import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import argparse  # 1. 引入 argparse

from src.dataset import EEGSeizureDataset
from src.models.sequence import EEGSeizureNet
from src.utils import load_config

# ==========================================
# 辅助函数
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        # === 技巧 1: 混合精度 (BFloat16) ===
        # RTX 30/40 系列专属福利，比 FP16 更稳，不用 Scaler
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # TensorBoard: 记录每个 Step 的 Loss
        global_step = (epoch - 1) * len(loader) + step
        writer.add_scalar('Loss/step_train', loss.item(), global_step)
        
    avg_loss = total_loss / len(loader)
    return avg_loss

def evaluate(model, loader, criterion, device, phase="Val"):
    """
    phase: 'Val' or 'Test'
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = [] # 记录概率用于 AUC
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"{phase}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1] # 取 Seizure 概率
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    metrics = {
        'loss': avg_loss,
        'acc': acc,
        'f1': f1,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }
    return metrics

# ==========================================
# 主流程
# ==========================================
def run_pipeline(config_path):
    print(f"Loading configuration from: {config_path}")
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 初始化 TensorBoard
    import time
    run_name = f"{cfg['train']['model_type']}_{int(time.time())}"
    log_dir = os.path.join(cfg['train'].get('log_dir', 'runs'), run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log dir: {log_dir}")
    
    # 2. 准备数据
    print("\n>>> Loading Training Data...")
    train_full_ds = EEGSeizureDataset(
        root_h5_dir=cfg['data']['train_root_dir'],
        annotation_json_path=cfg['data']['train_annotation'],
        fs=cfg['data']['fs'],
        seq_len=cfg['train']['seq_len'],
        stride=1.0,
        tmp_dir=cfg['data']['tmp_dir']
    )
    
    # 划分 Train/Val
    train_size = int(0.9 * len(train_full_ds))
    val_size = len(train_full_ds) - train_size
    train_ds, val_ds = random_split(train_full_ds, [train_size, val_size])
    
# === 技巧 3: DataLoader 参数 ===
    # num_workers: 设置为 CPU 核心数的一半左右。你由 16核，设为 8 或 10 比较合适。
    # pin_memory: 必须为 True！这会把数据锁在内存中，加速 CPU 到 GPU 的传输。
    # persistent_workers: True。避免每个 Epoch 结束后销毁进程再重建，节省开销。
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=8,        # 利用多核 CPU
        pin_memory=True,      # 加速 CPU->GPU 拷贝
        persistent_workers=True, # 保持进程存活
        prefetch_factor=4     # 让每个 worker 提前多读几个 batch
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=False, 
        num_workers=4,        # 验证集可以少一点
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. 初始化模型
    model = EEGSeizureNet(cfg).to(device)
    
    # === 技巧 2: 编译模型 ===
    # mode='reduce-overhead' 适合小 Batch 极速推理
    # mode='max-autotune' 适合长时间训练，追求极致吞吐
    # 对于你的 Transformer/LSTM，这是免费的加速
    print(">>> Compiling model with torch.compile...")
    model = torch.compile(model, mode='default')
    
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['train']['learning_rate']))
    criterion = nn.CrossEntropyLoss()
    
    # 4. 训练循环
    best_val_f1 = 0.0
    checkpoint_dir = "DeepSeizure/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{run_name}.pth")
    
    print(f"\n>>> ({cfg['train']['epochs']} epochs)...")
    for epoch in range(1, cfg['train']['epochs'] + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Val
        val_metrics = evaluate(model, val_loader, criterion, device, phase="Val")
        
        # TensorBoard Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('Acc/val', val_metrics['acc'], epoch)
        
        print(f"Ep {epoch} | Tr Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save Best
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> New Best F1! Model saved.")
            
    # 5. 测试阶段
    print("\n" + "="*40)
    print(">>> Training Finished. Starting Test...")
    print("="*40)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 加载测试集
    test_ds = EEGSeizureDataset(
        root_h5_dir=cfg['data']['test_root_dir'],
        annotation_json_path=cfg['data']['test_annotation'],
        fs=cfg['data']['fs'],
        seq_len=cfg['train']['seq_len'],
        stride=1.0,
        tmp_dir=cfg['data']['tmp_dir']
    )
    
    if len(test_ds) == 0:
        print("Warning: Test dataset is empty.")
    else:
        test_loader = DataLoader(test_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=2)
        test_metrics = evaluate(model, test_loader, criterion, device, phase="Test")
        
        # 计算高级指标
        y_true = np.array(test_metrics['labels'])
        y_pred = np.array(test_metrics['preds'])
        y_prob = np.array(test_metrics['probs'])
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0
            
        print(f"\nFinal Test Report [{cfg['train']['model_type']}]:")
        print(f"Accuracy:    {test_metrics['acc']:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1 Score:    {test_metrics['f1']:.4f}")
        print(f"ROC AUC:     {auc:.4f}")
        
        # TensorBoard HParams
        writer.add_hparams(
            {'lr': cfg['train']['learning_rate'], 'bs': cfg['train']['batch_size'], 'model': cfg['train']['model_type']},
            {'hparam/test_f1': test_metrics['f1'], 'hparam/test_auc': auc}
        )

    writer.close()
    print(f"\nDone! View logs via: tensorboard --logdir={cfg['train']['log_dir']}")

# 2. 修改入口逻辑
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeizure Training Pipeline")
    
    # 添加 --config 参数，设置默认值
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml', 
        help='Path to the configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # 将解析到的路径传给 run_pipeline
    run_pipeline(config_path=args.config)