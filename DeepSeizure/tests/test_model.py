import pytest
import torch
import sys
import os
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.dataset import EEGSeizureDataset
from src.models.sequence import EEGSeizureNet
from src.utils import load_config

@pytest.mark.parametrize("model_type", ["lstm", "transformer"])
def test_end_to_end_model_forward(model_type):
    """
    测试 Dataset (序列版) -> DataLoader -> Model 的完整数据流
    """
    print(f"\n\n=== Testing Model: {model_type.upper()} with Sequence Dataset ===")
    
    # 1. 加载配置
    cfg = load_config("configs/config.yaml")
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 修改配置
    cfg['train']['model_type'] = model_type
    cfg['features']['bispectrum']['batch_size'] = 16 
    # 我们让 Dataset 产生 5秒 的序列
    TEST_SEQ_LEN = 5
    
    # 2. 初始化 Dataset (原生序列模式)
    dataset = EEGSeizureDataset(
        root_h5_dir=cfg['data']['train_root_dir'],
        annotation_json_path=cfg['data']['train_annotation'],
        fs=cfg['data']['fs'],
        seq_len=TEST_SEQ_LEN,  # <--- 直接指定序列长度
        stride=1.0,
        tmp_dir='/tmp'
    )
    
    if len(dataset) == 0:
        pytest.skip("Dataset is empty. Check paths.")

    # 3. DataLoader 打包 Batch
    # 期望输出: [Batch_Size, Seq_Len, 17, 250]
    BATCH_SIZE = 2
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 获取一个 Batch
    try:
        x_batch, y_batch = next(iter(loader))
    except StopIteration:
        pytest.skip("Could not fetch batch from loader.")
        
    print(f"DataLoader Output Shape: {x_batch.shape}")
    # 验证: [2, 5, 17, 250]
    assert x_batch.shape == (BATCH_SIZE, TEST_SEQ_LEN, 17, 250)
    
    # 4. 模型前向传播
    device = torch.device(cfg['device'])
    x_batch = x_batch.to(device)
    
    model = EEGSeizureNet(cfg).to(device)
    
    with torch.no_grad():
        logits = model(x_batch)
        
    # 5. 验证输出
    print(f"Logits Shape: {logits.shape}")
    # 期望: [Batch_Size, 2] (二分类)
    assert logits.shape == (BATCH_SIZE, 2)
    
    print(f"✅ {model_type.upper()} End-to-End Test Passed!")

if __name__ == "__main__":
    # 手动调试
    test_end_to_end_model_forward("lstm")