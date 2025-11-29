import pytest
import torch
import sys
import os

# 路径修正
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.dataset import EEGSeizureDataset
from src.features_gpu import EEGFeatureLayer
from src.utils import load_config

# 目标测试文件
TARGET_SEGMENT_ID = "aaaaanme_s010_t012" 

def test_dataset_sequence_output():
    """
    测试 1: 验证 Dataset 是否正确返回了序列形状 [Seq_Len, 17, 250]
    """
    print("\n--- Test 1: Dataset Sequence Shape ---")
    cfg = load_config("configs/config.yaml")
    
    # 设置测试参数
    TEST_SEQ_LEN = 5 # 测试 5秒序列
    
    dataset = EEGSeizureDataset(
        root_h5_dir=cfg['data']['train_root_dir'],
        annotation_json_path=cfg['data']['train_annotation'],
        fs=cfg['data']['fs'],
        seq_len=TEST_SEQ_LEN, # <--- 关键：请求 5秒序列
        stride=1.0,
        tmp_dir='/tmp'
    )
    
    # 找到目标文件的一个样本
    target_idx = -1
    for i, sample in enumerate(dataset.samples):
        if TARGET_SEGMENT_ID in sample['file_path']:
            target_idx = i
            break
    
    if target_idx == -1:
        pytest.skip(f"Target file {TARGET_SEGMENT_ID} not found in dataset.")

    # 读取数据
    print(f"Reading index {target_idx}...")
    seq_x, label = dataset[target_idx]
    
    # 验证形状
    print(f"Output Shape: {seq_x.shape}")
    # 预期: [5, 17, 250] -> [Seq_Len, Channels, Time]
    assert seq_x.shape == (TEST_SEQ_LEN, 17, 250)
    assert label in [0, 1]
    
    print("✅ Dataset Sequence Shape Test Passed!")
    return seq_x # 返回给下一个测试用

def test_feature_layer_with_sequence():
    """
    测试 2: 将 Dataset 返回的一个序列，直接当作 Batch 喂给特征层
    """
    print("\n--- Test 2: Feature Layer on Sequence ---")
    
    # 1. 获取一段真实数据 [Seq_Len, 17, 250]
    # 为了独立运行，重新获取一次
    # (在实际 pytest 中可以通过 fixture 传递，这里为了简单直接复制逻辑)
    cfg = load_config("configs/config.yaml")
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 强制开启特征
    cfg['features']['bispectrum']['batch_size'] = 16 
    cfg['features']['bispectrum']['reduce_to_bands'] = True
    
    dataset = EEGSeizureDataset(
        root_h5_dir=cfg['data']['train_root_dir'],
        annotation_json_path=cfg['data']['train_annotation'],
        fs=cfg['data']['fs'],
        seq_len=5, # 5秒序列
        tmp_dir='/tmp'
    )
    # 随便取第0个样本
    seq_x, _ = dataset[0] 
    
    # 2. 放入 GPU
    device = torch.device(cfg['device'])
    # seq_x 是 [5, 17, 250]。
    # 对于 FeatureLayer 来说，它期望 [Batch, 17, Time]。
    # 我们可以把这 5 帧看作 Batch=5 进行并行计算！
    input_tensor = seq_x.to(device)
    
    # 3. 运行特征层
    model = EEGFeatureLayer(cfg).to(device)
    with torch.no_grad():
        out = model(input_tensor)
        
    # 4. 验证
    # 输入 Batch=5，输出的第一维度也应该是 5
    print(f"Feature Output Keys: {out.keys()}")
    
    if 'coherence' in out:
        assert out['coherence'].shape[0] == 5
        print(f"Coherence batch size matches sequence length (5).")
        
    if 'bispectrum_norm' in out:
        # [5, 136, 5, 5]
        assert out['bispectrum_norm'].shape[0] == 5
        print(f"Bispectrum batch size matches sequence length (5).")
        
    print("✅ Feature Layer Sequence Processing Passed!")

if __name__ == "__main__":
    # 允许直接运行调试
    test_dataset_sequence_output()
    test_feature_layer_with_sequence()