import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import json
import os
import random
from tqdm import tqdm # 引入 tqdm 显示预加载进度

class EEGSeizureDataset(Dataset):
    def __init__(self, root_h5_dir, annotation_json_path, fs=250, seq_len=10, stride=1.0, data_percentage=1.0, cache_to_ram=False):
        """
        Args:
            cache_to_ram (bool): 是否将所有数据预加载到内存。
        """
        self.root_h5_dir = root_h5_dir
        self.fs = fs
        self.seq_len_sec = seq_len
        self.seq_pts = int(seq_len * fs) 
        self.stride_pts = int(stride * fs)
        self.cache_to_ram = cache_to_ram # 保存标志位
        
        # 1. 加载完整 JSON
        with open(annotation_json_path, 'r') as f:
            full_annotations = json.load(f)
            
        # 2. 分层采样逻辑
        if data_percentage < 1.0:
            print(f"Dataset: 采样比例 {data_percentage}...")
            seizure_files = [x for x in full_annotations if x.get('seizure_duration_sec', 0) > 0]
            bckg_files = [x for x in full_annotations if x.get('seizure_duration_sec', 0) == 0]
            
            n_seizure_target = max(1, int(len(seizure_files) * data_percentage))
            n_bckg_target = max(1, int(len(bckg_files) * data_percentage))
            
            random.seed(42) 
            self.raw_annotations = random.sample(seizure_files, n_seizure_target) + random.sample(bckg_files, n_bckg_target)
        else:
            self.raw_annotations = full_annotations

        # 3. 构建索引
        self.samples = [] 
_prepare_indices()
        
        # 4. === 核心修改: 预加载到 RAM ===
        self.cached_data = [] # 用于存储 Tensor
        if self.cache_to_ram:
            self._preload_data()

    def _prepare_indices(self):
        print(f"Dataset: 正在扫描文件索引...")
        for entry in self.raw_annotations:
            rel_path = entry['file_path']
            if not rel_path.endswith('.h5'): rel_path += '.h5'
            full_path = os.path.join(self.root_h5_dir, rel_path)
            
            if not os.path.exists(full_path): continue

            seizure_intervals = []
            for event in entry['events']:
                if event['label'] != 'bckg':
                    seizure_intervals.append((event['start_time'], event['stop_time']))
            
            if 'data_points' in entry:
                n_points = entry['data_points']
            else:
                try:
                    with h5py.File(full_path, 'r') as f: n_points = f['eeg'].shape[1]
                except: continue
            
            for start_idx in range(0, n_points - self.seq_pts + 1, self.stride_pts):
                end_idx = start_idx + self.seq_pts
                t_start = start_idx / self.fs
                t_end = end_idx / self.fs
                
                label = 0
                for (sz_start, sz_end) in seizure_intervals:
                    if max(t_start, sz_start) < min(t_end, sz_end):
                        label = 1
                        break
                
                self.samples.append({
                    'file_path': full_path,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'label': label
                })
        print(f"Dataset: 索引构建完成！共 {len(self.samples)} 个样本。")

    def _preload_data(self):
        """将所有数据读取处理后存入 self.cached_data 列表"""
        print(f"\n>>> [RAM CACHE] 开始预加载 {len(self.samples)} 个样本到内存 (这可能需要几分钟)...")
        
        # 使用 tqdm 显示进度条
        for idx in tqdm(range(len(self.samples)), desc="Caching to RAM"):
            # 调用内部读取函数，获取处理好的 Tensor
            x, y = self._load_one_item(idx)
            self.cached_data.append((x, y))
            
        print(f">>> [RAM CACHE] 预加载完成！后续训练将不再读取磁盘。\n")

    def _load_one_item(self, idx):
        """内部函数：读取并预处理单个样本 (从磁盘)"""
        sample_info = self.samples[idx]
        original_path = sample_info['file_path']
        
        try:
            with h5py.File(original_path, 'r') as f:
                data = f['eeg'][:, sample_info['start_idx'] : sample_info['end_idx']]
            
            # Preprocessing
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True) + 1e-6
            data = (data - mean) / std
            
            # Reshape: [17, T] -> [T, 17, 250] (FeatureLayer 期望的输入)
            # 注意: 这里的 Reshape 逻辑需要和你之前的匹配
            # 你的 FeatureLayer 期望输入是 [Batch, Seq_Len, Channels, Time_per_sec]
            # 这里我们把 Seq_Len 秒的数据切成 [Seq_Len, 17, 250]
            
            raw_tensor = torch.tensor(data, dtype=torch.float32)
            n_channels = raw_tensor.shape[0]
            
            # [17, Seq_Len * 250] -> [17, Seq_Len, 250]
            reshaped = raw_tensor.view(n_channels, self.seq_len_sec, self.fs)
            # [17, Seq_Len, 250] -> [Seq_Len, 17, 250]
            seq_x = reshaped.permute(1, 0, 2)
            
            y = torch.tensor(sample_info['label'], dtype=torch.long)
            
            return seq_x, y

        except Exception as e:
            print(f"Error reading {original_path}: {e}")
            return torch.zeros((self.seq_len_sec, 17, self.fs)), torch.tensor(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 如果开启了缓存，直接从列表取值 (极速)
        if self.cache_to_ram:
            return self.cached_data[idx]
        
        # 否则从磁盘读取 (慢)
        return self._load_one_item(idx)