import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import json
import os
import random
from tqdm import tqdm

class EEGSeizureDataset(Dataset):
    def __init__(self, root_h5_dir, annotation_json_path, fs=250, seq_len=10, stride=1.0, data_percentage=1.0, cache_to_ram=False):
        """
        Args:
            cache_to_ram (bool): 是否将所有用到的 H5 源文件加载到内存。
        """
        self.root_h5_dir = root_h5_dir
        self.fs = fs
        self.seq_len_sec = seq_len
        self.seq_pts = int(seq_len * fs) 
        self.stride_pts = int(stride * fs)
        self.cache_to_ram = cache_to_ram
        
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
        self._prepare_indices()
        
        # 4. === 核心修改: 预加载源文件到 RAM ===
        self.file_cache = {} # Key: file_path, Value: torch.Tensor (Whole File)
        if self.cache_to_ram:
            self._preload_files()

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

    def _preload_files(self):
        """
        优化版缓存：只缓存 unique 的源文件，而不是缓存切片。
        这消除了滑动窗口带来的内存冗余，且无需 psutil 监控。
        """
        # 1. 找出所有需要用到的唯一文件路径
        unique_files = set(sample['file_path'] for sample in self.samples)
        print(f"\n>>> [RAM CACHE] 开始加载 {len(unique_files)} 个源文件到内存 (无冗余)...")
        
        for file_path in tqdm(unique_files, desc="Caching Files"):
            try:
                # 读取整个文件
                with h5py.File(file_path, 'r') as f:
                    # [Channels, All_Time] -> 存入内存
                    # 使用 float32 节省空间
                    whole_data = torch.tensor(f['eeg'][:], dtype=torch.float32)
                    self.file_cache[file_path] = whole_data
            except Exception as e:
                print(f"Failed to cache {file_path}: {e}")

        print(f">>> [RAM CACHE] 缓存完成。已缓存 {len(self.file_cache)} 个文件。")

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        start = sample_info['start_idx']
        end = sample_info['end_idx']
        
        # === 数据读取逻辑 ===
        if self.cache_to_ram and file_path in self.file_cache:
            # [路径 A] 极速模式：从内存中的大 Tensor 直接切片
            raw_data = self.file_cache[file_path][:, start:end]
        else:
            # [路径 B] 普通模式：从磁盘读取
            try:
                with h5py.File(file_path, 'r') as f:
                    data_numpy = f['eeg'][:, start:end]
                    raw_data = torch.tensor(data_numpy, dtype=torch.float32)
            except Exception as e:
                # 容错返回
                return torch.zeros((17, self.seq_pts // self.fs, self.fs)).permute(1, 0, 2), torch.tensor(0)

        # === 实时预处理 (On-the-fly Preprocessing) ===
        # 注意：这里我们对取出来的这一小段做处理
        
        # 1. Z-Score Normalization
        mean = raw_data.mean(dim=1, keepdim=True)
        std = raw_data.std(dim=1, keepdim=True) + 1e-6
        norm_data = (raw_data - mean) / std
        
        # 2. Reshape to [Seq_Len, Channels, Freq]
        n_channels = norm_data.shape[0]
        
        # 检查数据长度是否足够
        if norm_data.shape[1] != self.seq_pts:
             pad_len = self.seq_pts - norm_data.shape[1]
             norm_data = torch.nn.functional.pad(norm_data, (0, pad_len))

        reshaped = norm_data.view(n_channels, self.seq_len_sec, self.fs)
        # [17, 10, 250] -> [10, 17, 250]
        seq_x = reshaped.permute(1, 0, 2)
        
        y = torch.tensor(sample_info['label'], dtype=torch.long)
        
        return seq_x, y

    def __len__(self):
        return len(self.samples)