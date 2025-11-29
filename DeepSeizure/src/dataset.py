import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import json
import os
import random # 新增

class EEGSeizureDataset(Dataset):
    def __init__(self, root_h5_dir, annotation_json_path, fs=250, seq_len=10, stride=1.0, data_percentage=1.0):
        """
        Args:
            root_h5_dir (str): H5 数据根目录
            annotation_json_path (str): 标注 JSON 路径
            fs (int): 采样率
            seq_len (int): 序列长度 (秒)
            stride (float): 滑动窗口步长 (秒)
            data_percentage (float): 数据采样比例 (0.0 - 1.0)。例如 0.1 代表只使用 10% 的数据。
        """
        self.root_h5_dir = root_h5_dir
        self.fs = fs
        self.seq_len_sec = seq_len
        self.seq_pts = int(seq_len * fs) 
        self.stride_pts = int(stride * fs)
        
        # 1. 加载完整 JSON
        with open(annotation_json_path, 'r') as f:
            full_annotations = json.load(f)
            
        # 2. 实现分层采样逻辑 (Stratified Sampling)
        if data_percentage < 1.0:
            print(f"Dataset: 检测到采样比例 {data_percentage}，正在进行分层采样...")
            
            # 分组：有 Seizure vs 无 Seizure
            # 依据: seizure_duration_sec > 0
            seizure_files = [x for x in full_annotations if x.get('seizure_duration_sec', 0) > 0]
            bckg_files = [x for x in full_annotations if x.get('seizure_duration_sec', 0) == 0]
            
            # 计算目标数量
            n_seizure_target = int(len(seizure_files) * data_percentage)
            n_bckg_target = int(len(bckg_files) * data_percentage)
            
            # 确保至少有 1 个 (如果原始数据非空)
            if len(seizure_files) > 0 and n_seizure_target == 0: n_seizure_target = 1
            if len(bckg_files) > 0 and n_bckg_target == 0: n_bckg_target = 1
            
            # 随机抽取 (固定种子以保证可复现性)
            random.seed(42) 
            selected_seizure = random.sample(seizure_files, n_seizure_target)
            selected_bckg = random.sample(bckg_files, n_bckg_target)
            
            # 合并
            self.raw_annotations = selected_seizure + selected_bckg
            
            print(f"Dataset: 采样完成。")
            print(f"  - 原始: {len(full_annotations)} (Seizure: {len(seizure_files)}, Bckg: {len(bckg_files)})")
            print(f"  - 采样: {len(self.raw_annotations)} (Seizure: {len(selected_seizure)}, Bckg: {len(selected_bckg)})")
            
        else:
            # 全量数据
            self.raw_annotations = full_annotations

        self.samples = [] 
        self._prepare_indices()

    def _prepare_indices(self):
        # ... (这部分代码保持不变，它会遍历 self.raw_annotations) ...
        print(f"Dataset: 正在扫描文件并构建 {self.seq_len_sec}秒 的序列索引...")
        count = 0
        
        for entry in self.raw_annotations:
            # 1. 路径处理
            rel_path = entry['file_path']
            if not rel_path.endswith('.h5'): rel_path += '.h5'
            full_path = os.path.join(self.root_h5_dir, rel_path)
            
            if not os.path.exists(full_path): continue

            # 2. 获取 Seizure 区间
            seizure_intervals = []
            for event in entry['events']:
                if event['label'] != 'bckg':
                    seizure_intervals.append((event['start_time'], event['stop_time']))
            
            # 3. 获取数据总点数
            if 'data_points' in entry:
                n_points = entry['data_points']
            else:
                try:
                    with h5py.File(full_path, 'r') as f:
                        n_points = f['eeg'].shape[1]
                except: continue
            
            # 4. 生成序列索引 (只在文件内部滑动)
            for start_idx in range(0, n_points - self.seq_pts + 1, self.stride_pts):
                end_idx = start_idx + self.seq_pts
                
                # 简单实现：检查这个时间窗口与 Seizure 区间是否有交集
                t_start = start_idx / self.fs
                t_end = end_idx / self.fs
                
                label = 0
                for (sz_start, sz_end) in seizure_intervals:
                    # 只要有交集就算 (Overlap)
                    if max(t_start, sz_start) < min(t_end, sz_end):
                        label = 1
                        break
                
                self.samples.append({
                    'file_path': full_path,
                    'file_name': os.path.basename(full_path),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'label': label
                })
                count += 1
                
        print(f"Dataset: 索引构建完成！共生成 {len(self.samples)} 个序列样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ... (这部分代码保持不变) ...
        sample_info = self.samples[idx]
        original_path = sample_info['file_path']
        
        try:
            # 直接从原始路径读取数据
            with h5py.File(original_path, 'r') as f:
                # 一次性读取 seq_len 秒的数据 [Channels, Time]
                data = f['eeg'][:, sample_info['start_idx'] : sample_info['end_idx']]
            
            # Preprocessing: Z-Score Normalization
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True) + 1e-6
            data = (data - mean) / std
            
            # Reshape logic: [17, Seq_Len * fs] -> [Seq_Len, 17, fs]
            # 我们需要把连续的长波形，切成 1秒1秒 的块喂给 Feature Layer
            n_channels = data.shape[0]
            
            raw_tensor = torch.tensor(data, dtype=torch.float32)
            
            # [17, 2500] -> [17, 10, 250]
            reshaped = raw_tensor.view(n_channels, self.seq_len_sec, self.fs)
            # [17, 10, 250] -> [10, 17, 250]
            seq_x = reshaped.permute(1, 0, 2)
            
            y = torch.tensor(sample_info['label'], dtype=torch.long)
            
            return seq_x, y

        except Exception as e:
            print(f"Error reading {original_path}: {e}")
            # 返回全0数据占位
            return torch.zeros((self.seq_len_sec, 17, self.fs)), torch.tensor(0)