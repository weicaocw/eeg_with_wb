import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import json
import os
import shutil
import uuid

class EEGSeizureDataset(Dataset):
    def __init__(self, root_h5_dir, annotation_json_path, fs=250, seq_len=10, stride=1.0, tmp_dir='/tmp'):
        """
        Args:
            seq_len (int): 序列长度 (秒)。模型将一次性读取 seq_len 秒的数据作为一个样本。
            stride (float): 滑动窗口步长 (秒)。
        """
        self.root_h5_dir = root_h5_dir
        self.fs = fs
        
        # 核心变化：window_pts 现在代表整个序列的长度
        self.seq_len_sec = seq_len
        self.seq_pts = int(seq_len * fs) 
        self.stride_pts = int(stride * fs)
        
        self.tmp_dir = tmp_dir
        if not os.path.exists(self.tmp_dir): os.makedirs(self.tmp_dir, exist_ok=True)
        
        with open(annotation_json_path, 'r') as f:
            self.raw_annotations = json.load(f)
            
        self.samples = [] 
        self._prepare_indices()

    def _prepare_indices(self):
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
            
            # 4. 生成序列索引 (关键修正：只在文件内部滑动)
            # 只有当剩余数据 >= seq_pts (比如10秒) 时，才生成样本
            # 这样就绝对保证了同一个样本的数据来自同一个文件，且连续
            for start_idx in range(0, n_points - self.seq_pts + 1, self.stride_pts):
                end_idx = start_idx + self.seq_pts
                
                # 确定该序列的标签
                # 策略：如果序列的“最后一秒”在 seizure 区间内，或者序列中包含超过 50% 的 seizure，则标为 1
                # 这里我们使用更适合实时检测的策略：只要序列中有 Seizure 出现，就标记为 1 (偏敏感)，
                # 或者严格一点：序列最后时刻是 Seizure。
                
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
        sample_info = self.samples[idx]
        original_path = sample_info['file_path']
        unique_suffix = str(uuid.uuid4())[:8]
        tmp_name = f"{sample_info['file_name']}_{unique_suffix}.h5"
        tmp_path = os.path.join(self.tmp_dir, tmp_name)
        
        try:
            # Copy & Read
            shutil.copyfile(original_path, tmp_path)
            with h5py.File(tmp_path, 'r') as f:
                # 一次性读取 10秒 数据 [17, 2500]
                data = f['eeg'][:, sample_info['start_idx'] : sample_info['end_idx']]
            
            # Preprocessing
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True) + 1e-6
            data = (data - mean) / std
            
            # Reshape logic: [17, Seq_Len * fs] -> [Seq_Len, 17, fs]
            # 我们需要把连续的长波形，切成 1秒1秒 的块喂给 Feature Layer
            # data shape: [17, 2500]
            n_channels = data.shape[0]
            # view as [17, 10, 250] -> permute -> [10, 17, 250]
            # 前提：self.seq_pts 必须是 fs 的整数倍 (我们代码里保证了)
            
            raw_tensor = torch.tensor(data, dtype=torch.float32)
            
            # [17, 2500] -> [17, 10, 250]
            reshaped = raw_tensor.view(n_channels, self.seq_len_sec, self.fs)
            # [17, 10, 250] -> [10, 17, 250]
            seq_x = reshaped.permute(1, 0, 2)
            
            y = torch.tensor(sample_info['label'], dtype=torch.long)
            
            return seq_x, y

        except Exception as e:
            print(f"Error: {e}")
            # 返回全0数据占位
            return torch.zeros((self.seq_len_sec, 17, self.fs)), torch.tensor(0)
            
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass