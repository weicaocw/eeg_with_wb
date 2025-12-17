import os
import mne
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from src.features.manager import FeatureManager

logger = logging.getLogger(__name__)

class CHBDataLoader:
    def __init__(self, config, seizure_info):
        self.config = config
        self.seizure_info = seizure_info
        self.data_dir = config['paths']['data_dir']
        self.pairs = config['channel_pairs']
        self.fs = config['signal_processing']['fs']
        self.win_sec = config['signal_processing']['window_sec']
        self.step_sec = config['signal_processing']['step_sec']
        self.normalize = config['signal_processing'].get('normalize', False)
        
        self.feature_manager = FeatureManager(config)
        self.feature_names = None

    def process_file(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        intervals = self.seizure_info.get(filename, [])
        
        try:
            # Use verbose='error' to suppress duplicate channel warnings
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='error')
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return None, None
            
        # 统一通道名格式 (去除空格)
        # For duplicate channels (e.g., T8-P8-0, T8-P8-1), keep only the first one
        channels_to_drop = []
        clean_names = {}
        
        for ch in raw.ch_names:
            clean_ch = ch.strip()
            # If channel ends with -0, -1, etc. (MNE duplicate suffix)
            if clean_ch.endswith('-0'):
                # Keep this one (first occurrence) but rename without suffix
                clean_names[ch] = clean_ch[:-2]
            elif clean_ch.endswith('-1') or clean_ch.endswith('-2'):
                # Drop subsequent duplicates
                channels_to_drop.append(ch)
            else:
                clean_names[ch] = clean_ch
        
        # Drop duplicate channels first
        if channels_to_drop:
            logger.debug(f"Dropping duplicate channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)
        
        # Then rename remaining channels
        raw.rename_channels(clean_names)
        
        # 文件级标准化：计算每个通道的均值和标准差
        channel_stats = {}
        if self.normalize:
            for ch in raw.ch_names:
                data = raw.get_data(picks=ch)[0] * 1e6  # 转为 uV
                channel_stats[ch] = {'mean': np.mean(data), 'std': np.std(data)}
                if channel_stats[ch]['std'] == 0:
                    channel_stats[ch]['std'] = 1.0  # 避免除零
            logger.debug(f"Computed normalization stats for {len(channel_stats)} channels")
        
        X_file = []
        y_file = []
        
        total_samples = raw.n_times
        win_samples = int(self.win_sec * self.fs)
        step_samples = int(self.step_sec * self.fs)
        
        # --- 1. 正样本提取 (Seizure) ---
        seizure_mask = np.zeros(total_samples, dtype=bool)
        
        for start_sec, end_sec in intervals:
            s_idx = int(start_sec * self.fs)
            e_idx = int(end_sec * self.fs)
            seizure_mask[s_idx:e_idx] = True
            
            curr = s_idx
            while curr + win_samples <= e_idx:
                feats = self._extract_window(raw, curr, win_samples, channel_stats if self.normalize else None)
                if feats is not None:
                    X_file.append(feats)
                    y_file.append(1)
                curr += step_samples # 使用高重叠步长
                
        num_pos = len(y_file)
        
        # --- 2. 负样本提取 (Normal) ---
        # 目标数量
        target_neg = int(num_pos * self.config['data_balancing']['target_balance_ratio'])
        target_neg = min(target_neg, 20) # 至少取20个
        target_neg = max(target_neg, self.config['data_balancing']['negative_sample_limit'])
        
        count_neg = 0
        attempts = 0
        
        while count_neg < target_neg and attempts < target_neg * 10:
            attempts += 1
            rand_start = np.random.randint(0, total_samples - win_samples)
            # 检查是否触碰 seizure 区域
            if np.any(seizure_mask[rand_start : rand_start + win_samples]):
                continue
                
            feats = self._extract_window(raw, rand_start, win_samples, channel_stats if self.normalize else None)
            if feats is not None:
                X_file.append(feats)
                y_file.append(0)
                count_neg += 1
                
        logger.info(f"[{filename}] Seizure: {num_pos}, Normal: {count_neg}")
        
        if len(X_file) == 0:
            return None, None
        
        X = np.array(X_file)
        y = np.array(y_file)
        idx = np.random.permutation(len(y))
        
        return X[idx], y[idx]

    def _extract_window(self, raw, start_idx, n_samples, channel_stats=None):
        """提取单个窗口内所有通道对的特征，并拼接"""
        window_feats = []
        end_idx = start_idx + n_samples
        
        for ch1, ch2 in self.pairs:
            try:
                # 提取数据并转为 uV
                d1 = raw.get_data(picks=ch1, start=start_idx, stop=end_idx)[0] * 1e6
                d2 = raw.get_data(picks=ch2, start=start_idx, stop=end_idx)[0] * 1e6
                
                # 文件级标准化：使用预计算的均值和标准差
                if channel_stats is not None:
                    d1 = (d1 - channel_stats[ch1]['mean']) / channel_stats[ch1]['std']
                    d2 = (d2 - channel_stats[ch2]['mean']) / channel_stats[ch2]['std']
                
                # 调用 Manager 提取特征
                feats, names = self.feature_manager.extract(d1, d2)
                window_feats.extend(feats)
                
                # 记录特征名称 (只需做一次)
                if self.feature_names is None:
                    self.feature_names = [f"{ch1}|{ch2}_{n}" for n in names]
                    
            except ValueError:
                return None # 通道缺失
                
        return window_feats