import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGFeatureLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config['features']
        self.n_channels = 17
        self.fs = config['data']['fs']
        self.n_fft = self.cfg['n_fft']
        self.mi_bins = self.cfg['mi_bins']
        
        # 基础开关
        self.use_coh = self.cfg['use_coherence']
        self.use_mi = self.cfg['use_mi']
        
        # 双谱细分配置
        self.bi_cfg = self.cfg['bispectrum']
        self.use_bispectrum = self.bi_cfg['enabled']
        self.include_raw = self.bi_cfg['include_raw']
        self.include_norm = self.bi_cfg['include_normalized']
        self.reduce_bands = self.bi_cfg['reduce_to_bands']
        self.bispec_batch = self.bi_cfg['batch_size']
        
        # 1. 预计算通道对 (136对)
        self.register_buffer('pair_indices', torch.triu_indices(self.n_channels, self.n_channels, offset=1))
        self.n_pairs = self.pair_indices.shape[1]
        
        # 2. 窗函数
        self.register_buffer('window', torch.hann_window(self.fs)) 

        # 3. 双谱初始化
        if self.use_bispectrum:
            self._init_bispectrum_indices()
            # 无论是否 reduce，只要涉及频带定义，都建议初始化 mask (虽然只有 reduce 时才用)
            if 'frequency_bands' in self.cfg:
                self._init_band_masks(self.cfg['frequency_bands'])

    def _init_bispectrum_indices(self):
        n_freqs = self.n_fft // 2 + 1
        f1 = torch.arange(n_freqs)
        f2 = torch.arange(n_freqs)
        sum_idx = f1.unsqueeze(-1) + f2.unsqueeze(0)
        mask = sum_idx < n_freqs
        # 这里的 mask 乘法会在计算时剔除 f1+f2 > Nyquist
        self.register_buffer('freq_sum_idx', sum_idx * mask.long())
        self.register_buffer('freq_mask', mask.float()) 

    def _init_band_masks(self, bands_dict):
        n_freqs = self.n_fft // 2 + 1
        freqs = torch.linspace(0, self.fs/2, n_freqs)
        self.band_names = list(bands_dict.keys())
        band_masks = []
        for name in self.band_names:
            low, high = bands_dict[name]
            mask = (freqs >= low) & (freqs < high)
            band_masks.append(mask.float())
        self.register_buffer('band_masks', torch.stack(band_masks))

    def compute_spectral_features(self, x):
        """计算 FFT, 相干性, 互相关"""
        B, C, T = x.shape
        idx_i, idx_j = self.pair_indices[0], self.pair_indices[1]
        X = torch.fft.rfft(x * self.window, n=self.n_fft) 
        
        X_i = X[:, idx_i, :]
        X_j = X[:, idx_j, :]
        
        features = {}
        features['fft'] = X # 留给双谱用
        
        if self.use_coh:
            Pxy = X_i * torch.conj(X_j)
            Pxx = X_i.abs().pow(2)
            Pyy = X_j.abs().pow(2)
            features['coherence'] = Pxy.abs().pow(2) / (Pxx * Pyy + 1e-8)
            features['cross_corr'] = torch.fft.irfft(Pxy, n=self.n_fft)
        return features

    def _init_bispectrum_indices(self):
            n_freqs = self.n_fft // 2 + 1
            f1 = torch.arange(n_freqs)
            f2 = torch.arange(n_freqs)
            
            # [F, F]
            sum_idx = f1.unsqueeze(-1) + f2.unsqueeze(0)
            
            # Mask 1: Nyquist 限制 (f1 + f2 < n_freqs)
            mask_nyquist = sum_idx < n_freqs
            
            # Mask 2: 对称性去重 (f1 <= f2)
            # 如果你想保留全矩阵(为了CNN)，这个可以不做；
            # 如果你想去重，只保留上三角，可以加上这个。
            # 通常为了 CNN 输入，我们保留对称的矩阵，或者只保留上三角并置零下三角。
            # 这里为了最大化信息展示，我们保留全矩阵，只 mask 掉 nyquist 溢出部分。
            
            final_mask = mask_nyquist
            
            # 将溢出的索引设为 0，防止 gather 越界 (配合 mask 使用，结果会被置 0)
            safe_sum_idx = sum_idx * final_mask.long()
            
            self.register_buffer('freq_sum_idx', safe_sum_idx)
            self.register_buffer('freq_mask', final_mask.float()) 

    def compute_cross_bispectrum(self, X):
        """
        同时计算 Raw Bispectrum 和 Threenorm Bicoherence
        """
        B_size, _, n_freqs = X.shape
        idx_i, idx_j = self.pair_indices[0], self.pair_indices[1]
        
        list_raw = []
        list_norm = []
        
        # 预计算 |X|^3 用于归一化
        if self.include_norm:
            X_abs_cube = X.abs().pow(3)
        
        chunk_size = self.n_pairs if self.bispec_batch <= 0 else self.bispec_batch
        
        # 确保 mask 的维度正确用于广播 [1, 1, F, F]
        mask_broadcast = self.freq_mask.unsqueeze(0).unsqueeze(0)
        
        for k in range(0, self.n_pairs, chunk_size):
            end_k = min(k + chunk_size, self.n_pairs)
            
            sub_i = idx_i[k:end_k] 
            sub_j = idx_j[k:end_k]
            
            X_i = X[:, sub_i, :] 
            X_j = X[:, sub_j, :]
            
            # --- 分子 (Raw Bispectrum) ---
            # Term 1 & 2: Outer Product -> [B, chunk, F, F]
            term12 = X_i.unsqueeze(-1) * X_j.unsqueeze(-2)
            
            # Term 3: Gather X_j(f1+f2)
            X_j_conj = torch.conj(X_j)
            flat_indices = self.freq_sum_idx.view(-1)
            term3 = X_j_conj[:, :, flat_indices].view(B_size, end_k-k, n_freqs, n_freqs)
            
            # Raw Result: 计算并应用 Mask
            # 关键点：在这里直接乘 Mask，确保无效区域绝对为 0
            raw_chunk = (term12 * term3).abs() * mask_broadcast
            
            if self.include_raw:
                list_raw.append(raw_chunk)
            
            # --- 分母 (Normalization) ---
            if self.include_norm:
                # Denom formula: (|Xi(f1)|^3 * |Xj(f2)|^3 * |Xj(f1+f2)|^3)^(1/3)
                
                cube_i_f1 = X_abs_cube[:, sub_i, :].unsqueeze(-1)
                cube_j_f2 = X_abs_cube[:, sub_j, :].unsqueeze(-2)
                
                # Gather cube sum
                cube_j_sum = X_abs_cube[:, sub_j, :]
                cube_j_sum_flat = cube_j_sum[:, :, flat_indices].view(B_size, end_k-k, n_freqs, n_freqs)
                
                denom_prod = cube_i_f1 * cube_j_f2 * cube_j_sum_flat
                denominator = denom_prod.pow(1/3)
                
                # Norm Result = Raw / Denom
                # 关键修复：只在 Mask 有效的区域进行除法，避免 0/0 产生 NaN
                # 方法：分母加 epsilon，并且结果再次乘 Mask
                
                norm_chunk = raw_chunk / (denominator + 1e-8)
                
                # 再次应用 Mask 确保干净 (特别是 NaN 可能会污染)
                # 使用 torch.nan_to_num 处理可能的漏网之鱼
                norm_chunk = torch.nan_to_num(norm_chunk, nan=0.0, posinf=0.0, neginf=0.0)
                norm_chunk = norm_chunk * mask_broadcast
                
                list_norm.append(norm_chunk)

        # 拼接结果
        out_raw = torch.cat(list_raw, dim=1) if self.include_raw else None
        out_norm = torch.cat(list_norm, dim=1) if self.include_norm else None
        
        return out_raw, out_norm

    def compute_band_features(self, full_bispec):
        """将 65x65 -> 5x5"""
        if full_bispec is None: return None
        
        n_bands = self.band_masks.shape[0]
        B, P, _, _ = full_bispec.shape
        band_features = torch.zeros(B, P, n_bands, n_bands, device=full_bispec.device)
        
        for i in range(n_bands):
            for j in range(n_bands):
                mask1 = self.band_masks[i]
                mask2 = self.band_masks[j]
                joint_mask = mask1.unsqueeze(-1) * mask2.unsqueeze(0) 
                
                num_points = joint_mask.sum()
                if num_points > 0:
                    masked_val = full_bispec * joint_mask.unsqueeze(0).unsqueeze(0)
                    avg_val = masked_val.sum(dim=(-1, -2)) / num_points
                    band_features[:, :, i, j] = avg_val
        return band_features

    def compute_mi(self, x):
        """计算互信息 (One-Hot 矩阵乘法加速版)"""
        if not self.use_mi: return None
        B, C, T = x.shape
        idx_i, idx_j = self.pair_indices[0], self.pair_indices[1]
        
        x_i = x[:, idx_i, :]
        x_j = x[:, idx_j, :]
        
        # Binning
        min_val = x.min(dim=-1, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0]
        x_norm = (x - min_val) / (max_val - min_val + 1e-6)
        x_binned = (x_norm * self.mi_bins).long().clamp(0, self.mi_bins-1)
        
        bin_i = x_binned[:, idx_i, :]
        bin_j = x_binned[:, idx_j, :]
        
        # Matrix Multiplication for Joint Hist
        oh_i = F.one_hot(bin_i, num_classes=self.mi_bins).float()
        oh_j = F.one_hot(bin_j, num_classes=self.mi_bins).float()
        
        # [B, 136, Bins, T] @ [B, 136, T, Bins] -> [B, 136, Bins, Bins]
        joint_hist = torch.matmul(oh_i.transpose(-1, -2), oh_j)
        P_xy = joint_hist / T
        
        P_x = P_xy.sum(dim=-1)
        P_y = P_xy.sum(dim=-2)
        
        def entropy(p): return -(p * (p + 1e-8).log()).sum(dim=-1)
        
        H_x = entropy(P_x)
        H_y = entropy(P_y)
        H_xy = -(P_xy * (P_xy + 1e-8).log()).sum(dim=(-1, -2))
        
        return (H_x + H_y - H_xy).unsqueeze(-1) # [B, 136, 1]

    def forward(self, x):
        results = {}
        spec_feats = self.compute_spectral_features(x)
        results.update(spec_feats)
        X_fft = results.pop('fft') 
        
        if self.use_mi:
            results['mi'] = self.compute_mi(x)
            
        if self.use_bispectrum:
            # 1. 计算
            raw_full, norm_full = self.compute_cross_bispectrum(X_fft)
            
            # 2. 根据配置决定输出形式 (降维 OR 全尺寸)
            if self.reduce_bands:
                if self.include_raw:
                    results['bispectrum_raw'] = self.compute_band_features(raw_full) # [B, 136, 5, 5]
                if self.include_norm:
                    results['bispectrum_norm'] = self.compute_band_features(norm_full) # [B, 136, 5, 5]
            else:
                # 不降维，直接输出全尺寸 (小心显存)
                if self.include_raw:
                    results['bispectrum_raw'] = raw_full # [B, 136, 65, 65]
                if self.include_norm:
                    results['bispectrum_norm'] = norm_full # [B, 136, 65, 65]
                
        return results