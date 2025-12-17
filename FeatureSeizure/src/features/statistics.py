import numpy as np
from scipy import signal
from sklearn.metrics import mutual_info_score


def compute_cross_corr_stats(sig1, sig2, fs, **kwargs):
    """时域互相关特征"""
    sig1_z = sig1 - np.mean(sig1)
    sig2_z = sig2 - np.mean(sig2)
    denom = np.sqrt(np.sum(sig1_z ** 2) * np.sum(sig2_z ** 2)) + 1e-12
    corr = np.correlate(sig1_z, sig2_z, mode='full') / denom
    lags = np.arange(-len(sig1) + 1, len(sig1)) / fs
    abs_idx = np.argmax(np.abs(corr))
    return {
        'cross_corr_max': corr[abs_idx],
        'cross_corr_lag': lags[abs_idx],
        'cross_corr_mean': np.mean(corr)
    }


def compute_coherence_stats(sig1, sig2, fs, **kwargs):
    """频域相干度特征"""
    nperseg = min(256, len(sig1))
    if nperseg < 8:
        return {'coherence_mean': 0.0, 'coherence_max': 0.0, 'coherence_freq_at_max': 0.0}
    freqs, coh = signal.coherence(sig1, sig2, fs=fs, nperseg=nperseg)
    max_idx = int(np.argmax(coh)) if len(coh) else 0
    freq_at_max = freqs[max_idx] if len(freqs) > max_idx else 0.0
    return {
        'coherence_mean': float(np.mean(coh)) if len(coh) else 0.0,
        'coherence_max': float(np.max(coh)) if len(coh) else 0.0,
        'coherence_freq_at_max': float(freq_at_max)
    }


def compute_mutual_info(sig1, sig2, fs, **kwargs):
    """互信息特征（基于2D直方图的熵）"""
    bins = kwargs.get('mi_bins', 32)
    
    # 计算2D直方图（联合概率）
    hist_2d, _, _ = np.histogram2d(sig1, sig2, bins=bins)
    
    # 归一化为概率分布
    p_xy = hist_2d / (np.sum(hist_2d) + 1e-12)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # 计算互信息：MI = Σ p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 1e-12:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j] + 1e-12) + 1e-12)
    
    return {'mutual_info': float(mi)}
