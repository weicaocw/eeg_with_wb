import numpy as np
from scipy.stats import skew

# Monkey-patch pycwt.helpers.fft_kwargs BEFORE importing pycwt
import pycwt.helpers
def _fixed_fft_kwargs(signal, **kwargs):
    """Fixed version that uses int() instead of np.int()"""
    return {'n': int(2 ** np.ceil(np.log2(len(signal))))}
pycwt.helpers.fft_kwargs = _fixed_fft_kwargs

# Now import pycwt (it will use the patched version)
import pycwt.wavelet
# Also patch the reference in pycwt.wavelet module
from pycwt.helpers import fft_kwargs as original_fft_kwargs
import pycwt.wavelet as wavelet_module
# Replace the imported reference
wavelet_module.fft_kwargs = _fixed_fft_kwargs

import pycwt

def get_integer_freq_scales(fs, wavelet='morlet'):
    """
    辅助函数：生成对应于整数频率 (1Hz, 2Hz, ... Nyquist) 的 CWT 尺度。
    这保证了我们后续可以通过索引直接找到 f 和 2f。
    """
    # 1. 定义目标频率：从 1Hz 到 fs/2 (奈奎斯特频率)
    # 使用 floor 确保不越界
    nyquist = int(fs / 2)
    freqs = np.arange(1, nyquist + 1)
    return freqs

def compute_xwt_stats(sig1, sig2, fs, **kwargs):
    """
    计算 Cross Wavelet Transform 统计特征
    修正：强制使用整数频率尺度
    """
    # 1. 获取精确的整数频率
    freqs = get_integer_freq_scales(fs, wavelet='morlet')
    dt = 1.0 / fs
    wavelet = pycwt.wavelet.Morlet()
    # 2. 用 pycwt 计算 CWT（每个频率点都精确）
    # pycwt.cwt returns (W, scales, freqs, coi, fft, fftfreqs)
    wx = pycwt.cwt(sig1, dt, wavelet=wavelet, freqs=freqs)[0]
    wy = pycwt.cwt(sig2, dt, wavelet=wavelet, freqs=freqs)[0]
    # 3. XWT 计算
    xwt = wx * np.conj(wy)
    xwt_mag = np.abs(xwt)
    # 4. 统计特征
    feats = {
        'xwt_mean': np.mean(xwt_mag),
        'xwt_max': np.max(xwt_mag),
        'xwt_std': np.std(xwt_mag),
        'xwt_skew': skew(xwt_mag.flatten())
    }
    # 返回 wx, wy 和 freqs（xwb 需要频率信息）
    return feats, wx, wy, freqs

def compute_xwb_stats(sig1, sig2, fs, precomputed_cwt=None, **kwargs):
    """
    计算 Simplified Cross Wavelet Bispectrum 统计特征
    修正：使用精确索引匹配 f 和 2f，并在数学上严格限制 Nyquist 边界
    """
    # 1. 获取 CWT 数据
    if precomputed_cwt:
        wx, wy, freqs = precomputed_cwt
    else:
        freqs = get_integer_freq_scales(fs, wavelet='morlet')
        dt = 1.0 / fs
        wavelet = pycwt.wavelet.Morlet()
        wx = pycwt.cwt(sig1, dt, wavelet=wavelet, freqs=freqs)[0]
        wy = pycwt.cwt(sig2, dt, wavelet=wavelet, freqs=freqs)[0]
    
    # wx, wy 的形状是 (n_freqs, n_time)
    # freqs 是 [1, 2, 3, ..., Nyquist]
    
    # 2. 构建 f -> 2f 的索引映射
    # 假设 freqs 从 1Hz 开始，步长为 1Hz。
    # 索引 i 对应的频率是 i + 1 Hz。
    # 我们需要找 2 * (i + 1) Hz。
    # 2 * (i + 1) Hz 对应的索引是 2*(i+1) - 1 = 2*i + 1。
    
    n_freqs = len(freqs)
    
    # 生成基础频率 f 的索引列表 (i)
    # 限制条件：对应的 2f 索引 (2*i + 1) 必须小于总频率数 (n_freqs)
    # 解不等式：2*i + 1 < n_freqs  =>  2*i < n_freqs - 1  => i < (n_freqs - 1) / 2
    max_i = int((n_freqs - 1) / 2)
    
    # 如果有效频率太少，直接返回0
    if max_i < 1:
        return {'xwb_mean': 0.0, 'xwb_max': 0.0, 'xwb_std': 0.0, 'xwb_skew': 0.0}

    # 向量化生成索引
    idx_f = np.arange(0, max_i)      # 频率 f 的索引
    idx_2f = 2 * idx_f + 1           # 频率 2f 的精确索引
    
    # 3. 双谱计算 (Vectorized)
    # 公式: B(f) = Wx(f) * Wy(f) * conj(Wy(2f))
    # 取出对应的行进行计算
    term1 = wx[idx_f, :]       # Wx at f
    term2 = wy[idx_f, :]       # Wy at f
    term3 = np.conj(wy[idx_2f, :]) # Wy* at 2f
    
    # 结果是一个矩阵 (valid_freqs, n_time)
    b_val = np.abs(term1 * term2 * term3)
    
    # 4. 统计特征 (已添加 Skewness)
    feats = {
        'xwb_mean': np.mean(b_val),
        'xwb_max': np.max(b_val),
        'xwb_std': np.std(b_val),
        'xwb_skew': skew(b_val.flatten())
    }
    return feats