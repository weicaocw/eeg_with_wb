import numpy as np
import logging
from src.features.wavelet import compute_xwt_stats, compute_xwb_stats
from src.features.statistics import (
    compute_cross_corr_stats,
    compute_coherence_stats,
    compute_mutual_info,
)

logger = logging.getLogger(__name__)

class FeatureManager:
    def __init__(self, config):
        self.config_features = config['features']
        self.fs = config['signal_processing']['fs']
        
        # 特征注册表：将配置文件中的字符串映射到函数
        self.registry = {
            'xwt_stats': compute_xwt_stats,
            'xwb_stats': compute_xwb_stats,
            'cross_corr': compute_cross_corr_stats,
            'coherence': compute_coherence_stats,
            'mutual_info': compute_mutual_info,
        }
        
    def extract(self, sig1, sig2):
        """
        对一对信号提取所有配置的特征
        """
        combined_features = []
        feature_names = []
        
        # 缓存机制：因为 XWB 可以复用 XWT 的 CWT 结果
        cache_cwt = None
        
        # 1. 优先计算 XWT (因为它可以产生缓存)
        if 'xwt_stats' in self.config_features:
            res_dict, wx, wy, scales = compute_xwt_stats(sig1, sig2, self.fs)
            
            cache_cwt = (wx, wy, scales)
            
            for k, v in res_dict.items():
                combined_features.append(v)
                feature_names.append(k)
                
        # 2. 计算其他特征
        for feat_key in self.config_features:
            if feat_key == 'xwt_stats': continue # 已处理
            
            if feat_key in self.registry:
                func = self.registry[feat_key]
                # 注入缓存
                res_dict = func(sig1, sig2, self.fs, precomputed_cwt=cache_cwt)
                
                for k, v in res_dict.items():
                    combined_features.append(v)
                    feature_names.append(k)
            else:
                logger.warning(f"Warning: Feature {feat_key} not implemented.")
                
        return np.array(combined_features), feature_names