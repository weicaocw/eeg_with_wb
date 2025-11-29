import torch
import torch.nn as nn

class HybridEncoder(nn.Module):
    def __init__(self, n_pairs=136, bispec_size=5, embed_dim=256):
        """
        Args:
            n_pairs: 通道对数量 (136)
            bispec_size: 双谱图的边长 (如果降维了是 5，没降维是 65)
            embed_dim: 输出特征向量的维度
        """
        super().__init__()
        
        # --- 分支 A: 处理双谱图 (CNN) ---
        # 输入: [Batch, 136, H, W]
        # 我们使用 AdaptiveAvgPool，这样无论输入是 5x5 还是 65x65 都能处理
        self.bispec_cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(n_pairs, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 无论输入多大，最后都变成 1x1，提取全局特征
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten() # -> [Batch, 256]
        )
        
        # --- 分支 B: 处理线性特征 (MLP) ---
        # 这一层的输入维度需要在运行时动态确定，或者在 sequence.py 里计算好传入
        # 这里先定义好结构，输入层由 sequence.py 初始化
        self.linear_mlp = nn.Sequential(
            nn.Linear(1, 512), # 占位，会被替换
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # --- 融合层 ---
        self.fusion = nn.Linear(256 + 256, embed_dim)

    def forward(self, bispec, linear_feats):
        """
        bispec: [B, 136, H, W]
        linear_feats: [B, D]
        """
        # 1. CNN 提取双谱特征
        emb_bispec = self.bispec_cnn(bispec) # [B, 256]
        
        # 2. MLP 提取线性特征
        emb_linear = self.linear_mlp(linear_feats) # [B, 256]
        
        # 3. 融合
        combined = torch.cat([emb_bispec, emb_linear], dim=1)
        output = self.fusion(combined) # [B, embed_dim]
        
        return output