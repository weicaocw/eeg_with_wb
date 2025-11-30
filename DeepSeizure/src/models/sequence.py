import torch
import torch.nn as nn
import math
from src.features_gpu import EEGFeatureLayer
from src.models.encoder import HybridEncoder

# --- 辅助模块: 位置编码 (用于 Transformer) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        return x + self.pe[:, :x.size(1), :]

# --- 主模型 ---
class EEGSeizureNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 特征提取层
        self.feature_layer = EEGFeatureLayer(config)
        
        # 2. 计算维度参数
        n_pairs = 136
        # 根据配置判断双谱图大小
        use_bands = config['features']['bispectrum'].get('reduce_to_bands', True)
        # 如果降维了是 5 (Delta-Gamma)，否则是 n_freqs
        bispec_size = 5 if use_bands else (config['features']['n_fft'] // 2 + 1)
        
        # 计算线性特征维度 (Coherence + CrossCorr + MI)
        n_freqs = config['features']['n_fft'] // 2 + 1
        n_fft = config['features']['n_fft']
        # 粗略计算展平后的维度: 
        # Coherence(F) + CrossCorr(T) + MI(1)
        # 如果 output_band_averages=True，Coherence可能不需要降维，或者也可以平均，
        # 这里假设线性特征保持全分辨率以保留细节
        linear_input_dim = 0
        if config['features']['use_coherence']:
            linear_input_dim += n_pairs * n_freqs # Coherence
            linear_input_dim += n_pairs * n_fft   # Cross Corr
        if config['features']['use_mi']:
            linear_input_dim += n_pairs * 1       # MI
            
        embed_dim = 256
        
        # 3. 单帧编码器
        self.encoder = HybridEncoder(
            n_pairs=n_pairs, 
            bispec_size=bispec_size, 
            embed_dim=embed_dim
        )
        # 动态替换 MLP 输入层
        self.encoder.linear_mlp[0] = nn.Linear(linear_input_dim, 512)
        
        # 4. 时序骨干 (Backbone)
        model_type = config['train']['model_type'].lower()
        self.model_type = model_type
        
        print(f">> Initializing Sequence Backbone: {model_type.upper()}")
        
        if model_type == 'lstm':
            self.backbone = nn.LSTM(
                input_size=embed_dim, 
                hidden_size=128, 
                num_layers=2, 
                batch_first=True, 
                bidirectional=True
            )
            self.out_dim = 128 * 2 # Bidirectional
            
        elif model_type == 'transformer':
            self.pos_encoder = PositionalEncoding(embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=4, 
                dim_feedforward=512, 
                batch_first=True,
                dropout=0.1
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.out_dim = embed_dim
            
        elif model_type == 'mamba':
            # 严格检查，没有库直接报错
            try:
                from mamba_ssm import Mamba
            except ImportError:
                raise ImportError("Error: 'mamba_ssm' library not found. Please install it or choose another model_type.")
            
            self.backbone = Mamba(
                d_model=embed_dim, 
                d_state=16,  
                d_conv=4,    
                expand=2,    
            )
            self.out_dim = embed_dim
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2) 
        )

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Channels, Time] 
           e.g., [4, 60, 17, 250]
        """
        B, L, C, T = x.shape
        
        # --- A. 特征提取 & 编码 (Frame-wise) ---
        # 合并 Batch 和 Length: [B*L, C, T]
        x_flat = x.view(B * L, C, T)
        
        # 提取特征 (Feature Layer)
        feats = self.feature_layer(x_flat)
        
        # 准备双谱 (优先用 Raw，如果没有则用 Norm，根据配置)
        # Encoder 会处理 Normalization 的逻辑，通常我们把 Raw 和 Norm 拼起来或者只选一个
        # 这里为了演示，假设我们优先取 bispectrum_norm (归一化后的)，如果只有 raw 就取 raw
        if 'bispectrum_norm' in feats:
            bispec_input = feats['bispectrum_norm']
        elif 'bispectrum_raw' in feats:
            bispec_input = feats['bispectrum_raw']
        else:
            raise ValueError("No bispectrum feature found! Check config.")
            
        # 准备线性特征 (Concatenate & Flatten)
        linear_list = []
        if 'coherence' in feats: linear_list.append(feats['coherence'].flatten(1))
        if 'cross_corr' in feats: linear_list.append(feats['cross_corr'].flatten(1))
        if 'mi' in feats: linear_list.append(feats['mi'].flatten(1))
        
        linear_flat = torch.cat(linear_list, dim=1) # [B*L, linear_input_dim]
        
        # 编码 (Encoder) -> [B*L, Embed_Dim]
        embeddings = self.encoder(bispec_input, linear_flat)
        
        # 恢复时序维度 -> [B, L, Embed_Dim]
        seq_input = embeddings.view(B, L, -1)
        
        # --- B. 时序建模 (Sequence Modeling) ---
        if self.model_type == 'lstm':
            seq_out, _ = self.backbone(seq_input)
            # 取最后一个时间步
            final_feat = seq_out[:, -1, :] 
            
        elif self.model_type == 'transformer':
            # 加位置编码
            seq_input = self.pos_encoder(seq_input)
            seq_out = self.backbone(seq_input)
            # Mean Pooling over time
            final_feat = seq_out.mean(dim=1)
            
        elif self.model_type == 'mamba':
            seq_out = self.backbone(seq_input)
            # Mamba 通常取最后一步
            final_feat = seq_out[:, -1, :]
            
        # --- C. 分类 ---
        logits = self.classifier(final_feat)
        return logits