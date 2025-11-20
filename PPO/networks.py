"""
神经网络模块 - Actor-Critic网络和相关组件
"""

import torch
import torch.nn as nn
from typing import List

from config import TrainingConfig


class LSTMFactorCombiner(nn.Module):
    """使用LSTM学习因子组合 - 防过拟合增强版"""
    
    def __init__(self, n_factors: int, hidden_dim: int = 64, 
                 lstm_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
        
        # 动态调整hidden_dim防止过拟合
        if n_factors <= 3:
            self.hidden_dim = min(hidden_dim, 16)
        elif n_factors <= 5:
            self.hidden_dim = min(hidden_dim, 32)
        else:
            self.hidden_dim = hidden_dim
        
        # LSTM处理因子序列
        self.lstm = nn.LSTM(
            input_size=n_factors,
            hidden_size=self.hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 输出层：从隐藏状态到预测（增强dropout）
        self.output_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        for module in self.output_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, factor_values):
        """
        Args:
            factor_values: [batch_size, seq_len, n_factors] 或 [seq_len, n_factors]
        Returns:
            predictions: [batch_size, seq_len] 或 [seq_len]
        """
        if factor_values.dim() == 2:
            factor_values = factor_values.unsqueeze(0)  # [1, seq_len, n_factors]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(factor_values)
        
        # 对每个时间步生成预测
        predictions = self.output_net(lstm_out).squeeze(-1)
        
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions


class LSTMFeatureExtractor(nn.Module):
    """LSTM特征提取器"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        for module in self.output_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, padding_mask=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        embedded = self.embedding(x)
        lstm_output, (h_n, c_n) = self.lstm(embedded)
        lstm_output = self.layer_norm(lstm_output)
        
        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            masked_output = lstm_output * mask_expanded
            valid_lengths = mask_expanded.sum(dim=1).clamp(min=1)
            features = masked_output.sum(dim=1) / valid_lengths
        else:
            features = lstm_output.mean(dim=1)
        
        features = self.output_projection(features)
        
        return features, lstm_output



class TypeHead(nn.Module):
    """动作类型预测头"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, features):
        return self.head(features)



class PolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, vocab_size)
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, features):
        return self.head(features)



class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, features):
        return self.head(features).squeeze(-1)



class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, vocab_size: int, config: TrainingConfig, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        self.feature_extractor = LSTMFeatureExtractor(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout
        )
        
        self.type_head = TypeHead(config.hidden_dim, config.dropout)
        self.policy_head = PolicyHead(config.hidden_dim, vocab_size, config.dropout)
        self.value_head = ValueHead(config.hidden_dim, config.dropout)
    
    def forward(self, x, padding_mask=None):
        features, sequence_output = self.feature_extractor(x, padding_mask)
        type_logits = self.type_head(features)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return type_logits, action_logits, value
    
    def forward_batch(self, states_list):
        if not states_list:
            return torch.empty(0), torch.empty(0), torch.empty(0)
        
        batch_size = len(states_list)
        max_len = max(len(state) for state in states_list)
        
        device = next(self.parameters()).device
        padded_states = torch.full((batch_size, max_len), self.pad_token_id, 
                                   dtype=torch.long, device=device)
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        
        for i, state in enumerate(states_list):
            seq_len = len(state)
            padded_states[i, :seq_len] = torch.tensor(state, dtype=torch.long, device=device)
            padding_mask[i, :seq_len] = False
        
        return self.forward(padded_states, padding_mask)


