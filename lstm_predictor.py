"""
LSTM因子预测器模块 - 在PPO训练结束后对最佳因子组合进行训练
功能：
1. 接收PPO挖掘的最佳因子组合
2. 使用LSTM学习因子组合的时序模式
3. 生成最终的预测信号和交易策略

实盘适配重点：
- 消除标准化过程中的未来函数 (Stateful Normalization)
- 批量推理加速 (Batch Inference)
- 混合损失函数 (MSE + IC)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 确保从 networks 导入 LSTMFactorCombiner
# 如果 combiner 中没有导出 compute_factor_from_tokens，需要确保该辅助函数可用
try:
    from networks import LSTMFactorCombiner
except ImportError:
    # 简单的 fallback 定义，防止导入报错（实际使用时应确保 networks.py 存在）
    class LSTMFactorCombiner(nn.Module):
        def __init__(self, n_factors, hidden_dim, lstm_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(n_factors, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out)

from config import TrainingConfig

logger = logging.getLogger(__name__)


# ==========================================
# 辅助工具类
# ==========================================

class ICLoss(nn.Module):
    """IC 损失函数 (Pearson Correlation Loss) - 用于最大化预测值与目标的相关性"""
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # 归一化 (Batch 内)
        preds_mean = preds.mean()
        targets_mean = targets.mean()
        
        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean
        
        # 计算余弦相似度
        numerator = torch.sum(preds_centered * targets_centered)
        denominator = torch.sqrt(torch.sum(preds_centered ** 2)) * torch.sqrt(torch.sum(targets_centered ** 2))
        
        pearson_corr = numerator / (denominator + 1e-8)
        
        # 我们希望相关性最大化(接近1)，所以 Loss = 1 - Correlation
        return 1.0 - pearson_corr


class FactorSequenceDataset(Dataset):
    """因子序列数据集 - 支持训练和推理模式"""

    def __init__(self, factor_values: np.ndarray, targets: Optional[np.ndarray] = None,
                 sequence_length: int = 20):
        """
        Args:
            factor_values: [T, n_factors] 因子矩阵
            targets: [T] 目标收益率 (可选)
            sequence_length: 序列窗口长度
        """
        self.factor_values = torch.FloatTensor(factor_values)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.sequence_length = sequence_length
        
        # 有效样本数 = 总长度 - 窗口长度 + 1
        self.n_samples = len(factor_values) - sequence_length + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # 输入: 从 idx 到 idx+seq_len 的窗口
        x = self.factor_values[idx : idx + self.sequence_length]
        
        if self.targets is not None:
            # 目标: 序列最后一个时间步对应的收益 (假设 targets 已经是预先对齐好的未来收益)
            y = self.targets[idx + self.sequence_length - 1]
            return x, y
        else:
            return x


# ==========================================
# 主类：LSTMFactorPredictor
# ==========================================

class LSTMFactorPredictor:
    """LSTM因子预测器 - 具备实盘能力的独立预测模块"""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.n_factors = None
        
        # 🔥 关键：保存训练集的统计量，用于标准化验证集和实盘数据，防止未来函数
        self.factor_stats = {}  # {'mean': Series, 'std': Series}
        
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_ic': [], 'val_ic': [],
            'train_sharpe': [], 'val_sharpe': [],
        }

        logger.info(f"📊 LSTM Predictor initialized on {self.device}")

    def prepare_factor_matrix(self, alpha_pool: List[Dict], data: pd.DataFrame,
                              operators: Dict, fit_scaler: bool = False) -> pd.DataFrame:
        """
        准备因子矩阵并进行标准化
        
        Args:
            alpha_pool: 因子池信息
            data: 原始行情数据
            operators: 算子字典
            fit_scaler: True=计算并保存统计量(训练集); False=复用统计量(验证/测试/实盘)
        
        Returns:
            标准化后的因子 DataFrame
        """
        # 导入因子计算函数
        from factor_computation import compute_factor_from_tokens

        factor_dict = {}

        # 1. 计算原始因子值
        for i, alpha_info in enumerate(alpha_pool):
            tokens = alpha_info['tokens']
            try:
                # 调用 combiner 中的计算逻辑
                factor = compute_factor_from_tokens(tokens, data, operators)
                
                if factor is None:
                    logger.warning(f"Factor {i} computation returned None, using zeros")
                    factor_dict[f'factor_{i}'] = pd.Series(0.0, index=data.index)
                else:
                    factor_dict[f'factor_{i}'] = factor
            except Exception as e:
                logger.warning(f"Factor {i} computation failed: {e}, using zeros")
                factor_dict[f'factor_{i}'] = pd.Series(0.0, index=data.index)

        factor_matrix = pd.DataFrame(factor_dict)
        
        # 2. 基础清洗：处理 Inf 和 NaN (因果填充)
        factor_matrix = factor_matrix.replace([np.inf, -np.inf], np.nan)
        factor_matrix = factor_matrix.ffill().fillna(0.0)

        # 3. 标准化 (Strict No-Lookahead Bias)
        if fit_scaler:
            # 训练模式：计算并保存统计量
            self.factor_stats['mean'] = factor_matrix.mean()
            self.factor_stats['std'] = factor_matrix.std() + 1e-8
            logger.info("✅ Computed and saved factor normalization stats from Training Data")
        
        # 检查统计量是否存在
        if not self.factor_stats:
            # 如果没有统计量（例如直接预测而未加载模型），发出警告并使用当前数据（有风险）
            logger.warning("⚠️ Normalization stats not found! Using current batch stats (RISKY for Val/Test!)")
            current_mean = factor_matrix.mean()
            current_std = factor_matrix.std() + 1e-8
        else:
            # 使用保存的统计量
            current_mean = self.factor_stats['mean']
            current_std = self.factor_stats['std']

        # 应用标准化 (Z-Score)
        # 注意：对齐列名，防止 alpha_pool 变化导致 key error
        try:
            factor_matrix = (factor_matrix - current_mean) / current_std
        except Exception as e:
            logger.error(f"Normalization alignment error: {e}. Using raw values.")
            
        # 截断极值 (Clip outliers)
        factor_matrix = factor_matrix.clip(-5, 5)

        logger.info(f"✅ Prepared factor matrix: {factor_matrix.shape}")
        return factor_matrix

    def build_model(self, n_factors: int):
        """构建LSTM模型网络"""
        self.n_factors = n_factors

        self.model = LSTMFactorCombiner(
            n_factors=n_factors,
            hidden_dim=self.config.combiner_hidden_dim,
            lstm_layers=self.config.combiner_lstm_layers,
            dropout=self.config.combiner_dropout
        ).to(self.device)

        logger.info(f"🏗️ Built LSTM model: {n_factors} factors -> {self.config.combiner_hidden_dim}D hidden")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_factors: pd.DataFrame, train_targets: pd.Series,
              val_factors: pd.DataFrame, val_targets: pd.Series,
              epochs: int = 100, batch_size: int = 64, sequence_length: int = 20,
              early_stop_patience: int = 15) -> Dict:
        """
        训练 LSTM 模型
        """
        if self.model is None:
            self.build_model(train_factors.shape[1])

        # 1. 构建数据集
        train_dataset = FactorSequenceDataset(train_factors.values, train_targets.values, sequence_length)
        val_dataset = FactorSequenceDataset(val_factors.values, val_targets.values, sequence_length)

        # 训练集打乱，验证集不打乱
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 2. 优化器与调度器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.combiner_lr,
            weight_decay=self.config.combiner_weight_decay
        )
        
        # 学习率衰减
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 3. 损失函数：MSE + IC Loss
        mse_criterion = nn.MSELoss()
        ic_criterion = ICLoss()

        logger.info(f"🚀 Starting LSTM training | Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

        best_val_score = -float('inf')  # 优先优化 IC
        best_val_loss = float('inf')
        no_improve_count = 0
        best_model_state = None

        # 4. 训练循环
        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            train_losses = []
            train_preds_all = []
            train_targets_all = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                # 前向传播: 取序列最后一步的输出 [batch, 1]
                predictions = self.model(x_batch)[:, -1]
                
                # 混合 Loss: 0.5 MSE + 0.5 IC Loss
                # MSE 保证数值稳定性，IC Loss 保证排序能力
                loss = mse_criterion(predictions, y_batch) + 0.5 * ic_criterion(predictions, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_preds_all.append(predictions.detach().cpu().numpy())
                train_targets_all.append(y_batch.detach().cpu().numpy())

            # --- Validation ---
            self.model.eval()
            val_losses = []
            val_preds_all = []
            val_targets_all = []

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    predictions = self.model(x_batch)[:, -1]
                    
                    loss = mse_criterion(predictions, y_batch) + 0.5 * ic_criterion(predictions, y_batch)
                    val_losses.append(loss.item())
                    
                    val_preds_all.append(predictions.cpu().numpy())
                    val_targets_all.append(y_batch.cpu().numpy())

            # --- Metrics Calculation ---
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            # 合并 Batch 结果
            train_preds_flat = np.concatenate(train_preds_all)
            train_targets_flat = np.concatenate(train_targets_all)
            val_preds_flat = np.concatenate(val_preds_all)
            val_targets_flat = np.concatenate(val_targets_all)

            # 计算 IC (Spearman Rank Correlation)
            train_ic = spearmanr(train_preds_flat, train_targets_flat)[0]
            val_ic = spearmanr(val_preds_flat, val_targets_flat)[0]
            train_ic = train_ic if not np.isnan(train_ic) else 0.0
            val_ic = val_ic if not np.isnan(val_ic) else 0.0

            # 计算 Sharpe (近似值)
            val_returns = val_preds_flat * val_targets_flat
            val_sharpe = np.mean(val_returns) / (np.std(val_returns) + 1e-8) * np.sqrt(252 * 24) # 假设小时线数据

            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_ic'].append(train_ic)
            self.training_history['val_ic'].append(val_ic)
            self.training_history['val_sharpe'].append(val_sharpe)

            # 学习率调度
            scheduler.step(val_loss)

            # 早停检查 (以 IC 为准)
            current_score = val_ic
            if current_score > best_val_score:
                best_val_score = current_score
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                no_improve_count = 0
                
                logger.info(f"✨ Epoch {epoch+1}/{epochs} - New Best Val IC: {val_ic:.4f} (Loss: {val_loss:.5f})")
            else:
                no_improve_count += 1

            # 定期日志
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1} | Train IC: {train_ic:.4f} | Val IC: {val_ic:.4f} | Val Sharpe: {val_sharpe:.2f}")

            if no_improve_count >= early_stop_patience:
                logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"✅ Restored best model with Val IC={best_val_score:.4f}")

        return {
            'best_val_loss': best_val_loss,
            'best_val_ic': best_val_score,
            'epochs_trained': epoch + 1,
            'final_val_ic': val_ic,
        }

    def predict(self, factor_matrix: pd.DataFrame, sequence_length: int = 20, batch_size: int = 1024) -> pd.Series:
        """
        生成预测信号 (批量加速版)
        
        Args:
            factor_matrix: 因子矩阵 (必须已标准化)
            sequence_length: 序列长度
            batch_size: 推理批大小
            
        Returns:
            预测结果 Series (索引与输入对齐，前部填充0)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()
        
        # 使用 Dataset 封装数据，自动处理切片
        dataset = FactorSequenceDataset(factor_matrix.values, targets=None, sequence_length=sequence_length)
        
        # 使用 DataLoader 进行批量推理 (num_workers=0 避免多进程开销，对于纯计算通常够快)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        all_predictions = []
        
        with torch.no_grad():
            for x_batch in loader:
                x_batch = x_batch.to(self.device)
                # 模型输出 [batch, seq_len, 1] -> 取最后一步 -> [batch]
                preds = self.model(x_batch)[:, -1]
                all_predictions.extend(preds.cpu().numpy())
                
        # 数据对齐：由于 Dataset 会消耗掉前 (sequence_length - 1) 个点，需要补 0
        pad_length = sequence_length - 1
        full_predictions = [0.0] * pad_length + all_predictions
        
        # 确保长度一致
        if len(full_predictions) != len(factor_matrix):
            logger.warning(f"Prediction length mismatch: {len(full_predictions)} vs {len(factor_matrix)}")
            # 截断或填充逻辑
            full_predictions = full_predictions[:len(factor_matrix)]
        
        return pd.Series(full_predictions, index=factor_matrix.index)

    def generate_signals(self, predictions: pd.Series,
                        lookback: int = 100,
                        q_low: float = 0.3,
                        q_high: float = 0.7,
                        max_position: float = 1.0) -> pd.Series:
        """
        根据预测值生成交易信号 (Rolling Quantile Strategy)
        """
        # 滚动计算分位数，适应市场体制转换
        roll = predictions.rolling(lookback, min_periods=20)
        low_thresh = roll.quantile(q_low)
        high_thresh = roll.quantile(q_high)
        mid_val = roll.median()

        signals = pd.Series(0.0, index=predictions.index)

        # 做多：预测值 > 高分位数
        signals[predictions > high_thresh] = max_position
        
        # 做空：预测值 < 低分位数
        signals[predictions < low_thresh] = -max_position

        # 中性区域：微弱持仓 (可选)
        # mask_neutral = (predictions >= low_thresh) & (predictions <= high_thresh)
        # signals[mask_neutral & (predictions > mid_val)] = max_position * 0.1
        # signals[mask_neutral & (predictions < mid_val)] = -max_position * 0.1

        return signals.fillna(0.0)

    def evaluate(self, factor_matrix: pd.DataFrame, targets: pd.Series,
                sequence_length: int = 20) -> Dict:
        """
        评估模型性能 (Backtest)
        """
        # 生成预测
        predictions = self.predict(factor_matrix, sequence_length)

        # 对齐数据
        aligned = pd.DataFrame({
            'pred': predictions,
            'target': targets
        }).dropna()

        if len(aligned) < 100:
            return {'error': 'insufficient_data'}

        # 1. 基础 IC
        ic = spearmanr(aligned['pred'], aligned['target'])[0]
        ic = 0.0 if np.isnan(ic) else ic

        # 2. 预测值 Sharpe
        rets = aligned['pred'] * aligned['target']
        pred_sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252 * 24)

        # 3. 策略回测
        signals = self.generate_signals(predictions)
        # 对齐信号和收益 (假设 signal 是基于 t 时刻信息，target 是 t+1 收益)
        strat_rets = signals * targets
        strat_rets = strat_rets.dropna()
        
        strat_sharpe = 0.0
        cum_ret = 0.0
        if len(strat_rets) > 0:
            strat_sharpe = strat_rets.mean() / (strat_rets.std() + 1e-8) * np.sqrt(252 * 24)
            cum_ret = (1 + strat_rets).prod() - 1

        return {
            'ic': ic,
            'sharpe': pred_sharpe,
            'signal_sharpe': strat_sharpe,
            'cumulative_return': cum_ret,
            'n_samples': len(aligned),
        }

    def save_model(self, save_path: str):
        """保存模型和关键统计量"""
        if self.model is None:
            logger.warning("No model to save")
            return

        save_dict = {
            'model_state': self.model.state_dict(),
            'n_factors': self.n_factors,
            'factor_stats': self.factor_stats, # 🔥 必须保存统计量，否则无法正确推理
            'config': {
                'hidden_dim': self.config.combiner_hidden_dim,
                'lstm_layers': self.config.combiner_lstm_layers,
                'dropout': self.config.combiner_dropout,
            },
            'training_history': self.training_history,
        }

        torch.save(save_dict, save_path)
        logger.info(f"💾 Saved LSTM model and stats to {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        save_dict = torch.load(load_path, map_location=self.device)

        self.n_factors = save_dict['n_factors']
        self.factor_stats = save_dict.get('factor_stats', {}) # 🔥 恢复统计量
        
        # 重建模型结构
        self.build_model(self.n_factors)
        self.model.load_state_dict(save_dict['model_state'])
        self.training_history = save_dict.get('training_history', {})

        logger.info(f"📂 Loaded LSTM model from {load_path}")
        logger.info(f"   Factors: {self.n_factors}")
        logger.info(f"   Normalization Stats Present: {bool(self.factor_stats)}")

    def plot_training_history(self, save_path: str = 'lstm_training_history.png'):
        """绘制训练曲线"""
        if not self.training_history['train_loss']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss (MSE + IC)')
        axes[0, 0].legend()

        # IC
        axes[0, 1].plot(self.training_history['train_ic'], label='Train')
        axes[0, 1].plot(self.training_history['val_ic'], label='Val')
        axes[0, 1].set_title('IC (Spearman)')
        axes[0, 1].axhline(0, color='red', linestyle='--')
        axes[0, 1].legend()

        # Sharpe
        axes[1, 0].plot(self.training_history['val_sharpe'], label='Val Sharpe', color='green')
        axes[1, 0].set_title('Validation Sharpe Ratio')
        axes[1, 0].legend()

        # Summary
        axes[1, 1].axis('off')
        best_ic = max(self.training_history['val_ic']) if self.training_history['val_ic'] else 0
        info_text = f"Best Val IC: {best_ic:.4f}\nEpochs: {len(self.training_history['train_loss'])}"
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()