import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from config import TrainingConfig

logger = logging.getLogger(__name__)

class ICDiversityEvaluator:
    """
    基于组合增量夏普比率 (Incremental Sharpe) 的评估器
    遵循 AlphaGen 框架：挖掘能提升现有组合表现的协同因子
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.prediction_horizon = config.prediction_horizon
        self.bar_minutes = config.bar_minutes
        self.transaction_cost = getattr(config, 'transaction_cost', 0.0005)
        self.max_position = getattr(config, 'max_position', 0.1)
        self.sharpe_signal_lookback = getattr(config, 'sharpe_signal_lookback', 100)
        self.sharpe_signal_quantiles = getattr(config, 'sharpe_signal_quantiles', (0.3, 0.7))
        
        # 年化系数
        self.bars_per_year = 365 * 24 * 60 / max(self.bar_minutes, 1)
        
        # 引用 Combiner，不再自己维护 Pool 和 Model
        self.combiner = None 

        logger.info(f"Synergy Evaluator initialized (Target: Incremental Sharpe)")

    def set_combiner(self, combiner):
        """注入 Combiner 实例"""
        self.combiner = combiner

    def calculate_ic(self, predictions: pd.Series, targets: pd.Series) -> float:
        """计算 IC (仅作为观察指标，不参与 Reward)"""
        try:
            aligned = pd.DataFrame({'pred': predictions, 'target': targets}).dropna()
            if len(aligned) < 20: return 0.0
            if aligned['pred'].std() < 1e-8: return 0.0
            
            ic = aligned['pred'].corr(aligned['target'])
            return float(ic) if np.isfinite(ic) else 0.0
        except:
            return 0.0

    def calculate_turnover(self, predictions: pd.Series) -> float:
        """计算换手率 (用于惩罚项)"""
        try:
            lookback = max(int(self.sharpe_signal_lookback), 20)
            if len(predictions) < lookback + 20: return 0.0
            
            # 简单的分位数信号生成
            q_low, q_high = self.sharpe_signal_quantiles
            roll = predictions.rolling(window=lookback, min_periods=20)
            low, high = roll.quantile(q_low), roll.quantile(q_high)
            
            signals = pd.Series(0.0, index=predictions.index)
            signals[predictions > high] = 1.0
            signals[predictions < low] = -1.0
            signals = signals.fillna(0.0)
            
            # 计算平均换手
            turnover = signals.diff().abs().mean()
            return float(turnover) if np.isfinite(turnover) else 0.0
        except:
            return 0.0

    def calculate_rolling_sharpe_stability(self, predictions: pd.Series, targets: pd.Series, 
                                          window_days: int = 90, stability_penalty: float = 1.5) -> float:
        """
        🔥 计算滚动夏普的稳定性得分
        Score = Mean(Rolling_Sharpe) - lambda * Std(Rolling_Sharpe)
        """
        try:
            # 1. 计算净值曲线 (Net Returns)
            net_returns = self._get_net_returns(predictions, targets)
            
            if len(net_returns) < window_days * 2: return 0.0
            
            # 2. 计算滚动 Sharpe
            bars_per_day = 24 * 60 / max(self.bar_minutes, 1)
            window_bars = int(window_days * bars_per_day)
            
            # 滚动计算均值和标准差
            rolling_mean = net_returns.rolling(window=window_bars).mean()
            rolling_std = net_returns.rolling(window=window_bars).std()
            
            # 滚动年化 Sharpe
            rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)) * np.sqrt(self.bars_per_year)
            rolling_sharpe = rolling_sharpe.dropna()
            
            # 剔除极端值
            rolling_sharpe = rolling_sharpe.clip(-5, 5)
            
            if len(rolling_sharpe) < 10: return 0.0
            
            # 3. 计算稳定性得分
            mean_s = rolling_sharpe.mean()
            std_s = rolling_sharpe.std()
            
            # 核心公式：平均表现 - 不确定性惩罚
            stability_score = mean_s - stability_penalty * std_s
            
            return float(stability_score)
            
        except Exception as e:
            # logger.warning(f"Error in stability calc: {e}")
            return 0.0

    def _get_net_returns(self, predictions: pd.Series, targets: pd.Series) -> pd.Series:
        """辅助函数：提取净值收益逻辑"""
        valid_idx = predictions.index.intersection(targets.index)
        if len(valid_idx) < 100: return pd.Series([], dtype=float)
        
        pred_val = predictions.loc[valid_idx]
        target_val = targets.loc[valid_idx]
        
        lookback = max(int(self.sharpe_signal_lookback), 20)
        min_periods = min(lookback, 20)
        
        # 简单的滚动 z-score 信号
        roll = pred_val.rolling(window=lookback, min_periods=min_periods)
        mu = roll.mean()
        sigma = roll.std() + 1e-9
        z_scores = (pred_val - mu) / sigma
        
        signals = pd.Series(0.0, index=pred_val.index)
        signals[z_scores > 1.0] = self.max_position
        signals[z_scores < -1.0] = -self.max_position
        
        gross_returns = signals * target_val
        cost = signals.diff().abs().fillna(0.0) * self.transaction_cost
        return (gross_returns - cost).dropna()

    def _get_incremental_sharpe(self, predictions: pd.Series, targets: pd.Series, use_val: bool) -> float:
        """
        🔥 实现增量计算：调用 Combiner 试算 '假如加入该因子，Score 提升多少'
        """
        if self.combiner is None:
            # 如果没有 Combiner，退化为单因子评估
            return self.calculate_rolling_sharpe_stability(predictions, targets)

        # 调用 Combiner 的试算模式 (Trial Mode)
        # 注意：这里我们将 Val 部分留空，因为 Reward 通常只看 Training set 的增量
        result = self.combiner.evaluate_new_factor(
            alpha_info={},  # 暂时不需要具体 info
            train_factor=predictions,
            val_factor=pd.Series(dtype=float) 
        )
        
        return result.get('train_incremental_sharpe', 0.0)
    
    def evaluate(self, predictions: pd.Series, targets: pd.Series,
                 use_val: bool = False, add_to_history: bool = False,
                 is_single_factor: bool = True) -> Dict[str, float]:
        """
        评估函数
        """
        # 1. 基础指标
        ic = self.calculate_ic(predictions, targets)
        
        # 2. 计算 Synergy Reward (增量 Sharpe Stability)
        synergy_reward = self._get_incremental_sharpe(predictions, targets, use_val)
        
        # 3. 计算换手率惩罚
        turnover = self.calculate_turnover(predictions)
        penalty = 0.05 * max(0, turnover - 0.2)
        
        # 4. 最终得分
        final_score = synergy_reward - penalty
        
        # 单因子本身的 Sharpe (仅供参考)
        single_sharpe = self.calculate_rolling_sharpe_stability(predictions, targets)

        result = {
            'ic': ic,
            'kl_divergence': 0.0,
            'avg_kl': 0.0,
            'avg_correlation': 0.0,
            'diversity_score': 1.0,
            'sharpe': single_sharpe,  
            'composite_score': final_score,  # 🔥 真实的 Reward
            'metric_type': 'incremental_sharpe_stability'
        }

        # 5. 状态更新 (Commit Mode)
        if add_to_history and self.combiner is not None:
            self.add_factor(predictions, use_val=use_val, targets=targets)

        return result

    def add_factor(self, predictions: pd.Series, use_val: bool = False, targets: pd.Series = None):
        """
        更新因子池 - 委托给 Combiner
        """
        if self.combiner:
            # 在这里我们假设外部循环或者 Combiner 已经处理好了 targets 的设置
            # 这里的 add_factor 只是通知 Combiner 确认采纳当前因子
            # 注意：因为 evaluate 时传入的是 predictions，这里我们再次传入
            # 实际工程中可以用 cache 优化，但这里为了逻辑清晰直接传递
            self.combiner.add_alpha_and_optimize(
                alpha_info={}, 
                train_factor=predictions,
                val_factor=pd.Series(dtype=float) # Val 暂不用于更新权重逻辑
            )

    # 废弃方法的存根
    def calculate_kl_divergence(self, *args, **kwargs): return 0.0
    def get_average_kl(self, *args, **kwargs): return 0.0
    def add_kl_to_history(self, *args, **kwargs): pass
    def calculate_avg_correlation(self, *args, **kwargs): return 0.0
    # 兼容性存根：如果外部代码调用了这个属性，返回空列表
    @property
    def historical_factors_train(self): return []
    @property
    def historical_factors_val(self): return []