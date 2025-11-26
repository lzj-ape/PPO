"""
更新后的 ICDiversityEvaluator - 使用统一的信号生成和评估逻辑
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from config import TrainingConfig
from signals import SignalGenerator, PerformanceEvaluator

logger = logging.getLogger(__name__)


class ICDiversityEvaluator:
    """
    基于组合增量夏普比率 (Incremental Sharpe) 的评估器
    遵循 AlphaGen 框架：挖掘能提升现有组合表现的协同因子
    🆕 更新:
    ----
    - 使用统一的 SignalGenerator 生成信号
    - 使用统一的 PerformanceEvaluator 计算指标
    - 训练和回测逻辑完全一致
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
        self.bars_per_year = 365 * 24 * 60 / self.bar_minutes
        
        # 🆕 创建统一的信号生成器
        self.signal_generator = SignalGenerator(
            max_position=self.max_position,
            lookback=self.sharpe_signal_lookback,
            q_low=self.sharpe_signal_quantiles[0],
            q_high=self.sharpe_signal_quantiles[1],
            neutral_fraction=0.1,  # 中性区域1/10仓位
            min_periods=20
        )
        
        # 🆕 创建统一的性能评估器
        self.performance_evaluator = PerformanceEvaluator(
            prediction_horizon=self.prediction_horizon,
            bar_minutes=self.bar_minutes,
            transaction_cost=self.transaction_cost,
            signal_generator=self.signal_generator
        )
        
        # 引用 Combiner，不再自己维护 Pool 和 Model
        self.combiner = None 

        logger.info(f"Synergy Evaluator initialized (Target: Incremental Sharpe)")
        logger.info(f"  - Signal Generator: max_pos={self.max_position}, "
                   f"lookback={self.sharpe_signal_lookback}")

    def set_combiner(self, combiner):
        """注入 Combiner 实例"""
        self.combiner = combiner

    def calculate_ic(self, predictions: pd.Series, targets: pd.Series) -> float:
        """
        计算 IC (仅作为观察指标，不参与 Reward)
        委托给 PerformanceEvaluator
        """
        return self.performance_evaluator.calculate_ic(predictions, targets)

    def calculate_turnover(self, predictions: pd.Series) -> float:
        """
        计算换手率 (用于惩罚项)
        🆕 使用信号生成器计算
        """
        if len(predictions) < self.sharpe_signal_lookback + 20:
            return 0.0
        
        # 生成信号
        signals = self.signal_generator.generate_signals(predictions)
        
        # 计算换手率
        turnover = self.signal_generator.calculate_turnover(signals)
        
        return turnover

    def calculate_rolling_sharpe_stability(self, 
                                          predictions: pd.Series, 
                                          targets: pd.Series,
                                          window_days: int = 3, 
                                          stability_penalty: float = 1.5) -> float:
        """
        计算滚动夏普的稳定性得分
        🆕 委托给 PerformanceEvaluator
        
        Score = Mean(Rolling_Sharpe) - lambda * Std(Rolling_Sharpe)
        """
        return self.performance_evaluator.calculate_rolling_sharpe_stability(
            predictions, targets, window_days, stability_penalty
        )

    def _get_incremental_sharpe(self, predictions: pd.Series, targets: pd.Series, use_val: bool) -> float:
        """
        🔥 实现增量计算：调用 Combiner 试算 '假如加入该因子，Score 提升多少'
        """
        if self.combiner is None:
            # 如果没有 Combiner，退化为单因子评估
            score = self.calculate_rolling_sharpe_stability(predictions, targets)
            # 🔥 修复：如果计算失败，返回0
            return score if score is not None else 0.0

        # 调用 Combiner 的试算模式 (Trial Mode)
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
        🆕 使用统一的性能评估器
        """
        # 1. 基础指标
        ic = self.calculate_ic(predictions, targets)
        
        # 2. 计算 Synergy Reward (增量 Sharpe Stability)
        synergy_reward = self._get_incremental_sharpe(predictions, targets, use_val)
        
        # 3. 计算换手率惩罚 (使用统一的信号生成器)
        turnover = self.calculate_turnover(predictions)
        penalty = 0.05 * max(0, turnover - 0.2)
        
        # 4. 最终得分
        final_score = synergy_reward - penalty
        
        # 单因子本身的 Sharpe (仅供参考)
        single_sharpe = self.calculate_rolling_sharpe_stability(predictions, targets)
        # 🔥 修复：如果计算失败，使用0
        if single_sharpe is None:
            single_sharpe = 0.0

        result = {
            'ic': ic,
            'kl_divergence': 0.0,
            'avg_kl': 0.0,
            'avg_correlation': 0.0,
            'diversity_score': 1.0,
            'sharpe': single_sharpe,  
            'composite_score': final_score,  # 🔥 真实的 Reward
            'metric_type': 'incremental_sharpe_stability',
            'turnover': turnover,
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
            self.combiner.add_alpha_and_optimize(
                alpha_info={}, 
                train_factor=predictions,
                val_factor=pd.Series(dtype=float)
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