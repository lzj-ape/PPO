"""
高级Reward计算模块 - 针对加密货币因子挖掘

支持三种Reward策略：
1. 增量夏普比率 (Incremental Sharpe) - 最符合论文原意
2. 惩罚型夏普 (Penalized Sharpe) - 针对Crypto高成本特性
3. 滚动夏普稳定性 (Rolling Sharpe Stability) - 防止"一波流"过拟合
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Reward计算配置"""
    # 方案选择
    use_incremental_sharpe: bool = True      # 方案一：增量Sharpe
    use_penalty: bool = True                  # 方案二：惩罚项
    use_rolling_stability: bool = False       # 方案三：滚动稳定性（数据量大时开启）

    # 方案一参数
    incremental_weight: float = 5.0           # 增量Sharpe权重

    # 方案二参数
    complexity_lambda: float = 0.3            # 复杂度惩罚系数
    turnover_gamma: float = 2.0               # 换手率惩罚系数
    max_expr_length: int = 30                 # 最大表达式长度

    # 方案三参数
    rolling_window_ratio: float = 0.25        # 滚动窗口占比（例如0.25 = 每个窗口25%数据）
    rolling_stability_weight: float = 2.0     # 稳定性权重

    # 通用参数
    overfitting_threshold: float = 1.5        # 过拟合阈值（train_sharpe - val_sharpe）
    overfitting_penalty: float = 1.0          # 过拟合惩罚系数


class AdvancedRewardCalculator:
    """高级Reward计算器"""

    def __init__(self, config: RewardConfig):
        self.config = config
        logger.info(f"Advanced Reward Calculator initialized:")
        logger.info(f"  - Incremental Sharpe: {config.use_incremental_sharpe}")
        logger.info(f"  - Penalty: {config.use_penalty}")
        logger.info(f"  - Rolling Stability: {config.use_rolling_stability}")

    def calculate_reward(self,
                        new_train_eval: Dict,
                        new_val_eval: Dict,
                        old_train_eval: Dict,
                        old_val_eval: Dict,
                        alpha_info: Dict,
                        combination_series: Optional[pd.Series] = None,
                        evaluator = None) -> Dict:
        """
        计算综合Reward

        参数:
            new_train_eval: 新组合的训练集评估结果
            new_val_eval: 新组合的验证集评估结果
            old_train_eval: 旧组合的训练集评估结果
            old_val_eval: 旧组合的验证集评估结果
            alpha_info: 因子信息（包含tokens等）
            combination_series: 组合因子的时间序列（用于计算换手率和滚动稳定性）
            evaluator: 评估器实例（用于计算换手率）

        返回:
            包含reward和详细组件的字典
        """
        components = {}
        total_reward = 0.0

        # === 方案一：增量Sharpe ===
        if self.config.use_incremental_sharpe:
            incremental_reward = self._calculate_incremental_sharpe(
                new_val_eval, old_val_eval
            )
            components['incremental_sharpe'] = incremental_reward
            total_reward += incremental_reward * self.config.incremental_weight
        else:
            # 使用绝对Sharpe（原始方案）
            val_sharpe = new_val_eval.get('sharpe', 0)
            sharpe_normalized = np.tanh(val_sharpe / 2.0)
            absolute_reward = sharpe_normalized * 3.0
            components['absolute_sharpe'] = absolute_reward
            total_reward += absolute_reward

        # === 方案二：惩罚项 ===
        if self.config.use_penalty:
            # 2.1 复杂度惩罚
            complexity_penalty = self._calculate_complexity_penalty(alpha_info)
            components['complexity_penalty'] = complexity_penalty
            total_reward += complexity_penalty

            # 2.2 换手率惩罚（需要combination_series和evaluator）
            if combination_series is not None and evaluator is not None:
                turnover_penalty = self._calculate_turnover_penalty(
                    combination_series, evaluator
                )
                components['turnover_penalty'] = turnover_penalty
                total_reward += turnover_penalty
            else:
                components['turnover_penalty'] = 0.0

        # === 方案三：滚动Sharpe稳定性 ===
        if self.config.use_rolling_stability:
            if combination_series is not None and evaluator is not None:
                stability_reward = self._calculate_rolling_stability(
                    combination_series, evaluator
                )
                components['stability_reward'] = stability_reward
                total_reward += stability_reward * self.config.rolling_stability_weight
            else:
                components['stability_reward'] = 0.0
                logger.warning("Rolling stability requires combination_series and evaluator")

        # === 通用：过拟合惩罚 ===
        overfitting_penalty = self._calculate_overfitting_penalty(
            new_train_eval, new_val_eval
        )
        components['overfitting_penalty'] = overfitting_penalty
        total_reward += overfitting_penalty

        # === 记录所有关键指标 ===
        components.update({
            'old_val_sharpe': float(old_val_eval.get('sharpe', 0)),
            'new_val_sharpe': float(new_val_eval.get('sharpe', 0)),
            'new_train_sharpe': float(new_train_eval.get('sharpe', 0)),
            'sharpe_gap': float(new_train_eval.get('sharpe', 0) - new_val_eval.get('sharpe', 0)),
            'total_reward': float(total_reward),
        })

        return {
            'reward': total_reward,
            'components': components
        }

    def _calculate_incremental_sharpe(self, new_val_eval: Dict, old_val_eval: Dict) -> float:
        """
        方案一：计算增量Sharpe

        逻辑：
        - Reward = Sharpe(F_new) - Sharpe(F_old)
        - 强迫RL寻找能提升整体组合的因子，而不是单独高Sharpe的因子
        """
        old_sharpe = old_val_eval.get('sharpe', 0)
        new_sharpe = new_val_eval.get('sharpe', 0)

        delta_sharpe = new_sharpe - old_sharpe

        # 归一化（避免Reward过大）
        # 使用tanh将delta_sharpe压缩到(-1, 1)
        normalized_delta = np.tanh(delta_sharpe / 2.0)

        return float(normalized_delta)

    def _calculate_complexity_penalty(self, alpha_info: Dict) -> float:
        """
        方案二.1：复杂度惩罚

        逻辑：
        - 惩罚过长的表达式（Crypto单币种容易过拟合）
        - Penalty = -λ * (length / max_length)^2  （超过阈值后二次惩罚）
        """
        expr_length = len(alpha_info.get('tokens', [])) - 2  # 去除<START>和<END>
        max_length = self.config.max_expr_length

        if expr_length <= max_length:
            return 0.0

        # 超过阈值的部分进行二次惩罚
        excess_ratio = (expr_length - max_length) / max_length
        penalty = -self.config.complexity_lambda * (excess_ratio ** 2)

        return float(penalty)

    def _calculate_turnover_penalty(self, combination_series: pd.Series, evaluator) -> float:
        """
        方案二.2：换手率惩罚

        逻辑：
        - 计算因子信号的换手率
        - 高换手 = 高交易成本 = 实盘不可行
        - Penalty = -γ * Turnover
        """
        try:
            turnover = evaluator.calculate_turnover(combination_series)
            penalty = -self.config.turnover_gamma * turnover
            return float(penalty)
        except Exception as e:
            logger.debug(f"Turnover penalty calculation failed: {e}")
            return 0.0

    def _calculate_rolling_stability(self, combination_series: pd.Series, evaluator) -> float:
        """
        方案三：滚动Sharpe稳定性

        逻辑：
        - 将时间序列切分为多个窗口
        - 计算每个窗口的Sharpe
        - Reward = Avg(RollingSharpe) - β * Std(RollingSharpe)
        - 目的：寻找穿越牛熊的稳健因子
        """
        try:
            if len(combination_series) < 200:
                logger.warning("Data too short for rolling stability calculation")
                return 0.0

            window_size = int(len(combination_series) * self.config.rolling_window_ratio)
            window_size = max(window_size, 100)  # 最小窗口100个bar

            # 计算滚动窗口
            n_windows = len(combination_series) // window_size
            if n_windows < 3:
                logger.debug(f"Not enough windows for stability: {n_windows}")
                return 0.0

            rolling_sharpes = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                window_data = combination_series.iloc[start_idx:end_idx]

                # 计算该窗口的Sharpe（需要target，这里简化使用return）
                # 注意：实际使用时应该传入对应窗口的target
                window_return = window_data.diff().dropna()
                if len(window_return) > 10 and window_return.std() > 1e-8:
                    sharpe = window_return.mean() / window_return.std() * np.sqrt(252)
                    if np.isfinite(sharpe):
                        rolling_sharpes.append(sharpe)

            if len(rolling_sharpes) < 3:
                return 0.0

            # 计算平均Sharpe和稳定性
            avg_sharpe = np.mean(rolling_sharpes)
            std_sharpe = np.std(rolling_sharpes)

            # Reward = 平均 - 波动（鼓励稳定）
            beta = 0.5  # 稳定性权重
            stability_score = avg_sharpe - beta * std_sharpe

            # 归一化
            normalized_score = np.tanh(stability_score / 2.0)

            return float(normalized_score)

        except Exception as e:
            logger.debug(f"Rolling stability calculation failed: {e}")
            return 0.0

    def _calculate_overfitting_penalty(self, new_train_eval: Dict, new_val_eval: Dict) -> float:
        """
        过拟合惩罚

        逻辑：
        - 如果 train_sharpe - val_sharpe > threshold，进行惩罚
        - 防止模型在训练集上过度优化
        """
        train_sharpe = new_train_eval.get('sharpe', 0)
        val_sharpe = new_val_eval.get('sharpe', 0)
        sharpe_gap = train_sharpe - val_sharpe

        if sharpe_gap > self.config.overfitting_threshold:
            excess_gap = sharpe_gap - self.config.overfitting_threshold
            penalty = -self.config.overfitting_penalty * min(excess_gap, 3.0)
            return float(penalty)

        return 0.0


def create_reward_calculator(reward_type: str = 'hybrid') -> AdvancedRewardCalculator:
    """
    创建Reward计算器的工厂函数

    参数:
        reward_type:
            - 'incremental': 纯增量Sharpe（方案一）
            - 'penalized': 惩罚型Sharpe（方案二）
            - 'stable': 滚动稳定性（方案三）
            - 'hybrid': 混合模式（方案一+二，推荐）
            - 'full': 全部启用（方案一+二+三，数据量大时使用）

    返回:
        配置好的AdvancedRewardCalculator实例
    """
    if reward_type == 'incremental':
        # 方案一：纯增量Sharpe
        config = RewardConfig(
            use_incremental_sharpe=True,
            use_penalty=False,
            use_rolling_stability=False,
            incremental_weight=5.0
        )

    elif reward_type == 'penalized':
        # 方案二：惩罚型Sharpe
        config = RewardConfig(
            use_incremental_sharpe=False,
            use_penalty=True,
            use_rolling_stability=False,
            complexity_lambda=0.3,
            turnover_gamma=2.0
        )

    elif reward_type == 'stable':
        # 方案三：滚动稳定性
        config = RewardConfig(
            use_incremental_sharpe=False,
            use_penalty=False,
            use_rolling_stability=True,
            rolling_window_ratio=0.25,
            rolling_stability_weight=3.0
        )

    elif reward_type == 'hybrid':
        # 混合模式：方案一+二（推荐）
        config = RewardConfig(
            use_incremental_sharpe=True,
            use_penalty=True,
            use_rolling_stability=False,
            incremental_weight=5.0,
            complexity_lambda=0.3,
            turnover_gamma=2.0,
            max_expr_length=30
        )

    elif reward_type == 'full':
        # 全部启用：方案一+二+三（数据量大时）
        config = RewardConfig(
            use_incremental_sharpe=True,
            use_penalty=True,
            use_rolling_stability=True,
            incremental_weight=4.0,
            complexity_lambda=0.3,
            turnover_gamma=1.5,
            rolling_window_ratio=0.2,
            rolling_stability_weight=2.0
        )

    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    logger.info(f"Created reward calculator with type: {reward_type}")
    return AdvancedRewardCalculator(config)
