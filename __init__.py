"""
PPO因子挖掘系统 - 模块化版本
主要改进：
1. 使用KL散度损失函数优化线性组合器
2. 改进的奖励函数（正确处理泛化情况）
3. 移除强制删除因子的逻辑
"""

__version__ = "2.0.0"
__author__ = "Factor Mining Team"

# 导入主要类
from config.config import TrainingConfig
from factor.operators import TimeSeriesOperators
from factor.evaluator import ICDiversityEvaluator
from PPO.networks import ActorCriticNetwork, LSTMFeatureExtractor, LSTMFactorCombiner
from PPO.buffer import PPOBuffer
from factor.combiner import ImprovedCombinationModel
from PPO.miner_new import OptimizedSynergisticFactorMiner
from backtest.backtest import Backtester, MinerBacktester
from utils.utils import setup_logging, get_device

__all__ = [
    'TrainingConfig',
    'TimeSeriesOperators',
    'ICDiversityEvaluator',
    'ActorCriticNetwork',
    'LSTMFeatureExtractor',
    'PPOBuffer',
    'LSTMFactorCombiner',
    'ImprovedCombinationModel',
    'OptimizedSynergisticFactorMiner',
    'Backtester',
    'MinerBacktester',
    'setup_logging',
    'get_device',
]

