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
from config import TrainingConfig
from operators import TimeSeriesOperators
from evaluator import SharpeICEvaluator
from networks import ActorCriticNetwork, LSTMFeatureExtractor
from buffer import PPOBuffer
from combiner import LSTMFactorCombiner, ImprovedCombinationModel
from miner import OptimizedSynergisticFactorMiner
from backtest import SimpleBacktest
from utils import setup_logging, get_device

__all__ = [
    'TrainingConfig',
    'TimeSeriesOperators',
    'SharpeICEvaluator',
    'ActorCriticNetwork',
    'LSTMFeatureExtractor',
    'PPOBuffer',
    'LSTMFactorCombiner',
    'ImprovedCombinationModel',
    'OptimizedSynergisticFactorMiner',
    'RollingSignalGenerator',
    'CryptoBacktest',
    'setup_logging',
    'get_device',
]

