"""
因子挖掘器模块 - 主入口（向后兼容版本）
此文件保持与原miner.py的接口兼容性，内部委托给重构后的模块

重构结构：
- miner_core.py: 核心挖掘器逻辑
- expression_generator.py: 表达式生成
- factor_evaluator.py: 因子评估
- ppo_trainer.py: PPO训练
- visualization.py: 可视化工具
"""

import pandas as pd
import logging
from typing import Dict, List
from pathlib import Path

from config import TrainingConfig
from miner_core import FactorMinerCore

logger = logging.getLogger(__name__)


class OptimizedSynergisticFactorMiner:
    """
    因子挖掘器 - 向后兼容的入口类

    内部委托给重构后的FactorMinerCore
    保持所有原有接口不变，确保现有代码无需修改
    """

    def __init__(self,
                 data: pd.DataFrame,
                 target_col: str = 'future_return',
                 config: TrainingConfig = None,
                 max_factors: int = 15,
                 max_expr_len: int = 20):
        """
        初始化因子挖掘器

        Args:
            data: 完整数据集（包含OHLCV和技术指标）
            target_col: 目标变量列名
            config: 训练配置对象
            max_factors: 最大因子数量
            max_expr_len: 最大表达式长度
        """
        # 创建核心挖掘器实例
        self.core = FactorMinerCore(
            data=data,
            target_col=target_col,
            config=config,
            max_factors=max_factors,
            max_expr_len=max_expr_len
        )

        # 暴露常用属性以保持兼容性
        self.config = self.core.config
        self.device = self.core.device
        self.train_data = self.core.train_data
        self.val_data = self.core.val_data
        self.test_data = self.core.test_data
        self.train_target = self.core.train_target
        self.val_target = self.core.val_target
        self.test_target = self.core.test_target
        self.feature_names = self.core.feature_names
        self.operators = self.core.operators
        self.vocab = self.core.vocab
        self.token_to_id = self.core.token_to_id
        self.id_to_token = self.core.id_to_token
        self.actor_critic = self.core.actor_critic
        self.optimizer = self.core.optimizer
        self.evaluator = self.core.evaluator
        self.combination_model = self.core.combination_model
        self.ppo_buffer = self.core.ppo_buffer
        self.training_history = self.core.training_history
        self.best_val_score = self.core.best_val_score
        self.best_model_state = self.core.best_model_state

        logger.info("OptimizedSynergisticFactorMiner initialized (using refactored modules)")

    def mine_factors(self,
                    n_iterations: int = 500,
                    batch_size: int = 8,
                    train_interval: int = 20,
                    print_interval: int = 25,
                    early_stop_patience: int = 50,
                    min_delta: float = 1e-4):
        """
        主挖掘循环

        Args:
            n_iterations: 总迭代次数
            batch_size: 每次生成的表达式数量
            train_interval: PPO训练间隔（每N个iteration训练一次）
            print_interval: 打印进度间隔
            early_stop_patience: 早停patience（验证集无改进的最大iteration数）
            min_delta: 最小改进阈值（小于此值视为无改进）

        Returns:
            因子池列表
        """
        return self.core.mine_factors(
            n_iterations=n_iterations,
            batch_size=batch_size,
            train_interval=train_interval,
            print_interval=print_interval,
            early_stop_patience=early_stop_patience,
            min_delta=min_delta
        )

    def get_best_factors(self, top_k: int = 5) -> List[Dict]:
        """
        获取最佳因子

        Args:
            top_k: 返回前k个因子

        Returns:
            因子列表
        """
        factors = []

        if self.config.combiner_type == 'linear':
            # Linear模式：按权重排序
            for i, alpha_info in enumerate(self.combination_model.alpha_pool):
                factor = {
                    'tokens': alpha_info['tokens'],
                    'weight': self.combination_model.current_weights[i]
                              if self.combination_model.current_weights is not None else 0,
                    'timestamp': alpha_info.get('timestamp', 0)
                }
                factors.append(factor)

            return sorted(factors, key=lambda x: abs(x['weight']), reverse=True)[:top_k]
        else:
            # LSTM模式：按时间排序（最新的）
            for alpha_info in self.combination_model.alpha_pool:
                factor = {
                    'tokens': alpha_info['tokens'],
                    'timestamp': alpha_info.get('timestamp', 0)
                }
                factors.append(factor)

            return sorted(factors, key=lambda x: x['timestamp'], reverse=True)[:top_k]

    def plot_training_history(self):
        """
        绘制训练历史曲线

        包括：
        - 奖励变化
        - 组合得分（训练集/验证集）
        - Sharpe比率
        - IC指标
        - 迭代级性能变化
        - 因子池大小和接受率
        """
        self.core.plot_training_history()

    def analyze_performance_degradation(self, train_interval: int = 20):
        """
        分析训练间隔内的性能衰退模式

        Args:
            train_interval: PPO训练间隔
        """
        self.core.analyze_performance_degradation(train_interval)

    def train_lstm_predictor(self,
                            epochs: int = 100,
                            batch_size: int = 64,
                            sequence_length: int = 20,
                            early_stop_patience: int = 15,
                            save_model: bool = True,
                            model_path: str = 'lstm_predictor.pt'):
        """
        训练LSTM预测器（在PPO挖掘完成后）

        Args:
            epochs: LSTM训练轮数
            batch_size: 批大小
            sequence_length: LSTM序列长度
            early_stop_patience: 早停patience
            save_model: 是否保存模型
            model_path: 模型保存路径

        Returns:
            训练好的LSTMFactorPredictor实例
        """
        from lstm_predictor import LSTMFactorPredictor

        logger.info("=" * 70)
        logger.info("🚀 Starting LSTM Predictor Training (Post-PPO)")
        logger.info("=" * 70)

        # 创建LSTM预测器
        lstm_predictor = LSTMFactorPredictor(config=self.config)

        # 准备因子矩阵
        logger.info("\n📊 Preparing factor matrices...")
        train_factors = lstm_predictor.prepare_factor_matrix(
            self.combination_model.alpha_pool,
            self.train_data,
            self.operators
        )
        val_factors = lstm_predictor.prepare_factor_matrix(
            self.combination_model.alpha_pool,
            self.val_data,
            self.operators
        )
        test_factors = lstm_predictor.prepare_factor_matrix(
            self.combination_model.alpha_pool,
            self.test_data,
            self.operators
        )

        # 训练LSTM
        train_result = lstm_predictor.train(
            train_factors=train_factors,
            train_targets=self.train_target,
            val_factors=val_factors,
            val_targets=self.val_target,
            epochs=epochs,
            batch_size=batch_size,
            sequence_length=sequence_length,
            early_stop_patience=early_stop_patience
        )

        logger.info(f"\n✅ LSTM Training completed!")
        logger.info(f"   Best Val IC: {train_result['best_val_ic']:.4f}")

        # 保存模型
        if save_model:
            lstm_predictor.save_model(model_path)

        return lstm_predictor

    # ==================== 内部辅助方法（暴露以保持兼容性）====================

    def _tokens_to_expression(self, tokens: List[str]) -> str:
        """将RPN格式的tokens转换为可读表达式"""
        return self.core.expr_generator.tokens_to_expression(tokens)

    def _calculate_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算目标变量"""
        return self.core._calculate_target(data)

    def _compute_feature_scales(self):
        """计算特征数量级"""
        return self.core._compute_feature_scales()

    def _build_operators(self):
        """构建操作符"""
        return self.core._build_operators()

    def _build_vocab(self):
        """构建词汇表"""
        return self.core._build_vocab()

    def _init_networks(self):
        """初始化网络"""
        return self.core._init_networks()

    def generate_expression_batch(self, batch_size: int = 8):
        """生成表达式batch"""
        return self.core.expr_generator.generate_expression_batch(batch_size)

    def _evaluate_expression(self, tokens: List[str]) -> Dict:
        """评估表达式"""
        return self.core.factor_evaluator.evaluate_expression(tokens)

    def _compute_factor_values(self, expr_tokens: List[str], data: pd.DataFrame):
        """计算因子值"""
        return self.core.factor_evaluator.compute_factor_values(expr_tokens, data)

    def train_ppo_step(self) -> Dict[str, float]:
        """执行一次PPO训练步骤"""
        return self.core.ppo_trainer.train_ppo_step(
            self.core.expr_generator._get_valid_actions
        )


# ==================== 模块说明 ====================
__doc__ = """
因子挖掘器重构说明
==================

重构后的模块结构：

1. miner_core.py (FactorMinerCore)
   - 核心挖掘逻辑
   - 整合所有组件
   - 管理训练循环

2. expression_generator.py (ExpressionGenerator)
   - 基于PPO策略生成因子表达式
   - 层次化动作选择
   - 数量级兼容性检查

3. factor_evaluator.py (FactorEvaluator)
   - 计算因子值
   - 评估表达式有效性
   - 与组合模型交互

4. ppo_trainer.py (PPOTrainer)
   - PPO算法实现
   - GAE优势函数计算
   - 策略和价值网络更新

5. visualization.py (VisualizationTools)
   - 训练历史可视化
   - 性能分析工具
   - 结果展示

使用方式：
---------
# 方式1：直接使用（推荐，接口不变）
from miner import OptimizedSynergisticFactorMiner
miner = OptimizedSynergisticFactorMiner(data, config=config)
factors = miner.mine_factors(n_iterations=500

# 方式2：使用核心类（更灵活）
from miner_core import FactorMinerCore
core = FactorMinerCore(data, config=config)
factors = core.mine_factors(n_iterations=500)

优势：
-----
1. 模块化：每个组件职责清晰，易于维护和测试
2. 可扩展：可以轻松替换或增强单个模块
3. 可复用：各模块可以独立使用
4. 向后兼容：现有代码无需修改
5. 代码清晰：每个文件专注于单一功能

参考：
-----
- evaluator.py: 单一职责的评估器模块
- combiner.py: 清晰的组合模型接口
"""


if __name__ == '__main__':
    # 使用示例
    print(__doc__)
