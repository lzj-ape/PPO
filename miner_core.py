"""
核心挖掘器模块 - 主要的因子挖掘逻辑
整合表达式生成、因子评估、PPO训练等组件
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import logging
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

from config import TrainingConfig
from operators import TimeSeriesOperators
from evaluator import ICDiversityEvaluator
from networks import ActorCriticNetwork
from buffer import PPOBuffer
from combiner import ImprovedCombinationModel
from expression_generator import ExpressionGenerator
from factor_evaluator import FactorEvaluator
from ppo_trainer import PPOTrainer
from visualization import VisualizationTools

logger = logging.getLogger(__name__)


class FactorMinerCore:
    """核心因子挖掘器 - 整合所有组件的主类"""

    def __init__(self,
                 data: pd.DataFrame,
                 target_col: str = 'future_return',
                 config: TrainingConfig = None,
                 max_factors: int = 15,
                 max_expr_len: int = 20):
        """
        初始化因子挖掘器

        Args:
            data: 完整数据集
            target_col: 目标列名
            config: 训练配置
            max_factors: 最大因子数量
            max_expr_len: 最大表达式长度
        """
        self.config = config or TrainingConfig()
        self.target_col = target_col
        self.max_factors = max_factors
        self.max_expr_len = max_expr_len

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 计算目标变量
        if target_col not in data.columns:
            logger.info(f"Calculating target variable: {target_col}")
            data = self._calculate_target(data)

        # 数据分割（带Purging Gap）
        self._split_data(data)

        # 特征和目标
        self.feature_names = [col for col in data.columns if col != target_col]
        self.train_target = self.train_data[target_col]
        self.val_target = self.val_data[target_col]
        self.test_target = self.test_data[target_col]

        # 初始化操作符和词汇表
        self.ts_ops = TimeSeriesOperators()
        self._compute_feature_scales()
        self._build_operators()
        self._build_vocab()

        # 初始化网络
        self._init_networks()

        # 初始化评估器和组合模型
        self.evaluator = ICDiversityEvaluator(self.config)
        self.combination_model = ImprovedCombinationModel(
            config=self.config,
            max_alpha_count=max_factors
        )
        self.combination_model.set_evaluator(self.evaluator)
        self.combination_model.set_targets(self.train_target, self.val_target)
        self.evaluator.set_combiner(self.combination_model)

        # 初始化表达式生成器
        self.expr_generator = ExpressionGenerator(
            actor_critic=self.actor_critic,
            vocab=self.vocab,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            operators=self.operators,
            feature_names=self.feature_names,
            feature_scales=self.feature_scales,
            max_expr_len=max_expr_len,
            device=self.device,
            use_amp=self.use_amp
        )

        # 初始化因子评估器
        self.factor_evaluator = FactorEvaluator(
            operators=self.operators,
            feature_names=self.feature_names,
            combination_model=self.combination_model,
            train_data=self.train_data,
            val_data=self.val_data,
            train_target=self.train_target,
            val_target=self.val_target
        )

        # PPO缓冲区和训练器
        self.ppo_buffer = PPOBuffer(max_size=self.config.buffer_size)
        self.ppo_trainer = PPOTrainer(
            actor_critic=self.actor_critic,
            ppo_buffer=self.ppo_buffer,
            config=self.config,
            vocab=self.vocab,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            operators=self.operators,
            feature_names=self.feature_names,
            optimizer=self.optimizer,
            device=self.device,
            use_amp=self.use_amp
        )

        # 训练历史
        self.training_history = {
            'rewards': [],
            'train_metric1': [],
            'train_metric2': [],
            'train_kl': [],
            'train_composite': [],
            'val_metric1': [],
            'val_metric2': [],
            'val_kl': [],
            'val_composite': [],
            'policy_losses': [],
            'value_losses': [],
            'iteration_scores': [],
            'ppo_update_iterations': [],
            'pool_size_history': [],
            'factor_additions': [],
            'factor_rejections': [],
            'best_solutions': [],
        }

        self.best_val_score = -999.0
        self.best_model_state = None
        self.best_solution_snapshots: List[Dict] = []
        self.best_solution_path = Path(
            self.config.__dict__.get('best_solution_path', 'best_solutions.json')
        )

        # PPO更新追踪
        self.ppo_update_count = 0
        self.last_ppo_update_iter = -1

        # 奖励归一化
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []
        self.reward_momentum = 0.9

        logger.info(f"FactorMinerCore initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Combiner type: {self.config.combiner_type}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Vocab size: {len(self.vocab)}")

    def _calculate_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算目标变量"""
        horizon = self.config.prediction_horizon
        data['future_return'] = data['close'].pct_change(horizon).shift(-horizon).fillna(0)
        logger.info(f"Target calculated: {horizon}-period forward percentage returns")
        logger.info(f"  Mean: {data['future_return'].mean():.6f}")
        logger.info(f"  Std: {data['future_return'].std():.6f}")
        return data

    def _split_data(self, data: pd.DataFrame):
        """数据分割（带Purging Gap）"""
        train_size = int(len(data) * self.config.train_ratio)
        val_size = int(len(data) * self.config.val_ratio)
        gap_size = self.config.prediction_horizon
        self.train_data = data.iloc[:train_size].copy()
        self.val_data = data.iloc[train_size+gap_size:train_size+gap_size+val_size].copy()
        self.test_data = data.iloc[train_size+gap_size+val_size+gap_size:].copy()

        self.train_data = self.train_data.ffill().bfill().fillna(0)
        self.val_data = self.val_data.ffill().bfill().fillna(0)
        self.test_data = self.test_data.ffill().bfill().fillna(0)

        logger.info(f"Data split with Purging Gap={gap_size}:")
        logger.info(f"  Train: {len(self.train_data)} bars")
        logger.info(f"  Val: {len(self.val_data)} bars")
        logger.info(f"  Test: {len(self.test_data)} bars")

    def _compute_feature_scales(self):
        """计算特征数量级"""
        self.feature_scales = {}

        for feature in self.feature_names:
            if feature in self.train_data.columns:
                feature_data = self.train_data[feature]
                if isinstance(feature_data, pd.DataFrame):
                    feature_data = feature_data.iloc[:, 0]

                values = feature_data.replace([np.inf, -np.inf], np.nan).dropna()

                if len(values) == 0:
                    self.feature_scales[feature] = 1.0
                    continue

                # 检测bool/二值变量
                unique_values = values.unique()
                if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, 0.0, 1.0, True, False}):
                    self.feature_scales[feature] = 1.0
                    continue

                # 检测归一化数据
                val_min, val_max = values.min(), values.max()
                if -0.01 <= val_min and val_max <= 1.01:
                    self.feature_scales[feature] = 1.0
                    continue
                elif -1.01 <= val_min and val_max <= 1.01:
                    self.feature_scales[feature] = 1.0
                    continue

                # 使用中位数绝对值
                median_abs = np.abs(values.median())
                if median_abs < 1e-10:
                    mean_abs = np.abs(values.mean())
                    if mean_abs < 1e-10:
                        self.feature_scales[feature] = max(values.std(), 1e-10)
                    else:
                        self.feature_scales[feature] = mean_abs
                else:
                    self.feature_scales[feature] = median_abs
            else:
                self.feature_scales[feature] = 1.0

    def _build_operators(self):
        """构建操作符及其数量级规则 - 扩展到50个算子"""
        self.operators = {
            # ============ 基础算术 (5个) ============
            'add': {
                'arity': 2, 
                'func': self.ts_ops.add,
                'scale_rule': 'similar_only',
                'scale_threshold': 100.0
            },
            'sub': {
                'arity': 2, 
                'func': self.ts_ops.sub,
                'scale_rule': 'similar_only',
                'scale_threshold': 100.0
            },
            'mul': {
                'arity': 2, 
                'func': self.ts_ops.mul,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'div': {
                'arity': 2, 
                'func': self.ts_ops.div,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'pow': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.pow_op(x, 2.0),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            
            # ============ 基础变换 (7个) ============
            'abs': {
                'arity': 1, 
                'func': self.ts_ops.abs_op,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'sign': {
                'arity': 1, 
                'func': self.ts_ops.sign_op,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'log': {
                'arity': 1, 
                'func': self.ts_ops.log_op,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'exp': {
                'arity': 1, 
                'func': self.ts_ops.exp_op,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'sqrt': {
                'arity': 1, 
                'func': self.ts_ops.sqrt_op,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'sigmoid': {
                'arity': 1, 
                'func': self.ts_ops.sigmoid_op,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'tanh': {
                'arity': 1, 
                'func': self.ts_ops.tanh_op,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            
            # ============ 时间序列基础 (8个) ============
            'delay1': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.delay(x, 1),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'delay3': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.delay(x, 3),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'delta1': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.delta(x, 1),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'momentum5': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.momentum(x, 5),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'roc10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.rate_of_change(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'ts_rank10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.ts_rank(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'ts_min10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.ts_min(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'ts_max10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.ts_max(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            
            # ============ 移动平均 (8个) ============
            'sma5': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.sma(x, 5),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'sma10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.sma(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'sma20': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.sma(x, 20),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'ema5': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.ema(x, 5),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'ema10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.ema(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'wma10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.wma(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'dema10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.dema(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'tema10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.tema(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            
            # ============ 统计指标 (7个) ============
            'std10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.std(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'std20': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.std(x, 20),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'variance20': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.variance(x, 20),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'zscore20': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.zscore(x, 20),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'quantile20': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.quantile(x, 20, 0.5),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'mad20': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.mad(x, 20),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'covar': {
                'arity': 2, 
                'func': lambda x, y: self.ts_ops.covariance(x, y, 20),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            
            # ============ 技术指标 (5个 - 常用且不需要额外数据) ============
            'rsi14': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.rsi(x, 14),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'macd': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.macd(x, 12, 26),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'bb_upper': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.bbands_upper(x, 20, 2.0),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'bb_lower': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.bbands_lower(x, 20, 2.0),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            
            # ============ 比较与逻辑 (5个) ============
            'max': {
                'arity': 2, 
                'func': self.ts_ops.max_op,
                'scale_rule': 'similar_only',
                'scale_threshold': 100.0
            },
            'min': {
                'arity': 2, 
                'func': self.ts_ops.min_op,
                'scale_rule': 'similar_only',
                'scale_threshold': 100.0
            },
            'condition': {
                'arity': 3, 
                'func': self.ts_ops.condition,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'rank': {
                'arity': 1, 
                'func': self.ts_ops.rank,
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'scale': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.scale(x, 1.0),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            
            # ============ 高级算子 (3个) ============
            'corr20': {
                'arity': 2, 
                'func': lambda x, y: self.ts_ops.correlation(x, y, 20),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'decay10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.decay_linear(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
            'ts_prod10': {
                'arity': 1, 
                'func': lambda x: self.ts_ops.ts_prod(x, 10),
                'scale_rule': 'any',
                'scale_threshold': None
            },
        }

    def _build_vocab(self):
        """构建词汇表"""
        self.vocab = ['<PAD>', '<BEG>', '<SEP>'] + list(self.operators.keys()) + self.feature_names
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id['<PAD>']

    def _init_networks(self):
        """初始化网络"""
        vocab_size = len(self.vocab)
        self.actor_critic = ActorCriticNetwork(
            vocab_size, self.config, self.pad_token_id
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.config.lr_actor,
            weight_decay=1e-4
        )

        self.use_amp = torch.cuda.is_available()

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
            n_iterations: 迭代次数
            batch_size: 批大小
            train_interval: PPO训练间隔
            print_interval: 打印间隔
            early_stop_patience: 早停patience
            min_delta: 最小改进阈值
        """
        logger.info(f"Starting factor mining:")
        logger.info(f"  Iterations: {n_iterations}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Early stop patience: {early_stop_patience}")

        best_val_score = -999.0
        no_improve_count = 0
        recent_rewards = deque(maxlen=100)
        start_time = time.time()

        last_iter_end_train_eval = None
        last_iter_end_val_eval = None

        for iteration in range(n_iterations):
            # 获取初始评估
            if iteration == 0:
                iter_start_train_eval = self.combination_model.evaluate_combination(use_val=False)
                iter_start_val_eval = self.combination_model.evaluate_combination(use_val=True)
            else:
                iter_start_train_eval = last_iter_end_train_eval
                iter_start_val_eval = last_iter_end_val_eval

            # 生成表达式batch
            batch_results = self.expr_generator.generate_expression_batch(batch_size)

            # 评估表达式
            raw_rewards = []
            eval_results = []

            for tokens, state_ids, trajectory in batch_results:
                eval_result = self.factor_evaluator.evaluate_expression(tokens)

                if eval_result['valid']:
                    final_reward = eval_result['reward']
                else:
                    final_reward = -1.0

                raw_rewards.append(final_reward)
                eval_results.append(eval_result)

            # 归一化奖励
            normalized_rewards = self._normalize_rewards(raw_rewards)

            # 添加到buffer
            for i in range(batch_size):
                tokens, state_ids, trajectory = batch_results[i]
                final_reward_normalized = normalized_rewards[i]
                expression_length = len(trajectory['states'])

                # 步骤奖励分配
                step_rewards = self._compute_step_rewards(
                    final_reward_normalized, expression_length
                )

                for j in range(len(trajectory['states'])):
                    combined_log_prob = (trajectory['type_log_probs'][j] +
                                        trajectory['action_log_probs'][j])

                    self.ppo_buffer.add(
                        state=trajectory['states'][j],
                        action=trajectory['actions'][j],
                        reward=step_rewards[j],
                        log_prob=combined_log_prob,
                        value=trajectory['values'][j],
                        done=(j == len(trajectory['states']) - 1)
                    )

            # 收集统计信息
            recent_rewards.extend(raw_rewards)
            self.training_history['rewards'].extend(raw_rewards)

            # 更新训练历史
            for eval_result in eval_results:
                if eval_result['valid']:
                    train_eval = eval_result['train_eval']
                    val_eval = eval_result['val_eval']

                    self.training_history['train_metric1'].append(train_eval.get('sharpe', 0))
                    self.training_history['train_metric2'].append(train_eval.get('ic', 0))
                    self.training_history['val_metric1'].append(val_eval.get('sharpe', 0))
                    self.training_history['val_metric2'].append(val_eval.get('ic', 0))
                    self.training_history['train_composite'].append(train_eval['composite_score'])
                    self.training_history['val_composite'].append(val_eval['composite_score'])

            # 结束评估
            iter_end_train_eval = self.combination_model.evaluate_combination(use_val=False)
            iter_end_val_eval = self.combination_model.evaluate_combination(use_val=True)

            last_iter_end_train_eval = iter_end_train_eval
            last_iter_end_val_eval = iter_end_val_eval

            # PPO训练
            min_buffer_size = self.config.batch_size * 4
            if (self.ppo_buffer.is_full() or
                (iteration % train_interval == 0 and len(self.ppo_buffer) >= min_buffer_size)):

                self.ppo_update_count += 1
                train_stats = self.ppo_trainer.train_ppo_step(
                    self.expr_generator._get_valid_actions
                )

                if train_stats:
                    self.training_history['ppo_update_iterations'].append(iteration)
                    for key, value in train_stats.items():
                        if f'{key}s' not in self.training_history:
                            self.training_history[f'{key}s'] = []
                        self.training_history[f'{key}s'].append(value)

            # 早停检查
            current_val_score = iter_end_val_eval['composite_score']
            if current_val_score > best_val_score + min_delta:
                best_val_score = current_val_score
                self.best_val_score = best_val_score
                self.best_model_state = self.actor_critic.state_dict()
                no_improve_count = 0
                logger.info(f"✨ New best VAL score: {best_val_score:.4f} at iteration {iteration}")
            else:
                no_improve_count += 1

            if no_improve_count >= early_stop_patience:
                logger.info(f"🛑 Early stopping at iteration {iteration}")
                break

            # 定期打印
            if (iteration + 1) % print_interval == 0:
                avg_reward = np.mean(list(recent_rewards)) if recent_rewards else 0
                logger.info(f"Iteration {iteration + 1}/{n_iterations}")
                logger.info(f"  Avg Reward: {avg_reward:.4f}")
                logger.info(f"  Best VAL: {best_val_score:.4f}")
                logger.info(f"  Pool Size: {len(self.combination_model.alpha_pool)}")

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.actor_critic.load_state_dict(self.best_model_state)
            logger.info("✅ Restored best model")

        logger.info("🎉 MINING COMPLETED!")
    # 🔥 自定义返回格式
        return {
            'factors': self.combination_model.alpha_pool,
            'best_val_score': self.best_val_score,
            'training_history': self.training_history,
            'model_state': self.best_model_state,
            'statistics': {
                'total_iterations': n_iterations,
                'ppo_updates': self.ppo_update_count,
                'final_pool_size': len(self.combination_model.alpha_pool)
            },
            'evaluator': self.evaluator,
            'combination_model': self.combination_model
        }

    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """归一化奖励"""
        if not rewards:
            return []

        batch_mean = np.mean(rewards)
        batch_std = np.std(rewards) + 1e-8

        if len(self.reward_history) == 0:
            self.reward_mean = batch_mean
            self.reward_std = batch_std
        else:
            self.reward_mean = (self.reward_momentum * self.reward_mean +
                              (1 - self.reward_momentum) * batch_mean)
            self.reward_std = (self.reward_momentum * self.reward_std +
                             (1 - self.reward_momentum) * batch_std)

        self.reward_history.extend(rewards)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        normalized = [(r - self.reward_mean) / (self.reward_std + 1e-8) for r in rewards]
        normalized = [np.clip(r, -5.0, 5.0) for r in normalized]

        return normalized

    def _compute_step_rewards(self, final_reward: float, length: int) -> List[float]:
        """计算步骤奖励"""
        min_ratio = 0.5
        step_rewards = []
        total_weight = 0

        for j in range(length):
            progress = (j + 1) / length
            weight = min_ratio + (1 - min_ratio) * progress
            step_rewards.append(weight)
            total_weight += weight

        step_rewards = [r / total_weight * final_reward for r in step_rewards]
        return step_rewards

    def plot_training_history(self):
        """绘制训练历史"""
        vis_tools = VisualizationTools(self.training_history, self.config)
        vis_tools.plot_training_history()

    def analyze_performance_degradation(self, train_interval: int = 20):
        """分析性能衰退"""
        vis_tools = VisualizationTools(self.training_history, self.config)
        vis_tools.analyze_performance_degradation(train_interval)
