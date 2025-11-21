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

            # 🔥 阶段1: 纯试算（Trial）- 在统一的基准环境下评估所有因子
            # 避免因子顺序依赖导致的奖励不一致
            raw_rewards = []
            eval_results = []
            valid_candidates = []  # 记录合格候选因子

            for idx, (tokens, state_ids, trajectory) in enumerate(batch_results):
                # trial_only=True: 只计算奖励，不添加到池子
                eval_result = self.factor_evaluator.evaluate_expression(tokens, trial_only=True)

                if eval_result['valid']:
                    final_reward = eval_result['reward']
                    # 记录合格候选因子（达到阈值）
                    if eval_result.get('qualifies', False):
                        valid_candidates.append({
                            'idx': idx,
                            'tokens': tokens,
                            'reward': final_reward,
                            'eval_result': eval_result
                        })
                else:
                    # 🔥 无效表达式给予小的负奖励，而非-1.0
                    # 这样PPO能学习到"避免无效表达式"但不会被过大的惩罚干扰
                    final_reward = -0.1
                    # 调试：记录失败原因
                    if iteration < 3:  # 只在前几次迭代打印
                        logger.debug(f"Expression invalid: {tokens}, reason: {eval_result.get('reason', 'unknown')}")

                raw_rewards.append(final_reward)
                eval_results.append(eval_result)

            # 🔥 阶段2: 选择并提交（Commit）- 只提交本batch中最好的因子
            # 这样避免了同一batch内的因子相互影响奖励
            if valid_candidates:
                # 按奖励排序，选择top-1
                valid_candidates.sort(key=lambda x: x['reward'], reverse=True)
                best_candidate = valid_candidates[0]
                best_eval = best_candidate['eval_result']

                # 🔥 检查是否真的qualifies（达到阈值）
                if best_eval.get('qualifies', False):
                    # 真正提交最佳候选
                    commit_result = self.combination_model.add_alpha_and_optimize(
                        best_eval['alpha_info'],
                        best_eval['train_factor'],
                        best_eval['val_factor']
                    )
                    logger.debug(f"✅ Batch best factor committed (reward={best_candidate['reward']:.4f}), pool_size={commit_result.get('pool_size', 0)}")
                else:
                    logger.debug(f"❌ Batch best factor not qualified (reward={best_candidate['reward']:.4f}), skipping commit")

            # 🔥 移除归一化！直接使用原始增量Sharpe作为奖励
            # 原因：增量Sharpe是稀疏但真实的信号，归一化会破坏其意义
            # 调整clip范围：允许更大的正奖励以鼓励探索高质量因子
            # 同时保持适度的负奖励惩罚以避免过度惩罚
            clipped_rewards = [np.clip(r, -1.0, 10.0) for r in raw_rewards]

            # 添加到buffer
            for i in range(batch_size):
                tokens, state_ids, trajectory = batch_results[i]
                final_reward = clipped_rewards[i]  # 🔥 使用clipped而非normalized
                expression_length = len(trajectory['states'])

                # 步骤奖励分配
                step_rewards = self._compute_step_rewards(
                    final_reward, expression_length
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
                logger.info(f"🔄 PPO Update #{self.ppo_update_count} at iteration {iteration}")
                logger.info(f"  Buffer size: {len(self.ppo_buffer)}")

                train_stats = self.ppo_trainer.train_ppo_step(
                    self.expr_generator._get_valid_actions
                )

                if train_stats:
                    # 🔥 打印PPO训练详细信息
                    logger.info(f"  PPO Training Stats:")
                    logger.info(f"    Policy Loss: {train_stats.get('policy_loss', 0.0):.6f}")
                    logger.info(f"    Value Loss: {train_stats.get('value_loss', 0.0):.6f}")
                    logger.info(f"    Entropy Loss: {train_stats.get('entropy_loss', 0.0):.6f}")
                    logger.info(f"    Advantage - Mean: {train_stats.get('advantage_mean', 0.0):.4f}, Std: {train_stats.get('advantage_std', 0.0):.4f}")
                    logger.info(f"    Value - Mean: {train_stats.get('value_mean', 0.0):.4f}, Std: {train_stats.get('value_std', 0.0):.4f}")
                    logger.info(f"    Learning Rate: {train_stats.get('learning_rate', 0.0):.6f}")

                    # 记录到训练历史
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

                # 🔥 打印最佳因子组合信息
                if len(self.combination_model.alpha_pool) > 0:
                    logger.info(f"  Current Best Factor Combination:")
                    logger.info(f"    Train Score: {iter_end_train_eval['composite_score']:.4f}")
                    logger.info(f"    Val Score: {iter_end_val_eval['composite_score']:.4f}")

                    # 显示权重最大的前3个因子
                    if self.combination_model.current_weights is not None and len(self.combination_model.current_weights) > 0:
                        weights = self.combination_model.current_weights
                        abs_weights = np.abs(weights)
                        top_indices = np.argsort(abs_weights)[-3:][::-1]

                        logger.info(f"    Top 3 Factors by Weight:")
                        for rank, idx in enumerate(top_indices, 1):
                            if idx < len(self.combination_model.alpha_pool):
                                factor_info = self.combination_model.alpha_pool[idx]
                                tokens = factor_info.get('tokens', [])
                                weight = weights[idx]
                                contribution = self.combination_model.factor_contributions[idx] if idx < len(self.combination_model.factor_contributions) else 0.0
                                logger.info(f"      #{rank}: weight={weight:.4f}, incremental_contribution={contribution:.4f}")
                                logger.info(f"          expression: {' '.join(tokens)}")
                    else:
                        # 🔥 只有1个因子或没有权重时,直接显示因子信息
                        logger.info(f"    Factors in Pool:")
                        for idx, factor_info in enumerate(self.combination_model.alpha_pool[:3]):
                            tokens = factor_info.get('tokens', [])
                            contribution = self.combination_model.factor_contributions[idx] if idx < len(self.combination_model.factor_contributions) else 0.0
                            logger.info(f"      #{idx+1}: incremental_contribution={contribution:.4f}")
                            logger.info(f"          expression: {' '.join(tokens)}")

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

    # 🔥 已废弃：奖励归一化会破坏增量Sharpe的真实信号
    # 稀疏奖励（大部分接近0，少数>阈值）才能让PPO学习到"哪些因子真正有价值"
    # 如果归一化，会让PPO误以为"批次内相对好"="真正好"
    #
    # def _normalize_rewards(self, rewards: List[float]) -> List[float]:
    #     """归一化奖励 - DEPRECATED"""
    #     pass

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
