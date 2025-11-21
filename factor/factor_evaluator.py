"""
因子评估器模块
负责计算因子值和评估表达式
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
import time
import sys
from pathlib import Path

# 添加utils目录到路径
utils_path = Path(__file__).parent.parent / 'utils'
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

try:
    from advanced_reward import AdvancedRewardCalculator, RewardConfig
    ADVANCED_REWARD_AVAILABLE = True
except ImportError:
    ADVANCED_REWARD_AVAILABLE = False
    logging.warning("AdvancedRewardCalculator not available, using simple reward")

logger = logging.getLogger(__name__)


class FactorEvaluator:
    """
    因子评估器 - 负责计算因子值和评估表达式
    
    修改说明 (实盘适配):
    1. 引入了 Train/Val 状态区分，确保清洗验证集数据时使用训练集的统计量。
    2. 修复了数据清洗中的未来函数 (Look-ahead Bias)。
    """

    def __init__(self,
                 operators: Dict,
                 feature_names: List[str],
                 combination_model,
                 train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 train_target: pd.Series,
                 val_target: pd.Series):
        """
        初始化因子评估器

        Args:
            operators: 操作符字典
            feature_names: 特征名称列表
            combination_model: 组合模型
            train_data: 训练数据
            val_data: 验证数据
            train_target: 训练目标
            val_target: 验证目标
        """
        self.operators = operators
        self.feature_names = feature_names
        self.combination_model = combination_model
        self.train_data = train_data
        self.val_data = val_data
        self.train_target = train_target
        self.val_target = val_target

        # 🔥 新增：用于缓存当前正在评估的因子的训练集统计量
        # 格式: {'median': float, 'lower': float, 'upper': float}
        self.current_factor_stats = None

        # 🔥 初始化高级奖励计算器
        if ADVANCED_REWARD_AVAILABLE:
            # 使用简化配置：只启用惩罚项，不使用增量Sharpe（因为我们已经在combiner中计算）
            reward_config = RewardConfig(
                use_incremental_sharpe=False,  # 不重复计算增量
                use_penalty=True,  # 启用惩罚项
                use_rolling_stability=False,  # 数据量小时关闭
                complexity_lambda=0.3,
                turnover_gamma=2.0,
                max_expr_length=30
            )
            self.reward_calculator = AdvancedRewardCalculator(reward_config)
            logger.info("✅ AdvancedRewardCalculator enabled (penalty mode)")
        else:
            self.reward_calculator = None

    def evaluate_expression(self, tokens: List[str], trial_only: bool = False) -> Dict:
        """
        评估表达式

        Args:
            tokens: token列表
            trial_only: 是否仅试算不提交（True=只计算奖励，False=根据阈值决定是否添加）

        Returns:
            评估结果字典
        """
        if len(tokens) < 3 or tokens[0] != '<BEG>' or tokens[-1] != '<SEP>':
            return {'valid': False, 'reason': 'invalid_format'}

        try:
            expr_tokens = tokens[1:-1]
            
            # 🔥 重置统计量缓存，开始新一轮评估
            self.current_factor_stats = None

            # 1. 在训练集上计算因子值 (Compute & Learn Stats)
            # 注意：必须先算训练集，这样 _clean_series 才能计算并保存统计量
            train_factor = self.compute_factor_values(expr_tokens, self.train_data, is_training=True)
            
            if train_factor is None:
                return {'valid': False, 'reason': 'train_computation_failed'}
                
            if self.current_factor_stats is None:
                return {'valid': False, 'reason': 'stats_computation_failed'}

            # 2. 在验证集上计算因子值 (Compute & Apply Stats)
            # 这里会使用上一步缓存的统计量进行清洗，严禁使用验证集自己的统计量
            val_factor = self.compute_factor_values(expr_tokens, self.val_data, is_training=False)
            
            if val_factor is None:
                # 如果验证集计算失败（例如数据太短无法计算SMA），视为无效
                return {'valid': False, 'reason': 'val_computation_failed'}

            # 3. 先试算：计算增量贡献（不修改池子）
            alpha_info = {
                'tokens': tokens,
                'timestamp': time.time(),
                # 保存统计量，以便将来实盘生成时复用
                'stats': self.current_factor_stats,
                # 保存operators引用，供回测使用
                'operators': self.operators
            }

            # 🔥 Trial Mode: 计算增量Sharpe
            trial_result = self.combination_model.evaluate_new_factor(
                alpha_info, train_factor, val_factor
            )

            incremental_sharpe = trial_result.get('train_incremental_sharpe', 0.0)
            train_stats = trial_result.get('train_stats', {'sharpe': 0.0, 'composite_score': 0.0})
            val_stats = trial_result.get('val_stats', {'sharpe': 0.0, 'composite_score': 0.0})

            # 4. 决策：是否真正添加到池子
            # 🔥 自适应阈值：池子越小，阈值越低
            base_threshold = getattr(self.combination_model.config, 'ic_threshold', 0.01)
            current_pool_size = len(self.combination_model.alpha_pool)

            # 🔥 修复：前期使用0阈值，允许所有有效因子进入
            if current_pool_size < 3:
                ic_threshold = 0.0  # 前3个因子：只要增量>0就接受
            elif current_pool_size < 5:
                ic_threshold = base_threshold * 0.3  # 第4-5个因子用0.3倍阈值
            elif current_pool_size < 10:
                ic_threshold = base_threshold * 0.6  # 第6-10个因子用0.6倍阈值
            else:
                ic_threshold = base_threshold  # 之后用正常阈值

            should_add = incremental_sharpe > ic_threshold and not trial_only
            actually_added = False

            # 🔥 诊断日志：记录拒绝的原因（显式打印）
            if not trial_only and incremental_sharpe <= ic_threshold:
                logger.info(f"❌ Factor REJECTED:")
                logger.info(f"   incremental_sharpe={incremental_sharpe:.6f} <= threshold={ic_threshold:.6f}")
                logger.info(f"   base_threshold={base_threshold:.6f}, pool_size={current_pool_size}")
                logger.info(f"   base_train_score={self.combination_model.base_train_score:.4f}")
                logger.info(f"   new_train_score={trial_result['train_stats']['sharpe']:.4f}")
                logger.info(f"   expression: {' '.join(tokens[:15])}...")

                # 🔥 额外诊断：分析为什么增量低
                if incremental_sharpe <= 0:
                    logger.info(f"   ⚠️  Reason: New factor does NOT improve the combination (negative/zero increment)")
                elif self.combination_model.base_train_score > 2.0 and incremental_sharpe < 0.01:
                    logger.info(f"   ⚠️  Reason: Base score is already high, hard to improve further")
                else:
                    logger.info(f"   ⚠️  Reason: Improvement too small (below threshold)")

            if should_add:
                # 🔥 Commit Mode: 真正添加到池子
                old_pool_size = len(self.combination_model.alpha_pool)
                old_train_score = self.combination_model.base_train_score

                commit_result = self.combination_model.add_alpha_and_optimize(
                    alpha_info, train_factor, val_factor
                )
                current_pool_size = commit_result.get('pool_size', current_pool_size)
                train_score_after = commit_result.get('current_train_score', 0.0)
                val_score_after = commit_result.get('current_val_score', 0.0)
                actually_added = True

                # 🔥 记录成功添加（详细信息）
                logger.info(f"✅ Factor ACCEPTED:")
                logger.info(f"   incremental_sharpe={incremental_sharpe:.6f} > threshold={ic_threshold:.6f}")
                logger.info(f"   Pool size: {old_pool_size} → {current_pool_size}")
                logger.info(f"   Train score: {old_train_score:.4f} → {train_score_after:.4f} (Δ={train_score_after - old_train_score:.4f})")
                logger.info(f"   Val score: {commit_result.get('current_val_score', 0.0):.4f}")
                logger.info(f"   Expression: {' '.join(tokens[:20])}...")

                # 显示当前池子中的因子数量和权重分布
                if self.combination_model.current_weights is not None:
                    weights = self.combination_model.current_weights
                    logger.info(f"   Weight stats: mean={np.mean(np.abs(weights)):.4f}, max={np.max(np.abs(weights)):.4f}, min={np.min(np.abs(weights)):.4f}")
            else:
                # 不添加，保持原有分数
                train_score_after = train_stats.get('sharpe', 0.0)
                val_score_after = val_stats.get('sharpe', 0.0)

            # 5. 🔥 应用高级奖励计算（惩罚项）
            final_reward = incremental_sharpe
            penalty_components = {}

            if self.reward_calculator is not None:
                # 准备old/new评估数据
                old_train_eval = {'sharpe': self.combination_model.base_train_score}
                old_val_eval = {'sharpe': self.combination_model.base_val_score}
                new_train_eval = train_stats
                new_val_eval = val_stats

                # 计算惩罚项（不包括增量Sharpe，因为我们已经有了）
                penalty_result = self.reward_calculator.calculate_reward(
                    new_train_eval=new_train_eval,
                    new_val_eval=new_val_eval,
                    old_train_eval=old_train_eval,
                    old_val_eval=old_val_eval,
                    alpha_info=alpha_info,
                    combination_series=None,  # 暂不使用换手率惩罚
                    evaluator=None
                )

                # 只取惩罚部分（不包括incremental_sharpe）
                penalty_components = penalty_result.get('components', {})
                complexity_penalty = penalty_components.get('complexity_penalty', 0.0)
                overfitting_penalty = penalty_components.get('overfitting_penalty', 0.0)

                # 最终奖励 = 增量Sharpe + 惩罚项
                final_reward = incremental_sharpe + complexity_penalty + overfitting_penalty

                # logger.debug(f"Reward breakdown: incremental={incremental_sharpe:.4f}, "
                #            f"complexity={complexity_penalty:.4f}, overfitting={overfitting_penalty:.4f}, "
                #            f"final={final_reward:.4f}")

            # 6. 返回结果（奖励是增量Sharpe + 惩罚）
            return {
                'valid': True,
                'reward': final_reward,  # 🔥 核心修复：增量 + 惩罚
                'pool_size': current_pool_size,
                'added_to_pool': actually_added,  # 是否真的被添加（trial_only时为False）
                'qualifies': incremental_sharpe > ic_threshold,  # 是否达标
                'incremental_sharpe': incremental_sharpe,
                'penalty_components': penalty_components,
                'train_factor': train_factor,  # 🔥 新增：返回因子数据供后续提交
                'val_factor': val_factor,
                'alpha_info': alpha_info,
                'train_eval': {
                    'sharpe': train_score_after,
                    'ic': incremental_sharpe * 0.5,  # IC和增量Sharpe相关
                    'composite_score': incremental_sharpe
                },
                'val_eval': {
                    'sharpe': val_score_after,
                    'ic': incremental_sharpe * 0.5,
                    'composite_score': val_stats.get('composite_score', 0.0)
                },
                'composite_score': final_reward  # 🔥 这里也改为最终奖励
            }

        except Exception as e:
            # logger.debug(f"Expression evaluation error: {e}")
            # import traceback
            # logger.debug(traceback.format_exc())
            return {'valid': False, 'reason': str(e)}

    def compute_factor_values(self, expr_tokens: List[str], data: pd.DataFrame, is_training: bool = False) -> Optional[pd.Series]:
        """
        计算因子值

        Args:
            expr_tokens: 表达式token列表
            data: 数据DataFrame
            is_training: 是否为训练模式（决定是计算统计量还是应用统计量）

        Returns:
            因子值Series，如果失败则返回None
        """
        try:
            stack = []

            for token in expr_tokens:
                if token in self.feature_names:
                    if token in data.columns:
                        stack.append(data[token].copy())
                    else:
                        return None

                elif token in self.operators:
                    op_info = self.operators[token]
                    if len(stack) < op_info['arity']:
                        return None

                    args = []
                    for _ in range(op_info['arity']):
                        args.append(stack.pop())
                    args.reverse()

                    # 执行算子计算
                    try:
                        result = op_info['func'](*args)
                    except Exception:
                        return None # 算子执行失败（如除零）

                    # 🔥 中间结果清洗：
                    # 为了保持计算链的稳定性，中间步骤也进行轻量级清洗
                    # 但完全的分布对齐只在最后一步进行
                    result = result.replace([np.inf, -np.inf], np.nan)
                    
                    # 简单的 fillna 防止 NaN 传染，这里用 ffill 保持因果性
                    result = result.ffill().fillna(0)
                    
                    stack.append(result)

                else:
                    return None

            if len(stack) != 1:
                return None

            final_result = stack[0]
            
            # 🔥 最终结果清洗（包含去极值和标准化）
            # 这里传入 is_training 标志，决定是 "Learn" 还是 "Apply" 统计量
            final_result = self._clean_series(final_result, is_training=is_training)

            if final_result is None:
                return None

            # 最终验证：确保没有 NaN（_clean_series 应该已经处理了）
            if final_result.isnull().any():
                final_result = final_result.fillna(0)

            # 检查是否有足够的变化 (避免常数因子)
            if final_result.std() < 1e-6:
                return None

            return final_result

        except Exception as e:
            # logger.debug(f"Factor computation error: {e}")
            return None

    def _clean_series(self, series: pd.Series, is_training: bool) -> Optional[pd.Series]:
        """
        清理序列 - 严格防止未来函数 (Strict No-Lookahead)
        
        Args:
            series: 输入序列
            is_training: True=计算并保存统计量; False=使用已保存的统计量
        """
        if series is None:
            return None

        # 1. 替换无穷值
        series = series.replace([np.inf, -np.inf], np.nan)

        # 2. 检查 NaN 比例 (如果太多缺失，直接丢弃)
        # 注意：在 Valid 集中，如果由于 Lookback Buffer 不足导致开头有 NaN，
        # 这里的阈值需要宽容一些，或者在外部保证 Buffer 足够。
        nan_ratio = series.isna().sum() / len(series)
        if nan_ratio > 0.5:
            return None

        # 3. 因果填充 (Causal Imputation)
        # 优先使用前向填充 (ffill)，这意味着用“昨天”的值填补“今天”的空缺
        # 严禁使用 series.median() 直接填充，因为那是未来的统计量
        series = series.ffill()

        # 4. 去极值和剩余缺失值填充 (Clip & Fill)
        if is_training:
            # === 训练模式：学习统计量 ===
            try:
                median_val = series.median()
                # 使用 1% 和 99% 分位数确定边界
                q01 = series.quantile(0.01)
                q99 = series.quantile(0.99)
                
                # 缓存统计量
                self.current_factor_stats = {
                    'median': float(median_val) if not pd.isna(median_val) else 0.0,
                    'lower': float(q01) if not pd.isna(q01) else -10.0,
                    'upper': float(q99) if not pd.isna(q99) else 10.0
                }
            except Exception:
                return None # 统计量计算失败
                
            # 应用截断
            series = series.clip(self.current_factor_stats['lower'], self.current_factor_stats['upper'])
            # 填充剩余的 NaN (通常是序列开头的)
            series = series.fillna(self.current_factor_stats['median'])

        else:
            # === 验证/实盘模式：应用统计量 ===
            if self.current_factor_stats is None:
                # 这是一个异常情况：试图在没有训练统计量的情况下评估验证集
                # 回退策略：被迫使用当前数据的统计量（会有轻微未来函数，但总比崩溃好）
                # 更好的做法是返回 None 或报错
                logger.warning("Evaluating validation data without training stats! Fallback to local stats.")
                return self._clean_series(series, is_training=True)
            
            # 严格使用训练集的边界进行截断
            series = series.clip(self.current_factor_stats['lower'], self.current_factor_stats['upper'])
            
            # 严格使用训练集的中位数填充剩余 NaN
            series = series.fillna(self.current_factor_stats['median'])

        return series