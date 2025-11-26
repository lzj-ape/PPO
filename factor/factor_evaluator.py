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

            # 🔥 新增: 多样性检查 - 计算与池中现有因子的相似度
            diversity_penalty = 0.0
            if len(self.combination_model.alpha_pool) > 0:
                similarity_score = self._calculate_expression_similarity(tokens)
                # 相似度越高,惩罚越大
                if similarity_score > 0.7:
                    # 高度相似,重度惩罚
                    diversity_penalty = -0.5 * similarity_score
                elif similarity_score > 0.5:
                    # 中度相似,中度惩罚
                    diversity_penalty = -0.3 * similarity_score
                elif similarity_score > 0.3:
                    # 轻度相似,轻度惩罚
                    diversity_penalty = -0.1 * similarity_score
                # 否则不惩罚

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

            # 🔥 核心修复：统一使用增量Sharpe作为决策标准和PPO学习信号
            # 无论池子大小，都使用经过linear优化后的"增量Sharpe"来判断
            # 原因：
            # 1. 即使是单因子，combiner也会用Ridge优化权重，得到的是"组合"Sharpe
            # 2. 增量Sharpe = 新组合Sharpe - 旧组合Sharpe，才是真正的"贡献"
            # 3. 决策标准和PPO学习目标必须一致，否则策略会混乱

            # 根据池子大小调整阈值（而非改变评价指标）
            if current_pool_size < 3:
                # 前3个因子：允许轻微负值（因为样本少，不确定性大）
                ic_threshold = -0.03  # 允许-3%的负增量
            elif current_pool_size < 5:
                # 第4-5个因子：要求很小的正增量
                ic_threshold = 0.001  # 0.1%的增量即可
            elif current_pool_size < 10:
                # 第6-10个因子：要求中等增量
                ic_threshold = base_threshold * 0.3  # 0.3%的增量
            else:
                # 10个因子后：要求较高增量（池子已经很好了，新因子必须带来明显改进）
                ic_threshold = base_threshold * 0.6  # 0.6%的增量

            # 统一使用增量Sharpe
            decision_score = incremental_sharpe
            ppo_reward_signal = incremental_sharpe

            should_add = decision_score > ic_threshold and not trial_only
            actually_added = False

            # 🔥 诊断日志：记录拒绝的原因（显式打印）
            if not trial_only and decision_score <= ic_threshold:
                logger.info(f"❌ Factor REJECTED:")
                logger.info(f"   incremental_sharpe={decision_score:.6f} <= threshold={ic_threshold:.6f}")
                logger.info(f"   base_threshold={base_threshold:.6f}, pool_size={current_pool_size}")
                logger.info(f"   base_train_score={self.combination_model.base_train_score:.4f}")
                logger.info(f"   new_train_score={trial_result['train_stats']['sharpe']:.4f}")
                logger.info(f"   expression: {' '.join(tokens[:15])}...")

                # 🔥 额外诊断：分析为什么被拒绝
                if decision_score <= 0:
                    logger.info(f"   ⚠️  Reason: New factor does NOT improve the combination (negative/zero increment)")
                elif self.combination_model.base_train_score > 2.0 and decision_score < 0.01:
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
                logger.info(f"   incremental_sharpe={decision_score:.6f} > threshold={ic_threshold:.6f}")
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

            # 5. 🔥 应用高级奖励计算（惩罚项 + 多样性惩罚）
            # 使用 ppo_reward_signal 而非 incremental_sharpe，确保PPO学习到正确的信号
            final_reward = ppo_reward_signal + diversity_penalty
            penalty_components = {'diversity_penalty': diversity_penalty}

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
                penalty_components_extra = penalty_result.get('components', {})
                complexity_penalty = penalty_components_extra.get('complexity_penalty', 0.0)
                overfitting_penalty = penalty_components_extra.get('overfitting_penalty', 0.0)

                # 更新penalty_components
                penalty_components.update(penalty_components_extra)

                # 最终奖励 = PPO奖励信号 + 多样性惩罚 + 其他惩罚项
                final_reward = ppo_reward_signal + diversity_penalty + complexity_penalty + overfitting_penalty

                # logger.debug(f"Reward breakdown: ppo_signal={ppo_reward_signal:.4f}, "
                #            f"diversity={diversity_penalty:.4f}, complexity={complexity_penalty:.4f}, "
                #            f"overfitting={overfitting_penalty:.4f}, final={final_reward:.4f}")

            # 6. 返回结果（奖励是PPO reward signal + 惩罚）
            return {
                'valid': True,
                'reward': final_reward,  # 🔥 PPO学习信号（真实的增量Sharpe + 惩罚）
                'pool_size': current_pool_size,
                'added_to_pool': actually_added,  # 是否真的被添加（trial_only时为False）
                'qualifies': decision_score > ic_threshold,  # 🔥 修复：使用decision_score判断
                'incremental_sharpe': incremental_sharpe,  # 保持原始增量Sharpe供记录
                'ppo_reward_signal': ppo_reward_signal,  # 🔥 新增：显式返回PPO学习的信号
                'penalty_components': penalty_components,
                'train_factor': train_factor,
                'val_factor': val_factor,
                'alpha_info': alpha_info,
                'train_eval': {
                    'sharpe': train_score_after,
                    'ic': ppo_reward_signal * 0.5,  # 🔥 使用ppo_reward_signal
                    'composite_score': ppo_reward_signal
                },
                'val_eval': {
                    'sharpe': val_score_after,
                    'ic': ppo_reward_signal * 0.5,
                    'composite_score': val_stats.get('composite_score', 0.0)
                },
                'composite_score': final_reward
            }

        except Exception as e:
            # logger.debug(f"Expression evaluation error: {e}")
            # import traceback
            # logger.debug(traceback.format_exc())
            return {'valid': False, 'reason': str(e)}

    def _calculate_expression_similarity(self, tokens: List[str]) -> float:
        """
        计算新表达式与池中现有表达式的最大相似度

        相似度计算策略:
        1. Token序列的Jaccard相似度
        2. 结构相似度 (操作符序列)
        3. 返回最大相似度分数

        Args:
            tokens: 新表达式的token列表

        Returns:
            最大相似度分数 [0, 1]
        """
        if len(self.combination_model.alpha_pool) == 0:
            return 0.0

        new_tokens_set = set(tokens[1:-1])  # 去掉<BEG>和<SEP>
        new_operators = [t for t in tokens[1:-1] if t in self.operators]
        new_features = [t for t in tokens[1:-1] if t in self.feature_names]

        max_similarity = 0.0

        for alpha_info in self.combination_model.alpha_pool:
            existing_tokens = alpha_info['tokens']
            existing_tokens_set = set(existing_tokens[1:-1])
            existing_operators = [t for t in existing_tokens[1:-1] if t in self.operators]
            existing_features = [t for t in existing_tokens[1:-1] if t in self.feature_names]

            # 1. Token集合的Jaccard相似度
            if len(new_tokens_set) > 0 and len(existing_tokens_set) > 0:
                intersection = len(new_tokens_set & existing_tokens_set)
                union = len(new_tokens_set | existing_tokens_set)
                token_similarity = intersection / union if union > 0 else 0.0
            else:
                token_similarity = 0.0

            # 2. 操作符序列相似度
            if len(new_operators) > 0 and len(existing_operators) > 0:
                common_ops = len(set(new_operators) & set(existing_operators))
                total_ops = max(len(new_operators), len(existing_operators))
                operator_similarity = common_ops / total_ops if total_ops > 0 else 0.0
            else:
                operator_similarity = 0.0

            # 3. 特征序列相似度
            if len(new_features) > 0 and len(existing_features) > 0:
                common_features = len(set(new_features) & set(existing_features))
                total_features = max(len(new_features), len(existing_features))
                feature_similarity = common_features / total_features if total_features > 0 else 0.0
            else:
                feature_similarity = 0.0

            # 综合相似度 (加权平均)
            overall_similarity = (
                0.4 * token_similarity +
                0.4 * operator_similarity +
                0.2 * feature_similarity
            )

            max_similarity = max(max_similarity, overall_similarity)

        return max_similarity

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

                    # 🔥 中间结果清洗：更保守的策略，避免过度填充传播错误
                    result = result.replace([np.inf, -np.inf], np.nan)

                    # 🔥 修复：放宽NaN容忍度 0.5 → 0.7
                    # 原因：train_computation_failed 11/16，NaN检查过于严格导致计算失败
                    # 检查NaN比例，如果过高则认为计算失败
                    if len(result) > 0:
                        nan_ratio = result.isna().sum() / len(result)
                        if nan_ratio > 0.7:  # 从0.5提高到0.7
                            # NaN比例超过70%，中间步骤失败
                            return None

                    # 只在NaN比例不高时才填充
                    if result.isna().any():
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
        清理序列：去极值 + 标准化 (Z-Score)
        """
        if series is None:
            return None

        # 1. 基础清洗
        series = series.replace([np.inf, -np.inf], np.nan)
        # 🔥 修复：放宽NaN容忍度 0.5 → 0.7
        # 原因：train_computation_failed 11/16，NaN检查过于严格
        # 检查 NaN 比例
        if series.isna().sum() / len(series) > 0.7:  # 从0.5提高到0.7
            return None
        series = series.ffill()

        # 2. 计算/应用统计量
        if is_training:
            try:
                # 计算统计量
                median = series.median()
                # 这里的 quantile 范围可以适当放宽，比如 0.005 和 0.995
                lower = series.quantile(0.01)
                upper = series.quantile(0.99)
                
                # 先去极值，再算均值方差，这样更稳健
                clipped = series.clip(lower, upper)
                mean = clipped.mean()
                std = clipped.std()
                
                # 缓存统计量
                self.current_factor_stats = {
                    'median': float(median) if not pd.isna(median) else 0.0,
                    'lower': float(lower) if not pd.isna(lower) else -3.0,
                    'upper': float(upper) if not pd.isna(upper) else 3.0,
                    'mean': float(mean) if not pd.isna(mean) else 0.0,
                    'std': float(std) if not pd.isna(std) else 1.0,
                }
            except:
                return None
        
        # 检查是否有统计量可用
        if self.current_factor_stats is None:
            if not is_training:
                # 验证集没有统计量时的回退策略
                return self._clean_series(series, is_training=True)
            return None

        stats = self.current_factor_stats

        # 3. 执行清洗操作
        # A. 去极值 (Winsorization)
        series = series.clip(stats['lower'], stats['upper'])
        
        # B. 填充缺失值 (使用中位数)
        series = series.fillna(stats['median'])
        
        # C. 🔥 标准化 (Z-Score) - 这是你之前缺少的关键一步！
        if stats['std'] > 1e-8:
            series = (series - stats['mean']) / stats['std']
        else:
            series = series - stats['mean']
            
        return series