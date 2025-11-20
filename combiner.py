import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from config import TrainingConfig
# 注意：这里不再导入 ICDiversityEvaluator 以避免循环导入
# 我们将在运行时通过 set_evaluator 注入实例

logger = logging.getLogger(__name__)

class ImprovedCombinationModel:
    """
    基于 Ridge 回归的组合模型
    核心：Reward = Incremental Rolling Sharpe Stability Score
    """

    def __init__(self, config: TrainingConfig, max_alpha_count: int = 15):
        self.config = config
        self.max_alpha_count = max_alpha_count
        
        # 因子池信息
        self.alpha_pool: List[Dict] = []
        
        # 因子数据矩阵
        self.train_matrix: Optional[pd.DataFrame] = None
        self.val_matrix: Optional[pd.DataFrame] = None
        
        # 目标值
        self.train_target: Optional[pd.Series] = None
        self.val_target: Optional[pd.Series] = None
        
        # 模型与状态
        self.ridge_model = Ridge(alpha=1.0, fit_intercept=False) 
        self.current_weights: Optional[np.ndarray] = None
        self.evaluator = None # 类型: ICDiversityEvaluator
        
        # 缓存当前的基准分数
        self.base_train_score = 0.0
        self.base_val_score = 0.0
        
        # Rolling Sharpe 的参数
        self.rolling_window_days = getattr(config, 'rolling_window_days', 90)
        self.stability_penalty = getattr(config, 'stability_penalty', 1.5)

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def set_targets(self, train_target: pd.Series, val_target: pd.Series):
        self.train_target = train_target
        self.val_target = val_target

    def _align_and_clean(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """对齐特征和目标，并处理 NaN/Inf"""
        # 1. 确保索引交集
        valid_idx = X.index.intersection(y.index)
        
        # 2. 筛选并填充
        X_clean = X.loc[valid_idx].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        y_clean = y.loc[valid_idx].fillna(0.0)
        
        return X_clean, y_clean

    def evaluate_new_factor(self, alpha_info: Dict, 
                           train_factor: pd.Series, val_factor: pd.Series) -> Dict:
        """
        🔥 试算模式 (Trial Mode): 仅计算增量稳定性，不修改池子。
        """
        if self.evaluator is None or self.train_target is None: 
            return {'train_incremental_sharpe': 0.0, 'train_stats': {'sharpe': 0.0}, 'val_stats': {'sharpe': 0.0}}

        # 1. 对齐新因子数据到 Target 索引 (关键修复：防止索引错位)
        train_factor_aligned = train_factor.reindex(self.train_target.index).fillna(0.0)
        
        # 2. 构造临时训练矩阵
        if self.train_matrix is None or len(self.alpha_pool) == 0:
            # Case A: 池子为空
            temp_train_X = train_factor_aligned.to_frame(name='new')
        else:
            # Case B: 拼接现有矩阵 (使用 reindex 确保 train_matrix 也对齐)
            # 注意：这里为了效率，在实际大规模生产中应尽量避免每次都 concat DataFrame
            # 但为了代码清晰度，保持 concat
            current_X = self.train_matrix.reindex(self.train_target.index).fillna(0.0)
            temp_train_X = pd.concat([current_X, train_factor_aligned.rename('new')], axis=1)

        # 3. 拟合 Ridge 回归 (在 Train 上)
        X_train, y_train = self._align_and_clean(temp_train_X, self.train_target)
        
        if len(X_train) < 100: 
            return {'train_incremental_sharpe': 0.0, 'train_stats': {'sharpe': 0.0}, 'val_stats': {'sharpe': 0.0}}
        
        try:
            # 拟合
            self.ridge_model.fit(X_train.values, y_train.values)
            
            # 预测组合收益
            train_pred_vals = self.ridge_model.predict(X_train.values)
            train_pred_series = pd.Series(train_pred_vals, index=X_train.index)
            
            # 计算新的 Stability Score
            new_train_score = self.evaluator.calculate_rolling_sharpe_stability(
                train_pred_series, y_train,
                window_days=self.rolling_window_days, stability_penalty=self.stability_penalty
            )
            
            # 4. 计算增量 (Reward)
            incremental_score = new_train_score - self.base_train_score
            
            return {
                'train_incremental_sharpe': incremental_score, 
                'train_stats': {'sharpe': new_train_score, 'composite_score': new_train_score},
                # Val stats 暂略，以节省计算资源
                'val_stats': {'sharpe': 0.0, 'composite_score': 0.0},
            }
        except Exception as e:
            logger.error(f"Combiner trial failed: {e}")
            return {'train_incremental_sharpe': 0.0, 'train_stats': {'sharpe': 0.0}, 'val_stats': {'sharpe': 0.0}}

    def add_alpha_and_optimize(self, alpha_info: Dict, 
                              train_factor: pd.Series, val_factor: pd.Series) -> Dict:
        """
        🔥 提交模式 (Commit Mode): 真正将因子加入池子，并更新基准状态。
        """
        if self.train_target is None:
            return {}

        factor_name = f"alpha_{len(self.alpha_pool)}"
        
        # 1. 更新因子池元数据
        self.alpha_pool.append(alpha_info)
        
        # 2. 更新数据矩阵 (强制对齐)
        train_factor_aligned = train_factor.reindex(self.train_target.index).fillna(0.0)
        if self.val_target is not None:
            val_factor_aligned = val_factor.reindex(self.val_target.index).fillna(0.0)
        else:
            val_factor_aligned = pd.DataFrame()

        if self.train_matrix is None:
            self.train_matrix = train_factor_aligned.to_frame(name=factor_name)
            if not val_factor_aligned.empty:
                self.val_matrix = val_factor_aligned.to_frame(name=factor_name)
        else:
            self.train_matrix[factor_name] = train_factor_aligned
            if self.val_matrix is not None and not val_factor_aligned.empty:
                self.val_matrix[factor_name] = val_factor_aligned
            
        # 3. 重新拟合基准模型
        X_train, y_train = self._align_and_clean(self.train_matrix, self.train_target)
        
        if len(X_train) > 100:
            self.ridge_model.fit(X_train.values, y_train.values)
            self.current_weights = self.ridge_model.coef_
            
            # 4. 🔥 更新基准 Rolling Stability Score
            train_pred_vals = self.ridge_model.predict(X_train.values)
            train_pred_series = pd.Series(train_pred_vals, index=X_train.index)
            
            self.base_train_score = self.evaluator.calculate_rolling_sharpe_stability(
                train_pred_series, y_train,
                window_days=self.rolling_window_days, stability_penalty=self.stability_penalty
            )
            
            # Val Score Update (如果需要)
            if self.val_matrix is not None and self.val_target is not None:
                X_val, y_val = self._align_and_clean(self.val_matrix, self.val_target)
                if len(X_val) > 50:
                    val_pred = self.ridge_model.predict(X_val.values)
                    self.base_val_score = self.evaluator.calculate_rolling_sharpe_stability(
                        pd.Series(val_pred, index=X_val.index), y_val,
                        window_days=self.rolling_window_days, stability_penalty=self.stability_penalty
                    )
        
        # 5. 淘汰最差因子 (如果池子满了)
        if len(self.alpha_pool) > self.max_alpha_count:
            self._prune_factor()
            
        return {
            'pool_size': len(self.alpha_pool),
            'current_train_score': self.base_train_score,
            'current_val_score': self.base_val_score
        }

    def _prune_factor(self):
        """
        淘汰机制：移除权重绝对值最小的因子
        """
        if self.current_weights is None or len(self.alpha_pool) <= self.max_alpha_count: 
            return
        
        # 找到权重绝对值最小的索引
        min_idx = np.argmin(np.abs(self.current_weights))
        
        # 记录并移除
        col_to_drop = self.train_matrix.columns[min_idx]
        
        # 移除 Metadata
        self.alpha_pool.pop(min_idx)
        
        # 更新矩阵
        self.train_matrix.drop(columns=[col_to_drop], inplace=True)
        if self.val_matrix is not None:
            self.val_matrix.drop(columns=[col_to_drop], inplace=True)
            
        # 更新权重数组
        self.current_weights = np.delete(self.current_weights, min_idx)
        
        # 这里我们可以选择重新 fit，也可以暂时保持 current_weights 直到下一次 add
        # 为了保持 base_score 准确，建议重新 fit
        X_train, y_train = self._align_and_clean(self.train_matrix, self.train_target)
        if len(X_train) > 100:
            self.ridge_model.fit(X_train.values, y_train.values)
            self.current_weights = self.ridge_model.coef_
            
            # 更新 Base Score
            train_pred = self.ridge_model.predict(X_train.values)
            self.base_train_score = self.evaluator.calculate_rolling_sharpe_stability(
                pd.Series(train_pred, index=X_train.index), y_train,
                window_days=self.rolling_window_days, stability_penalty=self.stability_penalty
            )

    def evaluate_combination(self, use_val: bool = False) -> Dict:
        """返回当前的组合表现"""
        score = self.base_val_score if use_val else self.base_train_score
        return {'sharpe': score, 'composite_score': score}