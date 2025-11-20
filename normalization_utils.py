"""
标准化工具模块 - 统一的MAD去极值+滚动标准化方法
确保训练和回测使用完全相同的标准化逻辑
"""

import numpy as np
import pandas as pd
from typing import Optional


def normalize_series_with_mad(series: pd.Series, window: int = 100) -> Optional[pd.Series]:
    """
    滚动标准化 with MAD去极值 - 完全避免前视偏差

    步骤:
    1. 使用MAD (Median Absolute Deviation) 去极值
    2. 滚动标准化 (rolling + expanding混合避免前视偏差)

    参数:
        series: 原始因子值
        window: 滚动窗口大小

    返回:
        标准化后的序列（无前视偏差）
    """
    # 清理异常值
    series = series.replace([np.inf, -np.inf], np.nan)

    # ========== Step 1: MAD去极值 (Winsorization) ==========
    # 计算滚动中位数和MAD
    rolling_median = series.rolling(window=window, min_periods=20).median()
    rolling_mad = series.rolling(window=window, min_periods=20).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    # 计算expanding中位数和MAD (用于填充前window个点)
    expanding_median = series.expanding(min_periods=20).median()
    expanding_mad = series.expanding(min_periods=20).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    # 明确区分前window和后面的逻辑
    final_median = pd.Series(index=series.index, dtype=float)
    final_mad = pd.Series(index=series.index, dtype=float)

    # 前window个用expanding (更稳定)
    if len(series) >= window:
        final_median.iloc[:window] = expanding_median.iloc[:window]
        final_mad.iloc[:window] = expanding_mad.iloc[:window]

        # 之后用rolling (更敏感于市场变化)
        final_median.iloc[window:] = rolling_median.iloc[window:]
        final_mad.iloc[window:] = rolling_mad.iloc[window:]
    else:
        # 数据长度不足window，全部用expanding
        final_median = expanding_median
        final_mad = expanding_mad

    # 如果还有NaN (min_periods导致), 用expanding填充
    final_median = final_median.fillna(expanding_median)
    final_mad = final_mad.fillna(expanding_mad)

    # 定义上下限 (n_mad=5: 5倍MAD之外截断)
    n_mad = 5.0
    upper_limit = final_median + n_mad * final_mad
    lower_limit = final_median - n_mad * final_mad

    # 去极值: Winsorization (截断而不是删除)
    series_clipped = series.clip(lower=lower_limit, upper=upper_limit)

    # ========== Step 2: 滚动标准化 ==========
    # 计算去极值后的滚动均值和标准差
    rolling_mean = series_clipped.rolling(window=window, min_periods=20).mean()
    rolling_std = series_clipped.rolling(window=window, min_periods=20).std()

    expanding_mean = series_clipped.expanding(min_periods=20).mean()
    expanding_std = series_clipped.expanding(min_periods=20).std()

    # 明确区分前window和后面的逻辑
    final_mean = pd.Series(index=series.index, dtype=float)
    final_std = pd.Series(index=series.index, dtype=float)

    # 前window个用expanding
    if len(series) >= window:
        final_mean.iloc[:window] = expanding_mean.iloc[:window]
        final_std.iloc[:window] = expanding_std.iloc[:window]

        # 之后用rolling
        final_mean.iloc[window:] = rolling_mean.iloc[window:]
        final_std.iloc[window:] = rolling_std.iloc[window:]
    else:
        # 数据长度不足window，全部用expanding
        final_mean = expanding_mean
        final_std = expanding_std

    # 填充剩余NaN
    final_mean = final_mean.fillna(expanding_mean)
    final_std = final_std.fillna(expanding_std)

    # 检查标准差
    if final_std.mean() < 1e-6:
        return None

    # Z-Score标准化
    normalized = (series_clipped - final_mean) / (final_std + 1e-8)
    normalized = normalized.clip(-3, 3)  # 二次截断保护
    normalized = normalized.fillna(0)

    return normalized


def normalize_series_simple(series: pd.Series, window: int = 100) -> pd.Series:
    """
    简化版滚动标准化（不含MAD去极值）- 用于兼容性

    参数:
        series: 原始序列
        window: 滚动窗口

    返回:
        标准化序列
    """
    series = series.replace([np.inf, -np.inf], np.nan)

    rolling_mean = series.rolling(window=window, min_periods=20).mean()
    rolling_std = series.rolling(window=window, min_periods=20).std()

    expanding_mean = series.expanding(min_periods=20).mean()
    expanding_std = series.expanding(min_periods=20).std()

    final_mean = rolling_mean.fillna(expanding_mean)
    final_std = rolling_std.fillna(expanding_std)

    if final_std.mean() < 1e-6:
        return pd.Series(0, index=series.index)

    normalized = (series - final_mean) / (final_std + 1e-8)
    normalized = normalized.clip(-3, 3)
    normalized = normalized.fillna(0)

    return normalized
