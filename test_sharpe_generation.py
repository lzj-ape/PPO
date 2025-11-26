"""
单元测试：验证组合Sharpe生成逻辑的合理性
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "config"))
sys.path.append(str(ROOT / "factor"))

from config import TrainingConfig
from evaluator import ICDiversityEvaluator


def _make_targets_and_predictions(n_points: int = 800):
    """
    构造一个可重复的数据集：
    - target: 平滑信号 + 少量噪声
    - predictive: target 加微弱噪声（应取得更高Sharpe）
    - random_signal: 与target无关的噪声
    """
    rng = np.random.default_rng(2024)
    idx = pd.date_range("2024-01-01", periods=n_points, freq="15T")

    smooth_trend = np.sin(np.linspace(0, 20 * np.pi, n_points)) * 0.001
    noise = rng.normal(0, 0.0005, size=n_points)
    target = pd.Series(smooth_trend + noise, index=idx)

    predictive = target + rng.normal(0, 0.0001, size=n_points)
    random_signal = rng.normal(0, 0.001, size=n_points)

    return target, pd.Series(predictive, index=idx), pd.Series(random_signal, index=idx)


def test_predictive_signal_has_higher_sharpe():
    """
    高度相关的预测信号应当产生更高的滚动Sharpe稳定性得分。
    """
    config = TrainingConfig()
    evaluator = ICDiversityEvaluator(config)

    target, predictive, random_signal = _make_targets_and_predictions()

    sharpe_predictive = evaluator.calculate_rolling_sharpe_stability(predictive, target)
    sharpe_random = evaluator.calculate_rolling_sharpe_stability(random_signal, target)

    assert sharpe_predictive > sharpe_random + 0.05


def test_constant_signal_returns_zero_sharpe():
    """
    常数信号无法产生有效交易，Sharpe应为0。
    """
    config = TrainingConfig()
    evaluator = ICDiversityEvaluator(config)

    target, _, _ = _make_targets_and_predictions()
    constant_signal = pd.Series(0.0, index=target.index)

    sharpe_constant = evaluator.calculate_rolling_sharpe_stability(constant_signal, target)
    assert sharpe_constant == pytest.approx(0.0, abs=1e-9)

