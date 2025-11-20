"""
时间序列操作符模块 - 扩展至50个算子
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


class TimeSeriesOperators:
    """时间序列操作符集合 - 50个算子"""
    
    # ============ 基础算术运算 (1-5) ============
    @staticmethod
    def add(x: pd.Series, y: pd.Series) -> pd.Series:
        """加法"""
        return (x + y).fillna(0)
    
    @staticmethod
    def sub(x: pd.Series, y: pd.Series) -> pd.Series:
        """减法"""
        return (x - y).fillna(0)
    
    @staticmethod
    def mul(x: pd.Series, y: pd.Series) -> pd.Series:
        """乘法"""
        return (x * y).fillna(0)
    
    @staticmethod
    def div(x: pd.Series, y: pd.Series) -> pd.Series:
        """除法（安全）"""
        return (x / (y.replace(0, np.nan) + 1e-8)).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def pow_op(x: pd.Series, power: float = 2.0) -> pd.Series:
        """幂运算"""
        sign = np.sign(x)
        return (sign * np.power(x.abs(), power)).fillna(0).replace([np.inf, -np.inf], 0)
    
    # ============ 基础变换 (6-12) ============
    @staticmethod
    def abs_op(x: pd.Series) -> pd.Series:
        """绝对值"""
        return x.abs().fillna(0)
    
    @staticmethod
    def sign_op(x: pd.Series) -> pd.Series:
        """符号函数"""
        return np.sign(x).fillna(0)
    
    @staticmethod
    def log_op(x: pd.Series) -> pd.Series:
        """对数变换"""
        return np.log(x.abs() + 1e-8).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def exp_op(x: pd.Series) -> pd.Series:
        """指数变换（限制范围）"""
        return np.exp(np.clip(x, -10, 10)).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def sqrt_op(x: pd.Series) -> pd.Series:
        """平方根"""
        sign = np.sign(x)
        return (sign * np.sqrt(x.abs())).fillna(0)
    
    @staticmethod
    def sigmoid_op(x: pd.Series) -> pd.Series:
        """Sigmoid函数"""
        return (1 / (1 + np.exp(-np.clip(x, -10, 10)))).fillna(0)
    
    @staticmethod
    def tanh_op(x: pd.Series) -> pd.Series:
        """双曲正切"""
        return np.tanh(x).fillna(0)
    
    # ============ 时间序列基础 (13-20) ============
    @staticmethod
    def delay(x: pd.Series, periods: int = 1) -> pd.Series:
        """延迟/滞后"""
        return x.shift(periods).ffill().fillna(0)
    
    @staticmethod
    def delta(x: pd.Series, periods: int = 1) -> pd.Series:
        """差分"""
        return x.diff(periods).fillna(0)
    
    @staticmethod
    def momentum(x: pd.Series, window: int = 5) -> pd.Series:
        """动量（收益率）"""
        return (x / x.shift(window) - 1).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def rate_of_change(x: pd.Series, window: int = 10) -> pd.Series:
        """变化率"""
        return ((x - x.shift(window)) / (x.shift(window).abs() + 1e-8)).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def ts_rank(x: pd.Series, window: int = 10) -> pd.Series:
        """时序排名"""
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False
        ).fillna(0)
    
    @staticmethod
    def ts_min(x: pd.Series, window: int = 10) -> pd.Series:
        """滚动最小值"""
        return x.rolling(window=window, min_periods=1).min().fillna(0)
    
    @staticmethod
    def ts_max(x: pd.Series, window: int = 10) -> pd.Series:
        """滚动最大值"""
        return x.rolling(window=window, min_periods=1).max().fillna(0)
    
    @staticmethod
    def ts_argmin(x: pd.Series, window: int = 10) -> pd.Series:
        """最小值位置"""
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: s.argmin() / len(s), raw=True
        ).fillna(0)
    
    # ============ 移动平均 (21-25) ============
    @staticmethod
    def sma(x: pd.Series, window: int = 5) -> pd.Series:
        """简单移动平均"""
        return x.rolling(window=window, min_periods=1).mean().fillna(0)
    
    @staticmethod
    def ema(x: pd.Series, span: int = 5) -> pd.Series:
        """指数移动平均"""
        return x.ewm(span=span, adjust=False).mean().fillna(0)
    
    @staticmethod
    def wma(x: pd.Series, window: int = 10) -> pd.Series:
        """加权移动平均"""
        weights = np.arange(1, window + 1)
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: np.average(s, weights=weights[:len(s)]) if len(s) > 0 else 0, raw=True
        ).fillna(0)
    
    @staticmethod
    def dema(x: pd.Series, span: int = 10) -> pd.Series:
        """双重指数移动平均"""
        ema1 = x.ewm(span=span, adjust=False).mean()
        ema2 = ema1.ewm(span=span, adjust=False).mean()
        return (2 * ema1 - ema2).fillna(0)
    
    @staticmethod
    def tema(x: pd.Series, span: int = 10) -> pd.Series:
        """三重指数移动平均"""
        ema1 = x.ewm(span=span, adjust=False).mean()
        ema2 = ema1.ewm(span=span, adjust=False).mean()
        ema3 = ema2.ewm(span=span, adjust=False).mean()
        return (3 * ema1 - 3 * ema2 + ema3).fillna(0)
    
    # ============ 统计指标 (26-33) ============
    @staticmethod
    def std(x: pd.Series, window: int = 20) -> pd.Series:
        """标准差"""
        return x.rolling(window=window, min_periods=1).std().fillna(0)
    
    @staticmethod
    def variance(x: pd.Series, window: int = 20) -> pd.Series:
        """方差"""
        return x.rolling(window=window, min_periods=1).var().fillna(0)
    
    @staticmethod
    def skewness(x: pd.Series, window: int = 20) -> pd.Series:
        """偏度"""
        return x.rolling(window=window, min_periods=3).apply(
            lambda s: skew(s) if len(s) >= 3 else 0, raw=True
        ).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def kurt(x: pd.Series, window: int = 20) -> pd.Series:
        """峰度"""
        return x.rolling(window=window, min_periods=4).apply(
            lambda s: kurtosis(s) if len(s) >= 4 else 0, raw=True
        ).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def zscore(x: pd.Series, window: int = 20) -> pd.Series:
        """Z-Score标准化"""
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std()
        return ((x - mean) / (std + 1e-8)).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def quantile(x: pd.Series, window: int = 20, q: float = 0.5) -> pd.Series:
        """分位数"""
        return x.rolling(window=window, min_periods=1).quantile(q).fillna(0)
    
    @staticmethod
    def mad(x: pd.Series, window: int = 20) -> pd.Series:
        """平均绝对偏差"""
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: np.mean(np.abs(s - np.mean(s))), raw=True
        ).fillna(0)
    
    @staticmethod
    def covariance(x: pd.Series, y: pd.Series, window: int = 20) -> pd.Series:
        """协方差"""
        return x.rolling(window=window, min_periods=1).cov(y).fillna(0)
    
    # ============ 技术指标 (34-42) ============
    @staticmethod
    def rsi(x: pd.Series, window: int = 14) -> pd.Series:
        """相对强弱指标"""
        delta = x.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return (100 - 100 / (1 + rs)).fillna(50)
    
    @staticmethod
    def macd(x: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """MACD指标"""
        ema_fast = x.ewm(span=fast, adjust=False).mean()
        ema_slow = x.ewm(span=slow, adjust=False).mean()
        return (ema_fast - ema_slow).fillna(0)
    
    @staticmethod
    def bbands_upper(x: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """布林带上轨"""
        sma = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std()
        return (sma + num_std * std).fillna(0)
    
    @staticmethod
    def bbands_lower(x: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """布林带下轨"""
        sma = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std()
        return (sma - num_std * std).fillna(0)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均真实波幅（需要高开低收数据时使用）"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window, min_periods=1).mean().fillna(0)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """顺势指标"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window, min_periods=1).mean()
        mad_tp = tp.rolling(window=window, min_periods=1).apply(
            lambda s: np.mean(np.abs(s - np.mean(s))), raw=True
        )
        return ((tp - sma_tp) / (0.015 * mad_tp + 1e-8)).fillna(0).replace([np.inf, -np.inf], 0)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """威廉指标"""
        highest = high.rolling(window=window, min_periods=1).max()
        lowest = low.rolling(window=window, min_periods=1).min()
        return (-100 * (highest - close) / (highest - lowest + 1e-8)).fillna(-50)
    
    @staticmethod
    def stoch(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """随机振荡器"""
        lowest = low.rolling(window=window, min_periods=1).min()
        highest = high.rolling(window=window, min_periods=1).max()
        return (100 * (close - lowest) / (highest - lowest + 1e-8)).fillna(50)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均趋向指标（简化版）"""
        # 简化的ADX计算
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=window, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(window=window, min_periods=1).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(window=window, min_periods=1).mean() / (atr + 1e-8))
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        return dx.rolling(window=window, min_periods=1).mean().fillna(0).replace([np.inf, -np.inf], 0)
    
    # ============ 比较与逻辑 (43-47) ============
    @staticmethod
    def max_op(x: pd.Series, y: pd.Series) -> pd.Series:
        """最大值"""
        return np.maximum(x, y).fillna(0)
    
    @staticmethod
    def min_op(x: pd.Series, y: pd.Series) -> pd.Series:
        """最小值"""
        return np.minimum(x, y).fillna(0)
    
    @staticmethod
    def condition(cond: pd.Series, x: pd.Series, y: pd.Series) -> pd.Series:
        """条件选择：if cond > 0 then x else y"""
        result = np.where(cond > 0, x, y)
        return pd.Series(result, index=cond.index).fillna(0)
    
    @staticmethod
    def rank(x: pd.Series) -> pd.Series:
        """百分位排名"""
        return x.rank(pct=True).fillna(0.5)
    
    @staticmethod
    def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
        """缩放到[-a, a]"""
        x_abs_sum = x.abs().sum()
        if x_abs_sum < 1e-8:
            return x.fillna(0)
        return (x * a / x_abs_sum).fillna(0)
    
    # ============ 高级算子 (48-50) ============
    @staticmethod
    def correlation(x: pd.Series, y: pd.Series, window: int = 20) -> pd.Series:
        """滚动相关系数"""
        return x.rolling(window=window, min_periods=1).corr(y).fillna(0)
    
    @staticmethod
    def decay_linear(x: pd.Series, window: int = 10) -> pd.Series:
        """线性衰减加权"""
        weights = np.arange(1, window + 1)
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: np.average(s, weights=weights[:len(s)]) if len(s) > 0 else 0, raw=True
        ).fillna(0)
    
    @staticmethod
    def ts_prod(x: pd.Series, window: int = 10) -> pd.Series:
        """滚动连乘"""
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: np.prod(s), raw=True
        ).fillna(0).replace([np.inf, -np.inf], 0)

