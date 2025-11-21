"""
改进的数据生成器 - 避免未来函数并更贴近真实市场
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_realistic_market_data(
    n_rows: int = 10000,
    start_date: str = '2023-01-01',
    freq: str = '15min',
    seed: Optional[int] = 42,
    initial_price: float = 10000.0
) -> pd.DataFrame:
    """
    生成更贴近真实市场的OHLCV数据

    改进点:
    1. 严格的时间因果性 - 所有数据基于前期信息生成
    2. OHLC关系合理性 - High/Low基于Open/Close的合理范围
    3. Volume与波动率的相关性 - 大波动对应大成交量
    4. 避免未来信息泄露

    Args:
        n_rows: 数据行数
        start_date: 起始日期
        freq: 频率 (如 '15min', '1h', '1d')
        seed: 随机种子
        initial_price: 初始价格

    Returns:
        DataFrame with columns: [open, high, low, close, volume]
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n_rows, freq=freq)

    # ============================================
    # 第一步: 生成底层价格动力学
    # ============================================

    # 1.1 长期趋势 (周期性)
    trend = np.sin(np.linspace(0, 10 * np.pi, n_rows)) * 0.002

    # 1.2 动量效应 (分段随机游走)
    momentum_1 = np.cumsum(np.random.normal(0, 0.001, n_rows // 2))
    momentum_2 = np.cumsum(np.random.normal(0, 0.001, n_rows - n_rows // 2))
    momentum = np.concatenate([momentum_1, momentum_2])

    # 1.3 白噪音
    noise = np.random.normal(0, 0.001, n_rows)

    # 1.4 合成对数收益率
    log_returns = trend + momentum * 0.0005 + noise

    # 1.5 生成Close价格序列 (基于对数收益率)
    close_prices = initial_price * np.exp(np.cumsum(log_returns))

    # ============================================
    # 第二步: 基于Close生成OHLC (严格因果性)
    # ============================================

    ohlc_data = []

    for i in range(n_rows):
        current_close = close_prices[i]

        # 2.1 生成Open (基于前一周期Close + 跳空)
        if i == 0:
            # 第一根K线: Open基于初始价格的小幅随机偏移
            gap = np.random.normal(0, 0.0005)
            current_open = initial_price * (1 + gap)
        else:
            # 后续K线: Open = 前一周期Close + 跳空
            prev_close = close_prices[i - 1]
            gap = np.random.normal(0, 0.0003)  # 跳空幅度较小
            current_open = prev_close * (1 + gap)

        # 2.2 生成日内波动范围 (基于Open和Close)
        # 使用较大的值作为基准来计算high/low的范围
        reference_price = max(current_open, current_close)
        low_reference = min(current_open, current_close)

        # 日内波动幅度 (基于波动率)
        intraday_volatility = abs(log_returns[i]) * 0.5 + 0.0005

        # 2.3 生成High (必须 >= max(open, close))
        high_extension = abs(np.random.normal(0, intraday_volatility))
        current_high = reference_price * (1 + high_extension)

        # 2.4 生成Low (必须 <= min(open, close))
        low_extension = abs(np.random.normal(0, intraday_volatility))
        current_low = low_reference * (1 - low_extension)

        # 2.5 确保OHLC关系正确
        current_high = max(current_high, current_open, current_close)
        current_low = min(current_low, current_open, current_close)

        # 2.6 生成Volume (与波动率正相关)
        # 大波动 -> 大成交量
        price_range = (current_high - current_low) / current_close
        volume_base = 5000  # 基础成交量
        volume_volatility = abs(log_returns[i]) * 100000
        current_volume = volume_base + volume_volatility + np.random.uniform(0, 5000)
        current_volume = max(100, current_volume)  # 最小成交量100

        ohlc_data.append({
            'open': current_open,
            'high': current_high,
            'low': current_low,
            'close': current_close,
            'volume': current_volume
        })

    # ============================================
    # 第三步: 组装DataFrame
    # ============================================
    df = pd.DataFrame(ohlc_data, index=dates)

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    验证数据的完整性和合理性

    Args:
        df: 包含OHLCV的DataFrame

    Returns:
        验证结果字典
    """
    results = {
        'valid': True,
        'issues': []
    }

    # 检查1: NaN值
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        results['valid'] = False
        results['issues'].append(f'存在 {nan_count} 个NaN值')

    # 检查2: Inf值
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        results['valid'] = False
        results['issues'].append(f'存在 {inf_count} 个Inf值')

    # 检查3: OHLC关系
    if 'high' in df.columns and 'low' in df.columns:
        high_low_valid = (df['high'] >= df['low']).all()
        if not high_low_valid:
            results['valid'] = False
            results['issues'].append('存在 High < Low 的情况')

    if 'high' in df.columns and 'close' in df.columns:
        high_close_valid = (df['high'] >= df['close']).all()
        if not high_close_valid:
            results['valid'] = False
            results['issues'].append('存在 High < Close 的情况')

    if 'low' in df.columns and 'close' in df.columns:
        low_close_valid = (df['low'] <= df['close']).all()
        if not low_close_valid:
            results['valid'] = False
            results['issues'].append('存在 Low > Close 的情况')

    if 'high' in df.columns and 'open' in df.columns:
        high_open_valid = (df['high'] >= df['open']).all()
        if not high_open_valid:
            results['valid'] = False
            results['issues'].append('存在 High < Open 的情况')

    if 'low' in df.columns and 'open' in df.columns:
        low_open_valid = (df['low'] <= df['open']).all()
        if not low_open_valid:
            results['valid'] = False
            results['issues'].append('存在 Low > Open 的情况')

    # 检查4: Volume非负
    if 'volume' in df.columns:
        volume_valid = (df['volume'] >= 0).all()
        if not volume_valid:
            results['valid'] = False
            results['issues'].append('存在负Volume')

    return results


if __name__ == '__main__':
    # 测试数据生成
    print("生成测试数据...")
    df = generate_realistic_market_data(n_rows=5000)

    print("\n数据预览:")
    print(df.head())
    print("\n数据统计:")
    print(df.describe())

    print("\n数据验证:")
    validation = validate_data(df)
    if validation['valid']:
        print("✅ 数据验证通过")
    else:
        print("❌ 数据验证失败:")
        for issue in validation['issues']:
            print(f"  - {issue}")
