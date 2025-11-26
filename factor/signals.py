"""
统一的信号生成和评估模块
包含:
1. SignalGenerator: 信号生成器(训练和回测共用)
2. PerformanceEvaluator: 性能评估器(计算Sharpe、换手等)
3. 确保训练时的评估和回测时的计算完全一致
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    统一的信号生成器
    
    特性:
    ----
    1. 使用滚动分位数方法生成信号
    2. 始终保持持仓(避免空仓期)
    3. 强信号区域满仓,中性区域保持基础仓位
    4. 训练和回测使用相同的逻辑
    
    参数:
    ----
    max_position: float - 最大持仓比例(默认0.1即10%)
    lookback: int - 滚动窗口大小(默认100)
    q_low: float - 低分位数阈值(默认0.3)
    q_high: float - 高分位数阈值(默认0.7)
    neutral_fraction: float - 中性区域持仓比例(默认0.1,即1/10仓位)
    min_periods: int - 最小计算周期(默认20)
    """
    
    def __init__(self,
                 max_position: float = 0.1,
                 lookback: int = 100,
                 q_low: float = 0.3,
                 q_high: float = 0.7,
                 neutral_fraction: float = 0.1,
                 min_periods: int = 20):
        self.max_position = max_position
        self.lookback = lookback
        self.q_low = q_low
        self.q_high = q_high
        self.neutral_fraction = neutral_fraction
        self.min_periods = min_periods
        
        logger.info(f"SignalGenerator initialized: "
                   f"max_pos={max_position}, lookback={lookback}, "
                   f"quantiles=({q_low}, {q_high})")
    
    def generate_signals(self, factor: pd.Series) -> pd.Series:
        """
        生成交易信号 - 始终保持持仓
        
        逻辑:
        ----
        1. 计算滚动分位数阈值(q_low, q_high)
        2. 高于q_high: 满仓做多(max_position)
        3. 低于q_low: 满仓做空(-max_position)
        4. 中间区域: 保持基础持仓(max_position * neutral_fraction)
        
        参数:
        ----
        factor: pd.Series - 因子值(应该已经标准化)
        
        返回:
        ----
        pd.Series - 信号序列,范围[-max_position, +max_position]
        """
        # 计算滚动分位数阈值
        rolling_low = factor.rolling(
            window=self.lookback, 
            min_periods=self.min_periods
        ).quantile(self.q_low)
        
        rolling_high = factor.rolling(
            window=self.lookback, 
            min_periods=self.min_periods
        ).quantile(self.q_high)
        
        rolling_mid = factor.rolling(
            window=self.lookback, 
            min_periods=self.min_periods
        ).median()
        
        # 初始化信号
        signals = pd.Series(0.0, index=factor.index)
        
        # 强信号区域: 满仓
        signals[factor > rolling_high] = self.max_position
        signals[factor < rolling_low] = -self.max_position
        
        # 中性区域: 保持基础持仓
        neutral_mask = (factor >= rolling_low) & (factor <= rolling_high)
        base_position = self.max_position * self.neutral_fraction
        
        # 根据相对中位数位置决定多空方向
        signals[neutral_mask & (factor >= rolling_mid)] = base_position
        signals[neutral_mask & (factor < rolling_mid)] = -base_position
        
        # 填充初始NaN为0
        signals = signals.fillna(0.0)
        
        return signals
    
    def calculate_turnover(self, signals: pd.Series) -> float:
        """
        计算平均换手率
        
        换手率 = 平均每期持仓变化量
        
        参数:
        ----
        signals: pd.Series - 信号序列
        
        返回:
        ----
        float - 平均换手率
        """
        position_changes = signals.diff().abs()
        turnover = position_changes.mean()
        return float(turnover) if np.isfinite(turnover) else 0.0


class PerformanceEvaluator:
    """
    性能评估器 - 统一的Sharpe、收益等计算
    
    特性:
    ----
    1. 计算滚动Sharpe稳定性得分
    2. 计算IC、换手率、收益等指标
    3. 训练和回测使用相同的逻辑
    
    参数:
    ----
    prediction_horizon: int - 预测周期
    bar_minutes: int - K线周期(分钟)
    transaction_cost: float - 交易成本(默认0.0005)
    signal_generator: SignalGenerator - 信号生成器
    """
    
    def __init__(self,
                 prediction_horizon: int,
                 bar_minutes: int,
                 transaction_cost: float = 0.0005,
                 signal_generator: SignalGenerator = None):
        self.prediction_horizon = prediction_horizon
        self.bar_minutes = bar_minutes
        self.transaction_cost = transaction_cost
        
        # 如果未提供信号生成器,创建默认的
        self.signal_generator = signal_generator or SignalGenerator()
        
        # 年化系数
        self.bars_per_year = 365 * 24 * 60 / bar_minutes
        
        logger.info(f"PerformanceEvaluator initialized: "
                   f"horizon={prediction_horizon}, bars_per_year={self.bars_per_year:.0f}")
    
    def calculate_ic(self, predictions: pd.Series, targets: pd.Series) -> float:
        """计算IC (信息系数)"""
        aligned = pd.DataFrame({'pred': predictions, 'target': targets}).dropna()
        if len(aligned) < 20:
            return 0.0
        if aligned['pred'].std() < 1e-9:
            return 0.0
        
        ic = aligned['pred'].corr(aligned['target'])
        return float(ic) if np.isfinite(ic) else 0.0
    
    def calculate_net_returns(self, 
                             predictions: pd.Series, 
                             targets: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算净收益、总收益和信号
        
        返回:
        ----
        Tuple[net_returns, gross_returns, signals]
        """
        # 对齐数据
        valid_idx = predictions.index.intersection(targets.index)
        if len(valid_idx) < 100:
            return pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float)
        
        pred_val = predictions.loc[valid_idx]
        target_val = targets.loc[valid_idx]
        
        # 检查预测值是否有效
        if pred_val.std() < 1e-12 or len(pred_val.unique()) <= 1:
            return pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float)
        
        # 生成信号
        signals = self.signal_generator.generate_signals(pred_val)
        
        # 计算总收益
        gross_returns = signals * target_val
        
        # 计算交易成本
        position_changes = signals.diff().abs().fillna(signals.abs())
        transaction_costs = position_changes * self.transaction_cost
        
        # 净收益
        net_returns = (gross_returns - transaction_costs).dropna()
        
        return net_returns, gross_returns.loc[net_returns.index], signals.loc[net_returns.index]
    
    def calculate_rolling_sharpe_stability(self,
                                          predictions: pd.Series,
                                          targets: pd.Series,
                                          window_days: int = 3,
                                          stability_penalty: float = 1.5) -> float:
        """
        计算滚动夏普稳定性得分
        
        Score = Mean(Rolling_Sharpe) - lambda * Std(Rolling_Sharpe)
        
        参数:
        ----
        predictions: 预测值(应该已标准化)
        targets: 目标值(未来收益)
        window_days: 滚动窗口天数(默认3天)
        stability_penalty: 稳定性惩罚系数(默认1.5)
        
        返回:
        ----
        float - 稳定性得分,范围[-10, 10]
        """
        try:
            # 1. 计算净收益
            net_returns, _, _ = self.calculate_net_returns(predictions, targets)
            
            if len(net_returns) == 0:
                logger.debug("calculate_rolling_sharpe_stability: no valid returns")
                return None  # 🔥 修复：返回None表示计算失败，而非0.0

            # 2. 动态调整窗口大小
            bars_per_day = 24 * 60 / max(self.bar_minutes, 1)
            ideal_window_bars = int(window_days * bars_per_day)

            data_length = len(net_returns)
            # 🔥 修复：降低最小数据要求 150 → 80
            # 原因：train_computation_failed 11/16，数据量要求太高导致Sharpe无法计算
            if data_length < 80:
                logger.debug(f"calculate_rolling_sharpe_stability: data_length({data_length}) < 80")
                return None  # 🔥 修复：返回None表示计算失败，而非0.0

            # 动态窗口: 最小30根,最大不超过数据量的1/5
            window_bars = max(30, min(ideal_window_bars, data_length // 5))
            min_required_bars = window_bars * 2

            if data_length < min_required_bars:
                logger.debug(f"calculate_rolling_sharpe_stability: insufficient data "
                           f"({data_length} < {min_required_bars})")
                return None  # 🔥 修复：返回None表示计算失败，而非0.0
            
            logger.debug(f"calculate_rolling_sharpe_stability: data_length={data_length}, "
                        f"window_bars={window_bars}")
            
            # 3. 计算滚动Sharpe
            rolling_mean = net_returns.rolling(
                window=window_bars, 
                min_periods=window_bars//2
            ).mean()
            
            rolling_std = net_returns.rolling(
                window=window_bars, 
                min_periods=window_bars//2
            ).std()
            
            # 过滤掉无效的标准差
            rolling_std = rolling_std.replace(0, np.nan)
            
            # 年化Sharpe
            rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)) * np.sqrt(self.bars_per_year)
            rolling_sharpe = rolling_sharpe.dropna()

            # 🔥 修改：只clip下限，保留高Sharpe值
            # 原因：高Sharpe是好事，不应该被限制
            # 下限 -50 是为了防止极端负值（通常是计算错误）
            rolling_sharpe = rolling_sharpe.clip(lower=-50)
            
            if len(rolling_sharpe) < 10:
                logger.debug(f"calculate_rolling_sharpe_stability: too few valid sharpe values "
                           f"({len(rolling_sharpe)})")
                return None  # 🔥 修复：返回None表示计算失败，而非0.0
            
            # 4. 计算稳定性得分
            mean_s = rolling_sharpe.mean()
            std_s = rolling_sharpe.std()

            logger.debug(f"calculate_rolling_sharpe_stability: mean_s={mean_s:.4f}, std_s={std_s:.4f}")

            # 🔥 修改：直接使用 Mean(Rolling Sharpe)，不再惩罚波动率
            # 原因：避免过度惩罚表现好但波动大的因子
            # 旧公式：stability_score = mean_s - stability_penalty * std_s
            # 新公式：stability_score = mean_s

            stability_score = mean_s

            # 🔥 修改：只设置下限，不设置上限
            # 原因：高 Sharpe 是好事，不应该被限制
            # 下限设为 -10.0 是为了防止极端负值影响训练
            stability_score = max(stability_score, -10.0)

            logger.debug(f"calculate_rolling_sharpe_stability: score={stability_score:.4f} (no penalty, no upper limit)")

            return float(stability_score)
            
        except Exception as e:
            logger.warning(f"Error in calculate_rolling_sharpe_stability: {e}")
            return None  # 🔥 修复：返回None表示计算失败，而非0.0
    
    def calculate_comprehensive_metrics(self,
                                       predictions: pd.Series,
                                       targets: pd.Series,
                                       window_days: int = 3) -> Dict:
        """
        计算全面的性能指标
        
        返回:
        ----
        Dict - 包含IC、Sharpe、收益、换手等所有指标
        """
        # 1. 计算IC
        ic = self.calculate_ic(predictions, targets)
        
        # 2. 计算净收益和信号
        net_returns, gross_returns, signals = self.calculate_net_returns(predictions, targets)
        
        if len(net_returns) < 5:
            return {'error': 'Insufficient valid returns'}
        
        # 3. 计算换手率
        turnover = self.signal_generator.calculate_turnover(signals)
        
        # 4. 基础统计
        total_return = (1 + net_returns).prod() - 1
        gross_total_return = (1 + gross_returns).prod() - 1
        mean_return = net_returns.mean()
        std_return = net_returns.std()
        
        # 5. 年化收益
        num_periods = len(net_returns)
        periods_per_year = self.bars_per_year / self.prediction_horizon
        annual_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
        
        # 6. Sharpe比率
        if std_return > 1e-8:
            sharpe = np.sqrt(periods_per_year) * mean_return / std_return
        else:
            sharpe = 0.0
        
        # 7. 最大回撤
        cum_returns = (1 + net_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # 8. Calmar比率
        if max_drawdown > 1e-6:
            calmar = annual_return / max_drawdown
        else:
            calmar = 0.0
        
        # 9. 胜率
        win_rate = (net_returns > 0).sum() / len(net_returns) if len(net_returns) > 0 else 0.0
        
        # 10. 盈亏比
        winning_returns = net_returns[net_returns > 0]
        losing_returns = net_returns[net_returns < 0]
        
        if len(losing_returns) > 0 and losing_returns.mean() != 0:
            profit_loss_ratio = abs(winning_returns.mean() / losing_returns.mean())
        else:
            profit_loss_ratio = np.nan
        
        # 11. 信息比率
        if std_return > 1e-8:
            ir = mean_return / std_return
        else:
            ir = 0.0
        
        # 12. 波动率(年化)
        volatility = std_return * np.sqrt(periods_per_year)
        
        # 13. 平均持仓
        avg_position = signals.abs().mean()
        
        # 14. 手续费占比
        if gross_total_return > 0:
            cost_ratio = (gross_total_return - total_return) / (gross_total_return + 1e-8)
        else:
            cost_ratio = np.nan
        
        # 15. 滚动Sharpe稳定性
        sharpe_stability = self.calculate_rolling_sharpe_stability(
            predictions, targets, window_days
        )
        # 🔥 修复：如果计算失败，使用0
        if sharpe_stability is None:
            sharpe_stability = 0.0

        return {
            'ic': ic,
            'total_return': total_return,
            'gross_total_return': gross_total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sharpe_stability': sharpe_stability,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'information_ratio': ir,
            'num_periods': num_periods,
            'avg_position': avg_position,
            'turnover': turnover,
            'cost_ratio': cost_ratio,
            'mean_return': mean_return,
            'std_return': std_return,
            'equity_curve': (1 + net_returns).cumprod(),
            'gross_equity': (1 + gross_returns).cumprod(),
            'signals': signals,
            'returns': net_returns,
            'gross_returns': gross_returns,
        }


# ==================== 使用示例 ====================

def demo_unified_evaluator():
    """演示统一评估器的使用"""
    print("="*80)
    print("🚀 统一评估器演示")
    print("="*80)
    
    # 1. 创建示例数据
    np.random.seed(42)
    n_bars = 2000
    
    # 生成价格序列
    returns = np.random.randn(n_bars) * 0.015
    prices = 10000 * np.exp(np.cumsum(returns))
    
    # 生成因子(动量因子)
    factor = pd.Series(prices).pct_change(5)
    
    # 标准化因子(训练时应该做这一步)
    factor_mean = factor.rolling(100, min_periods=20).mean()
    factor_std = factor.rolling(100, min_periods=20).std()
    factor_normalized = (factor - factor_mean) / (factor_std + 1e-8)
    factor_normalized = factor_normalized.fillna(0).clip(-3, 3)
    
    # 目标:未来10期收益
    target = pd.Series(prices).pct_change(10).shift(-10)
    
    print(f"数据量: {n_bars} bars")
    print(f"因子有效值: {factor_normalized.notna().sum()}")
    print(f"目标有效值: {target.notna().sum()}")
    
    # 2. 创建信号生成器和评估器
    signal_generator = SignalGenerator(
        max_position=0.1,
        lookback=100,
        q_low=0.3,
        q_high=0.7
    )
    
    evaluator = PerformanceEvaluator(
        prediction_horizon=10,
        bar_minutes=15,
        transaction_cost=0.0005,
        signal_generator=signal_generator
    )
    
    # 3. 计算指标
    print("\n📊 计算性能指标...")
    metrics = evaluator.calculate_comprehensive_metrics(
        factor_normalized, 
        target
    )
    
    if 'error' in metrics:
        print(f"❌ 错误: {metrics['error']}")
        return
    
    # 4. 打印结果
    print("\n" + "="*80)
    print("📈 评估结果")
    print("="*80)
    print(f"IC:              {metrics['ic']:.4f}")
    print(f"Sharpe比率:      {metrics['sharpe_ratio']:.2f}")
    print(f"Sharpe稳定性:    {metrics['sharpe_stability']:.2f}")
    print(f"总收益:          {metrics['total_return']*100:.2f}%")
    print(f"年化收益:        {metrics['annual_return']*100:.2f}%")
    print(f"最大回撤:        {metrics['max_drawdown']*100:.2f}%")
    print(f"胜率:            {metrics['win_rate']*100:.1f}%")
    print(f"换手率:          {metrics['turnover']:.4f}")
    print(f"平均持仓:        {metrics['avg_position']:.4f}")
    print(f"手续费占比:      {metrics['cost_ratio']*100:.2f}%")
    print("="*80)
    
    # 5. 可视化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 权益曲线
    ax1 = axes[0, 0]
    ax1.plot(metrics['equity_curve'].values, label='Net', linewidth=2)
    ax1.plot(metrics['gross_equity'].values, label='Gross', linewidth=2, alpha=0.7)
    ax1.set_title('Equity Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 信号
    ax2 = axes[0, 1]
    ax2.plot(metrics['signals'].values)
    ax2.set_title('Trading Signals')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 收益分布
    ax3 = axes[1, 0]
    ax3.hist(metrics['returns'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_title('Returns Distribution')
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.grid(True, alpha=0.3)
    
    # 回撤
    ax4 = axes[1, 1]
    cum_returns = metrics['equity_curve']
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    ax4.fill_between(range(len(drawdown)), 0, drawdown.values * 100, alpha=0.5, color='red')
    ax4.set_title('Drawdown (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unified_evaluator_demo.png', dpi=300, bbox_inches='tight')
    print("\n✅ 可视化已保存: unified_evaluator_demo.png")
    plt.show()


if __name__ == "__main__":
    demo_unified_evaluator()