"""
回测模块 - 完整的因子回测和分析
包括：
1. 单因子回测（自动计算IC）
2. 因子组合回测（从alpha_pool计算）
3. 手续费优化
4. 训练集vs测试集对比
5. 因子相关性分析
6. 收益曲线对比图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
class Backtester:
    """高级回测器 - 带IC方向调整和手续费优化"""
    def __init__(self, 
                 prediction_horizon: int = 10,
                 bar_minutes: int = 15,
                 transaction_cost: float = 0.0005,
                 max_position: float = 0.1,
                 signal_threshold: float = 0.1):  # 兼容性参数，暂时保留
        """
        参数:
        ----
        max_position: 最大持仓比例（默认0.1即10%）
        注意：信号生成使用滚动分位数方法，与训练时保持一致
        """
        self.prediction_horizon = prediction_horizon
        self.bar_minutes = bar_minutes
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.signal_threshold = signal_threshold  # 兼容性保留，暂不使用
        
        self.bars_per_year = 365 * 24 * 60 / bar_minutes
        self.trades_per_year = self.bars_per_year / prediction_horizon
        
        print(f"📊 回测器初始化:")
        print(f"   预测周期: {prediction_horizon} bars")
        print(f"   交易成本: {transaction_cost*100:.3f}%")
        print(f"   最大持仓: {max_position*100:.1f}%")
        print(f"   信号生成: 滚动分位数方法 (q_low=0.3, q_high=0.7)")
        print(f"   持仓策略: 始终保持持仓，中性区域1/10仓位")
    
    def generate_signals(self, factor: pd.Series, lookback: int = 100) -> pd.Series:
        """
        生成交易信号 - 始终保持持仓

        使用滚动分位数方法：
        1. 计算滚动分位数阈值（q_low=0.3, q_high=0.7）
        2. 生成连续信号：
           - 高于q_high: 满仓做多 (max_position)
           - 低于q_low: 满仓做空 (-max_position)
           - 中间区域: 基础持仓 (max_position / 10)，即始终保持1/10仓位

        注意：
        - 始终保持持仓，避免空仓期
        - 中性区域保持小仓位，根据因子方向决定多空
        """
        # 使用滚动分位数
        q_low = 0.3
        q_high = 0.7

        # 计算滚动分位数阈值
        rolling_low = factor.rolling(window=lookback, min_periods=20).quantile(q_low)
        rolling_high = factor.rolling(window=lookback, min_periods=20).quantile(q_high)
        rolling_mid = factor.rolling(window=lookback, min_periods=20).median()

        # 生成信号：始终有持仓
        signals = pd.Series(0.0, index=factor.index)

        # 强信号区域：满仓
        signals[factor > rolling_high] = self.max_position
        signals[factor < rolling_low] = -self.max_position

        # 中性区域：保持基础持仓（1/10仓位）
        neutral_mask = (factor >= rolling_low) & (factor <= rolling_high)
        # 根据因子相对中位数的位置决定多空方向
        # signals[neutral_mask & (factor >= rolling_mid)] = self.max_position / 10
        # signals[neutral_mask & (factor < rolling_mid)] = -self.max_position / 10
        return signals
    
    def backtest_single_factor(self,
                               factor: pd.Series,
                               returns: pd.Series,
                               lookback: int = 100,
                               name: str = "Factor",
                               factor_is_normalized: bool = False) -> Dict:
        """
        单因子回测 - 与训练时完全一致

        关键逻辑：
        1. IC用原始因子值计算（更真实反映预测能力）
        2. 信号生成用标准化因子值（统一尺度）

        参数:
        ----
        factor: 因子值（原始值或标准化值）
        returns: 实际收益率（target）
        lookback: 回看窗口
        name: 因子名称
        factor_is_normalized: 因子是否已标准化（默认False，即原始值）

        返回:
        ----
        包含各种指标的字典，包括自动计算的IC和Sharpe
        """
        # 对齐数据
        aligned = pd.DataFrame({
            'factor': factor,
            'returns': returns
        }).dropna()
        if len(aligned) < lookback + self.prediction_horizon * 5:
            return {'error': 'Insufficient data', 'name': name}

        # ⭐ 计算IC：用原始因子值
        ic = aligned['factor'].corr(aligned['returns'])

        # ⭐ 生成信号：如果因子未标准化，先标准化
        if not factor_is_normalized:
            # 对因子进行滚动标准化（与训练时一致）
            factor_for_signal = aligned['factor'].copy()
            rolling_mean = factor_for_signal.rolling(window=lookback, min_periods=20).mean()
            rolling_std = factor_for_signal.rolling(window=lookback, min_periods=20).std()
            factor_for_signal = (factor_for_signal - rolling_mean) / (rolling_std + 1e-8)
            factor_for_signal = factor_for_signal.fillna(0).clip(-3, 3)
        else:
            factor_for_signal = aligned['factor']

        # 生成信号（基于标准化后的因子值）
        signals = self.generate_signals(factor_for_signal, lookback)
        
        # 非重叠采样
        valid_start = lookback
        valid_indices = list(range(valid_start,len(aligned)))
        
        if len(valid_indices) < 10:
            return {'error': 'Insufficient trades', 'name': name}
        
        signals_sampled = signals.iloc[valid_indices]
        returns_sampled = aligned['returns'].iloc[valid_indices]
        
        # 计算收益
        gross_returns = signals_sampled * returns_sampled
        
        # 计算交易成本（换手成本）
        position_changes = signals_sampled.diff().abs().fillna(signals_sampled.abs())
        turnover = position_changes.sum() / len(position_changes)  # 平均每期换手
        transaction_costs = position_changes * self.transaction_cost
        
        # 净收益
        net_returns = gross_returns - transaction_costs
        
        # 计算各种指标
        metrics = self._calculate_comprehensive_metrics(
            net_returns, gross_returns, signals_sampled, turnover
        )
        
        # 添加额外信息（包括IC）
        metrics.update({
            'name': name,
            'ic': ic,  # 🆕 IC作为输出结果
            'ic_direction': 'Long' if ic >= 0 else 'Short',
            'equity_curve': (1 + net_returns).cumprod(),
            'gross_equity': (1 + gross_returns).cumprod(),
            'signals': signals_sampled,
            'returns': net_returns,
            'gross_returns': gross_returns,
            'costs': transaction_costs,
            'turnover': turnover,
            'valid_indices': valid_indices,
        })
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, 
                                        net_returns: pd.Series,
                                        gross_returns: pd.Series,
                                        signals: pd.Series,
                                        turnover: float) -> Dict:
        """计算全面的回测指标"""
        if len(net_returns) < 5:
            return {}
        
        # 基础统计
        total_return = (1 + net_returns).prod() - 1
        gross_total_return = (1 + gross_returns).prod() - 1
        mean_return = net_returns.mean()
        std_return = net_returns.std()
        
        # 年化收益
        num_trades = len(net_returns)
        annual_return = (1 + total_return) ** (self.trades_per_year / num_trades) - 1
        
        # Sharpe比率
        if std_return > 1e-8:
            sharpe = np.sqrt(self.trades_per_year) * mean_return / std_return
        else:
            sharpe = 0
        
        # 最大回撤
        cum_returns = (1 + net_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Calmar比率
        if max_drawdown > 1e-6:
            calmar = annual_return / max_drawdown
        else:
            calmar = 0
        
        # 胜率 - 现在所有交易都有持仓
        win_rate = (net_returns > 0).sum() / len(net_returns) if len(net_returns) > 0 else 0
        num_active_trades = len(net_returns)

        # 盈亏比
        winning_returns = net_returns[net_returns > 0]
        losing_returns = net_returns[net_returns < 0]
        
        if len(losing_returns) > 0 and losing_returns.mean() != 0:
            profit_loss_ratio = abs(winning_returns.mean() / losing_returns.mean())
        else:
            profit_loss_ratio = np.nan
        
        # 信息比率
        if std_return > 1e-8:
            ir = mean_return / std_return
        else:
            ir = 0
        
        # 波动率（年化）
        volatility = std_return * np.sqrt(self.trades_per_year)
        
        # 平均持仓
        avg_position = signals.abs().mean()
        
        # 手续费占比
        cost_ratio = ((gross_total_return - total_return) / (gross_total_return + 1e-8)) if gross_total_return > 0 else np.nan
        
        return {
            'total_return': total_return,
            'gross_total_return': gross_total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'information_ratio': ir,
            'num_trades': num_trades,
            'num_active_trades': num_active_trades,
            'avg_position': avg_position,
            'cost_ratio': cost_ratio,
            'mean_return': mean_return,
            'std_return': std_return,
        }
    
    def backtest_factor_on_both_sets(self,
                                     train_factor: pd.Series,
                                     test_factor: pd.Series,
                                     train_returns: pd.Series,
                                     test_returns: pd.Series,
                                     name: str = "Factor") -> Dict:
        """
        在训练集和测试集上回测同一个因子
        
        参数:
        ----
        train_factor: 训练集因子值
        test_factor: 测试集因子值
        train_returns: 训练集收益率
        test_returns: 测试集收益率
        name: 因子名称
        
        返回:
        ----
        {
            'train': train_metrics,  # 包含IC
            'test': test_metrics,    # 包含IC
            'name': name
        }
        """
        print(f"\n📊 回测因子: {name}")
        
        # 训练集回测（自动计算IC）
        train_metrics = self.backtest_single_factor(
            train_factor, train_returns, name=f"{name}_train"
        )
        
        # 测试集回测（自动计算IC）
        test_metrics = self.backtest_single_factor(
            test_factor, test_returns, name=f"{name}_test"
        )
        
        # 打印IC信息
        if 'error' not in train_metrics and 'error' not in test_metrics:
            train_ic = train_metrics.get('ic', 0)
            test_ic = test_metrics.get('ic', 0)
            print(f"   训练集 IC: {train_ic:.4f} ({'Long' if train_ic >= 0 else 'Short'})")
            print(f"   测试集 IC: {test_ic:.4f} ({'Long' if test_ic >= 0 else 'Short'})")
        
            print(f"   训练集 - 收益: {train_metrics['total_return']*100:.2f}%, "
                  f"Sharpe: {train_metrics['sharpe_ratio']:.2f}, "
                  f"换手: {train_metrics['turnover']:.2f}")
            print(f"   测试集 - 收益: {test_metrics['total_return']*100:.2f}%, "
                  f"Sharpe: {test_metrics['sharpe_ratio']:.2f}, "
                  f"换手: {test_metrics['turnover']:.2f}")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'name': name
        }


class FactorAnalyzer:
    """因子分析器 - 可视化和相关性分析"""
    
    def __init__(self):
        self.results = []
    
    def add_backtest_result(self, result: Dict):
        """添加回测结果"""
        self.results.append(result)
    
    def plot_all_factors_comparison(self, save_path: str = 'factors_comparison.png'):
        """
        绘制所有因子的训练集vs测试集对比图
        每个因子一个子图，显示训练集和测试集的equity curve
        """
        valid_results = [r for r in self.results 
                        if 'error' not in r.get('train', {}) 
                        and 'error' not in r.get('test', {})]
        
        if not valid_results:
            print("❌ 没有有效的回测结果")
            return
        
        n_factors = len(valid_results)
        n_cols = 3
        n_rows = (n_factors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_factors > 1 else [axes]
        
        for idx, result in enumerate(valid_results):
            ax = axes[idx]
            
            train_metrics = result['train']
            test_metrics = result['test']
            name = result['name']
            ic = result.get('ic', 0)
            
            # 训练集equity curve
            train_equity = train_metrics['equity_curve']
            ax.plot(train_equity.values, label='Train', linewidth=2, alpha=0.8)
            
            # 测试集equity curve
            test_equity = test_metrics['equity_curve']
            ax.plot(test_equity.values, label='Test', linewidth=2, alpha=0.8)
            
            # 标题和统计
            train_return = train_metrics['total_return'] * 100
            test_return = test_metrics['total_return'] * 100
            train_sharpe = train_metrics['sharpe_ratio']
            test_sharpe = test_metrics['sharpe_ratio']
            
            title = (f"{name}\n"
                    f"IC={ic:.3f} | Train: {train_return:.1f}% (SR={train_sharpe:.2f}) | "
                    f"Test: {test_return:.1f}% (SR={test_sharpe:.2f})")
            ax.set_title(title, fontsize=10, fontweight='bold')
            
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Return')
        
        # 隐藏多余的子图
        for idx in range(n_factors, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 因子对比图已保存: {save_path}")
        plt.show()
    
    def plot_factors_correlation_heatmap(self, 
                                        train_factors: Dict[str, pd.Series],
                                        test_factors: Dict[str, pd.Series] = None,
                                        save_path: str = 'factors_correlation.png'):
        """
        绘制因子相关性热力图
        
        参数:
        ----
        train_factors: {factor_name: factor_series} 训练集因子
        test_factors: {factor_name: factor_series} 测试集因子（可选）
        """
        # 创建训练集因子矩阵
        train_df = pd.DataFrame(train_factors)
        train_df = train_df.dropna()
        
        # 计算相关系数矩阵
        train_corr = train_df.corr(method='spearman')
        
        # 如果有测试集，也计算
        if test_factors:
            test_df = pd.DataFrame(test_factors)
            test_df = test_df.dropna()
            test_corr = test_df.corr(method='spearman')
            
            # 创建2个子图
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # 训练集热力图
            sns.heatmap(train_corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=0, vmin=-1, vmax=1, square=True,
                       cbar_kws={'label': 'Spearman Correlation'},
                       ax=axes[0])
            axes[0].set_title('Training Set - Factor Correlation', 
                            fontsize=14, fontweight='bold', pad=20)
            
            # 测试集热力图
            sns.heatmap(test_corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=0, vmin=-1, vmax=1, square=True,
                       cbar_kws={'label': 'Spearman Correlation'},
                       ax=axes[1])
            axes[1].set_title('Test Set - Factor Correlation', 
                            fontsize=14, fontweight='bold', pad=20)
            
        else:
            # 只有训练集
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(train_corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=0, vmin=-1, vmax=1, square=True,
                       cbar_kws={'label': 'Spearman Correlation'},
                       ax=ax)
            ax.set_title('Factor Correlation Heatmap', 
                        fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 因子相关性热力图已保存: {save_path}")
        plt.show()
        
        # 打印高相关性的因子对
        print("\n🔍 高相关性因子对 (|corr| > 0.7):")
        high_corr_pairs = []
        for i in range(len(train_corr.columns)):
            for j in range(i+1, len(train_corr.columns)):
                corr_val = train_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((
                        train_corr.columns[i],
                        train_corr.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"   {f1} <-> {f2}: {corr:.3f}")
        else:
            print("   未发现高相关性因子对")
    
    def generate_summary_table(self) -> pd.DataFrame:
        """生成汇总表格"""
        valid_results = [r for r in self.results 
                        if 'error' not in r.get('train', {}) 
                        and 'error' not in r.get('test', {})]
        
        if not valid_results:
            return pd.DataFrame()
        
        summary_data = []
        for result in valid_results:
            train = result['train']
            test = result['test']
            
            summary_data.append({
                'Factor': result['name'],
                'IC': result.get('ic', 0),
                'Direction': 'Long' if result.get('ic', 0) >= 0 else 'Short',
                'Train_Return': train['total_return'],
                'Test_Return': test['total_return'],
                'Train_Sharpe': train['sharpe_ratio'],
                'Test_Sharpe': test['sharpe_ratio'],
                'Train_MaxDD': train['max_drawdown'],
                'Test_MaxDD': test['max_drawdown'],
                'Train_Turnover': train['turnover'],
                'Test_Turnover': test['turnover'],
                'Train_WinRate': train['win_rate'],
                'Test_WinRate': test['win_rate'],
                'Sharpe_Decay': train['sharpe_ratio'] - test['sharpe_ratio'],
            })
        
        df = pd.DataFrame(summary_data)
        
        # 格式化
        df['Train_Return'] = df['Train_Return'].apply(lambda x: f"{x*100:.2f}%")
        df['Test_Return'] = df['Test_Return'].apply(lambda x: f"{x*100:.2f}%")
        df['IC'] = df['IC'].apply(lambda x: f"{x:.4f}")
        df['Train_Sharpe'] = df['Train_Sharpe'].apply(lambda x: f"{x:.2f}")
        df['Test_Sharpe'] = df['Test_Sharpe'].apply(lambda x: f"{x:.2f}")
        df['Train_MaxDD'] = df['Train_MaxDD'].apply(lambda x: f"{x*100:.2f}%")
        df['Test_MaxDD'] = df['Test_MaxDD'].apply(lambda x: f"{x*100:.2f}%")
        df['Train_Turnover'] = df['Train_Turnover'].apply(lambda x: f"{x:.2f}")
        df['Test_Turnover'] = df['Test_Turnover'].apply(lambda x: f"{x:.2f}")
        df['Sharpe_Decay'] = df['Sharpe_Decay'].apply(lambda x: f"{x:.2f}")
        
        return df
    
    def print_summary(self):
        """打印汇总信息"""
        print("\n" + "="*100)
        print("📊 因子回测汇总")
        print("="*100)
        
        df = self.generate_summary_table()
        if len(df) > 0:
            print(df.to_string(index=False))
            
            # 保存到CSV
            csv_path = 'backtest_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✅ 汇总表格已保存: {csv_path}")
        else:
            print("❌ 没有有效的回测结果")

def compute_factor_from_tokens(
    tokens: List[str],
    data: pd.DataFrame,
    operators: Dict,
    rolling_window: int = 100,  # 滚动窗口大小，默认100个周期
    normalize: bool = True  # 是否进行滚动标准化
) -> Optional[pd.Series]:
    """
    根据tokens表达式计算因子值
    参数:
    ----
    tokens: list - 表达式tokens，如 ['<BEG>', 'return', 'low', 'mul', '<SEP>']
    data: pd.DataFrame - 原始特征数据
    operators: dict - 算子字典
    rolling_window: int - 滚动标准化窗口大小，默认100
    normalize: bool - 是否进行滚动标准化，默认True
    返回:
    ----
    pd.Series - 计算得到的因子值（根据normalize参数决定是否标准化）
    """
    # 移除<BEG>和<SEP>
    expr_tokens = tokens[1:-1]
    stack = []
    
    for token in expr_tokens:
        # 如果是特征名（列名）
        if token in data.columns:
            stack.append(data[token].copy())
        # 如果是算子
        elif token in operators:
            op_info = operators[token]
            arity = op_info['arity']
            func = op_info['func']
            
            # 检查栈是否有足够的操作数
            if len(stack) < arity:
                return None
            
            # 弹出操作数
            args = []
            for _ in range(arity):
                args.append(stack.pop())
            args.reverse()  # 恢复正确的顺序
            
            result = func(*args)
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.ffill().bfill().fillna(0)
            
            stack.append(result)
    
    if len(stack) != 1:
        return None

    factor_values = stack[0]

    # 处理异常值
    factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
    factor_values = factor_values.ffill().bfill().fillna(0)

    # 根据normalize参数决定是否标准化
    if normalize:
        rolling_mean = factor_values.rolling(window=rolling_window, min_periods=20).mean()
        rolling_std = factor_values.rolling(window=rolling_window, min_periods=20).std()
        rolling_std = rolling_std.replace(0, np.nan)
        factor_values = (factor_values - rolling_mean) / (rolling_std + 1e-8)
        factor_values = factor_values.fillna(0)
        factor_values = factor_values.clip(-3, 3)  # 与训练时保持一致：clip到[-3,3]

    return factor_values

def compute_linear_combination(alpha_pool: List[Dict], weights: np.ndarray, data: pd.DataFrame,
                              rolling_window: int = 100) -> pd.Series:
    """
    计算因子的线性组合（与训练时保持一致）

    关键逻辑：
    1. 对每个单独的因子进行滚动标准化（normalize=True）
    2. 加权组合标准化后的因子
    3. 组合结果不再标准化（与训练时一致）

    参数:
    ----
    alpha_pool: list - 因子池
    weights: np.ndarray - 权重（针对标准化后的因子）
    data: pd.DataFrame - 原始数据
    rolling_window: int - 滚动标准化窗口大小，默认100

    返回:
    ----
    pd.Series - 线性组合后的因子值（未标准化）
    """
    if len(alpha_pool) == 0:
        return pd.Series(0, index=data.index)

    # 计算所有因子值（每个因子都标准化）
    factor_values_list = []
    valid_weights = []

    for i, alpha_info in enumerate(alpha_pool):
        tokens = alpha_info['tokens']
        operators = alpha_info['operators']

        # ⭐ 关键：计算因子值时进行标准化（normalize=True）
        factor_values = compute_factor_from_tokens(
            tokens, data, operators,
            rolling_window=rolling_window,
            normalize=True  # 对单个因子标准化
        )

        if factor_values is not None:
            factor_values_list.append(factor_values)
            valid_weights.append(weights[i])

    if len(factor_values_list) == 0:
        return pd.Series(0, index=data.index)

    # 线性组合（不再标准化，与训练时一致）
    combination = pd.Series(0, index=data.index)
    for weight, factor_vals in zip(valid_weights, factor_values_list):
        combination += weight * factor_vals

    return combination


def calculate_future_returns(close_prices: pd.Series, periods: int = 10) -> pd.Series:
    future_returns = close_prices.pct_change(periods).shift(-periods)
    return future_returns


def backtest_miner_with_plot(miner, data_split: str = 'test', top_n: int = 10,
                             save_path: str = 'factors_backtest_comparison.png') -> Dict:
    """
    回测miner对象中的所有因子和组合，并画图对比
    
    参数:
    ----
    miner: OptimizedSynergisticFactorMiner - 训练好的挖掘器
    data_split: str - 'train', 'val', 或 'test'
    top_n: int - 回测前N个因子
    save_path: str - 图片保存路径
    
    返回:
    ----
    dict - 包含individual和combined回测结果
    """
    print(f"\n{'='*80}")
    print(f"📊 完整回测并画图：{data_split.upper()} 数据集")
    print(f"{'='*80}")
    
    # 1. 获取数据
    if data_split == 'train':
        data = miner.train_data
    elif data_split == 'val':
        data = miner.val_data
    else:
        data = miner.test_data
    print(f"数据量: {len(data)} bars")
    # 2. 获取因子池和权重
    combiner = miner.combination_model
    alpha_pool = combiner.alpha_pool
    if combiner.combiner_type == 'linear':
        weights = np.array(combiner.weights)
    else:
        weights = np.ones(len(alpha_pool)) / len(alpha_pool)
    n_factors = min(top_n, len(alpha_pool))
    print(f"因子池大小: {len(alpha_pool)}, 回测前{n_factors}个")
    
    # 3. 计算目标
    prediction_horizon = miner.config.prediction_horizon
    target_returns = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
    
    # 4. 创建回测器
    backtester = Backtester(
        prediction_horizon=prediction_horizon,
        bar_minutes=miner.config.bar_minutes,
        transaction_cost=0.0005,
        max_position=0.1
    )
    
    # 5. 回测每个单独的因子
    print("\n回测单个因子...")
    individual_results = {}
    
    for i in range(n_factors):
        alpha_info = alpha_pool[i]
        tokens = alpha_info['tokens']
        operators = alpha_info['operators']

        # ⭐ 计算因子值：使用原始值（normalize=False）用于IC计算
        factor_values = compute_factor_from_tokens(
            tokens, data, operators,
            rolling_window=100,
            normalize=False  # 使用原始因子值
        )

        if factor_values is None:
            continue

        # 回测（内部会标准化用于信号生成，但IC用原始值）
        result = backtester.backtest_single_factor(
            factor=factor_values,
            returns=target_returns,
            lookback=100,
            name=f"Factor_{i}",
            factor_is_normalized=False  # 明确告知是原始值
        )
        
        if 'error' not in result:
            individual_results[f"Factor_{i}"] = result
            tokens_short = '_'.join(tokens[1:min(3, len(tokens)-1)])
            print(f"  Factor {i} ({tokens_short}): IC={result['ic']:.3f}, "
                  f"Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']*100:.1f}%")
    
    # 6. 回测组合因子
    print("\n回测因子组合...")
    # 组合因子 = 标准化因子的加权和（结果未标准化，但尺度合理）
    combined_factor = compute_linear_combination(alpha_pool, weights, data, rolling_window=100)

    # 回测组合因子
    # 注意：组合因子不是标准化的，但尺度合理（输入是[-3,3]范围的标准化因子）
    # IC计算：用原始组合值
    # 信号生成：内部会标准化
    combined_result = backtester.backtest_single_factor(
        factor=combined_factor,
        returns=target_returns,
        lookback=100,
        name="Combined",
        factor_is_normalized=False  # 组合结果不是标准化的，需要在信号生成时标准化
    )
    
    if 'error' not in combined_result:
        print(f"  组合: IC={combined_result['ic']:.3f}, "
              f"Sharpe={combined_result['sharpe_ratio']:.2f}, "
              f"Return={combined_result['total_return']*100:.1f}%")
    
    # 7. 画图
    print(f"\n绘制收益曲线对比图...")
    plot_backtest_comparison(individual_results, combined_result, data_split, save_path)
    
    # 8. 打印汇总表
    print_backtest_summary_table(individual_results, combined_result)
    
    return {
        'individual': individual_results,
        'combined': combined_result
    }


def plot_backtest_comparison(individual_results: Dict, combined_result: Dict,
                            data_split: str, save_path: str):
    """
    绘制所有因子和组合的收益曲线对比图
    """
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Factor Backtest Comparison - {data_split.upper()} Set',
                fontsize=16, fontweight='bold')
    
    # 1. 左上：所有因子的权益曲线
    ax1 = axes[0, 0]
    
    # 画每个因子
    for name, result in individual_results.items():
        equity = result['equity_curve']
        ax1.plot(equity.values, label=name, alpha=0.6, linewidth=1)
    
    # 画组合（加粗）
    if combined_result and 'error' not in combined_result:
        combined_equity = combined_result['equity_curve']
        ax1.plot(combined_equity.values, label='Combined (Weighted)',
                color='red', linewidth=3, alpha=0.9)
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    ax1.set_title('Equity Curves: All Factors vs Combined', fontweight='bold')
    ax1.set_xlabel('Trading Periods')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. 右上：只显示Top5因子 + 组合
    ax2 = axes[0, 1]
    
    # 按Sharpe排序，取Top5
    sorted_factors = sorted(individual_results.items(),
                          key=lambda x: x[1]['sharpe_ratio'],
                          reverse=True)[:5]
    
    for name, result in sorted_factors:
        equity = result['equity_curve']
        ax2.plot(equity.values, label=f"{name} (SR={result['sharpe_ratio']:.1f})",
                linewidth=2, alpha=0.8)
    
    # 画组合
    if combined_result and 'error' not in combined_result:
        combined_equity = combined_result['equity_curve']
        ax2.plot(combined_equity.values,
                label=f"Combined (SR={combined_result['sharpe_ratio']:.1f})",
                color='red', linewidth=3, alpha=0.9)
    
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('Top 5 Factors by Sharpe + Combined', fontweight='bold')
    ax2.set_xlabel('Trading Periods')
    ax2.set_ylabel('Cumulative Return')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 左下：Sharpe比率对比
    ax3 = axes[1, 0]
    
    names = list(individual_results.keys())
    sharpes = [individual_results[n]['sharpe_ratio'] for n in names]
    
    # 添加组合
    if combined_result and 'error' not in combined_result:
        names.append('Combined')
        sharpes.append(combined_result['sharpe_ratio'])
    
    colors = ['steelblue'] * len(individual_results)
    if combined_result and 'error' not in combined_result:
        colors.append('red')
    
    bars = ax3.barh(range(len(names)), sharpes, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel('Sharpe Ratio')
    ax3.set_title('Sharpe Ratio Comparison', fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, sharpes)):
        ax3.text(val, i, f' {val:.2f}', va='center', fontsize=8)
    
    # 4. 右下：总收益和IC对比
    ax4 = axes[1, 1]
    
    returns = [individual_results[n]['total_return'] * 100 for n in individual_results.keys()]
    ics = [individual_results[n]['ic'] for n in individual_results.keys()]
    
    # 添加组合
    if combined_result and 'error' not in combined_result:
        returns.append(combined_result['total_return'] * 100)
        ics.append(combined_result['ic'])
    
    # 散点图
    scatter = ax4.scatter(ics[:-1] if len(ics) > len(individual_results) else ics,
                         returns[:-1] if len(returns) > len(individual_results) else returns,
                         s=100, alpha=0.6, c='steelblue', label='Individual Factors')
    
    # 组合点（红色加大）
    if combined_result and 'error' not in combined_result:
        ax4.scatter([ics[-1]], [returns[-1]], s=300, alpha=0.9,
                   c='red', marker='*', label='Combined', edgecolors='darkred', linewidths=2)
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('IC')
    ax4.set_ylabel('Total Return (%)')
    ax4.set_title('Return vs IC', fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存: {save_path}")
    plt.show()


def print_backtest_summary_table(individual_results: Dict, combined_result: Dict):
    """
    打印回测结果汇总表格
    """
    print(f"\n{'='*80}")
    print("📊 回测结果汇总表")
    print(f"{'='*80}")
    
    # 表头
    print(f"{'Factor':<15} {'IC':>8} {'Sharpe':>8} {'Return(%)':>12} {'MaxDD(%)':>12} {'WinRate(%)':>12}")
    print(f"{'-'*80}")
    
    # 单个因子
    for name, result in individual_results.items():
        print(f"{name:<15} {result['ic']:>8.4f} {result['sharpe_ratio']:>8.2f} "
              f"{result['total_return']*100:>12.2f} {result['max_drawdown']*100:>12.2f} "
              f"{result['win_rate']*100:>12.1f}")
    
    # 分隔线
    print(f"{'-'*80}")
    
    # 组合
    if combined_result and 'error' not in combined_result:
        print(f"{'COMBINED':<15} {combined_result['ic']:>8.4f} "
              f"{combined_result['sharpe_ratio']:>8.2f} "
              f"{combined_result['total_return']*100:>12.2f} "
              f"{combined_result['max_drawdown']*100:>12.2f} "
              f"{combined_result['win_rate']*100:>12.1f}")
    
    print(f"{'='*80}\n")


class MinerBacktester:
    """
    Miner回测器 - 一键回测miner对象

    使用方法：
    --------
    # 方法1: 从训练好的miner对象回测
    backtester = MinerBacktester(miner_linear)
    backtester.run('test')  # 自动回测并画图

    # 方法2: 从保存的JSON文件加载最佳因子组合回测
    backtester = MinerBacktester.from_best_solution('best_solutions.json', data, operators)
    backtester.run('test')
    """

    def __init__(self, miner):
        """
        参数:
        ----
        miner: OptimizedSynergisticFactorMiner - 训练好的挖掘器对象
        """
        self.miner = miner
        self.results = {}
        print(f"✅ 回测器已创建，因子池大小: {len(miner.combination_model.alpha_pool)}")
    
    def run(self, data_split: str = 'test', top_n: int = 10, 
            save_path: str = None, show_plot: bool = True):
        """
        运行回测并生成报告
        
        参数:
        ----
        data_split: 'train', 'val', 或 'test'
        top_n: 回测前N个因子
        save_path: 图片保存路径（默认自动生成）
        show_plot: 是否显示图表
        
        返回:
        ----
        dict - 回测结果
        """
        if save_path is None:
            save_path = f'{data_split}_factors_backtest.png'
        
        # 调用回测函数
        results = backtest_miner_with_plot(
            self.miner, 
            data_split=data_split, 
            top_n=top_n,
            save_path=save_path
        )
        
        self.results[data_split] = results
        return results
    
    def run_all(self, top_n: int = 10):
        """
        在训练集、验证集、测试集上都运行回测
        """
        print(f"\n{'='*80}")
        print("📊 在所有数据集上运行回测")
        print(f"{'='*80}\n")
        
        # 训练集
        self.run('train', top_n=top_n, save_path='train_backtest.png')
        
        # 验证集
        self.run('val', top_n=top_n, save_path='val_backtest.png')
        
        # 测试集
        self.run('test', top_n=top_n, save_path='test_backtest.png')
        
        # 打印对比
        self._print_comparison()
        
        return self.results
    
    def _print_comparison(self):
        """打印三数据集对比"""
        print(f"\n{'='*80}")
        print("📊 三数据集组合因子对比")
        print(f"{'='*80}")
        print(f"{'指标':<20} {'训练集':<15} {'验证集':<15} {'测试集':<15}")
        print(f"{'-'*80}")
        
        metrics_to_show = [
            ('IC', 'ic', '{:.4f}'),
            ('Sharpe比率', 'sharpe_ratio', '{:.2f}'),
            ('总收益(%)', 'total_return', '{:.2f}', 100),
            ('年化收益(%)', 'annual_return', '{:.2f}', 100),
            ('最大回撤(%)', 'max_drawdown', '{:.2f}', 100),
            ('胜率(%)', 'win_rate', '{:.2f}', 100),
        ]
        
        for metric_display, metric_key, fmt, *scale in metrics_to_show:
            multiplier = scale[0] if scale else 1
            
            train_val = self.results.get('train', {}).get('combined', {}).get(metric_key, 0) * multiplier
            val_val = self.results.get('val', {}).get('combined', {}).get(metric_key, 0) * multiplier
            test_val = self.results.get('test', {}).get('combined', {}).get(metric_key, 0) * multiplier
            
            print(f"{metric_display:<20} {fmt.format(train_val):<15} {fmt.format(val_val):<15} {fmt.format(test_val):<15}")
        
        print(f"{'='*80}\n")
    
    def get_combined_result(self, data_split: str = 'test'):
        """获取组合因子的回测结果"""
        if data_split not in self.results:
            print(f"❌ {data_split} 数据集尚未回测，请先运行 run('{data_split}')")
            return None
        return self.results[data_split]['combined']

    def get_individual_results(self, data_split: str = 'test'):
        """获取单个因子的回测结果"""
        if data_split not in self.results:
            print(f"❌ {data_split} 数据集尚未回测，请先运行 run('{data_split}')")
            return None
        return self.results[data_split]['individual']

    @classmethod
    def from_best_solution(cls, json_path: str, data: pd.DataFrame,
                          operators: Dict, config=None,
                          snapshot_index: int = -1):
        """
        从保存的best_solutions.json文件加载最佳因子组合并创建回测器

        参数:
        ----
        json_path: str - best_solutions.json文件路径
        data: pd.DataFrame - 完整数据（包含train/val/test）
        operators: Dict - 算子字典（需要与训练时相同）
        config: TrainingConfig - 配置对象（可选，用于数据分割）
        snapshot_index: int - 使用第几个快照（默认-1表示最后一个，即最佳）

        返回:
        ----
        MinerBacktester - 回测器对象

        示例:
        ----
        >>> # 加载数据和算子
        >>> data = pd.read_csv('your_data.csv')
        >>> from operators import TimeSeriesOperators
        >>> ts_ops = TimeSeriesOperators()
        >>> operators = {...}  # 构建算子字典
        >>>
        >>> # 从JSON加载并回测
        >>> backtester = MinerBacktester.from_best_solution(
        ...     'best_solutions.json', data, operators
        ... )
        >>> results = backtester.run('test')
        """
        import json
        from pathlib import Path
        from config import TrainingConfig

        # 读取JSON文件
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"❌ 找不到文件: {json_path}")

        with json_file.open('r', encoding='utf-8') as f:
            data_loaded = json.load(f)

        snapshots = data_loaded.get('snapshots', [])
        if not snapshots:
            raise ValueError(f"❌ JSON文件中没有快照数据")

        # 选择快照
        if snapshot_index < 0:
            snapshot_index = len(snapshots) + snapshot_index

        if snapshot_index < 0 or snapshot_index >= len(snapshots):
            raise IndexError(f"❌ 快照索引 {snapshot_index} 超出范围 [0, {len(snapshots)-1}]")

        snapshot = snapshots[snapshot_index]

        print(f"")
        print(f"{'='*70}")
        print(f"📂 从JSON加载最佳因子组合")
        print(f"{'='*70}")
        print(f"文件: {json_path}")
        print(f"快照: #{snapshot_index} / {len(snapshots)}")
        print(f"训练时间: Iteration {snapshot['iteration']}")
        print(f"验证分数: {snapshot['val_score']:.4f}")
        print(f"因子数量: {snapshot['pool_size']}")
        print(f"{'='*70}")

        # 创建一个伪miner对象（只包含必要的属性）
        class PseudoMiner:
            """伪miner对象 - 用于从JSON加载因子池"""
            def __init__(self, data, operators, snapshot, config):
                self.config = config or TrainingConfig()

                # 数据分割
                train_size = int(len(data) * self.config.train_ratio)
                val_size = int(len(data) * self.config.val_ratio)

                self.train_data = data.iloc[:train_size].copy()
                self.val_data = data.iloc[train_size:train_size+val_size].copy()
                self.test_data = data.iloc[train_size+val_size:].copy()

                # 填充缺失值
                self.train_data = self.train_data.ffill().bfill().fillna(0)
                self.val_data = self.val_data.ffill().bfill().fillna(0)
                self.test_data = self.test_data.ffill().bfill().fillna(0)

                # 创建伪组合模型
                class PseudoCombinationModel:
                    def __init__(self, snapshot, operators):
                        self.alpha_pool = []
                        self.weights = []
                        self.combiner_type = 'linear'  # 假设是linear

                        # 从snapshot加载因子池
                        for factor_info in snapshot['factors']:
                            alpha_info = {
                                'tokens': factor_info['tokens'],
                                'timestamp': factor_info.get('timestamp', 0),
                                'operators': operators
                            }
                            self.alpha_pool.append(alpha_info)

                            # 加载权重
                            weight = factor_info.get('weight')
                            if weight is not None:
                                self.weights.append(weight)

                        print(f"✅ 加载了 {len(self.alpha_pool)} 个因子")
                        if self.weights:
                            print(f"✅ 加载了 {len(self.weights)} 个权重")
                            print(f"   权重范围: [{min(self.weights):.4f}, {max(self.weights):.4f}]")

                self.combination_model = PseudoCombinationModel(snapshot, operators)

        pseudo_miner = PseudoMiner(data, operators, snapshot, config)

        # 创建回测器
        backtester = cls(pseudo_miner)
        print(f"✅ 回测器创建成功\n")

        return backtester


def load_and_backtest_from_json(json_path: str, data: pd.DataFrame,
                                operators: Dict, config=None,
                                data_split: str = 'test',
                                snapshot_index: int = -1,
                                save_path: str = None,
                                top_n: int = 10) -> Dict:
    backtester = MinerBacktester.from_best_solution(
        json_path, data, operators, config, snapshot_index
    )

    if save_path is None:
        save_path = f'{data_split}_backtest_from_json.png'

    results = backtester.run(data_split=data_split, save_path=save_path, top_n=top_n)

    return results


def demonstrate_backtest_workflow():
    """演示完整的回测流程"""
    print("="*80)
    print("🚀 因子回测演示")
    print("="*80)
    
    # 1. 创建示例数据
    print("\n📊 Step 1: 创建示例数据")
    n_bars = 2000
    returns = np.random.randn(n_bars) * 0.015
    prices = 10000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'returns': np.concatenate([[0], np.diff(np.log(prices))])
    })
    
    # 分割训练集和测试集
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"   训练集: {len(train_data)} bars")
    print(f"   测试集: {len(test_data)} bars")
    
    # 2. 创建示例因子
    print("\n🔧 Step 2: 创建示例因子")
    
    # 因子1: 动量因子
    train_data['factor1'] = train_data['close'].pct_change(5)
    test_data['factor1'] = test_data['close'].pct_change(5)
    
    # 因子2: 反转因子
    train_data['factor2'] = -train_data['close'].pct_change(20)
    test_data['factor2'] = -test_data['close'].pct_change(20)
    
    # 因子3: 波动率因子
    train_data['factor3'] = train_data['returns'].rolling(10).std()
    test_data['factor3'] = test_data['returns'].rolling(10).std()
    
    print("   因子已创建，IC将在回测时自动计算")
    
    # 3. 创建回测器和分析器
    print("\n📈 Step 3: 回测")
    backtester = Backtester(
        prediction_horizon=10,
        transaction_cost=0.0005,
        signal_threshold=0.1  # 降低换手
    )
    
    analyzer = FactorAnalyzer()
    
    # 回测每个因子（不需要预先计算IC）
    factors_info = [
        ('Factor1_Momentum', 'factor1'),
        ('Factor2_Reversal', 'factor2'),
        ('Factor3_Volatility', 'factor3'),
    ]
    
    for name, col in factors_info:
        result = backtester.backtest_factor_on_both_sets(
            train_data[col],
            test_data[col],
            train_data['returns'],
            test_data['returns'],
            name=name
        )
        analyzer.add_backtest_result(result)
    
    # 4. 可视化
    print("\n📊 Step 4: 生成可视化")
    
    # 绘制对比图
    analyzer.plot_all_factors_comparison('demo_factors_comparison.png')
    
    # 绘制相关性热力图
    train_factors = {
        'Factor1': train_data['factor1'],
        'Factor2': train_data['factor2'],
        'Factor3': train_data['factor3'],
    }
    
    test_factors = {
        'Factor1': test_data['factor1'],
        'Factor2': test_data['factor2'],
        'Factor3': test_data['factor3'],
    }
    
    analyzer.plot_factors_correlation_heatmap(
        train_factors, test_factors, 'demo_factors_correlation.png'
    )
    
    # 5. 打印汇总
    analyzer.print_summary()
    
    print("\n" + "="*80)
    print("✅ 演示完成！")
    print("="*80)


if __name__ == "__main__":
    demonstrate_backtest_workflow()