"""
可视化和分析工具模块
负责训练过程的可视化和性能分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class VisualizationTools:
    """可视化工具 - 提供训练历史和性能分析的可视化"""

    def __init__(self, training_history: Dict, config):
        """
        初始化可视化工具

        Args:
            training_history: 训练历史字典
            config: 配置对象
        """
        self.training_history = training_history
        self.config = config

    def plot_training_history(self, save_path: str = None):
        """绘制训练历史"""
        if save_path is None:
            save_path = f'training_history_{self.config.combiner_type}.png'

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))

        # Rewards
        if self.training_history['rewards']:
            axes[0, 0].plot(self.training_history['rewards'], alpha=0.3, label='Raw')
            window = 50
            if len(self.training_history['rewards']) > window:
                smoothed = pd.Series(self.training_history['rewards']).rolling(window).mean()
                axes[0, 0].plot(smoothed, linewidth=2, label=f'MA({window})')
            axes[0, 0].set_title('Rewards Over Time', fontweight='bold')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Composite Scores
        if self.training_history['train_composite']:
            axes[0, 1].plot(self.training_history['train_composite'], label='Train', alpha=0.7)
        if self.training_history['val_composite']:
            axes[0, 1].plot(self.training_history['val_composite'], label='Val', alpha=0.7)
        axes[0, 1].set_title(f'Composite Score ({self.config.combiner_type})', fontweight='bold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Sharpe Ratio
        if self.training_history['train_metric1']:
            axes[1, 0].plot(self.training_history['train_metric1'], alpha=0.5, label='Train Sharpe')
        if self.training_history['val_metric1']:
            axes[1, 0].plot(self.training_history['val_metric1'], alpha=0.5, label='Val Sharpe')
        axes[1, 0].set_title('Sharpe Ratio', fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Sharpe')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # IC (Information Coefficient)
        if self.training_history['train_metric2']:
            axes[1, 1].plot(self.training_history['train_metric2'], alpha=0.5, label='Train IC')
        if self.training_history['val_metric2']:
            axes[1, 1].plot(self.training_history['val_metric2'], alpha=0.5, label='Val IC')
        axes[1, 1].set_title('Information Coefficient', fontweight='bold')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('IC')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Iteration-level Performance Changes
        if self.training_history['iteration_scores']:
            iter_data = self.training_history['iteration_scores']
            iterations = [d['iteration'] for d in iter_data]
            train_deltas = [d['train_delta'] for d in iter_data]
            val_deltas = [d['val_delta'] for d in iter_data]

            axes[2, 0].plot(iterations, train_deltas, alpha=0.4, label='Train Δ', color='blue')
            axes[2, 0].plot(iterations, val_deltas, alpha=0.4, label='Val Δ', color='orange')

            # 添加移动平均
            if len(train_deltas) > 10:
                train_ma = pd.Series(train_deltas).rolling(10).mean()
                val_ma = pd.Series(val_deltas).rolling(10).mean()
                axes[2, 0].plot(iterations, train_ma, linewidth=2,
                              label='Train MA(10)', color='darkblue')
                axes[2, 0].plot(iterations, val_ma, linewidth=2,
                              label='Val MA(10)', color='darkorange')

            # 标记PPO更新点
            if self.training_history['ppo_update_iterations']:
                for ppo_iter in self.training_history['ppo_update_iterations']:
                    axes[2, 0].axvline(x=ppo_iter, color='green',
                                      alpha=0.3, linestyle='--', linewidth=0.8)

            axes[2, 0].axhline(y=0, color='red', linestyle='-', alpha=0.5, linewidth=1)
            axes[2, 0].set_title('Per-Iteration Performance Change (with PPO updates)',
                                fontweight='bold')
            axes[2, 0].set_xlabel('Iteration')
            axes[2, 0].set_ylabel('Score Change (Δ)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

        # Factor Pool and Acceptance Rate
        if self.training_history['pool_size_history']:
            iterations = list(range(len(self.training_history['pool_size_history'])))
            axes[2, 1].plot(iterations, self.training_history['pool_size_history'],
                           label='Pool Size', color='purple', linewidth=2)
            axes[2, 1].set_ylabel('Pool Size', color='purple')
            axes[2, 1].tick_params(axis='y', labelcolor='purple')
            axes[2, 1].set_xlabel('Iteration')
            axes[2, 1].set_title('Factor Pool Size & Acceptance Rate', fontweight='bold')
            axes[2, 1].grid(True, alpha=0.3)

            # 添加第二个y轴显示接受率
            ax2 = axes[2, 1].twinx()
            additions = np.array(self.training_history['factor_additions'])
            rejections = np.array(self.training_history['factor_rejections'])
            total_attempts = additions + rejections
            acceptance_rate = np.where(total_attempts > 0, additions / total_attempts * 100, 0)

            # 平滑接受率
            if len(acceptance_rate) > 10:
                acceptance_ma = pd.Series(acceptance_rate).rolling(10).mean()
                ax2.plot(iterations, acceptance_ma, label='Acceptance Rate (MA10)',
                        color='green', linewidth=2, alpha=0.7)
            else:
                ax2.plot(iterations, acceptance_rate, label='Acceptance Rate',
                        color='green', linewidth=2, alpha=0.7)

            ax2.set_ylabel('Acceptance Rate (%)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim([0, 100])

            # 合并图例
            lines1, labels1 = axes[2, 1].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[2, 1].legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved as '{save_path}'")
        plt.show()

    def analyze_performance_degradation(self, train_interval: int = 20):
        """
        分析训练间隔内的性能衰退模式

        Args:
            train_interval: PPO训练间隔
        """
        if not self.training_history['iteration_scores']:
            logger.warning("No iteration data to analyze")
            return

        iter_data = self.training_history['iteration_scores']
        ppo_updates = self.training_history['ppo_update_iterations']

        logger.info("=" * 70)
        logger.info("📊 PERFORMANCE DEGRADATION ANALYSIS")
        logger.info("=" * 70)

        # 1. 整体统计
        train_deltas = [d['train_delta'] for d in iter_data]
        val_deltas = [d['val_delta'] for d in iter_data]

        positive_train = sum(1 for d in train_deltas if d > 0)
        negative_train = sum(1 for d in train_deltas if d < 0)
        positive_val = sum(1 for d in val_deltas if d > 0)
        negative_val = sum(1 for d in val_deltas if d < 0)

        logger.info(f"\n1️⃣ Overall Statistics:")
        logger.info(f"  Train: {positive_train} improvements, {negative_train} degradations")
        logger.info(f"  Val:   {positive_val} improvements, {negative_val} degradations")
        logger.info(f"  Avg Train Δ: {np.mean(train_deltas):+.6f}")
        logger.info(f"  Avg Val Δ:   {np.mean(val_deltas):+.6f}")

        # 2. 按训练间隔分析
        if ppo_updates:
            logger.info(f"\n2️⃣ Performance by Training Interval:")

            intervals = []
            start_idx = 0

            for ppo_iter in ppo_updates + [len(iter_data)]:
                interval_data = [d for d in iter_data
                               if start_idx <= d['iteration'] < ppo_iter]

                if interval_data:
                    interval_train_deltas = [d['train_delta'] for d in interval_data]
                    interval_val_deltas = [d['val_delta'] for d in interval_data]

                    intervals.append({
                        'start': start_idx,
                        'end': ppo_iter - 1,
                        'train_sum': sum(interval_train_deltas),
                        'val_sum': sum(interval_val_deltas),
                        'train_mean': np.mean(interval_train_deltas),
                        'val_mean': np.mean(interval_val_deltas),
                        'additions': sum(d['additions'] for d in interval_data),
                        'rejections': sum(d['rejections'] for d in interval_data),
                    })

                start_idx = ppo_iter

            degraded_intervals = [i for i in intervals if i['val_sum'] < -0.001]
            logger.info(f"  Total intervals: {len(intervals)}")
            logger.info(f"  Degraded intervals: {len(degraded_intervals)} "
                       f"({len(degraded_intervals)/len(intervals)*100:.1f}%)")

            if degraded_intervals:
                worst_intervals = sorted(degraded_intervals, key=lambda x: x['val_sum'])[:5]
                logger.info(f"\n  🔴 Top 5 Worst Intervals:")
                for i, interval in enumerate(worst_intervals, 1):
                    logger.info(f"    {i}. Iter {interval['start']}-{interval['end']}: "
                              f"Val Δ={interval['val_sum']:+.4f}, "
                              f"Train Δ={interval['train_sum']:+.4f}, "
                              f"Added={interval['additions']}, "
                              f"Rejected={interval['rejections']}")

        # 3. 性能衰退的原因分析
        logger.info(f"\n3️⃣ Degradation Cause Analysis:")

        degraded_iters = [d for d in iter_data if d['val_delta'] < 0]
        improved_iters = [d for d in iter_data if d['val_delta'] > 0]

        if degraded_iters and improved_iters:
            avg_reject_degraded = np.mean([d['rejections'] for d in degraded_iters])
            avg_reject_improved = np.mean([d['rejections'] for d in improved_iters])

            logger.info(f"  Avg rejections when degraded: {avg_reject_degraded:.2f}")
            logger.info(f"  Avg rejections when improved: {avg_reject_improved:.2f}")

            if avg_reject_degraded > avg_reject_improved * 1.2:
                logger.info(f"  ⚠️  High rejection rate correlates with degradation!")

        # 4. PPO更新效果分析
        if len(ppo_updates) > 1:
            logger.info(f"\n4️⃣ PPO Update Effectiveness:")

            improvements_after_ppo = []
            for ppo_iter in ppo_updates:
                after_data = [d for d in iter_data
                            if ppo_iter <= d['iteration'] < ppo_iter + 5]
                if after_data:
                    avg_val_change = np.mean([d['val_delta'] for d in after_data])
                    improvements_after_ppo.append(avg_val_change)

            if improvements_after_ppo:
                positive_effect = sum(1 for x in improvements_after_ppo if x > 0)
                logger.info(f"  PPO updates with positive effect: "
                          f"{positive_effect}/{len(improvements_after_ppo)}")
                logger.info(f"  Avg performance change after PPO: "
                          f"{np.mean(improvements_after_ppo):+.6f}")

        logger.info("=" * 70)

    def plot_predictions_and_signals(self, df: pd.DataFrame, save_path: str):
        """绘制预测和信号的可视化图"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

        # 1. 价格走势
        axes[0].plot(df.index, df['close'], label='Close Price',
                    color='blue', linewidth=1.5)
        axes[0].set_title('Close Price', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. LSTM预测 vs 实际收益
        axes[1].plot(df.index, df['prediction'], label='LSTM Prediction',
                    color='orange', alpha=0.7)
        axes[1].plot(df.index, df['target'], label='Actual Return',
                    color='green', alpha=0.5)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.3)
        axes[1].set_title('LSTM Predictions vs Actual Returns',
                         fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. 交易信号
        axes[2].plot(df.index, df['signal'], label='Trading Signal',
                    color='purple', linewidth=1.5)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].fill_between(df.index, 0, df['signal'],
                            where=(df['signal'] > 0), color='green',
                            alpha=0.3, label='Long')
        axes[2].fill_between(df.index, 0, df['signal'],
                            where=(df['signal'] < 0), color='red',
                            alpha=0.3, label='Short')
        axes[2].set_title('Trading Signals', fontweight='bold', fontsize=12)
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Position')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved predictions plot to {save_path}")
        plt.close()
