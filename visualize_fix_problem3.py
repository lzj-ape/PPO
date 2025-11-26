"""
可视化修复前后的差异
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
def visualize_fix():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('问题3修复前后对比：PPO奖励信号分离', fontsize=16, fontweight='bold')

    # 场景：池子有2个因子，评估第3个候选因子
    # 绝对Sharpe=0.6, 增量Sharpe=-0.2（负贡献）

    # ============ 修复前 ============
    # 左上：修复前的逻辑流程
    ax1 = axes[0, 0]
    ax1.set_title('修复前：覆盖incremental_sharpe', fontsize=12, fontweight='bold', color='red')
    ax1.axis('off')

    flow_before = [
        "1️⃣ 计算指标:",
        "   - 绝对Sharpe = 0.6",
        "   - 增量Sharpe = -0.2 (负贡献)",
        "",
        "2️⃣ 判断逻辑 (pool_size=2 < 3):",
        "   - decision_score = 0.6 (绝对)",
        "   - incremental_sharpe = 0.6 ❌ 覆盖!",
        "",
        "3️⃣ 计算奖励:",
        "   - final_reward = 0.6 + 惩罚",
        "",
        "4️⃣ 结果:",
        "   - ✅ 接受因子 (0.6 > 0)",
        "   - ✅ PPO学到正奖励 +0.6",
        "   - ❌ 实际增量贡献 -0.2",
        "   - ❌ PPO学习信号错误!"
    ]

    y_pos = 0.95
    for line in flow_before:
        color = 'red' if '❌' in line else 'green' if '✅' in line else 'black'
        fontweight = 'bold' if '❌' in line or '✅' in line else 'normal'
        ax1.text(0.05, y_pos, line, fontsize=10, family='monospace',
                color=color, fontweight=fontweight, verticalalignment='top')
        y_pos -= 0.05

    # 左下：修复前的训练效果
    ax3 = axes[1, 0]
    ax3.set_title('修复前：PPO学习曲线', fontsize=12, color='red')

    # 模拟错误的学习曲线
    iterations = np.arange(0, 100)
    # PPO误以为生成绝对Sharpe高的因子好，导致振荡
    wrong_rewards = 0.6 * np.ones_like(iterations) + np.random.normal(0, 0.2, len(iterations))
    true_increments = -0.2 * np.ones_like(iterations) + np.random.normal(0, 0.1, len(iterations))

    ax3.plot(iterations, wrong_rewards, label='PPO学到的奖励 (绝对Sharpe)', color='orange', linewidth=2)
    ax3.plot(iterations, true_increments, label='真实增量贡献', color='red', linewidth=2, linestyle='--')
    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax3.set_xlabel('迭代次数', fontsize=10)
    ax3.set_ylabel('奖励值', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1, 1.5)

    # 添加标注
    ax3.annotate('PPO误以为这是好因子',
                xy=(50, 0.6), xytext=(60, 1.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, color='red', fontweight='bold')

    ax3.annotate('但实际对组合是负贡献',
                xy=(50, -0.2), xytext=(60, -0.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, color='red', fontweight='bold')

    # ============ 修复后 ============
    # 右上：修复后的逻辑流程
    ax2 = axes[0, 1]
    ax2.set_title('修复后：分离decision_score和ppo_reward_signal', fontsize=12, fontweight='bold', color='green')
    ax2.axis('off')

    flow_after = [
        "1️⃣ 计算指标:",
        "   - 绝对Sharpe = 0.6",
        "   - 增量Sharpe = -0.2 (负贡献)",
        "",
        "2️⃣ 判断逻辑 (pool_size=2 < 3):",
        "   - decision_score = 0.6 (绝对)",
        "   - ppo_reward_signal = -0.2 ✅ 保持真实值",
        "",
        "3️⃣ 计算奖励:",
        "   - final_reward = -0.2 + 惩罚",
        "",
        "4️⃣ 结果:",
        "   - ✅ 接受因子 (0.6 > 0)",
        "   - ✅ PPO学到负奖励 -0.2",
        "   - ✅ 实际增量贡献 -0.2",
        "   - ✅ PPO学习信号正确!"
    ]

    y_pos = 0.95
    for line in flow_after:
        color = 'red' if '❌' in line else 'green' if '✅' in line else 'black'
        fontweight = 'bold' if '❌' in line or '✅' in line else 'normal'
        ax2.text(0.05, y_pos, line, fontsize=10, family='monospace',
                color=color, fontweight=fontweight, verticalalignment='top')
        y_pos -= 0.05

    # 右下：修复后的训练效果
    ax4 = axes[1, 1]
    ax4.set_title('修复后：PPO学习曲线', fontsize=12, color='green')

    # 模拟正确的学习曲线
    correct_rewards = -0.2 * np.ones_like(iterations) + np.random.normal(0, 0.1, len(iterations))
    true_increments2 = -0.2 * np.ones_like(iterations) + np.random.normal(0, 0.1, len(iterations))

    ax4.plot(iterations, correct_rewards, label='PPO学到的奖励 (增量Sharpe)', color='blue', linewidth=2)
    ax4.plot(iterations, true_increments2, label='真实增量贡献', color='green', linewidth=2, linestyle='--')
    ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax4.set_xlabel('迭代次数', fontsize=10)
    ax4.set_ylabel('奖励值', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1, 1.5)

    # 添加标注
    ax4.annotate('PPO学到负奖励，\n逐渐避免生成此类因子',
                xy=(50, -0.2), xytext=(60, -0.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/Users/duanjin/Desktop/强化学习/PPO/factor/fix_problem3_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ 可视化图表已保存到: factor/fix_problem3_visualization.png")
    plt.show()


if __name__ == '__main__':
    visualize_fix()
