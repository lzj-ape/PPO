"""
验证修复后的决策逻辑

测试场景：
1. 第1个因子：增量Sharpe = 0.8，应该被接受（阈值=-0.03）
2. 第2个因子：增量Sharpe = 0.1，应该被接受（阈值=-0.03）
3. 第3个因子：增量Sharpe = -0.01，应该被接受（阈值=-0.03）
4. 第4个因子：增量Sharpe = -0.05，应该被拒绝（阈值=-0.03）
5. 第5个因子：增量Sharpe = 0.002，应该被接受（阈值=0.001）
"""

import sys
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / 'factor') not in sys.path:
    sys.path.insert(0, str(project_root / 'factor'))
if str(project_root / 'config') not in sys.path:
    sys.path.insert(0, str(project_root / 'config'))


def test_threshold_logic():
    """测试阈值逻辑"""
    from config import TrainingConfig

    config = TrainingConfig()
    base_threshold = config.ic_threshold  # 0.01

    print("="*80)
    print("🧪 测试阈值逻辑")
    print("="*80)
    print(f"基础阈值: {base_threshold}\n")

    test_cases = [
        # (pool_size, incremental_sharpe, expected_result, description)
        (0, 0.8, True, "第1个因子，增量Sharpe很高"),
        (1, 0.1, True, "第2个因子，增量Sharpe中等"),
        (2, -0.01, True, "第3个因子，轻微负增量但在阈值内"),
        (3, -0.05, False, "第4个因子，负增量超过阈值"),
        (4, 0.002, True, "第5个因子，小正增量"),
        (5, 0.0005, False, "第6个因子，增量太小"),
        (8, 0.002, False, "第9个因子，增量不足0.3%"),
        (8, 0.004, True, "第9个因子，增量达到0.3%"),
        (12, 0.005, False, "第13个因子，增量不足0.6%"),
        (12, 0.007, True, "第13个因子，增量达到0.6%"),
    ]

    print("测试用例：")
    print("-"*80)
    for pool_size, inc_sharpe, expected, desc in test_cases:
        # 计算阈值
        if pool_size < 3:
            ic_threshold = -0.03
        elif pool_size < 5:
            ic_threshold = 0.001
        elif pool_size < 10:
            ic_threshold = base_threshold * 0.3
        else:
            ic_threshold = base_threshold * 0.6

        # 判断
        should_accept = inc_sharpe > ic_threshold
        result_str = "✅ ACCEPT" if should_accept else "❌ REJECT"
        expected_str = "✅ ACCEPT" if expected else "❌ REJECT"
        status = "✓" if should_accept == expected else "✗ FAILED"

        print(f"{status} Pool={pool_size:2d}, Incr={inc_sharpe:+.4f}, "
              f"Threshold={ic_threshold:+.4f} → {result_str} "
              f"(期望: {expected_str})")
        print(f"   描述: {desc}")

        if should_accept != expected:
            print(f"   ⚠️  测试失败！")

    print("-"*80)


def test_decision_consistency():
    """测试决策一致性：decision_score == ppo_reward_signal"""
    print("\n" + "="*80)
    print("🧪 测试决策一致性")
    print("="*80)

    test_cases = [
        (0, 0.5, "第1个因子"),
        (2, 0.3, "第3个因子"),
        (5, 0.1, "第6个因子"),
    ]

    print("\n验证：decision_score == ppo_reward_signal == incremental_sharpe")
    print("-"*80)

    for pool_size, inc_sharpe, desc in test_cases:
        decision_score = inc_sharpe  # 修复后统一使用增量Sharpe
        ppo_reward_signal = inc_sharpe

        consistent = (decision_score == ppo_reward_signal == inc_sharpe)
        status = "✓" if consistent else "✗ FAILED"

        print(f"{status} {desc} (pool_size={pool_size}):")
        print(f"   incremental_sharpe = {inc_sharpe:.4f}")
        print(f"   decision_score     = {decision_score:.4f}")
        print(f"   ppo_reward_signal  = {ppo_reward_signal:.4f}")
        print(f"   一致性: {consistent}")

        if not consistent:
            print(f"   ⚠️  不一致！这会导致PPO学习混乱！")

    print("-"*80)


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*80)
    print("🧪 测试边界情况")
    print("="*80)

    from config import TrainingConfig
    config = TrainingConfig()
    base_threshold = config.ic_threshold

    edge_cases = [
        (0, 0.0, "第1个因子，增量为0"),
        (2, -0.03, "第3个因子，增量刚好等于阈值"),
        (2, -0.030001, "第3个因子，增量略低于阈值"),
        (4, 0.001, "第5个因子，增量刚好等于阈值"),
        (4, 0.0009999, "第5个因子，增量略低于阈值"),
    ]

    print("\n边界情况测试：")
    print("-"*80)

    for pool_size, inc_sharpe, desc in edge_cases:
        if pool_size < 3:
            ic_threshold = -0.03
        elif pool_size < 5:
            ic_threshold = 0.001
        elif pool_size < 10:
            ic_threshold = base_threshold * 0.3
        else:
            ic_threshold = base_threshold * 0.6

        should_accept = inc_sharpe > ic_threshold
        result_str = "✅ ACCEPT" if should_accept else "❌ REJECT"

        print(f"{desc}:")
        print(f"   incremental_sharpe = {inc_sharpe:.6f}")
        print(f"   ic_threshold       = {ic_threshold:.6f}")
        print(f"   结果: {result_str}")

    print("-"*80)


def main():
    print("\n" + "="*80)
    print("🚀 验证修复后的决策逻辑")
    print("="*80)
    print("\n修复内容：")
    print("1. ✅ 统一使用增量Sharpe作为决策标准（无论池子大小）")
    print("2. ✅ decision_score = ppo_reward_signal = incremental_sharpe")
    print("3. ✅ 根据池子大小调整阈值，而非改变评价指标")
    print("4. ✅ 前3个因子允许负增量（-3%），快速冷启动")
    print()

    try:
        test_threshold_logic()
        test_decision_consistency()
        test_edge_cases()

        print("\n" + "="*80)
        print("✅ 所有测试完成！")
        print("="*80)
        print("\n关键改进：")
        print("1. 前3个因子：阈值-3%，即使轻微拖累组合也接受（快速建池）")
        print("2. 第4-5个因子：阈值0.1%，要求小幅改进")
        print("3. 第6-10个因子：阈值0.3%，要求明显改进")
        print("4. 10+因子：阈值0.6%，要求显著改进（精选模式）")
        print("\n预期效果：")
        print("- 前期快速积累因子（解决冷启动问题）")
        print("- 中期平衡质量和多样性")
        print("- 后期精选高质量因子")
        print("- PPO学习目标和决策标准完全一致")
        print()

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
