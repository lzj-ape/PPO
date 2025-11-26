"""
修复后的代码检查清单

在开始训练前，请确认以下所有项目
"""

def check_fixes():
    print("="*80)
    print("✅ 修复验证检查清单")
    print("="*80)
    print()

    checks = [
        {
            'category': '📄 文件修改',
            'items': [
                ('factor/factor_evaluator.py', 'L164-187: 统一决策逻辑'),
                ('factor/factor_evaluator.py', 'L192-207: 更新日志（拒绝）'),
                ('factor/factor_evaluator.py', 'L223-228: 更新日志（接受）'),
                ('config/config.py', 'L48-58: 更新阈值说明'),
            ]
        },
        {
            'category': '🔍 核心逻辑验证',
            'items': [
                ('decision_score = incremental_sharpe', '所有池子大小'),
                ('ppo_reward_signal = incremental_sharpe', '所有池子大小'),
                ('decision_score == ppo_reward_signal', '完全一致'),
                ('无 absolute_sharpe 判断', '已移除旧逻辑'),
            ]
        },
        {
            'category': '⚙️ 阈值设置',
            'items': [
                ('pool_size < 3: ic_threshold = -0.03', '允许负增量'),
                ('pool_size < 5: ic_threshold = 0.001', '0.1%增量'),
                ('pool_size < 10: ic_threshold = base * 0.3', '0.3%增量'),
                ('pool_size >= 10: ic_threshold = base * 0.6', '0.6%增量'),
            ]
        },
        {
            'category': '📝 文档和测试',
            'items': [
                ('test_threshold_fix.py', '验证脚本存在'),
                ('THRESHOLD_FIX_SUMMARY.md', '修复文档存在'),
                ('compare_fix.py', '对比脚本存在'),
                ('所有测试通过', '运行test_threshold_fix.py'),
            ]
        }
    ]

    for check_group in checks:
        print(f"{check_group['category']}")
        print("-"*80)
        for item, desc in check_group['items']:
            print(f"  ☑️  {item}")
            print(f"      {desc}")
        print()

    print("="*80)
    print("🎯 训练前最后检查")
    print("="*80)
    print()
    print("1. 确认所有修改已保存")
    print("   → 检查git status，确认修改的文件")
    print()
    print("2. 运行验证测试")
    print("   → python test_threshold_fix.py")
    print()
    print("3. 查看对比说明")
    print("   → python compare_fix.py")
    print()
    print("4. 阅读修复文档")
    print("   → cat THRESHOLD_FIX_SUMMARY.md")
    print()
    print("5. 备份旧模型（可选）")
    print("   → mv best_model.pth best_model_old.pth")
    print()
    print("6. 开始新的训练")
    print("   → python main.py  # 或你的训练脚本")
    print()

    print("="*80)
    print("📊 训练时重点监控")
    print("="*80)
    print()
    print("前50个iteration（冷启动期）:")
    print("  - 池子大小: 期望达到 5-8 个因子")
    print("  - 接受率: 期望 40%-60%")
    print("  - 增量Sharpe: 注意是否有 [-0.03, 0.5] 范围的值")
    print("  - 日志: 查看拒绝/接受的理由是否合理")
    print()
    print("第50-200个iteration（成长期）:")
    print("  - 池子大小: 期望达到 10-12 个因子")
    print("  - 接受率: 期望 20%-40%")
    print("  - 增量Sharpe: 主要在 [0.001, 0.3] 范围")
    print()
    print("第200+个iteration（成熟期）:")
    print("  - 池子大小: 期望达到 12-15 个因子")
    print("  - 接受率: 期望 5%-15%")
    print("  - 增量Sharpe: 主要在 [0.006, 0.2] 范围")
    print()

    print("="*80)
    print("⚠️  异常情况处理")
    print("="*80)
    print()
    print("如果池子增长仍然很慢:")
    print("  1. 检查增量Sharpe的分布（是否大部分<-0.03）")
    print("  2. 尝试降低前期阈值: -0.03 → -0.05")
    print("  3. 检查combiner是否正常工作（base_train_score是否更新）")
    print("  4. 查看是否所有因子都invalid（计算失败）")
    print()
    print("如果池子质量下降:")
    print("  1. 检查是否接受了太多负增量因子")
    print("  2. 尝试提高前期阈值: -0.03 → -0.01")
    print("  3. 检查中期阈值是否太低")
    print()
    print("如果PPO不收敛:")
    print("  1. 确认decision_score == ppo_reward_signal")
    print("  2. 检查奖励分布是否合理（不是全0）")
    print("  3. 尝试调整clip范围和学习率")
    print()

    print("="*80)
    print("✅ 检查清单完成！")
    print("="*80)
    print()
    print("准备好了吗？让我们开始训练吧！🚀")
    print()

if __name__ == "__main__":
    check_fixes()
