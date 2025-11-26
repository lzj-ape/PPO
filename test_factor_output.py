"""
测试新的因子输出功能

模拟一个批次的因子生成和评估
"""

print("="*80)
print("🧪 测试因子输出功能")
print("="*80)
print()

# 模拟输出
iteration = 10
batch_size = 8

print(f"\n{'='*80}")
print(f"📊 Iteration {iteration}: Batch Evaluation ({batch_size} factors)")
print(f"{'='*80}")

# 模拟因子1: 合格
print(f"\n[Factor 1/{batch_size}] ✅ QUALIFIED")
print(f"  Expression: sma5(close)")
print(f"  Reward: 0.045678")
print(f"  Incremental Sharpe: 0.042345")
print(f"  Train Sharpe: 0.8765")
print(f"  Val Sharpe: 0.8234")

# 模拟因子2: 有效但未合格
print(f"\n[Factor 2/{batch_size}] ⚠️  VALID")
print(f"  Expression: add(close, volume)")
print(f"  Reward: 0.012345")
print(f"  Incremental Sharpe: 0.010234")
print(f"  Train Sharpe: 0.4567")
print(f"  Val Sharpe: 0.4321")

# 模拟因子3: 无效
print(f"\n[Factor 3/{batch_size}] ❌ INVALID")
print(f"  Expression: INVALID_EXPRESSION")
print(f"  Reason: train_computation_failed")
print(f"  RPN: <BEG> close sma20 volume std10...")

# 模拟因子4: 有效
print(f"\n[Factor 4/{batch_size}] ⚠️  VALID")
print(f"  Expression: sub(high, low)")
print(f"  Reward: 0.008765")
print(f"  Incremental Sharpe: 0.007654")
print(f"  Train Sharpe: 0.3456")
print(f"  Val Sharpe: 0.3234")

# 模拟因子5: 无效
print(f"\n[Factor 5/{batch_size}] ❌ INVALID")
print(f"  Expression: INVALID_EXPRESSION")
print(f"  Reason: invalid_format")
print(f"  RPN: <BEG> close <SEP> volume...")

# 模拟因子6: 有效
print(f"\n[Factor 6/{batch_size}] ⚠️  VALID")
print(f"  Expression: delta1(close)")
print(f"  Reward: 0.003456")
print(f"  Incremental Sharpe: 0.002345")
print(f"  Train Sharpe: 0.2345")
print(f"  Val Sharpe: 0.2123")

# 模拟因子7: 有效
print(f"\n[Factor 7/{batch_size}] ⚠️  VALID")
print(f"  Expression: mul(close, volume)")
print(f"  Reward: 0.001234")
print(f"  Incremental Sharpe: 0.000987")
print(f"  Train Sharpe: 0.1567")
print(f"  Val Sharpe: 0.1432")

# 模拟因子8: 无效
print(f"\n[Factor 8/{batch_size}] ❌ INVALID")
print(f"  Expression: INVALID_EXPRESSION")
print(f"  Reason: train_computation_failed")
print(f"  RPN: <BEG> close sma20 ema10 std20...")

print(f"\n{'='*80}")

# 批次决策
print(f"\n{'🎯 Batch Decision':^80}")
print(f"{'-'*80}")
print(f"✅ Best Factor in Batch:")
print(f"   Expression: sma5(close)")
print(f"   Reward: 0.045678")
print(f"   Incremental Sharpe: 0.042345")

print(f"\n🎉 COMMITTED TO POOL!")
print(f"   Pool size: 5")
print(f"   Train Score: 1.2345")
print(f"   Val Score: 1.1234")
print(f"   Incremental Contribution: 0.042345")

print(f"\n{'='*80}")

# 统计信息
print("\n📊 批次统计:")
print(f"  合格因子: 1/{batch_size} (12.5%)")
print(f"  有效因子: 4/{batch_size} (50.0%)")
print(f"  无效因子: 3/{batch_size} (37.5%)")
print()
print("  失败原因分布:")
print("    train_computation_failed: 2/3")
print("    invalid_format: 1/3")
print()

print("="*80)
print("✅ 输出功能测试完成！")
print("="*80)
print()
print("📖 查看 FACTOR_OUTPUT_GUIDE.md 了解详细的输出解读方法")
print()
