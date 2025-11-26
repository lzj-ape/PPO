---
noteId: "cf2ffae0c8fb11f08c769fd60f0ff98b"
tags: []

---

# 因子输出日志指南

**新增功能**: 详细输出每个生成因子的表达式和评分

---

## 📋 输出格式说明

### 1. 批次头部

```
================================================================================
📊 Iteration 10: Batch Evaluation (8 factors)
================================================================================
```

- **Iteration**: 当前迭代次数
- **8 factors**: 本批次生成的因子数量（batch_size）

---

### 2. 每个因子的详细信息

#### 2.1 合格因子（QUALIFIED）

```
[Factor 1/8] ✅ QUALIFIED
  Expression: sma5(close, delta1(volume))
  Reward: 0.023456
  Incremental Sharpe: 0.021234
  Train Sharpe: 0.6543
  Val Sharpe: 0.5234
```

**字段说明**:
- `Expression`: 可读的因子表达式（中缀表达式）
- `Reward`: PPO学习的最终奖励（包含惩罚项）
- `Incremental Sharpe`: 该因子对组合的增量贡献
- `Train Sharpe`: 训练集上的Sharpe比率
- `Val Sharpe`: 验证集上的Sharpe比率

**状态**: ✅ QUALIFIED = 达到接受阈值，可能被加入池子

#### 2.2 有效但未合格因子（VALID）

```
[Factor 2/8] ⚠️  VALID
  Expression: add(close, volume)
  Reward: 0.005432
  Incremental Sharpe: 0.004123
  Train Sharpe: 0.3210
  Val Sharpe: 0.2987
```

**状态**: ⚠️ VALID = 计算成功但未达到接受阈值

#### 2.3 无效因子（INVALID）

```
[Factor 3/8] ❌ INVALID
  Expression: INVALID_EXPRESSION
  Reason: train_computation_failed
  RPN: <BEG> close sma20 volume...
```

**字段说明**:
- `Reason`: 失败原因
  - `train_computation_failed`: 训练集计算失败
  - `invalid_format`: 表达式格式错误
  - `stats_computation_failed`: 统计量计算失败
- `RPN`: 原始的RPN格式tokens（前10个）

---

### 3. 批次决策

```
                           🎯 Batch Decision
--------------------------------------------------------------------------------
✅ Best Factor in Batch:
   Expression: sma5(close, delta1(volume))
   Reward: 0.023456
   Incremental Sharpe: 0.021234

🎉 COMMITTED TO POOL!
   Pool size: 5
   Train Score: 1.2345
   Val Score: 1.1234
   Incremental Contribution: 0.021234
```

**或者（未提交）**:

```
                           🎯 Batch Decision
--------------------------------------------------------------------------------
✅ Best Factor in Batch:
   Expression: add(close, volume)
   Reward: 0.005432
   Incremental Sharpe: 0.004123

❌ NOT COMMITTED (Did not meet threshold)
   Current pool size: 3
   Valid candidates: 2/8
```

**字段说明**:
- `Pool size`: 提交后的因子池大小
- `Train/Val Score`: 组合的总体得分
- `Incremental Contribution`: 该因子带来的真实增量

---

### 4. 完全失败的批次

```
                           🎯 Batch Decision
--------------------------------------------------------------------------------
❌ Batch iteration 10: NO valid candidates
   All 8 expressions failed validation
   Failure breakdown:
     train_computation_failed: 5/8
     invalid_format: 2/8
     stats_computation_failed: 1/8
```

---

## 📊 如何解读输出

### 场景1: 健康的训练

```
Iteration 10:
  [Factor 1/8] ✅ QUALIFIED (Reward: 0.05, Incr: 0.04)
  [Factor 2/8] ⚠️  VALID (Reward: 0.01, Incr: 0.008)
  [Factor 3/8] ⚠️  VALID (Reward: 0.003, Incr: 0.002)
  [Factor 4/8] ❌ INVALID (train_computation_failed)
  [Factor 5/8] ⚠️  VALID (Reward: 0.002, Incr: 0.001)
  [Factor 6/8] ❌ INVALID (invalid_format)
  [Factor 7/8] ⚠️  VALID (Reward: -0.001, Incr: -0.002)
  [Factor 8/8] ❌ INVALID (train_computation_failed)

🎉 COMMITTED: Factor 1 → Pool size: 5
```

**特征**:
- ✅ 有合格因子（1/8 = 12.5%）
- ✅ 多个有效因子（4/8 = 50%）
- ✅ 失败率可控（3/8 = 37.5%）
- ✅ 因子池持续增长

---

### 场景2: 需要关注的情况

```
Iteration 50:
  [Factor 1/8] ⚠️  VALID (Reward: 0.002, Incr: 0.001)
  [Factor 2/8] ⚠️  VALID (Reward: 0.001, Incr: 0.0008)
  [Factor 3/8] ⚠️  VALID (Reward: 0.0005, Incr: 0.0003)
  [Factor 4/8] ❌ INVALID (train_computation_failed)
  [Factor 5/8] ⚠️  VALID (Reward: -0.001, Incr: -0.002)
  [Factor 6/8] ❌ INVALID (train_computation_failed)
  [Factor 7/8] ❌ INVALID (train_computation_failed)
  [Factor 8/8] ❌ INVALID (train_computation_failed)

❌ NOT COMMITTED (Did not meet threshold)
   Current pool size: 8
   Valid candidates: 0/8
```

**问题**:
- ⚠️ 没有合格因子（0/8）
- ⚠️ 增量Sharpe都很小（< 0.002）
- ⚠️ 失败率高（4/8 = 50%）
- ⚠️ 池子停止增长

**可能原因**:
1. 池子质量已经很高，新因子难以超越
2. PPO陷入局部最优，生成的因子相似
3. 阈值太高（pool_size=8时，threshold=0.003）

**应对措施**:
1. 降低阈值（考虑修改config）
2. 增加探索（提高entropy_coeff）
3. 检查因子多样性

---

### 场景3: 严重问题

```
Iteration 100:
  [Factor 1/8] ❌ INVALID (train_computation_failed)
  [Factor 2/8] ❌ INVALID (train_computation_failed)
  [Factor 3/8] ❌ INVALID (train_computation_failed)
  [Factor 4/8] ❌ INVALID (train_computation_failed)
  [Factor 5/8] ❌ INVALID (train_computation_failed)
  [Factor 6/8] ❌ INVALID (train_computation_failed)
  [Factor 7/8] ❌ INVALID (train_computation_failed)
  [Factor 8/8] ❌ INVALID (train_computation_failed)

❌ NO valid candidates
   Failure breakdown:
     train_computation_failed: 8/8
```

**严重问题**:
- ❌ 100%失败率
- ❌ 全部是计算失败
- ❌ 无法积累因子

**应对措施**:
1. 检查数据质量（NaN/Inf比例）
2. 检查数据量是否足够
3. 降低数据长度要求（已在修复中完成）
4. 检查operators是否正常工作

---

## 🔍 关键指标监控

### 1. 合格率（Qualified Rate）

```
合格率 = QUALIFIED因子数 / batch_size

期望值:
- 前期（0-50 iter）: 10-30%
- 中期（50-200 iter）: 5-15%
- 后期（200+ iter）: 2-8%
```

### 2. 有效率（Valid Rate）

```
有效率 = VALID因子数 / batch_size

期望值:
- 任何阶段: > 50%
- 如果 < 30%: 需要检查计算失败原因
```

### 3. 增量Sharpe分布

```
期望分布:
- 前期: [-0.03, 0.5]，集中在 [0, 0.1]
- 中期: [0, 0.3]，集中在 [0.001, 0.05]
- 后期: [0, 0.2]，集中在 [0.005, 0.02]
```

### 4. 池子增长速度

```
期望速度:
- 前50 iter: 每5-10个iter增加1个
- 50-200 iter: 每10-20个iter增加1个
- 200+ iter: 每30-50个iter增加1个
```

---

## 💡 使用技巧

### 1. 快速定位问题

```bash
# 搜索所有合格因子
grep "✅ QUALIFIED" training.log

# 搜索所有提交记录
grep "🎉 COMMITTED" training.log

# 统计失败原因
grep "Failure breakdown" training.log
```

### 2. 分析因子质量

```bash
# 提取所有因子表达式和奖励
grep -A 3 "Expression:" training.log | grep -E "(Expression|Reward)"

# 查看增量Sharpe分布
grep "Incremental Sharpe:" training.log | awk '{print $4}' | sort -n
```

### 3. 监控池子状态

```bash
# 查看池子大小变化
grep "Pool size:" training.log | tail -20

# 查看训练得分变化
grep "Train Score:" training.log | tail -20
```

---

## 📝 示例完整日志

```
================================================================================
📊 Iteration 10: Batch Evaluation (8 factors)
================================================================================

[Factor 1/8] ✅ QUALIFIED
  Expression: sma5(close)
  Reward: 0.045678
  Incremental Sharpe: 0.042345
  Train Sharpe: 0.8765
  Val Sharpe: 0.8234

[Factor 2/8] ⚠️  VALID
  Expression: add(close, volume)
  Reward: 0.012345
  Incremental Sharpe: 0.010234
  Train Sharpe: 0.4567
  Val Sharpe: 0.4321

[Factor 3/8] ❌ INVALID
  Expression: INVALID_EXPRESSION
  Reason: train_computation_failed
  RPN: <BEG> close sma20 volume std10...

[Factor 4/8] ⚠️  VALID
  Expression: sub(high, low)
  Reward: 0.008765
  Incremental Sharpe: 0.007654
  Train Sharpe: 0.3456
  Val Sharpe: 0.3234

[Factor 5/8] ❌ INVALID
  Expression: INVALID_EXPRESSION
  Reason: invalid_format
  RPN: <BEG> close <SEP> volume...

[Factor 6/8] ⚠️  VALID
  Expression: delta1(close)
  Reward: 0.003456
  Incremental Sharpe: 0.002345
  Train Sharpe: 0.2345
  Val Sharpe: 0.2123

[Factor 7/8] ⚠️  VALID
  Expression: mul(close, volume)
  Reward: 0.001234
  Incremental Sharpe: 0.000987
  Train Sharpe: 0.1567
  Val Sharpe: 0.1432

[Factor 8/8] ❌ INVALID
  Expression: INVALID_EXPRESSION
  Reason: train_computation_failed
  RPN: <BEG> close sma20 ema10 std20...

================================================================================

                           🎯 Batch Decision
--------------------------------------------------------------------------------
✅ Best Factor in Batch:
   Expression: sma5(close)
   Reward: 0.045678
   Incremental Sharpe: 0.042345

🎉 COMMITTED TO POOL!
   Pool size: 5
   Train Score: 1.2345
   Val Score: 1.1234
   Incremental Contribution: 0.042345
```

---

**现在你可以清楚地看到每个因子的生成、评估和决策过程！** 🎉
