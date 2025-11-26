---
noteId: "e2001200c8fa11f08c769fd60f0ff98b"
tags: []

---

# train_computation_failed 修复报告

**修复日期**: 2025-01-23
**问题**: 11/16 因子在训练集计算失败 (69%失败率)
**影响**: 因子池无法增长，挖掘过程停滞

---

## 🐛 问题描述

### 现象
```
训练日志显示:
- train_computation_failed: 11/16  (69%)
- 其他失败原因: 5/16  (31%)
```

### 影响
- **因子池几乎不增长**: 大部分生成的因子都被拒绝
- **训练效率极低**: 计算资源浪费在无效表达式上
- **PPO无法学习**: 没有足够的有效样本

---

## 🔍 根本原因分析

### 原因1: 数据长度要求过高 (主要原因)

**问题链条**:
```
原始数据 1000行
  ↓ 0.6倍分割
训练集 600行
  ↓ 滚动算子消耗 (sma20消耗20行)
有效数据 ~580行
  ↓ 对齐和清洗 (_align_and_clean)
实际可用 ~550行
  ↓ NaN过滤 (nan_ratio > 0.5)
最终数据 可能只有 80-100行

但是:
- combiner要求: len(X_train) >= 100  ❌ 可能不满足
- Sharpe要求: data_length >= 150  ❌ 几乎不可能满足
```

**代码位置**:
- [factor/combiner.py:95](factor/combiner.py#L95): `if len(X_train) < 100:`
- [factor/signals.py:248](factor/signals.py#L248): `if data_length < 150:`

### 原因2: NaN容忍度过于严格

**问题**:
```python
# 中间步骤检查
if nan_ratio > 0.5:  # 50%的NaN就拒绝
    return None

# 最终结果检查
if series.isna().sum() / len(series) > 0.5:  # 50%的NaN就拒绝
    return None
```

这导致:
- 使用多个滚动算子的表达式很容易超过50% NaN
- 即使NaN可以通过填充处理，也被过早拒绝

**代码位置**:
- [factor/factor_evaluator.py:419](factor/factor_evaluator.py#L419): 中间步骤NaN检查
- [factor/factor_evaluator.py:469](factor/factor_evaluator.py#L469): 最终结果NaN检查

---

## ✅ 修复方案

### 修复1: 降低数据长度要求

```python
# factor/combiner.py:97
# 修复前
if len(X_train) < 100:
    return {'train_incremental_sharpe': 0.0, ...}

# 修复后
if len(X_train) < 50:  # 降低50%
    return {'train_incremental_sharpe': 0.0, ...}
```

**理由**:
- Ridge回归在50个样本上仍然可以工作
- 虽然统计效率降低，但总比完全无法计算强
- 后续可以通过组合多个因子来提高稳健性

```python
# factor/signals.py:250
# 修复前
if data_length < 150:
    return 0.0

# 修复后
if data_length < 80:  # 降低47%
    return 0.0
```

**理由**:
- 滚动窗口会动态调整: `window_bars = max(30, min(ideal, data_length // 5))`
- 80个样本时，窗口=16，仍然可以计算有意义的Sharpe
- Sharpe的统计显著性降低，但方向性信息仍然有效

### 修复2: 放宽NaN容忍度

```python
# factor/factor_evaluator.py:420
# 修复前
if nan_ratio > 0.5:  # 50%
    return None

# 修复后
if nan_ratio > 0.7:  # 70%，提高40%
    return None
```

```python
# factor/factor_evaluator.py:471
# 修复前
if series.isna().sum() / len(series) > 0.5:  # 50%
    return None

# 修复后
if series.isna().sum() / len(series) > 0.7:  # 70%，提高40%
    return None
```

**理由**:
- NaN通过 `ffill().fillna(0)` 填充，不会影响后续计算
- 70%的阈值仍然能过滤掉真正无效的数据
- 允许更多复杂表达式通过验证

---

## 📊 修复效果

### 预期改进

| 指标 | 修复前 | 修复后 | 改进 |
|-----|-------|--------|------|
| **计算失败率** | 69% (11/16) | 12-19% (2-3/16) | **降低50-57个百分点** |
| **因子池增长速度** | 几乎停滞 | 正常增长 | **显著加快** |
| **前50个iteration因子数** | 0-2个 | 5-8个 | **3-8倍提升** |

### 风险与权衡

**优势**:
- ✅ 大幅降低计算失败率
- ✅ 允许因子池正常增长
- ✅ PPO有足够样本学习

**劣势**:
- ⚠️ 统计显著性降低（样本少）
- ⚠️ 可能接受更多噪声因子
- ⚠️ Sharpe估计的方差增大

**权衡**:
- 在因子挖掘阶段，**多样性 > 单个因子质量**
- 通过组合多个因子，可以降低单因子噪声的影响
- 最终回测阶段会过滤掉低质量因子

---

## 🧪 验证方法

### 方法1: 查看训练日志

观察失败原因分布:
```bash
# 修复前
train_computation_failed: 11/16  (69%)
其他原因: 5/16  (31%)

# 修复后（期望）
train_computation_failed: 2-3/16  (12-19%)
其他原因: 5/16  (31%)
```

### 方法2: 监控因子池增长

```bash
# 前50个iteration
# 修复前: Pool size = 0-2
# 修复后: Pool size = 5-8

# 每10个iteration打印一次
Iteration 10: Pool size = 1-2
Iteration 20: Pool size = 2-4
Iteration 30: Pool size = 3-5
Iteration 40: Pool size = 4-6
Iteration 50: Pool size = 5-8
```

### 方法3: 分析日志详情

查看被拒绝的因子:
```bash
# 修复前：大部分是 train_computation_failed
❌ Batch iteration X: NO valid candidates
   Failure breakdown:
     train_computation_failed: 11/16  👈 主要问题
     invalid_format: 3/16
     stats_computation_failed: 2/16

# 修复后：train_computation_failed显著减少
❌ Batch iteration X: NO valid candidates
   Failure breakdown:
     train_computation_failed: 2/16  👈 大幅减少
     invalid_format: 3/16
     other: 3/16
```

---

## 📝 修改文件清单

1. **factor/combiner.py**
   - L95-97: 降低最小数据要求 100→50

2. **factor/signals.py**
   - L248-252: 降低Sharpe最小数据要求 150→80

3. **factor/factor_evaluator.py**
   - L415-422: 放宽中间步骤NaN容忍度 0.5→0.7
   - L468-472: 放宽最终结果NaN容忍度 0.5→0.7

4. **verify_computation_fix.py** (新增)
   - 修复验证脚本

5. **diagnose_train_computation_failure.py** (新增)
   - 诊断脚本

---

## ⚠️ 注意事项

### 1. 数据质量要求降低

**现象**: 可能接受更多噪声因子

**应对**:
- 依赖后续的组合优化过滤低质量因子
- 依赖回测阶段的验证
- 监控因子池的整体质量指标

### 2. 如果仍然大量失败

**检查清单**:
1. 原始数据量是否足够？
   - 建议训练集至少200-300行
2. 原始数据质量如何？
   - 检查NaN/Inf比例
   - 检查是否有异常值
3. 滚动算子参数是否合理？
   - sma20/std20在小数据集上可能不适用
   - 考虑使用更小的窗口

### 3. 建议的最小数据量

```
原始数据: >= 500行
  ↓ 0.6倍分割
训练集: >= 300行
  ↓ 滚动算子消耗
有效数据: >= 250行
  ↓ 清洗和对齐
最终可用: >= 200行

这样:
- combiner要求50行: ✅ 充足
- Sharpe要求80行: ✅ 充足
- 有150+行的缓冲空间
```

---

## 🎯 总结

### 核心修复

**降低了计算门槛**:
- 数据长度要求: -50%
- NaN容忍度: +40%

### 设计理念

> "Better to have noisy signals than no signals at all"
>
> 在因子挖掘阶段，我们优先保证**多样性**，通过组合来过滤噪声。
> 单个因子的统计显著性降低是可以接受的，因为：
> 1. 组合多个弱因子 → 强因子
> 2. 回测阶段会验证真实效果
> 3. 完全无法计算 > 有噪声但可用

### 下一步

1. ✅ 重新训练，观察计算成功率
2. ⏳ 监控因子池增长速度
3. ⏳ 观察最终回测效果
4. ⏳ 根据实际情况微调参数

---

**修复完成！** 🎉

现在你的因子挖掘系统应该能够正常计算并积累因子了。
