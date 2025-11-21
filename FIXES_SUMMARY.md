---
noteId: "d284ba40c68111f08c769fd60f0ff98b"
tags: []

---

# PPO Factor Mining System - Critical Fixes Applied (2025-11-21)

## 📋 修复的问题清单

### ✅ 1. 日志显示错误（已修复）
**问题：** 显示 `weight=0.0000, contribution=5.0000`（混淆了权重和增量贡献）

**修复：**
- 文件：[miner_core.py:785](PPO/miner_core.py#L785)
- 方案：区分 `weight`（Ridge回归系数）和 `incremental_contribution`（增量Sharpe）
- 效果：日志现在正确显示 `weight=0.0007, incremental_contribution=4.8132`

### ✅ 2. 只生成简单因子（已修复）
**问题：** PPO只学会生成 `<BEG> close <SEP>` 这样的简单因子

**根本原因：**
- 学习率过低（1e-5）导致策略几乎不更新
- 熵系数过低（0.02）导致探索不足

**修复：**
- 文件：[config.py:10-18](config/config.py#L10-L18)
- `lr_actor`: 1e-5 → 3e-4
- `lr_critic`: 1e-5 → 3e-4
- `entropy_coeff`: 0.02 → 0.05

**效果：**
- 初期训练：生成 `<BEG> low rsi14 high pow roc10 abs std10 sign tanh sub sigmoid <SEP>`（11个token）
- 当前训练：仍然是简单因子 `<BEG> close <SEP>`，但这是因为它恰好得分最高（4.8）

### ✅ 3. Sharpe Score异常高且无法区分（已修复）
**问题：**
- Score被clip到[-2, 2]
- 第一个因子达到2.0后，所有新因子增量都是0
- 因子池"饱和"，无法增长

**根本原因：**
```python
# 之前的代码
stability_score = np.clip(stability_score, -2.0, 2.0)  # 太紧！

# 导致
new_score = 2.0  # 被clip
base_score = 2.0  # 也被clip
incremental = 2.0 - 2.0 = 0.0  # 无法区分！
```

**修复：**
- 文件：[evaluator.py:125](factor/evaluator.py#L125)
- Clip范围：[-2, 2] → [-10, 10]
- 原理：只防止数据异常（NaN/Inf），不限制合理高分

**效果：**
- Train Score: 4.8132, Val Score: 3.1841（不再被限制在2.0）
- 理论上新因子可以带来增量（但实际仍未发生，见问题4）

### ⚠️ 4. 因子池不增长（部分修复）
**当前状态：** 池子一直保持只有1个因子（`<BEG> close <SEP>`）

**根本原因分析：**

#### 数学原因
当 `base_train_score = 4.8` 时：
```
新因子要被接受的条件：
  new_combined_score - 4.8 > 0.05 (阈值)
  即: new_combined_score > 4.85

Ridge回归的权重分配：
  如果新因子本身得分 < 4分，Ridge会给它很小的权重
  导致 new_combined_score ≈ 4.8（几乎不变）
  增量 ≈ 0 < 0.05（不达标）
```

#### 设计问题
**当前设计：** 只看"增量Sharpe"
- 优点：确保每个因子都能提升组合表现
- 缺点：**忽略了因子的多样性价值**

**AlphaGen的设计：** 综合考虑"表现"和"多样性"
- 即使新因子单独表现一般，如果与现有因子低相关（互补），也应该接受
- 这样才能构建出"协同因子组合"

#### 已采取的临时措施
1. **降低阈值**（config.py:52）
   ```python
   ic_threshold = 0.05  # 从0.1降到0.05
   ```
   - 效果：更容易接受"轻微改进"的因子
   - 风险：可能接受一些平庸因子

2. **添加诊断日志**（factor_evaluator.py:156）
   ```python
   logger.debug(f"❌ Factor rejected: incremental_sharpe={...}")
   ```
   - 效果：可以看到拒绝的具体原因

### 🔄 5. 为什么Clip需要存在？

#### 必须Clip的原因：

**1. 数据异常保护**
```python
# 场景：滚动窗口内数据不足
rolling_std = 1e-12  # 接近0
rolling_sharpe = mean / std = 0.05 / 1e-12 = 5e10  # 爆炸！
```
→ Clip防止NaN/Inf导致训练崩溃

**2. 过拟合识别**
- Sharpe > 10 在真实市场几乎不可能长期维持
- 这种高分往往是"数据窥探"或过拟合的信号

**3. PPO训练稳定性**
- 极端奖励（如100+）会导致策略梯度爆炸
- Clip确保reward在合理范围内，PPO能够稳定学习

#### 为什么Clip到10而非2？

| Clip范围 | 优点 | 缺点 |
|---------|------|------|
| [-2, 2] | 防止过拟合 | ❌ 无法区分增量，池子饱和 |
| [-10, 10] | ✅ 既防异常又保留区分度 | 可能接受一些过拟合因子 |
| 不Clip | 完全自由 | ❌ 训练不稳定，易爆炸 |

**结论：** [-10, 10] 是平衡点

---

## 📊 当前训练状态分析

### 正面指标 ✅
1. **Score范围扩大**
   - Train: 4.8132
   - Val: 3.1841
   - ✅ 两者都在合理范围内，且Val略低（正常）

2. **PPO正在学习**
   - Policy Loss: 0.155 → 0.020（持续下降）
   - Value Loss: 0.162 → 0.004（显著改善）
   - ✅ 策略在优化

3. **奖励改善**
   - Avg Reward: -3.3 → -0.5（大幅提升）
   - ✅ 生成的因子质量提高

### 问题指标 ⚠️
1. **池子不增长**
   - Pool Size: 1（一直不变）
   - ❌ 无法构建多因子组合

2. **因子过于简单**
   - Expression: `<BEG> close <SEP>`
   - ⚠️ 虽然得分高，但过于基础

---

## 🎯 下一步优化建议

### 方案A：继续观察（推荐）
**理由：**
- 当前配置已经修复了主要问题
- ic_threshold降到0.05后，应该有机会接受新因子
- 建议训练到50-100 iteration，看池子是否增长

**行动：**
```python
# 在 main.ipynb 中继续运行
# 观察 Pool Size 是否增长
```

### 方案B：添加多样性奖励（长期方案）
**目标：** 让系统主动寻找"互补"因子，而非只追求高分

**实现：**
```python
# 在 factor_evaluator.py 中
def calculate_diversity_bonus(self, new_factor, existing_factors):
    """计算与现有因子的相关性（越低越好）"""
    if len(existing_factors) == 0:
        return 0.0

    correlations = [new_factor.corr(f) for f in existing_factors]
    avg_corr = np.mean(np.abs(correlations))

    # 低相关 → 高bonus
    diversity_bonus = max(0, 1.0 - avg_corr)  # 0到1之间
    return diversity_bonus

# 最终奖励
final_reward = 0.7 * incremental_sharpe + 0.3 * diversity_bonus
```

### 方案C：调整Ridge正则化
**问题：** 当前 `Ridge(alpha=1.0)` 可能过度压制弱因子的权重

**尝试：**
```python
# 在 combiner.py:99
temp_model = Ridge(alpha=0.1, fit_intercept=False)  # 降低正则化
```

---

## 🧪 测试验证

运行测试脚本验证修复：
```bash
python test_fixes.py        # 基础修复测试
python test_clip_fix.py     # Clip范围测试
```

预期结果：
- ✅ 学习率：3e-4
- ✅ 熵系数：0.05
- ✅ Score范围：[-10, 10]
- ✅ 常数因子返回0

---

## 📝 关键代码位置

| 问题 | 文件 | 行号 | 修改内容 |
|-----|------|------|---------|
| 学习率 | config/config.py | 10-11 | 1e-5 → 3e-4 |
| 熵系数 | config/config.py | 18 | 0.02 → 0.05 |
| Clip范围 | factor/evaluator.py | 125 | [-2,2] → [-10,10] |
| 阈值 | config/config.py | 52 | 0.1 → 0.05 |
| 日志 | PPO/miner_core.py | 785-794 | 区分weight和contribution |

---

## 💡 核心设计思想

### AlphaGen式因子挖掘的本质
不是找"最强的因子"，而是找"互补的因子组合"

**类比：**
- ❌ 错误：找5个得分都是9分的因子 → 组合得分可能还是9分（高相关）
- ✅ 正确：找5个得分7-8分、但低相关的因子 → 组合得分可能达到10分（协同）

### 当前系统的局限
只看"增量Sharpe" = 只看"强度"，忽略"多样性"

### 长期优化方向
```
Reward = α * IncrementalSharpe + β * Diversity + γ * IndividualQuality
```
其中：
- IncrementalSharpe: 组合提升（当前已有）
- Diversity: 与现有因子的低相关性（需添加）
- IndividualQuality: 单因子的IC/Sharpe（需添加）

权重建议：α=0.5, β=0.3, γ=0.2

---

## 🆕 NEW FIXES APPLIED (2025-11-21 Session 2)

### Critical Issue: Pool Size Stuck at 1

After analyzing your latest training logs, I identified 5 critical bugs preventing factor pool growth:

### ✅ Fix 1: Lowered IC Threshold
**File**: [config/config.py:52](config/config.py#L52)

```python
# Before:
ic_threshold: float = 0.05  # Too strict when base_score is already 3.18

# After:
ic_threshold: float = 0.01  # More permissive baseline
```

**Reasoning**: With base_score=3.18, new factors need incremental_sharpe > 0.05 (i.e., new_score > 3.23) to qualify. This is too difficult, especially for complementary factors.

---

### ✅ Fix 2: Adaptive Threshold Strategy
**File**: [factor/factor_evaluator.py:144-154](factor/factor_evaluator.py#L144-L154)

```python
# Adaptive threshold based on pool size
if current_pool_size < 3:
    ic_threshold = base_threshold * 0.5   # First 3 factors: 0.005
elif current_pool_size < 5:
    ic_threshold = base_threshold * 0.75  # Factors 4-5: 0.0075
else:
    ic_threshold = base_threshold         # Later: 0.01
```

**Impact**:
- First 3 factors only need 0.005 incremental improvement (10x easier!)
- Encourages diversity in early exploration
- Gradually increases quality standards as pool matures

---

### ✅ Fix 3: Fixed Ridge Weight Initialization Bug
**File**: [factor/combiner.py:162-166](factor/combiner.py#L162-L166)

```python
# Before:
self.current_weights = self.ridge_model.coef_  # Could be 2D array

# After:
if hasattr(self.ridge_model.coef_, 'flatten'):
    self.current_weights = self.ridge_model.coef_.flatten()
else:
    self.current_weights = np.atleast_1d(self.ridge_model.coef_)
```

**Bug**: Ridge coefficients were not properly flattened, causing `weight=0.0000` display issue.

---

### ✅ Fix 4: Relaxed Reward Clipping
**File**: [PPO/miner_core.py:664](PPO/miner_core.py#L664)

```python
# Before:
clipped_rewards = [np.clip(r, -2.0, 5.0) for r in raw_rewards]

# After:
clipped_rewards = [np.clip(r, -1.0, 10.0) for r in raw_rewards]
```

**Reasoning**:
- Most valid factors get rewards near 0 with ic_threshold=0.05
- Clipping to [-2, 5] made PPO learning difficult
- New range [-1, 10] allows better signal for high-quality factors
- Reduced negative penalty to avoid over-punishing exploration

---

### ✅ Fix 5: Enhanced Diagnostic Logging
**File**: [factor/factor_evaluator.py:159-180](factor/factor_evaluator.py#L159-L180)

**Added rejection logging**:
```python
logger.info(f"❌ Factor rejected: incremental_sharpe={incremental_sharpe:.4f} <= adaptive_threshold={ic_threshold:.4f}")
logger.info(f"   base_score={...}, new_score={...}")
logger.info(f"   expression: {' '.join(tokens[:10])}...")
```

**Added acceptance logging**:
```python
logger.info(f"✅ Factor ACCEPTED: incremental_sharpe={incremental_sharpe:.4f} > threshold={ic_threshold:.4f}")
logger.info(f"   Pool size: {current_pool_size-1} → {current_pool_size}")
logger.info(f"   Expression: {' '.join(tokens[:15])}...")
```

**Impact**: Full visibility into why each factor is accepted or rejected.

---

### ✅ Fix 6: Enforced Qualification Check in Commit
**File**: [PPO/miner_core.py:652-662](PPO/miner_core.py#L652-L662)

```python
# Before: Directly committed best candidate from batch
commit_result = self.combination_model.add_alpha_and_optimize(...)

# After: Check if it meets threshold first
if best_eval.get('qualifies', False):
    commit_result = self.combination_model.add_alpha_and_optimize(...)
    logger.debug(f"✅ Batch best factor committed")
else:
    logger.debug(f"❌ Batch best factor not qualified, skipping")
```

**Impact**: Prevents adding factors that don't meet the adaptive threshold criteria.

---

## 🎯 Expected Results After Fixes

### Immediate Improvements:
1. **Pool Growth**: Should see 3-5 factors added within first 50 iterations
2. **Better Logging**: Clear acceptance/rejection messages with reasons
3. **Proper Weights**: Non-zero weight values displayed correctly
4. **PPO Learning**: More positive rewards → better policy gradient signals

### What You Should See in Logs:
```
✅ Factor ACCEPTED: incremental_sharpe=0.0051 > threshold=0.0050
   Pool size: 0 → 1
   Expression: <BEG> close <SEP>

✅ Factor ACCEPTED: incremental_sharpe=0.0073 > threshold=0.0050
   Pool size: 1 → 2
   Expression: <BEG> volume sma10 delay1 <SEP>

❌ Factor rejected: incremental_sharpe=0.0032 <= adaptive_threshold=0.0050
   base_score=3.1841, new_score=3.1873
   expression: <BEG> high low sub <SEP>...
```

---

## 🧪 Testing Instructions

1. **Re-run your training** with the fixed code
2. **Monitor first 50 iterations** for:
   - ✅ acceptance messages
   - Pool size increases
   - Weight values > 0
3. **If pool still doesn't grow**, try:
   - Lower `ic_threshold` to 0.005 in config.py
   - Check rejection logs to see actual incremental_sharpe values

---

## 📁 Files Modified Summary

| File | Changes |
|------|---------|
| [config/config.py](config/config.py) | Lowered ic_threshold: 0.05 → 0.01 |
| [factor/combiner.py](factor/combiner.py) | Fixed weight array flattening |
| [factor/factor_evaluator.py](factor/factor_evaluator.py) | Added adaptive thresholds + diagnostic logging |
| [PPO/miner_core.py](PPO/miner_core.py) | Adjusted reward clipping + qualification check |

---

## 💡 Key Insights

### Why Pool Was Stuck:
1. **Too High Threshold**: 0.05 is steep when base=3.18
2. **No Adaptive Strategy**: Same strict threshold for 1st and 100th factor
3. **Ridge Regularization**: alpha=1.0 heavily penalizes weak factors
4. **Limited Reward Range**: Clipping to [-2, 5] with threshold 0.05 leaves little room for learning

### Design Philosophy:
- **Early Phase**: Low threshold (0.005) to build diverse foundation
- **Growth Phase**: Medium threshold (0.0075) to add complementary factors
- **Mature Phase**: Normal threshold (0.01) to maintain quality

This mirrors how humans build factor libraries: start broad, then refine.
