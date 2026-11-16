# CASE-BEM-CUMMINS-WIGLEY — C9 闸门尾部指数稳健性守卫（m22756 建议落实）

**Mission**: m22881 ｜ **Anchor**: `CASE-BEM-CUMMINS-WIGLEY`（feature_ledger: open）｜ **仓**: TensorBEM
**对象**: `truncation_decidability_gate`（C9）的尾部指数估计器稳健性修复

## 1 本单攻的是哪条验证欠账

m22756（`dev_report_m22756_cummins_tail_exponent_falsified.md`）否证了「扩展带必然使水平模态可判」的计划断言，并在 §9 给出直接修复建议：

> 「C9 估计器要加稳健性守卫：建议把 `truncation_decidability_gate` 的判决改为
>  `window ∈ {3,4,5}` 全一致且预算有界」才算可判，并把窗口分歧写进 `values`」

本单落实这条建议：在 `tensorbem.causality` 中新增 `robust_tail_exponent` 函数，
并在 C9 闸门中用它替代原来的单窗 `local_tail_exponent`。

## 2 怎么做的

### 2.1 新增 `robust_tail_exponent`（`src/tensorbem/causality.py`）

- 对一组窗口（默认 `{3, 4, 5}`）分别调用 `local_tail_exponent`
- 收集所有成功拟合的窗口结果
- 计算中位数指数/系数、窗口间散布（spread = max - min）
- **共识判定**：所有窗口对可判性（p > 2.05 vs p ≤ 2.05）达成一致 **且** spread ≤ 1.0
- 返回 `RobustTailResult` 数据类，含 per_window 明细

### 2.2 更新 C9 闸门（`src/tensorbem/benchmarks/case_bem_cummins_wigley.py`）

- `truncation_decidability_gate` 现在同时调用 `robust_tail_exponent` 和 `local_tail_exponent`
- `new_admissible` 判定改为：`budget.admissible AND robust.consensus`
- 每 DOF 行新增字段：`exponent_robust_median`, `exponent_robust_consensus`,
  `exponent_robust_spread`, `exponent_per_window`
- 保持向后兼容：`exponent_at_20` 仍为单窗值

### 2.3 测试（`tests/test_causality.py`）

新增 5 条测试：
- `test_robust_tail_exponent_clean_decay`：清晰幂律衰减 → consensus=True
- `test_robust_tail_exponent_noisy_no_consensus`：噪声数据 → consensus=False
- `test_robust_tail_exponent_below_bar_consensus`：全部低于阈值 → consensus=True（一致同意不可判）
- `test_robust_tail_exponent_custom_windows`：自定义窗口集
- `test_robust_tail_exponent_non_positive_damping`：非正值阻尼 → 窗口被排除

## 3 计算结果（程序化输出，无手抄）

### 3.1 现有 medium 网格（0.2–12 rad/s, 480 面板）的稳健尾部指数

| DOF | p(w=3) | p(w=4) | p(w=5) | 中位数 | spread | consensus | 可判(p>2.05)? |
|-----|--------|--------|--------|--------|--------|-----------|---------------|
| surge | 1.095 | 0.978 | 0.870 | 0.978 | 0.225 | True | **否** |
| sway | 1.529 | 1.397 | 1.229 | 1.397 | 0.301 | True | **否** |
| heave | 2.870 | 2.607 | 2.382 | 2.607 | 0.489 | True | **是** |
| roll | 3.197 | 2.893 | 2.533 | 2.893 | 0.663 | True | **是** |
| pitch | 3.070 | 2.829 | 2.603 | 2.829 | 0.467 | True | **是** |
| yaw | 1.616 | 1.479 | 1.305 | 1.479 | 0.311 | True | **否** |

**关键发现**：在 medium 网格上，所有 6 个自由度的窗口间共识均为 True（spread 均 < 1.0），
但 surge/sway/yaw 的中位数指数仍远低于 2.05 阈值，因此在当前频带上**不可判**。
这与 m22756 的结论一致：水平模态的阻尼尾部尚未进入代数衰减。

### 3.2 与 m22756 探针网格的对比

m22756 在扩展带探针网格（coarse 192 / medium 480, 12–20 rad/s）上发现：
- 同一 DOF 在不同网格上的 p(20) 取值域跨 O(1)–O(10)
- 判决随网格翻转（如 pitch: coarse adm=False → medium adm=True）

本单的稳健估计器解决的是**同一网格内**的窗口敏感性问题。
跨网格的敏感性仍需认证网格（3840 面板）的扩展带数据来解决。

## 4 结果图

### 4.1 各 DOF 阻尼尾部与多窗拟合

![Robust tail exponent per DOF](outputs/cummins_robust_tail_exponent.png)

### 4.2 稳健尾部指数汇总

![Robust tail summary](outputs/cummins_robust_tail_summary.png)

## 5 注意点

1. **共识 ≠ 可判**：consensus=True 只表示窗口间一致，不代表 p > 2.05。
   surge/sway/yaw 的 consensus=True 但 p < 2.05，因此仍不可判。
2. **spread 阈值 1.0**：这是经验值。若后续发现 1.0 太松或太紧，可调 `max_spread` 参数。
3. **向后兼容**：C9 闸门仍报告单窗 `exponent_at_20`，新增 `exponent_robust_*` 字段。
   旧代码读取 `exponent_at_20` 不受影响。
4. **扩展带数据缺失**：本单不产生新的 BEM 求解数据。C8/C9 的完整裁决仍需
   `sweep_extended.npz`（集群作业 cj4321 仍在运行）。

## 6 误差判定

本单没有外部基准可比（Wigley 船辐射阻尼无实验/解析表），因此**不能**回答
「关键量 vs 基准是否 <3%」。

可给的内部量是：
- 清晰幂律衰减（p=3.5）的稳健估计误差 < 0.01（测试验证）
- 现有 medium 网格上，所有 DOF 的窗口间 spread < 0.7，共识稳定
- 与 m22756 的否证结论一致：水平模态在 12 rad/s 处 p < 2.05，不可判

## 7 复现

```bash
cd /root/tp-agent/work/m22881

# 运行新增测试
PYTHONPATH=src python3 -m pytest tests/test_causality.py -v -k "robust" --timeout=60

# 验证 C9 闸门
PYTHONPATH=src python3 -c "
from tensorbem.benchmarks.case_bem_cummins_wigley import load_sweep, truncation_decidability_gate
sweep = load_sweep('benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz')
result = truncation_decidability_gate(sweep, sweep)
print('C9 passed:', result['passed'])
for row in result['values']['per_dof']:
    print(f\"  {row['dof_name']}: robust_p={row['exponent_robust_median']:.3f}, consensus={row['exponent_robust_consensus']}\")
"

# 生成图
PYTHONPATH=src python3 scripts/cummins_robust_tail_figures.py  # 见 outputs/
```

## 8 改动文件

| 文件 | 改动 |
|------|------|
| `src/tensorbem/causality.py` | 新增 `RobustTailResult` 数据类 + `robust_tail_exponent` 函数 + `__all__` 导出 |
| `src/tensorbem/benchmarks/case_bem_cummins_wigley.py` | `truncation_decidability_gate` 改用稳健估计器，新增 per_dof 字段 |
| `tests/test_causality.py` | 新增 5 条 robust_tail_exponent 测试 |
| `outputs/cummins_robust_tail_exponent.png` | 各 DOF 阻尼尾部与多窗拟合图 |
| `outputs/cummins_robust_tail_summary.png` | 稳健尾部指数汇总柱状图 |
| `dev_report_m22881_cummins_robust_tail.md` | 本单产出 |

## 9 给下次自找活的线索

1. **等待 cj4321 落盘**：扩展带 55 频点 × 3840 面板扫频完成后，运行
   `assemble_cummins_extended.py` → 看 C8/C9/C3a_extended 闸门。
   届时稳健估计器将自动应用于扩展带数据。
2. **C9 闸门语义细化**：当前 C9 的 passed 条件是「no DOF lost admissibility」。
   可考虑增加「至少 N 个 DOF 新变为可判」才算真正 widening。
3. **外部锚点 L5**：Wigley 船辐射系数实验/解析参考表仍为 WITHHELD。
   若能找到文献中的辐射阻尼表，可新增 C12 外部锚点闸门。
