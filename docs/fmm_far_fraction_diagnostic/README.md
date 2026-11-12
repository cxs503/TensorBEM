# m16156 · `FMMOperator` 远近场诊断仪修复：未 `setup` 的算子不再静默给零，`far_fraction` 口径写明

**仓库** TensorBEM · **分支** `agent/m16156` · **模块** `src/tensorbem/fmm.py`
**工件** `docs/fmm_far_fraction_diagnostic/{evidence.json,tables.md,fig_far_fraction.png}`（由
`scripts/fmm_far_fraction_diagnostic.py` 生成，正文数字全部机读注入）
**测试** `tests/test_fmm_operator_setup_guard.py`（11 例，修复前 7 红）

---

## 1 案例介绍（物理背景 / 几何 / 参数）

被检对象是 BEM 单层算子 `S q = Σ_j q_j/(4π|x_i−y_j|)` 的快速多极加速器
`tensorbem.fmm.FMMOperator`：八叉树 + 实球谐多极展开，近场直接求和、远场多极求值。
m15669（八叉树几何缺陷）与 m16000（电荷缓存缺陷）都靠一个诊断量
`FMMOperator.far_fraction()` 判断"远场路径到底有没有被走到"，m14931 的代价审计也把
它当作 routing 口径发布。因此这个量一旦说谎，前两张单的因果结论一起失去依据。

m16000 遗留的公开矛盾（其报告"局限（如实）"第 2 条）：

> `far_fraction()` 在同几何返回 `0.0`（暗示无远场对），却与"修复前 2.46 的远场缓存误差"矛盾
> ⇒ 疑似独立诊断缺陷，本单未修。

同一几何（m16000 原样）：200 个源点 `N(0, 0.5²)` 聚簇 + 3 个远靶
`(4,0,0)/(8,0,0)/(0,6,0)`，`p=4..8`、`max_leaf=25`，纯 Python/NumPy 算子。

## 2 怎么做的（方法 / 关键设置）

1. **复现**：按 m16000 的几何把 `far_fraction` 与 `matvec` 都跑一遍（表 1、表 2）。
2. **定位**：`FMMOperator` 是 dataclass，`tree/leaves/_leaf_centers` 由 `setup()` 填充。
   证据脚本 `scripts/fmm_linearity_bench.py` 里写的是
   `FMMOperator(p=4, max_leaf=25).far_fraction(targets)` —— **从未调用 `setup`**。
   未 setup 时 `self.leaves == []`，于是 `far_fraction` 走
   `if len(targets) == 0 or not self.leaves: return 0.0`，`matvec` 走
   `if n == 0 or not self.leaves: return result`（全零）。**"无远场对"是 API 静默的产物，
   不是 routing 的缺陷。**
3. **修复（最小改动，`src/tensorbem/fmm.py`）**
   - 新增 `_require_setup(caller)`：`tree is None` / 无叶 / 无叶心 → `RuntimeError`，消息指明先调
     `setup(points[, charges])`；`matvec`、`far_fraction`、新增的 `far_work_fraction` 开头都过这道门。
   - `setup()` 对空点集显式 `ValueError`（原先抛 numpy 的
     `zero-size array to reduction operation minimum`）。
   - `far_fraction` 文档写明口径：**配对**计数（每个 (靶, 非空源叶) 对一票），并指向新增的
     `far_work_fraction`（按叶内源数加权的**代价**口径）。两个量在同一几何可以差一倍以上（表 2）。
   - 证据脚本 `scripts/fmm_linearity_bench.py` 的 far_fraction 调用改为先 `setup`，
     同时输出 `far_work_fraction`。
4. **不变量**（新增测试）：`far_fraction(targets) == 0` ⇒ `matvec` 就是稠密直接求和
   （表 3，相对误差 2.3e-16，仅求和顺序差）；反之远场份额 > 0 时清零多极系数必须改变答案。

## 3 计算结果（机读注入，见 `tables.md`）

- 未 setup：修复前 `far_fraction = 0.0`、`matvec` 全零；修复后两者均 `RuntimeError`（表 1）。
- setup 之后同一几何：`far_fraction = 0.173913`（恰 4/23 对），
  `far_work_fraction = 0.093333`，`matvec` 相对稠密误差 1.3e-6（表 2）。
  ⇒ m16000 的 2.46 误差**确有远场路径参与**，与其结论一致；矛盾纯属调用缺 `setup`。
- 多极份额梯子（icosphere, p=4, max_leaf=32）：level 1（80 面元/8 叶）为 0.0，level 2/3 约
  0.74，level 4 约 0.936（表 4）。其中 level 3 的 0.7400669642857143 与已发布的
  `docs/fmm_cost_audit/audit.json` 同项**逐位相同**，交叉证明历史审计数字可复现。
- 粗网格（80 面元）远场份额为 0 是一个必须知道的边界：网格太小时叶近邻图覆盖全部叶对，
  算子退化为"稠密求和 + 树记账"，此时任何"多极加速已被使用"的说法都是错的。

## 4 结果图

`fig_far_fraction.png`（左：未 setup 的 0.0 vs setup 后的 0.174，虚线为精确的 4/23；
右：配对口径与源数加权口径的远场份额随 N 变化）。图中数字全部读自 `evidence.json`。

## 5 注意点（坑 / 局限 / 适用范围）

- **坑**：`FMMOperator` 的 `setup` 依赖写在 dataclass 字段上，不 setup 也能构造成功；修复前
  任何"试一下"的探针都会拿到一个合法的 0.0/全零。同类 fail-silent 路径值得在其他 dataclass
  算子（`FMMOperator` 之外的 `_helmholtz_*`）继续普查——本单未做。
- **口径**：`far_fraction` 是配对口径，不是工作量口径；代价结论请用 `far_work_fraction`。
  历史审计（m14931/m15669）发布的 `near_pairs/far_pairs` 是配对计数，其结论按对计算，
  未受影响。
- 修复只影响"未 setup"与"空点集"两类调用；已 setup 的算子数值逐位不变
  （`tests/test_fmm.py` 23 例、`test_fmm_operator_linearity.py` 10 例、
  `test_fmm_cost_audit.py` 全部绿）。

## 6 误差判定

| 量 | 基准 | 实测 | 误差 | <3%? |
|---|---|---|---|---|
| `matvec`（m16000 几何, p=4） | 稠密直接求和 `direct_laplace_matvec` | 表 2 | 1.333e-06 | ✅ 是 |
| `far_fraction`（m16000 几何） | 手算配对分数 4/23 = 0.173913 | 0.173913043478 | 0（逐位） | ✅ 是 |
| `far_fraction`（icosphere level 3, p=4, max_leaf=32） | 历史审计 `fmm_cost_audit/audit.json` 0.7400669642857143 | 0.7400669642857143 | 0（逐位） | ✅ 是 |
| `matvec`（far_fraction = 0 的单叶几何） | 稠密直接求和 | 表 3 | 2.307e-16 | ✅ 是 |
| 未 setup 调用的错误可见性 | 期望"拒绝作答" | `RuntimeError` | 定性 | ✅ 是（修复前为静默 0） |

无"honest-FAIL"项残留；m16000 遗留矛盾（局限第 2 条）在本单**结案**。