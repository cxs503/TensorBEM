# m16156 · TensorBEM FMM 远近场诊断仪：未 `setup` 静默给零 → 响亮报错（m16000 遗留矛盾结案）

**仓库** TensorBEM · **分支** `agent/m16156` · **基线** `7f8a746`（main @ 2026-11-12）

## 缺陷（m16000 报告"局限（如实）"第 2 条，本单结案）

`FMMOperator` 是 dataclass，八叉树由 `setup()` 填充。未 setup 时 `self.leaves == []`：修复前

- `matvec(targets, sources, charges)` → **全零静默返回**（无异常）；
- `far_fraction(targets)` → **0.0**（读起来像"全部近场"）。

`scripts/fmm_linearity_bench.py` 恰好这么调（`FMMOperator(p=4, max_leaf=25).far_fraction(targets)`，
漏了 `setup`），于是 m16000 把 0.0 当成"无远场对"写进局限，与其自身 2.46 的远场缓存误差自相矛盾。
同一几何 `setup` 后 `far_fraction = 0.173913`（= 4/23 对），`far_work_fraction = 0.093333`，
`matvec` 相对稠密误差 1.333e-06 ⇒ 远场路径确实参与，矛盾消失。

## 修复（`src/tensorbem/fmm.py`，最小改动）

1. 新增 `_require_setup(caller)`：`tree is None` / 无叶 / 无叶心 → `RuntimeError`（消息指明先调
   `setup(points[, charges])`）；`matvec` / `far_fraction` / 新增 `far_work_fraction` 入口处过门。
2. `setup()` 对空点集显式 `ValueError`（原先是 numpy 的 `zero-size array to reduction` 内部报错）。
3. `far_fraction` docstring 写明**配对**口径；新增 `far_work_fraction`（按叶内源数加权的代价口径）——
   同一几何两者 0.174 vs 0.093，代价结论应取后者。
4. `scripts/fmm_linearity_bench.py` 的 far_fraction 探针改为先 `setup`，并输出 work 口径。

已 setup 的算子数值逐位不变。

## 测试（先红后绿）

`tests/test_fmm_operator_setup_guard.py`（11 例，修复前 7 红）：
未 setup 的 `matvec`/`far_fraction`/`far_work_fraction` 必抛 `RuntimeError`；空点集 `setup` 抛
`ValueError`；m16000 几何 `far_fraction == 4/23`；`far_fraction` 与 block_size 无关；配对口径与
源数加权口径手算一致且两者不相等；等占据叶时两者相等；**不变量** `far_fraction == 0 ⇒ matvec 即
稠密直接求和`（单叶几何，2.3e-16）；远场份额 > 0 时清零多极系数必改变答案。

回归：`tests/test_fmm.py` + `test_fmm_operator_linearity.py` + `test_fmm_cost_audit.py` +
新文件 = **60 passed**。

## 工件

- `docs/fmm_far_fraction_diagnostic/README.md`（六要素，数字由脚本注入 `tables.md`）
- `docs/fmm_far_fraction_diagnostic/evidence.json` / `tables.md` / `fig_far_fraction.png`
- `scripts/fmm_far_fraction_diagnostic.py`（A/B：`_UnsetProbe` 复现修复前静默语义，唯一差异=守卫）
- `tests/test_fmm_diagnostic_evidence.py`（把 evidence.json 的头条数字钉死，防文档漂移）

## 交叉复核

icosphere level 3（1280 面元, p=4, max_leaf=32）实测 `far_fraction = 0.7400669642857143`，与已发布
`docs/fmm_cost_audit/audit.json` 同项逐位相同 ⇒ 历史代价审计的 routing 口径可复现，未受本单影响。

## 局限（如实）

1. 只修了 `FMMOperator`；`fmm_helmholtz_matvec` / `fmm_degenerate_matvec` 等自由函数内部自建算子，
   未普查其 fail-silent 面（建议后续一单）。
2. 80 面元（level 1）远场份额为 0.0 是**真实**的边界（叶近邻图覆盖全部叶对），不是缺陷；
   但意味着小网格上 FMM 与稠密等价，基准不应在小网格上宣称加速。
