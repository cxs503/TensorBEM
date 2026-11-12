# TensorBEM m16000 — FMMOperator 电荷缓存失效缺陷（m15669 引入的静默错答）+ 修复

**仓库**：TensorBEM · **分支**：agent/m16000 · **基线**：4d63824 (main @ 2026-11-12)

## 缺陷

`FMMOperator.matvec` 把多极系数缓存在 `_multipole_matrix`，只在 `is None` 时构建：

- `setup(points)`（无电荷）→ 首次 `matvec` 用当时 `charges` 构建多极，**永远不再失效**；
- 之后每次 `matvec`：近场用当前电荷直和，远场读旧电荷的多极系数 ⇒ **算子非线性**。

实测（200 源点聚簇 + 3 远场靶点，max_leaf=25）：第 2 次调用（换电荷向量）相对稠密直接求和误差
**2.464e+00（246%）**，且在 p=1..8 恒为 2.46（不随阶改善 ⇒ 系统性错位，非截断误差）。
叠加性残差 1.428e-01。

影响面：多右端项、迭代求解器（每次迭代换电荷）、复用算子的批处理——全部静默返回错值。
触发于 m15669 合入的向量化 matvec 路径（缓存语义是既有的，m15669 的可复现几何使其显形）。

## 修复（最小改动，`src/tensorbem/fmm.py`）

1. 新增 `_cache_points` / `_cache_charges` 指纹字段（`copy=True`，原地改数组也能被发现）；
2. 新增 `_multipole_cache_valid(sources, charges)`，`matvec` 不再用 `is None` 判断，
   改为**指纹不匹配即重建**；
3. `setup()` 同步刷新指纹；不带电荷重 `setup` 时清空 `_multipole_matrix`（防旧点集多极残留）；
4. `matvec` 对 `sources`/`charges` 形状显式校验（原为裸 `IndexError`）。

首次调用数值**逐位不变**（8/8 个 p 上 `fixed_first_err == prefix_first_err`），修复只影响换电荷场景。

## 测试（先红后绿）

`tests/test_fmm_operator_linearity.py`（10 例）：
- 首次调用返回前该文件即在未改代码上失败（`second call error 2.464e+00 (first 1.048e-08)`）；
- 覆盖：跨调用叠加/齐次性、多 RHS 与直接法一致、原地改电荷、`setup(q)` 后换电荷、
  GMRES 迭代求解与稠密解一致（2 例）、电荷长度校验。

回归：`tests/test_fmm.py`(23) + `test_fmm_cost_audit.py` + 新文件 = **49 passed**。
另：`tests/test_capability_evidence.py` 4 例红在 HEAD（stash 验证，与本单无关，未触碰）。

## 工件

- `docs/fmm_operator_linearity/README.md`（六要素报告，表格由脚本从 evidence.json 注入）
- `docs/fmm_operator_linearity/evidence.json`（p 扫描 A/B + GMRES + 缓存开销，全部机读）
- `docs/fmm_operator_linearity/fig_error_vs_p.png` / `fig_gmres_residual.png`
- `scripts/fmm_linearity_bench.py`（A/B：`_PrefixCacheOperator` 复现修复前语义，唯一差异=缓存谓词）

## 局限（如实）

1. GMRES 实验（N=80 icosphere）对缺陷**不敏感**（该网格几乎全走近场，两条曲线重合，解误差 6.7e-16）
   ⇒ 它是「算子可入迭代」的语义回归，不是判别性证据；判别性证据= p 扫描 A/B。双远簇几何未做。
2. `far_fraction()` 在同几何返回 0.0，与「远场缓存导致 2.46 误差」矛盾 ⇒ 疑似独立诊断缺陷，
   本单未修，已登记建议后续。
3. 缓存开销：命中 4.24 ms / 重建 4.00 ms（N=200，比值 0.94）⇒ 指纹比较成本在噪声内。

## 平台报告

report id **15144**（scenario_key `tensorbem_fmm_operator_linearity`，含两张图）；
发布脚本 `post_report_m16000.py`（登录→POST /tensor/reports，数字全部从 evidence.json 注入）。
