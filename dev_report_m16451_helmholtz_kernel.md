# TensorBEM m16451 — Helmholtz 单层核算子：三处静默截断 + 修复（正确性优先）

**仓库** TensorBEM · **分支** agent/m16451 · **基线** daef8c3 (main)
**锚点** GAP-BEM-FMM-HELMHOLTZ-KERNEL（report 15492 提议、未入 feature_ledger → 本单按 NEW 处置）

## 缺陷（全部由「与解析核比对」暴露，既有测试零覆盖）

`fmm_helmholtz_matvec` 是平台唯一对外 Helmholtz 算子入口；`tests/test_fmm.py::TestFailClosed`
三条测试只断言**模式字符串与形状**，从不与 `Σ q e^{ikr}/(4πr)` 比对。实测（`docs/fmm_helmholtz_kernel/evidence.json`）：

1. **只返回实部**：`_helmholtz_direct` 取 `.real` ⇒ 实际在用 `cos(kr)/r` 求和。
   `cos kr` 是出射/入射等幅叠加（驻波），幅相都不对；对精确复场的相对误差 **28 %–99 %**。
2. **`ka < 0.1` 用 Laplace 核顶替 Helmholtz 核**（k 被忽略）且模式标签写成 `"fmm"`，
   误差 **5.3 %（ka=0.02）→ 26.3 %（ka=0.099）**，标签让调用方以为这是加速后的同一算子。
3. **复电荷被 `np.asarray(charges, dtype=float)` 静默丢弃虚部**（仅一个 `ComplexWarning`）；
   `ka=3` 复电荷实测误差 **96.0 %**。
4. 附带：`ka > ka_max` 的 `"not_supported"` 描述的是**未加速**而非算不了（数值本身是对的），
   旧实现把「成本标签」与「正确性标签」混为一谈。

## 修复（`src/tensorbem/fmm.py`，Helmholtz 段重写，约 +120 行）

- `_helmholtz_direct` → 返回 **complex128 完整核和**，按 `max_pairs` 分块（与既有 Laplace 直和同风格，分块对目标轴切片 ⇒ 逐位不变）；
- 模式标签改为真话：`complex_direct`（`ka ≤ ka_max`，精确）/ `not_supported`（`ka > ka_max`，仍精确，仅关加速带）/
  `laplace_approx`（**仅** `low_ka_asymptotic=True` 显式开启，且对真复电荷 raise）；
- 复电荷保留（`_validate_helmholtz_inputs` 转 `complex128`，不再 cast）；
- 形状/数值校验前移（显式 `ValueError`）；空目标/空源返回零向量；
- 新增 `source_extent()`、`HELMHOLTZ_MODES`；模块 docstring 的 fail-closed 段重写（明示本模块**没有** Helmholtz FMM）。

## 测试

- 新增 `tests/test_fmm_helmholtz_kernel.py`：解析钉扎（单源闭式 ≤1e-14）、双求积、模式契约 8 档 `ka` 扫描（逐档与解析核 ≤1e-12）、
  渐近路线 opt-in/拒绝复电荷、复电荷不截断、`(∇²+k²)Φ=0` 有限差残差二阶收敛（粗/细比 64.0 ≈ 8²）、出射相位 `arg Φ = kr` 与 1/r 衰减、
  空输入/形状守卫、分块逐位不变性。
- `tests/test_fmm.py` 两条只读模式字符串的测试改为对解析核钉扎（并注明改的理由）。
- 回归：`test_fmm.py + test_fmm_helmholtz_kernel.py + test_fmm_cost_audit/geometry_contract/setup_contract/operator_linearity`
  = **95 passed in 32.8 s**。

## 工件

- `docs/fmm_helmholtz_kernel/README.md`（六要素，表格由 `scripts/fmm_helmholtz_report.py` 从 evidence.json 注入，零手抄）
- `docs/fmm_helmholtz_kernel/evidence.json`（10 档 `ka` A/B；**修复前模块由 `git show daef8c3:src/tensorbem/fmm.py` 原样加载**，非手写复现）
- `docs/fmm_helmholtz_kernel/fig_error_vs_ka.png`（误差 vs ka + 被丢弃的虚部占比）
- `scripts/fmm_helmholtz_scan.py`（A/B 证据）· `scripts/fmm_helmholtz_report.py`（报告渲染）· `scripts/m16451_patch_fmm_helmholtz.py`（补丁脚本，留痕）

## 未完成（honest）

- GAP-BEM-FMM-HELMHOLTZ-KERNEL 的量化验收「`ka ≥ 25` 仍 <1e-6 且加速 ≥2×」**未达成**：本单只把算子做正确、标签做诚实，
  `ka > ka_max` 仍是 O(NM) 直和，成本特性无任何改善。真 Helmholtz FMM（对角移位 + 球谐平移）应单独立项。
- 已知 CI 红（**与本单无关，已在 HEAD 无我改动时复现**）：`tests/test_doc_claim_audit.py` 3 例
  （`test_historical_documents_are_still_registered_in_the_baseline` /
  `test_ratchet_against_baseline` / `test_committed_artifacts_are_current`），涉及 m15295/m15632/m14563 报告与
  `capytaine_crosscheck.md` 未登记，及 `docs/doc_claim_audit/audit.json` 陈旧。
- 容量纪律：本机 load15 ~109 ≥100，全程只做轻量计算（N=12×40 的核和、6 点有限差），未起重算力进程。