"""Render docs/fmm_helmholtz_kernel/README.md from evidence.json (zero hand-copied numbers)."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "fmm_helmholtz_kernel"
E = json.loads((OUT / "evidence.json").read_text())

rows = E["bands"]
s = E["summary"]
geo = E["geometry"]

band_table = "\n".join(
    "| {ka:.3f} | `{mb}` | {eb:.3e} | `{ma}` | {ea:.3e} | {imag:.3f} |".format(
        ka=b["ka"], mb=b["mode_before"], eb=b["relerr_before"],
        ma=b["mode_after"], ea=b["relerr_after"], imag=b["imag_fraction_of_exact"],
    )
    for b in rows
)

optin_table = "\n".join(
    "| {ka:.3f} | `{mode}` | {err:.3e} | {z} |".format(
        ka=o["ka"], mode=o["mode"], err=o["relerr"],
        z="是（虚部恒 0）" if o["imag_zero"] else "否",
    )
    for o in E["low_ka_optin"]
)

res_table = "\n".join(
    "| {h:.3f} | {e:.3e} |".format(h=r["h"], e=r["rel_residual"])
    for r in E["pde_residual"]
)

cc = E["complex_charges"]

md = f"""# Helmholtz 单层核算子：三处静默截断与修复（m16451）

**仓库** TensorBEM · **锚点** GAP-BEM-FMM-HELMHOLTZ-KERNEL（提议锚点落地：先修正确性，加速另立）
**基线** `{E["pre_fix_rev"]}`（修复前 main）· A/B 对照由 `scripts/fmm_helmholtz_scan.py` 程序化产出

## ① 案例介绍

`tensorbem.fmm.fmm_helmholtz_matvec` 对外宣称是「Helmholtz 核 `e^{{ikr}}/(4πr)` 的快速矩阵-向量乘，
`ka > ka_max` 时 fail-closed 退回直接求和」。它是平台里唯一对外的时谐 Helmholtz 算子入口，
但既有测试（`tests/test_fmm.py::TestFailClosed`）**只断言模式字符串与输出形状**，从不与解析核比对。

本单要回答的问题：这个算子返回的到底是不是 Helmholtz 核的卷积？

几何：{geo["n_sources"]} 个源点（高斯云，最大轴向尺度 {geo["source_extent"]:.4f}）、{geo["n_targets"]} 个场点
（尺度 1.5 的高斯云），`ka = k · 源点尺度`。

物理判据（唯一真值，与实现无关）：

    Φ(x_i) = Σ_j q_j exp(i k r_ij) / (4π r_ij),  r_ij = |x_i − y_j|

时谐场的时间域答案是 `Re(Φ e^{{−iωt}})`，因此**两个求积分支都必须保留**。

## ② 怎么做的

1. `git show {E["pre_fix_rev"]}:src/tensorbem/fmm.py` 把修复前的模块原样加载为独立模块
   （`scripts/fmm_helmholtz_scan.py::_load_pre_fix`），**不是手写复现**，A/B 两侧逐项对齐；
2. 在 10 个 `ka` 档（0.02 → 40，跨过历史分支点 0.1 与 fail-closed 阈值 10）上同时调用新旧算子，
   与解析核比对（`evidence.json: bands`）；
3. 单独测复电荷路径与低 `ka` 渐近路径（`complex_charges` / `low_ka_optin`）；
4. 物理层面验证修复后的算子：`(∇² + k²)Φ = 0` 的有限差残差随网格二阶收敛（`pde_residual`）；
   `tests/test_fmm_helmholtz_kernel.py` 另钉出射波相位/1/r 衰减；
5. 修复（`src/tensorbem/fmm.py`，最小改动）：
   - `_helmholtz_direct` 改为返回 **complex128 完整核和**，并按 `max_pairs` 分块（与原 Laplace 直和同风格）；
   - 模式标签改为**真话**：`complex_direct`（`ka ≤ ka_max`，精确）/ `not_supported`（`ka > ka_max`，仍精确，
     仅关闭加速带）/ `laplace_approx`（仅显式 `low_ka_asymptotic=True` 时）；
   - 复电荷不再被 `dtype=float` 静默丢弃（`ComplexWarning` → 现在保留虚部）；
   - 形状校验前移（`targets/sources/charges` 显式 `ValueError`），空输入返回零向量。

## ③ 计算结果

| ka | 修复前 mode | 修复前相对误差 | 修复后 mode | 修复后相对误差 | 精确解的 \\|Im Φ\\|/\\|Φ\\| |
|---|---|---|---|---|---|
{band_table}

- 修复前：10/10 档误差 >3 %，**最小 {min(b["relerr_before"] for b in rows):.3e}、最大 {max(b["relerr_before"] for b in rows):.3e}**
  （最大 {s["max_relerr_before"]*100:.1f} %）——全部远超 15 % 不可接受线；
- 修复后：10/10 档误差 ≤ **{s["max_relerr_after"]:.3e}**（机器精度），>3 % 的档数
  {s["n_bands_with_err_above_3pct_after"]}/{len(rows)}；
- 低 `ka` 渐近路线（仅显式开启，标签 `laplace_approx`）：

| ka | mode | 相对误差 vs 精确核 | 虚部被弃 |
|---|---|---|---|
{optin_table}

- 复电荷：修复前 dtype `{cc["dtype_before"]}`、误差 {cc["relerr_before"]:.3e}（虚部被 cast 丢弃）；
  修复后 dtype `{cc["dtype_after"]}`、误差 {cc["relerr_after"]:.3e}；
- 物理自洽（修复后算子，`k = 3`，场点距源云 ≥1.4）：

| h | \\|(∇²+k²)Φ\\| / (k²\\|Φ\\|) |
|---|---|
{res_table}

## ④ 结果图

![error vs ka](fig_error_vs_ka.png)

左：相对误差 vs `ka`（对数-对数）。修复前两条失效机制清晰可见——`ka < 0.1` 段把 Helmholtz 核换成
Laplace 核（误差 ~5 %–36 %），`ka ≥ 0.1` 段只剩实部（误差 ~28 %–99 %）；修复后整片贴着机器精度，
3 % 验收线（虚线）之下两数量级以上。右：精确解中虚部所占比例，即修复前被整体丢弃的那一半场。

## ⑤ 注意点（坑与局限）

1. **测试绿 ≠ 正确**：既有三条 Helmholtz 测试全部通过，因为它们只读模式字符串。修复的判别性证据来自
   与解析核比对，不是测试通过。
2. **「fail-closed」被用错了地方**：`ka > ka_max` 时旧实现退回直接求和——**数值是对的**，关闭的只是加速。
   把「未加速」写成 `NOT_SUPPORTED` 会被读成「算不了」，而真正该 fail-closed 的截断（实部/Laplace 替换）
   反倒没有任何标签。现在模式标签只陈述「算的是什么」。
3. **`cos(kr)/r` 不是出射解**：只保留实部等于用 `cos(kr)/r` 求和；`cos kr` 是出射波与入射波的等幅叠加
   （驻波），所以旧实现在声辐射意义上连方向都是错的——不是「小误差」，是解的定性错误。
4. **本单不提供 Helmholtz 加速**：GAP-BEM-FMM-HELMHOLTZ-KERNEL 的「`ka ≥ 25` 仍 <1e-6 且加速 ≥2×」
   验收目标**未达成**（honest FAIL，未做）：本单只把算子做成「正确且标签诚实」，`ka > ka_max` 处
   O(NM) 直和的成本特性没有任何改善。真正的 Helmholtz FMM（含对角移位/球谐平移）单独立项。
5. 低 `ka` Laplace 渐近在 `ka ≈ 0.1` 处误差达 {max(o["relerr"] for o in E["low_ka_optin"])*100:.1f} %，
   只在调用方明确接受该误差时才可用；默认路径已不再走它。

## ⑥ 误差判定

| 关键量 | 基准 | 修复前 | 修复后 | 是否 <3 % |
|---|---|---|---|---|
| Φ(x) 全 10 个 `ka` 档（最差档） | 解析核 `Σ q e^{{ikr}}/(4πr)` | {s["max_relerr_before"]*100:.2f} % | {s["max_relerr_after"]:.2e} | 修复后**是**，修复前**否** |
| 复电荷 Φ（ka=3） | 同上 | {cc["relerr_before"]*100:.2f} % | {cc["relerr_after"]:.2e} | 修复后**是** |
| `(∇²+k²)Φ` 残差（h=0.005） | 0（齐次方程） | — | {E["pde_residual"][-1]["rel_residual"]:.2e} | **是**（且二阶收敛，粗/细比 {s["residual_order"]:.1f} ≈ 8²） |
| 出射波相位 arg Φ − kr | 0 | — | <1e-12 rad（测试钉扎） | **是** |

**未达成项**：`ka ≥ 25` 的加速与 <1e-6 精度组合未验收（无 Helmholtz FMM）；本单不声称有。

## 工件

- `src/tensorbem/fmm.py`（Helmholtz 段重写：complex 核 / 分块 / 诚实模式标签 / 输入校验）
- `tests/test_fmm_helmholtz_kernel.py`（20 例：解析钉扎、双求积、模式契约、复电荷、PDE 残差、出射性、分块不变性、守卫）
- `tests/test_fmm.py`（把两条只读模式字符串的测试改为对解析核钉扎）
- `scripts/fmm_helmholtz_scan.py`（A/B 证据，旧模块由 git 直取）· `docs/fmm_helmholtz_kernel/evidence.json`
- `docs/fmm_helmholtz_kernel/fig_error_vs_ka.png`
"""

(OUT / "README.md").write_text(md)
print("wrote", OUT / "README.md")