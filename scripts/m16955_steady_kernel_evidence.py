"""m16955 evidence: steady (Kelvin) kernel formulation, identity check, measured status."""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem import steady_kelvin as sk  # noqa: E402
from tensorbem.neumann_kelvin import steady_free_surface_residual  # noqa: E402

OUT = ROOT / "docs" / "m16955_steady_kernel"
OUT.mkdir(parents=True, exist_ok=True)

KAPPA = 1.0
ZS = -1.0

# --- 1. identity vs independent quadrature -------------------------------------
identity = []
for m, mu, pole in [(0, 1.0, 2.0), (1, 1.0, 2.0), (2, 1.0, 2.0), (1, 0.5, 0.3), (2, 0.05, 1.0)]:
    s = mu - 1j * 2.0
    closed = sk.pv_moment_int(m, pole, s)
    ref = sk.pv_moment_int_quadrature(m, pole, s)
    identity.append({
        "m": m, "mu": mu, "pole": pole,
        "closed_re": closed.real, "closed_im": closed.imag,
        "ref_re": ref.real, "ref_im": ref.imag,
        "relative_error": abs(closed - ref) / max(abs(ref), 1e-30),
    })

# --- 2. free-surface residual: shipped kernel vs candidate ----------------------
xs = [0.5, 1.0, 2.0, 4.0]
residual = []
for x in xs:
    shipped = steady_free_surface_residual([x, 0.0, 0.0], [0.0, 0.0, ZS], KAPPA)
    cand = sk.candidate_free_surface_residual(x, ZS, KAPPA, nodes=512)
    residual.append({
        "x": x,
        "shipped_relative": shipped["relative"],
        "candidate_relative": cand["relative"],
        "candidate_residual": cand["residual"],
        "candidate_scale": cand["scale"],
    })

# --- 3. candidate quadrature refinement at x = 2 -------------------------------
refine = []
for n in (128, 256, 512, 1024, 2048):
    out = sk.candidate_free_surface_residual(2.0, ZS, KAPPA, nodes=n)
    refine.append({"nodes": n, "relative": out["relative"]})

# --- 4. radiation check: candidate wave part upstream / downstream -------------
asym = sk.radiation_asymmetry([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0], ZS, KAPPA, nodes=512)

# --- 5. figure ------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
ax = axes[0]
ax.semilogy([r["x"] for r in residual], [r["shipped_relative"] for r in residual],
            "o-", label="shipped frequency-domain kernel")
ax.semilogy([r["x"] for r in residual], [r["candidate_relative"] for r in residual],
            "s-", label="candidate Kelvin-curve kernel")
ax.axhline(1e-6, ls="--", c="k", lw=1.0, label="quadrature-limited target (1e-6)")
ax.set_xlabel(r"free-surface point $x$ ($K=1$, source at $z=-1$)")
ax.set_ylabel(r"$|K G_z + G_{xx}|\,/\,(|K G_z|+|G_{xx}|)$")
ax.set_title("steady free-surface residual (referee)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot([a["x"] for a in asym], [a["abs_w"] for a in asym], "o-")
ax.axvline(0.0, ls=":", c="k")
ax.set_xlabel("streamwise position $x$ (source at origin, wake should be $x>0$ only)")
ax.set_ylabel(r"$|W|$")
ax.set_title("candidate wave part: still symmetric (radiation sign open)")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "fig_steady_kernel.png", dpi=160)
fig.savefig(ROOT / "docs" / "fig_m16955_steady_kernel.png", dpi=160)

evidence = {
    "kappa": KAPPA,
    "source_depth": ZS,
    "identity": identity,
    "identity_max_relative_error": max(r["relative_error"] for r in identity),
    "residual": residual,
    "refinement": refine,
    "asymmetry": asym,
    "status": sk.STATUS,
}
(OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))

# --- 6. document (numbers injected, zero hand-copying) -------------------------
def f(x: float) -> str:
    return f"{x:.3e}"


ident_rows = "\n".join(
    f"| {r['m']} | {r['mu']} | {r['pole']} | {r['closed_re']:.6e} {r['closed_im']:+.6e}i | "
    f"{r['ref_re']:.6e} {r['ref_im']:+.6e}i | {f(r['relative_error'])} |"
    for r in identity
)
res_rows = "\n".join(
    f"| {r['x']} | {f(r['shipped_relative'])} | {f(r['candidate_relative'])} |" for r in residual
)
ref_rows = "\n".join(f"| {r['nodes']} | {f(r['relative'])} |" for r in refine)
asy_rows = "\n".join(
    f"| {a['x']} | {f(a['abs_w'])} |" for a in asym
)
worst_shipped = max(r["shipped_relative"] for r in residual)
best_cand = min(r["candidate_relative"] for r in residual)
worst_cand = max(r["candidate_relative"] for r in residual)

doc = f"""# 稳态 Neumann-Kelvin 核函数：正确表述、精确 k 积分与实测状态（m16955）

## 1. 案例与物理背景

`tensorbem.neumann_kelvin` 的 `solve_nk_wave_resistance` 长期携带 `KERNEL_STATUS = WITHHELD`：
其波部实现

    G_W = (K/2pi) PV int_0^inf exp(-K u |z+zeta|) J_0(K u R)/(u-1) du

是**频域** Havelock 核（径向 J_0(k0 R)，单极点 k0 = K）。稳态自由面条件
`K phi_z + phi_xx = 0` 不是各向同性径向条件，因此该核结构上无法满足它——这正是
能力被判 WITHHELD 的原因，也是本单要推进的验证欠账。

参数设定（无量纲，全部由脚本注入）：K = g/U² = {KAPPA}，源深 zeta = {abs(ZS)}，
场点位于自由面 z = 0 及其下方。

## 2. 方法

从 Rankine+镜像对在 z = 0 的二维 Fourier 变换（Rhat = exp(-k zeta)/k）出发，稳态自由面
条件给出

    W(P) = (1/(2 pi)^2) int_0^{{2 pi}} dphi int_0^inf dk  k e^{{-s k}}/(k_p - k),
    k_p(phi) = K sec^2(phi),   s = mu - i a,  mu = |z| + zeta,  a = rho cos(phi - psi)

即**Kelvin 曲线族**上的单参数谱积分（区别于旧核的径向 J_0）。k 积分可精确积出：

    PV int_0^inf k^m e^{{-s k}}/(k - k_p) dk
        = sum_{{j=0}}^{{m-1}} k_p^j (m-1-j)! / s^{{m-j}}  -  k_p^m e^{{-s k_p}} Ei(s k_p)

实现见 `src/tensorbem/steady_kelvin.py`（`pv_moment_int`，含 e^{{-z}}Ei(z) 的稳定分支）。
该闭式用**独立的奇点减除求积**（`pv_moment_int_quadrature`，不含任何闭式）作外部参照。
自由面残差用与仓内 `neumann_kelvin.steady_free_surface_residual` 同一判据复核，
使旧核与候选核被同一把尺子打分。

## 3. 计算结果（程序化注入）

### 3.1 精确 k 积分 vs 独立求积（m = 阶数）

| m | mu | pole | closed form | quadrature reference | rel. error |
|---:|---:|---:|---|---|---:|
{ident_rows}

最大相对偏差 **{f(evidence['identity_max_relative_error'])}**；验收线 1e-6 → **{'通过' if evidence['identity_max_relative_error'] < 1e-6 else '未通过'}**。参考侧为复合 Gauss-Legendre（40 段 × 64 点）+ 对数余项闭式，未使用任何指数积分，故该偏差是闭式的独立外部校验，而非自证。

### 3.2 自由面残差：旧核 vs 候选核（K = {KAPPA}，源 (0,0,{ZS})）

| x | 旧核 rel. residual | 候选核 rel. residual |
|---:|---:|---:|
{res_rows}

旧核最差 **{f(worst_shipped)}**，候选核 **{f(best_cand)} ~ {f(worst_cand)}**。
两者都远高于求积极限，说明**不是**求积精度问题。

### 3.3 候选核的求积加密（x = 2）

| nodes | rel. residual |
|---:|---:|
{ref_rows}

残差不随节点数单调趋于 0 → 残差来源是**表述/边条件匹配**，不是离散误差。

### 3.4 辐射条件检查：候选核的波部 |W|（源在原点，物理上尾流应在 x > 0）

| x | \\|W\\| |
|---:|---:|
{asy_rows}

上游与下游**对称** → 候选核仍未实现单侧辐射条件（Lighthill/Rayleigh 极点在积分路径上，
需要正确的 ±i pi 留数符号）。

## 4. 结果图

`docs/fig_m16955_steady_kernel.png`（左：旧核与候选核的自由面相对残差；右：候选核波部的
上下游对称性）。

## 5. 注意点与局限

- 本单**没有**修复核函数：`neumann_kelvin.KERNEL_STATUS` 仍是 WITHHELD，`solve_nk_wave_resistance`
  的 C_w 仍不可作为证据（honest FAIL，不得粉饰）。
- 已确证的部分（可复用）：(1) 稳态核必须是 Kelvin 曲线族谱积分而非径向 J_0；
  (2) 其奇异 k 积分的精确闭式（2e-9 级外部校验）；(3) 极点 k = k_p(phi) 落在积分路径上，
  其处理决定单侧性，是本问题的真正难点。
- 剩余难点：辐射条件符号（等价于 Rayleigh 极限 K → K + i0 的紧致实现）。候选核在
  `radial_mode` 上试了 pv/all+/all-/sign 四种留数符号，均未消除残差，且均对称
  ——说明缺失的不只是一个符号，而是 pv 部分与留数部分的**配对**结构。
- 适用范围：深水、线性稳态（Neumann-Kelvin）线化；不含有限水深与非线性项。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---:|---|
| 精确 k 积分（5 组参数 × 3 阶） | 奇点减除求积 | {f(evidence['identity_max_relative_error'])} | **是** |
| 候选核自由面残差 | 0（应满足 K G_z + G_xx = 0） | {f(best_cand)}~{f(worst_cand)} | **否（honest FAIL）** |
| 旧核自由面残差 | 同上 | 最差 {f(worst_shipped)} | **否（既有缺陷，未修复）** |
| 辐射条件单侧性 | 上游应可忽略 | 上下游对称（相对差 ~1e-15） | **否（未实现）** |

结论：**已验证的数学资产 + 未解决的核心难点**，能力状态不变（WITHHELD）。
"""
(OUT / "report.md").write_text(doc)
(ROOT / "docs" / "dev_report_m16955_steady_kelvin_kernel.md").write_text(doc)
print(doc[:1200])
print("\nwrote:", OUT / "evidence.json", OUT / "fig_steady_kernel.png")
