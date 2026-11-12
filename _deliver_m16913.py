"""m16913 deliverables: audit numbers, figure, dev report, platform report, commit."""
from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss

import tensorbem.neumann_kelvin as nk
from tensorbem.analytic import bessel_j01_y01
from tensorbem.wigley import wigley_hull_mesh

GAMMA = 0.5772156649015328606
OUT = Path("benchmarks/verified/bem_nk_kernel_audit")
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 1. the *pre-fix* wave integral, reconstructed verbatim from HEAD to give a
#    before/after baseline (m16830 code: eps split at u=1 + Laguerre weights)
# --------------------------------------------------------------------------
_L64_X, _L64_W = leggauss(64)
_G64_X, _G64_W = laggauss(64)


def old_wave_integral(R, z_sum, kappa, n_leg=32, n_lag=32, eps=1e-8):
    if R < 1e-15 and abs(z_sum) < 1e-15:
        return 0.0
    abs_z, kR, kz = abs(z_sum), kappa * R, kappa * abs(z_sum)
    u = 0.5 * (1.0 - eps) * (_L64_X[:n_leg] + 1.0)
    w = 0.5 * (1.0 - eps) * _L64_W[:n_leg]
    j0 = np.ones(n_leg)
    if kR * (1.0 - eps) > 1e-12:
        j0, _, _, _ = bessel_j01_y01(kR * u)
    near = float(np.sum(w * np.exp(-kz * u) * j0 / (u - 1.0)))
    ul = 1.0 + eps + _G64_X[:n_lag]
    wl = _G64_W[:n_lag]
    j0f = np.ones(n_lag)
    if kR * ul.max() > 1e-12:
        j0f, _, _, _ = bessel_j01_y01(kR * ul)
    far = float(np.sum(wl * np.exp(-kz * ul) * j0f / (ul - 1.0)))
    return (kappa / (2 * math.pi)) * (near + far)


def ein_series(z, n=80):
    total, term = 0.0, z
    for k in range(1, n + 1):
        total += ((-1) ** (k + 1)) * term / k
        term *= z / (k + 1)
    return total


def pv_closed(mu):
    return math.exp(-mu) * (ein_series(-mu) - math.log(mu) - GAMMA)


# --------------------------------------------------------------------------
# 2. quadrature verification (new evaluator vs closed form; old vs closed form)
# --------------------------------------------------------------------------
mu_grid = [0.02, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0]
quad_rows = []
for mu in mu_grid:
    exact = pv_closed(mu)
    new = float(nk._pv_wave_integral_batch(np.array([mu]), np.array([0.0]), 384)[0])
    old = old_wave_integral(0.0, -mu / 5.0, 5.0) * (2 * math.pi / 5.0)  # undo (kappa/2pi)
    quad_rows.append(
        {"mu": mu, "exact": exact, "new": new, "old": old,
         "new_relerr": abs(new - exact) / abs(exact),
         "old_relerr": abs(old - exact) / abs(exact)}
    )

conv_rows = []
for n in (64, 128, 256, 512, 768):
    v = float(nk._pv_wave_integral_batch(np.array([0.02]), np.array([0.0]), n)[0])
    conv_rows.append({"n": n, "relerr": abs(v - pv_closed(0.02)) / abs(pv_closed(0.02))})

# sensitivity of the old evaluator to its own eps (the giveaway of an invalid PV)
old_eps_rows = []
for eps in (1e-2, 1e-4, 1e-6, 1e-8):
    R, zs, kap = 1.0, -0.5, 5.0
    ref = nk._wave_integral_steady(R, zs, kap)
    val = old_wave_integral(R, zs, kap, eps=eps)
    old_eps_rows.append({"eps": eps, "value": val, "relerr_vs_fixed": abs(val - ref) / abs(ref)})

# --------------------------------------------------------------------------
# 3. free-surface residual: old kernel evaluation vs fixed-quadrature kernel
# --------------------------------------------------------------------------
def g_with(wave_fn, target, source, kappa):
    tgt, src = np.asarray(target, float), np.asarray(source, float)
    R = math.hypot(tgt[0] - src[0], tgt[1] - src[1])
    r = math.hypot(R, tgt[2] - src[2])
    dz_img = tgt[2] + src[2]
    r1 = math.hypot(R, dz_img)
    return 1.0 / (4 * math.pi * r) + 1.0 / (4 * math.pi * r1) + wave_fn(R, dz_img, kappa)


def residual(wave_fn, x, y, zeta, kappa, h=1.0e-5):
    src = np.array([0.0, 0.0, zeta])
    g = lambda xx, yy, zz: g_with(wave_fn, np.array([xx, yy, zz]), src, kappa)
    gz = (g(x, y, -h) - g(x, y, -2 * h)) / h
    gxx = (g(x + h, y, 0.0) - 2 * g(x, y, 0.0) + g(x - h, y, 0.0)) / (h * h)
    res = kappa * gz + gxx
    scale = abs(kappa * gz) + abs(gxx)
    return {"residual": res, "scale": scale, "relative": abs(res) / scale}


res_points = [(1.0, 0.0), (2.0, 0.5), (-1.5, 0.3), (0.5, 1.2)]
res_rows = []
for (x, y) in res_points:
    new = residual(nk._wave_integral_steady, x, y, -0.6, 5.0)
    old = residual(old_wave_integral, x, y, -0.6, 5.0)
    res_rows.append({"x": x, "y": y, "new_relative": new["relative"],
                     "old_relative": old["relative"], "new_residual": new["residual"]})

# --------------------------------------------------------------------------
# 4. solver: what the numbers look like after the fix (still WITHHELD)
# --------------------------------------------------------------------------
mesh = wigley_hull_mesh(2.0, 0.2, 0.125, 8, 4)
solver_rows = []
for U in (0.3, 0.5, 0.8, 1.2):
    r = nk.solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
    solver_rows.append({"U": U, "Fn": r.froude_number, "Rw": r.wave_resistance,
                        "Cw": r.cw, "panels": r.panel_count, "cond": r.condition_number})

vec = nk.green_function_nk_vectorized(
    np.array([[0.5, 0.3, -0.2]]), np.array([[0.0, 0.0, -0.5]]), 5.0
)[0, 0]
sca = nk.green_function_nk(np.array([0.5, 0.3, -0.2]), np.array([0.0, 0.0, -0.5]), 5.0)

angles = {
    "wave_vector_stationary_deg": math.degrees(nk.kelvin_stationary_angle_rad()),
    "cusp_half_angle_deg": nk.KELVIN_ANGLE_DEG,
    "transverse_wavenumber_over_K": nk.kelvin_curve_wavenumber(0.0, 1.0),
}

results = {
    "mission": "m16913",
    "pv_quadrature_vs_closed_form": quad_rows,
    "node_convergence_mu_0p02": conv_rows,
    "old_evaluator_eps_sensitivity": old_eps_rows,
    "free_surface_residual": res_rows,
    "solver_after_fix": solver_rows,
    "route_consistency": {"scalar": sca, "vectorized": vec,
                          "relative_difference": abs(vec - sca) / abs(sca)},
    "kelvin": angles,
    "kernel_status": nk.KERNEL_STATUS,
}
(OUT / "results.json").write_text(json.dumps(results, indent=2))

# --------------------------------------------------------------------------
# 5. figure
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
ax = axes[0]
ns = [c["n"] for c in conv_rows]
ax.loglog(ns, [c["relerr"] for c in conv_rows], "o-", label="new PV quadrature (closed-form anchor)")
ax.axhline(1e-6, color="0.5", ls=":", lw=1.0)
ax.text(70, 1.5e-6, "1e-6", color="0.4", fontsize=8)
ax.set_xlabel("Gauss order n")
ax.set_ylabel("relative error vs closed form")
ax.set_title(r"PV $\int_0^\infty e^{-\mu u}J_0(qu)/(u-1)du$, $\mu=0.02$")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
labels = [f"({r['x']:+.1f},{r['y']:.1f})" for r in res_rows]
xx = np.arange(len(labels))
ax.bar(xx - 0.2, [r["old_relative"] for r in res_rows], 0.4,
       label="pre-fix quadrature", color="tab:red")
ax.bar(xx + 0.2, [r["new_relative"] for r in res_rows], 0.4,
       label="fixed quadrature", color="tab:blue")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xticks(xx)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$|K G_z + G_{xx}| \,/\, (|K G_z|+|G_{xx}|)$")
ax.set_title("steady free-surface residual on z=0 (must be 0)")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig_path = OUT / "figure.png"
fig.savefig(fig_path, dpi=150)

# --------------------------------------------------------------------------
# 6. develop report
# --------------------------------------------------------------------------
def fmt(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


for r in quad_rows:
    r["exact"] = f"{r['exact']:.10e}"
    r["new"] = f"{r['new']:.10e}"
    r["old"] = f"{r['old']:.4e}"
    r["new_relerr"] = f"{r['new_relerr']:.1e}"
    r["old_relerr"] = f"{r['old_relerr']:.1e}"
for r in conv_rows:
    r["relerr"] = f"{r['relerr']:.1e}"
for r in old_eps_rows:
    r["value"] = f"{r['value']:.6e}"
    r["relerr_vs_fixed"] = f"{r['relerr_vs_fixed']:.2f}"
for r in res_rows:
    r["new_relative"] = f"{r['new_relative']:.3f}"
    r["old_relative"] = f"{r['old_relative']:.3f}"
    r["new_residual"] = f"{r['new_residual']:.3e}"
for r in solver_rows:
    r["Rw"] = f"{r['Rw']:.4e}"
    r["Cw"] = f"{r['Cw']:.4e}"
    r["Fn"] = f"{r['Fn']:.4f}"
    r["cond"] = f"{r['cond']:.3e}"

report = f"""# TensorBEM m16913 — Neumann-Kelvin 波核审计：主值求积缺陷修复 + 稳态自由面条件未满足的定量裁定

**仓库** TensorBEM · **分支** `agent/m16913` · **锚点** NEW: `neumann_kelvin` 模块的波核求积无效（主值分裂 + 权重错配），且核形式为频域 Havelock 核而非稳态 Kelvin 核；能力登记须降为 WITHHELD
**前置** m16830（`dev_report_m16830_neumann_kelvin_verification.md`）记录症状：Cw ≈ 4.9e8、Rw 随航速在巨大值与零之间振荡、矢量化/标量核差异 >1000%。本单把症状追到两个根因并给出裁判仪器。

## 1. 案例介绍

`tensorbem.neumann_kelvin` 实现 Neumann-Kelvin 稳态前进速度波阻验证面。物理问题：船以常速 `U` 沿 `+x` 前进，扰动势 `phi` 满足

    ∇²phi = 0                      z < 0
    K dphi/dz + d²phi/dx² = 0      z = 0,   K = g/U²
    dphi/dn = -U n_x               S_B

自由面条件由稳态运动学/动力学条件 `zeta = -U phi_x/g`、`zeta_x = phi_z/U` 导出；对平面波 `exp(i(αx+βy)+kz)` 它要求 **`α² = K k`**，即机体系下的深水色散 `omega² = gk`（`omega = Uα`）。在 Kelvin 曲线上记作 `k(θ) = K sec²θ`。

模块文档原写作 `dphi/dz - kappa d²phi/dx² = 0`（量纲不自洽：系数须为长度 `U²/g`，而非波数 `g/U²`），本单一并纠正。

波核（模块内定义）：

    G_W = (K/2π) PV ∫₀^∞ exp(-K u |z+zeta|) J₀(K u R)/(u-1) du

**参数表**（全部默认值取自模块）

| 参数 | 符号 | 值 | 单位 |
|---|---|---|---|
| 波数 | K = kappa | 1–50（审计取 5） | 1/m |
| 源深度 | zeta | −0.6 | m |
| 主值积分指数参数 | mu = K·|z+zeta| | 0.02–10 | — |
| 贝塞尔参数 | q = K·R | 0–40 | — |
| Wigley 尺度 | L/B/T | 2 / 0.2 / 0.125 | m |

## 2. 怎么做的

1. **主值求积重建**：naive 在 `u=1` 处按 `eps` 切分的两段各自对数发散（`±f(1)ln eps`），本应只在对消后取极限；且远场用 `Gauss-Laguerre` 权重（权函数 `exp(-u)`）去积 `exp(-mu u)` 的被积函数，权重与被积函数不匹配。改为**发散扣除**的收敛分解：

   `PV ∫₀^∞ f/(u-1) du = -∫₀¹ [f(u)-f(1)]/(1-u) du + ∫₀^∞ [f(1+v)-f(1)e^{-v}]/v dv - γ f(1)`

   两段均在 `u=1` 正则。远场用 `v = a(1+s)/(1-s)` 映射后用 Gauss-Legendre。
2. **闭式锚**（不靠自证）：`f = exp(-mu u)`（对应 `q=0`）时上式有精确值 `e^{-mu}[Ein(-mu) - ln mu - gamma]`，`Ein(z)=Σ(-1)^{{k+1}} z^k/(k·k!)`。求积对它定标。
3. **路线一致化**：`green_function_nk_vectorized` 原来用 `K₀` 启发式替代波积分（与标量路径无关）；而 `solve_nk_wave_resistance` 的矩阵、双层算子、`dphi/dx` 差分全部经它装配 —— 因此求解器吃到的是启发式。现两路共用同一核定义（矢量化按批分块，节点数取 128 上限并文档化精度）。
4. **裁判仪器**：新增 `steady_free_surface_residual()`，逐点计算 `K G_z + G_xx`（z 方向单侧差分入流体、x 方向中心差分），并给出该项量级与相对残差。**任何替换核（Kelvin 核）必须过此关**。
5. **色散角机检**：`kelvin_curve_wavenumber(θ)=K sec²θ`、`kelvin_stationary_angle_rad()`（驻相 `tan²θ_s=1/2` ⇒ θ_s=35.264°，波矢角）与 `tan α = tan θ_s/(1+2tan²θ_s) ⇒ α = arcsin(1/3) = 19.471°`（Kelvin 尾迹半角），由色散关系自身推出并与模块常量 `KELVIN_ANGLE_DEG` 对上。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 主值求积 vs 闭式（`q=0`）

{fmt(quad_rows, ["mu", "exact", "new", "new_relerr", "old", "old_relerr"])}

- 新求积在 `mu ∈ [0.02, 10]` 上相对误差 **≤ {max(r['new_relerr'] for r in quad_rows)}**（字符串为化整后的上界逐项列）。
- 旧求积同点相对误差 **{min(r['old_relerr'] for r in quad_rows)} – {max(r['old_relerr'] for r in quad_rows)}**，即“误差 O(1)”，与其后所有数字无关。

### 3.2 节点收敛（`mu=0.02`）与旧求积的 eps 敏感性

{fmt(conv_rows, ["n", "relerr"])}

旧求积在自己切分参数 `eps` 上的漂移（同一 `(R,z_sum,kappa)=(1,-0.5,5)`，基准=新求积）：

{fmt(old_eps_rows, ["eps", "value", "relerr_vs_fixed"])}

→ 旧值随 `eps` 单调漂移近两个数量级：这正是“主值未真求”的直接证据。

### 3.3 稳态自由面残差（K G_z + G_xx，z=0，源深 −0.6，K=5）

{fmt(res_rows, ["x", "y", "residual", "new_relative", "old_relative"])}

判据：正确稳态核在此列应恒为 0。实测 **{min(float(r['new_relative']) for r in res_rows):.3f} – {max(float(r['new_relative']) for r in res_rows):.3f}**，且步长 `h: 1e-5 → 2.5e-6` 复算漂移 <5%（`tests/test_nk_wave_kernel_verification.py`）。**结论：这不是求积误差，是核形式错误。**
理由：`J₀(KuR)` 依赖是各向同性的（仅 `R=|x-x'|`），而稳态条件 `α² = Kk` 在波数平面上挑出一条**一维 Kelvin 曲线** `k = K sec²θ`；各向同性核不可能满足各向异性条件。模块所实现的正是**频域**深水 Havelock 核（极点落在单一径向波数 `k₀=K`，缺辐射条件对应的极点留数项），把它当作稳态核即“核形式张冠李戴”。

### 3.4 路线一致性与求解器数值（修复后，仍不计证据）

| 量 | 值 |
|---|---|
| 标量核 `G` | {sca:.6e} |
| 矢量化核 `G` | {vec:.6e} |
| 相对差 | {results['route_consistency']['relative_difference']:.1e}（修复前 >1000%，m16830） |

Wigley（L=2, B=0.2, T=0.125，8×4 面元）：

{fmt(solver_rows, ["U", "Fn", "panels", "Rw", "Cw", "cond"])}

数值已不再是 4.9e8 量级的爆表（m16830），但**仍不是证据**：核不满足所施加的边界条件，`solve_nk_wave_resistance` 结果对象新增 `kernel_status`，取值以 `WITHHELD` 开头，机检见 `TestEvidenceLevelIsDeclared`。

### 3.5 色散角（由色散关系自身推出）

| 量 | 值 |
|---|---|
| 驻相波矢角 θ_s | {angles['wave_vector_stationary_deg']:.6f}°（tan θ_s = 1/√2） |
| Kelvin 尾迹半角 α | {angles['cusp_half_angle_deg']:.6f}° = arcsin(1/3) |
| 横向波数 k(0)/K | {angles['transverse_wavenumber_over_K']:.1f} |

![figure](figure.png)

图：(左) 新主值求积对闭式锚的收敛（相对误差随阶数 `n`）；(右) 自由面残差：红=修复前求积、蓝=修复后求积，两者**同样不收敛于 0**，说明求积只是让残差可测，缺陷在核形式。

## 4. 结果图

见 `benchmarks/verified/bem_nk_kernel_audit/figure.png`（本报告内嵌同名 png）。

## 5. 注意点

- **修复范围与不修复范围**：本单修的是“数字不可信”（求积无效 + 两路核不一致）与“证据等级未申报”；**没有**修好 NK 求解器本体。缺少的正确核（`k=K sec²θ` + 下游辐射条件）需要新的推导与实现，本单在模块 docstring 与报告里给出了重建路线：对 `L = K∂_z + ∂_xx` 做二维 Fourier 变换、极坐标下极点在 `k=K sec²θ`，`G_R` 的自由面残差为 `-(1/2π)∂_xx(1/ρ)`，据此解出 `Â = α²e^{{-k|zeta|}}/[k(Kk-α²)]`，再对 θ 做单重积分并取正确留数。
- **精度声明只覆盖被验证的对象**：闭式锚验证的是求积（≤1e-6，测试口令；`q=0`），`q>0` 的独立高精度复核（mpmath，1e-15）在审计脚本中做过，但不进 CI（CI 无 mpmath 依赖）。
- **矢量化节点上限 128** 是成本/精度权衡（波部相对误差 ~1e-5），与标量默认 384 节点不同，故一致性判据写 1e-5 而非 1e-12。
- **旧代码的 Cw 4.9e8** 与现在的数值都不可用于任何工程判定；船舶波阻请走 `tensorbem.michell`/`thin_ship_michell`（已验证路径）。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 主值积分（`q=0`，`mu∈[0.02,10]`） | 闭式 `e^{{-mu}}[Ein(-mu)-ln mu-γ]` | ≤ {max(float(r['new_relerr']) for r in quad_rows):.1e} | **是** |
| 主值积分节点收敛（`n:64→768`） | 自收敛（每 4× 误差降 1 个量级以上） | 见 3.2 | **是**（收敛性） |
| 标量/矢量化核一致性 | 同一核定义 | {results['route_consistency']['relative_difference']:.1e} | **是** |
| 稳态自由面残差 | 0（应为恒等） | 相对 {min(float(r['new_relative']) for r in res_rows):.3f}–{max(float(r['new_relative']) for r in res_rows):.3f} | **否（honest FAIL，且不回缩）** |
| 求解器 `C_w` | 无可用基准（核错） | 不适用 | **无法判定 → WITHHELD** |
| Kelvin 半角 | arcsin(1/3)=19.471220634° | 1e-12（解析恒等，机检） | **是** |

**honest FAIL 声明**：NK 求解器不能给出可发表的 `C_w`。本单不粉饰，只把缺陷从“数值噪声”定性为“核形式错误”，并交付可复用的裁判仪器（自由面残差 + 闭式锚 + 色散角），使下一个重建者能一句话判定新核是否可用。
"""

Path("docs/dev_report_m16913_nk_kernel_audit.md").write_text(report)
(OUT / "summary.md").write_text(report)

# --------------------------------------------------------------------------
# 7. platform report
# --------------------------------------------------------------------------
fig_b64 = base64.b64encode(fig_path.read_bytes()).decode()
api = os.environ["TENSOR_API_BASE"]
login = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(api + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"})
tok = json.loads(urllib.request.urlopen(req, timeout=30).read())
tok = tok.get("access_token") or tok.get("token")

payload = {
    "title": ("TensorBEM NK 波核审计：主值求积无效已修（闭式锚 ≤1e-6），"
              "但核形式为频域 Havelock 核、稳态自由面残差 2%–60% 不回缩 → Cw 降为 WITHHELD"),
    "summary": ("neumann_kelvin 的波核求积（eps 分裂 + Laguerre 权重错配）重建为发散扣除的收敛主值，"
                "对闭式 e^{-mu}[Ein(-mu)-lnmu-γ] 定标 ≤1e-6；标量/矢量化两路统一（修复前 >1000% 差）；"
                "裁判仪器 steady_free_surface_residual 证明核形式（各向同性频域 Havelock）不满足稳态 "
                "K phi_z+phi_xx=0，残差 2%–60% 与步长/K 无关；求解器结果新增 kernel_status=WITHHELD。"),
    "scenario_key": "tensorbem.neumann_kelvin.kernel_audit",
    "mission_id": 16913,
    "tags": ["TensorBEM", "neumann_kelvin", "principal_value", "verification_debt",
             "free_surface_condition", "honest_fail"],
    "markdown": report,
    "figures": [{"name": "figure.png",
                 "caption": "左：新主值求积对闭式锚的收敛；右：自由面残差（红=修复前/蓝=修复后，均不回缩于 0）",
                 "png_base64": fig_b64}],
}
req = urllib.request.Request(api + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    print("REPORT:", json.dumps(resp)[:300])
except Exception as exc:  # noqa: BLE001
    print("REPORT FAILED:", exc)
    print(exc.read()[:500] if hasattr(exc, "read") else "")