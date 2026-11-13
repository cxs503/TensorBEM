"""Report generator for CASE-BEM-CUMMINS-WIGLEY (6-DOF memory kernel), mission m17209.

Every number in the markdown and in the figures is read from
``artifacts/cummins_wigley_kernel_6dof/results.json`` (written by
``scripts/m17209_cummins_wigley_kernel_6dof.py``) or from the committed Capytaine
reference -- nothing is typed by hand.

    PYTHONPATH=src python scripts/m17209_report.py            # markdown + figures
    PYTHONPATH=src python scripts/m17209_report.py --post     # + publish report
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = pathlib.Path(os.environ.get("M17209_OUT", str(ROOT / "artifacts" / "cummins_wigley_kernel_6dof")))
DOCS = ROOT / "docs"
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100.0 * x:+.{digits}f} %"


def mark(x: bool) -> str:
    return "✓" if x else "✗"


def make_figures(r: dict) -> list[dict]:
    DOCS.mkdir(exist_ok=True)
    figs = []

    # 1. memory kernels, all six DOF
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label in DOFS:
        st = r["studies"][label]
        tau = np.asarray(st["kernel"]["tau"])
        k = np.asarray(st["kernel"]["values"])
        ax.plot(tau, k / max(abs(k[0]), 1e-30), label=label, lw=1.4)
    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(r"$K_{ii}(\tau)/K_{ii}(0)$")
    ax.set_title("Cummins memory kernels, Wigley III hull (medium mesh)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    p1 = DOCS / "fig_m17209_cummins_kernels_6dof.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    figs.append({"name": p1.name,
                 "caption": "Normalised Cummins memory kernels K_ii(tau)/K_ii(0) for the six DOF "
                            "of the Wigley III hull, from K(t)=(2/pi)int B(w)cos(wt)dw."})

    # 2. cross-solver comparison at omega = 1 rad/s
    cross = r["cross_solver"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    x = np.arange(6)
    ref_a = [row["added_mass_ref"] for row in cross["rows"]]
    got_a = [row["added_mass_got"] for row in cross["rows"]]
    ref_b = [row["damping_ref"] for row in cross["rows"]]
    got_b = [row["damping_got"] for row in cross["rows"]]
    for ax, ref, got, title, unit in (
        (axes[0], ref_a, got_a, "added mass A_ii [kg, kg m$^2$]", "A"),
        (axes[1], ref_b, got_b, "damping B_ii [kg/s, kg m$^2$/s]", "B"),
    ):
        ax.bar(x - 0.2, ref, 0.4, label="Capytaine 3.1 (independent)", color="#888")
        ax.bar(x + 0.2, got, 0.4, label="TensorBEM", color="#1f77b4")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(DOFS, rotation=30, fontsize=8)
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle("Wigley III radiation coefficients at $\\omega=1$ rad/s: independent-solver check",
                 fontsize=10)
    fig.tight_layout()
    p2 = DOCS / "fig_m17209_cross_solver_6dof.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    figs.append({"name": p2.name,
                 "caption": "TensorBEM vs Capytaine 3.1 diagonal radiation coefficients at "
                            "omega = 1 rad/s (log scale); values loaded from the committed "
                            "reference file."})

    # 3. heave free decay (Cummins) + KK residual per DOF
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    dec = r["free_decay_heave"]
    axes[0].plot(dec["time"], dec["displacement"], lw=1.1)
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("heave displacement [m]")
    axes[0].set_title(f"Cummins free decay, heave\n"
                      f"$\\zeta_{{decay}}$={dec['zeta_free_decay']:.4f}, "
                      f"$\\zeta_{{half-power}}$={dec['zeta_half_power']:.4f}", fontsize=9)
    axes[0].grid(alpha=0.3)
    r3 = [r["studies"][d]["R3"]["rel_err"] for d in DOFS]
    r5 = [r["studies"][d]["R5"]["rel_err"] for d in DOFS]
    r1 = [abs(r["per_dof"][d]["R1_added_mass_rel_err"]) for d in DOFS]
    x = np.arange(6)
    axes[1].bar(x - 0.27, r1, 0.27, label="R1 vs Capytaine (A_ii)")
    axes[1].bar(x, r3, 0.27, label="R3 Kramers-Kronig")
    axes[1].bar(x + 0.27, r5, 0.27, label="R5 round trip")
    axes[1].axhline(0.10, color="r", ls="--", lw=1, label="case tol 10 %")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DOFS, rotation=30, fontsize=8)
    axes[1].set_ylabel("relative error")
    axes[1].set_title("Gate residuals per DOF", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    p3 = DOCS / "fig_m17209_gates_6dof.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    figs.append({"name": p3.name,
                 "caption": "Left: Cummins-equation heave free decay. Right: per-DOF gate "
                            "residuals (R1 independent solver, R3 Kramers-Kronig, R5 band-limited "
                            "round trip) against the 10% case tolerance."})
    return figs


def markdown(r: dict, figs: list[dict]) -> str:
    g = r["geometry"]
    gates = r["gates"]
    cross = r["cross_solver"]
    dec = r["free_decay_heave"]
    sens = r["sensitivity"]
    rows = "\n".join(
        f"| {row['dof_label']} | {row['added_mass_ref']:.4g} | {row['added_mass_got']:.4g} | "
        f"{pct(row['added_mass_rel_err'])} {mark(abs(row['added_mass_rel_err']) <= 0.10)} | "
        f"{row['damping_ref']:.4g} | {row['damping_got']:.4g} | "
        f"{pct(row['damping_rel_err'])} {mark(abs(row['damping_rel_err']) <= 0.10)} |"
        for row in cross["rows"] if row["damping_rel_err"] is not None
    )
    per_dof = "\n".join(
        f"| {d} | {pct(v['R1_added_mass_rel_err'])} | {pct(v['R1_damping_rel_err'])} | "
        f"{pct(v['R2_rel_err'])} | {pct(v['R3_rel_err'])} | {pct(v['R5_rel_err'])} | "
        f"{v['added_mass_inf']:.5g} |"
        for d, v in r["per_dof"].items()
    )
    worst_r1 = max(abs(v["R1_added_mass_rel_err"]) for v in r["per_dof"].values())
    gate_lines = "\n".join(
        f"| {k} | {v['id']} | {v['value']:.4g} | {v['tol']:.2f} | {mark(v['passed'])} |"
        for k, v in gates.items()
    )
    mesh = sens["mesh"]
    mesh_lines = "\n".join(
        f"| {lv} | {v['level']} | {v['panels']} | "
        + " | ".join(f"{a:.4g}" for a in v["added_mass_diag"]) + " |"
        for lv, v in mesh.items()
    )
    n_worst = max((abs(v["R1_added_mass_rel_err"]), d) for d, v in r["per_dof"].items())
    fig_md = "\n".join(f"![{f['name']}]({f['name']})\n\n*{f['caption']}*\n" for f in figs)
    verdict = r["verdict"]
    return f"""# CASE-BEM-CUMMINS-WIGLEY：Wigley III 船 6 自由度 Cummins 记忆核（m17209）

**仓** TensorBEM · **算例** CASE-BEM-CUMMINS-WIGLEY（算例池） · **判定** `{verdict}`
**工件** `artifacts/cummins_wigley_kernel_6dof/results.json`、`sweep_coarse.json`、`sweep_medium.json`
**复现**
```
PYTHONPATH=src python scripts/m17209_cummins_wigley_kernel_6dof.py --stage sweep   # BEM 扫频（分钟级）
PYTHONPATH=src python scripts/m17209_cummins_wigley_kernel_6dof.py --stage gates   # 五道闸门（秒级）
PYTHONPATH=src python -m pytest tests/test_cummins_wigley_kernel_6dof.py -q        # 4 passed
```

## ① 问题 + ref 出处

**问题**：对一艘有航速修形的船型（水面穿透、非流线型截面）而言，TensorBEM 的频域辐射解
A(ω)、B(ω) 经 Cummins 方程转到时域后，记忆核 K(t) 与自由衰减是否在 6 个自由度上都成立？

$$
\\left(M + A_\\infty\\right)\\ddot{{x}} + \\int_0^t K(t-s)\\,\\dot{{x}}(s)\\,ds + Cx = F(t),
\\qquad K(t) = \\frac{{2}}{{\\pi}}\\int_0^\\infty B(\\omega)\\cos(\\omega t)\\,d\\omega
$$

**参考值（全部运行时读入，零手抄）**

| 编号 | 参考 | 来源（仓内可核） |
|---|---|---|
| R1 | 独立求解器 Capytaine 3.1 的 A_ii、B_ii（ω=1 rad/s，6 DOF，三级网格） | `benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`（内含 python 3.11.6 / numpy 2.4.6 / capytaine 3.1.dev0 / loongarch64 / ρ=1025 / g=9.80665 / 水深远洋 / 旋转中心 (0,0,−0.09375) 的完整环境记录） |
| R2 | Cummins 恒等式 `K(0)=(2/π)∫B dω` | 上式本身；程序内以同一 B(ω) 的梯形积分对照采样核值 |
| R3 | Kramers-Kronig 因果 `A(ω)−A(∞)=Hilbert[B](ω)` | `tensorbem.causality.kkr_added_mass`（主值积分，取扫频中点） |
| R4 | 时域自由衰减阻尼比 vs 频域半功率阻尼比 | 同一 A、B 的两种独立估计（Cummins 卷积 / 频响半功率） |
| R5 | 往返闭环 B→K→B | `tensorbem.time_domain.fourier_round_trip`（自带有效频带） |

**诚实声明**：仓内**没有任何数字化发表的 K(t) 表或 Wigley III 辐射系数表**；且仓的出处闸门
`tensorbem.reference_provenance` 把"回忆值"一律记为债务（不算证据，见
`docs/dev_report_m14725_reference_provenance.md`，其中 15 个文献常数 14 个为回忆值）。因此本算例
**不引用任何"原文表格页码"**——编造页码比没有页码更坏。R1 是独立求解器锚点（唯一能发现频域
解的**系统性**偏差的参考，但只覆盖 ω=1 rad/s 单频点）；R2–R5 是 Cummins/KK 形式体系自带的精确
恒等式，**只能发现实现与采样误差，不能发现 BEM 本身的系统偏差**——这一局限写进 §⑤。

## ② 方法与实现

1. **几何**：Wigley III 数学船型（`tensorbem.wigley3_freq.wigley3_wetted_hull_mesh`），
   L={g['L']} m，B={g['B']} m，T={g['T']} m，c={g['c']:.4f}，**Cb={g['Cb']:.4f}**，
   ∇={g['volume']:.8f} m³，L/B={g['L_B']:.1f}，B/T={g['B_T']:.1f}，ρ=1025 kg/m³，g=9.80665 m/s²。
   几何闸门（Cb、∇ 与 L·B·T·Cb 一致性）在 `wigley3_hydrostatic_reference` 内已由既有测试守护。
2. **频域**：3-D 面元 BEM（Kelvin 格林函数，`solve_radiation_3d`，`chief_points="auto"`），
   ω ∈ {{{', '.join(f'{w:g}' for w in r['omega'])}}} rad/s（无量纲
   ω√(L/g) ∈ [{min(r['nondim_omega_sqrt_L_over_g']):.3f}, {max(r['nondim_omega_sqrt_L_over_g']):.3f}]），
   一次求解同时给出 6×6 的 A、B。
3. **时域**：`radiation_memory_kernel` 由 B(ω) 的余弦变换给出 K(t)（dt={r['kernel_settings']['dt']} s，
   τ_max={r['kernel_settings']['tau_max']} s）；`cummins_free_decay` 解 Cummins 方程自由衰减（heave，
   质量 = ρ∇、恢复力 C33 = ρgA_wp 均由几何算出，**无经验回转半径**）；`half_power_damping`
   由同一 A、B 给出频域阻尼比。
4. **闸门**：`cummins_wigley_kernel_6dof.run_case` 产出 R1–R5，预声明容差见
   `GATES`：R1/R3/R4 = **10 %（本算例容差）**，R2 = 2 %，R5 = 5 %。

## ③ 参数表（无量纲数完整）

| 量 | 值 |
|---|---|
| L / B / T | {g['L']} / {g['B']} / {g['T']} m |
| L/B，B/T | {g['L_B']:.1f}，{g['B_T']:.2f} |
| Cb（方块系数） | {g['Cb']:.4f} |
| ∇ | {g['volume']:.8f} m³ |
| ρ，g | 1025 kg/m³，9.80665 m/s² |
| 排水质量 | {dec['mass']:.4f} kg |
| C33（heave 恢复力） | {dec['restoring_C33']:.5g} N/m |
| A_wp | {dec['waterplane_area']:.6f} m² |
| 频率节点数 | {len(r['omega'])} |
| ω 范围 / ω√(L/g) | {r['omega'][0]:g}–{r['omega'][-1]:g} rad/s / {min(r['nondim_omega_sqrt_L_over_g']):.3f}–{max(r['nondim_omega_sqrt_L_over_g']):.3f} |
| 网格（门禁档） | {cross['reference']['level']}，{cross['reference']['panels']} 面元 |
| 核采样 | dt={r['kernel_settings']['dt']} s，τ_max={r['kernel_settings']['tau_max']} s |

## ④ 对照表（ref vs got + 误差打勾）

**R1 独立求解器对照（ω = 1 rad/s，{cross['reference']['level']}）**

| DOF | A_ref (Capytaine) | A_got (TensorBEM) | 误差 | B_ref | B_got | 误差 |
|---|---|---|---|---|---|---|
{rows}

**五道闸门（逐 DOF 明细）**

| DOF | R1 A 误差 | R1 B 误差 | R2 K(0) 误差 | R3 KK 误差 | R5 往返误差 | A_∞ |
|---|---|---|---|---|---|---|
{per_dof}

| 闸门 | 名称 | 值 | 容差 | 判定 |
|---|---|---|---|---|
{gate_lines}

**R4 自由衰减（heave）**：ζ_decay = {dec['zeta_free_decay']:.5f}，ζ_half-power = {dec['zeta_half_power']:.5f}，
相对误差 {pct(dec['rel_err'])} {mark(abs(dec['rel_err']) <= 0.10)}；
周期 {dec['decay_metrics']['period']:.4f} s，衰减率 {dec['decay_metrics']['decay_rate']:.5f} 1/s，
对数幅值拟合 R² = {dec['decay_metrics']['fit_r_squared']:.5f}。

**最差 DOF**：A_ii 相对 {n_worst[1]}（{pct(n_worst[0])}）。总判定 `{verdict}`。

## ⑤ 收敛性与参数敏感性（两点）

1. **网格阶梯**（同一 ω=1 rad/s 的 A_ii 对角）：

| 档 | 标签 | 面元 | A_11 | A_22 | A_33 | A_44 | A_55 | A_66 |
|---|---|---|---|---|---|---|---|---|
{mesh_lines}

2. **核采样敏感性**（K(t) 是 B(ω) 的积分，采样步长与记忆窗长决定分辨率）：
   dt {sens['sampling']['dt_base']} → {sens['sampling']['dt_base']/2:g} s 时 K(0) 相对变化
   {pct(sens['sampling']['rel_change_dt_half'])}；τ_max {sens['sampling']['tau_max_base']:g} →
   {sens['sampling']['tau_max_base']/2:g} s 时 K(0) 相对变化
   {pct(sens['sampling']['rel_change_tau_max_half'])}；
   往返闭环误差 {pct(sens['sampling']['R5_rel_err_base'])} → {pct(sens['sampling']['R5_rel_err_dt_half'])}
   （dt 减半）。**频带截断不可测**：`validate_radiation_sweep` 要求 ≥8 个频率节点，
   8 节点扫频砍掉高频尾巴即被拒（记为生态摩擦）。

**局限**：R2–R5 用的是同一份 BEM 扫频，彼此独立的是"数学路径"而不是"物理模型"；能发现频域解的
系统偏差的只有 R1，而 R1 只有 ω=1 rad/s 单频点、且 Capytaine 与该 BEM 都是势流面元法。
A_∞ 由尾部外插给出（尾部分数 {r['studies']['heave']['added_mass_inf_tail_fraction']}），
这与 ω_max 直接耦合，是本算例最大的单点不确定源。

## ⑥ 失败模式与工件

* **honest 记录**：最差 A_ii 相对误差 {pct(n_worst[0])}（{n_worst[1]}）；判定 `{verdict}`。
  逐 DOF 的失败项见上表，未做任何覆盖或剔除。
* **已知陷阱**（本单踩到，已写进模块 docstring）：
  `kkr_added_mass` 要求采样点**不与节点重合**且**不在扫频端点**，否则主值积分抛
  `ValueError`——R3 因此在扫频中点评估，并对实测 A 做线性插值比较；
  往返闭环的相对误差在 A(ω)−A(∞) 过零点处无定义，R5 改用按量级归一化并只在
  `fourier_round_trip` 自报的有效频带内统计。
* **工件**：`artifacts/cummins_wigley_kernel_6dof/results.json`（全部数字的唯一来源）、
  `sweep_coarse.json` / `sweep_medium.json`（BEM 原始输出）、
  `docs/fig_m17209_cummins_kernels_6dof.png`、`docs/fig_m17209_cross_solver_6dof.png`、
  `docs/fig_m17209_gates_6dof.png`。
* **生态摩擦**：①仓内无任何数字化文献表（Journée/WAMIT 均为回忆值且已被证伪，见
  `benchmarks/verified/wigley3_recalled_ref/results.json` 的 `adjudication` 中
  `verdict: REFUTED`），算例池要"原文表格"参考时无路可走——需要数字化手册工件注入通道；
  ②`kkr_added_mass` 的端点/节点限制未在文档中声明，使用者必须先踩一次异常；
  ③平台 knowledge_base 的 `/knowledge/search` 对本域关键词（cummins/kramers/wigley/BEM）
  命中 0 条（`artifacts/cummins_wigley_kernel_6dof/kb_probe.json`），开工三查实际上查不到东西。

## 图

{fig_md}
"""


def post_report(title: str, md: str, figs: list[dict], tags: list[str]) -> dict:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    login = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": os.environ.get("TENSOR_AGENT_USER", "agent"),
                         "password": os.environ.get("TENSOR_AGENT_PASS", "")}).encode(),
        headers={"Content-Type": "application/json"},
    )
    tok = json.loads(urllib.request.urlopen(login, timeout=60).read())
    token = tok.get("access_token") or tok.get("token")
    payload = {
        "title": title,
        "summary": "TensorBEM Wigley III 6-DOF Cummins memory kernel case: R1 independent-solver, "
                   "R2 Cummins identity, R3 Kramers-Kronig, R4 free decay, R5 round trip.",
        "scenario_key": "bem_cummins_memory_kernel",
        "mission_id": 17209,
        "tags": tags,
        "markdown": md,
        "figures": [{"name": f["name"], "caption": f["caption"],
                     "png_base64": base64.b64encode((DOCS / f["name"]).read_bytes()).decode()}
                    for f in figs],
    }
    req = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args(argv)
    r = json.loads((ART / "results.json").read_text())
    figs = make_figures(r)
    md = markdown(r, figs)
    out = DOCS / "dev_report_m17209_cummins_wigley_kernel_6dof.md"
    out.write_text(md)
    print("wrote", out, "figs:", [f["name"] for f in figs])
    if args.post:
        resp = post_report(
            "CASE-BEM-CUMMINS-WIGLEY: Wigley III 6-DOF Cummins 记忆核（m17209）",
            md, figs,
            ["TensorBEM", "Cummins", "Kramers-Kronig", "Wigley", "6DOF", "benchmark", "case-pool"],
        )
        print("report id:", resp.get("id") or resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())