"""Far-field mean drift force of a bottom-mounted cylinder from a real BEM Kochin function.

Node ``p40w4-n02-bem-drift-force-3d`` (m9286).  Driver behind
``docs/drift_far_field_3d.md`` and ``artifacts/drift_far_field_3d/study.json``:
every number in the report is written here, nothing is transcribed by hand.

What is measured
----------------
1. ``ka`` scan on a fixed-shape mesh family (depth ``5/ka`` so the mesh shape --
   and therefore the discretisation error -- does not drift with ``ka``), at two
   mesh levels.  For each ``ka`` and level: BEM diffraction solve -> Kochin
   function (:func:`tensorbem.drift_far_field3d.bem_kochin`) -> near-field
   Pinkster estimate, far-field Maruo estimate from that Kochin function, the
   exact MacCamy-Fuchs references (near *and* far), and the optical-theorem
   (energy identity) residual.
2. Panel-doubling convergence of the far-field estimate, per ``ka`` and on a
   three-level ladder at ``ka = 1``.
3. Control-radius scan ``r/R = 5 ... 40`` of the un-reduced radiation-stress
   route on the BEM Kochin function, with an explicit extrapolation-eligibility
   verdict (expected **not** evaluable -- that is the m5242 honest FAIL being
   measured, not overwritten).
4. Local log-log slope of ``F(ka)`` for the BEM far-field route and for the
   exact reference, plus the boundary of the ``(ka)^3`` band.

Usage
-----
    python -m tensorbem.benchmarks.drift_far_field_3d_study \
        --json artifacts/drift_far_field_3d/study.json \
        --figures artifacts/drift_far_field_3d
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..cylinder_drift import far_field_cylinder_drift, near_field_cylinder_drift
from ..drift import (
    bottom_mounted_cylinder_mesh,
    far_field_mean_drift_3d,
    near_field_mean_drift_3d,
)
from ..drift_far_field3d import (
    bem_kochin,
    local_loglog_slope,
    maruo_far_field_drift,
    optical_theorem_residual,
    panel_richardson,
    radius_scan,
    slope_band,
)
from ..free_surface_solver3d import solve_diffraction_3d

DENSITY = 1025.0
GRAVITY = 9.81
RADIUS = 1.0
DEPTH_OVER_KA = 5.0

KA_VALUES = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
LEVELS = (("base", 16, 6, 1.35), ("fine", 32, 12, 1.25))
LADDER_KA = 1.0
LADDER = ((16, 6, 1.35), (24, 9, 1.30), (32, 12, 1.25))
RADII = (5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0)
GATE_NEAR_FAR = 0.03
GATE_ORDER = 1.5
GATE_RICHARDSON = 0.02
GATE_SLOPE = 0.2
GATE_DOUBLING = 0.02
GATE_ENERGY = 1.0e-3


def _rel(value: float, reference: float) -> float:
    return float(abs(value - reference) / abs(reference)) if reference != 0.0 else float("inf")


def run_case(
    ka: float,
    n_theta: int = 16,
    n_z: int = 6,
    growth: float = 1.35,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    n_azimuth: int = 180,
) -> dict:
    """One cylinder case: BEM solve, three drift estimates, energy identity."""
    start = time.perf_counter()
    k = float(ka) / RADIUS
    mesh = bottom_mounted_cylinder_mesh(
        RADIUS, DEPTH_OVER_KA / k, n_theta=n_theta, n_z=n_z, growth=growth
    )
    diffraction = solve_diffraction_3d(
        mesh, k, amplitude=1.0, density=density, gravity=gravity
    )
    koch = bem_kochin(mesh, diffraction, n_azimuth=n_azimuth)

    near = near_field_mean_drift_3d(mesh, diffraction, density=density, gravity=gravity)
    far_repo = far_field_mean_drift_3d(mesh, diffraction, density=density, n_azimuth=n_azimuth)
    maruo = maruo_far_field_drift(
        koch.kochin,
        koch.azimuth,
        k,
        amplitude=1.0,
        density=density,
        gravity=gravity,
    )
    energy = optical_theorem_residual(
        koch.kochin,
        koch.azimuth,
        k,
        radius=RADIUS,
        amplitude=1.0,
        gravity=gravity,
    )
    exact_far = far_field_cylinder_drift(
        float(ka), amplitude=1.0, radius=RADIUS, density=density, gravity=gravity
    )
    exact_near = near_field_cylinder_drift(
        float(ka), amplitude=1.0, radius=RADIUS, density=density, gravity=gravity
    )
    near_x = float(near.mean_force[0])
    far_x = float(maruo["force_x"])
    excess = float(far_repo["force"][0])
    exact_far_x = float(exact_far["mean_drift_force_x"])
    exact_near_x = float(exact_near["mean_drift_force_x"])
    return {
        "ka": float(ka),
        "wavenumber": k,
        "n_theta": int(n_theta),
        "n_z": int(n_z),
        "growth": float(growth),
        "panels": int(mesh.panel_count),
        "depth_over_radius": DEPTH_OVER_KA / float(ka),
        "near_field_x": near_x,
        "near_pressure_term": float(np.sum(near.pressure_term)),
        "near_waterline_term": float(np.sum(near.waterline_term)),
        "far_field_x": far_x,
        "far_field_repo_channel_x": excess,
        "exact_far_x": exact_far_x,
        "exact_near_x": exact_near_x,
        "kochin_integral_total": float(maruo["integral_total"]),
        "kochin_integral_weighted": float(maruo["integral_weighted"]),
        "near_vs_far_gap": _rel(far_x, near_x),
        "far_vs_exact": _rel(far_x, exact_far_x),
        "near_vs_exact": _rel(near_x, exact_near_x),
        "near_route_error": _rel(near_x, exact_near_x),
        "far_route_error": _rel(far_x, exact_far_x),
        "repo_channel_vs_maruo": _rel(excess, far_x),
        "azimuth_rel_change": koch.azimuth_rel_change,
        "kochin_max_abs": float(np.abs(koch.kochin).max()),
        "solver_relative_residual": koch.solver_relative_residual,
        "energy_residual": energy["residual_relative"],
        "energy_integral": energy["integral_total"],
        "energy_predicted": energy["predicted"],
        "seconds": time.perf_counter() - start,
    }


def radius_study(
    ka: float = LADDER_KA,
    *,
    n_theta: int = 16,
    n_z: int = 6,
    growth: float = 1.35,
    n_azimuth: int = 180,
) -> dict:
    """Control-radius scan of the un-reduced route on a real BEM Kochin function."""
    k = float(ka) / RADIUS
    mesh = bottom_mounted_cylinder_mesh(
        RADIUS, DEPTH_OVER_KA / k, n_theta=n_theta, n_z=n_z, growth=growth
    )
    diffraction = solve_diffraction_3d(mesh, k, amplitude=1.0)
    koch = bem_kochin(mesh, diffraction, n_azimuth=n_azimuth)
    scan = radius_scan(
        koch.kochin, koch.azimuth, k, radii=RADII, amplitude=1.0, n_theta=8192
    )
    exact = far_field_cylinder_drift(
        float(ka), amplitude=1.0, radius=RADIUS
    )["mean_drift_force_x"]
    for row in scan["rows"]:
        row["gap_vs_reduced"] = _rel(row["force_x"], float(exact))
    scan["ka"] = float(ka)
    scan["exact_x"] = float(exact)
    scan["panels"] = int(mesh.panel_count)
    scan["reduced_route_x"] = float(
        maruo_far_field_drift(koch.kochin, koch.azimuth, k, amplitude=1.0)["force_x"]
    )
    return scan


def run_study(
    ka_values=KA_VALUES,
    *,
    n_azimuth: int = 180,
    levels=LEVELS,
    ladder=LADDER,
    ladder_ka: float = LADDER_KA,
) -> dict:
    start = time.perf_counter()
    cases: dict[str, list[dict]] = {}
    for name, n_theta, n_z, growth in levels:
        cases[name] = [
            run_case(ka, n_theta, n_z, growth, n_azimuth=n_azimuth) for ka in ka_values
        ]

    ladder_rows = [
        run_case(ladder_ka, n_theta, n_z, growth, n_azimuth=n_azimuth)
        for (n_theta, n_z, growth) in ladder
    ]
    panels = [row["panels"] for row in ladder_rows]
    far_ladder = [row["far_field_x"] for row in ladder_rows]
    ladder_change = _rel(far_ladder[-1], far_ladder[0])
    ladder_step = (
        _rel(far_ladder[-1], far_ladder[-2]) if len(far_ladder) > 1 else float("nan")
    )
    ladder_near_far = [row["near_vs_far_gap"] for row in ladder_rows]

    radius = radius_study(ladder_ka, n_azimuth=n_azimuth)

    fine = cases[levels[-1][0]]
    base = cases[levels[0][0]]
    doubling_per_ka = {
        f"{case['ka']:.3f}": _rel(fine_row["far_field_x"], case["far_field_x"])
        for case, fine_row in zip(base, fine)
    }
    worst_doubling = max(doubling_per_ka.values())

    bem_slope = local_loglog_slope(
        [c["ka"] for c in fine], [c["far_field_x"] for c in fine]
    )
    exact_slope = local_loglog_slope(
        [c["ka"] for c in fine], [c["exact_far_x"] for c in fine]
    )
    bem_band = slope_band(bem_slope)
    exact_band = slope_band(exact_slope)

    gap_fine = max(c["near_vs_far_gap"] for c in fine)
    gap_base = max(c["near_vs_far_gap"] for c in base)
    energy_fine = max(c["energy_residual"] for c in fine)
    energy_base = max(c["energy_residual"] for c in base)
    far_exact_fine = max(c["far_vs_exact"] for c in fine)
    near_exact_fine = max(c["near_vs_exact"] for c in fine)
    worst_azimuth = max(c["azimuth_rel_change"] for c in fine)
    worst_slope_dev = float(np.max(np.abs(bem_slope["slope"] - 3.0)))
    exact_worst_slope_dev = float(np.max(np.abs(exact_slope["slope"] - 3.0)))
    slope_first = float(bem_slope["slope"][0])
    exact_slope_first = float(exact_slope["slope"][0])

    gates = {
        "near_vs_far_within_3pct_fine_mesh": {
            "target": GATE_NEAR_FAR,
            "measured_fine_max": gap_fine,
            "measured_base_max": gap_base,
            "pass": bool(gap_fine <= GATE_NEAR_FAR),
        },
        "radius_scan_order_ge_1p5_and_richardson_le_2pct": {
            "target_order": GATE_ORDER,
            "target_richardson": GATE_RICHARDSON,
            "measured_order": radius["observed_order"],
            "measured_richardson": radius["richardson_residual_relative"],
            "measured_spread": radius["spread_relative"],
            "evaluable": radius["richardson_evaluable"],
            "pass": bool(
                radius["richardson_evaluable"]
                and radius["observed_order"] >= GATE_ORDER
                and radius["richardson_residual_relative"] <= GATE_RICHARDSON
            ),
        },
        "asymptotic_local_slope_3pm0p2": {
            "target": 3.0,
            "tolerance": GATE_SLOPE,
            "ka": float(bem_slope["ka"][0]),
            "bem_slope": slope_first,
            "exact_slope": exact_slope_first,
            "pass": bool(abs(slope_first - 3.0) <= GATE_SLOPE),
        },
        "full_band_0p2_3_slope_3pm0p2": {
            "target": 3.0,
            "tolerance": GATE_SLOPE,
            "bem_worst_deviation": worst_slope_dev,
            "exact_worst_deviation": exact_worst_slope_dev,
            "bem_band_max_ka": bem_band["band_max_ka"],
            "exact_band_max_ka": exact_band["band_max_ka"],
            "pass": bool(worst_slope_dev <= GATE_SLOPE),
        },
        "panel_doubling_far_field_le_2pct": {
            "target": GATE_DOUBLING,
            "per_ka": doubling_per_ka,
            "ladder_finest_step": ladder_step,
            "ladder_levels": len(far_ladder),
            "ladder_overall": ladder_change,
            "pass": bool(worst_doubling <= GATE_DOUBLING),
        },
        "energy_identity_residual_le_1e-3": {
            "target": GATE_ENERGY,
            "measured_fine_max": energy_fine,
            "measured_base_max": energy_base,
            "pass": bool(energy_fine <= GATE_ENERGY),
        },
    }
    return {
        "node": "p40w4-n02-bem-drift-force-3d",
        "mission": 9286,
        "geometry": {
            "body": "bottom-mounted truncated vertical cylinder",
            "radius_m": RADIUS,
            "depth": "5 / ka (fixed mesh shape across the ka family)",
            "density": DENSITY,
            "gravity": GRAVITY,
            "amplitude_m": 1.0,
            "wave_angle_rad": 0.0,
        },
        "levels": [list(level) for level in levels],
        "ladder": [list(level) for level in ladder],
        "ladder_panels": panels,
        "n_azimuth": int(n_azimuth),
        "cases": cases,
        "ladder_rows": ladder_rows,
        "radius_scan": radius,
        "doubling_per_ka": doubling_per_ka,
        "bem_local_slope": {
            "ka": [float(v) for v in bem_slope["ka"]],
            "slope": [float(v) for v in bem_slope["slope"]],
            "in_band": [bool(v) for v in bem_slope["in_band"]],
            "slope_long_wave": bem_slope["slope_long_wave"],
        },
        "exact_local_slope": {
            "ka": [float(v) for v in exact_slope["ka"]],
            "slope": [float(v) for v in exact_slope["slope"]],
            "in_band": [bool(v) for v in exact_slope["in_band"]],
            "slope_long_wave": exact_slope["slope_long_wave"],
        },
        "bands": {"bem": bem_band, "exact": exact_band},
        "worst": {
            "near_vs_far_gap_fine": gap_fine,
            "near_vs_far_gap_base": gap_base,
            "far_route_error_fine": far_exact_fine,
            "near_route_error_fine": near_exact_fine,
            "energy_residual_fine": energy_fine,
            "energy_residual_base": energy_base,
            "azimuth_rel_change": worst_azimuth,
            "ladder_near_far": ladder_near_far,
        },
        "gates": gates,
        "seconds": time.perf_counter() - start,
    }




# ---------------------------------------------------------------------------
# document block (programmatic injection: no digit in the report is hand typed)
# ---------------------------------------------------------------------------
BEGIN_BLOCK = "<!-- BEGIN GENERATED: drift-far-field-3d -->"
END_BLOCK = "<!-- END GENERATED: drift-far-field-3d -->"


def _pct(value: float, digits: int = 2) -> str:
    return "n/a" if value != value else f"{100.0 * value:.{digits}f}%"


def render_document_block(study: dict) -> str:
    """Markdown tables derived from ``study`` (the committed artifact)."""
    lines: list[str] = [BEGIN_BLOCK, "", "### 生成数字（`artifacts/drift_far_field_3d/study.json`）", ""]
    gate = study["gates"]

    lines.append(
        "**门判据**（`pass` 逐条来自驱动，未手改）："
    )
    lines.append("")
    lines.append("| 判据 | 目标 | 实测 | 通过 |")
    lines.append("|---|---|---:|---|")
    nf = gate["near_vs_far_within_3pct_fine_mesh"]
    lines.append(
        f"| 近场 vs 远场（最细网格） | < 3% | {_pct(nf['measured_fine_max'])} "
        f"(粗网格 {_pct(nf['measured_base_max'])}) | {nf['pass']} |"
    )
    rs = gate["radius_scan_order_ge_1p5_and_richardson_le_2pct"]
    lines.append(
        f"| r 扫描收敛阶 ≥1.5 且 Richardson ≤2% | ≥1.5 / ≤2% | "
        f"order={rs['measured_order']:.3f}、残差={rs['measured_richardson']:.3f}、"
        f"spread={_pct(rs['measured_spread'],1)}、evaluable={rs['evaluable']} | {rs['pass']} |"
    )
    asym = gate["asymptotic_local_slope_3pm0p2"]
    lines.append(
        f"| 渐近带局部斜率 3±0.2 (ka={asym['ka']:.2f}) | 3.0±0.2 | BEM {asym['bem_slope']:.3f} / "
        f"精确 {asym['exact_slope']:.3f} | {asym['pass']} |"
    )
    full = gate["full_band_0p2_3_slope_3pm0p2"]
    lines.append(
        f"| 全带 ka∈[0.2,3] 斜率 3±0.2 | ≤0.2 | BEM 最大偏差 {full['bem_worst_deviation']:.3f} / "
        f"精确 {full['exact_worst_deviation']:.3f}；出带边界 ka≈{full['bem_band_max_ka']:.2f} "
        f"(精确 {full['exact_band_max_ka']:.2f}) | {full['pass']} |"
    )
    dbl = gate["panel_doubling_far_field_le_2pct"]
    lines.append(
        f"| 面板数加倍（16×6 → 32×12）远场变化 | ≤2% | 逐 ka 最大 {_pct(max(dbl['per_ka'].values()))} | {dbl['pass']} |"
    )
    en = gate["energy_identity_residual_le_1e-3"]
    lines.append(
        f"| 能量恒等式残差（最细网格） | ≤1e-3 | {_pct(en['measured_fine_max'],3)} "
        f"(粗网格 {_pct(en['measured_base_max'],3)}) | {en['pass']} |"
    )
    lines.append("")

    lines.append("**ka 扫描（最细网格）**：近场 / 远场 / 精确 / 三条带证据")
    lines.append("")
    lines.append(
        "| ka | 面元 | 近场 F_x (N) | 远场 F_x (N) | 精确远场 (N) | 近/远差 | 远场误差 | 能量残差 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    fine = study["cases"][study["levels"][-1][0]]
    for c in fine:
        lines.append(
            f"| {c['ka']:.2f} | {c['panels']} | {c['near_field_x']:.4g} | {c['far_field_x']:.4g} | "
            f"{c['exact_far_x']:.4g} | {_pct(c['near_vs_far_gap'])} | {_pct(c['far_vs_exact'])} | "
            f"{_pct(c['energy_residual'],3)} |"
        )
    lines.append("")

    lines.append("**近/远场偏差的路线归属**（同网格，对照精确解拆账）")
    lines.append("")
    lines.append("| ka | 近场路线误差 | 远场路线误差 | 近/远差 |")
    lines.append("|---:|---:|---:|---:|")
    for c in fine:
        lines.append(
            f"| {c['ka']:.2f} | {_pct(c['near_route_error'])} | {_pct(c['far_route_error'])} | "
            f"{_pct(c['near_vs_far_gap'])} |"
        )
    lines.append("")

    lines.append("**控制圆柱半径扫描（未约化辐射应力通道，BEM Kochin，ka = %.1f）**" % study["radius_scan"]["ka"])
    lines.append("")
    lines.append("| r/R | kR | F_x (N) | 与约化解之差 | 散射项 (N) | 干扰项 (N) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in study["radius_scan"]["rows"]:
        lines.append(
            f"| {row['radius_factor']:.1f} | {row['kR']:.2f} | {row['force_x']:.5g} | "
            f"{_pct(row['gap_vs_reduced'],1)} | {row['flux_scattered']:.5g} | {row['flux_interference']:.5g} |"
        )
    lines.append("")
    lines.append(
        f"- 约化解（半径无关）: F_x = {study['radius_scan']['reduced_route_x']:.6g} N；"
        f"精确值 = {study['radius_scan']['exact_x']:.6g} N；"
        f"半径族相对散布 = {_pct(study['radius_scan']['spread_relative'],1)}；"
        f"Richardson 可评 = {study['radius_scan']['richardson_evaluable']}"
    )
    lines.append("")

    lines.append("**局部 log-log 斜率** `d ln F / d ln(ka)`")
    lines.append("")
    lines.append("| ka | BEM 斜率 | 精确斜率 | BEM 在 3±0.2 带内 |")
    lines.append("|---:|---:|---:|---|")
    for ka, sb, se, inb in zip(
        study["bem_local_slope"]["ka"],
        study["bem_local_slope"]["slope"],
        study["exact_local_slope"]["slope"],
        study["bem_local_slope"]["in_band"],
    ):
        lines.append(f"| {ka:.2f} | {sb:.3f} | {se:.3f} | {inb} |")
    lines.append("")

    lines.append("**面板阶梯（ka = %.1f）**" % study["ladder_rows"][0]["ka"])
    lines.append("")
    lines.append("| 面元 | 远场 F_x (N) | 近/远差 | 能量残差 |")
    lines.append("|---:|---:|---:|---:|")
    for row in study["ladder_rows"]:
        lines.append(
            f"| {row['panels']} | {row['far_field_x']:.5g} | {_pct(row['near_vs_far_gap'])} | "
            f"{_pct(row['energy_residual'],3)} |"
        )
    lines.append("")
    ladder = study["ladder_rows"]
    panels = [row["panels"] for row in ladder]
    lines.append("**面板阶梯相邻步变化 + 面板轴 Richardson（ka = %.1f）**" % ladder[0]["ka"])
    lines.append("")
    lines.append("| 步 | 面板比 | 远场变化 | 近场变化 |")
    lines.append("|---|---:|---:|---:|")
    for prev, cur in zip(ladder, ladder[1:]):
        lines.append(
            f"| {prev['panels']} → {cur['panels']} | {cur['panels'] / prev['panels']:.2f}× | "
            f"{_pct(abs(cur['far_field_x'] - prev['far_field_x']) / abs(cur['far_field_x']))} | "
            f"{_pct(abs(cur['near_field_x'] - prev['near_field_x']) / abs(cur['near_field_x']))} |"
        )
    lines.append("")
    for label, key, reference in (
        ("远场", "far_field_x", ladder[-1]["exact_far_x"]),
        ("近场", "near_field_x", ladder[-1]["exact_near_x"]),
    ):
        try:
            rich = panel_richardson(panels, [row[key] for row in ladder])
        except ValueError as exc:
            lines.append(f"- {label}轴 Richardson：不可评（{exc}）")
            continue
        lines.append(
            f"- {label}轴 Richardson：观测阶 {rich['observed_order']:.2f}、"
            f"外推值 {rich['extrapolated']:.6g} N、外推自残差 "
            f"{_pct(rich['residual_relative'],3)}、与精确解偏差 "
            f"{_pct(abs(rich['extrapolated'] - reference) / abs(reference),3)}"
        )
    lines.append("")
    lines.append("**审计余量**（最细网格）：" + "、".join(
        [
            f"远场路线最大误差 {_pct(study['worst']['far_route_error_fine'])}",
            f"近场路线最大误差 {_pct(study['worst']['near_route_error_fine'])}",
            f"方位分辨率相对变化 {study['worst']['azimuth_rel_change']:.2e}",
            f"驱动总耗时 {study['seconds']:.0f} s",
        ]
    ))
    lines.append("")
    lines.append(END_BLOCK)
    return "\n".join(lines)


def inject_document(study: dict, doc_path: Path) -> None:
    """Replace the generated block of ``doc_path`` with the block from ``study``."""
    text = doc_path.read_text()
    start = text.index(BEGIN_BLOCK)
    stop = text.index(END_BLOCK) + len(END_BLOCK)
    doc_path.write_text(text[:start] + render_document_block(study) + text[stop:])



# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(study: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # 1. radius scan
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rs = [row["radius_factor"] for row in study["radius_scan"]["rows"]]
    fs = [row["force_x"] for row in study["radius_scan"]["rows"]]
    ax.plot(rs, fs, "o-", label="un-reduced control-circle flux (BEM Kochin)")
    ax.axhline(study["radius_scan"]["exact_x"], color="k", ls="--",
               label="reduced Maruo / exact (radius free)")
    ax.set_xlabel("control radius  r / R")
    ax.set_ylabel("mean drift force $F_x$ (N)")
    ax.set_title(
        "Finite-radius route does not converge\n(spread %.0f%% over r/R = 5..40, ka = %.1f)"
        % (100.0 * study["radius_scan"]["spread_relative"], study["radius_scan"]["ka"])
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_radius_scan.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # 2. local slope
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(study["bem_local_slope"]["ka"], study["bem_local_slope"]["slope"], "o-",
            label="BEM Kochin far field")
    ax.plot(study["exact_local_slope"]["ka"], study["exact_local_slope"]["slope"], "s--",
            label="exact MacCamy-Fuchs reference")
    ax.axhspan(2.8, 3.2, color="green", alpha=0.12, label=r"$3 \pm 0.2$ band")
    ax.set_xscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"local slope  d ln $F$ / d ln(ka)")
    ax.set_title("Local log-log slope of the cylinder drift force")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_local_slope.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # 3. near / far / exact per ka (fine mesh)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fine = study["cases"][study["levels"][-1][0]]
    ka = [c["ka"] for c in fine]
    ax.plot(ka, [c["near_field_x"] for c in fine], "^-", label="near field (Pinkster, BEM)")
    ax.plot(ka, [c["far_field_x"] for c in fine], "o-", label="far field (BEM Kochin, Maruo)")
    ax.plot(ka, [c["exact_far_x"] for c in fine], "s--", label="exact MacCamy-Fuchs (far)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel("mean drift force $F_x$ (N)")
    ax.set_title("Estimators of the same quantity, finest mesh (%d panels)"
                 % fine[0]["panels"])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_estimators.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # 4. energy identity residual + near/far gap vs ka
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(ka, [c["energy_residual"] for c in fine], "o-", label="energy identity residual")
    ax.plot(ka, [c["near_vs_far_gap"] for c in fine], "^-", label="near vs far gap")
    ax.axhline(1.0e-3, color="r", ls=":", label="1e-3 gate")
    ax.axhline(0.03, color="k", ls=":", label="3% gate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel("relative residual")
    ax.set_title("Audit residuals on the finest mesh")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_residuals.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("artifacts/drift_far_field_3d/study.json"))
    parser.add_argument("--figures", type=Path, default=Path("artifacts/drift_far_field_3d"))
    parser.add_argument("--n-azimuth", type=int, default=180)
    parser.add_argument("--inject-docs", type=Path, default=None)
    args = parser.parse_args(argv)
    study = run_study(n_azimuth=args.n_azimuth)
    study["figures"] = make_figures(study, args.figures)
    if args.inject_docs is not None and args.inject_docs.exists():
        inject_document(study, args.inject_docs)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(study, indent=2, default=float))
    print(json.dumps(study["gates"], indent=2, default=float))
    print(f"wrote {args.json} ({study['seconds']:.1f} s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())