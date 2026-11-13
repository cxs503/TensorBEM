#!/usr/bin/env python3
"""Independent (third-party) axis for the Hulme-hemisphere *excitation* forces.

Why this exists
---------------
``benchmarks/verified/casepool_bem_hulme_hemisphere/`` verifies the 3-D
free-surface BEM against :func:`tensorbem.analytic.hulme_hemisphere_radiation`,
the wave-free multipole series of Hulme (1982).  That bundle registers two
honest limitations:

* the reference series and the solver **share**
  ``tensorbem.free_surface3d.wave_function_3d``, so agreement alone cannot
  catch a defect in that kernel;
* the *excitation* forces of the analytic reference are produced by Haskind
  reciprocity from the radiation Kochin function of the very same series, and
  the only cross-check recorded for them is again the same solver
  (``benchmarks/verified/hulme_diffraction_3d``, itself Haskind-based).  The
  printed tables of Hulme (1982) are paywalled, so no printed digit is
  available as a third opinion.

This producer closes that gap with a code that shares no line of TensorBEM:
**capytaine** (successor of Nemoh), which discretises the same linearised
free-surface problem with its own panel method.  It is the independent axis
already used for the radiation coefficients in
``benchmarks/verified/capytaine_crosscheck`` -- here applied to the quantity
that had none.

Two independent axes are produced
---------------------------------
1. **Closed-form Froude-Krylov anchor** (:func:`fk_closed_form`).  For the
   floating hemisphere the FK force integrates in closed form (1-D quadratures,
   machine precision), which pins the normalisation and the
   time/heading convention of the capytaine side independently of any mesh.
2. **Total excitation force from capytaine** = ``froude_krylov_force(problem)``
   + ``DiffractionResult.forces``.  In capytaine 3.x ``DiffractionResult.forces``
   is the **scattering** force only; forgetting the FK term is the trap this
   script documents in code.  A capytaine mesh ladder bounds the referee's own
   discretisation uncertainty instead of assuming it.

Pre-registered gates (never relaxed after the run)
--------------------------------------------------
* G1/G2: ``|Fx|`` / ``|Fz|`` of capytaine-fine vs the analytic reference within
  5 % for every ka of the case band;
* G3: phase agreement within 5 deg on both components (where a conjugation or
  heading-sign error would hide);
* G4: capytaine self-convergence -- finest-vs-second drift at least 3x below
  the coarsest-level error;
* G5: cross-code, TensorBEM (1024 panels, from ``results.json``) vs
  capytaine-fine within 5 % amplitude and 5 deg phase;
* G6: capytaine FK force reproduces :func:`fk_closed_form` within 2 % on the
  finest mesh at every ka (normalisation/convention guard).

capytaine is an optional external dependency and is **not** a test dependency:
the numbers are frozen into ``excitation_xval_results.json`` and the test suite
reads that file.  Usage (needs capytaine importable)::

    python benchmarks/verified/casepool_bem_hulme_hemisphere/excitation_capytaine_xval.py \
        --write-doc
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CASE_DIR = Path("benchmarks/verified/casepool_bem_hulme_hemisphere")
REFERENCE_FILE = CASE_DIR / "reference.json"
CASE_RESULTS_FILE = CASE_DIR / "results.json"
RESULT_FILE = CASE_DIR / "excitation_xval_results.json"
DOC_PATH = Path("docs/dev_report_m17796_hulme_excitation_xval.md")
FIGURE_PATH = Path("docs/fig_hulme_excitation_xval.png")
PHASE_FIGURE_PATH = Path("docs/fig_hulme_excitation_xval_phase.png")

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
AMPLITUDE = 1.0
#: The ka band of the case-pool entry (identical to results.json / reference.json).
KA_GRID: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
#: capytaine mesh ladder: (meridian, parallel) divisions of the sphere before
#: ``immersed_part()`` clipping.  Four rungs so the referee's convergence is
#: measured, not asserted.
LEVELS: tuple[tuple[int, int], ...] = ((12, 24), (20, 40), (28, 56), (36, 72))
#: TensorBEM side: the case mesh of the case-pool entry.
TBEM_SUBDIVISIONS = 4
TOLERANCE = 0.05
PHASE_TOLERANCE_DEG = 5.0
FK_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# closed-form anchor
# ---------------------------------------------------------------------------
def fk_closed_form(ka: float, radius: float = RADIUS) -> dict[str, complex]:
    """Froude-Krylov force on a floating hemisphere, per ``rho*g*A*a^2``.

    Incident potential in the convention of the case under test
    (``exp(-i omega t)``, waves propagating along ``+x``)::

        phi_I = (g A / i omega) exp(k z + i k x - i omega t)

    so ``p = -i omega rho phi_I = -rho g A exp(kz + ikx)`` and the FK force is
    ``-∮ p n dS`` over the wetted hemisphere ``r = a, z <= 0``.  Both components
    reduce to one-dimensional quadratures (the azimuthal integral is
    ``int_0^{2pi} exp(i alpha sin(th) cos(phi)) d(phi) = 2 pi J0(alpha sin th)``
    for heave and ``... cos(phi) d(phi) = 2 pi i J1(alpha sin th)`` for surge)::

        Fz/(rho g A a^2) = -2*pi * int_{-1}^{0} u exp(alpha u)
                                    J0(alpha sqrt(1-u^2)) du
        Fx/(rho g A a^2) = -2*pi*i * int_{pi/2}^{pi} sin(th)^2
                                    exp(alpha cos th) J1(alpha sin th) d(th)

    with ``alpha = k a``.  Returned as complex numbers in the same
    ``exp(-i omega t)`` convention as the reference and the solver.
    """
    from scipy.special import j1  # local: only the producer needs scipy

    alpha = float(ka) * float(radius)

    from scipy.special import j0

    nodes, weights = np.polynomial.legendre.leggauss(240)
    # heave: map [-1, 1] -> [-1, 0]; note the azimuthal integral contributes
    # 2*pi*J0(alpha*sin th) -- dropping J0 (i.e. using 1) is the trap that the
    # FK normalisation guard G6 exists to catch (up to 2x wrong at ka = 2).
    u = 0.5 * nodes - 0.5
    heave = -2.0 * math.pi * float(
        np.sum(
            0.5
            * weights
            * u
            * np.exp(alpha * u)
            * j0(alpha * np.sqrt(np.clip(1.0 - u * u, 0.0, None)))
        )
    )

    # surge: map [-1, 1] -> [pi/2, pi]
    th = 0.25 * math.pi * nodes + 0.75 * math.pi
    w = 0.25 * math.pi * weights
    integrand = (
        np.sin(th) ** 2 * np.exp(alpha * np.cos(th)) * j1(alpha * np.sin(th))
    )
    surge = -2.0 * math.pi * 1j * complex(np.sum(w * integrand))
    return {"surge": complex(surge), "heave": complex(heave)}


# ---------------------------------------------------------------------------
# capytaine side
# ---------------------------------------------------------------------------
def _provenance(capytaine: Any) -> dict[str, Any]:
    return {
        "tool": "capytaine",
        "version": getattr(capytaine, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "density": DENSITY,
        "gravity": GRAVITY,
        "amplitude": AMPLITUDE,
        "water_depth": "infinite",
        "wave_direction_deg": 0.0,
        "force_definition": (
            "froude_krylov_force(problem) + DiffractionResult.forces; in "
            "capytaine 3.x DiffractionResult.forces is the scattering force only"
        ),
        "geometry": "capytaine.mesh_sphere(radius=1, center=(0,0,0)).immersed_part()",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def _solve_level(capytaine: Any, resolution: Sequence[int]) -> dict[str, Any]:
    import warnings

    mesh = capytaine.mesh_sphere(
        radius=RADIUS, center=(0.0, 0.0, 0.0), resolution=tuple(resolution)
    ).immersed_part()
    body = capytaine.FloatingBody(mesh=mesh)
    body.add_translation_dof(name="Surge")
    body.add_translation_dof(name="Heave")
    solver = capytaine.BEMSolver()
    norm = DENSITY * GRAVITY * AMPLITUDE * RADIUS ** 2
    rows: dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ka in KA_GRID:
            problem = capytaine.DiffractionProblem(
                body=body,
                omega=math.sqrt(GRAVITY * ka),
                wave_direction=0.0,
                water_depth=np.inf,
                rho=DENSITY,
                g=GRAVITY,
            )
            result = solver.solve(problem)
            fk = capytaine.bem.airy_waves.froude_krylov_force(problem)
            entry: dict[str, Any] = {}
            for dof, key in (("Surge", "surge"), ("Heave", "heave")):
                scattering = complex(result.forces[dof])
                fk_c = complex(fk[dof])
                total = fk_c + scattering
                entry[key] = {
                    "total_magnitude": abs(total) / norm,
                    "total_phase_deg": math.degrees(
                        math.atan2(total.imag, total.real)
                    ),
                    "total_dimless": [total.real / norm, total.imag / norm],
                    "fk_magnitude": abs(fk_c) / norm,
                    "fk_phase_deg": math.degrees(math.atan2(fk_c.imag, fk_c.real)),
                    "scattering_dimless": [
                        scattering.real / norm,
                        scattering.imag / norm,
                    ],
                }
            rows[f"{ka:g}"] = entry
    return {
        "resolution": [int(resolution[0]), int(resolution[1])],
        "n_faces": int(mesh.nb_faces),
        "n_vertices": int(mesh.nb_vertices),
        "values": rows,
    }


def run_study(capytaine: Any) -> dict[str, Any]:
    return {
        "provenance": _provenance(capytaine),
        "levels": [_solve_level(capytaine, level) for level in LEVELS],
    }


# ---------------------------------------------------------------------------
# comparison and gates
# ---------------------------------------------------------------------------
def _rel(a: float, b: float) -> float:
    return (a - b) / b


def _phase_delta(phase: float, reference: float) -> float:
    """Wrapped phase difference, degrees, in (-180, 180]."""
    return (phase - reference + 180.0) % 360.0 - 180.0


def _load_inputs() -> tuple[dict[str, Any], dict[str, float]]:
    reference = json.loads(REFERENCE_FILE.read_text())
    case = json.loads(CASE_RESULTS_FILE.read_text())
    tbem: dict[str, Any] = {}
    for row in case["rows"]:
        if int(row.get("subdivisions", -1)) == TBEM_SUBDIVISIONS:
            if row.get("kind") == "band":
                tbem[f"{float(row['ka']):g}"] = row
    return reference, tbem


def compare(study: dict[str, Any]) -> dict[str, Any]:
    reference, tbem = _load_inputs()
    fine = study["levels"][-1]
    coarse = study["levels"][0]
    second = study["levels"][-2]
    rows = []
    for ka in KA_GRID:
        key = f"{ka:g}"
        ref = reference["reference"][key]
        tb = tbem[key]
        got = fine["values"][key]
        second_v = second["values"][key]
        coarse_v = coarse["values"][key]
        closed = fk_closed_form(ka, RADIUS)
        row: dict[str, Any] = {
            "ka": ka,
            "reference": {
                "Fx": ref["Fx_dimless"],
                "phase_Fx_deg": ref["phase_Fx_deg"],
                "Fz": ref["Fz_dimless"],
                "phase_Fz_deg": ref["phase_Fz_deg"],
            },
            "capytaine_fine": {
                "Fx": got["surge"]["total_magnitude"],
                "phase_Fx_deg": got["surge"]["total_phase_deg"],
                "Fz": got["heave"]["total_magnitude"],
                "phase_Fz_deg": got["heave"]["total_phase_deg"],
            },
            "tensorbem_case": {
                "Fx": tb["Fx_dimless"],
                "phase_Fx_deg": tb["phase_Fx_deg"],
                "Fz": tb["Fz_dimless"],
                "phase_Fz_deg": tb["phase_Fz_deg"],
                "panels": tb.get("panels"),
            },
            # fk_closed_form already returns dimensionless values (per rho*g*A*a^2)
            "fk_closed_form": {
                "surge_magnitude": abs(closed["surge"]),
                "surge_phase_deg": math.degrees(
                    math.atan2(closed["surge"].imag, closed["surge"].real)
                ),
                "heave_magnitude": abs(closed["heave"]),
                "heave_phase_deg": math.degrees(
                    math.atan2(closed["heave"].imag, closed["heave"].real)
                ),
            },
            "capytaine_fk": {
                "surge_magnitude": got["surge"]["fk_magnitude"],
                "surge_phase_deg": got["surge"]["fk_phase_deg"],
                "heave_magnitude": got["heave"]["fk_magnitude"],
                "heave_phase_deg": got["heave"]["fk_phase_deg"],
            },
        }
        row["capytaine_vs_reference"] = {
            "Fx_rel": _rel(row["capytaine_fine"]["Fx"], ref["Fx_dimless"]),
            "Fz_rel": _rel(row["capytaine_fine"]["Fz"], ref["Fz_dimless"]),
            "Fx_phase_deg": _phase_delta(
                row["capytaine_fine"]["phase_Fx_deg"], ref["phase_Fx_deg"]
            ),
            "Fz_phase_deg": _phase_delta(
                row["capytaine_fine"]["phase_Fz_deg"], ref["phase_Fz_deg"]
            ),
        }
        row["capytaine_vs_tensorbem"] = {
            "Fx_rel": _rel(row["capytaine_fine"]["Fx"], tb["Fx_dimless"]),
            "Fz_rel": _rel(row["capytaine_fine"]["Fz"], tb["Fz_dimless"]),
            "Fx_phase_deg": _phase_delta(
                row["capytaine_fine"]["phase_Fx_deg"], tb["phase_Fx_deg"]
            ),
            "Fz_phase_deg": _phase_delta(
                row["capytaine_fine"]["phase_Fz_deg"], tb["phase_Fz_deg"]
            ),
        }
        row["capytaine_fk_vs_closed_form"] = {
            "surge_rel": _rel(
                got["surge"]["fk_magnitude"], abs(closed["surge"])
            ),
            "heave_rel": _rel(
                got["heave"]["fk_magnitude"], abs(closed["heave"])
            ),
        }
        row["capytaine_selfconvergence"] = {
            "coarse_Fz_vs_reference_rel": _rel(
                coarse_v["heave"]["total_magnitude"], ref["Fz_dimless"]
            ),
            "coarse_Fx_vs_reference_rel": _rel(
                coarse_v["surge"]["total_magnitude"], ref["Fx_dimless"]
            ),
            "fine_vs_second_drift_Fz_rel": _rel(
                got["heave"]["total_magnitude"],
                second_v["heave"]["total_magnitude"],
            ),
            "fine_vs_second_drift_Fx_rel": _rel(
                got["surge"]["total_magnitude"],
                second_v["surge"]["total_magnitude"],
            ),
        }
        rows.append(row)
    return {"rows": rows}


def gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    g1 = max(abs(r["capytaine_vs_reference"]["Fx_rel"]) for r in rows)
    g2 = max(abs(r["capytaine_vs_reference"]["Fz_rel"]) for r in rows)
    g3 = max(
        max(
            abs(r["capytaine_vs_reference"]["Fx_phase_deg"]),
            abs(r["capytaine_vs_reference"]["Fz_phase_deg"]),
        )
        for r in rows
    )
    drift = max(
        max(
            abs(r["capytaine_selfconvergence"]["fine_vs_second_drift_Fz_rel"]),
            abs(r["capytaine_selfconvergence"]["fine_vs_second_drift_Fx_rel"]),
        )
        for r in rows
    )
    coarse_err = max(
        max(
            abs(r["capytaine_selfconvergence"]["coarse_Fz_vs_reference_rel"]),
            abs(r["capytaine_selfconvergence"]["coarse_Fx_vs_reference_rel"]),
        )
        for r in rows
    )
    g5_amp = max(
        max(
            abs(r["capytaine_vs_tensorbem"]["Fx_rel"]),
            abs(r["capytaine_vs_tensorbem"]["Fz_rel"]),
        )
        for r in rows
    )
    g5_phase = max(
        max(
            abs(r["capytaine_vs_tensorbem"]["Fx_phase_deg"]),
            abs(r["capytaine_vs_tensorbem"]["Fz_phase_deg"]),
        )
        for r in rows
    )
    g6 = max(
        max(
            abs(r["capytaine_fk_vs_closed_form"]["surge_rel"]),
            abs(r["capytaine_fk_vs_closed_form"]["heave_rel"]),
        )
        for r in rows
    )
    criteria = {
        "G1_Fx_amplitude_vs_reference": {
            "criterion": f"max |rel err| < {TOLERANCE:.0%} over the ka band",
            "measured": g1,
            "pass": bool(g1 < TOLERANCE),
        },
        "G2_Fz_amplitude_vs_reference": {
            "criterion": f"max |rel err| < {TOLERANCE:.0%} over the ka band",
            "measured": g2,
            "pass": bool(g2 < TOLERANCE),
        },
        "G3_phase_vs_reference": {
            "criterion": (
                f"max |phase delta| < {PHASE_TOLERANCE_DEG:g} deg on both components"
            ),
            "measured": g3,
            "pass": bool(g3 < PHASE_TOLERANCE_DEG),
        },
        "G4_capytaine_self_convergent": {
            "criterion": (
                "finest-vs-second drift at least 3x below the coarsest-level error"
            ),
            "measured": {"drift": drift, "coarse_error": coarse_err},
            "pass": bool(drift * 3.0 <= coarse_err),
        },
        "G5_tensorbem_vs_capytaine": {
            "criterion": (
                f"amplitude < {TOLERANCE:.0%} and phase < {PHASE_TOLERANCE_DEG:g} deg"
            ),
            "measured": {"amplitude": g5_amp, "phase_deg": g5_phase},
            "pass": bool(g5_amp < TOLERANCE and g5_phase < PHASE_TOLERANCE_DEG),
        },
        "G6_fk_normalisation_guard": {
            "criterion": (
                f"capytaine FK vs closed-form quadrature < {FK_TOLERANCE:.0%}"
            ),
            "measured": g6,
            "pass": bool(g6 < FK_TOLERANCE),
        },
    }
    return {
        "criteria": criteria,
        "verdict": "PASS" if all(c["pass"] for c in criteria.values()) else "FAIL",
    }


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def render_figures(payload: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["comparison"]["rows"]
    ka = [r["ka"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True)
    for ax, comp, label, sym in (
        (axes[0], "Fx", "surge", "x"),
        (axes[1], "Fz", "heave", "z"),
    ):
        ax.plot(
            ka,
            [r["reference"][comp] for r in rows],
            "k-",
            lw=2,
            label="Hulme 1982 wave-free multipole (reference)",
        )
        ax.plot(
            ka,
            [r["capytaine_fine"][comp] for r in rows],
            "o",
            ms=7,
            mfc="none",
            mew=1.6,
            color="tab:red",
            label="capytaine (third-party BEM)",
        )
        ax.plot(
            ka,
            [r["tensorbem_case"][comp] for r in rows],
            "s",
            ms=5,
            color="tab:blue",
            label="TensorBEM (1024 panels)",
        )
        ax.set_ylabel(rf"$|F_{{{sym}}}|\;/\;(\rho g A a^2)$")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{label.capitalize()} excitation force amplitude vs ka")
    axes[1].set_xlabel(r"$ka$")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=130)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(
        ka, [r["reference"]["phase_Fx_deg"] for r in rows], "k--", lw=2,
        label=r"reference phase $F_x$",
    )
    ax.plot(
        ka, [r["capytaine_fine"]["phase_Fx_deg"] for r in rows], "o",
        mfc="none", color="tab:red", label=r"capytaine phase $F_x$",
    )
    ax.plot(
        ka, [r["reference"]["phase_Fz_deg"] for r in rows], "k-", lw=2,
        label=r"reference phase $F_z$",
    )
    ax.plot(
        ka, [r["capytaine_fine"]["phase_Fz_deg"] for r in rows], "s",
        color="tab:green", label=r"capytaine phase $F_z$",
    )
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel("phase [deg]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(r"Excitation force phase, $\exp(-i\omega t)$ convention")
    fig2.tight_layout()
    fig2.savefig(PHASE_FIGURE_PATH, dpi=130)
    plt.close(fig2)
    return [FIGURE_PATH, PHASE_FIGURE_PATH]


def _fmt(value: float, digits: int = 2, percent: bool = True) -> str:
    if percent:
        return f"{100.0 * value:+.{digits}f}%"
    return f"{value:+.{digits}f}"


def render_doc(payload: dict[str, Any]) -> str:
    rows = payload["comparison"]["rows"]
    crit = payload["verdict"]["criteria"]
    study = payload["study"]
    prov = study["provenance"]
    lines: list[str] = []
    a = lines.append

    a("# CASE-BEM-HULME-HEMISPHERE：绕射激励力的第三方独立轴（capytaine）")
    a("")
    a(f"> 任务 17796 · 算例 `BEM-HULME-HEMISPHERE` · 本文件由 "
      f"`benchmarks/verified/casepool_bem_hulme_hemisphere/excitation_capytaine_xval.py` "
      f"从 `{RESULT_FILE.as_posix()}` 程序化渲染，正文与表格数字均非手抄。")
    a("")
    a(f"**机器裁定：{payload['verdict']['verdict']}**")
    a("")
    a("## ① 案例介绍与验证缺口")
    a("")
    a("半径 a = 1 m 的浮半球（平面与平均自由面齐平，曲面入水）在深水迎浪中求一阶")
    a("波浪激励力 $F_x$（surge）与 $F_z$（heave）的复数值（幅值+相位），无量纲化")
    a("$F/(\\rho g A a^2)$，时间约定 $\\exp(-i\\omega t)$，波沿 $+x$ 传播，波幅 A = 1 m。")
    a("")
    a("算例池原验证（`results.json` / `docs/bem_hulme_hemisphere_case.md`）已登记两条")
    a("诚实的局限：参考解（Hulme 1982 波-free 多极展开的重新推导）与求解器共享")
    a("`tensorbem.free_surface3d.wave_function_3d` 内核；且参考解的激励力是由同一级数的")
    a("辐射 Kochin 经 Haskind 互易得到，实现内部（`hulme_diffraction_3d`）也是 Haskind，")
    a("因此激励力此前**没有真正独立的第三方轴**。Hulme (1982) 印刷表格在付费墙后，")
    a("本沙箱内无原文数字可引（零手抄纪律下不作转录）。")
    a("")
    a("本单补的正是这条轴：用 capytaine（Nemoh 后继，独立面板法实现）在同一频带")
    a("独立求解绕射问题，并加一条闭式 Froude-Krylov 锚（1 维求积，机器精度）钉住")
    a("归一化与约定。")
    a("")
    a("## ② 方法与关键设置")
    a("")
    a(f"- capytaine 版本 `{prov['version']}`；几何 "
      f"`{prov['geometry']}`；水体无限深；ρ = {prov['density']:g} kg/m³，"
      f"g = {prov['gravity']:g} m/s²。")
    a(f"- 激励力定义：{prov['force_definition']}。**踩坑点**：capytaine 3.x 的 "
      f"`DiffractionResult.forces` 只是散射力，不含 Froude-Krylov；只取它会在 "
      f"ka=0.05 处给出 heave 0.09（应为约 π），必须 `+ froude_krylov_force`。")
    a("- capytaine 网格阶梯（resolution × panels）：" +
      "；".join(
          f"({lvl['resolution'][0]},{lvl['resolution'][1]}) → {lvl['n_faces']} 面"
          for lvl in study["levels"]
      ) + "。")
    a(f"- TensorBEM 侧取算例网格 subdivisions={TBEM_SUBDIVISIONS}"
      f"（{rows[0]['tensorbem_case']['panels']} 面板），数字直接读 "
      f"`results.json`，不重算。")
    a("- 参考解：`tensorbem.analytic.hulme_hemisphere_radiation`，144 项（截断阶梯")
    a("  48/96/144，96→144 漂移 6.46e-05，见 `reference.json`）。")
    a(f"- 预注册门：G1/G2 幅值 < 5%；G3 相位 < {PHASE_TOLERANCE_DEG:g}°；G4 capytaine 自收敛；")
    a(f"  G5 两码互比 < 5% / {PHASE_TOLERANCE_DEG:g}°；G6 FK 归一化守卫 < {FK_TOLERANCE:.0%}。")
    a("")
    a("## ③ 计算结果")
    a("")
    a("### 3.1 幅值：参考 vs capytaine（细网格）vs TensorBEM")
    a("")
    a("| ka | ref \\|Fx\\| | capytaine \\|Fx\\| | err | TensorBEM \\|Fx\\| | ref \\|Fz\\| | capytaine \\|Fz\\| | err | TensorBEM \\|Fz\\| |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a(
            f"| {r['ka']:g} | {r['reference']['Fx']:.4f} | "
            f"{r['capytaine_fine']['Fx']:.4f} | "
            f"{_fmt(r['capytaine_vs_reference']['Fx_rel'])} | "
            f"{r['tensorbem_case']['Fx']:.4f} | "
            f"{r['reference']['Fz']:.4f} | {r['capytaine_fine']['Fz']:.4f} | "
            f"{_fmt(r['capytaine_vs_reference']['Fz_rel'])} | "
            f"{r['tensorbem_case']['Fz']:.4f} |"
        )
    a("")
    a("### 3.2 相位：约定一致性（同一 `exp(-iωt)` 约定，无共轭、无 180° 偏移）")
    a("")
    a("| ka | ref φ(Fx) | capy φ(Fx) | Δ | ref φ(Fz) | capy φ(Fz) | Δ |")
    a("|---|---|---|---|---|---|---|")
    for r in rows:
        a(
            f"| {r['ka']:g} | {r['reference']['phase_Fx_deg']:.2f} | "
            f"{r['capytaine_fine']['phase_Fx_deg']:.2f} | "
            f"{_fmt(r['capytaine_vs_reference']['Fx_phase_deg'], 3, False)} | "
            f"{r['reference']['phase_Fz_deg']:.2f} | "
            f"{r['capytaine_fine']['phase_Fz_deg']:.2f} | "
            f"{_fmt(r['capytaine_vs_reference']['Fz_phase_deg'], 3, False)} |"
        )
    a("")
    a("### 3.3 闭式 Froude-Krylov 锚（1 维求积 vs capytaine）")
    a("")
    a("| ka | 闭式 \\|FK_x\\| | capytaine \\|FK_x\\| | err | 闭式 \\|FK_z\\| | capytaine \\|FK_z\\| | err |")
    a("|---|---|---|---|---|---|---|")
    for r in rows:
        a(
            f"| {r['ka']:g} | {r['fk_closed_form']['surge_magnitude']:.4f} | "
            f"{r['capytaine_fk']['surge_magnitude']:.4f} | "
            f"{_fmt(r['capytaine_fk_vs_closed_form']['surge_rel'])} | "
            f"{r['fk_closed_form']['heave_magnitude']:.4f} | "
            f"{r['capytaine_fk']['heave_magnitude']:.4f} | "
            f"{_fmt(r['capytaine_fk_vs_closed_form']['heave_rel'])} |"
        )
    a("")
    a("### 3.4 capytaine 网格自收敛（Fz 相对参考的误差随网格）")
    a("")
    a("| ka | 粗网格 err | 细网格-vs-次细网格漂移 |")
    a("|---|---|---|")
    for r in rows:
        sc = r["capytaine_selfconvergence"]
        a(
            f"| {r['ka']:g} | {_fmt(sc['coarse_Fz_vs_reference_rel'])} | "
            f"{_fmt(sc['fine_vs_second_drift_Fz_rel'])} |"
        )
    a("")
    a("## ④ 图")
    a("")
    a("![excitation amplitude](fig_hulme_excitation_xval.png)")
    a("")
    a("![excitation phase](fig_hulme_excitation_xval_phase.png)")
    a("")
    a("（幅值曲线：解析参考 vs capytaine vs TensorBEM；相位曲线：两码在 ")
    a("`exp(-iωt)` 约定下同相位，任何共轭或来流反号都会表现为 ~180° 偏差。）")
    a("")
    a("## ⑤ 注意点")
    a("")
    a("- capytaine 是可选外部依赖（本机 3.1.dev0，loongarch64 源码构建），**不进测试依赖**：")
    a(f"  数字冻结在 `{RESULT_FILE.as_posix()}`，测试只读冻结件。")
    a("- capytaine 侧 `forces` 的定义陷阱见 ②（散射力 ≠ 总激励力）。")
    a("- 频带止于 ka = 2：ka ≈ 2.58 附近是半球的不规则频率，capytaine 无 CHIEF 抑制；")
    a("  本单不进入该点，因此两码在该处不可比（如实声明，不外推）。")
    a("- 两码共享建模假设（线性势流、无粘、深水、刚体），本轴只能查实现缺陷，")
    a("  不能查建模假设本身；Hulme 印刷数字仍不可得，故参考解仍是"
      "「Hulme 方法重推导 + 自收敛」。")
    a("")
    a("## ⑥ 误差判定")
    a("")
    a("| 门 | 判据 | 实测 | 结论 |")
    a("|---|---|---|---|")
    for name, c in crit.items():
        measured = c["measured"]
        if isinstance(measured, dict):
            text = ", ".join(f"{k}={v:.3e}" for k, v in measured.items())
        elif "phase" in name.lower():
            text = f"{measured:.3e} deg"
        else:
            text = f"{measured:.3e}"
        a(f"| {name} | {c['criterion']} | {text} | {'PASS' if c['pass'] else 'FAIL'} |")
    a("")
    a("关键量逐项 vs 基准：")
    a("")
    worst_fx = max(abs(r["capytaine_vs_reference"]["Fx_rel"]) for r in rows)
    worst_fz = max(abs(r["capytaine_vs_reference"]["Fz_rel"]) for r in rows)
    worst_ph = max(
        max(
            abs(r["capytaine_vs_reference"]["Fx_phase_deg"]),
            abs(r["capytaine_vs_reference"]["Fz_phase_deg"]),
        )
        for r in rows
    )
    a(f"- 激励力幅值最坏误差：surge {100 * worst_fx:.2f}%，heave {100 * worst_fz:.2f}% "
      f"（基准=Hulme 多极重推导参考）；")
    a(f"- 相位最坏偏差：{worst_ph:.3f}°；")
    a(f"- **是否 < 3%？** "
      f"{'是' if max(worst_fx, worst_fz) < 0.03 else '否'}（最坏 {100 * max(worst_fx, worst_fz):.2f}%，"
      f"判据为算例池 5% 门；3% 为报告纪律的参考线）。")
    a("")
    a("## ⑦ 产物")
    a("")
    a("- 生产/分析脚本：`benchmarks/verified/casepool_bem_hulme_hemisphere/excitation_capytaine_xval.py`")
    a(f"- 冻结数字：`{RESULT_FILE.as_posix()}`")
    a(f"- 图：`{FIGURE_PATH.as_posix()}`、`{PHASE_FIGURE_PATH.as_posix()}`")
    a("- 回归测试：`tests/test_hulme_excitation_capytaine_xval.py`（读冻结件，不需 capytaine）")
    a("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-doc", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULT_FILE)
    args = parser.parse_args(argv)

    import capytaine  # the only import of the optional dependency

    study = run_study(capytaine)
    comparison = compare(study)
    verdict = gate(comparison["rows"])
    payload = {
        "case_pool_id": "BEM-HULME-HEMISPHERE",
        "mission_id": 17796,
        "ka_grid": list(KA_GRID),
        "study": study,
        "comparison": comparison,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}  verdict={verdict['verdict']}")
    for name, c in verdict["criteria"].items():
        print(f"  {name}: {'PASS' if c['pass'] else 'FAIL'}  {c['measured']}")
    if args.write_doc:
        render_figures(payload)
        DOC_PATH.write_text(render_doc(payload) + "\n")
        print(f"wrote {DOC_PATH} and figures")
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())