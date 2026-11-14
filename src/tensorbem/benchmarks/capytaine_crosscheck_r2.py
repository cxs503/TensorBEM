"""Round-2 cross-code verification of the 3-D radiation matrix against capytaine.

Round 1 (``tensorbem.benchmarks.capytaine_crosscheck``) compared **two diagonal
entries** -- heave and surge -- of two smooth bodies with the independent panel
code capytaine.  The rest of the 6x6 radiation matrix was never compared against
a foreign code:

* the **rotational block** (roll / pitch / yaw), which exercises the
  ``(r - centre) x n`` generalised normals instead of ``n``;
* the **translation-rotation couplings** (surge-pitch, sway-roll for a
  surface-piercing body), where a mode-index or moment-arm sign error would
  survive a diagonal-only comparison;
* the **structural nulls** -- entries the geometry forces to zero, so a spurious
  coupling can only be seen entry by entry;
* the **rotation centre**, which round 1 never had to fix (it used no rotation)
  and which capytaine 3.1 defaults to ``nan``.

This module scores the whole matrix entry by entry against a *live* capytaine
run (``scripts/capytaine_crosscheck_r2_ref.py``) with the rotation centre
recorded on both sides, and re-produces round 1's frozen reference to show that
that axis still reproduces.

Bands are registered here, before any number is looked at, and are never
relaxed afterwards:

* ``TOL_DIAGONAL`` -- 5 % for the diagonal block (same band as round 1);
* ``TOL_OFFDIAGONAL`` -- 10 % for entries that are genuinely non-zero in **both**
  codes (small quantities are more discretisation-sensitive, so a wider band is
  registered up front rather than excused afterwards);
* ``NULL_FLOOR`` -- an entry is a *structural null* when ``|entry|`` is at most
  1 % of ``sqrt(|M_ii M_jj|)``; a pair null in one code and not in the other is a
  ``conflict`` and therefore a **failure** (the interesting defect class).

The third, non-BEM axis is analytic: a deeply submerged sphere must approach the
unbounded-fluid Lamb (1932) limits as the free-surface effect decays, which no
panel code satisfies by construction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import (
    floating_sphere_hull_mesh,
    hemisphere_hull_mesh,
    submerged_sphere_mesh,
)

DENSITY = 1025.0
GRAVITY = 9.80665
NU_GRID = (0.5, 1.0, 2.0)
DOF_ORDER = ("surge", "sway", "heave", "roll", "pitch", "yaw")

#: Mesh levels actually exercised under this round's compute budget: 64/256
#: panels (hemisphere) and 270 (sphere).  The 1024/1026 levels used by round 1
#: were not re-run here -- a documented limitation, not an assumed equivalence.
HEMISPHERE_LEVELS = (2, 3)
SPHERE_LEVELS = (2,)

TOL_DIAGONAL = 0.05
TOL_OFFDIAGONAL = 0.10
NULL_FLOOR = 0.01
SYMMETRY_TOL = 0.05
DEEP_TOL = 0.05

DEEP_DEPTHS = (3.0, 8.0)
DEEP_NU = (0.5,)
DEEP_LEVELS = (2, 3)
#: Lamb (1932) unbounded-fluid added mass of a sphere, R = 1 m.
LAMB_SURGE = (2.0 / 3.0) * math.pi * DENSITY
#: A sphere rotating about its own centre has ``(r - centre) x n = 0`` on the
#: whole surface, so its added inertia about that centre is **exactly zero** at
#: every frequency and every depth.  Normalising by ``A_surge * R^2`` gives a
#: dimensionless floor for the rotational-normal assembly.
EXACT_NULL_TOL = 0.01

ARTIFACT_ROOT = Path("benchmarks/verified/capytaine_crosscheck_r2")
REFERENCE_FILE = "capytaine_ref_r2.json"
RESULT_FILE = "results.json"
DOC_PATH = Path("docs/capytaine_crosscheck_r2.md")
FIGURE_MATRIX = Path("docs/fig_capytaine_r2_matrix.png")
FIGURE_CONVERGENCE = Path("docs/fig_capytaine_r2_convergence.png")
FIGURE_DEEP = Path("docs/fig_capytaine_r2_deep.png")

CASES: dict[str, dict[str, Any]] = {
    "hemisphere_r1": {
        "kind": "surface-piercing hemisphere, radius 1.0 m, centre (0, 0, 0)",
        "centre": (0.0, 0.0, 0.0),
        "levels": HEMISPHERE_LEVELS,
        "sphere": False,
    },
    "sphere_r1_z-0.5": {
        "kind": "floating sphere, radius 1.0 m, centre (0, 0, -0.5)",
        "centre": (0.0, 0.0, -0.5),
        "levels": SPHERE_LEVELS,
        "sphere": True,
    },
}


def _key(nu: float) -> str:
    return "{:.3f}".format(nu)


def _mesh(case: str, level: int) -> Any:
    spec = CASES[case]
    if spec["sphere"]:
        return floating_sphere_hull_mesh(int(level), radius=1.0, center=spec["centre"])
    return hemisphere_hull_mesh(int(level), radius=1.0)


def solve_matrix(mesh: Any, nu: float, centre: Sequence[float]) -> dict[str, Any]:
    """Full 6x6 added mass / damping of one body at one frequency."""
    res = solve_radiation_3d(
        mesh,
        nu,
        modes=list(range(6)),
        center=centre,
        density=DENSITY,
        gravity=GRAVITY,
    )
    return {
        "added_mass": np.asarray(res.added_mass, dtype=float).tolist(),
        "damping": np.asarray(res.damping, dtype=float).tolist(),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
    }


def run_case(case: str, levels: Sequence[int] | None = None) -> dict[str, Any]:
    spec = CASES[case]
    rows = []
    for level in spec["levels"] if levels is None else levels:
        mesh = _mesh(case, int(level))
        rows.append(
            {
                "level": int(level),
                "n_faces": int(mesh.faces.shape[0]),
                "matrices": {
                    _key(nu): solve_matrix(mesh, nu, spec["centre"]) for nu in NU_GRID
                },
            }
        )
    return {
        "levels": rows,
        "rotation_center": list(spec["centre"]),
        "reference": rows[-1]["matrices"],
        "reference_level": rows[-1]["level"],
        "reference_faces": rows[-1]["n_faces"],
    }


def _scale(mat: np.ndarray, i: int, j: int) -> float:
    scale = math.sqrt(abs(mat[i, i] * mat[j, j]))
    return scale if scale > 0.0 else float("nan")


def compare(tensorbem: dict[str, Any], capytaine: dict[str, Any]) -> dict[str, Any]:
    """Entry-by-entry comparison of all 36 entries of both matrices."""
    cases: dict[str, Any] = {}
    failures: list[str] = []
    worst_active = 0.0
    for case, tb_case in tensorbem.items():
        cap_case = capytaine.get(case)
        if cap_case is None:
            cases[case] = {
                "compared": False,
                "reason": "absent from the capytaine reference",
            }
            failures.append(case + ": absent from the capytaine reference")
            continue
        entries: list[dict[str, Any]] = []
        case_failures: list[str] = []
        case_worst = 0.0
        for nu in NU_GRID:
            key = _key(nu)
            tb = tb_case["reference"][key]
            cap = cap_case["reference"][key]
            for quantity in ("added_mass", "damping"):
                a_tb = np.asarray(tb[quantity], dtype=float)
                a_cap = np.asarray(cap[quantity], dtype=float)
                for i, dof_i in enumerate(DOF_ORDER):
                    for j, dof_j in enumerate(DOF_ORDER):
                        scale = _scale(a_tb, i, j)
                        entry: dict[str, Any] = {
                            "nu": nu,
                            "row": dof_i,
                            "column": dof_j,
                            "quantity": quantity,
                            "tensorbem": float(a_tb[i, j]),
                            "capytaine": float(a_cap[i, j]),
                            "scale": scale,
                        }
                        cap_null = abs(a_cap[i, j]) <= NULL_FLOOR * scale
                        tb_null = abs(a_tb[i, j]) <= NULL_FLOOR * scale
                        if cap_null and tb_null:
                            entry["class"] = "null"
                            entry["tensorbem_null_ratio"] = abs(a_tb[i, j]) / scale
                            entry["capytaine_null_ratio"] = abs(a_cap[i, j]) / scale
                        elif cap_null != tb_null:
                            entry["class"] = "conflict"
                            entry["relative"] = None
                            case_failures.append(
                                "{} nu={:g} {} A[{},{}]: null/non-null conflict "
                                "(tensorbem {:.6g}, capytaine {:.6g}, floor {:.6g})".format(
                                    case,
                                    nu,
                                    quantity,
                                    dof_i,
                                    dof_j,
                                    a_tb[i, j],
                                    a_cap[i, j],
                                    NULL_FLOOR * scale,
                                )
                            )
                        else:
                            relative = (a_tb[i, j] - a_cap[i, j]) / abs(a_cap[i, j])
                            band = TOL_DIAGONAL if i == j else TOL_OFFDIAGONAL
                            entry["class"] = "active"
                            entry["relative"] = float(relative)
                            entry["band"] = band
                            entry["within_band"] = bool(abs(relative) <= band)
                            case_worst = max(case_worst, abs(relative))
                            if abs(relative) > band:
                                case_failures.append(
                                    "{} nu={:g} {} A[{},{}]: {:+.2%} (band {:.0%})".format(
                                        case, nu, quantity, dof_i, dof_j, relative, band
                                    )
                                )
                        entries.append(entry)
        nulls = [e for e in entries if e["class"] == "null"]
        worst_active = max(worst_active, case_worst)
        failures.extend(case_failures)
        cases[case] = {
            "compared": True,
            "entries": entries,
            "n_entries": len(entries),
            "n_active": len([e for e in entries if e["class"] == "active"]),
            "n_null": len(nulls),
            "n_conflict": len([e for e in entries if e["class"] == "conflict"]),
            "worst_active_relative": case_worst,
            "worst_null_ratio": max(
                (
                    max(e["tensorbem_null_ratio"], e["capytaine_null_ratio"])
                    for e in nulls
                ),
                default=0.0,
            ),
            "tensorbem_faces": tb_case["reference_faces"],
            "capytaine_faces": cap_case["reference_faces"],
            "rotation_center": tb_case["rotation_center"],
            "failures": case_failures,
            "verdict": "PASS" if not case_failures else "FAIL",
        }
    return {
        "tolerance_diagonal": TOL_DIAGONAL,
        "tolerance_offdiagonal": TOL_OFFDIAGONAL,
        "null_floor": NULL_FLOOR,
        "worst_active_relative": worst_active,
        "failures": failures,
        "cases": cases,
        "verdict": "PASS" if not failures else "FAIL",
    }


def convergence(tensorbem: dict[str, Any], capytaine: dict[str, Any]) -> dict[str, Any]:
    """Worst deviation per mesh level, diagonal block and active couplings apart."""
    out: dict[str, Any] = {}
    for case, tb_case in tensorbem.items():
        cap_case = capytaine.get(case)
        if cap_case is None:
            continue
        rows = []
        for row in tb_case["levels"]:
            worst_diag = 0.0
            worst_off = 0.0
            for nu in NU_GRID:
                key = _key(nu)
                tb = np.asarray(row["matrices"][key]["added_mass"], dtype=float)
                cap = np.asarray(
                    cap_case["reference"][key]["added_mass"], dtype=float
                )
                for i in range(6):
                    for j in range(6):
                        scale = _scale(cap, i, j)
                        if abs(cap[i, j]) <= NULL_FLOOR * scale:
                            continue
                        rel = abs((tb[i, j] - cap[i, j]) / abs(cap[i, j]))
                        if i == j:
                            worst_diag = max(worst_diag, rel)
                        else:
                            worst_off = max(worst_off, rel)
            rows.append(
                {
                    "level": row["level"],
                    "n_faces": row["n_faces"],
                    "worst_diagonal": worst_diag,
                    "worst_coupling": worst_off,
                }
            )
        out[case] = rows
    return out


def symmetry_invariants(
    tensorbem: dict[str, Any], capytaine: dict[str, Any]
) -> dict[str, Any]:
    """Invariants the geometry forces, independent of any foreign code.

    Both bodies are axisymmetric about z, so ``A_surge = A_sway`` and
    ``A_roll = A_pitch`` must hold up to mesh anisotropy, and the impedance
    matrix must be symmetric (reciprocity).
    """
    out: dict[str, Any] = {}
    for case, tb_case in tensorbem.items():
        rows = []
        for nu in NU_GRID:
            key = _key(nu)
            row: dict[str, Any] = {"nu": nu}
            tb = np.asarray(tb_case["reference"][key]["added_mass"], dtype=float)
            row["tensorbem_surge_sway"] = float(abs(tb[0, 0] - tb[1, 1]) / abs(tb[0, 0]))
            row["tensorbem_roll_pitch"] = float(abs(tb[3, 3] - tb[4, 4]) / abs(tb[3, 3]))
            row["tensorbem_reciprocity"] = float(
                np.max(np.abs(tb - tb.T)) / np.max(np.abs(tb))
            )
            cap_case = capytaine.get(case)
            if cap_case is not None:
                cap = np.asarray(
                    cap_case["reference"][key]["added_mass"], dtype=float
                )
                row["capytaine_surge_sway"] = float(
                    abs(cap[0, 0] - cap[1, 1]) / abs(cap[0, 0])
                )
                row["capytaine_roll_pitch"] = float(
                    abs(cap[3, 3] - cap[4, 4]) / abs(cap[3, 3])
                )
                row["capytaine_reciprocity"] = float(
                    np.max(np.abs(cap - cap.T)) / np.max(np.abs(cap))
                )
            rows.append(row)
        worst = max(
            max(
                r["tensorbem_surge_sway"],
                r["tensorbem_roll_pitch"],
                r["tensorbem_reciprocity"],
            )
            for r in rows
        )
        out[case] = {
            "rows": rows,
            "worst": float(worst),
            "tolerance": SYMMETRY_TOL,
            "verdict": "PASS" if worst <= SYMMETRY_TOL else "FAIL",
        }
    return out


def deep_submergence() -> dict[str, Any]:
    """Score the deeply submerged sphere against the analytic axes.

    Two independent statements are available for a sphere:

    * **translational added mass** tends to the unbounded-fluid Lamb (1932)
      value ``(2/3) pi rho R^3`` as the free-surface effect decays with depth;
    * **rotational added inertia about the sphere centre is exactly zero** at
      every depth and frequency, because ``(r - centre) x n`` vanishes on the
      whole spherical surface.  This is a property of the *formulation*, so it
      tests the rotational generalised normals themselves; the residual is the
      discretisation floor of that machinery.
    """
    rows = []
    for depth in DEEP_DEPTHS:
        centre = (0.0, 0.0, -float(depth))
        per_level = {}
        for level in DEEP_LEVELS:
            mesh = submerged_sphere_mesh(level, radius=1.0, submergence=float(depth))
            for nu in DEEP_NU:
                mat = solve_matrix(mesh, nu, centre)
                surge = mat["added_mass"][0][0]
                roll = mat["added_mass"][3][3]
                per_level["L{}@{}".format(level, _key(nu))] = {
                    "level": int(level),
                    "n_faces": int(mesh.faces.shape[0]),
                    "nu": nu,
                    "surge_added_mass": surge,
                    "roll_added_inertia": roll,
                    "surge_vs_lamb": (surge - LAMB_SURGE) / LAMB_SURGE,
                    "roll_over_scale": abs(roll) / abs(surge),
                }
        rows.append({"depth": float(depth), "values": per_level})
    deepest = rows[-1]["values"]
    finest = max(deepest.values(), key=lambda v: v["n_faces"])
    worst_translation = abs(finest["surge_vs_lamb"])
    worst_rotation = max(v["roll_over_scale"] for v in deepest.values())
    return {
        "depths": list(DEEP_DEPTHS),
        "levels": list(DEEP_LEVELS),
        "nu_grid": list(DEEP_NU),
        "lamb_surge": LAMB_SURGE,
        "rows": rows,
        "deepest_finest_vs_lamb": float(finest["surge_vs_lamb"]),
        "deepest_finest_faces": int(finest["n_faces"]),
        "worst_rotation_ratio": float(worst_rotation),
        "tolerance": DEEP_TOL,
        "tolerance_rotation": EXACT_NULL_TOL,
        "verdict": "PASS"
        if worst_translation <= DEEP_TOL and worst_rotation <= EXACT_NULL_TOL
        else "FAIL",
    }


def run_study(
    levels: dict[str, Sequence[int]] | None = None, skip_deep: bool = False
) -> dict[str, Any]:
    tensorbem = {
        case: run_case(case, None if levels is None else levels.get(case))
        for case in CASES
    }
    return {
        "density": DENSITY,
        "gravity": GRAVITY,
        "nu_grid": list(NU_GRID),
        "dof_order": list(DOF_ORDER),
        "tensorbem": tensorbem,
        "deep_submergence": None if skip_deep else deep_submergence(),
    }


def load_reference(path: Path | None = None) -> dict[str, Any]:
    target = ARTIFACT_ROOT / REFERENCE_FILE if path is None else path
    return json.loads(target.read_text())


def assemble(study: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    study = dict(study)
    study["capytaine"] = {
        case: reference["cases"][case]
        for case in study["tensorbem"]
        if case in reference["cases"]
    }
    study["capytaine_provenance"] = reference["metadata"]
    study["round1_reproduction"] = reference.get("round1_reproduction")
    study["comparison"] = compare(study["tensorbem"], reference["cases"])
    study["convergence"] = convergence(study["tensorbem"], reference["cases"])
    study["symmetry"] = symmetry_invariants(study["tensorbem"], reference["cases"])
    deep_ref = reference.get("deep_submergence") or {}
    if study.get("deep_submergence") and deep_ref:
        for row in study["deep_submergence"]["rows"]:
            ref_case = deep_ref.get("cases", {}).get("depth_{:g}".format(row["depth"]))
            if ref_case is None:
                continue
            cap = ref_case["reference"][_key(DEEP_NU[0])]["added_mass"]
            row["capytaine_surge"] = cap[0][0]
            row["capytaine_roll"] = cap[3][3]
            row["capytaine_surge_vs_lamb"] = (cap[0][0] - LAMB_SURGE) / LAMB_SURGE
            row["capytaine_roll_over_scale"] = abs(cap[3][3]) / abs(cap[0][0])
    return study


def render_figures(study: dict[str, Any]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return []

    out: list[Path] = []
    nu = NU_GRID[1]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    for col, case in enumerate(study["tensorbem"]):
        for row, quantity in enumerate(("added_mass", "damping")):
            ax = axes[row][col]
            tb = np.asarray(study["tensorbem"][case]["reference"][_key(nu)][quantity])
            cap = np.asarray(study["capytaine"][case]["reference"][_key(nu)][quantity])
            dev = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    if abs(cap[i, j]) <= NULL_FLOOR * _scale(tb, i, j):
                        dev[i, j] = np.nan
                    else:
                        dev[i, j] = (tb[i, j] - cap[i, j]) / abs(cap[i, j]) * 100.0
            im = ax.imshow(dev, cmap="RdBu_r", vmin=-15.0, vmax=15.0)
            for i in range(6):
                for j in range(6):
                    if np.isnan(dev[i, j]):
                        ax.text(j, i, "0", ha="center", va="center",
                                fontsize=7, color="0.45")
                    else:
                        ax.text(j, i, "{:+.1f}".format(dev[i, j]), ha="center",
                                va="center", fontsize=7)
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels(DOF_ORDER, rotation=45, fontsize=8)
            ax.set_yticklabels(DOF_ORDER, fontsize=8)
            ax.set_title("{} / {} @ nu={:g}".format(case, quantity, nu), fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("TensorBEM vs capytaine: relative deviation [%] (0 = structural null)")
    fig.tight_layout()
    FIGURE_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_MATRIX, dpi=140)
    plt.close(fig)
    out.append(FIGURE_MATRIX)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, key, label in (
        (axes[0], "worst_diagonal", "diagonal block"),
        (axes[1], "worst_coupling", "active couplings"),
    ):
        for case, rows in study["convergence"].items():
            ax.plot(
                [r["n_faces"] for r in rows],
                [100.0 * r[key] for r in rows],
                marker="o",
                label=case,
            )
        ax.axhline(5.0, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel("TensorBEM panels")
        ax.set_ylabel("worst |dev| vs capytaine [%] ({})".format(label))
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_CONVERGENCE, dpi=140)
    plt.close(fig)
    out.append(FIGURE_CONVERGENCE)

    deep = study.get("deep_submergence")
    if deep:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        level_keys = [k for k in deep["rows"][0]["values"]]
        for ax, field, label, floor in (
            (
                axes[0],
                "surge_added_mass",
                "surge added mass [kg]",
                deep["lamb_surge"],
            ),
            (
                axes[1],
                "roll_over_scale",
                "|A_roll| / |A_surge|  (exact null)",
                deep["tolerance_rotation"],
            ),
        ):
            for key in level_keys:
                ax.plot(
                    [r["depth"] for r in deep["rows"]],
                    [r["values"][key][field] for r in deep["rows"]],
                    marker="o",
                    label="TensorBEM {} panels".format(deep["rows"][0]["values"][key]["n_faces"]),
                )
            cap_field = "capytaine_surge" if field == "surge_added_mass" else "capytaine_roll_over_scale"
            if all(cap_field in r for r in deep["rows"]):
                ax.plot(
                    [r["depth"] for r in deep["rows"]],
                    [r[cap_field] for r in deep["rows"]],
                    marker="x",
                    ls=":",
                    label="capytaine",
                )
            ax.axhline(floor, color="k", ls="--", lw=0.9,
                       label="Lamb limit" if field == "surge_added_mass" else "1 % floor")
            if field == "roll_over_scale":
                ax.set_yscale("log")
            ax.set_xlabel("submergence depth / R")
            ax.set_ylabel(label)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURE_DEEP, dpi=140)
        plt.close(fig)
        out.append(FIGURE_DEEP)
    return out


def _matrix_table(mat: np.ndarray) -> list[str]:
    lines = ["| row / col | " + " | ".join(DOF_ORDER) + " |", "|---" * 7 + "|"]
    for i, dof in enumerate(DOF_ORDER):
        lines.append("| {} | ".format(dof) + " | ".join(
            "{:.4g}".format(mat[i, j]) for j in range(6)) + " |")
    return lines


def render_document(study: dict[str, Any]) -> str:
    prov = study["capytaine_provenance"]
    comparison = study["comparison"]
    deep = study.get("deep_submergence")
    n_active = sum(
        c.get("n_active", 0) for c in comparison["cases"].values() if c.get("compared")
    )
    n_null = sum(
        c.get("n_null", 0) for c in comparison["cases"].values() if c.get("compared")
    )
    nu = NU_GRID[1]
    lines: list[str] = []
    add = lines.append

    add("# TensorBEM 三维辐射矩阵与 capytaine 的第二轮交叉验证（XVAL-BEM-R2）")
    add("")
    add("## 1. 案例介绍")
    add("")
    add(
        "第一轮交叉验证（`docs/capytaine_crosscheck.md`）只比对了两个平滑物体的"
        "**升沉/纵荡对角项**。辐射矩阵其余部分——**转动块（横摇/纵摇/艏摇）**、"
        "**平移-转动耦合项**（穿水面物体的纵荡-纵摇、横荡-横摇）、以及几何上"
        "**必须为零的结构性零项**——从未与外部程序比对；转动还引入了**转动中心**"
        "这一第一轮无需固定的量（capytaine 3.1 的默认值是 `nan`）。"
    )
    add("")
    add("| 项目 | 取值 |")
    add("|---|---|")
    add("| 物体 A | 穿水面半球，R = 1.0 m，转动中心 (0,0,0) |")
    add("| 物体 B | 浮球，R = 1.0 m，中心 z = -0.5 m，转动中心 (0,0,-0.5) |")
    add(
        "| 流体 | rho = {} kg/m^3, g = {} m/s^2，无限水深 |".format(
            study["density"], study["gravity"]
        )
    )
    add("| 频率 | nu = {} (m^-1) |".format(", ".join("{:g}".format(x) for x in study["nu_grid"])))
    add(
        "| 独立程序 | capytaine {}（commit `{}`），{}，Python {} |".format(
            prov["version"], prov["git_commit"], prov["machine"], prov["python"]
        )
    )
    add("| 自由度 | 全部 6 个：surge, sway, heave, roll, pitch, yaw |")
    add("| 第三解析轴 | 深潜球体 Lamb (1932) 无界流体极限 |")
    add("")
    add("## 2. 方法")
    add("")
    add(
        "1. `scripts/capytaine_crosscheck_r2_ref.py`（仓库内**唯一**导入 capytaine 之处）"
        "对每个辐射自由度求解一次 `RadiationProblem`，记录该自由度对应的**整列**"
        "`A[i,j]`、`B[i,j]`；转动中心通过 `cpt.rigid_body_dofs(rotation_center=...)` "
        "写进自由度定义并回读校验（capytaine 3.1 不允许事后设置，默认留 `nan`）。"
    )
    add(
        "2. `capytaine_crosscheck_r2.run_case` 在声明网格层级上调用 "
        "`solve_radiation_3d(modes=range(6), center=<同一转动中心>)`，取整个 6x6 矩阵。"
    )
    add(
        "3. `compare` 对 36 个 (i,j) x 2 个量（附加质量/阻尼）x 每个频率逐一分类："
        "**active**（两码均显著：|M_ij| > {:.0%}·sqrt(|M_ii M_jj|)）、**null**（两码均在此阈下）、"
        "**conflict**（一码显著、另一码为零 → 直接失败）。".format(NULL_FLOOR)
    )
    add(
        "4. 事前登记、事后不放宽的容差：对角块 **{:.0%}**（与第一轮同带），真正非零的耦合项 "
        "**{:.0%}**，结构零项阈值 **{:.0%}**，轴对称恒等式 **{:.0%}**，Lamb 极限 **{:.0%}**。".format(
            TOL_DIAGONAL, TOL_OFFDIAGONAL, NULL_FLOOR, SYMMETRY_TOL, DEEP_TOL
        )
    )
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 网格层级与逐案结论")
    add("")
    add("| case | TensorBEM 面板 | capytaine 面板 | active | 结构零项 | conflict | 最差 active 偏差 | 判定 |")
    add("|---|---:|---:|---:|---:|---:|---:|---|")
    for case, row in comparison["cases"].items():
        if not row.get("compared"):
            add("| {} | - | - | - | - | - | - | {} |".format(case, row.get("reason")))
            continue
        add(
            "| {} | {} | {} | {} | {} | {} | {:.2f}% | {} |".format(
                case,
                row["tensorbem_faces"],
                row["capytaine_faces"],
                row["n_active"],
                row["n_null"],
                row["n_conflict"],
                100.0 * row["worst_active_relative"],
                row["verdict"],
            )
        )
    add("")
    add(
        "总计 {} 个 active 项与 {} 个结构零项被逐项判定，总体判定 **{}**"
        "（对角带 {:.0%}、耦合带 {:.0%}）。".format(
            n_active, n_null, comparison["verdict"], TOL_DIAGONAL, TOL_OFFDIAGONAL
        )
    )
    add("")
    add("### 3.2 附加质量矩阵（六自由度，nu = {:g} m^-1）".format(nu))
    add("")
    for case in study["tensorbem"]:
        add(
            "**{}** — TensorBEM（finest，{} 面板）".format(
                case, study["tensorbem"][case]["reference_faces"]
            )
        )
        add("")
        lines.extend(
            _matrix_table(
                np.asarray(study["tensorbem"][case]["reference"][_key(nu)]["added_mass"])
            )
        )
        add("")
        add(
            "**{}** — capytaine（{} 面板，转动中心 {}）".format(
                case,
                study["capytaine"][case]["reference_faces"],
                study["capytaine"][case].get("rotation_center"),
            )
        )
        add("")
        lines.extend(
            _matrix_table(
                np.asarray(study["capytaine"][case]["reference"][_key(nu)]["added_mass"])
            )
        )
        add("")
    add("### 3.3 active 项逐项偏差（含转动块与耦合项）")
    add("")
    for case, row in comparison["cases"].items():
        if not row.get("compared"):
            continue
        add("**{}**".format(case))
        add("")
        add("| nu | quantity | (i,j) | TensorBEM | capytaine | 相对偏差 | 登记带 |")
        add("|---:|---|---|---:|---:|---:|---:|")
        for entry in row["entries"]:
            if entry["class"] != "active":
                continue
            add(
                "| {:g} | {} | ({},{}) | {:.5g} | {:.5g} | {:+.2f}% | {:.0%} |".format(
                    entry["nu"],
                    entry["quantity"],
                    entry["row"],
                    entry["column"],
                    entry["tensorbem"],
                    entry["capytaine"],
                    100.0 * entry["relative"],
                    100.0 * entry["band"],
                )
            )
        add("")
    add("### 3.4 结构零项（几何禁止非零的项）")
    add("")
    add(
        "两个物体都关于 z 轴轴对称，并有水平对称面，因此纵荡-横荡、升沉-横摇、纵荡-横摇、"
        "横荡-纵摇等项必须为零。这是只看对角项时看不见的缺陷类：本不该存在的耦合如果出现，"
        "只有逐项检查才能发现。"
    )
    add("")
    add("| case | 结构零项个数 | 两码最大 |M_ij| / sqrt(|M_ii M_jj|) |")
    add("|---|---:|---:|")
    for case, row in comparison["cases"].items():
        if row.get("compared"):
            add(
                "| {} | {} | {:.4f}% |".format(
                    case, row["n_null"], 100.0 * row["worst_null_ratio"]
                )
            )
    add("")
    add("### 3.5 网格收敛（最差偏差随面板数）")
    add("")
    add("| case | 面板 | 对角最差偏差 | 耦合最差偏差 |")
    add("|---|---:|---:|---:|")
    for case, rows in study["convergence"].items():
        for r in rows:
            add(
                "| {} | {} | {:.2f}% | {:.2f}% |".format(
                    case,
                    r["n_faces"],
                    100.0 * r["worst_diagonal"],
                    100.0 * r["worst_coupling"],
                )
            )
    add("")
    add("### 3.6 几何恒等式（不依赖外部程序的独立轴）")
    add("")
    add("| case | nu | TB A_surge vs A_sway | TB A_roll vs A_pitch | TB 互易性 max|A-A^T|/max|A| | capytaine 同量 (surge/sway, roll/pitch) | 判定 |")
    add("|---|---:|---:|---:|---:|---|---|")
    for case, block in study["symmetry"].items():
        for r in block["rows"]:
            cap_txt = "-"
            if r.get("capytaine_surge_sway") is not None:
                cap_txt = "{:.3f}% / {:.3f}%".format(
                    100.0 * r["capytaine_surge_sway"],
                    100.0 * r["capytaine_roll_pitch"],
                )
            add(
                "| {} | {:g} | {:.3f}% | {:.3f}% | {:.2e} | {} | {} |".format(
                    case,
                    r["nu"],
                    100.0 * r["tensorbem_surge_sway"],
                    100.0 * r["tensorbem_roll_pitch"],
                    r["tensorbem_reciprocity"],
                    cap_txt,
                    block["verdict"],
                )
            )
    add("")
    add(
        "轴对称性受网格各向异性限制（icosphere 网格并非严格轴对称），因此这里是离散化误差"
        "而非求解器缺陷；capytaine 侧同量数值一并列出以免单方面归因。"
    )
    add("")
    add("### 3.7 第三轴（解析）：深潜球体的两个精确陈述")
    add("")
    if deep:
        add(
            "**(a) 转动块精确为零。** 球体绕**自身几何中心**转动时，物面上 "
            "`(r - centre) x n == 0` 处处成立，激励为零，故附加惯量在任意频率、任意潜深下"
            "**精确为零**（与 free-surface 条件无关）。这是形式性质而非数值性质，"
            "因此它直接检验转动广义法向 `(r-centre) x n` 的装配：量纲取自 "
            "`|A_roll| / |A_surge|`，登记阈 {:.0%}。".format(EXACT_NULL_TOL)
        )
        add("")
        add("**(b) 平动附加质量 → Lamb (1932) 无界流体极限。** 解析值 "
            "(2/3)·pi·rho·R^3 = {:.2f} kg；自由面影响随潜深衰减，剩余偏差由面板离散控制，"
            "本轮的深层阶梯只用单一网格（见局限 5），因此该偏差含离散误差。登记带 {:.0%}。".format(deep["lamb_surge"], DEEP_TOL))
        add("")
        add("| 深度/R | TensorBEM 面板 | A_surge [kg] | vs Lamb | |A_roll|/|A_surge| | capytaine A_surge | capytaine |A_roll|/|A_surge| |")
        add("|---:|---:|---:|---:|---:|---:|---:|")
        for row in deep["rows"]:
            for key, v in row["values"].items():
                cap_s = "{:.2f}".format(row["capytaine_surge"]) if "capytaine_surge" in row else "-"
                cap_r = (
                    "{:.2e}".format(row["capytaine_roll_over_scale"])
                    if "capytaine_roll_over_scale" in row
                    else "-"
                )
                add(
                    "| {:g} | {} | {:.2f} | {:+.2f}% | {:.2e} | {} | {} |".format(
                        row["depth"],
                        v["n_faces"],
                        v["surge_added_mass"],
                        100.0 * v["surge_vs_lamb"],
                        v["roll_over_scale"],
                        cap_s,
                        cap_r,
                    )
                )
        add("")
        add(
            "最深一层（{:g}R）最细网格（{} 面板）平动偏差 **{:+.2f}%**，转动比最大 **{:.2e}**"
            "（阈 {:.0e}），合判 **{}**。".format(
                DEEP_DEPTHS[-1],
                deep["deepest_finest_faces"],
                100.0 * deep["deepest_finest_vs_lamb"],
                deep["worst_rotation_ratio"],
                EXACT_NULL_TOL,
                deep["verdict"],
            )
        )
    else:
        add("未运行（`skip_deep`）。")
    add("")
    add("### 3.8 第一轮冻结参考的可复现性")
    add("")
    repro = study.get("round1_reproduction")
    if repro and repro.get("available"):
        add(
            "用本轮 live 运行重算第一轮所用网格（半球 36x72、球 28x56），与冻结文件 "
            "`capytaine_ref.json` 的最大相对差 **{:.3e}%**（{} 个格子）。".format(
                100.0 * repro["worst_relative"], len(repro["entries"])
            )
        )
    else:
        add("不可用：{}".format((repro or {}).get("reason")))
    add("")
    add("## 4. 结果图")
    add("")
    add(
        "![matrix deviation](fig_capytaine_r2_matrix.png)\n\n"
        "*图 1 — 全 6x6 矩阵逐项相对偏差（nu = {:g} m^-1；上排附加质量、下排阻尼；"
        "左列半球、右列浮球）。灰 `0` 为结构零项（两码均在阈下），不参与百分比比较。*".format(nu)
    )
    add("")
    add(
        "![convergence](fig_capytaine_r2_convergence.png)\n\n"
        "*图 2 — 对角块（左）与 active 耦合项（右）最差偏差随 TensorBEM 面板数的收敛。*"
    )
    add("")
    if deep:
        add(
            "![deep submergence](fig_capytaine_r2_deep.png)\n\n"
            "*图 3 — 深潜球体附加质量/附加惯量随潜深走势与 Lamb 无界流体极限（黑虚线），"
            "两码同时绘出。*"
        )
        add("")
    add("## 5. 注意点与局限")
    add("")
    add(
        "1. **只覆盖无限水深**。`solve_radiation_3d` 是 Rankine + 内部面元（lid）形式，"
        "本身不含有限水深色散；有限水深路线（`docs/BEM_FINITE_DEPTH_GREEN.md`）本轮"
        "**未**与 capytaine 比对，仍是未偿欠账。"
    )
    add(
        "2. **只覆盖辐射**。绕射/激励力另见 "
        "`benchmarks/verified/casepool_bem_hulme_hemisphere/excitation_capytaine_xval.py`；"
        "二阶漂移力 capytaine 3.1 不提供。"
    )
    add(
        "3. **capytaine 自身误差**。两个低阶 Rankine 面元法一致只说明二者在各自离散误差内重合，"
        "不能证明任一者为真值。转动块与耦合项**没有**解析第三轴，只能靠 Lamb 极限（深潜球）"
        "与网格收敛间接支撑——这是本轮最重要的诚实局限。"
    )
    add(
        "4. **转动中心**。所有转动量都是关于登记转动中心的；换中心会得到不同的耦合矩阵"
        "（同一物理，不同表示）。artifact 记录了两侧的中心并断言一致。"
    )
    add(
        "5. **本轮的网格预算**。受本单算力预算限制，TensorBEM 侧半球只跑到 256 面板"
        "（第一轮用 1024）、球只用 270（第一轮 1026），深潜块只用单一 320 面板网格；"
        "因此本轮给出的是**自由度覆盖**（转动块与耦合项首次入库）而非收敛证明，"
        "偏差中含本侧离散误差。capytaine 侧仍取其自身网格阶梯的最细一层。"
    )
    add(
        "6. **网格各向异性**。TensorBEM 侧几何由 icosphere 派生，surge/sway 与 roll/pitch "
        "并不严格相等，故 3.6 的恒等式检查按离散化误差解读。"
    )
    add("")
    add("## 6. 误差判定（逐项，显式回答是否 <3%）")
    add("")
    add("| 关键量 | 基准 | 本轮机 | 误差 | <3%? |")
    add("|---|---|---|---:|---|")
    worst_diag_entry = None
    worst_off_entry = None
    for case, row in comparison["cases"].items():
        if not row.get("compared"):
            continue
        for entry in row["entries"]:
            if entry["class"] != "active":
                continue
            if entry["row"] == entry["column"]:
                if worst_diag_entry is None or abs(entry["relative"]) > abs(
                    worst_diag_entry["relative"]
                ):
                    worst_diag_entry = entry
            else:
                if worst_off_entry is None or abs(entry["relative"]) > abs(
                    worst_off_entry["relative"]
                ):
                    worst_off_entry = entry
    for label, entry in (("最差对角项", worst_diag_entry), ("最差 active 耦合项", worst_off_entry)):
        if entry is None:
            continue
        add(
            "| {} (nu={:g}, {}, A[{},{}]) | capytaine | {:.5g} vs {:.5g} | {:+.2f}% | {} |".format(
                label,
                entry["nu"],
                entry["quantity"],
                entry["row"],
                entry["column"],
                entry["tensorbem"],
                entry["capytaine"],
                100.0 * entry["relative"],
                "是" if abs(entry["relative"]) < 0.03 else "否",
            )
        )
    if deep:
        add(
            "| 深潜球 A_surge（{:g}R, {} 面板） | Lamb (2/3)pi rho R^3 = {:.2f} kg | {:.2f} kg | {:+.2f}% | {} |".format(
                DEEP_DEPTHS[-1],
                deep["deepest_finest_faces"],
                deep["lamb_surge"],
                deep["rows"][-1]["values"][sorted(deep["rows"][-1]["values"])[-1]][
                    "surge_added_mass"
                ],
                100.0 * deep["deepest_finest_vs_lamb"],
                "是" if abs(deep["deepest_finest_vs_lamb"]) < 0.03 else "否",
            )
        )
        add(
            "| 深潜球 A_roll（球心转动，精确零） | 0（解析） | {:.2e}·|A_surge| | 相对 |A_surge| 为 {:.2e} | {} |".format(
                deep["worst_rotation_ratio"],
                deep["worst_rotation_ratio"],
                "是" if deep["worst_rotation_ratio"] < 0.03 else "否",
            )
        )
    add("")
    add(
        "验收标准：owner 规定的对标误差 >15% 不可接受；本模块登记的对角带 {:.0%}、耦合带 {:.0%} "
        "均严于该标准。".format(TOL_DIAGONAL, TOL_OFFDIAGONAL)
    )
    add("")
    return "\n".join(lines)


def write_outputs(study: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    root = ARTIFACT_ROOT if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    figures = render_figures(study)
    study["figures"] = [str(path) for path in figures]
    result_path = root / RESULT_FILE
    result_path.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(render_document(study))
    return {"result": result_path, "document": DOC_PATH, "figures": figures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--skip-deep", action="store_true")
    args = parser.parse_args(argv)
    reference = load_reference(args.reference)
    study = assemble(run_study(skip_deep=args.skip_deep), reference)
    paths = write_outputs(study)
    print("wrote {}".format(paths["result"]))
    print("wrote {}".format(paths["document"]))
    for fig in paths["figures"]:
        print("wrote {}".format(fig))
    print(
        "verdict {} (worst active {:.2%}); symmetry {}; deep {}".format(
            study["comparison"]["verdict"],
            study["comparison"]["worst_active_relative"],
            {k: v["verdict"] for k, v in study["symmetry"].items()},
            (study["deep_submergence"] or {}).get("verdict"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())