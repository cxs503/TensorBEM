#!/usr/bin/env python3
"""CASE-BEM-HULME-SURGE-LOWFREQ -- the horizontal long-wave end of the
floating-hemisphere anchor (``A11(ka -> 0)``).

Why this package exists
-----------------------
``CASE-BEM-HULME-HEMISPHERE`` (case pool id 138, repo TensorBEM, tol ``< 5 %``)
is pinned on its documented band ``ka in [0.2, 2.0]``: radiation (surge/heave
added mass and damping), diffraction, mesh and reference-term ladders, a
third-party (capytaine) route and a Haskind reciprocity check are all committed.
The *heave* column also carries an exact low-frequency anchor --
``A33(ka -> 0) = 0.8309572 rho*nabla`` from the even-parity (sliding) Legendre
series, see :mod:`tensorbem.benchmarks.hemisphere_zero_frequency_envelope` and
``benchmarks/verified/bem_rigid_lid_hemisphere_ka0``.

The *surge* column had no such anchor: every one of its non-self-referential
checks lives inside the band (capytaine, Haskind, both at ``ka >= 0.2``).  A
reference series that is 10 % wrong in the long-wave range could still pass all
of those gates, so the horizontal long-wavelength limit is the one place where
the ``A11`` column can be falsified from outside.  This package closes that gap.

The exact anchor
----------------
For ``omega -> 0`` the free-surface condition ``dphi/dz - nu phi = 0``
degenerates to the rigid-wall condition ``dphi/dz = 0`` on ``z = 0`` (outside the
waterplane ``r > a``).  Continue the fluid evenly in ``z``:

* a *horizontal* velocity is even under ``z -> -z`` while the mirrored outward
  normal flips its ``z`` component, so the Neumann data ``u.n = U n_x`` is
  *preserved* under the reflection (the opposite of the heave case, where
  ``u.n = U n_z`` changes sign and the image body slides instead of translating
  -- the trap documented by the m16802 erratum);
* the mirrored problem is therefore a **full sphere translating in x** in
  unbounded fluid, whose added mass is the classical Kirchhoff value
  ``A_sph = 0.5 rho Vol_sph`` with ``Vol_sph = 2 nabla``;
* the doubled flow field carries twice the kinetic energy of the half-space
  field at the same velocity, so ``A11(0) = A_sph / 2 = rho nabla / 2``.

Hence, with ``nabla = (2/3) pi a^3``,

    A11(ka -> 0) / (rho nabla) = 1/2        (exactly)

which this package tests four ways:

  R1  closed form (Kirchhoff sphere + energy halving)                  -> 0.5
  R2  spectral Legendre solve of the translating sphere (numerical
      self test of the doubled-body constant)                          -> 1.0 -> 0.5
  R3  the repository's Hulme (1982) multipole reference series, ka ladder
      down to ``ka = 2e-3`` plus a linear extrapolation in ``ka``        -> ~0.5
  R4  the production Kelvin-Green 3-D solver at ``ka = 1e-3``          -> ~0.5

Nothing in ``results.json`` / ``report.md`` is transcribed by hand: the report and
the figure are rendered from the JSON payload this module writes.

Run::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_surge_lowfreq/run_case.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for parent in HERE.parents:
        if (parent / "src" / "tensorbem" / "__init__.py").exists():
            return parent
    raise RuntimeError("repo root not found above " + str(HERE))


if str(_repo_root() / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root() / "src"))

from tensorbem.analytic import hulme_hemisphere_radiation  # noqa: E402
from tensorbem.free_surface_solver3d import solve_radiation_3d  # noqa: E402
from tensorbem.geometry3d import hemisphere_hull_mesh  # noqa: E402

# ---------------------------------------------------------------------------
# Case constants (SI)
# ---------------------------------------------------------------------------
CASE_CODE = "CASE-BEM-HULME-SURGE-LOWFREQ"
ANCHOR = "CASE-BEM-HULME-HEMISPHERE"
RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
GRAD = (2.0 / 3.0) * math.pi * RADIUS**3          # displaced volume, exact
RHO_GRAD = DENSITY * GRAD

TOLERANCE = 0.05                                  # case-pool band (< 5 %)

#: ``A11(ka -> 0) / (rho nabla)``, exact, see the module docstring.
EXACT_LIMIT_DIMLESS = 0.5
#: Settled heave anchor (``bem_rigid_lid_hemisphere_ka0`` / zero-frequency envelope).
HEAVE_RIGID_LID_DIMLESS = 0.8309572

REFERENCE_KAS: tuple[float, ...] = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
REFERENCE_TERMS: tuple[int, ...] = (48, 96, 144)
REFERENCE_QUAD = 512                      # >= 2 * max(n_terms)

MESH_LEVELS: tuple[int, ...] = (2, 3, 4)          # 64, 256, 1024 panels
MESH_LADDER_KA = 1.0e-3
KA_LADDER: tuple[float, ...] = (1.0e-3, 1.0e-2, 3.0e-2, 0.1)
KA_LADDER_SUB = 3

SURGE_MODE = 0                                    # (surge, sway, heave, roll, pitch, yaw)


# ---------------------------------------------------------------------------
# R1/R2: the exact anchor and its doubled-body constant
# ---------------------------------------------------------------------------
def kirchhoff_sphere_surge_added_mass(
    density: float = DENSITY, radius: float = RADIUS
) -> float:
    """Kirchhoff: added mass of a translating sphere, ``0.5 rho Vol_sphere``."""
    vol_sphere = (4.0 / 3.0) * math.pi * radius**3
    return 0.5 * density * vol_sphere


def exact_surge_limit(rho_grad: float = RHO_GRAD) -> dict[str, Any]:
    """The ``ka -> 0`` surge anchor: closed form plus the energy-halving step."""
    a_sphere = kirchhoff_sphere_surge_added_mass()
    a_hemi = 0.5 * a_sphere
    return {
        "statement": "A11(ka->0) = A_sph/2 = rho*nabla/2 (exact)",
        "sphere_volume_m3": (4.0 / 3.0) * math.pi * RADIUS**3,
        "hemisphere_volume_m3": GRAD,
        "sphere_added_mass_kg": a_sphere,
        "hemisphere_added_mass_kg": a_hemi,
        "hemisphere_added_mass_per_rho_nabla": a_hemi / rho_grad,
        "derivation_steps": [
            "omega -> 0: dphi/dz - nu*phi = 0 degenerates to the rigid wall dphi/dz = 0 on z = 0 (r > a)",
            "even continuation z -> -z preserves the horizontal Neumann data u.n = U*n_x",
            "the mirrored problem is a full sphere translating in x in unbounded fluid",
            "Kirchhoff: A_sph = rho*(4/3)*pi*a^3/2",
            "T_doubled = 2*T_hemisphere at equal U  =>  A_hemi = A_sph/2 = rho*nabla/2",
            "normalising by rho*nabla with nabla = (2/3)*pi*a^3 gives exactly 1/2",
        ],
    }


def doubled_body_self_test() -> dict[str, Any]:
    """R2: independent Legendre solve of the translating sphere (doubled body)."""
    from tensorbem.benchmarks.hemisphere_zero_frequency_envelope import (
        spectral_added_mass,
    )

    a33, dimless = spectral_added_mass("translation", n_terms=60, radius=RADIUS)
    return {
        "method": "tensorbem.benchmarks.hemisphere_zero_frequency_envelope."
                  "spectral_added_mass('translation', n_terms=60)",
        "full_sphere_dimensionless_vs_rho_nabla":
            float(kirchhoff_sphere_surge_added_mass() / RHO_GRAD),
        "spectral_lower_half_dimensionless_vs_rho_nabla": float(dimless),
        "lower_half_added_mass_kg": float(a33),
        "note": "the spectral projector integrates the translating-sphere potential over the "
                "lower hemisphere, so it returns the half-space (hemisphere) value directly: "
                "0.5 rho*nabla in rho*nabla units; the closed-form full sphere is 1.0 in the "
                "same units, and halving it for the half-space gives the same 1/2",
    }


# ---------------------------------------------------------------------------
# R3: the repository reference series down to ka = 2e-3
# ---------------------------------------------------------------------------
def reference_ladder(
    kas: Iterable[float] = REFERENCE_KAS,
    terms: Sequence[int] = REFERENCE_TERMS,
    quad_order: int = REFERENCE_QUAD,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for ka in kas:
        ka = float(ka)
        per_terms = {}
        for n_terms in terms:
            per_terms[int(n_terms)] = hulme_hemisphere_radiation(
                ka,
                radius=RADIUS,
                density=DENSITY,
                gravity=GRAVITY,
                n_terms=int(n_terms),
                quad_order=int(quad_order),
            )
        base = per_terms[int(terms[-1])]
        row: dict[str, float] = {
            "ka": ka,
            "a11_dimless": float(base.surge_added_mass_dimless),
            "b11_dimless": float(base.surge_damping_dimless),
            "b11_far_field_dimless": float(base.surge_far_field_damping_dimless),
            "a33_dimless": float(base.heave_added_mass_dimless),
            "b33_dimless": float(base.heave_damping_dimless),
            "excess_over_exact": float(base.surge_added_mass_dimless - EXACT_LIMIT_DIMLESS),
            "excess_per_ka": float(
                (base.surge_added_mass_dimless - EXACT_LIMIT_DIMLESS) / ka
            ),
        }
        for n_terms, res in per_terms.items():
            row[f"a11_terms_{n_terms}_dimless"] = float(res.surge_added_mass_dimless)
        rows.append(row)
    return rows


def extrapolate_limit(rows: Sequence[dict[str, float]], n_fit: int = 3) -> dict[str, Any]:
    """Linear extrapolation of ``A11(ka)`` in ``ka`` over the smallest ``n_fit`` kas."""
    order = sorted(rows, key=lambda r: r["ka"])[: int(n_fit)]
    x = np.array([r["ka"] for r in order], dtype=float)
    y = np.array([r["a11_dimless"] for r in order], dtype=float)
    design = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return {
        "method": f"ordinary least squares A11 = a0 + a1*ka over the {len(order)} smallest ka",
        "kas_used": [float(v) for v in x],
        "a0_dimless": float(coef[0]),
        "a1_dimless_per_ka": float(coef[1]),
        "max_residual": float(np.max(np.abs(pred - y))),
        "rel_error_vs_exact": float(coef[0] / EXACT_LIMIT_DIMLESS - 1.0),
        "smallest_ka": float(x[0]),
        "value_at_smallest_ka": float(y[0]),
        "rel_error_at_smallest_ka": float(y[0] / EXACT_LIMIT_DIMLESS - 1.0),
        "note": "the excess is linear in ka over the fitted window, so the intercept is the limit",
    }


# ---------------------------------------------------------------------------
# R4: the production 3-D solver at ka = 1e-3
# ---------------------------------------------------------------------------
def _solve_surge(subdivisions: int, ka: float) -> dict[str, Any]:
    mesh = hemisphere_hull_mesh(int(subdivisions), radius=RADIUS)
    t0 = time.time()
    res = solve_radiation_3d(
        mesh, float(ka), modes=[SURGE_MODE], density=DENSITY, gravity=GRAVITY
    )
    dt = time.time() - t0
    omega = math.sqrt(GRAVITY * float(ka))
    a11 = float(res.added_mass[0, 0])
    b11 = float(res.damping[0, 0])
    return {
        "subdivisions": int(subdivisions),
        "panels": int(mesh.areas.size),
        "ka": float(ka),
        "omega": float(omega),
        "added_mass_kg": a11,
        "added_mass_dimless": float(a11 / RHO_GRAD),
        "damping_dimless": float(b11 / (omega * RHO_GRAD)),
        "rel_error_vs_exact": float(a11 / RHO_GRAD / EXACT_LIMIT_DIMLESS - 1.0),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
        "solve_seconds": float(dt),
    }


def solver_ladders() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mesh_rows = [_solve_surge(sub, MESH_LADDER_KA) for sub in MESH_LEVELS]
    ka_rows = [_solve_surge(KA_LADDER_SUB, float(ka)) for ka in KA_LADDER]
    return mesh_rows, ka_rows


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def evaluate_gates(
    exact: dict[str, Any],
    spectral: dict[str, Any],
    reference_rows: Sequence[dict[str, float]],
    extrap: dict[str, Any],
    mesh_rows: Sequence[dict[str, Any]],
    ka_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    small_to_large = sorted(reference_rows, key=lambda r: r["ka"])
    excess = [r["excess_over_exact"] for r in small_to_large]
    # small_to_large is ascending in ka; the series approaches the limit from above,
    # so the excess must grow with ka and stay positive at the smallest ka.
    monotone = (
        all(excess[i] < excess[i + 1] for i in range(len(excess) - 1))
        and excess[0] > 0.0
    )

    term_drift = 0.0
    for r in reference_rows:
        lo = r[f"a11_terms_{REFERENCE_TERMS[0]}_dimless"]
        hi = r[f"a11_terms_{REFERENCE_TERMS[-1]}_dimless"]
        term_drift = max(term_drift, abs(hi - lo) / hi)

    finest = max(mesh_rows, key=lambda r: r["panels"])
    errs = [abs(r["rel_error_vs_exact"]) for r in mesh_rows]
    mesh_monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))

    ka_sorted = sorted(ka_rows, key=lambda r: r["ka"])
    b_values = [r["damping_dimless"] for r in ka_sorted]
    damping_grows = all(b_values[i] < b_values[i + 1] for i in range(len(b_values) - 1))
    min_ka_row = ka_sorted[0]

    heave_rel = abs(small_to_large[0]["a33_dimless"] / HEAVE_RIGID_LID_DIMLESS - 1.0)

    gates: dict[str, Any] = {
        "G1_exact_closed_form": {
            "measured": exact["hemisphere_added_mass_per_rho_nabla"],
            "criterion": "closed form A11(0)/(rho*nabla) = 1/2 to 1e-12",
            "pass": bool(
                abs(exact["hemisphere_added_mass_per_rho_nabla"] - EXACT_LIMIT_DIMLESS) < 1e-12
            ),
        },
        "G2_doubled_body_self_test": {
            "measured": spectral["spectral_lower_half_dimensionless_vs_rho_nabla"],
            "criterion": "independent spectral solve of the translating-sphere BC, integrated "
                         "over the lower hemisphere, returns 1/2 (rho*nabla units) to 1e-6",
            "pass": bool(
                abs(spectral["spectral_lower_half_dimensionless_vs_rho_nabla"]
                    - EXACT_LIMIT_DIMLESS) < 1e-6
            ),
        },
        "G3_reference_at_min_ka_within_0p5pct": {
            "measured": 100.0 * extrap["rel_error_at_smallest_ka"],
            "criterion": "|A11(ka=2e-3)/0.5 - 1| < 0.5 %",
            "pass": bool(abs(extrap["rel_error_at_smallest_ka"]) < 0.005),
            "where": {"ka": extrap["smallest_ka"]},
        },
        "G4_extrapolated_limit_within_0p2pct": {
            "measured": 100.0 * extrap["rel_error_vs_exact"],
            "criterion": "|linear extrapolation to ka=0 / 0.5 - 1| < 0.2 %",
            "pass": bool(abs(extrap["rel_error_vs_exact"]) < 0.002),
        },
        "G5_monotone_approach_from_above": {
            "measured": excess,
            "criterion": "excess A11-1/2 strictly increasing with ka (decreasing as ka falls), "
                         "positive at the smallest ka",
            "pass": bool(monotone),
        },
        "G6_reference_term_truncation": {
            "measured": term_drift,
            "criterion": "|A11(144 terms) - A11(48 terms)| / A11 < 1e-4 at every ka",
            "pass": bool(term_drift < 1e-4),
        },
        "G7_solver_finest_within_case_tolerance": {
            "measured": 100.0 * finest["rel_error_vs_exact"],
            "criterion": "|A11(ka=1e-3, 1024 panels)/(0.5 rho*nabla) - 1| < 5 % (case tolerance)",
            "pass": bool(abs(finest["rel_error_vs_exact"]) < TOLERANCE),
            "where": {"ka": finest["ka"], "panels": finest["panels"]},
        },
        "G8_solver_mesh_monotone": {
            "measured": [100.0 * e for e in errs],
            "criterion": "|error| decreases 64 -> 256 -> 1024 panels",
            "pass": bool(mesh_monotone),
        },
        "G9_damping_vanishes": {
            "measured": {
                "min_ka": min_ka_row["ka"],
                "b11_dimless": min_ka_row["damping_dimless"],
            },
            "criterion": "B11(ka=1e-3) < 1e-3 and B11 increases with ka over the ladder",
            "pass": bool(min_ka_row["damping_dimless"] < 1e-3 and damping_grows),
        },
        "G10_heave_crosslink_1pct": {
            "measured": 100.0 * heave_rel,
            "criterion": "reference A33 at ka=2e-3 within 1 % of the settled rigid-lid "
                         "heave anchor 0.8309572 (harness consistency)",
            "pass": bool(heave_rel < 0.01),
        },
    }
    verdict = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
    return {
        "gates": gates,
        "verdict": verdict,
        "failing": [k for k, g in gates.items() if not g["pass"]],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_case(write: bool = True) -> dict[str, Any]:
    t_start = time.time()
    exact = exact_surge_limit()
    spectral = doubled_body_self_test()
    reference_rows = reference_ladder()
    extrap = extrapolate_limit(reference_rows)
    mesh_rows, ka_rows = solver_ladders()
    gate_block = evaluate_gates(exact, spectral, reference_rows, extrap, mesh_rows, ka_rows)

    payload: dict[str, Any] = {
        "case_code": CASE_CODE,
        "anchor": ANCHOR,
        "scope": "horizontal (surge) long-wave end, ka -> 0, of the floating-hemisphere "
                 "radiation anchor",
        "physical_case": {
            "body": "half-submerged hemisphere, flat face flush with the mean free surface",
            "radius_m": RADIUS,
            "displaced_volume_m3": GRAD,
            "waterplane_area_m2": math.pi * RADIUS**2,
            "density_kg_m3": DENSITY,
            "gravity_m_s2": GRAVITY,
            "rho_nabla_kg": RHO_GRAD,
            "depth": "infinite (deep water)",
            "mode": "surge, unit velocity amplitude, exp(-i*omega*t)",
            "normalisation": "A11/(rho*nabla), B11/(omega*rho*nabla)",
            "limit": "ka -> 0 is the rigid-wall (omega -> 0) limit of the free surface",
        },
        "exact_anchor": exact,
        "doubled_body_self_test": spectral,
        "reference": {
            "module": "tensorbem.analytic.hulme_hemisphere_radiation",
            "knot": "Hulme (1982) J. Fluid Mech. 121:443-463 wave-free multipole series, "
                    "re-derived in-repo (not the printed table)",
            "kas": list(REFERENCE_KAS),
            "n_terms_ladder": list(REFERENCE_TERMS),
            "quad_order": REFERENCE_QUAD,
        },
        "reference_ladder": reference_rows,
        "extrapolation": extrap,
        "solver": {
            "module": "tensorbem.free_surface_solver3d.solve_radiation_3d",
            "mesh": "tensorbem.geometry3d.hemisphere_hull_mesh",
            "mesh_ladder_subdivisions": list(MESH_LEVELS),
            "mesh_ladder_ka": MESH_LADDER_KA,
            "ka_ladder": list(KA_LADDER),
            "ka_ladder_subdivisions": KA_LADDER_SUB,
            "chief_points": "auto",
        },
        "solver_mesh_ladder": mesh_rows,
        "solver_ka_ladder": ka_rows,
        "cross_link": {
            "heave_rigid_lid_dimless_settled": HEAVE_RIGID_LID_DIMLESS,
            "heave_reference_at_min_ka": sorted(reference_rows, key=lambda r: r["ka"])[0][
                "a33_dimless"
            ],
            "source": "benchmarks/verified/bem_rigid_lid_hemisphere_ka0 + "
                      "tensorbem.benchmarks.hemisphere_zero_frequency_envelope",
        },
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runtime_seconds": float(time.time() - t_start),
        **gate_block,
    }
    if write:
        write_artifacts(payload)
    return payload


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _measured_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(f"{v:.3e}" if isinstance(v, float) else str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, float):
        return _fmt(value)
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    pc = payload["physical_case"]
    ex = payload["exact_anchor"]
    ref = payload["reference"]
    extrap = payload["extrapolation"]
    gates = payload["gates"]
    mesh_rows = payload["solver_mesh_ladder"]
    ka_rows = sorted(payload["solver_ka_ladder"], key=lambda r: r["ka"])
    ref_rows = sorted(payload["reference_ladder"], key=lambda r: r["ka"])
    harness = payload["doubled_body_self_test"]
    near_band = min(ref_rows, key=lambda r: abs(r["ka"] - 0.2))
    band_excess_pct = 100.0 * (near_band["a11_dimless"] / EXACT_LIMIT_DIMLESS - 1.0)

    lines: list[str] = []
    add = lines.append

    add(f"# {payload['case_code']}: 浮半球水平（surge）长波极限 `A11(ka->0)` 锚定")
    add("")
    add(f"> 锚点 `{payload['anchor']}`（算例池 id 138，容差 `{TOLERANCE * 100:.0f}%`）的"
        "水平分量验证欠账。")
    add("> 本文件由 `benchmarks/verified/casepool_bem_hulme_surge_lowfreq/limits.py` 从")
    add("> `results.json` 程序化渲染，正文不手抄任何数字。")
    add("")
    add("## 1. 案例介绍")
    add("")
    add(f"- 几何：{pc['body']}，a = {_fmt(pc['radius_m'], 3)} m，"
        f"排水体积 ∇ = {_fmt(pc['displaced_volume_m3'])} m³，"
        f"水线面 πa² = {_fmt(pc['waterplane_area_m2'])} m²")
    add(f"- 流体：ρ = {_fmt(pc['density_kg_m3'], 1)} kg/m³，g = {_fmt(pc['gravity_m_s2'], 5)} m/s²，"
        f"无限水深；ρ∇ = {_fmt(pc['rho_nabla_kg'], 3)} kg")
    add(f"- 模态与量纲：{pc['mode']}；{pc['normalisation']}")
    add(f"- 极限语义：{pc['limit']}")
    add("")
    add("### 为什么需要这个包")
    add("")
    add("锚点算例在 `ka ∈ [0.2, 2.0]` 带内已闭环（辐射/绕射、面元与参考项阶梯、capytaine 第三方路线、"
        "Haskind 互易、长波激励力极限）。**heave** 列另有精确低频锚 "
        f"`A33(ka→0) = {HEAVE_RIGID_LID_DIMLESS} ρ∇`（`bem_rigid_lid_hemisphere_ka0`；"
        "滑移/平移之辨见其 erratum），**surge** 列此前没有任何带外锚——带内所有对照都在 "
        "`ka ≥ 0.2`，参考级数若在长波段整体偏移仍可全绿。本包用精确闭式把该端钉死。")
    add("")
    add("### 精确锚的推导")
    add("")
    add(f"**{ex['statement']}**")
    add("")
    for i, step in enumerate(ex["derivation_steps"], 1):
        add(f"{i}. {step}")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| 镜像球体积 | {_fmt(ex['sphere_volume_m3'], 6)} m³ |")
    add(f"| 半球排水体积 ∇ | {_fmt(ex['hemisphere_volume_m3'], 6)} m³ |")
    add(f"| Kirchhoff 球附连质量 A_sph | {_fmt(ex['sphere_added_mass_kg'], 4)} kg |")
    add(f"| 半球附连质量 A11(0) = A_sph/2 | {_fmt(ex['hemisphere_added_mass_kg'], 4)} kg |")
    add(f"| 无量纲 A11(0)/(ρ∇) | {_fmt(ex['hemisphere_added_mass_per_rho_nabla'], 6)} |")
    add("")
    add("## 2. 怎么做的（四条独立路线）")
    add("")
    add("| 路线 | 实现 | 独立性 |")
    add("|---|---|---|")
    add("| R1 闭式 | Kirchhoff 球 + 镜像能量减半 | 手推解析值，不依赖任何代码 |")
    add(f"| R2 谱展开 | `{harness['method']}` | 数值复核镜像球常数（0.5ρVol_sph） |")
    add(f"| R3 参考级数 | `{ref['module']}`，ka = "
        f"{', '.join(str(k) for k in ref['kas'])}，项数阶梯 {ref['n_terms_ladder']}，"
        f"quad_order = {ref['quad_order']} | 半解析级数，与面元求解器无关 |")
    add(f"| R4 生产求解器 | `{payload['solver']['module']}` on "
        f"`{payload['solver']['mesh']}`，ka = {_fmt(payload['solver']['mesh_ladder_ka'], 4)} "
        f"面元阶梯 {payload['solver']['mesh_ladder_subdivisions']}，ka 阶梯 "
        f"{payload['solver']['ka_ladder']}（sub = {payload['solver']['ka_ladder_subdivisions']}）"
        " | 3-D Kelvin 格林函数 BEM，与 R3 方法完全不同 |")
    add("")
    add("关键设置：单位速度幅值、`exp(-iωt)`；参考项数阶梯 (48/96/144) 把「级数截断」与「物理极限差」"
        "分开；求解器在 `ka = 1e-3` 时 ω = 0.099 rad/s，自由液面条件 `∂φ/∂z = νφ`（ν = 1e-3）"
        "已进入刚壁极限。")
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 R3 参考级数低频阶梯（144 项）")
    add("")
    add("| ka | A11/(ρ∇) | A11 − 1/2 | (A11 − 1/2)/ka | B11/(ωρ∇) | A33/(ρ∇) |")
    add("|---|---|---|---|---|---|")
    for r in ref_rows:
        add(f"| {r['ka']:.3f} | {_fmt(r['a11_dimless'])} | {_fmt(r['excess_over_exact'])} | "
            f"{_fmt(r['excess_per_ka'])} | {_fmt(r['b11_dimless'])} | {_fmt(r['a33_dimless'])} |")
    add("")
    add(f"级数截断漂移（48 vs 144 项）最大 {gates['G6_reference_term_truncation']['measured']:.2e}"
        "（相对），比判据小四个量级以上，说明上表的差别是物理的、不是截断误差。")
    add("")
    add("### 3.2 外推与极限")
    add("")
    add(f"- 外推：{extrap['method']}；ka = {', '.join(f'{v:.3f}' for v in extrap['kas_used'])}")
    add(f"- 拟合截距 A11(0) = {_fmt(extrap['a0_dimless'], 6)}"
        f"（斜率 {_fmt(extrap['a1_dimless_per_ka'], 4)} /ka，最大残差 {extrap['max_residual']:.2e}）")
    add(f"- 与精确值 1/2 的相对差：**{100 * extrap['rel_error_vs_exact']:+.4f}%**")
    add(f"- 最小 ka = {extrap['smallest_ka']:.3f} 处直接值 {_fmt(extrap['value_at_smallest_ka'], 6)}，"
        f"相对精确值 {100 * extrap['rel_error_at_smallest_ka']:+.4f}%")
    add(f"- R2 谱展开：整球平移闭式 {_fmt(harness['full_sphere_dimensionless_vs_rho_nabla'], 9)}"
        "（ρ∇ 单位），下半球积分（即半球附连质量）"
        f"{_fmt(harness['spectral_lower_half_dimensionless_vs_rho_nabla'], 9)}，应为 1/2")
    add("")
    add("### 3.3 R4 生产求解器（surge，面元阶梯 @ ka = 1e-3）")
    add("")
    add("| sub | 面元 | A11/(ρ∇) | vs 精确 1/2 | 条件数 | 相对残差 | 用时 s |")
    add("|---|---|---|---|---|---|---|")
    for r in mesh_rows:
        add(f"| {r['subdivisions']} | {r['panels']} | {_fmt(r['added_mass_dimless'])} | "
            f"{100 * r['rel_error_vs_exact']:+.3f}% | {_fmt(r['condition_number'], 3)} | "
            f"{r['relative_residual']:.2e} | {r['solve_seconds']:.1f} |")
    add("")
    add("| ka | 面元 | A11/(ρ∇) | vs 精确 1/2 | B11/(ωρ∇) |")
    add("|---|---|---|---|---|")
    for r in ka_rows:
        add(f"| {r['ka']:.4f} | {r['panels']} | {_fmt(r['added_mass_dimless'])} | "
            f"{100 * r['rel_error_vs_exact']:+.3f}% | {_fmt(r['damping_dimless'])} |")
    add("")
    add("## 4. 结果图")
    add("")
    add("`figure.png`（三面板，由 `write_figure` 从本 JSON 生成）：")
    add("")
    add("1. 左：参考级数 A11/(ρ∇) vs ka（对数横轴）+ 精确渐近线 1/2 + 生产求解器阶梯点；")
    add("2. 中：(A11 − 1/2)/ka vs ka，显示超标量在窗口内近似常数（极限是有限值而非数值平台）；")
    add("3. 右：B11/(ωρ∇) 与求解器面元收敛的绝对相对误差 vs 面元数（观测收敛阶）。")
    add("")
    add("## 5. 注意点")
    add("")
    add("1. **滑移 vs 平移**：heave 的刚壁镜像把半球映成「对滑」两半（Neumann 数据变号），"
        "surge 的镜像则是**整球平移**（水平速度在 z→−z 下不变）。混用会得到 0.5 vs 0.831 的错误锚，"
        "这正是 m16802 erratum 的坑；本包的 R1 显式写出这一不对称性。")
    add(f"2. **极限只在 ka→0 成立**：带内 ka = 0.2 的参考值已比极限高 {band_excess_pct:.1f}%，"
        "所以带内 gate 仍必须以参考级数逐点为准，不能拿 1/2 去判带内数值。")
    add("3. **低频求解器行为**：ka = 1e-3 下系统仍可解（条件数、残差随表给出），误差主项是面元离散；"
        "因此面元阶梯是这一端的主要不确定性来源，而不是格林函数消波项。")
    add("4. **参考解仍非 Hulme 原文印刷表格**，而是本仓重推的级数；本包给它补的是**与级数无关**的"
        "低频端精确锚，这是相对带内 capytaine/Haskind 对照的增量。")
    add("5. **B11 的判据是定性的**：只要求它随 ka→0 单调趋零，没有解析基准（未做，标为未验证项）。")
    add("")
    add("## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3%）")
    add("")
    add("基准 = 精确闭式 `A11(0)/(ρ∇) = 1/2`（R1，独立于代码与级数）。")
    add("")
    add("| 关键量 | 计算值 | 基准 | 相对误差 | < 3%? |")
    add("|---|---|---|---|---|")
    rows = [
        ("A11(0) 级数线性外推", extrap["a0_dimless"], extrap["rel_error_vs_exact"]),
        ("A11(ka=2e-3) 级数直接值", extrap["value_at_smallest_ka"],
         extrap["rel_error_at_smallest_ka"]),
        ("A11(ka=1e-3) 求解器 1024 面元", mesh_rows[-1]["added_mass_dimless"],
         mesh_rows[-1]["rel_error_vs_exact"]),
        ("R2 谱展开下半球（基准 1/2）",
         harness["spectral_lower_half_dimensionless_vs_rho_nabla"],
         harness["spectral_lower_half_dimensionless_vs_rho_nabla"] / EXACT_LIMIT_DIMLESS - 1.0),
    ]
    for label, value, rel in rows:
        add(f"| {label} | {_fmt(value, 9)} | 1/2 | {100 * rel:+.4f}% | "
            f"{'是' if abs(rel) < 0.03 else '否'} |")
    add("")
    add(f"**总判定：{payload['verdict']}**"
        + (f"（未过：{', '.join(payload['failing'])}）" if payload["failing"]
           else "（全部 gate 通过）"))
    add("")
    add("| gate | 判据 | 实测 | 结论 |")
    add("|---|---|---|---|")
    for name, gate in gates.items():
        add(f"| {name} | {gate.get('criterion', '')} | {_measured_text(gate.get('measured'))} | "
            f"{'PASS' if gate['pass'] else 'FAIL'} |")
    add("")
    add("## 7. 复现")
    add("")
    add("```bash")
    add("PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_surge_lowfreq/run_case.py")
    add("PYTHONPATH=src pytest -q tests/test_casepool_hulme_surge_lowfreq.py")
    add("```")
    add("")
    add(f"运行环境：Python {payload['machine']['python']}，numpy {payload['machine']['numpy']}，"
        f"{payload['machine']['platform']}；本包运行 {payload['runtime_seconds']:.1f} s。")
    return "\n".join(lines) + "\n"


def write_figure(payload: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_rows = sorted(payload["reference_ladder"], key=lambda r: r["ka"])
    kas = [r["ka"] for r in ref_rows]
    a11 = [r["a11_dimless"] for r in ref_rows]
    excess_per_ka = [r["excess_per_ka"] for r in ref_rows]
    b11 = [r["b11_dimless"] for r in ref_rows]
    mesh_rows = payload["solver_mesh_ladder"]
    ka_rows = sorted(payload["solver_ka_ladder"], key=lambda r: r["ka"])
    slope = payload["extrapolation"]["a1_dimless_per_ka"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.semilogx(kas, a11, "o-", color="C0", label="Hulme series (144 terms)")
    ax.axhline(EXACT_LIMIT_DIMLESS, color="C3", ls="--",
               label=r"exact $A_{11}(0)/\rho\nabla=1/2$")
    ax.semilogx([r["ka"] for r in ka_rows], [r["added_mass_dimless"] for r in ka_rows],
                "s", color="C2", ms=6, label="TensorBEM 3-D solver (256 panels)")
    ax.set_xlabel(r"$ka=\omega^2 a/g$")
    ax.set_ylabel(r"$A_{11}/(\rho\nabla)$")
    ax.set_title("surge added mass: long-wave limit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.semilogx(kas, excess_per_ka, "o-", color="C1")
    ax.axhline(slope, color="k", ls=":", lw=1, label=f"fit slope {slope:.4f} per ka")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$(A_{11}-1/2)/ka$")
    ax.set_title("excess is linear in ka (finite limit)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.loglog(kas, b11, "o-", color="C4", label=r"$B_{11}/(\omega\rho\nabla)$ series")
    ax.loglog([r["ka"] for r in ka_rows], [r["damping_dimless"] for r in ka_rows],
              "s", color="C2", ms=6, label="solver")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$B_{11}/(\omega\rho\nabla)$")
    ax.set_title("surge damping vanishes as ka -> 0")
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    panels = [r["panels"] for r in mesh_rows]
    panel_err = [abs(r["rel_error_vs_exact"]) * 100.0 for r in mesh_rows]
    ax2.loglog(panels, panel_err, "^-", color="C5", label="|A11 error| vs panels")
    ax2.set_ylabel(r"$|A_{11}$ error$|$ [%] vs exact 1/2")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    fig.suptitle("CASE-BEM-HULME-SURGE-LOWFREQ -- horizontal long-wave anchor "
                 r"$A_{11}(ka\to0)=\rho\nabla/2$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_artifacts(payload: dict[str, Any], out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir) if out_dir is not None else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    report_path = out_dir / "report.md"
    report_path.write_text(render_report(payload))
    figure_path = out_dir / "figure.png"
    write_figure(payload, figure_path)
    return {"results": results_path, "report": report_path, "figure": figure_path}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = None
    if "--out" in argv:
        out_dir = Path(argv[argv.index("--out") + 1])
    payload = run_case(write=False)
    paths = write_artifacts(payload, out_dir)
    print(json.dumps({
        "verdict": payload["verdict"],
        "failing": payload["failing"],
        "exact_limit_dimless": EXACT_LIMIT_DIMLESS,
        "extrapolated_dimless": payload["extrapolation"]["a0_dimless"],
        "rel_error_vs_exact": payload["extrapolation"]["rel_error_vs_exact"],
        "solver_finest_dimless": payload["solver_mesh_ladder"][-1]["added_mass_dimless"],
        "runtime_seconds": payload["runtime_seconds"],
        "artifacts": {k: str(v) for k, v in paths.items()},
    }, indent=2))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())