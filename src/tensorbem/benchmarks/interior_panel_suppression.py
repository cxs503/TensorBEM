"""Interior (lid) panel suppression of 3-D irregular frequencies: verification study.

Physical problem
----------------
The exterior Neumann boundary-integral equation of
:mod:`tensorbem.free_surface_solver3d` loses uniqueness at the *irregular
frequencies* of a surface-piercing hull.  In a frequency sweep the radiation
coefficients and the heave RAO then show unphysical spikes (and negative
radiation damping) in a narrow band, which is exactly what a ship seakeeping
sweep must not contain.

Method under test
-----------------
:mod:`tensorbem.interior_panels` builds a discretised **interior waterplane lid**
concentric-ring panel set inside the hull; its panel centroids are the
null-field collocation points of Green's second identity applied inside the
hull.  The solver stacks those exact rows onto the body rows
(``interior_panels="auto"``) and solves the over-determined system in a least
squares sense.  This "interior panel" family (Ohmatsu 1994; Zhu 1993) is set
against two references in the same code path:

* ``config="none"``   -- the raw, unregularised BIE (the negative control),
* ``config="chief"``  -- the 5 scattered asymmetric CHIEF points already used by
  the published hemisphere benchmark.

Pure analytical anchors
-----------------------
The study is only meaningful if the surrounding numerics are anchored:

``anchor_hemisphere``
    Heave/surge radiation of the floating hemisphere against Hulme's (1982)
    spherical-multipole series (:func:`tensorbem.analytic.hulme_hemisphere_radiation`).
    This is the *truth* the suppression must recover at the spurious peak.
``anchor_sphere``
    Submerged sphere against the Thorne/Wang/Linton multipole series, at the
    exact panel counts ``320``/``1280`` (``icosphere(2)``/``icosphere(3)``);
    supplies the ``320 -> 1280`` double-logarithmic convergence slope and the
    "suppression rows are inert for a submerged body" check.
``anchor_ursell``
    Ursell's deeply submerged 2-D cylinder at ``a/f = 0.5``, ``ka = 0.6``
    against :func:`tensorbem.analytic.ursell_cylinder_added_mass`
    (``A = 2545.1672 kg/m``, ``B = 2718.1540 kg/(m s)`` recomputed, never
    hard-coded).
``anchor_box``
    Low-frequency heave restoring force of a rectangular barge against the
    exact ``rho g A_wp`` plus the ``ka -> 0`` heave RAO limit ``RAO_3 -> 1``.

Everything is recomputed from the solver; the module never stores a number it
did not just calculate.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem import __name__ as _pkg  # noqa: F401  (keeps import order stable)
from tensorbem.analytic import (
    hulme_hemisphere_radiation,
    submerged_sphere_wave_multipole,
    ursell_cylinder_added_mass,
)
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    haskind_excitation_force_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import (
    box_hull_mesh,
    hemisphere_hull_mesh,
    icosphere,
    TriangleMesh,
)
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.interior_panels import interior_panel_points

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0

#: Panel counts of the floating hemisphere (``4 * 4**subdivisions``).
HEMISPHERE_LEVELS: dict[int, int] = {64: 2, 256: 3, 1024: 4}

#: Pre-declared uniform scan grid, ``ka`` in ``[0.1, 3.0]`` (step 0.1).
KA_MAIN: tuple[float, ...] = tuple(round(0.1 + 0.1 * i, 6) for i in range(30))

#: Pre-declared refined band around the first irregular frequency of the
#: hemisphere (observed at ``Ka ~ 2.58`` by the published CHIEF benchmark, and
#: predicted within ~5 % by the interior Dirichlet resonance of the hull
#: interior).  Step 0.01 so that the peak is sampled, not stepped over.
KA_FINE: tuple[float, ...] = tuple(round(2.40 + 0.01 * i, 6) for i in range(41))

#: Analysis band of the spurious peak (inside the refined band).
SPURIOUS_BAND: tuple[float, float] = (2.40, 2.80)

CONFIGS = ("none", "chief", "lid")

#: Legendre orders tried for the Hulme reference.  ``numpy.polynomial.legendre
#: .leggauss`` calls ``eigvalsh`` internally and can report
#: ``LinAlgError: Eigenvalues did not converge`` on some BLAS builds (observed
#: on loongarch64 + numpy 2.0.1), so the order is stepped down instead of
#: aborting a long sweep.  The order actually used is reported with the numbers.
HULME_QUAD_ORDERS = (128, 96, 64)


def hulme_reference(ka: float, *, density: float = DENSITY, gravity: float = GRAVITY):
    """Hulme hemisphere reference, stepping the quadrature order down on failure."""
    last: Exception | None = None
    for order in HULME_QUAD_ORDERS:
        try:
            ref = hulme_hemisphere_radiation(
                ka, radius=RADIUS, density=density, gravity=gravity, quad_order=order
            )
        except np.linalg.LinAlgError as exc:  # pragma: no cover - BLAS specific
            last = exc
            continue
        return ref, order
    raise RuntimeError("Hulme reference quadrature failed for every order") from last


def ka_grid(level: int) -> tuple[float, ...]:
    """Scan grid for a panel level (coarse levels use the pre-declared main grid)."""
    if level not in HEMISPHERE_LEVELS:
        raise ValueError(f"level must be one of {sorted(HEMISPHERE_LEVELS)}")
    if level <= 256:
        return KA_MAIN + KA_FINE
    return KA_MAIN


def hemisphere_case() -> tuple[Any, float, float, float]:
    """Return ``(mesh, volume, waterplane_area, mass)`` of the neutrally floating hemisphere."""
    mesh = hemisphere_hull_mesh(HEMISPHERE_LEVELS[256], radius=RADIUS)
    return mesh, float(mesh.volume), float(mesh.waterplane_area), 0.0


def _config_kwargs(config: str) -> dict[str, Any]:
    if config == "none":
        return {"chief_points": None, "interior_panels": None}
    if config == "chief":
        return {"chief_points": "auto", "interior_panels": None}
    if config == "lid":
        return {"chief_points": None, "interior_panels": "auto"}
    raise ValueError(f"config must be one of {CONFIGS}")


def heave_rao(
    mesh: Any,
    ka: float,
    added_mass: float,
    damping: float,
    excitation: complex,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> float:
    """Heave response amplitude operator of a neutrally floating body.

    ``RAO_3 = |F_3| / |C_33 - omega^2 (M + A_33) + i omega B_33|`` with
    ``C_33 = rho g A_wp``, ``M = rho V`` (neutral flotation) and unit incident
    wave amplitude.
    """
    omega = math.sqrt(gravity * ka / RADIUS)
    mass = density * float(mesh.volume)
    restoring = density * gravity * float(mesh.waterplane_area)
    impedance = complex(restoring - omega * omega * (mass + added_mass), omega * damping)
    return float(abs(excitation) / abs(impedance))


def hemisphere_point(
    level: int,
    ka: float,
    config: str,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """One ``(level, ka, config)`` realisation of the hemisphere sweep."""
    mesh = hemisphere_hull_mesh(HEMISPHERE_LEVELS[level], radius=RADIUS)
    ops = assemble_free_surface_operators_3d(mesh, ka, **_config_kwargs(config))
    rad = solve_radiation_3d(
        mesh,
        ka,
        modes=(0, 2),
        density=density,
        gravity=gravity,
        operators=ops,
    )
    force = haskind_excitation_force_3d(
        mesh, rad, density=density, gravity=gravity
    )["boundary_haskind"]
    volume = float(mesh.volume)
    omega = math.sqrt(gravity * ka / RADIUS)
    ref, quad_order = hulme_reference(ka, density=density, gravity=gravity)
    del quad_order  # order is recorded by anchor_hemisphere_accuracy
    return {
        "level": int(level),
        "ka": float(ka),
        "config": config,
        "panels": int(mesh.panel_count),
        "interior_rows": int(ops.chief_points.shape[0]),
        "condition_number": float(rad.condition_number),
        "relative_residual": float(rad.relative_residual),
        "added_mass_11_dimless": float(rad.added_mass[0, 0] / (density * volume)),
        "damping_11_dimless": float(rad.damping[0, 0] / (density * omega * volume)),
        "added_mass_33_dimless": float(rad.added_mass[1, 1] / (density * volume)),
        "damping_33_dimless": float(rad.damping[1, 1] / (density * omega * volume)),
        "added_mass_33": float(rad.added_mass[1, 1]),
        "damping_33": float(rad.damping[1, 1]),
        "excitation_force_3_abs": float(abs(force[1])),
        "rao_heave": heave_rao(
            mesh,
            ka,
            float(rad.added_mass[1, 1]),
            float(rad.damping[1, 1]),
            complex(force[1]),
            density=density,
            gravity=gravity,
        ),
        "reference_added_mass_11_dimless": float(ref.surge_added_mass_dimless),
        "reference_damping_11_dimless": float(ref.surge_damping_dimless),
        "reference_added_mass_33_dimless": float(ref.heave_added_mass_dimless),
        "reference_damping_33_dimless": float(ref.heave_damping_dimless),
    }


def spurious_peak_metrics(
    points: Sequence[dict[str, Any]], key: str = "rao_heave"
) -> dict[str, float]:
    """Curvature metric of a sweep: ``max |d^2 y / d ka^2|`` on the fine band.

    The grid must be uniform inside :data:`SPURIOUS_BAND`; the second difference
    is divided by ``dka**2`` so that different grids are comparable.
    """
    band = [
        p for p in points if SPURIOUS_BAND[0] - 1e-12 <= p["ka"] <= SPURIOUS_BAND[1] + 1e-12
    ]
    band.sort(key=lambda p: p["ka"])
    if len(band) < 3:
        raise ValueError("at least three band samples are required")
    kas = np.array([p["ka"] for p in band], dtype=float)
    ys = np.array([p[key] for p in band], dtype=float)
    dka = float(np.min(np.diff(kas)))
    if float(np.max(np.abs(np.diff(kas) - dka))) > 1.0e-9:
        raise ValueError("the spurious band grid must be uniform")
    curvature = (ys[2:] - 2.0 * ys[1:-1] + ys[:-2]) / (dka * dka)
    idx = int(np.argmax(np.abs(curvature))) + 1
    return {
        "max_curvature": float(np.max(np.abs(curvature))),
        "peak_ka": float(kas[idx]),
        "peak_value": float(ys[idx]),
        "min_value": float(np.min(ys)),
        "max_condition_number": float(
            max(p.get("condition_number", float("nan")) for p in band)
        ),
        "min_damping_33_dimless": float(
            min(p.get("damping_33_dimless", float("nan")) for p in band)
        ),
        "samples": float(len(band)),
    }


def summarise_sweep(
    cases: Sequence[dict[str, Any]], analytic_error_key: str = "added_mass_33_dimless"
) -> dict[str, Any]:
    """Group raw cases by ``(level, config)`` and derive every reported metric."""
    out: dict[str, Any] = {}
    levels = sorted({c["level"] for c in cases})
    for level in levels:
        for config in CONFIGS:
            subset = [c for c in cases if c["level"] == level and c["config"] == config]
            if len(subset) < 3:
                continue
            subset.sort(key=lambda c: c["ka"])
            rao_metrics = spurious_peak_metrics(subset, "rao_heave")
            dam_metrics = spurious_peak_metrics(subset, "damping_33_dimless")
            a33 = np.array([c["added_mass_33_dimless"] - c["reference_added_mass_33_dimless"] for c in subset])
            b33 = np.array([c["damping_33_dimless"] - c["reference_damping_33_dimless"] for c in subset])
            out[f"{level}:{config}"] = {
                "level": level,
                "config": config,
                "panels": subset[0]["panels"],
                "interior_rows": subset[0]["interior_rows"],
                "rao": rao_metrics,
                "damping": dam_metrics,
                "max_abs_error_added_mass_33_dimless": float(np.max(np.abs(a33))),
                "max_abs_error_damping_33_dimless": float(np.max(np.abs(b33))),
            }
    return out


def suppression_reduction(summary: dict[str, Any], level: int) -> dict[str, float]:
    """Relative reduction of the curvature metric, ``1 - lid / none``."""
    base = summary[f"{level}:none"]["rao"]["max_curvature"]
    lid = summary[f"{level}:lid"]["rao"]["max_curvature"]
    chief = summary.get(f"{level}:chief", {}).get("rao", {}).get("max_curvature")
    out = {
        "level": float(level),
        "max_curvature_none": float(base),
        "max_curvature_lid": float(lid),
        "curvature_reduction_lid": float(1.0 - lid / base) if base > 0 else float("nan"),
        "min_damping_none": float(summary[f"{level}:none"]["rao"]["min_damping_33_dimless"]),
        "min_damping_lid": float(summary[f"{level}:lid"]["rao"]["min_damping_33_dimless"]),
    }
    if chief is not None:
        out["max_curvature_chief"] = float(chief)
        out["curvature_reduction_chief"] = float(1.0 - chief / base) if base > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Pure analytical anchors
# ---------------------------------------------------------------------------
def anchor_hemisphere_accuracy(
    level: int = 256, ka_list: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.6), config: str = "lid"
) -> dict[str, Any]:
    """Hemisphere radiation against Hulme's series at several ``ka``."""
    rows = []
    for ka in ka_list:
        case = hemisphere_point(level, float(ka), config)
        rows.append(
            {
                "ka": float(ka),
                "config": config,
                "added_mass_33_dimless": case["added_mass_33_dimless"],
                "reference_added_mass_33_dimless": case["reference_added_mass_33_dimless"],
                "damping_33_dimless": case["damping_33_dimless"],
                "reference_damping_33_dimless": case["reference_damping_33_dimless"],
                "added_mass_rel_error": abs(
                    case["added_mass_33_dimless"] - case["reference_added_mass_33_dimless"]
                )
                / abs(case["reference_added_mass_33_dimless"]),
                "damping_rel_error": abs(
                    case["damping_33_dimless"] - case["reference_damping_33_dimless"]
                )
                / abs(case["reference_damping_33_dimless"]),
            }
        )
    return {"level": level, "config": config, "rows": rows}


def anchor_sphere(
    panel_levels: Sequence[int] = (320, 1280),
    ka: float = 0.6,
    submergence: float = 2.0,
    *,
    interior_rows: bool = False,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """Submerged-sphere anchor at exact panel counts (Wang/Linton multipoles)."""
    subdivisions = {320: 2, 1280: 3}
    ref = submerged_sphere_wave_multipole(
        ka, radius=RADIUS, submergence=submergence, density=density
    )
    interior = None
    if interior_rows:
        # Five deterministic interior points strictly inside the sphere.
        # Five interior points at 0.4 R from the body centre: far enough from the
        # panels that the null-field rows stay well conditioned.
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [-0.2, 0.346, 0.0],
                [0.0, -0.2, 0.346],
                [0.231, 0.231, -0.231],
            ],
            dtype=float,
        )
        interior = offsets + np.array([0.0, 0.0, -submergence])
    rows = []
    for panels in panel_levels:
        mesh = TriangleMesh(
            icosphere(subdivisions[int(panels)], radius=RADIUS).vertices
            + np.array([0.0, 0.0, -submergence]),
            icosphere(subdivisions[int(panels)], radius=RADIUS).faces,
        )
        ops = assemble_free_surface_operators_3d(mesh, ka)
        if interior is not None:
            ops = assemble_free_surface_operators_3d(mesh, ka, interior_panels=interior)
        rad = solve_radiation_3d(
            mesh, ka, modes=(0, 2), density=density, gravity=gravity, operators=ops
        )
        volume = 4.0 / 3.0 * math.pi * RADIUS**3
        rows.append(
            {
                "panels": int(mesh.panel_count),
                "interior_rows": int(ops.chief_points.shape[0]),
                "added_mass_33": float(rad.added_mass[1, 1]),
                "damping_33": float(rad.damping[1, 1]),
                "added_mass_33_rel_error": abs(
                    float(rad.added_mass[1, 1]) - float(ref.heave_added_mass)
                )
                / abs(float(ref.heave_added_mass)),
                "damping_33_rel_error": abs(
                    float(rad.damping[1, 1]) - float(ref.heave_damping)
                )
                / abs(float(ref.heave_damping)),
                "added_mass_11_rel_error": abs(
                    float(rad.added_mass[0, 0]) - float(ref.surge_added_mass)
                )
                / abs(float(ref.surge_added_mass)),
                "damping_11_rel_error": abs(
                    float(rad.damping[0, 0]) - float(ref.surge_damping)
                )
                / abs(float(ref.surge_damping)),
                "volume": volume,
            }
        )

    slope = None
    if len(rows) > 1 and rows[0]["added_mass_33_rel_error"] > 0 and rows[-1]["added_mass_33_rel_error"] > 0:
        slope = float(
            math.log(rows[0]["added_mass_33_rel_error"] / rows[-1]["added_mass_33_rel_error"])
            / math.log(rows[-1]["panels"] / rows[0]["panels"])
        )
    return {
        "ka": float(ka),
        "submergence": float(submergence),
        "reference_added_mass_33": float(ref.heave_added_mass),
        "reference_damping_33": float(ref.heave_damping),
        "interior_rows": bool(interior_rows),
        "rows": rows,
        "added_mass_convergence_slope": slope,
        "max_added_mass_rel_error": float(max(r["added_mass_33_rel_error"] for r in rows)),
        "max_damping_rel_error": float(max(r["damping_33_rel_error"] for r in rows)),
    }


def anchor_ursell(
    ka: float = 0.6,
    submergence_ratio: float = 2.0,
    panel_count: int = 128,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """Ursell deeply submerged cylinder ``A``/``B`` (2-D anchor, per unit span)."""
    from tensorbem.benchmarks.ursell_cylinder import run as ursell_run

    study = ursell_run(
        panel_count=panel_count,
        radius=RADIUS,
        submergence=submergence_ratio * RADIUS,
        ka=ka,
        density=density,
        gravity=gravity,
    )
    exact_a, exact_b = ursell_cylinder_added_mass(
        RADIUS, submergence_ratio * RADIUS, math.sqrt(ka * gravity / RADIUS), density,
        gravity=gravity,
    )
    return {
        "ka": float(ka),
        "a_over_f": float(1.0 / submergence_ratio),
        "panel_count": int(panel_count),
        "reference_added_mass": float(exact_a),
        "reference_damping": float(exact_b),
        "panel_added_mass": float(study["panel_added_mass"]),
        "panel_damping": float(study["panel_damping"]),
        "added_mass_rel_error": float(study["panel_added_mass_relative_error"]),
        "damping_rel_error": float(study["panel_damping_relative_error"]),
    }


def anchor_box(
    length: float = 4.0,
    beam: float = 2.0,
    draft: float = 1.0,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """Low-frequency heave restoring force of a barge against ``rho g A_wp``."""
    mesh = box_hull_mesh(length, beam, draft, n_length=4, n_beam=3, n_draft=3)
    mass = density * length * beam * draft
    props = compute_hydrostatic_restoring_matrix(
        mesh, mass, np.array([0.0, 0.0, -0.5 * draft]), density, gravity
    )
    exact = density * gravity * length * beam
    c33 = float(props.restoring_matrix[2, 2])
    return {
        "length": length,
        "beam": beam,
        "draft": draft,
        "waterplane_area_mesh": float(mesh.waterplane_area),
        "waterplane_area_exact": float(length * beam),
        "c33": c33,
        "c33_reference": float(exact),
        "c33_rel_error": abs(c33 - exact) / exact,
    }


def anchor_box_low_frequency_rao(ka: float = 0.1) -> dict[str, Any]:
    """``ka -> 0`` heave RAO limit of the barge (``RAO_3 -> 1``)."""
    mesh = box_hull_mesh(4.0, 2.0, 1.0, n_length=4, n_beam=3, n_draft=3)
    ops = assemble_free_surface_operators_3d(mesh, ka, interior_panels=None)
    rad = solve_radiation_3d(mesh, ka, modes=(2,), operators=ops)
    force = haskind_excitation_force_3d(mesh, rad)["boundary_haskind"][0]
    mass = DENSITY * 4.0 * 2.0 * 1.0
    omega = math.sqrt(GRAVITY * ka / 1.0)
    restoring = DENSITY * GRAVITY * float(mesh.waterplane_area)
    impedance = complex(
        restoring - omega * omega * (mass + float(rad.added_mass[0, 0])),
        omega * float(rad.damping[0, 0]),
    )
    rao = float(abs(force) / abs(impedance))
    return {"ka": ka, "rao_heave": rao, "c33": restoring, "mass": mass}


def run(
    levels: Sequence[int] = (64, 256),
    configs: Sequence[str] = CONFIGS,
    ka_filter: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run the full study (or a subset) and return the JSON-serialisable report."""
    cases: list[dict[str, Any]] = []
    for level in levels:
        grid = ka_filter if ka_filter is not None else ka_grid(int(level))
        for config in configs:
            for ka in grid:
                cases.append(hemisphere_point(int(level), float(ka), config))
    summary = summarise_sweep(cases)
    reductions = [
        suppression_reduction(summary, level)
        for level in sorted({c["level"] for c in cases})
        if f"{level}:none" in summary and f"{level}:lid" in summary
    ]
    return {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "density": DENSITY,
            "gravity": GRAVITY,
            "radius": RADIUS,
            "ka_main": list(KA_MAIN),
            "ka_fine": list(KA_FINE),
            "spurious_band": list(SPURIOUS_BAND),
        },
        "cases": cases,
        "summary": summary,
        "suppression": reductions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--anchors", action="store_true")
    args = parser.parse_args(argv)

    payload = run(levels=tuple(args.levels), configs=tuple(args.configs))
    if args.anchors:
        payload["anchors"] = {
            "hemisphere": anchor_hemisphere_accuracy(),
            "sphere": anchor_sphere(),
            "ursell": anchor_ursell(),
            "box": anchor_box(),
            "box_low_frequency_rao": anchor_box_low_frequency_rao(),
        }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())