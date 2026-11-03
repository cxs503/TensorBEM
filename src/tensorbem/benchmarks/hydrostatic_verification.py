"""Closed-form verification of the hydrostatic / mass / long-wave RAO chain.

Four independent checks, each with an analytic reference, are bundled here:

1. **Wall-sided barge** (:func:`run_barge_verification`). Every panel of
   :func:`tensorbem.geometry3d.box_hull_mesh` is planar and every waterline edge
   is a straight segment, so the boundary integrals of
   :mod:`tensorbem.hydrostatic` must reproduce the classical barge formulas

   .. math::
       \\nabla = L B T,\\quad A_{wp} = LB,\\quad I_T = \\frac{L B^3}{12},
       \\quad I_L = \\frac{B L^3}{12},\\quad
       BM_T = \\frac{B^2}{12T},\\quad BM_L = \\frac{L^2}{12T}

   to machine precision. This separates *integrator* correctness from geometric
   discretisation error.

2. **Floating hemisphere convergence**
   (:func:`run_hemisphere_convergence`). The radially projected octahedron of
   :func:`tensorbem.geometry3d.hemisphere_hull_mesh` is an inscribed polyhedron,
   so the computed hydrostatics approach the sphere values with the panel size.
   The study reports relative errors together with the observed convergence
   order and therefore answers "how many panels are needed for < 1 %?".

3. **Rigid-body mass matrix identity** (:func:`run_mass_matrix_energy_check`).
   For random rigid-body velocities the quadratic form must equal
   ``0.5*m*|v + omega x r_g|^2 + 0.5*omega.J_cg.omega``.

4. **Long-wave heave RAO** (:func:`run_long_wave_heave_check`). For a
   wall-sided body in the long-wave limit the heave RAO tends to one; the
   residual is a genuine end-to-end check of hydrostatics + Froude-Krylov
   excitation + added mass + impedance solve.

Every number in the JSON artifact is produced programmatically (no hand
transcription), and ``main()`` renders the report figures.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..geometry3d import box_hull_mesh, hemisphere_hull_mesh
from ..hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
    compute_waterplane_properties,
)
from ..rao import compute_rao_at_frequency

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665


@dataclass(frozen=True)
class Comparison:
    """One computed-vs-reference scalar pair with its relative error."""

    name: str
    computed: float
    reference: float

    @property
    def relative_error(self) -> float:
        scale = abs(self.reference)
        if scale == 0.0:
            return abs(self.computed)
        return abs(self.computed - self.reference) / scale

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "computed": float(self.computed),
            "reference": float(self.reference),
            "relative_error": float(self.relative_error),
        }


# --------------------------------------------------------------------------- #
# 1. Wall-sided barge: machine-precision verification
# --------------------------------------------------------------------------- #
def barge_reference(
    length: float,
    beam: float,
    draft: float,
    kg: float,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Closed-form hydrostatics of a wall-sided rectangular barge.

    ``kg`` is the centre-of-gravity height **above the keel** (m).
    """
    volume = length * beam * draft
    area = length * beam
    i_t = length * beam**3 / 12.0        # about the longitudinal (x) axis
    i_l = beam * length**3 / 12.0        # about the transverse (y) axis
    kb = 0.5 * draft
    bm_t = i_t / volume
    bm_l = i_l / volume
    gm_t = kb + bm_t - kg
    gm_l = kb + bm_l - kg
    return {
        "displaced_volume": volume,
        "waterplane_area": area,
        "waterplane_i_xx": i_t,
        "waterplane_i_yy": i_l,
        "center_of_buoyancy_z": -0.5 * draft,
        "kb_above_keel": kb,
        "bm_transverse": bm_t,
        "bm_longitudinal": bm_l,
        "gm_transverse": gm_t,
        "gm_longitudinal": gm_l,
        "c_heave": density * gravity * area,
        "c_roll": density * gravity * volume * gm_t,
        "c_pitch": density * gravity * volume * gm_l,
    }


def run_barge_verification(
    length: float = 30.0,
    beam: float = 8.0,
    draft: float = 2.5,
    kg_fraction: float = 0.5,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    panelizations: Sequence[tuple[int, int]] = ((1, 1), (3, 2)),
    center_xy: tuple[float, float] = (2.0, -1.0),
) -> dict[str, Any]:
    """Compare barge hydrostatics against the closed-form barge formulas."""
    kg = kg_fraction * draft
    ref = barge_reference(length, beam, draft, kg, density, gravity)
    mass = density * ref["displaced_volume"]
    cg = np.array([0.0, 0.0, -draft + kg])

    cases: list[dict[str, Any]] = []
    for nx, ny in panelizations:
        hull = box_hull_mesh(
            length,
            beam,
            draft,
            center_xy=center_xy,
            length_subdivisions=nx,
            beam_subdivisions=ny,
        )
        props = compute_hydrostatic_restoring_matrix(
            hull, mass, cg, density=density, gravity=gravity
        )
        volume, cb = compute_displaced_volume_and_buoyancy(hull)
        area, wp_center, moments = compute_waterplane_properties(hull)
        c = props.restoring_matrix

        comparisons = [
            Comparison("displaced_volume", volume, ref["displaced_volume"]),
            Comparison("center_of_buoyancy_z", float(cb[2]), ref["center_of_buoyancy_z"]),
            Comparison("center_of_buoyancy_x", float(cb[0]), center_xy[0]),
            Comparison("center_of_buoyancy_y", float(cb[1]), center_xy[1]),
            Comparison("waterplane_area", area, ref["waterplane_area"]),
            Comparison("waterplane_center_x", float(wp_center[0]), center_xy[0]),
            Comparison("waterplane_center_y", float(wp_center[1]), center_xy[1]),
            Comparison("waterplane_i_xx", float(moments[0]), ref["waterplane_i_xx"]),
            Comparison("waterplane_i_yy", float(moments[1]), ref["waterplane_i_yy"]),
            Comparison("waterplane_i_xy", float(moments[2]), 0.0),
            Comparison(
                "gm_transverse",
                float(c[3, 3]) / (density * gravity * props.displaced_volume),
                ref["gm_transverse"],
            ),
            Comparison(
                "gm_longitudinal",
                float(c[4, 4]) / (density * gravity * props.displaced_volume),
                ref["gm_longitudinal"],
            ),
            Comparison("c_heave", float(c[2, 2]), ref["c_heave"]),
            Comparison("c_roll", float(c[3, 3]), ref["c_roll"]),
            Comparison("c_pitch", float(c[4, 4]), ref["c_pitch"]),
        ]
        cases.append(
            {
                "length_subdivisions": int(nx),
                "beam_subdivisions": int(ny),
                "panel_count": int(hull.panel_count),
                "max_relative_error": max(
                    (cmp.relative_error for cmp in comparisons), default=0.0
                ),
                "comparisons": [cmp.as_dict() for cmp in comparisons],
            }
        )

    return {
        "case": {
            "length": length,
            "beam": beam,
            "draft": draft,
            "kg_above_keel": kg,
            "density": density,
            "gravity": gravity,
            "waterplane_center": list(center_xy),
        },
        "reference": ref,
        "panelizations": cases,
    }


# --------------------------------------------------------------------------- #
# 2. Floating hemisphere: convergence against the sphere solution
# --------------------------------------------------------------------------- #
def hemisphere_reference(
    radius: float = 1.0,
    kg: float | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Closed-form hydrostatics of a floating hemisphere of ``radius``.

    ``kb`` is measured from the keel (the pole of the hemisphere), so
    ``KB = R - 3R/8 = 5R/8``; ``BM_T = I_T / nabla = 3R/8``.
    """
    r = float(radius)
    volume = 2.0 * math.pi * r**3 / 3.0
    area = math.pi * r**2
    i_xx = math.pi * r**4 / 4.0
    kb = r - 3.0 * r / 8.0
    bm_t = i_xx / volume
    kg_keel = 0.5 * r if kg is None else float(kg)
    gm_t = kb + bm_t - kg_keel
    return {
        "displaced_volume": volume,
        "waterplane_area": area,
        "waterplane_i_xx": i_xx,
        "waterplane_i_yy": i_xx,
        "center_of_buoyancy_z": -3.0 * r / 8.0,
        "kb_above_keel": kb,
        "bm_transverse": bm_t,
        "gm_transverse": gm_t,
        "c_heave": density * gravity * area,
        "c_roll": density * gravity * volume * gm_t,
    }


def run_hemisphere_convergence(
    subdivisions: Sequence[int] = (1, 2, 3, 4, 5),
    radius: float = 1.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Panel-refinement study of hemisphere hydrostatics vs closed forms."""
    kg = 5.0 * radius / 8.0   # CG at the centroid of the solid, 5R/8 above keel
    ref = hemisphere_reference(radius, kg, density, gravity)
    mass = density * ref["displaced_volume"]
    cg = np.array([0.0, 0.0, -radius + kg])

    rows: list[dict[str, Any]] = []
    for sub in subdivisions:
        hull = hemisphere_hull_mesh(subdivisions=int(sub), radius=radius)
        props = compute_hydrostatic_restoring_matrix(
            hull, mass, cg, density=density, gravity=gravity
        )
        kb = float(props.center_of_buoyancy[2] + radius)
        bm = float(props.waterplane_moments[0]) / props.displaced_volume
        gm = kb + bm - kg
        surface = 4.0 * math.pi * radius**2
        h = math.sqrt(hull.surface_area / max(hull.panel_count, 1))
        rows.append(
            {
                "subdivisions": int(sub),
                "panel_count": int(hull.panel_count),
                "panel_size": h,
                "relative_panel_size": h / surface**0.5,
                "volume_relative_error": abs(props.displaced_volume - ref["displaced_volume"])
                / ref["displaced_volume"],
                "waterplane_area_relative_error": abs(props.waterplane_area - ref["waterplane_area"])
                / ref["waterplane_area"],
                "waterplane_i_xx_relative_error": abs(
                    float(props.waterplane_moments[0]) - ref["waterplane_i_xx"]
                )
                / ref["waterplane_i_xx"],
                "kb_relative_error": abs(kb - ref["kb_above_keel"]) / ref["kb_above_keel"],
                "gm_relative_error": abs(gm - ref["gm_transverse"]) / ref["gm_transverse"],
                "computed_volume": float(props.displaced_volume),
                "computed_waterplane_area": float(props.waterplane_area),
                "computed_gm_transverse": float(gm),
            }
        )

    fields = (
        "volume_relative_error",
        "waterplane_area_relative_error",
        "waterplane_i_xx_relative_error",
        "kb_relative_error",
        "gm_relative_error",
    )
    orders = {field: observed_order(rows, field) for field in fields}
    return {
        "case": {
            "radius": radius,
            "kg_above_keel": kg,
            "density": density,
            "gravity": gravity,
        },
        "reference": ref,
        "levels": rows,
        "observed_order": orders,
    }


def observed_order(rows: Sequence[dict[str, Any]], field: str) -> float:
    """Least-squares slope of ``log(error)`` against ``log(panel size)``."""
    usable = [(r["panel_size"], r[field]) for r in rows if r[field] > 0.0]
    if len(usable) < 2:
        return float("nan")
    h = np.log(np.array([u[0] for u in usable], dtype=float))
    e = np.log(np.array([u[1] for u in usable], dtype=float))
    design = np.vstack((h, np.ones_like(h))).T
    slope = np.linalg.lstsq(design, e, rcond=None)[0][0]
    return float(slope)


# --------------------------------------------------------------------------- #
# 3. Rigid-body mass matrix: kinetic-energy identity
# --------------------------------------------------------------------------- #
def run_mass_matrix_energy_check(
    mass: float = 1000.0,
    center_of_gravity: Sequence[float] = (0.3, -0.4, -0.6),
    inertia: Sequence[Sequence[float]] | None = None,
    samples: int = 16,
    seed: int = 20261103,
) -> dict[str, Any]:
    """Check ``0.5*x^T M x`` against the rigid-body kinetic energy."""
    r_g = np.asarray(center_of_gravity, dtype=float)
    j_cg = np.diag([5.0e2, 7.0e2, 9.0e2]) if inertia is None else np.asarray(inertia, dtype=float)
    m = compute_mass_matrix(mass, r_g, inertia=j_cg)

    rng = np.random.default_rng(seed)
    errors: list[float] = []
    for _ in range(samples):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        x = np.concatenate((v, w))
        quadratic = 0.5 * float(x @ m @ x)
        v_g = v + np.cross(w, r_g)
        energy = 0.5 * mass * float(v_g @ v_g) + 0.5 * float(w @ j_cg @ w)
        errors.append(abs(quadratic - energy) / max(abs(energy), 1e-12))

    return {
        "mass": mass,
        "center_of_gravity": r_g.tolist(),
        "inertia_about_cg": j_cg.tolist(),
        "samples": int(samples),
        "symmetric": bool(np.allclose(m, m.T, atol=1e-12)),
        "max_relative_energy_error": float(max(errors)),
        "coupling_block": m[:3, 3:].tolist(),
    }


# --------------------------------------------------------------------------- #
# 4. Long-wave heave RAO of a wall-sided barge
# --------------------------------------------------------------------------- #
def run_long_wave_heave_check(
    length: float = 30.0,
    beam: float = 8.0,
    draft: float = 2.5,
    omegas: Sequence[float] = (0.2, 0.4, 0.6),
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Heave RAO of a wall-sided barge against the long-wave limit ``RAO -> 1``."""
    hull = box_hull_mesh(length, beam, draft)
    mass = density * length * beam * draft
    cg = np.array([0.0, 0.0, -0.5 * draft])
    props = compute_hydrostatic_restoring_matrix(hull, mass, cg, density, gravity)

    rows: list[dict[str, Any]] = []
    for omega in omegas:
        result = compute_rao_at_frequency(
            hull,
            float(omega),
            0.0,
            mass,
            cg,
            hydrostatic_props=props,
            density=density,
            gravity=gravity,
            chief_points=None,
        )
        amplitude = float(abs(result.rao[2]))
        rows.append(
            {
                "omega": float(omega),
                "ka": float(omega * omega * length / (2.0 * gravity)),
                "heave_rao_amplitude": amplitude,
                "relative_error_vs_long_wave_limit": abs(amplitude - 1.0),
                "heave_rao_phase": float(np.angle(result.rao[2])),
                "impedance_residual": float(result.impedance_residual),
            }
        )
    return {
        "case": {
            "length": length,
            "beam": beam,
            "draft": draft,
            "panel_count": int(hull.panel_count),
            "density": density,
            "gravity": gravity,
        },
        "levels": rows,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(
    *,
    quick: bool = False,
    subdivisions: Sequence[int] | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Execute the full hydrostatic verification suite."""
    subs = (1, 2, 3, 4) if quick else tuple((1, 2, 3, 4, 5) if subdivisions is None else subdivisions)
    omegas = (0.2, 0.4) if quick else (0.2, 0.4, 0.6)
    return {
        "benchmark": "hydrostatic_analytic_verification",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "barge": run_barge_verification(density=density, gravity=gravity),
        "hemisphere_convergence": run_hemisphere_convergence(
            subdivisions=subs, density=density, gravity=gravity
        ),
        "mass_matrix_energy": run_mass_matrix_energy_check(),
        "long_wave_heave_rao": run_long_wave_heave_check(
            omegas=omegas, density=density, gravity=gravity
        ),
    }


def render_figures(report: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render the three verification figures and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Figure 1: hemisphere convergence against the analytic sphere solution.
    levels = report["hemisphere_convergence"]["levels"]
    h = np.array([row["panel_size"] for row in levels])
    fields = (
        ("volume_relative_error", r"displaced volume $\nabla$"),
        ("waterplane_area_relative_error", r"waterplane area $A_{wp}$"),
        ("waterplane_i_xx_relative_error", r"$I_{xx}$"),
        ("kb_relative_error", r"$KB$ (from keel)"),
        ("gm_relative_error", r"$GM_T$"),
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for field, label in fields:
        err = np.array([row[field] for row in levels])
        ax.loglog(h, err, "o-", label=label)
    guide = 1.1e-1 * (h / h[0]) ** 2
    ax.loglog(h, guide, "k--", lw=1.0, label=r"$O(h^2)$ guide")
    ax.set_xlabel("panel size h (m)")
    ax.set_ylabel("relative error vs analytic sphere")
    ax.set_title("Floating hemisphere: hydrostatic convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "hydrostatic_hemisphere_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # Figure 2: error budget for the barge (machine precision) versus the
    # hemisphere at the coarse mesh used by the original test suite (s=2) and
    # at the convergent mesh (s=4).
    barge = report["barge"]["panelizations"][0]
    barge_err = {cmp["name"]: cmp["relative_error"] for cmp in barge["comparisons"]}
    by_sub = {row["subdivisions"]: row for row in levels}
    names = ["displaced_volume", "waterplane_area", "waterplane_i_xx", "gm_transverse"]
    labels = [r"$\nabla$", r"$A_{wp}$", r"$I_{xx}$", r"$GM_T$"]

    def hemi_error(row: dict[str, Any], name: str) -> float:
        mapping = {
            "displaced_volume": "volume_relative_error",
            "waterplane_area": "waterplane_area_relative_error",
            "waterplane_i_xx": "waterplane_i_xx_relative_error",
            "gm_transverse": "gm_relative_error",
        }
        return float(row[mapping[name]])

    coarse = by_sub.get(2, levels[0])
    fine = by_sub.get(4, levels[-1])
    barge_map = {
        "displaced_volume": barge_err["displaced_volume"],
        "waterplane_area": barge_err["waterplane_area"],
        "waterplane_i_xx": barge_err["waterplane_i_xx"],
        "gm_transverse": barge_err["gm_transverse"],
    }
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    floor = 1e-18
    ax.bar(
        x - width,
        [max(barge_map[n], 0.0) + floor for n in names],
        width,
        label=f"barge (planar panels, {barge['panel_count']} panels)",
    )
    ax.bar(
        x,
        [max(hemi_error(coarse, n), floor) for n in names],
        width,
        label=f"hemisphere subdivisions={coarse['subdivisions']} ({coarse['panel_count']} panels)",
    )
    ax.bar(
        x + width,
        [max(hemi_error(fine, n), floor) for n in names],
        width,
        label=f"hemisphere subdivisions={fine['subdivisions']} ({fine['panel_count']} panels)",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("relative error vs closed form")
    ax.set_title("Hydrostatic error budget (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "hydrostatic_error_budget.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # Figure 3: long-wave heave RAO against the limit value 1.
    rows = report["long_wave_heave_rao"]["levels"]
    om = [row["omega"] for row in rows]
    amp = [row["heave_rao_amplitude"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(om, amp, "o-", label="BEM heave RAO (barge)")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="long-wave limit (RAO = 1)")
    for xv, yv, row in zip(om, amp, rows):
        ax.annotate(
            f"{row['relative_error_vs_long_wave_limit'] * 100:.3f}%",
            (xv, yv),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=7,
        )
    ax.set_xlabel(r"wave frequency $\omega$ (rad/s)")
    ax.set_ylabel("heave RAO (-)")
    ax.set_title("Wall-sided barge: heave RAO approaches the long-wave limit")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "hydrostatic_long_wave_heave_rao.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--output", type=Path, help="write machine-readable JSON here")
    parser.add_argument("--figures", type=Path, help="directory for the PNG figures")
    args = parser.parse_args()

    report = run(quick=args.quick)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.figures is not None:
        for path in render_figures(report, args.figures):
            print(f"wrote {path}")

    barge = report["barge"]["panelizations"][0]
    print(f"barge max relative error: {barge['max_relative_error']:.3e}")
    for row in report["hemisphere_convergence"]["levels"]:
        print(
            f"subdivisions={row['subdivisions']:2d} panels={row['panel_count']:5d} "
            f"volume={row['volume_relative_error']:.3e} GM_T={row['gm_relative_error']:.3e}"
        )
    print(
        "observed order (volume): "
        f"{report['hemisphere_convergence']['observed_order']['volume_relative_error']:.2f}"
    )
    print(
        "max mass-matrix energy error: "
        f"{report['mass_matrix_energy']['max_relative_energy_error']:.3e}"
    )


if __name__ == "__main__":
    main()