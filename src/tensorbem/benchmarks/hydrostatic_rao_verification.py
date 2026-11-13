"""Closed-form verification of the hydrostatic restoring + 6-DOF RAO chain.

The hydrostatic/RAO layer (``tensorbem.hydrostatic`` and ``tensorbem.rao``) is a
*post-processing* layer: it consumes panel-method radiation/diffraction results
and turns them into restoring matrices, rigid-body loads and RAOs.  Its own
correctness therefore cannot be established by the radiation/diffraction
benchmarks alone, and it must not be accepted on plausibility checks such as
"the RAO is positive and finite".

This module verifies that layer against references that are independent of the
panel operators:

1. **Rectangular pontoon (box) hydrostatics** — exact closed-form values for
   waterplane area, displaced volume, second moments and the restoring matrix
   (Biran & Pulido, *Ship Hydrostatics and Stability*; Rawson & Tupper,
   *Basic Ship Theory*):

       A_wp = L B,  nabla = L B T,  I_T = L B^3/12,  I_L = B L^3/12
       C33 = rho g A_wp,  C44 = rho g nabla GM_T,  C55 = rho g nabla GM_L
       GM_T = KB + BM_T - KG,  KB = T/2,  BM_T = I_T/nabla

2. **Rigid-body mass matrix** — exact block form
   ``M = [[m I, -m S^T], [-m S, J_o + m S S^T]]`` with ``S = skew(r_G)``.

3. **Hulme (1982) floating hemisphere** — literature coefficients
   ``A11, B11, A33, B33`` and complex exciting forces ``F1, F3``
   (J. Fluid Mech. 121:443-463).  The reference is evaluated with the
   independent multi-pole series in :mod:`tensorbem.analytic`.

4. **Long-wave heave limit** — for a freely floating body in deep water the
   heave RAO must tend to unity as ``ka -> 0`` (the body follows the free
   surface), so ``|xi_3(ka=0.2)| = 1`` within panel discretisation error.

5. **Chain integrity** — the A/B matrices and exciting forces used inside the
   RAO solve must be bit-comparable with the stand-alone free-surface solves,
   and the returned impedance residual ``|Z xi - F|`` must vanish.

Nothing here is a model test: the references are closed form or published.

Run ``python -m tensorbem.benchmarks.hydrostatic_rao_verification`` for JSON
and ``--figure`` for the comparison plot.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..analytic import hulme_hemisphere_radiation
from ..free_surface_solver3d import solve_diffraction_3d, solve_radiation_3d
from ..geometry3d import WettedHullMesh, hemisphere_hull_mesh
from ..rigid_body_mass import energy_residual
from ..hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
    compute_waterplane_properties,
)
from ..rao import compute_rao_at_frequency

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


def _rel(value: float, reference: float) -> float:
    """Relative error, guarded against a vanishing reference."""
    scale = abs(reference)
    if scale == 0.0:
        return abs(value)
    return abs(value - reference) / scale


def box_pontoon_hull(
    length: float = 4.0, beam: float = 2.0, draft: float = 1.0
) -> WettedHullMesh:
    """Wetted surface of a rectangular pontoon: bottom plus four flat sides.

    Ten triangles, every face planar, so both the waterplane polygon and the
    enclosed volume are represented *exactly* by the mesh.  The hull is open at
    ``z = 0`` (no waterplane lid), as required by :class:`WettedHullMesh`.
    """
    half_l, half_b = 0.5 * length, 0.5 * beam
    base = [(-half_l, -half_b), (half_l, -half_b), (half_l, half_b), (-half_l, half_b)]
    vertices = [(x, y, -draft) for x, y in base] + [(x, y, 0.0) for x, y in base]
    bottom = [0, 1, 2, 3]
    top = lambda i: i + 4  # noqa: E731 - keel/waterline index offset
    faces = [
        (bottom[0], bottom[2], bottom[1]),
        (bottom[0], bottom[3], bottom[2]),
    ]
    for i in range(4):
        j = (i + 1) % 4
        faces.append((bottom[i], bottom[j], top(j)))
        faces.append((bottom[i], top(j), top(i)))
    return WettedHullMesh(np.array(vertices, dtype=float), np.array(faces, dtype=np.int64))


def box_reference(
    length: float = 4.0,
    beam: float = 2.0,
    draft: float = 1.0,
    kg: float = 0.5,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Exact hydrostatics of a wall-sided pontoon (z=0 free surface, z=-T keel)."""
    volume = length * beam * draft
    i_transverse = length * beam**3 / 12.0
    i_longitudinal = beam * length**3 / 12.0
    kb = 0.5 * draft
    bm_t = i_transverse / volume
    bm_l = i_longitudinal / volume
    gm_t = kb + bm_t - kg
    gm_l = kb + bm_l - kg
    return {
        "waterplane_area": length * beam,
        "displaced_volume": volume,
        "waterplane_moment_xx": i_transverse,
        "waterplane_moment_yy": i_longitudinal,
        "center_of_buoyancy_z": -0.5 * draft,
        "kb": kb,
        "bm_transverse": bm_t,
        "bm_longitudinal": bm_l,
        "gm_transverse": gm_t,
        "gm_longitudinal": gm_l,
        "c33": density * gravity * length * beam,
        "c44": density * gravity * volume * gm_t,
        "c55": density * gravity * volume * gm_l,
    }


def run_box_hydrostatics(
    length: float = 4.0,
    beam: float = 2.0,
    draft: float = 1.0,
    kg: float = 0.5,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Compare the panel-mesh hydrostatics with the closed-form pontoon values."""
    hull = box_pontoon_hull(length, beam, draft)
    ref = box_reference(length, beam, draft, kg, density, gravity)

    area, centroid, moments = compute_waterplane_properties(hull)
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)
    props = compute_hydrostatic_restoring_matrix(
        hull, density * volume, np.array([0.0, 0.0, -kg]), density, gravity
    )
    restoring = props.restoring_matrix

    measured = {
        "waterplane_area": area,
        "displaced_volume": volume,
        "waterplane_moment_xx": float(moments[0]),
        "waterplane_moment_yy": float(moments[1]),
        "center_of_buoyancy_z": float(center_of_buoyancy[2]),
        "c33": float(restoring[2, 2]),
        "c44": float(restoring[3, 3]),
        "c55": float(restoring[4, 4]),
    }
    errors = {key: _rel(measured[key], ref[key]) for key in measured}
    max_offdiag = float(
        np.max(np.abs(restoring - np.diag(np.diag(restoring))))
    )
    return {
        "geometry": {"length": length, "beam": beam, "draft": draft, "kg": kg},
        "panel_count": int(hull.panel_count),
        "waterplane_centroid": centroid.tolist(),
        "measured": measured,
        "reference": {
            key: ref[key]
            for key in (
                "waterplane_area",
                "displaced_volume",
                "waterplane_moment_xx",
                "waterplane_moment_yy",
                "center_of_buoyancy_z",
                "c33",
                "c44",
                "c55",
                "gm_transverse",
                "gm_longitudinal",
            )
        },
        "relative_error": errors,
        "max_relative_error": max(errors.values()),
        "max_offdiagonal_restoring": max_offdiag,
    }


def run_mass_matrix_check(mass: float = 8200.0, kg: float = 0.5) -> dict[str, Any]:
    """Compare the rigid-body mass matrix with the exact block form."""
    cg = np.array([0.0, 0.0, -kg])
    inertia = np.diag([1.0e4, 8.0e3, 1.2e4]) + np.array(
        [[0.0, 0.0, 2.0e2], [0.0, 0.0, 0.0], [2.0e2, 0.0, 0.0]]
    )  # symmetric, about the origin
    s = np.array([[0.0, -cg[2], cg[1]], [cg[2], 0.0, -cg[0]], [-cg[1], cg[0], 0.0]])
    zeros = np.zeros((3, 3))
    # Closed block form (Fossen 2011, eq. 3.44) with the inertia tensor already
    # referred to the origin, so the lower-right block *is* ``inertia``: the
    # parallel-axis transfer is already inside it.  Mission m17324: the previous
    # "exact" block here was the transposed/negated one
    # ([-m S^T]; [-m S, I + m S S^T]), i.e. it encoded the assembly defect
    # instead of the physics, which is why the symmetry-only tests stayed green.
    exact = np.block(
        [
            [mass * np.eye(3), -mass * s],
            [mass * s, inertia],
        ]
    )
    measured = compute_mass_matrix(mass, cg, inertia)
    inertia_cg = inertia - mass * (float(cg @ cg) * np.eye(3) - np.outer(cg, cg))
    return {
        "mass": mass,
        "center_of_gravity": cg.tolist(),
        "inertia_tensor_about_origin": inertia.tolist(),
        "inertia_tensor_about_cg": inertia_cg.tolist(),
        "max_abs_difference": float(np.max(np.abs(measured - exact))),
        "scale": float(np.max(np.abs(exact))),
        "max_relative_error": float(np.max(np.abs(measured - exact)))
        / float(np.max(np.abs(exact))),
        "symmetry_error": float(np.max(np.abs(measured - measured.T))),
        "energy_identity_error": energy_residual(measured, mass, inertia_cg, cg),
        "diagonal": np.diag(measured).tolist(),
    }


def run_hulme_crosscheck(
    subdivisions: int = 2,
    ka_values: Sequence[float] = (0.2, 0.5, 1.0),
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    cg_depth_ratio: float = 0.5,
    chief_points: int | str | None = "auto",
    with_coherence: bool = True,
) -> dict[str, Any]:
    """Full RAO chain versus Hulme (1982) coefficients and exciting forces.

    The floating body is set to mass ``rho * nabla_mesh`` (hydrostatic
    equilibrium consistent with the discretised hull) and given an explicit
    inertia tensor, so no heuristic mass property enters the comparison.
    """
    hull = hemisphere_hull_mesh(subdivisions, radius=radius)
    area, _, _ = compute_waterplane_properties(hull)
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    mass = density * volume
    cg = np.array([0.0, 0.0, -cg_depth_ratio * radius])
    # Explicit sphere-like inertia about the origin: a documented test device,
    # not a claim about a real hemisphere.
    radius_of_gyration_sq = 0.4 * radius**2
    inertia = mass * radius_of_gyration_sq * np.eye(3)
    props = compute_hydrostatic_restoring_matrix(hull, mass, cg, density, gravity)

    rows = []
    for ka in ka_values:
        nu = float(ka) / radius
        omega = math.sqrt(gravity * nu)
        ref = hulme_hemisphere_radiation(
            float(ka), radius=radius, density=density, gravity=gravity
        )
        rao_result = compute_rao_at_frequency(
            hull,
            omega,
            wave_angle=0.0,
            mass=mass,
            center_of_gravity=cg,
            hydrostatic_props=props,
            density=density,
            gravity=gravity,
            inertia=inertia,
            chief_points=chief_points,
        )
        # Chain integrity: the coefficients inside the RAO must be those of the
        # stand-alone free-surface solves.
        if with_coherence:
            rad = solve_radiation_3d(
                hull, nu, density=density, gravity=gravity, chief_points=chief_points
            )
            diff = solve_diffraction_3d(
                hull,
                nu,
                amplitude=1.0,
                wave_angle=0.0,
                density=density,
                gravity=gravity,
                chief_points=chief_points,
            )
            coherence = {
                "added_mass": float(
                    np.max(np.abs(rao_result.added_mass - rad.added_mass))
                ),
                "damping": float(np.max(np.abs(rao_result.damping - rad.damping))),
                "excitation": float(
                    np.max(np.abs(rao_result.excitation_force - diff.excitation_force))
                ),
            }
            condition_number = float(rad.condition_number)
            radiation_residual = float(rad.relative_residual)
        else:
            coherence = {"added_mass": 0.0, "damping": 0.0, "excitation": 0.0}
            condition_number = float("nan")
            radiation_residual = float("nan")
        rows.append(
            {
                "ka": float(ka),
                "omega": omega,
                "wavenumber": nu,
                "heave_added_mass": float(rao_result.added_mass[2, 2]),
                "heave_added_mass_reference": float(ref.heave_added_mass),
                "heave_added_mass_error": _rel(
                    float(rao_result.added_mass[2, 2]), float(ref.heave_added_mass)
                ),
                "heave_damping": float(rao_result.damping[2, 2]),
                "heave_damping_reference": float(ref.heave_damping),
                "heave_damping_error": _rel(
                    float(rao_result.damping[2, 2]), float(ref.heave_damping)
                ),
                "surge_added_mass": float(rao_result.added_mass[0, 0]),
                "surge_added_mass_reference": float(ref.surge_added_mass),
                "surge_added_mass_error": _rel(
                    float(rao_result.added_mass[0, 0]), float(ref.surge_added_mass)
                ),
                "surge_damping": float(rao_result.damping[0, 0]),
                "surge_damping_reference": float(ref.surge_damping),
                "surge_damping_error": _rel(
                    float(rao_result.damping[0, 0]), float(ref.surge_damping)
                ),
                "excitation_heave": abs(complex(rao_result.excitation_force[2])),
                "excitation_heave_reference": abs(complex(ref.excitation_force_z)),
                "excitation_heave_error": _rel(
                    abs(complex(rao_result.excitation_force[2])),
                    abs(complex(ref.excitation_force_z)),
                ),
                "excitation_surge": abs(complex(rao_result.excitation_force[0])),
                "excitation_surge_reference": abs(complex(ref.excitation_force_x)),
                "excitation_surge_error": _rel(
                    abs(complex(rao_result.excitation_force[0])),
                    abs(complex(ref.excitation_force_x)),
                ),
                "excitation_heave_phase_error_rad": float(
                    np.angle(
                        complex(rao_result.excitation_force[2])
                        / complex(ref.excitation_force_z)
                    )
                ),
                "heave_rao_amplitude": float(rao_result.rao_amplitude[2]),
                "surge_rao_amplitude": float(rao_result.rao_amplitude[0]),
                "impedance_residual": float(rao_result.impedance_residual),
                "chain_coherence_max_abs": max(coherence.values()),
                "chain_condition_number": condition_number,
                "radiation_relative_residual": radiation_residual,
            }
        )

    return {
        "subdivisions": int(subdivisions),
        "panel_count": int(hull.panel_count),
        "radius": radius,
        "mass": mass,
        "displaced_volume_mesh": volume,
        "waterplane_area_mesh": area,
        "waterplane_area_exact": math.pi * radius**2,
        "waterplane_area_error": _rel(area, math.pi * radius**2),
        "displaced_volume_exact": 2.0 * math.pi * radius**3 / 3.0,
        "displaced_volume_error": _rel(volume, 2.0 * math.pi * radius**3 / 3.0),
        "c33": float(props.restoring_matrix[2, 2]),
        "frequencies": rows,
        "max_heave_added_mass_error": max(r["heave_added_mass_error"] for r in rows),
        "max_heave_damping_error": max(r["heave_damping_error"] for r in rows),
        "max_heave_excitation_error": max(r["excitation_heave_error"] for r in rows),
        "max_surge_added_mass_error": max(r["surge_added_mass_error"] for r in rows),
        "max_surge_damping_error": max(r["surge_damping_error"] for r in rows),
        "max_surge_excitation_error": max(r["excitation_surge_error"] for r in rows),
        "max_impedance_residual": max(r["impedance_residual"] for r in rows),
    }


def run_long_wave_limit(
    ka: float = 0.2,
    subdivisions: int = 2,
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Heave RAO in the long-wave limit must approach unity."""
    hull = hemisphere_hull_mesh(subdivisions, radius=radius)
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    area, _, _ = compute_waterplane_properties(hull)
    mass = density * volume
    cg = np.array([0.0, 0.0, -0.5 * radius])
    inertia = mass * 0.4 * radius**2 * np.eye(3)
    props = compute_hydrostatic_restoring_matrix(hull, mass, cg, density, gravity)
    nu = float(ka) / radius
    omega = math.sqrt(gravity * nu)
    result = compute_rao_at_frequency(
        hull,
        omega,
        wave_angle=0.0,
        mass=mass,
        center_of_gravity=cg,
        hydrostatic_props=props,
        density=density,
        gravity=gravity,
        inertia=inertia,
        chief_points="auto",
    )
    heave = float(result.rao_amplitude[2])
    return {
        "ka": float(ka),
        "subdivisions": int(subdivisions),
        "panel_count": int(hull.panel_count),
        "heave_rao_amplitude": heave,
        "reference": 1.0,
        "abs_deviation": abs(heave - 1.0),
        "relative_deviation": abs(heave - 1.0),
        "heave_excitation_over_rho_g_awp": abs(
            abs(complex(result.excitation_force[2]))
        )
        / (density * gravity * area),
        "impedance_residual": float(result.impedance_residual),
    }


def run_mesh_convergence(
    ka: float = 0.5,
    subdivisions: Sequence[int] = (2, 3),
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Show that the residual errors against Hulme are discretisation, not model."""
    studies = {
        int(sub): run_hulme_crosscheck(
            subdivisions=int(sub),
            ka_values=(float(ka),),
            radius=radius,
            density=density,
            gravity=gravity,
            with_coherence=False,
        )
        for sub in subdivisions
    }
    sub_list = sorted(studies)
    coarse, fine = studies[sub_list[0]], studies[sub_list[-1]]
    coarse_row, fine_row = coarse["frequencies"][0], fine["frequencies"][0]
    quantities = (
        "surge_added_mass_error",
        "surge_damping_error",
        "heave_added_mass_error",
        "heave_damping_error",
        "excitation_surge_error",
        "excitation_heave_error",
    )
    ratios = {
        q: (coarse_row[q] / fine_row[q]) if fine_row[q] > 0.0 else math.inf
        for q in quantities
    }
    return {
        "ka": float(ka),
        "coarse_subdivisions": sub_list[0],
        "fine_subdivisions": sub_list[-1],
        "coarse_panels": coarse["panel_count"],
        "fine_panels": fine["panel_count"],
        "errors": {
            q: {
                "coarse": coarse_row[q],
                "fine": fine_row[q],
                "reduction_factor": ratios[q],
            }
            for q in quantities
        },
        "fine_max_error": max(fine_row[q] for q in quantities),
    }


def run(
    subdivisions: Sequence[int] = (2, 3),
    ka_values: Sequence[float] = (0.2, 0.5, 1.0),
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    *,
    quick: bool = False,
    include_convergence: bool = True,
) -> dict[str, Any]:
    """Execute the closed-form verification of the hydrostatic/RAO layer."""
    subs = tuple(int(s) for s in subdivisions)
    coarse = min(subs) if subs else 2
    report: dict[str, Any] = {
        "benchmark": "hydrostatic_restoring_rao_closed_form_verification",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "references": {
            "box_hydrostatics": "Biran & Pulido, Ship Hydrostatics and Stability; "
            "Rawson & Tupper, Basic Ship Theory (wall-sided pontoon closed forms)",
            "hemisphere": "Hulme (1982) J. Fluid Mech. 121:443-463, tables 1-2",
            "long_wave_limit": "freely floating body heave RAO -> 1 as ka -> 0",
        },
        "box_hydrostatics": run_box_hydrostatics(density=density, gravity=gravity),
        "mass_matrix": run_mass_matrix_check(),
        "hulme_crosscheck": run_hulme_crosscheck(
            subdivisions=coarse,
            ka_values=ka_values if not quick else tuple(ka_values)[:2],
            radius=radius,
            density=density,
            gravity=gravity,
        ),
        "long_wave_limit": run_long_wave_limit(
            subdivisions=coarse, radius=radius, density=density, gravity=gravity
        ),
    }
    if include_convergence and not quick:
        report["mesh_convergence"] = run_mesh_convergence(
            subdivisions=subs, radius=radius, density=density, gravity=gravity
        )
    return report


def plot(report: dict[str, Any], path: Path) -> Path:
    """Two-panel verification figure: RAO physics and reference-error decay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cross = report["hulme_crosscheck"]
    rows = cross["frequencies"]
    ka = np.array([r["ka"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.plot(ka, [r["heave_rao_amplitude"] for r in rows], "o-", label="heave RAO, BEM chain")
    ax.axhline(1.0, color="0.35", ls="--", lw=1.2, label="long-wave limit $|\\xi_3|\\to 1$")
    ax.plot(
        ka,
        [r["excitation_heave"] / cross["c33"] for r in rows],
        "s--",
        color="tab:orange",
        label="$|F_3|/(\\rho g A_{wp})$ from BEM",
    )
    ax.set_xlabel("$ka$")
    ax.set_ylabel("amplitude / ratio")
    ax.set_title("heave response of the floating hemisphere")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    if "mesh_convergence" in report:
        conv = report["mesh_convergence"]["errors"]
        labels = {
            "surge_added_mass_error": "$A_{11}$",
            "surge_damping_error": "$B_{11}$",
            "heave_added_mass_error": "$A_{33}$",
            "heave_damping_error": "$B_{33}$",
            "excitation_surge_error": "$|F_1|$",
            "excitation_heave_error": "$|F_3|$",
        }
        xs = np.arange(len(labels))
        coarse = [conv[k]["coarse"] for k in labels]
        fine = [conv[k]["fine"] for k in labels]
        ax.bar(xs - 0.2, np.array(coarse) * 100.0, width=0.4, label="64 panels")
        ax.bar(xs + 0.2, np.array(fine) * 100.0, width=0.4, label="256 panels")
        ax.set_xticks(xs)
        ax.set_xticklabels(list(labels.values()))
        ax.set_yscale("log")
        ax.axhline(3.0, color="crimson", ls="--", lw=1.2, label="3 % target")
        ax.set_ylabel("relative error vs Hulme (1982) [%]")
        ax.set_title("$ka=%.1f$: error is discretisation, not model" % report["mesh_convergence"]["ka"])
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--ka", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=_DEFAULT_DENSITY)
    parser.add_argument("--gravity", type=float, default=_DEFAULT_GRAVITY)
    parser.add_argument("--quick", action="store_true", help="fast CI smoke run")
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    parser.add_argument("--figure", type=Path, help="also write the comparison figure")
    args = parser.parse_args()

    report = run(
        args.subdivisions,
        args.ka,
        args.radius,
        args.density,
        args.gravity,
        quick=args.quick,
        include_convergence=not args.skip_convergence,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.figure is not None:
        plot(report, args.figure)
    print(payload)


if __name__ == "__main__":
    main()