"""Mutual added mass of two circular cylinders: BEM versus the multipole series.

The reference for this case is :func:`tensorbem.analytic.two_cylinder_added_mass`
-- a truncated multipole series for two equal circular cylinders in an
unbounded fluid, independent of every layer matrix here. It is the first
reference in this repository that carries *mutual* (body-to-body)
hydrodynamic coefficients, so it exercises two things the single-body
benchmarks cannot:

* the 2-D exterior Neumann operator assembled over **two disjoint loops**
  (:class:`tensorbem.multiloop.MultiLoopMesh`, both loops marked as holes so
  the domain-outward normals point into the bodies);
* the interaction terms ``A[0,2]`` (in line with the centres, negative) and
  ``A[1,3]`` (transverse, positive), which are what a two-body motion actually
  feels, and which no single-body check can see.

Formulation (constant panels, midpoint collocation): with ``phi`` the
radiation potential of a unit rigid-body velocity ``U`` on one body and zero
on the other, the direct exterior second-kind equation reads

    0.5*phi - (H phi) = (S q_e),      q_e = dphi/dn_e,  n_e = fluid-outward,

and the added mass follows from ``A_ij = -rho * closed_integral(phi_j n_i ds)``
with the body-outward normal ``n_i``. ``S`` and ``H`` are exactly the
single- and double-layer matrices of
:func:`tensorbem.solver.assemble_matrices`, so this case adds no new kernel --
only a mesh whose two loops are *both* marked as holes. Marking them flips the
domain-outward normals to point into the bodies, which flips the sign of the
double-layer matrix: ``H`` then satisfies ``H (1 on one body) = +1/2`` there, so
the assembled system must be ``0.5 I + H``. Using ``0.5 I - H`` instead is the
sign that belongs to normals pointing into the fluid; it leaves the self
coefficient nearly unchanged (0.3-3 %) but annihilates the interaction terms.
That mistake is the documented injected-error control (``--inject layer_sign``).

The condition number of the system is ``O(1)`` (2.2-2.5 in every case
reported), so this benchmark can afford 48-96 panels per body.

Reproducibility: no random data, no time steps, no iteration counts to tune.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.two_cylinder_added_mass \
        --panels 48 64 96 --separation-ratio 3.0 --json artifacts/two_cylinder/add.json

The ``--inject`` switch deliberately assembles the *wrong* double-layer sign;
it exists so a test can prove that the assertions in
``tests/test_two_cylinder_added_mass.py`` actually fire. Note that the
self coefficient survives that injection almost unchanged (it stays within
0.3 %), while the mutual coefficients collapse towards zero: the interaction
term, not the self term, is what detects a broken coupling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..analytic import two_cylinder_added_mass
from ..geometry import PanelMesh
from ..multiloop import MultiLoopMesh
from ..solver import assemble_matrices

MODES = ("body0_surge", "body0_heave", "body1_surge", "body1_heave")
DENSITY = 1025.0


def circle_body(radius: float, panels: int, center: complex) -> PanelMesh:
    """A regular polygon on a circle, counter-clockwise about ``center``."""
    if panels < 8:
        raise ValueError("panels must be at least 8")
    angles = 2.0 * np.pi * np.arange(panels) / panels
    offset = np.array([float(np.real(center)), float(np.imag(center))])
    return PanelMesh(
        radius * np.column_stack((np.cos(angles), np.sin(angles))) + offset[None, :]
    )


def two_circle_mesh(radius: float, separation: float, panels: int) -> MultiLoopMesh:
    """Two disjoint circular bodies as one multiply connected exterior boundary."""
    if separation <= 2.0 * radius:
        raise ValueError("separation must exceed 2*radius")
    half = 0.5 * separation
    loops = (circle_body(radius, panels, complex(-half, 0.0)),
             circle_body(radius, panels, complex(half, 0.0)))
    # Both loops bound bodies, so the fluid lies outside each: mark both as
    # holes and their domain-outward normals point into the bodies.
    return MultiLoopMesh(loops=loops, hole_loops=(0, 1))


def solve_added_mass_potentials(
    mesh: MultiLoopMesh, density: float = DENSITY, layer_sign: float = 1.0
) -> tuple[np.ndarray, float, np.ndarray]:
    """Added-mass matrix, condition number and the four boundary potentials.

    Columns of the returned potential array follow :data:`MODES` (panel order
    matches ``mesh``), so a caller can compare boundary fields, not only the
    integrated coefficients.
    """
    if layer_sign not in (1.0, -1.0):
        raise ValueError("layer_sign must be +1 or -1")
    single, double = assemble_matrices(mesh)
    panels = mesh.panel_count
    system = 0.5 * np.eye(panels) + layer_sign * double
    normals = mesh.normals  # fluid-outward: into each body
    rhs = []
    for _, loop_slice in enumerate(mesh.loop_slices):
        for axis in range(2):
            flux = np.zeros(panels)
            flux[loop_slice] = normals[loop_slice, axis]
            rhs.append(single @ flux)
    potentials = np.linalg.solve(system, np.column_stack(rhs))
    matrix = np.zeros((4, 4))
    for body, loop_slice in enumerate(mesh.loop_slices):
        for axis in range(2):
            modal_normal = mesh.normals[loop_slice, axis] * mesh.lengths[loop_slice]
            matrix[2 * body + axis, :] = density * (
                potentials[loop_slice, :] * modal_normal[:, None]
            ).sum(axis=0)
    return matrix, float(np.linalg.cond(system)), potentials


def solve_added_mass(
    mesh: MultiLoopMesh, density: float = DENSITY, layer_sign: float = 1.0
) -> tuple[np.ndarray, float]:
    """BEM added-mass matrix per unit span for the ``4`` rigid-body modes.

    ``layer_sign`` is a negative-control switch: ``-1`` assembles
    ``(1/2 I - H)``, which is the exterior operator for normals pointing *into*
    the fluid and is therefore wrong for hole loops. It is never used outside
    the documented injected-error control.
    """
    matrix, condition, _ = solve_added_mass_potentials(mesh, density, layer_sign)
    return matrix, condition


def _summarise(matrix: np.ndarray) -> dict[str, float]:
    return {
        "self": float(matrix[0, 0]),
        "mutual_along": float(matrix[0, 2]),
        "mutual_transverse": float(matrix[1, 3]),
        "in_phase_surge": float(matrix[0, 0] + matrix[0, 2]),
        "in_phase_heave": float(matrix[1, 1] + matrix[1, 3]),
    }


def _relative_error(computed: float, reference: float) -> float:
    return abs(computed - reference) / abs(reference)


def run(
    panels: int = 64,
    separation_ratio: float = 3.0,
    radius: float = 1.0,
    density: float = DENSITY,
    *,
    terms: int = 24,
    layer_sign: float = 1.0,
) -> dict[str, Any]:
    """One mesh: BEM coefficients against the multipole reference."""
    for name, value in (
        ("radius", radius),
        ("separation_ratio", separation_ratio),
        ("density", density),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if separation_ratio <= 2.0:
        raise ValueError("separation_ratio must exceed 2 (non-touching bodies)")
    separation = separation_ratio * radius
    reference = two_cylinder_added_mass(
        radius, separation, density, terms=terms
    )
    mesh = two_circle_mesh(radius, separation, panels)
    matrix, condition = solve_added_mass(mesh, density, layer_sign)
    bem = _summarise(matrix)
    exact = _summarise(reference.added_mass_matrix)
    errors = {
        key: _relative_error(bem[key], exact[key]) for key in exact
    }
    checks = {
        "self_within_1_percent": {"value": errors["self"], "limit": 1.0e-2,
                                  "passed": errors["self"] < 1.0e-2},
        "mutual_along_within_2_percent": {
            "value": errors["mutual_along"], "limit": 2.0e-2,
            "passed": errors["mutual_along"] < 2.0e-2,
        },
        "mutual_transverse_within_2_percent": {
            "value": errors["mutual_transverse"], "limit": 2.0e-2,
            "passed": errors["mutual_transverse"] < 2.0e-2,
        },
        "in_phase_heave_within_1_percent": {
            "value": errors["in_phase_heave"], "limit": 1.0e-2,
            "passed": errors["in_phase_heave"] < 1.0e-2,
        },
        "interaction_is_resolved": {
            # A decoupled (single-body) treatment returns mutual = 0; require the
            # computed mutual term to be a real, non-degenerate fraction of the
            # self term so the check above cannot pass vacuously.
            "value": abs(bem["mutual_transverse"]) / abs(bem["self"]),
            "limit": 0.05,
            "passed": abs(bem["mutual_transverse"]) / abs(bem["self"]) > 0.05,
        },
    }
    return {
        "benchmark": "two_cylinder_mutual_added_mass",
        "panels_per_body": int(panels),
        "radius": float(radius),
        "separation": float(separation),
        "separation_ratio": float(separation_ratio),
        "fluid_density": float(density),
        "multipole_terms": int(terms),
        "single_layer_condition_number": condition,
        "reference_boundary_residual": reference.boundary_residual,
        "reference_reflection_symmetry_error": reference.reflection_symmetry_error,
        "displaced_mass_per_span": reference.displaced_mass_per_span,
        "bem_added_mass_matrix": matrix.tolist(),
        "reference_added_mass_matrix": reference.added_mass_matrix.tolist(),
        "modes": list(MODES),
        "bem": bem,
        "reference": {
            key: (float(exact[key])) for key in exact
        },
        "relative_errors": errors,
        "max_relative_error": max(errors.values()),
        "checks": checks,
        "all_checks_passed": all(item["passed"] for item in checks.values()),
        "layer_sign": float(layer_sign),
    }


def interaction_decay(
    separation_ratios: Sequence[float] = (3.0, 5.0, 10.0, 20.0, 40.0),
    radius: float = 1.0,
    density: float = DENSITY,
    *,
    terms: int = 24,
) -> list[dict[str, float]]:
    """Reference mutual coefficient versus separation (BEM-free asymptotic check).

    For well-separated bodies the interaction decays like ``(a/d)**2``:
    ``mutual * separation**2`` must approach a constant. This is an
    independent consistency statement about the series itself, not a
    comparison with any BEM number.
    """
    rows = []
    for ratio in separation_ratios:
        if ratio <= 2.0:
            raise ValueError("separation ratios must exceed 2")
        reference = two_cylinder_added_mass(
            radius, ratio * radius, density, terms=terms
        )
        separation = ratio * radius
        rows.append(
            {
                "separation_ratio": float(ratio),
                "self_over_displaced": reference.self_added_mass
                / reference.displaced_mass_per_span,
                "mutual_along_over_displaced": reference.mutual_added_mass_along
                / reference.displaced_mass_per_span,
                "mutual_transverse_over_displaced": reference.mutual_added_mass_transverse
                / reference.displaced_mass_per_span,
                "in_phase_heave_over_displaced": reference.in_phase_heave_added_mass
                / reference.displaced_mass_per_span,
                "mutual_transverse_times_separation_squared": reference.mutual_added_mass_transverse
                * separation**2,
            }
        )
    return rows


def refinement_study(
    panels: Sequence[int] = (48, 64, 96),
    separation_ratio: float = 3.0,
    radius: float = 1.0,
    density: float = DENSITY,
    *,
    terms: int = 24,
) -> dict[str, Any]:
    """Mesh-refinement ladder against a single multipole reference."""
    rows = [
        run(panels=n, separation_ratio=separation_ratio, radius=radius,
            density=density, terms=terms)
        for n in panels
    ]
    errors = [row["max_relative_error"] for row in rows]
    decreasing = all(
        later < earlier for earlier, later in zip(errors, errors[1:])
    )
    observed = [
        None
        if earlier <= 0.0 or later <= 0.0
        else float(math.log(earlier / later) / math.log(n_fine / n_coarse))
        for n_coarse, n_fine, earlier, later in zip(panels, panels[1:], errors, errors[1:])
    ]
    return {
        "benchmark": "two_cylinder_mutual_added_mass_refinement",
        "separation_ratio": float(separation_ratio),
        "radius": float(radius),
        "panels_per_body": [int(n) for n in panels],
        "max_relative_errors": errors,
        "monotone_decrease": decreasing,
        "observed_orders": observed,
        "rows": rows,
    }


def plot_refinement(study: dict[str, Any], output_path: str | Path) -> None:
    """Convergence of self and mutual coefficients against the reference."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = study["panels_per_body"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    keys = (
        ("self", "self (blockage)"),
        ("mutual_along", "mutual, in line"),
        ("mutual_transverse", "mutual, transverse"),
        ("in_phase_heave", "in-phase heave"),
    )
    for key, label in keys:
        errors = [row["relative_errors"][key] for row in study["rows"]]
        axes[0].loglog(panels, errors, "o-", label=label)
    axes[0].set_xlabel("panels per body")
    axes[0].set_ylabel("relative error vs multipole series")
    axes[0].set_title(f"two cylinders, d/a = {study['separation_ratio']}")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    first = study["rows"][-1]
    exact = first["reference"]
    bem = first["bem"]
    labels = ["self", "mutual\n(in line)", "mutual\n(transverse)", "in-phase\nheave"]
    source = ["self", "mutual_along", "mutual_transverse", "in_phase_heave"]
    positions = np.arange(len(labels))
    axes[1].bar(
        positions - 0.2,
        [exact[key] / exact["self"] for key in source],
        0.4,
        label="multipole reference",
    )
    axes[1].bar(
        positions + 0.2,
        [bem[key] / bem["self"] for key in source],
        0.4,
        label=f"BEM ({first['panels_per_body']} panels/body)",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("coefficient / self coefficient")
    axes[1].set_title("normalised added-mass coefficients")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, nargs="+", default=[48, 64, 96])
    parser.add_argument("--separation-ratio", type=float, default=3.0)
    parser.add_argument("--terms", type=int, default=24)
    parser.add_argument("--inject", choices=("none", "layer_sign"), default="none")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args(argv)

    layer_sign = -1.0 if args.inject == "layer_sign" else 1.0
    if layer_sign > 0.0:
        payload: dict[str, Any] = dict(
            refinement_study(
                panels=args.panels,
                separation_ratio=args.separation_ratio,
                terms=args.terms,
            )
        )
        payload["interaction_decay"] = interaction_decay(terms=args.terms)
        study = payload
    else:
        # Documented negative control: the same mesh ladder, with the wrong
        # double-layer sign. Nothing is written to the figures directory.
        payload = {
            "benchmark": "two_cylinder_mutual_added_mass_injected_sign",
            "rows": [
                run(
                    panels=n,
                    separation_ratio=args.separation_ratio,
                    terms=args.terms,
                    layer_sign=-1.0,
                )
                for n in args.panels
            ],
        }
        study = None

    text = json.dumps(payload, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
        print(f"wrote {args.json}")
    else:
        print(text)
    if args.figures is not None and study is not None:
        plot_refinement(study, args.figures / "two_cylinder_refinement.png")
        print(f"wrote {args.figures / 'two_cylinder_refinement.png'}")


if __name__ == "__main__":  # pragma: no cover
    main()