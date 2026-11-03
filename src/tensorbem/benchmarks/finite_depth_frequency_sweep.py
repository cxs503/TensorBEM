"""Frequency and submergence sweep for the finite-depth radiation solver.

The finite-depth radiation module (:mod:`tensorbem.finite_depth_solver`) was
first validated at a single operating point, ``ka = 0.6`` and ``f/a = 2`` in a
deep basin (``docs/finite_depth_radiation.md`` section 5, limitation 5).  A
single point cannot distinguish "the kernel is right" from "the kernel is right
at that ``nu*h``": the evanescent branch of the finite-depth kernel decays with
``nu*h`` and with the body clearance, so a mode count that is adequate at
``ka = 0.6`` may be under-resolved at short waves, and the free-surface part of
the kernel becomes stiffer when the section sits closer to the surface.

This benchmark sweeps ``ka`` and ``f/a`` at ``h/a = 50`` (where the finite-depth
coefficients must reproduce the deep-water ones) and reports five checks per
case:

* **deep-water collapse** - the finite-depth coefficients are compared both with
  Ursell's closed-form deep-water multipole solution
  (:func:`tensorbem.analytic.ursell_cylinder_added_mass`, no shared code) and
  with the deep-water *panel* solver (:mod:`tensorbem.free_surface_solver`) on
  the identical mesh.  The second comparison isolates the finite-depth kernel
  from the panel discretisation, which is what a ``ka`` sweep must do: the
  discretisation error grows with ``ka`` by itself;
* **energy identity** - ``B = 2*rho*g*c_g*|eta_a|^2`` from the radiated wave
  amplitude must equal the damping read off the surface pressure;
* **mode-sum adequacy** - the same case is re-run with ``mode_factor`` times the
  automatic evanescent mode count;
* **panel refinement** - the coarsest and finest case of the sweep are refined
  from ``N`` to ``2N`` panels to expose the observed order;
* **heave/sway split** - in deep water the two coefficients must coincide; the
  residual split measures how much of the finite depth is still felt at
  ``h/a = 50``.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_frequency_sweep \
        --json artifacts/finite_depth_frequency_sweep/sweep.json \
        --figures artifacts/finite_depth_frequency_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks.finite_depth_radiation import submerged_circle_mesh
from tensorbem.finite_depth import _resolve_modes
from tensorbem.finite_depth_solver import solve_radiation
from tensorbem.free_surface_solver import solve_radiation_indirect as solve_deep

GRAVITY = 9.80665
DENSITY = 1025.0
RADIUS = 1.0
MOTIONS = {"heave": 1, "sway": 0}

KA_VALUES = (0.2, 0.6, 1.0, 1.5)
RATIOS = (1.5, 2.0, 3.0)
DEPTH_RATIO = 50.0
PANELS = 64


def _analytic(ka: float, ratio: float) -> tuple[float, float]:
    """Ursell deep-water ``(A, B)`` for the given ``ka`` and ``f/a``."""
    submergence = ratio * RADIUS
    omega = math.sqrt(ka * GRAVITY / RADIUS)
    return ursell_cylinder_added_mass(
        RADIUS, submergence, omega, DENSITY, gravity=GRAVITY
    )


def _finite_depth(
    ka: float, ratio: float, depth: float, panels: int, modes: int | None
) -> dict:
    """Solve heave and sway separately and package both coefficients.

    The indirect formulation returns a two-component force, and for a body that
    is symmetric about ``x = 0`` the off-direction component is *exactly* zero
    (a heave motion produces no horizontal force).  Reading both components out
    of a single solve would therefore compare zero against zero, so each motion
    is solved with its own body condition.
    """
    nu = ka / RADIUS
    mesh = submerged_circle_mesh(panels, RADIUS, ratio * RADIUS)
    payload = {
        "added_mass": {},
        "damping": {},
        "far_field_damping": {},
        "far_over_near": {},
        "condition_number": {},
        "relative_residual": {},
    }
    started = time.time()
    for motion in MOTIONS:
        result, far = solve_radiation(
            depth=depth,
            mesh=mesh,
            nu=nu,
            direction=motion,
            density=DENSITY,
            gravity=GRAVITY,
            modes=modes,
        )
        index = MOTIONS[motion]
        payload["added_mass"][motion] = float(result.added_mass[index])
        payload["damping"][motion] = float(result.damping[index])
        payload["far_field_damping"][motion] = float(far)
        payload["far_over_near"][motion] = (
            float(far / result.damping[index]) if result.damping[index] else float("nan")
        )
        payload["condition_number"][motion] = float(result.condition_number)
        payload["relative_residual"][motion] = float(result.relative_residual)
    payload["elapsed_seconds"] = time.time() - started
    return payload


def _deep_panel(ka: float, ratio: float, panels: int) -> dict:
    """Deep-water panel solution on the identical mesh, one solve per motion."""
    nu = ka / RADIUS
    mesh = submerged_circle_mesh(panels, RADIUS, ratio * RADIUS)
    out: dict[str, dict[str, float]] = {"added_mass": {}, "damping": {}}
    for motion in MOTIONS:
        result = solve_deep(mesh, nu, motion, density=DENSITY)
        index = MOTIONS[motion]
        out["added_mass"][motion] = float(result.added_mass[index])
        out["damping"][motion] = float(result.damping[index])
    return out


def run(
    ka_values: tuple[float, ...] = KA_VALUES,
    ratios: tuple[float, ...] = RATIOS,
    depth_ratio: float = DEPTH_RATIO,
    panels: int = PANELS,
    mode_factor: int = 2,
    refinement_panels: tuple[int, ...] = (32, 64, 128),
) -> dict:
    """Run the ``(ka, f/a)`` sweep and the supporting studies."""
    depth = depth_ratio * RADIUS
    started = time.time()
    cases = []
    for ka in ka_values:
        for ratio in ratios:
            analytic_mass, analytic_damping = _analytic(ka, ratio)
            case = _finite_depth(ka, ratio, depth, panels, None)
            deep = _deep_panel(ka, ratio, panels)
            nu = ka / RADIUS
            entry = {
                "ka": float(ka),
                "submergence_ratio": float(ratio),
                "clearance_ratio": float(depth_ratio - ratio - 1.0),
                "nu": float(nu),
                "nu_h": float(nu * depth),
                "modes_auto": int(_resolve_modes(nu, depth, None)),
                "panels": int(panels),
                "finite_depth": case,
                "deep_panel": deep,
                "analytic": {"added_mass": float(analytic_mass),
                             "damping": float(analytic_damping)},
            }
            for motion in MOTIONS:
                entry[f"dev_from_analytic_{motion}"] = {
                    "added_mass": case["added_mass"][motion] / analytic_mass - 1.0,
                    "damping": case["damping"][motion] / analytic_damping - 1.0,
                }
                entry[f"dev_from_deep_panel_{motion}"] = {
                    "added_mass": case["added_mass"][motion]
                    / deep["added_mass"][motion] - 1.0,
                    "damping": case["damping"][motion]
                    / deep["damping"][motion] - 1.0,
                }
            entry["heave_sway_split"] = (
                case["added_mass"]["heave"] / case["added_mass"]["sway"] - 1.0
            )
            cases.append(entry)

    # Mode-sum adequacy: the case with the largest nu*h (short wave, deep basin)
    # and the case with the smallest clearance (stiffest free-surface kernel).
    mode_cases = []
    for ka, ratio in ((max(ka_values), min(ratios)),):
        base = _finite_depth(ka, ratio, depth, panels, None)
        rich_modes = mode_factor * int(_resolve_modes(ka / RADIUS, depth, None))
        rich = _finite_depth(ka, ratio, depth, panels, rich_modes)
        mode_cases.append(
            {
                "ka": float(ka),
                "submergence_ratio": float(ratio),
                "modes_auto": int(_resolve_modes(ka / RADIUS, depth, None)),
                "modes_requested": int(rich_modes),
                "rel_diff_added_mass": {
                    motion: rich["added_mass"][motion] / base["added_mass"][motion] - 1.0
                    for motion in MOTIONS
                },
                "rel_diff_damping": {
                    motion: rich["damping"][motion] / base["damping"][motion] - 1.0
                    for motion in MOTIONS
                },
            }
        )

    # Panel refinement on the coarsest and the finest point of the sweep.
    refinement = []
    for ka, ratio in ((min(ka_values), ratios[1] if len(ratios) > 1 else ratios[0]),
                      (max(ka_values), min(ratios))):
        analytic_mass, analytic_damping = _analytic(ka, ratio)
        rows = []
        for count in refinement_panels:
            case = _finite_depth(ka, ratio, depth, count, None)
            rows.append(
                {
                    "panels": int(count),
                    "added_mass_heave": case["added_mass"]["heave"],
                    "damping_heave": case["damping"]["heave"],
                    "abs_err_added_mass": abs(case["added_mass"]["heave"] / analytic_mass - 1.0),
                    "abs_err_damping": abs(case["damping"]["heave"] / analytic_damping - 1.0),
                }
            )
        orders = []
        for column in ("abs_err_added_mass", "abs_err_damping"):
            for coarse, fine in zip(rows, rows[1:]):
                if fine[column] > 0.0 and coarse[column] > 0.0:
                    orders.append(
                        {
                            "quantity": column,
                            "panels": [coarse["panels"], fine["panels"]],
                            "observed_order": math.log(coarse[column] / fine[column])
                            / math.log(fine["panels"] / coarse["panels"]),
                        }
                    )
        refinement.append(
            {
                "ka": float(ka),
                "submergence_ratio": float(ratio),
                "rows": rows,
                "observed_orders": orders,
            }
        )

    return {
        "benchmark": "finite_depth_frequency_sweep",
        "gravity": GRAVITY,
        "density": DENSITY,
        "radius": RADIUS,
        "depth": depth,
        "depth_ratio": float(depth_ratio),
        "panels": int(panels),
        "mode_factor": int(mode_factor),
        "ka_values": [float(value) for value in ka_values],
        "submergence_ratios": [float(value) for value in ratios],
        "cases": cases,
        "mode_study": mode_cases,
        "refinement": refinement,
        "elapsed_seconds": time.time() - started,
    }


def summary(payload: dict) -> dict:
    """Aggregate the checks into the numbers the report quotes."""
    cases = payload["cases"]
    kernel_errors = [
        max(
            abs(case["dev_from_deep_panel_heave"]["added_mass"]),
            abs(case["dev_from_deep_panel_sway"]["added_mass"]),
        )
        for case in cases
    ]
    panel_errors = [
        max(
            abs(case["dev_from_analytic_heave"]["added_mass"]),
            abs(case["dev_from_analytic_sway"]["added_mass"]),
        )
        for case in cases
    ]
    energy = [
        abs(case["finite_depth"]["far_over_near"][motion] - 1.0)
        for case in cases
        for motion in MOTIONS
    ]
    splits = [abs(case["heave_sway_split"]) for case in cases]
    return {
        "max_kernel_error_vs_deep_panel": max(kernel_errors),
        "max_panel_error_vs_analytic": max(panel_errors),
        "max_energy_identity_error": max(energy),
        "max_heave_sway_split": max(splits),
        "worst_panel_error_case": max(
            cases,
            key=lambda case: max(
                abs(case["dev_from_analytic_heave"]["added_mass"]),
                abs(case["dev_from_analytic_sway"]["added_mass"]),
            ),
        )["ka"],
        "max_mode_rel_diff": max(
            max(abs(value) for value in case["rel_diff_added_mass"].values())
            for case in payload["mode_study"]
        ),
    }


def plot(payload: dict, directory: Path) -> list[Path]:
    """Error-vs-``ka`` panels and the energy identity across the sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cases = payload["cases"]
    ratios = payload["submergence_ratios"]
    colours = {1.5: "#d62728", 2.0: "#1f77b4", 3.0: "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for motion, linestyle, name in (("heave", "-", "heave"), ("sway", "--", "sway")):
        for ratio in ratios:
            subset = sorted(
                (case for case in cases if case["submergence_ratio"] == ratio),
                key=lambda case: case["ka"],
            )
            axes[0].semilogy(
                [case["ka"] for case in subset],
                [abs(case[f"dev_from_analytic_{motion}"]["added_mass"]) for case in subset],
                linestyle, marker="o", color=colours[ratio],
                label=rf"{name}, $f/a={ratio}$ vs Ursell",
            )
            axes[1].semilogy(
                [case["ka"] for case in subset],
                [abs(case[f"dev_from_deep_panel_{motion}"]["added_mass"]) for case in subset],
                linestyle, marker="s", color=colours[ratio],
                label=rf"{name}, $f/a={ratio}$ vs deep panel",
            )
    for axis, title in (
        (axes[0], "Added mass vs closed-form deep-water Ursell"),
        (axes[1], "Added mass vs deep-water panel solver (same mesh)"),
    ):
        axis.set_xlabel(r"$ka$")
        axis.set_ylabel(r"$|A/A_{\rm ref} - 1|$")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    path = directory / "ka_sweep_errors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    fig, axis = plt.subplots(figsize=(6.6, 4.0))
    for ratio in ratios:
        subset = sorted(
            (case for case in cases if case["submergence_ratio"] == ratio),
            key=lambda case: case["ka"],
        )
        for motion, marker in (("heave", "o"), ("sway", "s")):
            axis.semilogx(
                [case["ka"] for case in subset],
                [case["finite_depth"]["far_over_near"][motion] for case in subset],
                marker + "-", color=colours[ratio], alpha=0.85,
                label=rf"{motion}, $f/a={ratio}$",
            )
    axis.axhline(1.0, color="k", ls="--", lw=1.1)
    axis.set_xlabel(r"$ka$")
    axis.set_ylabel(r"$B_{\rm far}/B_{\rm near}$")
    axis.set_title(r"Energy identity across the sweep ($h/a=50$, 64 panels)", fontsize=10)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "energy_identity_ka.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--panels", type=int, default=PANELS)
    parser.add_argument("--depth-ratio", type=float, default=DEPTH_RATIO)
    args = parser.parse_args(argv)
    payload = run(panels=args.panels, depth_ratio=args.depth_ratio)
    payload["summary"] = summary(payload)
    if args.json is not None:
        _write_json(payload, args.json)
    if args.figures is not None:
        for path in plot(payload, args.figures):
            print(f"wrote {path}")
    for case in payload["cases"]:
        print(
            f"ka={case['ka']:4.2f} f/a={case['submergence_ratio']:4.2f} "
            f"A={case['finite_depth']['added_mass']['heave']:9.2f} "
            f"dev_ursell={case['dev_from_analytic_heave']['added_mass']:+.2e} "
            f"dev_deeppanel={case['dev_from_deep_panel_heave']['added_mass']:+.2e} "
            f"Bfar/Bnear={case['finite_depth']['far_over_near']['heave']:7.4f} "
            f"modes={case['modes_auto']}"
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())