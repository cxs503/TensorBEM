"""Panel-convergence ladder for the Dean–Ursell–Ogilvie submerged-cylinder
diffraction benchmark.

``docs/marine_bem_benchmarks.md`` §4.8 (Table 7) verifies the circular-cylinder
zero-reflection / unit-transmission / circular-polarisation / Haskind-reciprocity
chain at a *single* panel count (N=64).  A single point cannot separate
"the solver is correct" from "the solver happens to be close at this one
resolution".  This module turns that single point into a convergence ladder:

    N = 16, 32, 64, 128, 256

and reports, for each of the four physical laws, the absolute error against the
independent Ursell (1950) / Ogilvie (1963) multipole closed form and the
*observed order* of convergence.  The closed form is evaluated by
:mod:`tensorbem.analytic` (zero shared code with the panel solver), so the
reference is independent of the discretisation under audit.

The four laws audited (circular cross-section, deep water, ``e^{-i omega t}``):

1. **Dean zero reflection**  ``R = 0``  (Dean 1948).
2. **Unit transmission**     ``|T| = 1``  (energy conservation).
3. **Circular polarisation** ``F_z = -i F_x``  (Ogilvie 1963).
4. **Haskind–Newman reciprocity**  ``F_diff = F_haskind``  (no diffraction
   potential on the right-hand side).

A fifth, *relative* yardstick is the MFS (method of fundamental solutions)
independent discretisation: same kernel, point sources, no panels, no singular
quadrature.  It is not an absolute reference (same kernel) but it exposes
panel-discretisation bias the way the Ursell re-validation used it.

Run::

    python -m tensorbem.benchmarks.dean_diffraction_convergence \
        --json outputs/dean_diffraction_convergence/convergence.json \
        --figure outputs/dean_diffraction_convergence/convergence.png
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from . import dean_diffraction

#: Panel counts for the convergence ladder.
PANEL_COUNTS: tuple[int, ...] = (16, 32, 64, 128, 256)

#: The four physical laws and the result key that carries their error.
LAW_KEYS: tuple[tuple[str, str, str], ...] = (
    ("dean_zero_reflection", "reflection_modulus", "|R| (exact 0)"),
    ("unit_transmission", "transmission_error", "|T - T_exact|"),
    ("circular_polarisation", "circular_polarisation_error",
     "|F_z + i F_x| / |F_x|"),
    ("haskind_reciprocity_x", "haskind_x_rel_diff",
     "Haskind vs diffraction, F_x"),
)


def _observed_order(err_coarse: float, err_fine: float,
                    h_coarse: float, h_fine: float) -> float | None:
    """Observed order ``log(e_c/e_f)/log(h_c/h_f)``; ``None`` if undefined."""
    if (err_coarse <= 0.0 or err_fine <= 0.0
            or h_coarse <= 0.0 or h_fine <= 0.0):
        return None
    if err_coarse == err_fine:
        return None
    return math.log(err_coarse / err_fine) / math.log(h_coarse / h_fine)


def run(panel_counts: tuple[int, ...] = PANEL_COUNTS,
        radius: float = 1.0,
        submergence: float = 2.0,
        ka: float = 0.6,
        amplitude: float = 1.0,
        density: float = 1025.0,
        gravity: float = 9.80665,
        panel_nodes: int = 24,
        source_count: int = 32) -> dict[str, Any]:
    """Run the convergence ladder and assemble the report dict."""
    from ..free_surface_solver import (
        solve_diffraction_indirect, solve_radiation_indirect,
        haskind_excitation_force,
    )
    from ..analytic import ursell_cylinder_coefficients
    from .dean_diffraction import submerged_circle_mesh
    rows: list[dict[str, Any]] = []
    for n in panel_counts:
        nu = ka / radius
        circle = submerged_circle_mesh(n, radius, submergence)
        panel = solve_diffraction_indirect(
            circle, nu, amplitude, density=density, gravity=gravity,
            panel_nodes=panel_nodes)
        rad_sway = solve_radiation_indirect(
            circle, nu, "sway", density=density, panel_nodes=panel_nodes)
        rad_heave = solve_radiation_indirect(
            circle, nu, "heave", density=density, panel_nodes=panel_nodes)
        hx = haskind_excitation_force(
            circle, rad_sway, nu, amplitude, density=density, gravity=gravity)
        hz = haskind_excitation_force(
            circle, rad_heave, nu, amplitude, density=density, gravity=gravity)
        exact = ursell_cylinder_coefficients(
            radius, submergence, nu, amplitude, density=density,
            gravity=gravity)
        fx = complex(panel.excitation_force[0])
        fz = complex(panel.excitation_force[1])
        exf = complex(exact["force_x"])
        et = complex(exact["transmission"])
        rows.append({
            "panels": int(n),
            "h_max": 2.0 * math.pi * radius / n,
            "condition_number": panel.condition_number,
            "relative_residual": panel.relative_residual,
            "reflection_modulus": abs(panel.reflection_coefficient),
            "transmission_modulus": abs(panel.transmission_coefficient),
            "transmission_error": abs(panel.transmission_coefficient - et),
            "transmission_phase_degrees": math.degrees(
                math.atan2(panel.transmission_coefficient.imag,
                           panel.transmission_coefficient.real)),
            "force_x_relative_error": abs(fx - exf) / abs(exf),
            "force_z_relative_error": abs(fz - complex(exact["force_z"]))
                / abs(complex(exact["force_z"])),
            "circular_polarisation_error":
                abs(fz - (-1.0j * fx)) / abs(fx),
            "haskind_x_rel_diff": abs(fx - complex(hx)) / abs(fx),
            "haskind_z_rel_diff": abs(fz - complex(hz)) / abs(fz),
            "energy_balance_error": panel.energy_balance_error,
        })

    # Observed orders between adjacent levels (h ~ 1/N).
    orders: dict[str, list[float | None]] = {k: [] for k, _, _ in LAW_KEYS}
    for i in range(1, len(rows)):
        h_c, h_f = rows[i - 1]["h_max"], rows[i]["h_max"]
        for key, err_key, _ in LAW_KEYS:
            e_c = rows[i - 1][err_key]
            e_f = rows[i][err_key]
            orders[key].append(_observed_order(e_c, e_f, h_c, h_f))

    # Pin the N=64 row to the documented Table 7 values (regression guard).
    row64 = next(r for r in rows if r["panels"] == 64)
    table7 = {
        "reflection_modulus": 2.20e-15,
        "transmission_error": 8.39e-3,
        "circular_polarisation_error": 1.10e-14,
        "haskind_x_rel_diff": 2.16e-5,
        "mfs_force_x_relative_error": 1.51e-3,
    }

    return {
        "benchmark": "dean_submerged_cylinder_diffraction_convergence",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "parameters": {
            "radius": radius, "submergence": submergence, "ka": ka,
            "amplitude": amplitude, "density": density, "gravity": gravity,
            "panel_nodes": panel_nodes, "source_count": source_count,
        },
        "exact_reference": {
            "method": "ursell_ogilvie_multipole_closed_form",
            "transmission_modulus": 1.0,
            "transmission_phase_degrees":
                rows[0]["transmission_phase_degrees"],
            "force_x_relative_error_at_256":
                rows[-1]["force_x_relative_error"],
        },
        "rows": rows,
        "observed_orders": orders,
        "table7_pin": {
            "documented": table7,
            "measured": {
                "reflection_modulus": row64["reflection_modulus"],
                "transmission_error": row64["transmission_error"],
                "circular_polarisation_error":
                    row64["circular_polarisation_error"],
                "haskind_x_rel_diff": row64["haskind_x_rel_diff"],
                "mfs_force_x_relative_error":
                    row64["mfs_force_x_relative_error"],
            },
        },
    }


def plot(report: dict[str, Any], path: Path) -> Path:
    """Two-panel log-log convergence figure (errors vs panel count)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    ns = [r["panels"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.loglog(ns, [r["reflection_modulus"] for r in rows], "o-",
             label="|R| (exact 0)")
    ax.loglog(ns, [r["transmission_error"] for r in rows], "s-",
              label="|T − T_exact|")
    ax.loglog(ns, [r["circular_polarisation_error"] for r in rows], "^-",
              label="|F_z + iF_x|/|F_x|")
    ax.loglog(ns, [r["haskind_x_rel_diff"] for r in rows], "D-",
              label="Haskind vs diffraction (F_x)")
    ax.set_xlabel("panels N")
    ax.set_ylabel("error")
    ax.set_title("Circular-cylinder diffraction: error vs N (ka=0.6, f=2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.loglog(ns, [r["force_x_relative_error"] for r in rows], "o-",
              label="F_x vs exact")
    ax.loglog(ns, [r["force_z_relative_error"] for r in rows], "s-",
              label="F_z vs exact")
    ax.loglog(ns, [r["mfs_force_x_relative_error"] for r in rows], "--",
              label="MFS F_x vs exact")
    ax.set_xlabel("panels N")
    ax.set_ylabel("relative error")
    ax.set_title("Wave forces vs Ursell–Ogilvie closed form")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--figure", type=Path, default=None)
    args = ap.parse_args()

    report = run()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    if args.figure is not None:
        p = plot(report, args.figure)
        print(f"wrote {p}")
    # Console summary (zero hand-copy: read from the same dict).
    print(f"{'N':>5} {'|R|':>12} {'|T-Tex|':>12} {'circ.pol':>12} "
          f"{'Haskind_x':>12} {'Fx_err':>12} {'Fz_err':>12}")
    for r in report['rows']:
        print(f"{r['panels']:>5} {r['reflection_modulus']:>12.3e} "
              f"{r['transmission_error']:>12.3e} "
              f"{r['circular_polarisation_error']:>12.3e} "
              f"{r['haskind_x_rel_diff']:>12.3e} "
              f"{r['force_x_relative_error']:>12.3e} "
              f"{r['force_z_relative_error']:>12.3e}")
    print("observed orders (16->32->64->128->256):")
    for key, _, label in LAW_KEYS:
        print(f"  {label:32s} " +
              "  ".join(f"{o:.2f}" if o is not None else "  n/a"
                     for o in report["observed_orders"][key]))


if __name__ == "__main__":
    main()
