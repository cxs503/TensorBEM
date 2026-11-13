"""CASE-BEM-DEAN-DIFFRACTION (m17203) — 2-D deep-water wave diffraction past a
submerged circular cylinder, TensorBEM.

Pre-declared design (experimental-design skill, thresholds fixed before runs):

  decision      Does TensorBEM's 2-D deep-water diffraction route reproduce the
                published Dean/Ursell/Ogilvie multipole reference within the
                mission tolerance (<5%) for the anchor case, and does it stay
                within 1% at production resolution?
  alternatives  H1: all measured quantities agree within tolerance (route valid);
                H0: at least one quantity leaves tolerance (route defective).
  metrics       relative error of the complex excitation force components
                F_x, F_z; absolute |R| (Dean's theorem: exactly 0); |T|-1;
                transmission-phase error (deg); Haskind-vs-direct difference;
                free-surface energy-balance error.
  unit          one (ka, f/a, N_panels, nodes/panel) computation.
  baseline      Ursell (1950)/Ogilvie (1963) multipole closed form
                (``tensorbem.analytic.dean_cylinder_diffraction``), plus two
                independent numeric routes on the same geometry:
                MFS collocation reference and Haskind–Newman reciprocity.
  thresholds    mission: <5% on every force/phase metric; internal target 1%
                at N=256; |R| < 1e-9 absolute (theorem, not an error bar);
                nominal resolution N=128.
  sensitivity   ka sweep, submergence sweep, panels N, quadrature nodes/panel.
  analysis      relative error |got-ref|/|ref| (forces), absolute difference
                (phase in deg, |R|, |T|-1); no outlier removal; all runs kept.

Usage:
  PYTHONPATH=src python3 scripts/m17203_dean_case.py --stage core
  PYTHONPATH=src python3 scripts/m17203_dean_case.py --stage panels
  PYTHONPATH=src python3 scripts/m17203_dean_case.py --stage nodes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tensorbem.analytic import dean_cylinder_diffraction
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import (
    diffraction_reference,
    haskind_excitation_force,
    solve_diffraction_indirect,
    solve_radiation_indirect,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "m17203_dean_diffraction"
OUT.mkdir(parents=True, exist_ok=True)

A = 1.0            # cylinder radius [m]
F_NOM = 2.0        # axis submergence [m]  (f/a = 2)
KA_NOM = 0.6       # wavenumber x radius
AMP = 1.0          # incident wave amplitude [m]
RHO = 1025.0       # water density [kg/m^3]
G = 9.80665        # gravity [m/s^2]
N_NOM = 128        # nominal panel count
NODES_NOM = 24     # Gauss nodes per panel for the free-surface integrals


def _c(v: complex) -> list[float]:
    return [float(v.real), float(v.imag)]


def rel(got: complex, ref: complex) -> float:
    return abs(got - ref) / abs(ref)


def one_run(
    ka: float = KA_NOM,
    submergence: float = F_NOM,
    panels: int = N_NOM,
    panel_nodes: int = NODES_NOM,
    source_count: int = 32,
) -> dict:
    """One computation: analytic reference + panel BEM + MFS + Haskind routes."""
    t0 = time.time()
    nu = ka / A
    mesh = submerged_circle_mesh(panels, A, submergence)
    exact = dean_cylinder_diffraction(A, submergence, ka, AMP, RHO, gravity=G)

    panel = solve_diffraction_indirect(
        mesh, nu, AMP, density=RHO, gravity=G, panel_nodes=panel_nodes
    )
    mfs = diffraction_reference(
        mesh, nu, AMP, source_count=source_count, density=RHO, gravity=G
    )
    rad_sway = solve_radiation_indirect(
        mesh, nu, "sway", density=RHO, panel_nodes=panel_nodes
    )
    rad_heave = solve_radiation_indirect(
        mesh, nu, "heave", density=RHO, panel_nodes=panel_nodes
    )
    hask_x = complex(haskind_excitation_force(mesh, rad_sway, nu, AMP, density=RHO, gravity=G))
    hask_z = complex(haskind_excitation_force(mesh, rad_heave, nu, AMP, density=RHO, gravity=G))

    ex_fx = complex(exact["force_x"])
    ex_fz = complex(exact["force_z"])
    ex_t = complex(exact["transmission"])
    p_fx = complex(panel.excitation_force[0])
    p_fz = complex(panel.excitation_force[1])
    m_fx = complex(mfs.excitation_force[0])

    phase_exact = math.degrees(math.atan2(ex_t.imag, ex_t.real))
    phase_panel = math.degrees(
        math.atan2(panel.transmission_coefficient.imag, panel.transmission_coefficient.real)
    )
    # Circular polarisation: F_z = -i F_x  =>  phase(F_z) - phase(F_x) = -90 deg
    dphi_exact = math.degrees(np.angle(ex_fz) - np.angle(ex_fx))
    dphi_panel = math.degrees(np.angle(p_fz) - np.angle(p_fx))
    # Morison inertia limit (f/a -> infinity):  |F_x| -> 2*pi*rho*g*a*A*ka*exp(-k f)
    morison = 2.0 * math.pi * RHO * G * A * AMP * ka * math.exp(-ka * submergence / A)

    return {
        "ka": ka,
        "submergence_over_radius": submergence / A,
        "wavenumber_times_submergence": ka * submergence / A,
        "panels": panels,
        "panel_nodes": panel_nodes,
        "sources": source_count,
        "exact_force_x": _c(ex_fx),
        "exact_force_z": _c(ex_fz),
        "exact_force_x_modulus": abs(ex_fx),
        "exact_force_x_nondim": abs(ex_fx) / (RHO * G * A * AMP),
        "exact_transmission_modulus": abs(ex_t),
        "exact_transmission_phase_degrees": phase_exact,
        "exact_phase_z_minus_x_degrees": dphi_exact,
        "morison_limit_force_x": morison,
        "morison_over_exact": morison / abs(ex_fx),
        "panel_force_x": _c(p_fx),
        "panel_force_z": _c(p_fz),
        "panel_force_x_relative_error": rel(p_fx, ex_fx),
        "panel_force_z_relative_error": rel(p_fz, ex_fz),
        "panel_reflection_modulus": abs(panel.reflection_coefficient),
        "panel_transmission_modulus": abs(panel.transmission_coefficient),
        "panel_transmission_phase_degrees": phase_panel,
        "panel_transmission_phase_error_degrees": abs(phase_panel - phase_exact),
        "panel_energy_balance_error": panel.energy_balance_error,
        "panel_phase_z_minus_x_degrees": dphi_panel,
        "panel_condition_number": panel.condition_number,
        "panel_relative_residual": panel.relative_residual,
        "mfs_force_x": _c(m_fx),
        "mfs_force_x_relative_error": rel(m_fx, ex_fx),
        "mfs_reflection_modulus": abs(mfs.reflection_coefficient),
        "mfs_transmission_modulus_error": abs(abs(mfs.transmission_coefficient) - 1.0),
        "haskind_force_x": _c(hask_x),
        "haskind_force_z": _c(hask_z),
        "haskind_vs_direct_x": rel(hask_x, p_fx),
        "haskind_vs_direct_z": rel(hask_z, p_fz),
        "haskind_vs_exact_x": rel(hask_x, ex_fx),
        "runtime_seconds": time.time() - t0,
    }


def sweep(name: str, cases: list[dict]) -> list[dict]:
    rows = []
    for kwargs in cases:
        row = one_run(**kwargs)
        row["sweep"] = name
        rows.append(row)
        print(
            f"[{name}] ka={row['ka']:.2f} f/a={row['submergence_over_radius']:.1f} "
            f"N={row['panels']} nodes={row['panel_nodes']} | "
            f"Fx err={row['panel_force_x_relative_error']*100:.3f}% "
            f"phase err={row['panel_transmission_phase_error_degrees']:.3f} deg "
            f"|R|={row['panel_reflection_modulus']:.2e} "
            f"haskind={row['haskind_vs_direct_x']*100:.3f}% "
            f"({row['runtime_seconds']:.1f}s)",
            flush=True,
        )
    return rows


STAGES = {
    "core": lambda: sweep(
        "core",
        [
            {"ka": KA_NOM, "submergence": F_NOM, "panels": N_NOM, "panel_nodes": NODES_NOM},
        ],
    ),
    "panels": lambda: sweep(
        "panel_convergence",
        [{"panels": n, "panel_nodes": NODES_NOM} for n in (32, 64, 96, 128, 192, 256, 384)],
    ),
    "nodes": lambda: sweep(
        "quadrature_sensitivity",
        [{"panels": N_NOM, "panel_nodes": n} for n in (6, 8, 12, 16, 24, 32, 48)],
    ),
    "ka": lambda: sweep(
        "frequency_sweep",
        [{"ka": k, "panels": N_NOM, "panel_nodes": NODES_NOM}
         for k in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)],
    ),
    "submergence": lambda: sweep(
        "submergence_sweep",
        [{"submergence": f, "panels": N_NOM, "panel_nodes": NODES_NOM}
         for f in (1.5, 2.0, 3.0, 5.0, 8.0)],
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core", choices=sorted(STAGES) + ["all"])
    args = ap.parse_args(argv)

    stages = sorted(STAGES) if args.stage == "all" else [args.stage]
    all_rows: list[dict] = []
    for stage in stages:
        rows = STAGES[stage]()
        (OUT / f"stage_{stage}.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "environment": {
                        "python": sys.version.split()[0],
                        "numpy": np.__version__,
                        "platform": platform.platform(),
                    },
                    "parameters": {
                        "radius_m": A, "amplitude_m": AMP, "density_kg_m3": RHO,
                        "gravity_m_s2": G, "nominal_ka": KA_NOM,
                        "nominal_submergence_m": F_NOM, "nominal_panels": N_NOM,
                        "nominal_panel_nodes": NODES_NOM,
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )
        all_rows.extend(rows)

    if len(stages) > 1:
        keys = sorted({k for r in all_rows for k in r if k != "sweep"})
        with (OUT / "all_stages.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["sweep"] + keys)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                                 for k, v in row.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())