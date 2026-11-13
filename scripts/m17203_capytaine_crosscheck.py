"""CASE-BEM-DEAN-DIFFRACTION — independent cross-software check (capytaine 3-D BEM).

The 2-D deep-water reference (Ursell/Ogilvie multipole) is compared with
TensorBEM's 2-D panel route in ``scripts/m17203_dean_case.py``.  This script
adds a third, *external* route: capytaine's 3-D free-surface BEM applied to a
long horizontal cylinder (axis along y, radius a=1 m, axis depth f=2 m,
infinite water depth).  For a cylinder of length L the force per unit length
F_x/L converges to the 2-D value as L/lambda grows, so the L-sweep doubles as
the 2-D limit extrapolation.

Output: artifacts/m17203_dean_diffraction/capytaine_crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import capytaine as cpy
from capytaine.bem.airy_waves import froude_krylov_force
from tensorbem.analytic import dean_cylinder_diffraction

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "m17203_dean_diffraction"
OUT.mkdir(parents=True, exist_ok=True)

A = 1.0
F = 2.0
KA = 0.6
AMP = 1.0
RHO = 1025.0
G = 9.80665


def run_one(length: float, ntheta: int, nx: int) -> dict:
    t0 = time.time()
    k = KA / A
    omega = math.sqrt(G * k)
    # Closed cylinder: nr>0 end caps, otherwise the shell is open and the
    # inner-surface pressure cancels roughly half of the outer force.
    mesh = cpy.mesh_horizontal_cylinder(
        length=length,
        radius=A,
        center=(0.0, 0.0, -F),
        resolution=(4, ntheta, nx),
    ).rotated_z(0.5 * math.pi)  # axis along y, waves travel along +x
    body = cpy.FloatingBody(mesh=mesh)
    body.add_translation_dof(direction=(1.0, 0.0, 0.0), name="Surge")
    body.add_translation_dof(direction=(0.0, 0.0, 1.0), name="Heave")
    problem = cpy.DiffractionProblem(
        body=body, wave_direction=0.0, omega=omega, rho=RHO, g=G, water_depth=None
    )
    solver = cpy.BEMSolver()
    result = solver.solve(problem)
    # capytaine's DiffractionResult.forces holds the DIFFRACTION force only;
    # the excitation force is Froude-Krylov + diffraction (docs: diffraction_force
    # returned separately from Froude_Krylov_force).
    fk = froude_krylov_force(problem)
    fx = complex(result.forces["Surge"]) + complex(fk["Surge"])
    fz = complex(result.forces["Heave"]) + complex(fk["Heave"])
    return {
        "length_over_lambda": length / (2.0 * math.pi / k),
        "length": length,
        "ntheta": ntheta,
        "nx": nx,
        "n_panels": int(mesh.nb_faces),
        "force_x_total": [fx.real, fx.imag],
        "force_z_total": [fz.real, fz.imag],
        "froude_krylov_x_per_length": [complex(fk["Surge"]).real / length,
                                       complex(fk["Surge"]).imag / length],
        "diffraction_x_per_length": [complex(result.forces["Surge"]).real / length,
                                     complex(result.forces["Surge"]).imag / length],
        "force_x_per_length": [fx.real / length, fx.imag / length],
        "force_z_per_length": [fz.real / length, fz.imag / length],
        "runtime_seconds": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=float, nargs="+", default=[10.0, 20.0, 40.0])
    ap.add_argument("--ntheta", type=int, default=32)
    ap.add_argument("--panels-per-length", type=float, default=4.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    exact = dean_cylinder_diffraction(A, F, KA, AMP, RHO, gravity=G)
    ex_fx = complex(exact["force_x"])
    ex_fz = complex(exact["force_z"])

    rows = []
    for length in args.lengths:
        nx = max(4, int(round(args.panels_per_length * length)))
        row = run_one(length, args.ntheta, nx)
        fx = complex(*row["force_x_per_length"])
        fz = complex(*row["force_z_per_length"])
        row["exact_force_x_per_length"] = [ex_fx.real, ex_fx.imag]
        row["exact_force_z_per_length"] = [ex_fz.real, ex_fz.imag]
        row["force_x_relative_error"] = abs(fx - ex_fx) / abs(ex_fx)
        row["force_z_relative_error"] = abs(fz - ex_fz) / abs(ex_fz)
        rows.append(row)
        print(
            f"L={length:5.1f} (L/lam={row['length_over_lambda']:4.2f}) N={row['n_panels']:5d} "
            f"|Fx|/L={abs(fx):10.2f} err={row['force_x_relative_error']*100:7.3f}% "
            f"|Fz|/L={abs(fz):10.2f} err={row['force_z_relative_error']*100:7.3f}% "
            f"({row['runtime_seconds']:.1f}s)",
            flush=True,
        )

    payload = {
        "solver": "capytaine " + cpy.__version__,
        "case": {"radius_m": A, "submergence_m": F, "ka": KA, "amplitude_m": AMP,
                 "density_kg_m3": RHO, "gravity_m_s2": G, "water_depth": "infinite"},
        "reference": "Ursell(1950)/Ogilvie(1963) multipole closed form (2-D)",
        "note": "3-D force divided by cylinder length; L/lambda -> inf recovers the"
                " 2-D force per unit length.",
        "rows": rows,
    }
    name = f"capytaine_crosscheck{args.tag}.json"
    (OUT / name).write_text(json.dumps(payload, indent=2))
    print("wrote", OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())