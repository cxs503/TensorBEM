#!/usr/bin/env python3
"""Generate the m17262 (CASE-BEM-SPHERE-ADDED-MASS) run artifacts.

Produces, under runs/m17262/:
  sphere_added_mass_sub{1,2,3,4}.json  per-level run artifacts (CLI entry)
  refinement_benchmark.json             L4 benchmark: ref vs got + orders + negative control
  entry.json                            entry artifact (command + full output)
  sensitivity_sub2.json                 (rho,U) rescaling invariance of the relative error
  flux_panels_sub2.csv                  per-panel computed vs exact flux (320 panels)
  fig_convergence.png                   rel error vs panels, 3% line, slope-2 reference
  fig_flux_field.png                    per-panel flux error on the icosphere (sub2)

All numbers are produced by the production entry
``python -m tensorbem.benchmarks.sphere_added_mass`` (subprocess, no re-implementation).
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "src"

LEVELS = (1, 2, 3, 4)
TOLERANCE = 0.03


def run_cli(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.sphere_added_mass", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CLI failed: {args}\n{proc.stderr}")
    return json.loads(proc.stdout)


def solve_sub2():
    """Re-run the sub2 solve directly for per-panel flux data (same production path)."""
    sys.path.insert(0, str(SRC))
    from tensorbem.geometry3d import icosphere
    from tensorbem.solver3d import solve_laplace_surface

    radius, speed, density = 1.0, 1.0, 1025.0
    mesh = icosphere(2, radius=radius)
    centroids = mesh.centroids
    cos_theta = centroids[:, 0] / np.linalg.norm(centroids, axis=1)
    boundary_potential = -0.5 * speed * radius * cos_theta
    result = solve_laplace_surface(mesh, boundary_potential, domain="exterior")
    exact_flux = -speed * cos_theta
    return mesh, cos_theta, result.normal_derivative, exact_flux


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) per-level run artifacts via the production CLI entry
    for s in LEVELS:
        out = OUT / f"sphere_added_mass_sub{s}.json"
        if not out.exists():
            run_cli(["--subdivisions", str(s), "--output", str(out)])
        print(f"level sub{s}: {out.name} ok")

    # 2) L4 benchmark ladder (ref vs got, base-2 orders, negative control)
    bench = run_cli(
        ["--ladder", *[str(s) for s in LEVELS], "--tolerance", str(TOLERANCE),
         "--output", str(OUT / "refinement_benchmark.json")]
    )
    print(
        "ladder: finest rel_err "
        f"{bench['finest_rel_error']:.4e}, first passing level "
        f"{bench['first_passing_level']} panels"
    )

    # 3) entry artifact: the single-level entry command and its output
    entry_cmd = (
        "PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass "
        "--subdivisions 2"
    )
    entry_out = run_cli(["--subdivisions", "2"])
    (OUT / "entry.json").write_text(
        json.dumps(
            {"entry_cmd": entry_cmd, "output": entry_out},
            indent=2, sort_keys=True, allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 4) sensitivity: rho=1000, U=2 -> relative error must be unchanged (linearity)
    base = json.loads((OUT / "sphere_added_mass_sub2.json").read_text())
    scaled = run_cli(["--subdivisions", "2", "--density", "1000.0", "--speed", "2.0"])
    (OUT / "sensitivity_sub2.json").write_text(
        json.dumps(
            {
                "base": {"density": 1025.0, "speed": 1.0,
                         "rel_error": base["added_mass_relative_error"]},
                "scaled": {"density": 1000.0, "speed": 2.0,
                           "rel_error": scaled["added_mass_relative_error"]},
                "rel_error_diff": abs(
                    scaled["added_mass_relative_error"]
                    - base["added_mass_relative_error"]
                ),
                "invariant": bool(
                    abs(scaled["added_mass_relative_error"]
                        - base["added_mass_relative_error"]) < 1e-12
                ),
            },
            indent=2, sort_keys=True, allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 5) per-panel flux dump (sub2)
    mesh, cos_theta, q_comp, q_exact = solve_sub2()
    centroids = mesh.centroids
    with (OUT / "flux_panels_sub2.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["panel", "cos_theta", "q_computed", "q_exact", "area"])
        for i in range(len(cos_theta)):
            writer.writerow(
                [i, f"{cos_theta[i]:.17g}", f"{q_comp[i]:.17g}",
                 f"{q_exact[i]:.17g}", f"{mesh.areas[i]:.17g}"]
            )

    # 6) figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [r["panels"] for r in bench["levels"]]
    errs = [r["rel_error"] for r in bench["levels"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(panels, errs, "o-", label="added-mass rel. error")
    ax.axhline(TOLERANCE, color="r", ls="--", lw=1, label="3% tolerance")
    ax.loglog(panels, [errs[0] * (panels[0] / p) ** 2 for p in panels],
              "k:", lw=1, label="slope 2 reference")
    ax.set_xlabel("panels")
    ax.set_ylabel("rel. error vs 0.5*rho*V")
    ax.set_title("CASE-BEM-SPHERE-ADDED-MASS refinement (icosphere, exterior Dirichlet)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergence.png")
    plt.close(fig)

    q_err = np.abs(q_comp - q_exact)
    az = np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0]))
    pol = np.degrees(np.arcsin(np.clip(centroids[:, 2] / np.linalg.norm(centroids, axis=1), -1, 1)))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.pcolormesh(az, pol, q_err, shading="auto", cmap="viridis")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("polar angle (deg)")
    ax.set_title("sub2 (320 panels): |q_computed - q_exact| per panel")
    fig.colorbar(im, ax=ax, label="|dq|")
    fig.tight_layout()
    fig.savefig(OUT / "fig_flux_field.png")
    plt.close(fig)

    print("artifacts complete")


if __name__ == "__main__":
    main()
