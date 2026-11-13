"""Field figure for CASE-BEM-HULME-HEMISPHERE: the solved body potential.

The comparison curves (figure.png) show the integral quantities; this figure
shows the *field* those integrals come from: the per-panel radiation potential
``phi_j`` on the wetted hemisphere for the surge (j = 1, along the wave) and
heave (j = 3) unit-velocity problems, at ka = 1 and ka = 2.  The potential is
the second-kind solution of (1/2 I - H) phi = S(-n_j); the phase map shows the
fore/aft antisymmetry of surge and the axisymmetry of heave, which is the visual
counterpart of the A11/A33/B11/B33 columns of the case report.

    python benchmarks/verified/casepool_bem_hulme_hemisphere/make_field_figure.py

Output: ``figure_body_potential.png`` next to this file (drawn from a real
``solve_radiation_3d`` run, 256-panel mesh; the numbers are not reused from
results.json because the field is not stored there).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.benchmarks import hulme_hemisphere_case as case
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import hemisphere_hull_mesh


def solve_fields(ka: float, subdivisions: int) -> dict:
    mesh = hemisphere_hull_mesh(subdivisions, radius=case.RADIUS)
    nu = ka / case.RADIUS
    result = solve_radiation_3d(mesh, nu, modes=(0, 2), density=case.DENSITY,
                                gravity=case.GRAVITY, chief_points="auto")
    centres = np.asarray(mesh.centroids, dtype=float)
    phi = np.asarray(result.potentials, dtype=complex)
    return {
        "ka": ka,
        "panels": mesh.panel_count,
        "condition_number": float(result.condition_number),
        "centres": centres.tolist(),
        "surge_abs": np.abs(phi[:, 0]).tolist(),
        "surge_phase": np.angle(phi[:, 0]).tolist(),
        "heave_abs": np.abs(phi[:, 1]).tolist(),
        "heave_phase": np.angle(phi[:, 1]).tolist(),
    }


def draw(fields: list[dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 6.4))
    for row, field in enumerate(fields):
        centres = np.asarray(field["centres"])
        for col, (key, title) in enumerate((
            ("surge_abs", f"|phi| surge (j=1), ka={field['ka']:g}"),
            ("heave_abs", f"|phi| heave (j=3), ka={field['ka']:g}"),
            ("surge_phase", f"arg phi surge (j=1), ka={field['ka']:g}"),
        )):
            ax = fig.add_subplot(len(fields), 3, row * 3 + col + 1,
                                 projection="3d")
            values = np.asarray(field[key])
            if key.endswith("phase"):
                sc = ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2],
                                c=values, cmap="twilight", s=6, vmin=-math.pi,
                                vmax=math.pi)
            else:
                sc = ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2],
                                c=values, cmap="viridis", s=6)
            ax.set_title(title, fontsize=8)
            ax.set_box_aspect((1, 1, 0.55))
            ax.set_xlabel("x/a", fontsize=7)
            ax.set_ylabel("y/a", fontsize=7)
            ax.set_zlabel("z/a", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    fig.suptitle("CASE-BEM-HULME-HEMISPHERE: per-panel radiation potential on the "
                 "wetted hemisphere (constant panels, CHIEF on)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subdivisions", type=int, default=case.COARSE_SUBDIVISIONS)
    parser.add_argument("--ka", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [solve_fields(ka, args.subdivisions) for ka in args.ka]
    figure = draw(fields, out_dir / "figure_body_potential.png")
    (out_dir / "field_data.json").write_text(
        json.dumps({"subdivisions": args.subdivisions,
                    "density": case.DENSITY, "gravity": case.GRAVITY,
                    "fields": fields}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {figure} and field_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())