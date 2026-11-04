"""Symmetry-reduction benchmark: half-hull vs full-hull 3-D Kelvin BEM.

Generates every number quoted in ``docs/symmetry3d.md`` (nothing is transcribed by
hand) plus the two figures: ``fig_agreement.png`` and ``fig_cost.png``.

Run:
    PYTHONPATH=src python -m tensorbem.benchmarks.symmetry_reduction \
        --subdivisions 2 3 --output docs/symmetry3d
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.free_surface_solver3d import (
    _auto_chief_points_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.symmetry3d import (
    MODE_PARITY,
    assemble_symmetric_blocks,
    mirror_patch,
    solve_diffraction_3d_symmetric,
    solve_radiation_3d_symmetric,
    split_patch,
    verify_generalized_normal_parity,
)

NU = 1.0
PARITY = np.array([MODE_PARITY[k] for k in range(6)], dtype=float)
SAME = np.equal.outer(PARITY, PARITY)
MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def run(subdivisions: list[int], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "wavenumber": NU,
        "panels": {},
    }

    for sub in subdivisions:
        mesh = hemisphere_hull_mesh(sub)
        upper, lower = split_patch(mesh)
        entry: dict[str, Any] = {
            "full_panels": int(mesh.panel_count),
            "half_panels": int(upper.panel_count),
            "half_area_match": float(abs(upper.surface_area - lower.surface_area)),
            "parity_error": [float(v) for v in verify_generalized_normal_parity(upper)],
            "mirror_panels": int(mirror_patch(upper).panel_count),
        }

        blocks, t_blocks = _timed(assemble_symmetric_blocks, upper, NU)
        entry["assembly_entries_half"] = int(blocks.assembly_entries())
        entry["assembly_entries_full"] = int(mesh.panel_count**2)

        full_rad, t_full_rad = _timed(solve_radiation_3d, mesh, NU)
        half_rad, t_half_rad = _timed(
            solve_radiation_3d_symmetric, upper, NU, blocks=blocks
        )
        scale_a = float(np.abs(full_rad.added_mass).max())
        scale_b = float(np.abs(full_rad.damping).max())
        entry["added_mass_full"] = full_rad.added_mass.tolist()
        entry["added_mass_half"] = half_rad.added_mass.tolist()
        entry["damping_full"] = full_rad.damping.tolist()
        entry["damping_half"] = half_rad.damping.tolist()
        entry["added_mass_max_rel_err_same_parity"] = float(
            np.abs(half_rad.added_mass[SAME] - full_rad.added_mass[SAME]).max() / scale_a
        )
        entry["damping_max_rel_err_same_parity"] = float(
            np.abs(half_rad.damping[SAME] - full_rad.damping[SAME]).max() / scale_b
        )
        entry["mixed_parity_residue_full"] = float(
            np.abs(full_rad.added_mass[~SAME]).max() / scale_a
        )
        entry["mixed_parity_residue_half"] = float(
            np.abs(half_rad.added_mass[~SAME]).max()
        )
        entry["condition_full"] = float(full_rad.condition_number)
        entry["condition_half_max_parity"] = float(half_rad.condition_number)
        entry["residual_full"] = float(full_rad.relative_residual)
        entry["residual_half"] = float(half_rad.relative_residual)
        entry["timing_radiation_full_s"] = float(t_full_rad)
        entry["timing_radiation_half_s"] = float(t_half_rad)
        entry["timing_blocks_s"] = float(t_blocks)

        diff = []
        for beta in (0.0, 0.6, math.pi / 2):
            full_dif, t_full_dif = _timed(solve_diffraction_3d, mesh, NU, wave_angle=beta)
            half_dif, t_half_dif = _timed(
                solve_diffraction_3d_symmetric, upper, NU, wave_angle=beta, blocks=blocks
            )
            scale = float(np.abs(full_dif.excitation_force).max())
            diff.append(
                {
                    "beta": round(beta, 6),
                    "force_scale": scale,
                    "max_rel_err": float(
                        np.abs(half_dif.excitation_force - full_dif.excitation_force).max()
                        / scale
                    ),
                    "surge_force_full": complex(full_dif.excitation_force[0]).__repr__(),
                    "heave_force_full": complex(full_dif.excitation_force[2]).__repr__(),
                    "timing_full_s": float(t_full_dif),
                    "timing_half_s": float(t_half_dif),
                }
            )
        entry["diffraction"] = diff

        # irregular frequency: CHIEF rows must fold exactly too
        nu_irr = 2.58
        chief = _auto_chief_points_3d(mesh)
        full_irr = solve_radiation_3d(mesh, nu_irr, modes=[2], chief_points=chief)
        half_irr = solve_radiation_3d_symmetric(
            upper, nu_irr, modes=[2], chief_points=chief
        )
        half_irr_d = solve_diffraction_3d_symmetric(
            upper, nu_irr, wave_angle=0.6, chief_points=chief
        )
        full_irr_d = solve_diffraction_3d(mesh, nu_irr, wave_angle=0.6, chief_points=chief)
        entry["chief"] = {
            "nu": nu_irr,
            "added_mass_full": float(full_irr.added_mass[0, 0]),
            "added_mass_half": float(half_irr.added_mass[0, 0]),
            "damping_full": float(full_irr.damping[0, 0]),
            "damping_half": float(half_irr.damping[0, 0]),
            "relative_error_added_mass": float(
                abs(half_irr.added_mass[0, 0] - full_irr.added_mass[0, 0])
                / abs(full_irr.added_mass[0, 0])
            ),
            "relative_error_damping": float(
                abs(half_irr.damping[0, 0] - full_irr.damping[0, 0])
                / abs(full_irr.damping[0, 0])
            ),
            "diffraction_max_rel_err": float(
                np.abs(half_irr_d.excitation_force - full_irr_d.excitation_force).max()
                / np.abs(full_irr_d.excitation_force).max()
            ),
            "residual_full": float(full_irr.relative_residual),
            "residual_half": float(half_irr.relative_residual),
            "condition_full": float(full_irr.condition_number),
            "condition_half": float(half_irr.condition_number),
        }
        results["panels"][str(mesh.panel_count)] = entry

    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "table.md").write_text(_tables(results) + "\n")
    _figures(results, output)
    return results


def _tables(results: dict[str, Any]) -> str:
    lines = ["<!-- generated by tensorbem.benchmarks.symmetry_reduction -->", ""]
    lines += [
        "| full panels | half panels | unknown | A rel. err | B rel. err | mixed-parity residue (full) | cond (full / half) |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, e in results["panels"].items():
        lines.append(
            f"| {e['full_panels']} | {e['half_panels']} | {e['half_panels']} "
            f"| {e['added_mass_max_rel_err_same_parity']:.2e} "
            f"| {e['damping_max_rel_err_same_parity']:.2e} "
            f"| {e['mixed_parity_residue_full']:.2e} "
            f"| {e['condition_full']:.3f} / {e['condition_half_max_parity']:.3f} |"
        )
    lines += ["", "| full panels | assembly entries (full) | assembly entries (folded) | radiation time full [s] | folded [s] | speed-up |", "|---:|---:|---:|---:|---:|---:|"]
    for key, e in results["panels"].items():
        lines.append(
            f"| {e['full_panels']} | {e['assembly_entries_full']} | {e['assembly_entries_half']} "
            f"| {e['timing_radiation_full_s']:.2f} | {e['timing_radiation_half_s']:.2f} "
            f"| x{e['timing_radiation_full_s'] / e['timing_radiation_half_s']:.2f} |"
        )
    lines += ["", "| full panels | wave heading beta [rad] | force scale | max rel. error |", "|---:|---:|---:|---:|"]
    for key, e in results["panels"].items():
        for d in e["diffraction"]:
            lines.append(
                f"| {e['full_panels']} | {d['beta']:.3f} | {d['force_scale']:.1f} "
                f"| {d['max_rel_err']:.2e} |"
            )
    lines += [
        "",
        "| full panels | A33 full | A33 folded | rel. err | B33 full | B33 folded | rel. err | cond (full / half) |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, e in results["panels"].items():
        c = e["chief"]
        lines.append(
            f"| {e['full_panels']} | {c['added_mass_full']:.4f} | {c['added_mass_half']:.4f} "
            f"| {c['relative_error_added_mass']:.2e} | {c['damping_full']:.4f} "
            f"| {c['damping_half']:.4f} | {c['relative_error_damping']:.2e} "
            f"| {c['condition_full']:.3f} / {c['condition_half']:.3f} |"
        )
    return "\n".join(lines)


def _figures(results: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(results["panels"], key=int)
    finest = results["panels"][keys[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ref = np.asarray(finest["added_mass_full"])
    red = np.asarray(finest["added_mass_half"])
    ax = axes[0]
    image = ax.imshow(ref, cmap="viridis")
    ax.set_title(f"full hull, A [kg], {finest['full_panels']} panels")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(MODE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(MODE_NAMES, fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)

    ax = axes[1]
    scale = max(np.abs(ref).max(), 1e-30)
    residual = np.abs(red - ref) / scale
    image = ax.imshow(
        np.log10(np.maximum(residual, 1e-18)), cmap="magma_r", vmin=-17, vmax=-4
    )
    ax.set_title("log10 |A_folded - A_full| / max|A|")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(MODE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(MODE_NAMES, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(
                j,
                i,
                "0" if residual[i, j] == 0.0 else f"{residual[i, j]:.0e}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if residual[i, j] < 1e-10 else "white",
            )
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output / "fig_agreement.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    panels = np.array([results["panels"][k]["full_panels"] for k in keys], dtype=float)
    t_full = np.array([results["panels"][k]["timing_radiation_full_s"] for k in keys])
    t_half = np.array([results["panels"][k]["timing_radiation_half_s"] for k in keys])
    ax = axes[0]
    ax.loglog(panels, t_full, "o-", label="full hull (N unknowns)")
    ax.loglog(panels, t_half, "s-", label="folded half (N/2, 2 parities)")
    ax.set_xlabel("panels on the full hull")
    ax.set_ylabel("6-DOF radiation wall time [s]")
    ax.set_title("assembly + solve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for key in keys:
        entry = results["panels"][key]
        betas = [d["beta"] for d in entry["diffraction"]]
        errs = [d["max_rel_err"] for d in entry["diffraction"]]
        ax.semilogy(betas, errs, "o-", label=f"{entry['full_panels']} panels")
    ax.axhline(1e-6, color="k", ls="--", lw=1, label="1e-6 gate")
    ax.set_xlabel("wave heading beta [rad]")
    ax.set_ylabel("max rel. error, exciting force")
    ax.set_title("diffraction: folded vs full")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fig_cost.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--output", type=Path, default=Path("docs/symmetry3d"))
    args = parser.parse_args()
    results = run(sorted(set(int(s) for s in args.subdivisions)), args.output)
    for key, entry in results["panels"].items():
        print(
            f"{entry['full_panels']:5d} panels: A rel.err "
            f"{entry['added_mass_max_rel_err_same_parity']:.2e}, B rel.err "
            f"{entry['damping_max_rel_err_same_parity']:.2e}, "
            f"speed-up x{entry['timing_radiation_full_s'] / entry['timing_radiation_half_s']:.2f}"
        )


if __name__ == "__main__":
    main()