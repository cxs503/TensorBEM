"""Regenerate the finite-depth radiation artefacts (JSON table + figure).

Run from the repository root with ``PYTHONPATH=src``::

    PYTHONPATH=src python3 docs/finite_depth_radiation/make_artifacts.py

A full run takes about two and a half minutes because each finite-depth
boundary-element assembly is O(panels^2 * nodes * modes); nothing in
``finite_depth_radiation.json`` or ``fig_finite_depth_radiation.png`` is
transcribed by hand.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorbem.benchmarks.finite_depth_radiation import (
    REFERENCE_ADDED_MASS,
    REFERENCE_DAMPING,
    build_record,
)

HERE = pathlib.Path(__file__).resolve().parent


def make_figure(record: dict) -> pathlib.Path:
    limit = record["deep_water_limit"]
    identity = record["energy_identity"]
    convergence = record["panel_convergence"]

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axis = axes[0]
    nu_h = [row["nu_times_depth"] for row in limit]
    axis.loglog(nu_h, [row["mfs_added_mass_vs_ursell"] for row in limit], "o-",
                label="added mass")
    axis.loglog(nu_h, [row["mfs_damping_vs_ursell"] for row in limit], "s-",
                label="damping")
    axis.set_xlabel(r"$\nu h$")
    axis.set_ylabel("relative error vs Ursell (deep water)")
    axis.set_title("(a) depth $\\to \\infty$ limit (MFS)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    axis = axes[1]
    labels = [row["case"].replace("deep water", "deep") for row in identity]
    index = range(len(labels))
    axis.bar([i - 0.2 for i in index],
             [row["panel_relative_residual"] for row in identity], width=0.4,
             label="panel method")
    axis.bar([i + 0.2 for i in index],
             [row["mfs_relative_residual"] for row in identity], width=0.4,
             label="MFS")
    axis.set_yscale("log")
    axis.set_xticks(list(index))
    axis.set_xticklabels(labels, rotation=20)
    axis.set_ylabel(r"$|B_{far}/B_{near} - 1|$")
    axis.set_title("(b) energy identity, depth independent")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()

    axis = axes[2]
    panels = [row["panels"] for row in convergence]
    axis.loglog(panels, [row["added_mass_relative_error"] for row in convergence],
                "o-", label="added mass")
    axis.loglog(panels, [row["damping_relative_error"] for row in convergence],
                "s-", label="damping")
    axis.set_xlabel("panels")
    axis.set_ylabel("relative error vs MFS")
    axis.set_title(f"(c) panel convergence, $h = {convergence[0]['depth']:g}$ m")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    figure.tight_layout()
    path = HERE / "fig_finite_depth_radiation.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    record = build_record()
    json_path = HERE / "finite_depth_radiation.json"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    figure_path = make_figure(record)
    print(f"wrote {json_path} and {figure_path}")
    print(
        "deep-water reference: A = "
        f"{REFERENCE_ADDED_MASS}, B = {REFERENCE_DAMPING}"
    )


if __name__ == "__main__":
    main()