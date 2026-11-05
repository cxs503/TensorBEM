"""Artifacts for the classical-layer pin closure (mission m3508).

Reproduces the two numbers the README quotes as the classical-layer headline and
plots their convergence:

* 2-D elliptic cylinder, both motion axes, 16/32/64/128 panels (fresh runs);
* 3-D ellipsoid ``(1.5, 1.0, 0.75)``, 20/80/320/1280 panels.  The 320- and
  1280-panel points are read back out of ``docs/benchmark_pins.json`` -- they
  were measured by ``python -m tensorbem.benchmarks.doc_pins --write`` and
  re-measured by ``--check --include-slow``, so re-running the 1280-panel case
  (~90 s) here would only duplicate evidence.

Run:  PYTHONPATH=src python3 docs/m3508/make_artifacts.py
Writes: docs/m3508/outputs/*.json, docs/m3508/figures/classical_layer_pins.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem.benchmarks import ellipse_added_mass, ellipsoid_added_mass  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PINS = REPO_ROOT / "docs" / "benchmark_pins.json"
OUT = HERE / "outputs"
FIG = HERE / "figures"

PANELS_2D = (16, 32, 64, 128)
SUBDIVISIONS_3D = (0, 1)  # 20 and 80 panels; 320/1280 come from the pin file


def ellipse_study() -> dict[str, object]:
    rows = {}
    for panels in PANELS_2D:
        runs = {axis: ellipse_added_mass.run(panels, axis=axis) for axis in ("x", "y")}
        rows[str(panels)] = {
            axis: {
                "h_max": runs[axis]["h_max"],
                "added_mass_per_unit_span": runs[axis]["added_mass_per_unit_span"],
                "added_mass_relative_error": runs[axis]["added_mass_relative_error"],
                "normal_derivative_relative_l2_error": runs[axis][
                    "normal_derivative_relative_l2_error"
                ],
            }
            for axis in ("x", "y")
        }
    return {
        "benchmark": "two_dimensional_translating_ellipse",
        "semi_major": 1.5,
        "semi_minor": 0.75,
        "density": 1025.0,
        "rows": rows,
    }


def ellipsoid_study() -> dict[str, object]:
    pins = json.loads(PINS.read_text(encoding="utf-8"))["pins"]
    rows: dict[str, object] = {}
    for subdivisions in SUBDIVISIONS_3D:
        result = ellipsoid_added_mass.run(subdivisions)
        rows[str(result["panels"])] = {
            "source": "fresh run",
            "subdivisions": subdivisions,
            "h_max": result["h_max"],
            "volume_ratio": result["volume_ratio"],
            "axes": [
                {
                    "added_mass": row["added_mass"],
                    "exact_added_mass": row["exact_added_mass"],
                    "added_mass_relative_error": row["added_mass_relative_error"],
                    "normal_derivative_relative_l2_error": row[
                        "normal_derivative_relative_l2_error"
                    ],
                }
                for row in result["axes"]
            ],
        }
    for panels, case in ((320, "ellipsoid_added_mass_320"),
                         (1280, "ellipsoid_added_mass_1280")):
        metrics = pins[case]["metrics"]
        rows[str(panels)] = {
            "source": f"docs/benchmark_pins.json::{case}",
            "h_max": metrics["h_max"],
            "volume_ratio": metrics["volume_ratio"],
            "axes": [
                {
                    "added_mass": metrics[f"axes.{axis}.added_mass"],
                    "added_mass_relative_error": metrics[
                        f"axes.{axis}.added_mass_relative_error"
                    ],
                    "normal_derivative_relative_l2_error": metrics[
                        f"axes.{axis}.normal_derivative_relative_l2_error"
                    ],
                }
                for axis in range(3)
            ],
        }
    return {
        "benchmark": "three_dimensional_translating_ellipsoid",
        "semi_axes": [1.5, 1.0, 0.75],
        "density": 1025.0,
        "rows": rows,
    }


def _order(previous: float, current: float, h_ratio: float) -> float:
    return math.log(previous / current) / math.log(h_ratio)


def figure(ellipse: dict[str, object], ellipsoid: dict[str, object]) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for axis, colour in (("x", "C0"), ("y", "C1")):
        h = [ellipse["rows"][str(n)][axis]["h_max"] for n in PANELS_2D]
        error = [
            ellipse["rows"][str(n)][axis]["added_mass_relative_error"] for n in PANELS_2D
        ]
        left.loglog(h, error, "o-", color=colour, label=f"ellipse {axis} axis")
        left.annotate(
            f"{error[-1]:.3e}",
            (h[-1], error[-1]),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
            color=colour,
        )
    # Reference slopes through the finest 2-D point.
    h_ref = [ellipse["rows"][str(n)]["x"]["h_max"] for n in PANELS_2D]
    e_ref = ellipse["rows"]["128"]["x"]["added_mass_relative_error"]
    for slope, style in ((2.0, "--"), (1.0, ":")):
        left.loglog(
            h_ref,
            [e_ref * (value / h_ref[-1]) ** slope for value in h_ref],
            style,
            color="grey",
            lw=1.0,
            label=f"slope {slope:g}",
        )
    left.set_xlabel(r"$h_{\max}$")
    left.set_ylabel("added-mass relative error")
    left.set_title("2-D elliptic cylinder (exact exterior operator)")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=8)

    counts = sorted(int(key) for key in ellipsoid["rows"])
    colours = ("C0", "C1", "C2")
    for axis in range(3):
        h = [ellipsoid["rows"][str(n)]["h_max"] for n in counts]
        error = [
            ellipsoid["rows"][str(n)]["axes"][axis]["added_mass_relative_error"]
            for n in counts
        ]
        right.loglog(h, error, "o-", color=colours[axis], label=f"ellipsoid axis {axis}")
        right.annotate(
            f"{error[-1]:.4e}",
            (h[-1], error[-1]),
            textcoords="offset points",
            xytext=(4, -11),
            fontsize=8,
            color=colours[axis],
        )
    h_all = [ellipsoid["rows"][str(n)]["h_max"] for n in counts]
    e_fine = ellipsoid["rows"]["1280"]["axes"][0]["added_mass_relative_error"]
    right.loglog(
        h_all,
        [e_fine * (value / h_all[-1]) ** 2.0 for value in h_all],
        "--",
        color="grey",
        lw=1.0,
        label="slope 2",
    )
    right.set_xlabel(r"$h_{\max}$")
    right.set_ylabel("added-mass relative error")
    right.set_title("3-D ellipsoid (1.5, 1.0, 0.75)")
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize=8)

    fig.suptitle(
        "Classical-layer claims now pinned: README 7.54e-4 / 9.00e-4 (128 panels) and "
        "4.1679e-3 / 4.4621e-3 / 4.9359e-3 (1280 panels)",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "classical_layer_pins.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ellipse = ellipse_study()
    ellipsoid = ellipsoid_study()
    (OUT / "ellipse_convergence.json").write_text(
        json.dumps(ellipse, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "ellipsoid_convergence.json").write_text(
        json.dumps(ellipsoid, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path = figure(ellipse, ellipsoid)
    print("figure:", path)
    for panels in PANELS_2D:
        row = ellipse["rows"][str(panels)]
        print(f"ellipse {panels:4d}  x={row['x']['added_mass_relative_error']:.6e} "
              f"y={row['y']['added_mass_relative_error']:.6e}")
    for panels in sorted(int(key) for key in ellipsoid["rows"]):
        row = ellipsoid["rows"][str(panels)]
        errors = " ".join(f"{axis['added_mass_relative_error']:.6e}" for axis in row["axes"])
        print(f"ellipsoid {panels:5d}  {errors}  [{row['source']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())