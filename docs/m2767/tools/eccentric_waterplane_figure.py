"""Artifacts for the eccentric-waterplane verification: figure + machine-readable numbers.

Every number in ``docs/hydrostatic_eccentric_waterplane.md`` comes from
``artifacts/hydrostatic_eccentric/waterplane_checks.json`` produced here; nothing
is transcribed by hand.

Run:
    PYTHONPATH=src python3 docs/m2767/tools/eccentric_waterplane_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from test_hydrostatic_eccentric_waterplane import (  # noqa: E402
    OBLIQUE_PARALLELOGRAM,
    OFFSET_RECTANGLE,
    OFFSET_TRIANGLE,
    prism_mesh,
)
from tensorbem.hydrostatic import (  # noqa: E402
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)

OUT = REPO / "artifacts" / "hydrostatic_eccentric"

CASES = [
    {
        "name": "offset rectangle",
        "polygon": OFFSET_RECTANGLE,
        "draft": 1.3,
        "exact": {
            "area": 4.0,
            "centroid": (1.5, 0.5),
            "moments": (4.0 / 3.0, 4.0 / 3.0, 0.0),
            "source": "A = 2*2; I_xx = 2*2^3/12; I_yy = 2*2^3/12",
        },
    },
    {
        "name": "offset right triangle",
        "polygon": OFFSET_TRIANGLE,
        "draft": 0.9,
        "exact": {
            "area": 1.2,
            "centroid": ((0.3 + 2.3 + 0.3) / 3.0, (0.2 + 0.2 + 1.4) / 3.0),
            "moments": (1.2 * 1.2**2 / 18.0, 1.2 * 2.0**2 / 18.0, -1.2 * 2.0 * 1.2 / 36.0),
            "source": "A = b h/2; I_xx = A h^2/18; I_yy = A b^2/18; I_xy = -A b h/36",
        },
    },
    {
        "name": "oblique parallelogram",
        "polygon": OBLIQUE_PARALLELOGRAM,
        "draft": 1.1,
        "exact": {
            "area": 2.76,
            "centroid": (
                float(np.mean([p[0] for p in OBLIQUE_PARALLELOGRAM])),
                float(np.mean([p[1] for p in OBLIQUE_PARALLELOGRAM])),
            ),
            "moments": (2.76 * (0.4**2 + 1.5**2) / 12.0,
                        2.76 * (2.0**2 + 0.6**2) / 12.0,
                        2.76 * (2.0 * 0.4 + 0.6 * 1.5) / 12.0),
            "source": "I_xx = A(u_y^2+v_y^2)/12; I_yy = A(u_x^2+v_x^2)/12; "
                      "I_xy = A(u_x u_y+v_x v_y)/12, u=(2,0.4), v=(0.6,1.5)",
        },
    },
]

LABELS = ["area", "c_x", "c_y", "I_xx", "I_yy", "I_xy"]


def relative_error(computed: np.ndarray, exact: np.ndarray, scale: float) -> np.ndarray:
    """Relative error, falling back to a scale-normalised absolute one at zero."""
    computed = np.asarray(computed, dtype=float)
    exact = np.asarray(exact, dtype=float)
    out = np.empty_like(computed)
    for i, (c, e) in enumerate(zip(computed, exact)):
        if abs(e) > 1e-12:
            out[i] = abs(c - e) / abs(e)
        else:
            out[i] = abs(c - e) / scale
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"cases": [], "figure": str(OUT / "eccentric_waterplane.png")}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax_left, ax_right = axes

    for index, case in enumerate(CASES):
        polygon = np.asarray(case["polygon"], dtype=float)
        hull = prism_mesh(case["polygon"], draft=case["draft"])
        area, centroid, moments = compute_waterplane_properties(hull)
        volume, centre_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)

        exact = case["exact"]
        computed_vector = np.array([area, centroid[0], centroid[1], *moments])
        exact_vector = np.array(
            [exact["area"], exact["centroid"][0], exact["centroid"][1], *exact["moments"]]
        )
        scale = float(np.max(np.abs(exact_vector)))
        error = relative_error(computed_vector, exact_vector, scale)
        payload["cases"].append(
            {
                "name": case["name"],
                "polygon": case["polygon"],
                "draft": case["draft"],
                "computed": {
                    "area": float(area),
                    "centroid": [float(centroid[0]), float(centroid[1])],
                    "moments": [float(m) for m in moments],
                    "displaced_volume": float(volume),
                    "centre_of_buoyancy": [float(v) for v in centre_of_buoyancy],
                },
                "exact": {
                    "area": float(exact["area"]),
                    "centroid": [float(exact["centroid"][0]), float(exact["centroid"][1])],
                    "moments": [float(m) for m in exact["moments"]],
                    "closed_form": exact["source"],
                },
                "relative_error": {label: float(e) for label, e in zip(LABELS, error)},
            }
        )

        closed = np.vstack([polygon, polygon[0]])
        ax_left.fill(polygon[:, 0], polygon[:, 1], alpha=0.22, color=f"C{index}")
        ax_left.plot(closed[:, 0], closed[:, 1], color=f"C{index}", lw=1.4)
        ax_left.plot(*centroid, "o", ms=9, color=f"C{index}", mfc="white",
                     label=f"{case['name']}: library centroid")
        ax_left.plot(*exact["centroid"], "+", ms=11, mew=2.0, color=f"C{index}")
        ax_right.bar(
            np.arange(len(LABELS)) + 0.26 * (index - 1),
            np.maximum(error, 1e-18),
            width=0.25,
            color=f"C{index}",
            label=case["name"],
        )

    ax_left.axhline(0.0, color="0.6", lw=0.8)
    ax_left.axvline(0.0, color="0.6", lw=0.8)
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("x [m]")
    ax_left.set_ylabel("y [m]")
    ax_left.set_title("Waterplanes do not straddle the origin\n(circles = library centroid, + = closed form)")
    ax_left.legend(fontsize=7, loc="upper left")
    ax_left.grid(alpha=0.3)

    ax_right.set_yscale("log")
    ax_right.set_xticks(np.arange(len(LABELS)))
    ax_right.set_xticklabels(LABELS)
    ax_right.axhline(1e-12, color="0.4", ls="--", lw=1.0, label="double precision floor")
    ax_right.set_ylabel("relative error vs closed form")
    ax_right.set_title("Every waterplane integral is exact to rounding")
    ax_right.legend(fontsize=7)
    ax_right.grid(alpha=0.3, axis="y")

    fig.suptitle("Eccentric waterplane verification (tensorbem.hydrostatic)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "eccentric_waterplane.png", dpi=150, bbox_inches="tight")

    closure_path = OUT / "closure.json"
    if closure_path.exists():
        payload["closure"] = json.loads(closure_path.read_text(encoding="utf-8"))
    for tag in ("baseline", "after"):
        report = REPO / "artifacts" / "mutation_closure" / tag / "mutation_report.json"
        if report.exists():
            data = json.loads(report.read_text(encoding="utf-8"))
            payload[f"mutation_{tag}"] = {
                module: {
                    "killed": stats["killed"],
                    "survived": stats["survived"],
                    "mutation_score": stats["mutation_score"],
                    "survivors": stats["survivors"],
                }
                for module, stats in data["modules"].items()
            }
    (OUT / "waterplane_checks.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    worst = max(
        e for case in payload["cases"] for e in case["relative_error"].values()
    )
    print(f"worst relative error over all waterplane quantities: {worst:.3e}")
    print(f"artifacts: {OUT}")


if __name__ == "__main__":
    main()