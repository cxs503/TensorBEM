"""Regenerate the verification figure from the committed study artifact.

    python docs/mixed_boundary/make_figure.py

Left panel: measured relative L2 errors against the closed-form harmonic
solutions, with the observed orders annotated.  Right panel: the flux error of
the mixed solve versus the all-Dirichlet solve of the same analytic field,
which shows the first-order level is set by the square corners and not by the
boundary-condition mix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "outputs" / "study.json"
FIGURE = HERE / "figures" / "mixed_boundary_convergence.png"


def main() -> int:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in document["cases"]}
    mixed = cases["adjacent_mixed"]
    pure = cases["pure_neumann"]
    baseline = document["dirichlet_flux_baseline"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    series = (
        (mixed, "q_unknown_side_rel_l2", "mixed: $q$ on Dirichlet sides", "o", "-"),
        (mixed, "u_unknown_side_rel_l2", "mixed: $u$ on Neumann sides", "s", "-"),
        (mixed, "interior_field_rel_l2", "mixed: interior field", "^", "-"),
        (pure, "u_unknown_side_rel_l2", "pure Neumann: $u$ (gauge)", "D", "--"),
        (pure, "interior_field_rel_l2", "pure Neumann: interior field", "v", "--"),
    )
    for case, key, label, marker, style in series:
        x = [row["panel_count"] for row in case["rows"]]
        y = [row[key] for row in case["rows"]]
        orders = case["observed_order"].get(key, [])
        tag = f"{label} (order {orders[-1]:.2f})" if orders else label
        left.loglog(x, y, style, marker=marker, label=tag, linewidth=1.4, markersize=5)
    panels = mixed["rows"]
    reference = [panels[0]["panel_count"], panels[-1]["panel_count"]]
    anchor = panels[0]["interior_field_rel_l2"] * 2.0
    left.loglog(
        reference,
        [anchor, anchor * (reference[0] / reference[1]) ** 2],
        "k:",
        linewidth=1.0,
        label="slope 2",
    )
    left.set_xlabel("boundary panels")
    left.set_ylabel("relative $L_2$ error")
    left.set_title("(a) closed-form harmonic verification")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=7.5, loc="lower left")

    x = [row["panel_count"] for row in baseline["rows"]]
    right.loglog(
        x,
        [row["q_rel_l2"] for row in baseline["rows"]],
        "-o",
        label="all Dirichlet ($u=e^x\\cos y$)",
        linewidth=1.4,
        markersize=5,
    )
    right.loglog(
        [row["panel_count"] for row in panels],
        [row["q_unknown_side_rel_l2"] for row in panels],
        "-s",
        label="mixed Dirichlet/Neumann",
        linewidth=1.4,
        markersize=5,
    )
    right.set_xlabel("boundary panels")
    right.set_ylabel("relative $L_2$ error in $q$")
    right.set_title("(b) flux error is corner-limited, not mixing-limited")
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize=8)
    first_order = baseline["rows"][0]["q_rel_l2"] * 6.0
    right.loglog(
        [x[0], x[-1]],
        [first_order, first_order * x[0] / x[-1]],
        "k:",
        linewidth=1.0,
        label="slope 1",
    )
    right.legend(fontsize=8)

    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=110)
    print(f"wrote {FIGURE} ({FIGURE.stat().st_size / 1024:.1f} kB)")
    print("interior field order:", mixed["observed_order"]["interior_field_rel_l2"])
    print(
        "u trace order:",
        mixed["observed_order"]["u_unknown_side_rel_l2"],
        math.nan,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())