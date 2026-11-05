"""Adjudicate the one genuinely contested table cell of PR #139.

`docs/marine_bem_benchmarks.md` table 10, row 2 (16 -> 64 panels), last column:
m3306 kept the merge-base value 1.62, main recomputed it as 1.80. Both describe
the observed order of the surge wave damping B11' between the 16- and
64-panel meshes; they differ only in the mesh-size metric used in
``log(e0/e1)/log(h0/h1)``:

* nominal refinement factor 2 (``sqrt(N1/N0)``)            -> 1.62
* the meshes' actual ``h_mean`` ratio (1.87)               -> 1.80

The convention the document itself uses for the rest of the column -- and the
one m3306 argues for in the paragraph it adds -- is the measured ``h_mean``
ratio, and the merged registry pins main's 1.80 for this cell. This script
recomputes both numbers from the released code so the choice is evidence-based
rather than a coin flip.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tensorbem.benchmarks import floating_hemisphere as hemi

OUT = Path(__file__).resolve().parent / "table10_row2_order_adjudication.json"

study = hemi.run_refinement_study(subdivisions=(1, 2, 3), ka=1.0)
levels = study["levels"]
rows = []
for level in levels:
    rows.append(
        {
            "subdivisions": level["subdivisions"],
            "panel_count": level["panel_count"],
            "h_mean": level["h_mean"],
            "surge_damping_rel_error": level["surge_damping_rel_error"],
        }
    )

# 16 -> 64 panels = subdivision level 1 -> level 2
lo, hi = rows[0], rows[1]
e0, e1 = lo["surge_damping_rel_error"], hi["surge_damping_rel_error"]
order_h_mean = math.log(e0 / e1) / math.log(lo["h_mean"] / hi["h_mean"])
order_nominal = math.log(e0 / e1) / math.log(math.sqrt(hi["panel_count"] / lo["panel_count"]))
# 64 -> 256 panels: the cell m3306 corrected (1.88 -> 1.94)
mid = rows[2]
order_h_mean_256 = math.log(e1 / mid["surge_damping_rel_error"]) / math.log(hi["h_mean"] / mid["h_mean"])
order_nominal_256 = math.log(e1 / mid["surge_damping_rel_error"]) / math.log(
    math.sqrt(mid["panel_count"] / hi["panel_count"])
)

payload = {
    "benchmark": "floating_hemisphere.run_refinement_study (Hulme 1982, Ka=1.0, a=1)",
    "levels": rows,
    "table10_row2_cell": {
        "documented_by_m3306": 1.62,
        "documented_by_main": 1.80,
        "recomputed_h_mean_convention": order_h_mean,
        "recomputed_nominal_refinement_convention": order_nominal,
        "h_mean_ratio_16_to_64": lo["h_mean"] / hi["h_mean"],
        "verdict": "1.80 (measured h_mean ratio, the column's convention)",
        "abs_diff_vs_h_mean": abs(order_h_mean - 1.80),
        "abs_diff_vs_nominal": abs(order_h_mean - 1.62),
    },
    "table10_row3_cell_m3306_correction": {
        "old_value_in_base": 1.88,
        "m3306_value": 1.94,
        "recomputed_h_mean_convention": order_h_mean_256,
        "recomputed_nominal_refinement_convention": order_nominal_256,
        "h_mean_ratio_64_to_256": hi["h_mean"] / mid["h_mean"],
    },
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload["table10_row2_cell"], indent=2))
print(json.dumps(payload["table10_row3_cell_m3306_correction"], indent=2))