"""Reconcile the two registry families that the merge put side by side.

The merge unions two independently developed registrations of the *same*
document tables:

* the PR branch (``agent/m3181``) generates one ``DeclaredNumber`` per table
  cell from :data:`_TABLE_HULME` / :data:`_TABLE_WANG_LINTON`
  (groups ``hulme3d`` / ``wang_linton3d``);
* ``main`` states the same cells inline (groups ``hulme_hemisphere`` /
  ``submerged_sphere``), after the m3150 recomputation corrected two columns of
  the document.

A pure line-union keeps both families, but two cells then carry *mutually
exclusive* claims: the generated family still quotes the pre-m3150 digits.
Keeping them would either resurrect a corrected value or turn the merged suite
red, so the generated family is brought in line with the (documented, evidence-
backed) main-branch convention:

1. B11 observed order.  The generated family derived it as
   ``log(e0/e1)/log 2`` (ideal mesh halving), while every other column -- and
   ``main``'s ``_study_rows`` -- uses the *measured* mean-edge ratio
   ``log(e0/e1)/log(h0/h1)``, ``h0/h1 = 1.87`` at these meshes.  Table 10 now
   quotes the latter (1.80 / 1.94), so the runner is fixed and the two cells
   are updated.
2. Rotational coupling ratio of the submerged sphere.  m3150 shows the old
   two-significant-digit cells (2.2e-4 / 2.6e-5) are mis-rounded copies of
   2.151e-4 / 2.648e-5; table 12 quotes the reproducible pair.

Every value here is taken from the main branch, not invented; the script only
makes the two families agree.  Run from the repository root::

    python docs/m3751/reconcile_table_cells.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = ROOT / "src/tensorbem/benchmarks/declared_numbers.py"

EDITS: tuple[tuple[str, str], ...] = (
    # --- 1. B11 observed order: measured edge ratio, not log2 --------------
    (
        """            if previous is not None:
                # B11 is quoted in table 10 but not emitted by the benchmark.
                values[f"{tag}_observed_order_b11"] = math.log(
                    previous["surge_damping_rel_error"]
                    / level["surge_damping_rel_error"]
                ) / math.log(2.0)
""",
        """            if previous is not None:
                # B11 is quoted in table 10 but not emitted by the benchmark;
                # it follows from the consecutive-level error ratio divided by
                # the *measured* mean-edge ratio (m3150 口径: the module's other
                # observed orders use log(h0/h1), not the ideal log 2).
                error_ratio = math.log(
                    previous["surge_damping_rel_error"]
                    / level["surge_damping_rel_error"]
                )
                h0, h1 = float(previous["h_mean"]), float(level["h_mean"])
                values[f"{tag}_observed_order_b11"] = error_ratio / (
                    math.log(h0 / h1) if h1 < h0 else math.log(2.0)
                )
""",
    ),
    (
        '    (2, 64, "observed_order_b11", 1.62),',
        '    (2, 64, "observed_order_b11", 1.80),',
    ),
    (
        '    (3, 256, "observed_order_b11", 1.88),',
        '    (3, 256, "observed_order_b11", 1.94),',
    ),
    # --- 2. Submerged-sphere rotational coupling ratio --------------------
    (
        '    (1, 80, "rotational_added_mass_max_ratio", 2.2e-4),',
        '    (1, 80, "rotational_added_mass_max_ratio", 2.151e-4),',
    ),
    (
        '    (2, 320, "rotational_added_mass_max_ratio", 2.6e-5),',
        '    (2, 320, "rotational_added_mass_max_ratio", 2.648e-5),',
    ),
    (
        """#: ``rotational_added_mass_max_ratio`` is quoted at two significant digits only
#: (2.5e-32 / 2.2e-4 / 2.6e-5), i.e. with a rounding half-width of ~1.9 %; the
#: 20-panel cell is a machine-zero claim and is therefore registered as an upper
#: bound, while the two finite cells get a tolerance equal to that band.""",
        """#: ``rotational_added_mass_max_ratio`` is quoted at three significant digits
#: (2.5e-32 / 2.151e-4 / 2.648e-5, m3150 re-rounding of the old 2.2e-4 / 2.6e-5),
#: i.e. with a rounding half-width of ~1.9 %; the 20-panel cell is a machine-zero
#: claim and is therefore registered as an upper bound, while the two finite
#: cells get a tolerance equal to that band.""",
    ),
)


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    for old, new in EDITS:
        if text.count(old) != 1:
            print(f"FAILED: expected exactly one occurrence of:\n{old}", file=sys.stderr)
            return 1
        text = text.replace(old, new)
    TARGET.write_text(text, encoding="utf-8")
    print(f"reconciled {len(EDITS)} cells/conventions in {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())