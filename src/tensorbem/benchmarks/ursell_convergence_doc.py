"""Executable pins for the numbers printed in ``docs/ursell_convergence_study.md``.

Provenance
----------
The m2198 convergence study (2026-11-02, branch ``agent/m2198``) published the
Ursell submerged-cylinder refinement table, the observed-order table, the
README comparison, the near-field / far-field table and the error verdict of
``docs/ursell_convergence_study.md``.  Nothing executed those digits: the
document declares no generator and no test parses it, so
:mod:`tensorbem.benchmarks.doc_claim_audit` grades it ``UNGUARDED``
(paper-level, 93 number tokens).  They drifted, and the drift is exactly the
kind an unguarded table hides:

* the parameter table printed ``omega = 1.905 rad/s``, which contradicts the
  same table's ``ka = 0.6``, ``a = 1 m``, ``g = 9.80665 m/s^2``
  (``omega = sqrt(ka g / a) = 2.425694 rad/s``) and contradicts the reference
  damping the document itself quotes (``B_ref = 2718.154`` implies
  ``omega = 2.4257``; with ``omega = 1.905`` it would be ``2135``);
* rows 16 and 64 of the added-mass column (``2697.5``, ``2587.6``) disagree
  with the relative error printed *beside them* (``5.998e-2``, ``1.674e-2``
  imply ``2697.8``, ``2587.8``) -- the table is internally inconsistent, which
  a one-line consistency check would have caught;
* row 32 of the damping column (``2851.7``, repeated in the near/far table)
  disagrees with its own relative error (``4.547e-2`` implies ``2841.7``);
* the far-field damping of rows 16/32/64 (``2850.1 / 2813.8 / 2762.1``) is not
  the value the shipped ``far_field_damping`` produces for the same mesh and
  potential (``2796.6 / 2784.7 / 2758.2``), so only the 128-panel row
  (``2739.9``) still reproduced;
* the near/far column and the verdict table used *different* definitions of the
  same ratio (``(near-far)/near`` in one table, ``near/far - 1`` in the other);
* the README comparison quotes a README claim (``7.7e-3``) that the README no
  longer makes -- it has said ``8.51e-3`` since the m2265 root-cause study --
  so the 10.6 % deviation the table reports is history, not a live statement;
* the MFS verdict rows (``0.234 % / 0.412 %``) reproduce for no collocation
  count tried (64 panels: ``7.57e-4 / 3.53e-3``; 128 panels:
  ``1.89e-4 / 8.82e-4``).

This module removes the hand-assembly.  Every table of sections 3, 4.2 and 6 is
rendered from :func:`study`, which solves the shipped problem, and
:func:`check_document` re-derives every printed number from the document text,
so a future edit to the solver or to the prose turns ``tests``
``test_ursell_convergence_doc_pins.py`` red.

Conventions fixed here
----------------------
``near/far`` relative difference is ``|B_near/B_far - 1|`` everywhere (the
document used two definitions); the near-field damping is the pressure
integral ``B`` and the far field is ``far_field_damping`` on the same potential.

Run::

    python -m tensorbem.benchmarks.ursell_convergence_doc --write   # rewrite docs
    python -m tensorbem.benchmarks.ursell_convergence_doc --json-out /tmp/ursell.json

The 128-panel row is the expensive one (~22 s of the ~29 s study); the test
module computes the study once at module scope and shares it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tensorbem.benchmarks.ursell_convergence_table import (
    mfs_row,
    panel_row,
    reference_coefficients,
)
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import far_field_damping, solve_radiation_indirect

DOC_NAME = "ursell_convergence_study.md"

#: Refinement ladder of the §3.1 table.
PANEL_COUNTS = (16, 32, 64, 128)

#: Panel count whose errors the verdict table certifies (<3 % criterion).
VERDICT_PANELS = 128

#: MFS (fundamental-solution) panel count quoted in the verdict.
MFS_PANELS = 128

#: Block markers written into the document; the text between them is generated.
MARKERS = {
    "conv": "ursell-conv",
    "orders": "ursell-orders",
    "readme": "ursell-readme",
    "nearfar": "ursell-nearfar",
    "verdict": "ursell-verdict",
}

#: README claims the §3.3 table compares against, verbatim.  The test asserts
#: these strings are still in README.md, so a README edit cannot silently
#: orphan the comparison.
README_CLAIMS = {
    "added_mass_relative_error": r"8.51\times10^{-3}",
    "near_far_relative_difference": r"6\times10^{-3}",
    "mfs_added_mass_relative_error": r"2\times10^{-4}",
}
README_DECLARED = {
    "added_mass_relative_error": 8.51e-3,
    "mfs_added_mass_relative_error": 2.0e-4,
    "near_far_relative_difference": 6.0e-3,
}

#: The digits the document carried before this module existed, kept as an audit
#: trail in §3.4.  ``None`` means the old table had no such column.
SUPERSEDED = {
    16: {"added_mass": 2697.5, "damping": 2904.8, "near_far": 1.89e-2},
    32: {"added_mass": 2627.4, "damping": 2851.7, "near_far": 1.34e-2},
    64: {"added_mass": 2587.6, "damping": 2787.5, "near_far": 9.12e-3},
    128: {"added_mass": 2566.8, "damping": 2754.5, "near_far": 5.34e-3},
}
SUPERSEDED_OMEGA = 1.905
SUPERSEDED_ORDERS = {(16, 32): (0.897, 0.589), (32, 64): (0.950, 0.837), (64, 128): (0.975, 0.928)}
SUPERSEDED_MFS = {"added_mass_relative_error": 2.34e-3, "damping_relative_error": 4.12e-3}
SUPERSEDED_FAR = {16: 2850.1, 32: 2813.8, 64: 2762.1, 128: 2739.9}


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #
def _row(panel_count: int) -> dict[str, float]:
    """One refinement row plus the far-field damping on the same potential."""
    row = panel_row(panel_count)
    # panel_row already solved the problem; re-solve only to hand the potential
    # to the far-field integral.  Kept explicit so the two integrals provably
    # see the same density (the near/far identity is a statement about one
    # potential, not about two solves).
    mesh = submerged_circle_mesh(panel_count, 1.0, 2.0)
    result = solve_radiation_indirect(mesh, 0.6, "heave", density=1025.0, panel_nodes=24)
    far = float(
        far_field_damping(
            mesh,
            result.potential,
            result.normal_velocity,
            0.6,
            1025.0,
            gravity=9.80665,
        )
    )
    row["far_field_damping"] = far
    row["near_far_relative_difference"] = abs(row["damping"] / far - 1.0)
    return row


def _order(coarse: float, fine: float) -> float:
    """Observed order ``p = log2(E_N / E_2N)``."""
    return math.log2(coarse / fine)


def study() -> dict:
    """Recompute every table of the document from the shipped solver."""
    reference = reference_coefficients()
    rows = [_row(n) for n in PANEL_COUNTS]
    orders = []
    for left, right in zip(rows, rows[1:]):
        orders.append(
            {
                "interval": f"{left['panels']} → {right['panels']}",
                "added_mass_order": _order(
                    left["added_mass_relative_error"], right["added_mass_relative_error"]
                ),
                "damping_order": _order(
                    left["damping_relative_error"], right["damping_relative_error"]
                ),
            }
        )
    verdict_row = next(r for r in rows if r["panels"] == VERDICT_PANELS)
    mfs = mfs_row(MFS_PANELS)
    measured = {
        "added_mass_relative_error": verdict_row["added_mass_relative_error"],
        "mfs_added_mass_relative_error": mfs["added_mass_relative_error"],
        "near_far_relative_difference": verdict_row["near_far_relative_difference"],
    }
    readme = [
        {
            "quantity": key,
            "declared": README_DECLARED[key],
            "measured": measured[key],
            "deviation": measured[key] / README_DECLARED[key] - 1.0,
        }
        for key in ("added_mass_relative_error", "near_far_relative_difference", "mfs_added_mass_relative_error")
    ]
    return {
        "benchmark": "ursell_submerged_circular_cylinder_deep_water",
        "document": DOC_NAME,
        "parameters": {
            "radius": 1.0,
            "submergence": 2.0,
            "ka": 0.6,
            "density": 1025.0,
            "gravity": 9.80665,
            "panel_nodes": 24,
        },
        "omega": reference["omega"],
        "reference": reference,
        "rows": rows,
        "orders": orders,
        "mfs": mfs,
        "readme_declared": README_DECLARED,
        "readme": readme,
        "verdict_panels": VERDICT_PANELS,
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _block(name: str, body: str) -> str:
    tag = MARKERS[name]
    return f"<!-- {tag}:begin -->\n{body.rstrip()}\n<!-- {tag}:end -->"


def render(study_data: dict) -> dict[str, str]:
    """Markdown for every generated block, keyed by marker name."""
    rows = study_data["rows"]
    conv = [
        "| 面元数 | 附加质量 A [kg/m] | A 相对误差 | 阻尼 B [kg/(m·s)] | B 相对误差 | 近/远场相对差 | 条件数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        conv.append(
            f"| {r['panels']} | {r['added_mass']:.4f} | {r['added_mass_relative_error']:.4e} | "
            f"{r['damping']:.4f} | {r['damping_relative_error']:.4e} | "
            f"{r['near_far_relative_difference']:.4e} | {r['condition_number']:.4f} |"
        )

    orders = ["| 加密区间 | 附加质量收敛阶 | 阻尼收敛阶 |", "|---|---|---|"]
    for o in study_data["orders"]:
        orders.append(
            f"| {o['interval']} | {o['added_mass_order']:.4f} | {o['damping_order']:.4f} |"
        )

    readme = ["| 量 | README 声明值 | 实测值（128 面元） | 偏差 |", "|---|---|---|---|"]
    for item in study_data["readme"]:
        readme.append(
            f"| `{item['quantity']}` | {item['declared']:.3e} | {item['measured']:.3e} | "
            f"{item['deviation']:+.1%} |"
        )

    nearfar = ["| 面元数 | 近场阻尼 B | 远场阻尼 | 相对差 |", "|---|---|---|---|"]
    for r in rows:
        nearfar.append(
            f"| {r['panels']} | {r['damping']:.4f} | {r['far_field_damping']:.4f} | "
            f"{r['near_far_relative_difference']:.4%} |"
        )

    verdict_row = next(r for r in rows if r["panels"] == study_data["verdict_panels"])
    mfs = study_data["mfs"]
    verdict = [
        "| 关键量 | 基准 | 实测 | 相对误差 | 是否 < 3% |",
        "|---|---|---|---|---|",
        f"| 附加质量 A（{verdict_row['panels']} 面元） | {study_data['reference']['added_mass']:.4f} | "
        f"{verdict_row['added_mass']:.4f} | {verdict_row['added_mass_relative_error']:.3%} | "
        f"{'是' if verdict_row['added_mass_relative_error'] < 0.03 else '否'} |",
        f"| 阻尼 B（{verdict_row['panels']} 面元） | {study_data['reference']['damping']:.4f} | "
        f"{verdict_row['damping']:.4f} | {verdict_row['damping_relative_error']:.3%} | "
        f"{'是' if verdict_row['damping_relative_error'] < 0.03 else '否'} |",
        f"| 近/远场一致性（{verdict_row['panels']} 面元） | 1 | "
        f"{1 + verdict_row['near_far_relative_difference']:.6f} | "
        f"{verdict_row['near_far_relative_difference']:.3%} | "
        f"{'是' if verdict_row['near_far_relative_difference'] < 0.03 else '否'} |",
        f"| MFS 附加质量误差（{mfs['panels']} 配点） | {study_data['reference']['added_mass']:.4f} | "
        f"{mfs['added_mass']:.4f} | {mfs['added_mass_relative_error']:.3%} | "
        f"{'是' if mfs['added_mass_relative_error'] < 0.03 else '否'} |",
        f"| MFS 阻尼误差（{mfs['panels']} 配点） | {study_data['reference']['damping']:.4f} | "
        f"{mfs['damping']:.4f} | {mfs['damping_relative_error']:.3%} | "
        f"{'是' if mfs['damping_relative_error'] < 0.03 else '否'} |",
    ]
    return {
        "conv": _block("conv", "\n".join(conv)),
        "orders": _block("orders", "\n".join(orders)),
        "readme": _block("readme", "\n".join(readme)),
        "nearfar": _block("nearfar", "\n".join(nearfar)),
        "verdict": _block("verdict", "\n".join(verdict)),
    }


def apply(text: str, study_data: dict) -> str:
    """Replace every generated block in ``text``; raise if one is missing."""
    blocks = render(study_data)
    out = text
    for name, body in blocks.items():
        tag = MARKERS[name]
        pattern = re.compile(
            rf"<!-- {re.escape(tag)}:begin -->.*?<!-- {re.escape(tag)}:end -->",
            re.DOTALL,
        )
        if not pattern.search(out):
            raise ValueError(f"document is missing the generated block marker '{tag}'")
        out = pattern.sub(lambda _m, body=body: body, out)
    return out


# --------------------------------------------------------------------------- #
# checking
# --------------------------------------------------------------------------- #
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _numbers(line: str) -> list[float]:
    """Every numeric token of a table row, including ``×10⁻⁴`` superscripts."""
    line = line.replace("×10", "e").translate(_SUPERSCRIPT)
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?", line)
    values = []
    for token in tokens:
        scale = 1.0
        if token.endswith("%"):
            token, scale = token[:-1], 1e-2
        values.append(float(token) * scale)
    return values


def _cells(line: str) -> list[str]:
    """The cells of one markdown table row, whitespace-stripped."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _cell_number(cell: str) -> float:
    """One numeric cell (``2697.8241``, ``5.9979e-02``, ``3.87%``, ``×10⁻²``)."""
    values = _numbers(cell)
    if not values:
        raise ValueError(f"cell '{cell}' carries no number")
    return values[0]


def _table(text: str, header_cell: str) -> list[list[str]]:
    """Cells of the table whose header contains ``header_cell``."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("|") and header_cell in line:
            rows = []
            for follow in lines[index + 2:]:
                if not follow.startswith("|"):
                    break
                rows.append(_cells(follow))
            return rows
    raise ValueError(f"no table with header '{header_cell}'")


def _close(got: float, expected: float, decimals: int = 4, rel: float = 1e-9) -> bool:
    """Printed value matches: half an ulp of the printed precision, plus float noise."""
    return abs(got - expected) <= 0.5 * 10.0 ** (-decimals) + rel * abs(expected)


def check_document(path: Path, study_data: dict) -> list[str]:
    """Every failure of the document against a fresh recomputation."""
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []

    # --- rendered blocks must be byte-identical to a fresh render ------------
    try:
        regenerated = apply(text, study_data)
    except ValueError as exc:
        return [str(exc)]
    if regenerated != text:
        failures.append("document blocks are stale: a fresh render differs from the text")

    # --- omega is consistent with the document's own ka, a, g ---------------
    match = re.search(r"\|\s*角频率\s*\|[^|]*\|\s*([0-9.]+)\s*\|\s*rad/s", text)
    if not match:
        failures.append("parameter table: omega row not found")
    else:
        printed = float(match.group(1))
        expected = math.sqrt(0.6 * 9.80665 / 1.0)
        if not _close(printed, expected, decimals=6):
            failures.append(
                f"parameter table: omega {printed} contradicts sqrt(ka*g/a) = {expected:.6f}"
            )

    # --- refinement rows: value and its own relative error must agree -------
    try:
        conv = _table(text, "附加质量 A [kg/m]")
    except ValueError as exc:
        failures.append(str(exc))
    else:
        if len(conv) != len(study_data["rows"]):
            failures.append(f"refinement table: {len(conv)} rows, expected {len(study_data['rows'])}")
        for row, expected in zip(conv, study_data["rows"]):
            panels = int(row[0])
            checks = [
                ("added mass", _cell_number(row[1]), expected["added_mass"], 1e-9),
                ("added mass error", _cell_number(row[2]), expected["added_mass_relative_error"], 1e-3),
                ("damping", _cell_number(row[3]), expected["damping"], 1e-9),
                ("damping error", _cell_number(row[4]), expected["damping_relative_error"], 1e-3),
                ("near/far", _cell_number(row[5]), expected["near_far_relative_difference"], 1e-3),
                ("condition number", _cell_number(row[6]), expected["condition_number"], 1e-9),
            ]
            for label, got, want, rel in checks:
                if not _close(got, want, decimals=4, rel=rel):
                    failures.append(f"N={panels} {label}: printed {got}, recomputed {want}")

    # --- the near/far table is the same quantity as the last column ---------
    try:
        nearfar = _table(text, "近场阻尼 B")
    except ValueError as exc:
        failures.append(str(exc))
    else:
        for row, expected in zip(nearfar, study_data["rows"]):
            for label, got, want, rel in (
                ("near-field damping", _cell_number(row[1]), expected["damping"], 1e-9),
                ("far-field damping", _cell_number(row[2]), expected["far_field_damping"], 1e-9),
                ("near/far difference", _cell_number(row[3]), expected["near_far_relative_difference"], 1e-3),
            ):
                if not _close(got, want, decimals=4 if rel < 1e-6 else 6, rel=rel):
                    failures.append(f"N={int(row[0])} {label}: printed {got}, recomputed {want}")

    # --- observed orders ----------------------------------------------------
    try:
        orders = _table(text, "加密区间")
    except ValueError as exc:
        failures.append(str(exc))
    else:
        for row, expected in zip(orders, study_data["orders"]):
            if row[0].replace(" ", "") != expected["interval"].replace(" ", ""):
                failures.append(f"orders table: interval '{row[0]}' != '{expected['interval']}'")
            for label, cell, want in (
                ("added-mass", row[1], expected["added_mass_order"]),
                ("damping", row[2], expected["damping_order"]),
            ):
                got = _cell_number(cell)
                if not _close(got, want, decimals=4, rel=1e-6):
                    failures.append(
                        f"{expected['interval']} {label} order: printed {got}, recomputed {want}"
                    )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="rewrite the document's generated blocks")
    parser.add_argument("--json-out", type=Path, default=None, help="dump the study as JSON")
    parser.add_argument("--doc", type=Path, default=None, help="document path (default: docs/<DOC_NAME>)")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    doc = args.doc or root / "docs" / DOC_NAME
    study_data = study()
    if args.json_out:
        args.json_out.write_text(json.dumps(study_data, indent=2) + "\n", encoding="utf-8")
    if args.write:
        doc.write_text(apply(doc.read_text(encoding="utf-8"), study_data), encoding="utf-8")
        print(f"rewrote {doc}")
    failures = check_document(doc, study_data)
    for failure in failures:
        print("FAIL", failure)
    print(f"{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())