"""Generate the tables of ``docs/finite_depth_energy_identity/report.md``.

Every number in the document is produced here from
``outputs/finite_depth_energy_identity/convergence.json``; ``pytest`` compares
the generated blocks with the text of the report, so a hand-edited (or stale)
number fails the suite instead of surviving in prose.

    PYTHONPATH=src python3 docs/finite_depth_energy_identity/build_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "outputs" / "finite_depth_energy_identity" / "convergence.json"
REPORT = Path(__file__).resolve().parent / "report.md"

DEPTH_LABELS = {"shallow": "3.2", "medium": "5.0", "deep": "20.0"}


def _sci(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def load(path: Path = ARTIFACT) -> dict:
    return json.loads(path.read_text())


def refinement_table(payload: dict) -> str:
    lines = [
        "| 水深 \\(h/a\\) | 运动 | 面元 \\(N\\) | \\(k_0\\Delta s\\) | \\(A\\) | \\(B_{\\rm near}\\) | \\(B_{\\rm far}\\) | \\(B_{\\rm far,q}\\) | \\(B_{\\rm far}/B_{\\rm near}-1\\) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in DEPTH_LABELS:
        for row in sorted(
            (r for r in payload["rows"] if r["depth_label"] == label),
            key=lambda r: (r["motion"], r["panels"]),
        ):
            lines.append(
                "| {depth} | {motion} | {panels} | {k0ds:.4f} | {mass:.2f} | {near:.2f} | {far:.2f} | {farq:.2f} | {offset:+.4e} |".format(
                    depth=DEPTH_LABELS[label],
                    motion=row["motion"],
                    panels=row["panels"],
                    k0ds=row["k0_panel_length"],
                    mass=row["added_mass"],
                    near=row["damping_near"],
                    far=row["damping_far"],
                    farq=row["damping_far_quadrature"],
                    offset=row["energy_offset"],
                )
            )
    return "\n".join(lines)


def summary_table(payload: dict) -> str:
    lines = [
        "| 算例 | 观测阶 \\(p\\) | 外推 \\((B_{\\rm far}/B_{\\rm near}-1)_\\infty\\) | \\(B_{\\rm near}\\) 极限 | \\(B_{\\rm far}\\) 极限 | 两极限相对差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, summary in payload["summaries"].items():
        label, motion = key.split(":")
        lines.append(
            "| \\(h/a={d}\\), {m} | {order:.2f} | {offset:+.3e} | {near:.2f} | {far:.2f} | {rel:.2e} |".format(
                d=DEPTH_LABELS[label],
                m="垂荡" if motion == "heave" else "横荡",
                order=summary["median_order"],
                offset=summary["offset_limit"],
                near=summary["damping_near_limit"],
                far=summary["damping_far_limit"],
                rel=summary["near_vs_far_limit_rel"],
            )
        )
    return "\n".join(lines)


def mfs_table(payload: dict) -> str:
    lines = [
        "| MFS 源数 | \\(A\\) | \\(B\\) | 与等源数无关的相对变化 |",
        "|---:|---:|---:|---:|",
    ]
    rows = payload["mfs_source_sweep"]
    base = rows[0]
    for row in rows:
        lines.append(
            "| {n} | {mass:.2f} | {damping:.2f} | {change:+.2e} |".format(
                n=row["source_count"],
                mass=row["added_mass"],
                damping=row["damping"],
                change=row["damping"] / base["damping"] - 1.0,
            )
        )
    return "\n".join(lines)


def node_table(payload: dict) -> str:
    lines = [
        "| 每面元 Gauss 点 | \\(A\\) | \\(B_{\\rm near}\\) | \\(B_{\\rm far}/B_{\\rm near}-1\\) |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["node_sweep"]:
        lines.append(
            "| {n} | {mass:.4f} | {near:.4f} | {offset:+.4e} |".format(
                n=row["panel_nodes"],
                mass=row["added_mass"],
                near=row["damping_near"],
                offset=row["energy_offset"],
            )
        )
    return "\n".join(lines)


def deep_table(payload: dict) -> str:
    lines = [
        "| 运动 | 面元 \\(N\\) | \\(B_{\\rm near}\\) | \\(B_{\\rm far}\\) | \\(B_{\\rm far}/B_{\\rm near}-1\\) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["deep_water_control"],
                      key=lambda r: (r["motion"], r["panels"])):
        lines.append(
            "| {m} | {n} | {near:.2f} | {far:.2f} | {offset:+.4e} |".format(
                m=row["motion"], n=row["panels"], near=row["damping_near"],
                far=row["damping_far"], offset=row["energy_offset"],
            )
        )
    return "\n".join(lines)


BLOCKS = {
    "REFINEMENT": refinement_table,
    "SUMMARY": summary_table,
    "MFS": mfs_table,
    "NODES": node_table,
    "DEEP": deep_table,
}


def blocks(payload: dict) -> dict[str, str]:
    return {name: builder(payload) for name, builder in BLOCKS.items()}


def inject(report_text: str, table_blocks: dict[str, str]) -> str:
    """Replace every ``BEGIN/END`` marker pair by the generated block."""
    text = report_text
    for name, block in table_blocks.items():
        begin = f"<!-- BEGIN:{name} -->"
        end = f"<!-- END:{name} -->"
        if begin not in text or end not in text:
            raise SystemExit(f"marker pair for {name} missing from {REPORT}")
        head, rest = text.split(begin, 1)
        _stale, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{block}\n{end}{tail}"
    return text


def main(argv: list[str] | None = None) -> int:
    payload = load()
    table_blocks = blocks(payload)
    argv = sys.argv[1:] if argv is None else argv
    if "--inject" in argv:
        REPORT.write_text(inject(REPORT.read_text(), table_blocks))
        print(f"injected {len(table_blocks)} blocks into {REPORT}", file=sys.stderr)
        return 0
    for name, text in table_blocks.items():
        print(f"===== {name} =====")
        print(text)
    print(f"\n(report: {REPORT})", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())