"""Re-measure the m4796 CI wall-clock evidence (zero hand-copied values).

``docs/dev_report_m4796_ci_red_fix.md`` makes three kinds of numeric claims
about the red ``pytest`` job on ``main@fa868b64`` (2026-11-06):

* how many recent runs were sampled and how they concluded,
* the wall-clock distribution of those runs (min / median / max / > bound),
* the wall clock of the target run, read off the Actions API.

All of them are derivable from the recorded evidence
``docs/m4796/ci_duration_evidence.json``, so none of them needs to be typed by
hand -- this module is the single place that turns the JSON into numbers, and
``tests/test_m4796_ci_evidence.py`` re-runs it against the report to keep the
prose from drifting away from the data.

Usage::

    python3 tools/m4796_ci_evidence.py            # summary as JSON
    python3 tools/m4796_ci_evidence.py --check-report
    python3 tools/m4796_ci_evidence.py --plot     # redraw the committed figure

A note on the target run, because two traces of it disagree slightly: the
*run* is 29.8 min (``created_at -> updated_at``) while the *check run* span is
29.6 min (``started_at -> completed_at``), because a run exists ~10 s before its
check run starts and the two traces round differently.  Both are reported; the
0.7 % gap is what the report's 29.8 min claim rests on.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "m4796" / "ci_duration_evidence.json"
FIGURE = ROOT / "docs" / "m4796" / "ci_wallclock.png"
REPORT = ROOT / "docs" / "dev_report_m4796_ci_red_fix.md"

#: ``timeout-minutes`` the workflow now carries; the report counts the runs that
#: would have been cut off by it.
BOUND_MIN = 90
TARGET_SHA = "fa868b64"

#: conclusion -> colour, shared by the figure and the tests that describe it.
COLOURS = {
    "success": "#2a9d3f",
    "failure": "#c62828",
    "cancelled": "#e8890c",
    "queued": "#8a8a8a",
}


def load(path: Path | str = EVIDENCE) -> dict:
    """Load the recorded evidence."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _minutes(start: str, end: str) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    delta = datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
    return round(delta.total_seconds() / 60.0, 1)


def summarise(evidence: dict) -> dict:
    """Every number the report claims, recomputed from the evidence."""
    runs = evidence["recent_runs"]
    walls = [float(run["wall_min"]) for run in runs]
    target = next(run for run in runs if run["sha"].startswith(TARGET_SHA))
    check_run = evidence["check_run"]
    return {
        "n": len(runs),
        "min_min": min(walls),
        "median_min": round(statistics.median(walls), 1),
        "max_min": max(walls),
        "over_bound": sum(1 for wall in walls if wall > BOUND_MIN),
        "bound_min": BOUND_MIN,
        "conclusions": dict(collections.Counter(run["conclusion"] for run in runs)),
        "target_sha": target["sha"],
        "target_wall_min": float(target["wall_min"]),
        "target_conclusion": target["conclusion"],
        "target_check_run_min": _minutes(check_run["started_at"], check_run["completed_at"]),
        "target_annotations": check_run.get("annotations") or [],
    }


def report_claims(text: str) -> dict[str, str]:
    """The ``| label | value |`` cells of the report, label -> value."""
    claims: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        claims.setdefault(cells[0], cells[1])
    return claims


def _first_number(text: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    assert match is not None, f"no number in {text!r}"
    return float(match.group(0))


def conclusion_text(conclusions: dict[str, int]) -> str:
    """``cancelled 10 / failure 11 / ...`` -- the report's fixed ordering."""
    order = ["cancelled", "failure", "success", "queued"]
    tail = sorted(k for k in conclusions if k not in order)
    return " / ".join(
        f"`{name}` {conclusions[name]}" for name in order + tail if name in conclusions
    )


def check_report(summary: dict, text: str) -> list[str]:
    """Compare the report's claims with ``summary``; return the mismatches."""
    claims = report_claims(text)
    problems: list[str] = []

    def expect(label: str, actual: float | str) -> None:
        if label not in claims:
            problems.append(f"report has no row {label!r}")
            return
        # The value cell carries prose after the number ("0.0（queued，未起跑）"),
        # so read the leading number instead of parsing the whole cell.
        claimed = _first_number(claims[label]) if isinstance(actual, float) else claims[label]
        if isinstance(actual, float):
            ok = abs(claimed - actual) <= 0.051
        else:
            ok = claimed == actual
        if not ok:
            problems.append(f"{label!r}: report says {claims[label]!r}, evidence says {actual!r}")

    expect("样本数 n", float(summary["n"]))
    expect("最短", float(summary["min_min"]))
    expect("中位", float(summary["median_min"]))
    expect("最长", float(summary["max_min"]))
    expect("超过新上界 90 min 的 run 数", float(summary["over_bound"]))

    target_label = next((k for k in claims if k.startswith(f"`{TARGET_SHA}`")), None)
    if target_label is None:
        problems.append(f"report has no row for the target run {TARGET_SHA}")
    else:
        expect(target_label, float(summary["target_wall_min"]))
        if summary["target_conclusion"] not in claims[target_label]:
            problems.append(
                f"{target_label!r}: report never mentions conclusion "
                f"{summary['target_conclusion']!r}"
            )

    if "结论分布" in claims:
        if claims["结论分布"] != conclusion_text(summary["conclusions"]):
            problems.append(
                "结论分布: report says %r, evidence says %r"
                % (claims["结论分布"], conclusion_text(summary["conclusions"]))
            )
    else:
        problems.append("report has no 结论分布 row")

    return problems


def plot(summary_evidence: dict, out: Path | str = FIGURE) -> Path:
    """Redraw ``ci_wallclock.png`` from the evidence (same data, same figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = summary_evidence["recent_runs"]
    labels = [run["sha"][:8] for run in runs]
    walls = [float(run["wall_min"]) for run in runs]
    colours = [COLOURS.get(run["conclusion"], "#555555") for run in runs]

    figure, axes = plt.subplots(figsize=(11.7, 5.46), dpi=100)
    axes.bar(range(len(runs)), walls, color=colours, width=0.72)
    axes.axhline(BOUND_MIN, linestyle="--", linewidth=1.6, color="#1565c0")
    axes.text(
        len(runs) - 0.5,
        BOUND_MIN + 3,
        f"new job bound: timeout-minutes: {BOUND_MIN}",
        ha="right",
        va="bottom",
        color="#1565c0",
        fontsize=9,
    )
    for index, run in enumerate(runs):
        if run["sha"].startswith(TARGET_SHA):
            axes.annotate(
                f"{TARGET_SHA} — {run['conclusion']}\n"
                f"{run['wall_min']:.1f} min, cancelled by owner",
                xy=(index, walls[index]),
                xytext=(index - 3.5, min(BOUND_MIN + 40, max(walls) * 0.94)),
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.0},
            )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOURS[name])
        for name in ("success", "failure", "cancelled", "queued")
    ]
    axes.legend(
        handles,
        ["success", "failure", "cancelled", "queued"],
        loc="upper left",
        fontsize=9,
        frameon=False,
    )
    axes.set_xticks(range(len(runs)))
    axes.set_xticklabels(labels, rotation=90, fontsize=7)
    axes.set_ylabel("wall clock (min)")
    axes.set_xlabel("run (SHA, oldest last as returned by the API)")
    axes.set_title(
        f"TensorBEM `tests` workflow — {len(runs)} most recent runs, "
        f"median {statistics.median(walls):.1f} min, max {max(walls):.1f} min"
    )
    axes.set_ylim(0, max(walls) * 1.12)
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    out = Path(out)
    figure.savefig(out)
    plt.close(figure)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-report", action="store_true", help="compare the dev report with the evidence")
    parser.add_argument("--plot", action="store_true", help=f"redraw {FIGURE.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    evidence = load()
    summary = summarise(evidence)
    if args.plot:
        print(f"wrote {plot(evidence)}")
    if args.check_report:
        problems = check_report(summary, REPORT.read_text(encoding="utf-8"))
        for problem in problems:
            print(f"MISMATCH {problem}")
        print("report matches the evidence" if not problems else f"{len(problems)} mismatch(es)")
        return 1 if problems else 0
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())