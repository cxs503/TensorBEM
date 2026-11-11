"""Measure the artifact-namespace collision before/after the fix (m15365).

Two checkouts are compared on the same command::

    PYTHONPATH=src python3 -m pytest <files> -q -rf

* ``before`` = ``git archive HEAD`` of the branch point (pre-fix main),
* ``after``  = the working tree this branch is committed from.

Every number printed/plotted here is parsed from pytest's own summary line, so
nothing is hand-typed.  Outputs: a JSON table (``--out``) and a PNG figure.

    python scripts/m15365_namespace_split_measure.py \
        --before /tmp/head_snapshot --after . \
        --json benchmarks/verified/bem_verified_anchors/m15365_before_after.json \
        --png docs/figures/m15365_artifact_namespace_split.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

FILES = [
    "tests/test_bem_verified_anchors.py",
    "tests/test_thin_ship_anchor_artifacts.py",
    "tests/test_figure_integrity.py",
    "tests/test_bem_finite_depth_anchors.py",
    "tests/test_w20_wigley_anchor_artifacts.py",
]
SUMMARY = re.compile(r"(\d+) (passed|failed)")
FAILED = re.compile(r"^FAILED (\S+?)::", re.MULTILINE)


def run(root: pathlib.Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *FILES, "-q", "-rf"],
        cwd=root,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        capture_output=True,
        text=True,
    )
    text = proc.stdout + proc.stderr
    passed = failed = 0
    for count, kind in SUMMARY.findall(text):
        if kind == "passed":
            passed += int(count)
        else:
            failed += int(count)
    per_file = {name: 0 for name in FILES}
    for node in FAILED.findall(text):
        key = node if node.endswith(".py") else node.split("::")[0]
        if key in per_file:
            per_file[key] += 1
    return {"passed": passed, "failed": failed, "per_file_failed": per_file,
            "tail": text.strip().splitlines()[-1] if text.strip() else ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=pathlib.Path)
    parser.add_argument("--after", required=True, type=pathlib.Path)
    parser.add_argument("--json", required=True, type=pathlib.Path)
    parser.add_argument("--png", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    table = {"files": FILES,
             "before": run(args.before.resolve()),
             "after": run(args.after.resolve())}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [pathlib.Path(name).stem.replace("test_", "") for name in FILES]
    before = np.array([table["before"]["per_file_failed"][n] for n in FILES], float)
    after = np.array([table["after"]["per_file_failed"][n] for n in FILES], float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    ax.bar(x - 0.2, before, 0.4, label=f"before fix (red {int(before.sum())})", color="#b22222")
    ax.bar(x + 0.2, after, 0.4, label=f"after fix (red {int(after.sum())})", color="#2e7d32")
    ax.set_ylabel("failing nodes (pytest, parsed)")
    ax.set_title("TensorBEM artifact-gate reds per test file, same command before/after m15365")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.99, 0.95,
            f"passed nodes: {table['before']['passed']} -> {table['after']['passed']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=130)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())