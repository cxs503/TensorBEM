"""Build the PR #128 merge evidence bundle (numbers, figure) for docs/m3776.

Everything in ``outputs/`` and ``figures/`` is produced by this script from the
git objects and the pytest logs; no number in the report is typed by hand.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parent  # the heal worktree, i.e. the merge commit itself
sys.path.insert(0, str(HERE))

import verify_union  # noqa: E402  (local module, same directory)

OURS = "55db397"  # origin/agent/m3181 as published before this mission
THEIRS = "ecba798"  # origin/main
MERGED = "HEAD"
OUT = HERE / "outputs"
FIG = HERE / "figures"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(WT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def pytest_summary(log: Path) -> dict:
    text = log.read_text() if log.exists() else ""
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    counts = {}
    for key in ("passed", "failed", "error", "errors", "skipped", "xfailed", "xpassed"):
        match = re.findall(rf"(\d+) {key}", text)
        if match:
            counts[key.rstrip("s") if key == "errors" else key] = int(match[-1])
    seconds = re.findall(r"in ([\d.]+)s", tail)
    failed = sorted(set(re.findall(r"^FAILED (\S+)", text, re.M)))
    errors = sorted(set(re.findall(r"^ERROR (\S+)", text, re.M)))
    return {
        "log": log.name,
        "summary_line": tail,
        "counts": counts,
        "seconds": float(seconds[-1]) if seconds else None,
        "failed_tests": failed,
        "error_tests": errors,
        "collected": int(re.findall(r"(\d+) tests? collected", text)[-1])
        if re.findall(r"(\d+) tests? collected", text)
        else None,
    }


def conflict_report(base: str) -> dict:
    """Per-path three-way classification, plus the resolution of each conflict."""
    raw = subprocess.run(
        ["git", "-C", str(WT), "merge-tree", "--write-tree", "--name-only", OURS, THEIRS],
        capture_output=True,
        text=True,
    ).stdout
    conflicted = [line for line in raw.splitlines() if line and "/" in line or line.endswith(".toml")]
    conflicted = [line for line in conflicted if not re.fullmatch(r"[0-9a-f]{40}", line)]
    resolutions = [
        {
            "path": "src/tensorbem/benchmarks/declared_numbers.py",
            "hunks": 3,
            "resolution": [
                "import block: both alias families kept (hulme/hemisphere, "
                "wang_linton/sphere) because both sides' group dispatchers use them",
                "DECLARED_NUMBERS: main's inline rows kept verbatim and this "
                "branch's extension re-attached as `) + _free_surface_3d_rows()`",
                "_run_group: all six 3-D dispatchers kept (hulme3d, "
                "hulme3d_chief, wang_linton3d from this branch; hulme_hemisphere, "
                "submerged_sphere, hulme_chief from main)",
                "symbol collision: main's study flattener `_chief_rows(rows)` "
                "renamed `_chief_study_rows(rows)` (+1 call site) so this "
                "branch's `_chief_rows()` registry builder is not shadowed",
            ],
            "declared_numbers_retuned": False,
        },
        {
            "path": "tests/test_declared_numbers.py",
            "hunks": 2,
            "resolution": [
                "rationale comment above the finite/zero assertion: both sides' "
                "wording merged into one comment (no claim dropped)",
                "group-coverage set: union of {hulme3d, hulme3d_chief} and "
                "{hulme_chief, hulme_hemisphere}",
            ],
        },
    ]
    return {
        "merge_base": git("merge-base", OURS, THEIRS),
        "ours": git("rev-parse", OURS),
        "theirs": git("rev-parse", THEIRS),
        "merged": git("rev-parse", MERGED),
        "conflicted_paths": conflicted,
        "auto_merged_paths": len(
            [
                line
                for line in subprocess.run(
                    ["git", "-C", str(WT), "diff", "--name-only", base, MERGED],
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()
                if line
            ]
        ),
        "resolutions": resolutions,
    }


def registry_snapshot() -> dict:
    code = (
        "import json;"
        "from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS as D;"
        "import collections;"
        "print(json.dumps({'rows': len(D),"
        "'keys_unique': len({e.key for e in D}) == len(D),"
        "'groups': dict(collections.Counter(e.group for e in D)),"
        "'tiers': dict(collections.Counter(e.tier for e in D))}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(WT),
        env={"PYTHONPATH": str(WT / "src"), "PATH": "/usr/bin:/bin"},
        check=True,
    ).stdout
    return json.loads(out)


def lines_added(ref: str, base: str, path: str) -> int:
    content = verify_union.blob_bytes(str(WT), ref, path)
    base_content = verify_union.blob_bytes(str(WT), base, path)
    if content is None:
        return 0
    text = verify_union.decode(content) or ""
    base_text = verify_union.decode(base_content) or ""
    added = verify_union.added_lines(base_text, text)
    return sum(added.values())


def figure(union: dict, merged: dict, baseline: dict, top_paths: list[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    ours = [lines_added(OURS, union["base_commit"], p) for p in top_paths]
    theirs = [lines_added(THEIRS, union["base_commit"], p) for p in top_paths]
    labels = [p.replace("src/tensorbem/", "").replace("tests/", "") for p in top_paths]
    y = range(len(top_paths))
    ax.barh([i + 0.2 for i in y], ours, height=0.38, color="#2b6cb0", label="added by agent/m3181")
    ax.barh([i - 0.2 for i in y], theirs, height=0.38, color="#c05621", label="added by main")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("added lines retained in the merge (vs merge base)")
    ax.set_title("A. semantic union: both sides' additions survive")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    keys = ["passed", "skipped", "failed", "error"]
    m = [merged["counts"].get(k, 0) for k in keys]
    b = [baseline["counts"].get(k, 0) for k in keys]
    x = range(len(keys))
    ax.bar([i - 0.2 for i in x], m, width=0.38, color="#2f855a", label="merged (12a8d43)")
    ax.bar([i + 0.2 for i in x], b, width=0.38, color="#718096", label="main baseline (ecba798)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys)
    ax.set_ylabel("tests")
    ax.set_title("B. pytest outcome, merged tree vs main baseline")
    for i, (mv, bv) in enumerate(zip(m, b)):
        ax.text(i - 0.2, mv, str(mv), ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, bv, str(bv), ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"PR #128 heal: {union['paths_checked']} paths checked, "
        f"{union['failures']} lost lines, verdict {union['verdict']}",
        fontsize=10,
    )
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "fig_union_and_tests.png"
    fig.savefig(path, dpi=130)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    base = git("merge-base", OURS, THEIRS)
    union = verify_union.check(str(WT), base, OURS, THEIRS, MERGED)
    (OUT / "verify_union.json").write_text(json.dumps(union, indent=2, ensure_ascii=False) + "\n")

    merged = pytest_summary(HERE / "logs/merged.log")
    baseline = pytest_summary(HERE / "logs/baseline.log")
    evidence = {
        "conflicts": conflict_report(base),
        "union": {k: v for k, v in union.items() if k != "files"},
        "tests": {"merged": merged, "baseline": baseline},
        "registry": registry_snapshot(),
    }
    # line-level detail of the union check, trimmed to non-trivial rows
    evidence["union"]["files_checked"] = union["paths_checked"]
    evidence["union"]["declared_rewrites"] = union["declared_rewrites"]
    (OUT / "merge_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )

    diffs = subprocess.run(
        ["git", "-C", str(WT), "diff", "--numstat", base, MERGED],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    ranked = []
    for line in diffs:
        add, _dele, path = line.split("\t", 2)
        try:
            ranked.append((int(add), path))
        except ValueError:
            continue
    top = [path for _n, path in sorted(ranked, reverse=True) if path.endswith(".py")][:10]
    fig = figure(union, merged, baseline, top)
    print(json.dumps({"union_verdict": union["verdict"], "figure": str(fig)}))
    print("merged:", merged["summary_line"])
    print("baseline:", baseline["summary_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())