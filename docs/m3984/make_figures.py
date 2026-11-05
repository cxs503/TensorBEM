#!/usr/bin/env python3
"""Figures for the PR #109 merge-healing report.

Every plotted number is read from a file the pipeline already produced -- the
resolution's own recomputation dump (``docs/m2966/outputs/merge_checks/
merge_checks.json``, written by ``docs/m2966/merge_resolution_checks.py`` from
the shipped solver) and the two pytest logs of the baseline/merged runs.  No
number is transcribed by hand.

    python docs/m3984/make_figures.py \
        --checks docs/m2966/outputs/merge_checks/merge_checks.json \
        --base-log /tmp/m3984/base_main.log \
        --merged-log /tmp/m3984/merged_full.log
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("docs/m3984/figures")


def mfs_rows(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_mfs(data: dict, out: Path) -> dict:
    rows = data["mfs_rows"]["rows"]
    ref = data["mfs_rows"]["reference"]
    panels = np.array([r["panels"] for r in rows], dtype=float)
    docs = {
        "added mass": np.array([r["added_mass_rel_error"] for r in rows]),
        "damping": np.array([r["damping_rel_error"] for r in rows]),
    }

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for a, (name, err) in zip(ax, docs.items()):
        a.loglog(panels, err, "o-", label="MFS rows published in §4.6")
        # the O(N^-1) trend the panel method actually shows
        a.loglog(
            panels,
            err[0] * (panels[0] / panels) ** 1.0,
            "k--",
            lw=1.0,
            label=r"$O(N^{-1})$ reference",
        )
        a.axhline(0.03, color="crimson", lw=1.2, ls=":", label="3 % acceptance")
        for p, e in zip(panels, err):
            a.annotate(f"{e:.1e}", (p, e), textcoords="offset points", xytext=(4, -11),
                       fontsize=7)
        a.set_xlabel("MFS panels $N$ (24 sources)")
        a.set_ylabel(f"relative error vs Ursell closed form")
        a.set_title(f"{name} (reference {ref['added_mass' if name == 'added mass' else 'damping']:.2f})")
        a.set_xticks(panels)
        a.set_xticklabels([str(int(p)) for p in panels])
        a.grid(True, which="both", alpha=0.25)
        a.legend(fontsize=7, loc="lower left")
    fig.suptitle("PR #109 union: main's generated labels × the branch's third MFS row")
    fig.tight_layout()
    path = out / "mfs_rows_vs_reference.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "figure": str(path),
        "panels": panels.tolist(),
        "added_mass_rel_error": docs["added mass"].tolist(),
        "damping_rel_error": docs["damping"].tolist(),
        "worst_relative_error": float(max(docs["added mass"].max(), docs["damping"].max())),
        "all_under_3pct": bool(
            docs["added mass"].max() < 0.03 and docs["damping"].max() < 0.03
        ),
    }


def fig_referee(data: dict, out: Path) -> dict:
    pr = data["panel_referee"]
    pieces = np.array(pr["pieces"], dtype=float)
    err = np.array(pr["abs_errors"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.loglog(pieces, err, "o-", label="uniform-piece Gauss (20 pt) vs closed form")
    ax.loglog(pieces, err[2] * (pieces[2] / pieces), "k--", lw=1.0,
              label=r"$O(1/N)$ (observed order %.3f)" % pr["headline_observed_order"])
    ax.axhline(1e-6, color="darkorange", lw=1.1, ls=":",
               label=r"$10^{-6}$: 400 pieces still here")
    ax.axhline(1e-8, color="crimson", lw=1.1, ls=":", label=r"$10^{-8}$: unreachable")
    ax.set_xlabel("uniform pieces across the coplanar panel")
    ax.set_ylabel(r"$|I_{\rm Gauss}-I_{\rm closed}|$")
    ax.set_title("Why a uniform Gauss referee cannot certify the log-panel primitive")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "panel_referee_resolution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "figure": str(path),
        "observed_order": pr["headline_observed_order"],
        "error_at_400_pieces": float(err[list(pieces).index(400)]),
        "single_panel_error": pr["single_panel_error"],
        "closed_form": pr["closed_form"],
    }


_COUNT = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")


def pytest_counts(path: Path) -> dict:
    text = path.read_text(errors="replace") if path.exists() else ""
    counts = {k: int(v) for v, k in _COUNT.findall(text.splitlines()[-1] if text else "")} \
        if False else {}
    # the summary is the last non-empty line that mentions 'passed'/'failed'
    for line in reversed(text.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            for n, what in _COUNT.findall(line):
                counts[what.rstrip("s")] = int(n)
            break
    counts["log"] = str(path)
    counts["finished"] = bool(re.search(r"in [\d.]+s", text))
    return counts


def fig_tests(base: dict, merged: dict, subsets: dict, out: Path) -> dict:
    labels = ["main baseline", "merged (PR #109)"]
    passed = [base.get("passed", 0), merged.get("passed", 0)]
    failed = [base.get("failed", 0), merged.get("failed", 0)]
    skipped = [base.get("skipped", 0), merged.get("skipped", 0)]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8))
    x = np.arange(2)
    ax[0].bar(x - 0.22, passed, 0.22, label="passed", color="#2a7f62")
    ax[0].bar(x, failed, 0.22, label="failed", color="#b23a48")
    ax[0].bar(x + 0.22, skipped, 0.22, label="skipped", color="#8d8d8d")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("tests")
    ax[0].set_title("full suite: main vs merged tree")
    ax[0].legend(fontsize=8)
    for i, (p, f, s) in enumerate(zip(passed, failed, skipped)):
        ax[0].annotate(f"{p}p/{f}f/{s}s", (i, max(p, 1)), textcoords="offset points",
                       xytext=(0, 4), ha="center", fontsize=8)

    mods = list(subsets["main"].keys())
    b = [subsets["main"][m] for m in mods]
    g = [subsets["merged"][m] for m in mods]
    y = np.arange(len(mods))
    ax[1].barh(y - 0.2, b, 0.4, label="main baseline", color="#4d7ea8")
    ax[1].barh(y + 0.2, g, 0.4, label="merged", color="#2a7f62")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(mods, fontsize=7)
    ax[1].set_xlabel("tests run")
    ax[1].set_title("affected surface, per module")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "test_parity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"figure": str(path), "main": base, "merged": merged, "subsets": subsets}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="docs/m2966/outputs/merge_checks/merge_checks.json")
    ap.add_argument("--base-log", default="/tmp/m3984/base_main.log")
    ap.add_argument("--merged-log", default="/tmp/m3984/merged_full.log")
    ap.add_argument("--base-subset", default="/tmp/m3984/base_subset.log")
    ap.add_argument("--merged-subset", default="/tmp/m3984/merged_subset.log")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = mfs_rows(Path(args.checks))
    summary = {
        "mfs_rows": fig_mfs(data, out),
        "referee": fig_referee(data, out),
        "tests": fig_tests(
            pytest_counts(Path(args.base_log)),
            pytest_counts(Path(args.merged_log)),
            {
                "main": {"affected surface (4 modules)": pytest_counts(Path(args.base_subset)).get("passed", 0)},
                "merged": {"affected surface (4 modules)": pytest_counts(Path(args.merged_subset)).get("passed", 0)},
            },
            out,
        ),
    }
    (out.parent / "outputs").mkdir(parents=True, exist_ok=True)
    (out.parent / "outputs" / "figure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))