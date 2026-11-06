"""Evidence generator for m4862: doc-claim audit bookkeeping gate.

Every number quoted in ``dev_report_m4862_doc_claim_audit_gate.md`` is produced
here, so nothing is hand-copied.  The script compares the *live* audit against
the artifacts committed on ``origin/main`` (pre-fix state) and against the
artifacts committed on this branch (post-fix state).

Usage::

    PYTHONPATH=src python tools/m4862_audit_gate_evidence.py \
        --json artifacts/m4862_audit_gate_evidence.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tensorbem.benchmarks import doc_claim_audit as audit

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_show(ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def make_figure(evidence: dict, path: Path) -> Path:
    """Two panels: published vs live tier counts, and the gate before/after."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = ("PINNED", "ARTIFACT", "UNGUARDED")
    colors = {"PINNED": "#27ae60", "ARTIFACT": "#e67e22", "UNGUARDED": "#c0392b"}
    before_counts = evidence["pre_fix"]["committed_counts"]
    after_counts = evidence["live"]["counts"]
    before = [before_counts[t.lower()] for t in tiers]
    after = [after_counts[t.lower()] for t in tiers]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    idx = range(len(tiers))
    ax.bar([i - 0.2 for i in idx], before, width=0.38, label="published artifact",
           color="#95a5a6")
    ax.bar([i + 0.2 for i in idx], after, width=0.38, label="live audit",
           color=[colors[t] for t in tiers])
    ax.set_xticks(list(idx))
    ax.set_xticklabels([f"{t}\n({b}->{a})" for t, b, a in zip(tiers, before, after)],
                       fontsize=8)
    ax.set_ylabel("documents")
    ax.set_title(
        "tier counts: committed audit.json vs live tree\n"
        f"documents {before_counts['documents']} -> {after_counts['documents']}, "
        f"number tokens {before_counts['number_tokens']} -> "
        f"{after_counts['number_tokens']}",
        fontsize=8,
    )
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    labels = ["artifact\ndrift", "ratchet\nfailures"]
    pre = [
        len(evidence["pre_fix"]["artifact_drift_failures"]),
        len(evidence["pre_fix"]["ratchet_failures"]),
    ]
    post = [
        len(evidence["post_fix"]["artifact_drift_failures"]),
        len(evidence["post_fix"]["ratchet_failures"]),
    ]
    ax.bar([i - 0.2 for i in range(2)], pre, width=0.38, color="#c0392b",
           label="before (origin/main artifacts)")
    ax.bar([i + 0.2 for i in range(2)], post, width=0.38, color="#27ae60",
           label="after (this branch)")
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("failures")
    ax.set_title("gate output before / after the fix", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    for i, (a, b) in enumerate(zip(pre, post)):
        ax.text(i - 0.2, a + 0.3, str(a), ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.3, str(b), ha="center", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument(
        "--figure", type=Path, default=REPO_ROOT / "artifacts" / "m4862_audit_gate.png"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "artifacts" / "m4862_audit_gate_evidence.json",
    )
    args = parser.parse_args(argv)

    report = audit.audit()
    live_tiers = {r["doc"]: (r["tier"], r["number_tokens"]) for r in report["records"]}

    # ---- pre-fix: artifacts and baseline as committed on the base revision ----
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        old_json = tmpdir / "audit.json"
        old_json.write_text(git_show(args.ref, "docs/doc_claim_audit/audit.json"), "utf-8")
        old_md = tmpdir / "doc_claim_audit.md"
        old_md.write_text(git_show(args.ref, "docs/doc_claim_audit.md"), "utf-8")
        pre_drift = audit.artifact_drift(report, old_json, old_md)
        old_baseline = json.loads(
            git_show(args.ref, "docs/doc_claim_audit/baseline.json")
        )
        old_committed = json.loads(old_json.read_text(encoding="utf-8"))

    pre_check = audit.check(report, old_baseline)

    # ---- post-fix: artifacts and baseline committed on this branch ----
    post_drift = audit.artifact_drift(
        report, audit.JSON_PATH, audit.MARKDOWN_PATH
    )
    baseline = json.loads(audit.JSON_PATH.parent.joinpath("baseline.json").read_text("utf-8"))
    post_check = audit.check(report, baseline)

    old_records = {r["doc"]: r for r in old_committed.get("records", [])}
    new_docs = sorted(set(live_tiers) - set(old_records))
    retired_docs = sorted(set(old_records) - set(live_tiers))
    upgraded = sorted(
        d
        for d in set(live_tiers) & set(old_records)
        if old_records[d]["tier"] != live_tiers[d][0]
    )

    evidence = {
        "mission": "m4862",
        "subject": "doc-claim audit bookkeeping: ratchet baseline + published artifacts",
        "base_ref": args.ref,
        "live": {
            "counts": report["counts"],
            "coverage": report["coverage"],
            "gated_by_cli": {
                "--check": "OK" if not post_check else "FAIL",
                "--check-artifacts": "OK" if not post_drift else "FAIL",
            },
        },
        "pre_fix": {
            "committed_counts": old_committed.get("counts"),
            "committed_coverage": old_committed.get("coverage"),
            "documents_missing_from_artifact": new_docs,
            "documents_retired_from_tree": retired_docs,
            "tier_changes": {
                d: {"was": old_records[d]["tier"], "now": live_tiers[d][0]}
                for d in upgraded
            },
            "artifact_drift_failures": pre_drift,
            "ratchet_failures": pre_check,
            "number_tokens_unreported": (
                report["counts"]["number_tokens"]
                - (old_committed.get("counts") or {}).get("number_tokens", 0)
            ),
        },
        "post_fix": {
            "artifact_drift_failures": post_drift,
            "ratchet_failures": post_check,
            "baseline_unguarded": len(baseline["unguarded"]),
            "baseline_unguarded_paper_level": len(baseline["unguarded_paper_level"]),
        },
    }

    fig_path = args.figure
    if fig_path is not None:
        make_figure(evidence, fig_path)
        evidence["figure"] = str(fig_path.relative_to(REPO_ROOT))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({k: evidence[k] for k in ("live", "post_fix")}, indent=2))
    print(f"drift failures before fix: {len(pre_drift)}")
    print(f"ratchet failures before fix: {len(pre_check)}")
    print(f"evidence written: {args.json}")
    ok = not pre_drift or True  # pre-fix failures are the finding, not an error
    return 0 if ok and not post_drift and not post_check else 1


if __name__ == "__main__":
    raise SystemExit(main())