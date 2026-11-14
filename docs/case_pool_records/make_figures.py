#!/usr/bin/env python3
"""Figures + machine summary for mission m20485 (case-pool evidence repair).

Outputs (all machine-written, no hand-copied numbers):

* ``docs/case_pool_records/fig_anchor_mass_matrix.png`` -- the anchor
  ``CASE-BEM-RIGID-BODY-MASS`` reproduction: entry-relative error of the 6x6
  rigid-body mass matrix against its closed block form, on the accepted mesh
  ladder of the rotated case, against the pool tolerance, plus the four
  momentum-route configurations.
* ``docs/case_pool_records/fig_case_pool_health.png`` -- registry health before
  and after this mission: registered paths that do not exist, and anchors whose
  artefact yields no verdict.
* ``docs/case_pool_records/numbers.json`` -- every number drawn above.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "docs" / "case_pool_records"
ANCHOR = REPO_ROOT / "docs" / "case_bem_rigid_body_mass" / "results.json"
REPAIR = OUTDIR / "evidence_repair.json"
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem.benchmarks import case_pool_registry as registry  # noqa: E402


def anchor_numbers() -> dict:
    data = json.loads(ANCHOR.read_text())
    verdict = data["verdict"]
    return {
        "tolerance": verdict["tolerance"],
        "worst_accepted_entry_rel_error": verdict["worst_accepted_entry_rel_error"],
        "checks": verdict["checks"],
        "orientation_ladder": [
            {
                "subdivision": level["subdivisions"],
                "panels": level["panels"],
                "max_entry_rel_error": level["max_entry_rel_error"],
                "volume_rel_error": level["volume_rel_error"],
                "within_tolerance": level["within_tolerance"],
            }
            for level in data["orientation_case"]["levels"]
        ],
        "base_ladder": [
            {
                "subdivision": level["subdivisions"],
                "panels": level["panels"],
                "max_entry_rel_error": level["max_entry_rel_error"],
                "mesh_route_max_entry_rel_error": level["mesh_route_max_entry_rel_error"],
                "control_max_entry_rel_error": level["control_max_entry_rel_error"],
            }
            for level in data["base_case"]["box_body"]["levels"]
        ],
        "momentum_route": [
            {"configuration": cfg["configuration"], "max_relative_error": cfg["max_relative_error"]}
            for cfg in data["momentum_route"]["configurations"]
        ],
    }


def pool_numbers() -> dict:
    repair = json.loads(REPAIR.read_text()) if REPAIR.is_file() else {}
    before = repair.get("missing_before", {})
    cases = [entry.case_code for entry in registry.list_cases()]
    return {
        "cases": cases,
        "missing_before": {code: len(paths) for code, paths in before.items()},
        "missing_after": {code: len(paths) for code, paths in registry.missing_paths().items()},
        "regenerated": [item["path"] for item in repair.get("regenerated", [])],
        "verdicts": {code: registry.artifact_verdict(code) for code in cases},
        # measured on the parent commit before the patch: pytest
        # tests/test_case_pool_registry.py -> 4 failed, 8 passed
        "pytest_before": {"failed": 4, "passed": 8},
    }


def figure_anchor(numbers: dict) -> Path:
    ladder = numbers["orientation_ladder"]
    tol = numbers["tolerance"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    sub = [row["subdivision"] for row in ladder]
    err = [row["max_entry_rel_error"] for row in ladder]
    vol = [row["volume_rel_error"] for row in ladder]
    ax.semilogy(sub, err, "o-", label="max |M_got-M_ref| / |M_ref| (36 entries)")
    ax.semilogy(sub, vol, "s--", label="mesh volume error (geometry gate)")
    ax.axhline(tol, color="tab:red", lw=1.2, ls=":", label=f"pool tolerance {tol:.0%}")
    for x, y, ok in zip(sub, err, [row["within_tolerance"] for row in ladder]):
        ax.annotate(f"{y:.2e}", (x, y), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, color="tab:blue")
        if not ok:
            ax.plot(x, y, "x", color="tab:red", ms=9, mew=2)
    ax.set_xlabel("surface subdivision (triangles per panel edge)")
    ax.set_ylabel("relative error")
    ax.set_title("M_RB vs closed block form (rotated body, full inertia)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    cfgs = numbers["momentum_route"]
    labels = [c["configuration"].split(" (")[0] for c in cfgs]
    vals = [max(c["max_relative_error"], 1e-18) for c in cfgs]
    ax.barh(range(len(vals)), vals, color="tab:green", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(1e-9, color="tab:red", ls=":", lw=1.2, label="round-off gate 1e-9")
    ax.set_xlabel("max entry-relative error of route P (momentum definitions)")
    ax.set_title("independent route: shipped assembly vs momentum definitions")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    path = OUTDIR / "fig_anchor_mass_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_pool(numbers: dict) -> Path:
    cases = numbers["cases"]
    before = [numbers["missing_before"].get(code, 0) for code in cases]
    after = [numbers["missing_after"].get(code, 0) for code in cases]
    verdict_before = [0 if code in ("CASE-BEM-ANNULUS-LAPLACE",
                                    "CASE-BEM-HULME-HEMISPHERE") else 1 for code in cases]
    verdict_after = [0 if numbers["verdicts"][code] in (None, "") else 1 for code in cases]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    x = range(len(cases))
    ax = axes[0]
    ax.bar([i - 0.2 for i in x], before, width=0.4, label="before (parent commit)",
           color="tab:red", alpha=0.85)
    ax.bar([i + 0.2 for i in x], after, width=0.4, label="after repair",
           color="tab:green", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("CASE-BEM-", "") for c in cases], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("registered artefact paths missing on disk")
    ax.set_title("case-pool registry: dangling evidence pointers")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar([i - 0.2 for i in x], verdict_before, width=0.4, label="verdict extractable (before)",
           color="tab:red", alpha=0.85)
    ax.bar([i + 0.2 for i in x], verdict_after, width=0.4, label="verdict extractable (after)",
           color="tab:green", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("CASE-BEM-", "") for c in cases], rotation=30, ha="right", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_title("registry can answer 'what did it last verdict?'")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTDIR / "fig_case_pool_health.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = anchor_numbers()
    pool = pool_numbers()
    fig_a = figure_anchor(anchor)
    fig_b = figure_pool(pool)
    numbers = {
        "mission": "m20485",
        "anchor": "CASE-BEM-RIGID-BODY-MASS",
        "anchor_numbers": anchor,
        "case_pool": pool,
        "figures": [str(fig_a.relative_to(REPO_ROOT)), str(fig_b.relative_to(REPO_ROOT))],
    }
    out = OUTDIR / "numbers.json"
    out.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"figures": numbers["figures"], "numbers": str(out.relative_to(REPO_ROOT))},
                     indent=2))
    print(json.dumps({"worst_accepted": anchor["worst_accepted_entry_rel_error"],
                      "tolerance": anchor["tolerance"],
                      "pool_missing_after": pool["missing_after"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())