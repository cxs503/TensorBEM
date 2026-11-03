"""Reproduce the doc-claim gate evidence for m2918 (TensorBEM).

Writes ``results.json`` and ``doc_claim_gate.png`` next to this file.
Run: PYTHONPATH=src python3 docs/m2918/outputs/doc_claim_gate/make_figure.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem import doc_claims  # noqa: E402

OUT = Path(__file__).resolve().parent
root = doc_claims.REPO_ROOT

# --------------------------------------------------------------------------
# 1. which artifact does the gate use on this tree?
# --------------------------------------------------------------------------
artifact = doc_claims.resolve_artifact()
live_exists = doc_claims.ARTIFACT.exists()
pinned_exists = doc_claims.PINNED_ARTIFACT.exists()
measured = json.loads(artifact.read_text())["values"]

# --------------------------------------------------------------------------
# 2. per-claim relative difference, stated (documents) vs measured (code)
# --------------------------------------------------------------------------
claimed = doc_claims.parse_ursell_table(doc_claims.MARINE_DOC.read_text())
claimed.update(doc_claims.parse_readme_claims(doc_claims.MARINE_DOC.read_text()))
claimed.update(doc_claims.parse_readme_claims(doc_claims.README.read_text()))

rows = []
for claim in doc_claims.CLAIMS:
    if claim.key not in claimed or claim.key not in measured:
        continue
    stated = float(claimed[claim.key])
    expected = float(measured[claim.key])
    rel = abs(stated - expected) / abs(expected) if expected else 0.0
    rows.append(
        {
            "key": claim.key,
            "where": claim.where,
            "stated": stated,
            "measured": expected,
            "rel_diff": rel,
            "rel_tol": claim.rel_tol,
            "pass": rel <= claim.rel_tol,
        }
    )

report = doc_claims.check_documents()
payload = {
    "repo_root": str(root),
    "repo_root_source": "TENSORBEM_REPO_ROOT" if os.environ.get(doc_claims.ROOT_ENV_VAR)
    else "discovered (module file -> parents, then cwd -> parents)",
    "artifact_used": str(artifact),
    "artifact_kind": "generated artifacts/" if live_exists and artifact == doc_claims.ARTIFACT
    else "tracked tests/data/ (fresh-clone fallback)",
    "pinned_artifact_exists": pinned_exists,
    "generated_artifact_exists": live_exists,
    "claims_checked": report.checked,
    "claims_failed": len(report.failures),
    "claims_missing": len(report.missing),
    "gate_ok": report.ok,
    "max_relative_difference": max((r["rel_diff"] for r in rows), default=0.0),
    "rows": rows,
}
(OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

# --------------------------------------------------------------------------
# 3. figure: per-claim margin + the two defects the gate caught
# --------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 6.2))
order = np.argsort([r["rel_diff"] for r in rows])
rows_sorted = [rows[i] for i in order]
y = np.arange(len(rows_sorted))
reldiff = np.array([max(r["rel_diff"], 1e-12) for r in rows_sorted])
tol = np.array([r["rel_tol"] for r in rows_sorted])

ax0.barh(y, reldiff, height=0.62, color="#3b7dd8", label="|stated - measured| / measured")
ax0.scatter(tol, y, marker="|", s=180, color="#c0392b", label="per-claim tolerance")
ax0.set_yticks(y)
ax0.set_yticklabels([r["key"].replace("panel_added_mass", "A").replace(
    "panel_damping", "B").replace("relative_error", "relerr").replace(
    "near_far_field_relative_difference", "near/far dev") for r in rows_sorted],
    fontsize=6.4)
ax0.set_xscale("log")
ax0.set_xlabel("relative difference (log scale)")
ax0.set_title(f"all {len(rows)} published numbers reproduce\ntolerance is derived from printed digits")
ax0.axvline(1e-2, color="0.6", ls=":", lw=1)
ax0.legend(fontsize=8, loc="lower right")
ax0.grid(axis="x", which="both", alpha=0.25)

# right: the two defects
labels = ["README headline\nadded-mass error\n@128 panels", "table near/far-field\ndamping deviation\n@128 panels"]
before = [7.7e-3, 5.3e-3]
after = [8.5130305e-3, 5.3413678e-3]
ypos = np.arange(2)
yp = np.linspace(-0.16, 0.16, 2)
for i in range(2):
    ax1.plot([before[i], after[i]], [ypos[i], ypos[i]], color="0.4", lw=2, zorder=1)
ax1.scatter(before, ypos, s=110, color="#e67e22", label="stated before fix", zorder=3)
ax1.scatter(after, ypos, s=110, color="#27ae60", label="measured by code", zorder=3)
for i, (b, a) in enumerate(zip(before, after)):
    ax1.annotate(f"{b:.3g}\n({abs(b - a) / a * 100:.2f}% off)", (b, ypos[i]),
                 textcoords="offset points", xytext=(-6, 16), ha="right", fontsize=8)
    ax1.annotate(f"{a:.4g}", (a, ypos[i]), textcoords="offset points",
                 xytext=(8, -12), fontsize=8)
ax1.set_yticks(ypos)
ax1.set_yticklabels(labels, fontsize=8)
ax1.set_xlabel("value printed in the document / produced by the code")
ax1.set_title("the two numbers the gate flagged red on 2026-11-03\n(README was 9.55% off; ratio tolerance was mis-derived)")
ax1.legend(fontsize=8, loc="center left")
ax1.grid(axis="x", alpha=0.25)

fig.suptitle("TensorBEM doc-claim gate: portable checkout discovery + tracked fallback artifact (m2918)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "doc_claim_gate.png", dpi=95)
print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, ensure_ascii=False))
print("figure:", OUT / "doc_claim_gate.png")