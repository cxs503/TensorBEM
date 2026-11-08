"""m6502 evidence pack: mutation controls, audit before/after, energy-Hessian figure.

Every number quoted in dev_report_m6502_*.md comes from the JSON this writes.
Run from the worktree root: ``PYTHONPATH=src python outputs/m6502/build_evidence.py``
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m6502")
OUT = ROOT / "outputs" / "m6502"
OUT.mkdir(parents=True, exist_ok=True)
ENV = {"PYTHONPATH": str(ROOT / "src"), "OPENBLAS_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"}

sys.path.insert(0, str(ROOT / "src"))
from tensorbem.benchmarks import doc_claim_audit as dca  # noqa: E402
from tensorbem.benchmarks import tank_free_surface_verification as tfsv  # noqa: E402


def pytest_summary(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "--no-header"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    failed = [ln.split("::")[-1].split(" ")[0] for ln in proc.stdout.splitlines() if ln.startswith("FAILED")]
    return {"exit": proc.returncode, "summary": tail, "failed_tests": failed}


evidence: dict = {"mission": "m6502", "repo": "TensorBEM"}

# ---- 1. doc-claim audit state (the ratchet) --------------------------------
report = dca.audit()
baseline = json.loads((ROOT / "docs" / "doc_claim_audit" / "baseline.json").read_text())
tank = next(r for r in report["records"] if r["doc"] == "tank_free_surface_correction.md")
evidence["doc_claim_audit"] = {
    "counts": report["counts"],
    "coverage": report["coverage"],
    "failures_now": dca.check(report, baseline),
    "tank_doc_tier": tank["tier"],
    "tank_doc_read_by": list(tank["reading_tests"]),
    "tank_doc_generators": list(tank["generators"]),
    "paper_level_unguarded_still": [
        r["doc"] for r in report["records"] if r["tier"] == "UNGUARDED" and r["paper_level"]
    ],
}

# ---- 2. the study numbers (energy Hessian) --------------------------------
study = tfsv.run(quick=False)
c8 = study["checks"]["C8_energy_hessian"]
evidence["energy_hessian"] = {
    "depth": c8["depth"],
    "fd_steps": c8["fd_steps"],
    "cases": c8["cases"],
    "attribution": [
        {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in a.items()}
        for a in c8["attribution"]
    ],
    "not_run": study["not_run"],
}

# ---- 3. mutation controls (are the new tests real gates?) -----------------
module_copy = OUT / "_tank_free_surface_orig.py"
bench_copy = OUT / "_verification_orig.py"
shutil.copy(ROOT / "src/tensorbem/tank_free_surface.py", module_copy)
shutil.copy(ROOT / "src/tensorbem/benchmarks/tank_free_surface_verification.py", bench_copy)
target_tests = ["tests/test_tank_free_surface.py", "tests/test_tank_free_surface_doc_pins.py"]
mutations = {}

mutations["clean"] = pytest_summary(target_tests)

# M1: flip the sign of dC[3,4] in the module under test
s = module_copy.read_text().replace(
    "        delta_c[3, 4] += w * tank.i_xy\n        delta_c[4, 3] += w * tank.i_xy\n",
    "        delta_c[3, 4] -= w * tank.i_xy\n        delta_c[4, 3] -= w * tank.i_xy\n",
)
(ROOT / "src/tensorbem/tank_free_surface.py").write_text(s)
mutations["M1_cross_sign_flipped"] = pytest_summary(target_tests)
shutil.copy(module_copy, ROOT / "src/tensorbem/tank_free_surface.py")

# M2: reinstate the historical origin-based free-surface plane in the energy helper
s = bench_copy.read_text().replace("    if centroidal:\n        u = y", "    if False:\n        u = y")
tgt = ROOT / "src/tensorbem/benchmarks/tank_free_surface_verification.py"
tgt.write_text(s)
mutations["M2_origin_plane_reintroduced"] = pytest_summary(target_tests)

# M3: drop the frozen-liquid baseline
s = bench_copy.read_text().replace('key = "delta" if reference else "free"', 'key = "free"')
tgt.write_text(s)
mutations["M3_no_frozen_reference"] = pytest_summary(target_tests)
shutil.copy(bench_copy, tgt)
mutations["restored_clean"] = pytest_summary(target_tests)
evidence["mutation_controls"] = mutations
for f in (module_copy, bench_copy):
    f.unlink()

# ---- 4. figure -------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
labels, rels = [], []
for case, block in c8["cases"].items():
    for row in block["rows"]:
        for qty, val in row["relerr"].items():
            labels.append(f"{case}\n{qty}\neps={row['eps']:.0e}")
            rels.append(val)
xpos = np.arange(len(rels))
ax.bar(xpos, rels, color="#2c7fb8")
ax.axhline(0.03, color="#c0392b", ls="--", lw=1.4, label="3 % acceptance line")
ax.set_yscale("log")
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=5.5, rotation=90)
ax.set_ylabel("relative error of energy Hessian vs analytic")
ax.set_title("liquid-energy Hessian vs analytic -rho_l g i_ij (grid 800)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")

ax = axes[1]
names = ["centroidal plane\n+ frozen reference", "origin plane\n(m6452 symptom)", "no frozen\nreference"]
ratios = [a["ratio"] for a in c8["attribution"]]
bars = ax.bar(range(3), ratios, color=["#2ecc71", "#e67e22", "#c0392b"])
ax.axhline(1.0, color="k", ls="--", lw=1.2, label="analytic (ratio 1)")
for b, r in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width() / 2, r * 1.1, f"{r:.4f}x", ha="center", fontsize=9)
ax.set_yscale("log")
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("d2(Delta U)/dphi2 / analytic (rectangle)")
ax.set_title("why the energy path once looked 4x wrong")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")
fig.tight_layout()
fig_path = OUT / "fig_energy_hessian.png"
fig.savefig(fig_path, dpi=150)
plt.close(fig)
evidence["figure"] = str(fig_path.relative_to(ROOT))

(OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, default=float) + "\n")
print(json.dumps({k: evidence[k] for k in ("doc_claim_audit", "mutation_controls")}, indent=2, default=float)[:2500])
