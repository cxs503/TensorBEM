"""Emit the maturity-gate evidence bundle for the W20 item-5 delivery.

The maturity gate (owner 2026-11-11 达标判断方案) re-checks every claimed
level against machine-verifiable evidence keys:

    L2  impl_files + entry + run_artifacts
    L3  + integration_callers
    L4  + benchmark (ref vs got) + repro_cmd
    L5  + external_anchor

This tool derives ``benchmarks/verified/bem_kp505_openwater_w20/
evidence.json`` *from the artefacts themselves* (gci.json, the
transcription, the raw cache, git HEAD), so the evidence cannot drift
from the numbers it points at.  Run with ``python -m
tools.emit_w20_evidence``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = REPO_ROOT / "benchmarks" / "verified" / "bem_kp505_openwater_w20"
GCI_REL = "benchmarks/verified/bem_kp505_openwater_w20/gci.json"
TRANSCRIPTION_REL = "papers/data/w20-anchors/kp505_transcription.json"
CACHE_REL = "runs/w20_kp505_gci_cache"
SCHEMA = "tensorbem.maturity_evidence/1"

REPRO_CMD = (
    "python -m tools.build_w20_kp505_transcription && "
    "python -m tensorbem.benchmarks.kp505_w20_gci && "
    "python -m tools.emit_w20_report && "
    "python -m tools.emit_w20_evidence && "
    "pytest tests/test_w20_kp505_transcription.py tests/test_w20_kp505_gci.py "
    "tests/test_w20_kp505_integration.py -q"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> int:
    gci = json.loads((REPO_ROOT / GCI_REL).read_text())
    transcription = json.loads((REPO_ROOT / TRANSCRIPTION_REL).read_text())
    verdict = gci["verdict"]
    design = verdict["design_point"]
    tie = gci["history_tie"]
    anchor = gci["anchor"]

    primary = {
        cell["J"]: cell for cell in gci["cells"] if cell["core_policy"] == "fixed"
    }
    benchmark = {
        "reference": {
            "source": anchor["source"],
            "kind": anchor["kind"],
            "citation": transcription["anchor"]["citation"],
            "doi": transcription["anchor"]["doi"],
            "values": anchor["values"],
        },
        "got": {
            str(j): {
                "KT_fine": primary[j]["quantities"]["KT"]["levels"][-1],
                "10KQ_fine": primary[j]["quantities"]["10KQ"]["levels"][-1],
                "delta_vs_anchor_pct": {
                    "KT": primary[j]["quantities"]["KT"]["delta_vs_anchor_pct"],
                    "10KQ": primary[j]["quantities"]["10KQ"]["delta_vs_anchor_pct"],
                },
                "gci_fine_pct": {
                    "KT": primary[j]["quantities"]["KT"]["gci_fine_pct"],
                    "10KQ": primary[j]["quantities"]["10KQ"]["gci_fine_pct"],
                },
            }
            for j in (0.5, 0.7, 0.9)
        },
        "design_point_J0.70": {
            "dKT_pct": design["dKT_pct"],
            "dKQ_pct": design["dKQ_pct"],
            "gci_KT_pct": design["gci_KT_pct"],
            "gci_10KQ_pct": design["gci_10KQ_pct"],
            "answer_under_3pct": design["answer_under_3pct"],
            "threshold_pct": 3.0,
        },
        "history_29pct_reproduction": {
            "J": tie["J"],
            "dKT_pct": tie["dKT_pct"],
            "stated_report_8822_dKT_pct": tie["stated_report_8822_dKT_pct"],
        },
    }

    cache_files = sorted(
        p.name for p in (REPO_ROOT / CACHE_REL).glob("cell_*.json")
    )
    # Each artefact is listed in both path dialects the gate resolves:
    # repo-relative (audit worktree root) and eco-prefixed (ECO root).
    rel_artifacts = [
        "benchmarks/verified/bem_kp505_openwater_w20/gci.json",
        "benchmarks/verified/bem_kp505_openwater_w20/levels.csv",
        "benchmarks/verified/bem_kp505_openwater_w20/gci_table.csv",
        "benchmarks/verified/bem_kp505_openwater_w20/figures/w20_kt_kq_eta_vs_anchor.png",
        "benchmarks/verified/bem_kp505_openwater_w20/figures/w20_order_and_gci.png",
        "benchmarks/verified/bem_kp505_openwater_w20/figures/w20_deviation_split.png",
        "papers/data/w20-anchors/kp505_transcription.json",
        "runs/w20_kp505_gci_cache",
    ]
    run_artifacts = rel_artifacts + ["TensorBEM/" + r for r in rel_artifacts]
    impl_rel = [
        "src/tensorbem/benchmarks/kp505_w20_gci.py",
        "src/tensorbem/propeller.py",
        "src/tensorbem/uq.py",
    ]
    impl_files = impl_rel + ["TensorBEM/" + r for r in impl_rel]
    entry_rel = "src/tensorbem/benchmarks/kp505_w20_gci.py"

    evidence = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
        "mission": 14569,
        "original_mission": 9641,
        "claimed_level": 4,
        "impl_files": impl_files,
        "entry": entry_rel,
        "entry_eco": "TensorBEM/" + entry_rel,
        "entry_command": "python -m tensorbem.benchmarks.kp505_w20_gci",
        "run_artifacts": run_artifacts,
        "integration_callers": [
            "tests/test_w20_kp505_integration.py::test_live_l1_evaluation_matches_the_artefact",
            "tests/test_w20_kp505_integration.py::test_driver_is_exported_and_wired_to_the_main_solver",
            "src/tensorbem/benchmarks/kp505_w20_gci.py:151 (evaluate -> propeller.open_water_point)",
        ],
        "benchmark": benchmark,
        "repro_cmd": REPRO_CMD,
        "external_anchor": {
            "file": TRANSCRIPTION_REL,
            "file_eco": "TensorBEM/" + TRANSCRIPTION_REL,
            "sha256": sha256_file(REPO_ROOT / TRANSCRIPTION_REL),
            "citation": transcription["anchor"]["citation"],
            "doi": transcription["anchor"]["doi"],
            "kind": transcription["anchor"]["kind"],
        },
        "tests": [
            "tests/test_w20_kp505_transcription.py",
            "tests/test_w20_kp505_gci.py",
            "tests/test_w20_kp505_integration.py",
        ],
        "artefact_sha256": {
            "gci.json": sha256_file(REPO_ROOT / GCI_REL),
            "kp505_transcription.json": sha256_file(REPO_ROOT / TRANSCRIPTION_REL),
        },
        "verdict": verdict,
    }
    out = OUTDIR / "evidence.json"
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
