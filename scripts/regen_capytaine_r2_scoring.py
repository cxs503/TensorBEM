"""Re-score the frozen round-2 cross-validation without re-solving.

The comparison, its scoring and the rendered document are pure functions of the
stored per-panel matrices (``tensorbem`` + the frozen capytaine reference), so a
criterion change can be re-applied to the frozen numbers instead of re-running
several hundred radiation solves.  Everything that *is* a solve -- the
degeneracy audit, the rotation-centre transfer identity, the deep-submergence
ladder -- is carried over untouched from the artifact.

Usage::

    PYTHONPATH=src python scripts/regen_capytaine_r2_scoring.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2  # noqa: E402

RESULT = ROOT / r2.ARTIFACT_ROOT / r2.RESULT_FILE


def main() -> int:
    study = json.loads(RESULT.read_text())
    reference = r2.load_reference()

    # recomputed: everything that depends on the invariant criterion
    study["comparison"] = r2.compare(study["tensorbem"], reference["cases"])
    study["invariant_scoring"] = r2.invariant_scoring(
        study["comparison"], study["tensorbem"]
    )
    study["ledger_acceptance"] = {
        "anchor": "XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE",
        "threshold": 0.15,
        "metric": "worst active relative deviation of the 6x6 radiation matrices",
        "worst_active_relative": study["comparison"]["worst_active_relative"],
        "satisfied": bool(study["comparison"]["worst_active_relative"] <= 0.15),
        "internal_bands": {
            "diagonal": r2.TOL_DIAGONAL,
            "offdiagonal": r2.TOL_OFFDIAGONAL,
            "note": (
                "self-imposed QoS bands, stricter than the 0.15 acceptance; the "
                "internal verdict may stay FAIL while the acceptance is met"
            ),
        },
    }

    paths = r2.write_outputs(study)
    print("wrote", paths["result"])
    print("wrote", paths["document"])
    for fig in paths.get("figures", []):
        print("wrote", fig)
    print(
        "internal verdict {verdict}  worst active {worst:.2%}  "
        "invariant {inv}  acceptance met: {acc}".format(
            verdict=study["comparison"]["verdict"],
            worst=study["comparison"]["worst_active_relative"],
            inv=study["invariant_scoring"]["verdict"],
            acc=study["ledger_acceptance"]["satisfied"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
