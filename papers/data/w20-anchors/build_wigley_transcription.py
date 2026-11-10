#!/usr/bin/env python3
"""W20 anchor-repair, seakeeping-truncation item 4: Wigley repo-benchmark transcription.

Why this script exists
----------------------
W18 exp#1 reported an honest FAIL on transcribing the *experimental* Wigley
RAO table (Wigley 1942 / ITTC seakeeping benchmark): the source was not
reachable and no table could be copied.  This task therefore re-routes the
anchor to the two benchmark records that *do* live in this repository,

    benchmarks/verified/bem_wigley_wave_resistance/results.json
    benchmarks/verified/bem_wigley_hydrostatics/results.json

and transcribes them **field by field** with RFC-6901 JSON pointers, so that no
number in any downstream comparison or report is hand-copied.

What "100 % complete" means here (machine-checkable, not a claim)
----------------------------------------------------------------
Every scalar leaf of both source documents is enumerated, given a pointer, read
back through :func:`resolve` and compared with the source leaf.  Completeness is
therefore ``transcribed_leaves / total_leaves`` with a round-trip equality check
per leaf -- 100 % cannot be faked by omitting hard fields.

Reference typing (the honest part)
----------------------------------
The ledger classifies every *reference* quantity of the two files:

  * ``analytic``   -- closed form evaluated to machine precision
                      (hydrostatics: V = 4LBT/9, KB = 5T/8, ...);
  * ``numerical``  -- a theory that is integrated numerically
                      (Michell thin-ship with converged composite
                      Gauss-Legendre quadrature, 480 subintervals x 20 nodes);
  * ``experiment`` -- model-test data.  **None is present in either file.**
                      Both files say so explicitly (see the pointers recorded in
                      ``reference_type_ledger.experiment_anchor``), therefore no
                      number of this repository may be used as an experimental
                      anchor.

Usage::

    python3 papers/data/w20-anchors/build_wigley_transcription.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "wigley_repo_benchmark_transcription.json"

SOURCES = {
    "wave_resistance": "benchmarks/verified/bem_wigley_wave_resistance/results.json",
    "hydrostatics": "benchmarks/verified/bem_wigley_hydrostatics/results.json",
}
REPORTS = {
    "wave_resistance": "benchmarks/verified/bem_wigley_wave_resistance/report.md",
    "hydrostatics": "benchmarks/verified/bem_wigley_hydrostatics/report.md",
}

#: SI unit per leaf name.  ``None`` = not declared by the source and not
#: inferable without guessing -- recorded as null instead of invented.
UNITS = {
    "L": "m", "B": "m", "T": "m", "KB": "m", "KG": "m", "speed": "m/s",
    "froude_number": "1", "L_over_B": "1", "B_over_T": "1",
    "rho": "kg/m^3", "g": "m/s^2", "density": "kg/m^3", "gravity": "m/s^2",
    "volume": "m^3", "displaced_volume": "m^3", "Cb": "1", "Cm": "1", "Cp": "1",
    "waterplane_area": "m^2", "midship_section_area": "m^2", "wetted_area": "m^2",
    "I_xx": "m^4", "I_yy": "m^4", "BM_T": "m", "BM_L": "m", "GM_T": "m", "GM_L": "m",
    "C33": "N/m", "C44": "N*m/rad", "C55": "N*m/rad",
    "Rw_reference": "N", "Rw_production": "N", "Rw": "N",
    "Cw_reference": "1", "Cw_production": "1",
    "center_of_buoyancy": "m", "LCB_x": "m", "wetted_area_reference": "m^2",
}

#: Field classes a usable anchor must expose.  Each is a list of pointer
#: prefixes; a class counts as present only if at least one leaf matches.
REQUIRED_CLASSES = {
    "geometry (L, B, T)": ["/geometry/L", "/geometry/B", "/geometry/T"],
    "hydrostatic reference (Cb, volume)": ["/reference/Cb", "/reference/volume"],
    "fluid (rho, g)": ["/fluid/rho", "/fluid/g"],
    "case definition (Froude numbers)": ["/cases/0/froude_number", "/cases/1/froude_number",
                                         "/cases/2/froude_number"],
    "quantities (Rw / Cw)": ["/cases/0/Rw_production", "/cases/0/Cw_production"],
    "mesh / discretisation record": ["/levels/0/n_sections", "/levels/0/panel_count",
                                     "/production_mesh/n_sections"],
    "convergence record (GCI)": ["/gci/volume/gci_fine", "/cases/0/gci/gci_fine"],
    "acceptance criteria": ["/criteria/Cb_error/measured", "/cases/0/criteria/wave_resistance_error/measured"],
    "evidence level string": ["/evidence_level"],
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def resolve(doc, pointer: str):
    """RFC-6901 resolution ('' -> whole document)."""
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise KeyError(f"pointer must start with '/': {pointer}")
    cur = doc
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


def walk_leaves(node, prefix: str = ""):
    """Yield (pointer, value, dtype) for every scalar leaf of a JSON document."""
    if isinstance(node, dict):
        for key, val in node.items():
            tok = str(key).replace("~", "~0").replace("/", "~1")
            yield from walk_leaves(val, f"{prefix}/{tok}")
    elif isinstance(node, list):
        for idx, val in enumerate(node):
            yield from walk_leaves(val, f"{prefix}/{idx}")
    else:
        if isinstance(node, bool):
            dtype = "bool"
        elif isinstance(node, int):
            dtype = "int"
        elif isinstance(node, float):
            dtype = "float"
        elif node is None:
            dtype = "null"
        else:
            dtype = "str"
        yield prefix, node, dtype


def tail_name(pointer: str) -> str:
    return pointer.rsplit("/", 1)[-1]


def main() -> int:
    docs = {k: json.loads((REPO / p).read_text()) for k, p in SOURCES.items()}

    inventory: dict[str, dict] = {}
    total = transcribed = 0
    roundtrip_failures: list[str] = []

    for study, path in SOURCES.items():
        leaves = []
        for pointer, value, dtype in walk_leaves(docs[study]):
            total += 1
            back = resolve(docs[study], pointer)
            ok = (back == value) and (type(back) is type(value) or
                                      (isinstance(back, (int, float)) and not isinstance(back, bool)))
            if not ok:
                roundtrip_failures.append(f"{path}#{pointer}")
                continue
            transcribed += 1
            leaves.append({
                "pointer": pointer,
                "value": value,
                "dtype": dtype,
                "unit": UNITS.get(tail_name(pointer)),
                "source": path,
            })
        inventory[study] = {
            "source_path": path,
            "source_sha256": sha256_of(REPO / path),
            "source_bytes": (REPO / path).stat().st_size,
            "n_leaves": len(leaves),
            "leaves": leaves,
        }

    classes = {}
    all_leaves = {leaf["pointer"] for inv in inventory.values() for leaf in inv["leaves"]}
    for name, prefixes in REQUIRED_CLASSES.items():
        hit = [p for p in prefixes if p in all_leaves]
        classes[name] = {
            "expected_pointers": prefixes,
            "found_pointers": hit,
            "present": len(hit) == len(prefixes),
        }

    # ---------------------------------------------------------------- typing
    ledger = {
        "purpose": ("classify every reference number of the two files so that a "
                    "repository-computed value can never be quoted as a model test"),
        "entries": [
            {
                "quantity": "wave_resistance.reference (Rw_reference, Cw_reference)",
                "pointer": "/cases/*/Rw_reference",
                "type": "numerical",
                "theory": "Michell thin-ship integral, composite Gauss-Legendre "
                          "(480 subintervals x 20 nodes) -- converged quadrature",
                "pointer_evidence": ["/method/reference"],
                "is_experiment": False,
                "may_be_called_test_data": False,
            },
            {
                "quantity": "wave_resistance.production (Rw_production, Cw_production)",
                "pointer": "/cases/*/Rw_production",
                "type": "numerical",
                "theory": "Michell thin-ship integral, 256-node Gauss-Legendre, "
                          "lambda = 1 + s^2 substitution (repo production path)",
                "pointer_evidence": ["/method/production"],
                "is_experiment": False,
                "may_be_called_test_data": False,
            },
            {
                "quantity": "hydrostatics.reference (V, Cb, KB, A_wp, I_xx, I_yy, BM, GM, C33/C44/C55)",
                "pointer": "/reference/*",
                "type": "analytic",
                "theory": "closed form of the Wigley III parabolic hull "
                          "(V = 4LBT/9, Cb = 4/9, KB = 5T/8, A_wp = 2LB/3, "
                          "I_xx = 4B^3L/105, I_yy = B L^3/30)",
                "pointer_evidence": ["/reference"],
                "is_experiment": False,
                "may_be_called_test_data": False,
            },
            {
                "quantity": "hydrostatics.levels[*] (mesh-integrated values)",
                "pointer": "/levels/*",
                "type": "numerical",
                "theory": "divergence-theorem volume/buoyancy integrals and waterline "
                          "polygon moments over the panelised wetted surface",
                "pointer_evidence": ["/levels/0/panel_count"],
                "is_experiment": False,
                "may_be_called_test_data": False,
            },
            {
                "quantity": "hydrostatics.wetted_area_reference",
                "pointer": "/wetted_area_reference",
                "type": "numerical",
                "theory": "2-D Gauss quadrature of sqrt(1 + y_x^2 + y_z^2) over the "
                          "analytic Wigley surface (semi-analytic: quadrature of a closed form)",
                "pointer_evidence": ["/wetted_area_reference", "/levels/0/wetted_area"],
                "is_experiment": False,
                "may_be_called_test_data": False,
            },
        ],
        "experiment_anchor": {
            "status": "WITHHELD",
            "what_is_missing": [
                "Wigley (1942) model-test resistance and RAO tables",
                "ITTC seakeeping benchmark RAO table for the Wigley hull",
                "any measured heave/pitch RAO ordinate at any Froude number",
            ],
            "in_file_negative_evidence": [
                {
                    "pointer": "/evidence_level",
                    "value": docs["wave_resistance"].get("evidence_level"),
                    "reading": "the file itself declares the evidence as a converged "
                               "theory reference, not a test comparison",
                },
                {
                    "report_path": REPORTS["wave_resistance"],
                    "quote": "This verification does **not** compare with experimental data.",
                    "sha256": sha256_of(REPO / REPORTS["wave_resistance"]),
                },
            ],
            "consequence": ("criterion-2 comparisons may carry the labels analytic / "
                            "numerical only; any table row labelled 'experiment' for "
                            "this hull would be fabricated"),
        },
        "upstream_w18_experiments": {
            "status": "not_retrievable_in_this_worktree",
            "what_was_looked_for": "W18 exp#2 / exp#3 domain-truncation result files",
            "searched": [
                "papers/ (absent before this task)",
                "docs/", "benchmarks/", "tools/",
                "git log --all -- '*truncat*'",
            ],
            "consequence": ("the truncation comparison is re-executed from the in-repo "
                            "machinery at the same hull/fluid caliber instead of being "
                            "rescaled from exp#2/#3; no exp#2/#3 number is quoted"),
        },
    }

    doc = {
        "schema": "w20-wigley-anchor-transcription/v1",
        "task": "W20 seakeeping-truncation item 4 (anchor repair), mission 9647",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": "TensorBEM",
        "git": {
            "base_commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": git("describe", "--always", "--dirty"),
        },
        "builder_sha256": sha256_of(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sources": [
            {"study": s, "path": p, "report_path": REPORTS[s],
             "sha256": sha256_of(REPO / p),
             "report_sha256": sha256_of(REPO / REPORTS[s])}
            for s, p in SOURCES.items()
        ],
        "field_inventory": inventory,
        "completeness": {
            "leaves_total": total,
            "leaves_transcribed": transcribed,
            "rate": transcribed / total if total else 0.0,
            "roundtrip_method": "RFC-6901 resolve(pointer) == source leaf, type-checked",
            "roundtrip_failures": roundtrip_failures,
            "required_classes": classes,
            "required_classes_all_present": all(c["present"] for c in classes.values()),
        },
        "reference_type_ledger": ledger,
    }

    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: {transcribed}/{total} leaves "
          f"({doc['completeness']['rate']:.4%}), "
          f"required classes all present = {doc['completeness']['required_classes_all_present']}, "
          f"wave_resistance sha256 = {doc['sources'][0]['sha256'][:16]}...")
    return 0 if not roundtrip_failures and doc["completeness"]["required_classes_all_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())