"""Build the Wageningen B-series anchor artefact (machine extraction, zero hand typing).

W25 mission ``paper-exp-propeller-openwater-bem-7`` needs an external quantitative
anchor: the Oosterveld & van Oossanen (1975) regression of the Wageningen B-screw
series (NSMB Publication 479; also *International Shipbuilding Progress* 22(251)),
i.e. the 39-term ``KT`` polynomial and the 47-term ``KQ`` polynomial.

The published tables are not available offline in this worktree, so the anchor
enters through a *declared reproduction chain* rather than a hand copy:

1. Oosterveld & van Oossanen (1975) printed tables (39 / 47 terms).
2. The platform ecosystem's reference implementation
   ``backend/services/wageningen_b_series.py`` in the ship-performance-platform
   worktree, whose own provenance note records that its table was extracted
   programmatically from two independent public implementations
   (``AlfredMylne/PropCalc``, ``isulovsky/Wageningen-B-series-propeller-calculator``)
   which agree term for term.
3. This script: imports that module **by file path**, reads
   ``KT_COEFFICIENTS`` / ``KQ_COEFFICIENTS`` programmatically, computes probe
   values with that (independent) implementation, and writes
   ``benchmarks/anchors/wageningen_b_series.json`` with the sha256 of both the
   source file and the emitted table.

No coefficient in the emitted JSON is typed by a human: the only literals in this
script are the *probe points* (J, P/D, Ae/Ao, Z) and the citation metadata.

Usage::

    python tools/build_wageningen_anchor.py \
        --source /root/tp-agent/work/m6783/backend/services/wageningen_b_series.py \
        --out benchmarks/anchors/wageningen_b_series.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_terms(terms) -> str:
    """Fingerprint of a coefficient table: JSON with fixed formatting."""

    payload = json.dumps([list(float(c) if i == 0 else int(c) for i, c in enumerate(t))
                          for t in terms], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_wageningen_source", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # register before execution: the source defines dataclasses, and
    # dataclasses resolves annotations through sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Probe points for the cross-implementation check.  They deliberately span the
#: published validity domain: B4-70 at its design P/D = 1.0 (the W25 case), a
#: 5-blade / P/D = 0.95 point (KP505-like), and the P/D / Ae/Ao corners.
PROBE_POINTS = [
    # (J, P/D, Ae/Ao, Z)
    (0.10, 1.00, 0.70, 4),
    (0.30, 1.00, 0.70, 4),
    (0.50, 1.00, 0.70, 4),
    (0.70, 1.00, 0.70, 4),
    (0.90, 1.00, 0.70, 4),
    (1.00, 1.00, 0.70, 4),
    (1.20, 1.00, 0.70, 4),
    (0.50, 0.60, 0.70, 4),
    (0.50, 1.40, 0.70, 4),
    (0.50, 1.00, 0.30, 4),
    (0.50, 1.00, 1.05, 4),
    (0.70, 0.95, 0.80, 5),
    (0.70, 1.00, 0.70, 7),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="/root/tp-agent/work/m6783/backend/services/wageningen_b_series.py",
        help="reference implementation of the O&O (1975) regression",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/anchors/wageningen_b_series.json",
        help="output anchor path (relative to the repository root)",
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    module = load_module(source)
    kt_terms = [list(t) for t in module.KT_COEFFICIENTS]
    kq_terms = [list(t) for t in module.KQ_COEFFICIENTS]

    if len(kt_terms) != 39 or len(kq_terms) != 47:
        raise SystemExit(
            f"term counts must be 39/47 per O&O (1975); got {len(kt_terms)}/{len(kq_terms)}"
        )

    probes = []
    for advance, p_d, ae_ao, z in PROBE_POINTS:
        probes.append(
            {
                "J": advance,
                "P_D": p_d,
                "Ae_Ao": ae_ao,
                "Z": z,
                "KT": module.kt(advance, p_d, ae_ao, z),
                "KQ": module.kq(advance, p_d, ae_ao, z),
                "eta0": module.eta0(advance, p_d, ae_ao, z),
            }
        )

    # B4-70 (design P/D = 1.0) curve on the mission grid, evaluated by the
    # reference implementation: this is the "anchor curve" the BEM is scored on.
    curve = []
    for step in range(0, 29):  # J = 0.05 .. 1.40, step 0.05
        advance = round(0.05 * step, 10)
        curve.append(
            {
                "J": advance,
                "KT": module.kt(advance, 1.0, 0.70, 4, strict=False),
                "KQ": module.kq(advance, 1.0, 0.70, 4, strict=False),
            }
        )
    for row in curve:
        kq_value = row["KQ"]
        row["eta0"] = (
            0.0 if kq_value <= 0.0 else row["J"] * row["KT"] / (2.0 * math.pi * kq_value)
        )

    j0 = module.zero_thrust_advance_ratio(1.0, 0.70, 4)

    payload = {
        "anchor_id": "AN-WAGENINGEN-BSERIES-1975",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "citation": {
            "authors": "Oosterveld, M.W.C. & van Oossanen, P.",
            "year": 1975,
            "title": (
                "Further computer-analyzed data of the Wageningen B-screw series"
            ),
            "journal": "International Shipbuilding Progress, 22(251)",
            "report": "Netherlands Ship Model Basin (NSMB) Publication 479",
            "content": (
                "39-term KT and 47-term KQ multiple regression of the Wageningen "
                "B-screw series open-water tests, valid Z=2-7, Ae/Ao=0.30-1.05, "
                "P/D=0.5-1.4, Re=2e6 at the 0.75R chord, hub ratio 0.18-0.20"
            ),
        },
        "model": {
            "KT": "sum_i C_i * J^s_i * (P/D)^t_i * (Ae/Ao)^u_i * Z^v_i",
            "KQ": "sum_i D_i * J^s_i * (P/D)^t_i * (Ae/Ao)^u_i * Z^v_i",
            "eta0": "J * KT / (2 pi KQ)",
            "term_layout": ["C", "s[J]", "t[P/D]", "u[Ae/Ao]", "v[Z]"],
            "reference_reynolds": 2.0e6,
        },
        "domain": dict(module.DOMAIN),
        "provenance": {
            "depth": "SECONDARY_REPRODUCTION",
            "declared_limit": (
                "the printed NSMB tables are not available offline in this worktree; "
                "the coefficients enter through the two-step machine chain below and "
                "the regression is additionally adjudicated against physics anchors "
                "(tests/test_wageningen_anchor.py) rather than trusted on faith"
            ),
            "chain": [
                {
                    "step": 1,
                    "what": "published regression tables of Oosterveld & van Oossanen (1975)",
                    "artifact": "NSMB Publication 479 / ISP 22(251)",
                    "verification_in_worktree": "not held",
                },
                {
                    "step": 2,
                    "what": (
                        "platform reference implementation; its own provenance note records a "
                        "programmatic extraction from two independent public implementations "
                        "(AlfredMylne/PropCalc, isulovsky/Wageningen-B-series-propeller-calculator, "
                        "fetched 2026-11-01, agreeing term for term)"
                    ),
                    "artifact": str(source),
                    "sha256": sha256_file(source),
                },
                {
                    "step": 3,
                    "what": "this extraction (importlib read of the coefficient tuples, no retyping)",
                    "tool": "tools/build_wageningen_anchor.py",
                    "command": (
                        "python tools/build_wageningen_anchor.py --source "
                        f"{source} --out {args.out}"
                    ),
                },
            ],
        },
        "kt_terms": kt_terms,
        "kq_terms": kq_terms,
        "fingerprints": {
            "n_kt_terms": len(kt_terms),
            "n_kq_terms": len(kq_terms),
            "kt_table_sha256": sha256_terms(kt_terms),
            "kq_table_sha256": sha256_terms(kq_terms),
        },
        "source_implementation_probes": {
            "note": (
                "values produced by the step-2 implementation at PROBE_POINTS; the "
                "TensorBEM loader must reproduce them to 1e-12 (cross-implementation check)"
            ),
            "rows": probes,
        },
        "b4_70_pd1_curve": {
            "geometry": {"Z": 4, "Ae_Ao": 0.70, "P_D": 1.0, "hub_ratio_range": [0.18, 0.20]},
            "zero_thrust_advance_ratio": j0,
            "J_step": 0.05,
            "rows": curve,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {out_path}")
    print(f"  source sha256      {payload['provenance']['chain'][1]['sha256']}")
    print(f"  kt term sha256     {payload['fingerprints']['kt_table_sha256']}")
    print(f"  kq term sha256     {payload['fingerprints']['kq_table_sha256']}")
    print(f"  B4-70 J0 (P/D=1.0) {j0:.6f}")
    print(f"  file sha256        {sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())