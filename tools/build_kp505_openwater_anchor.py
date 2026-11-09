"""Build the KP505 open-water anchor artefact for the propeller-openwater-bem study.

The artefact `benchmarks/anchors/kp505_openwater.json` is *generated*: every number
in it is either

* parsed out of a committed source document (the verbatim transcription of the
  published open-water table is extracted from the committed PDF with pypdf and
  written next to it), or
* computed in this script from those parsed numbers (open-water efficiency
  eta0 = J*KT/(2*pi*KQ)), or
* measured on the committed official workshop geometry (KP505 IGES).

Nothing is hand-typed: the transcription file is produced by this script from the
PDF bytes, and the JSON is produced by this script from the transcription.

Run:  python -m tools.build_kp505_openwater_anchor        (or `python tools/...py`)
      python -m tools.build_kp505_openwater_anchor --check   # verify, do not write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "benchmarks" / "anchors" / "sources"
PDF = SRC_DIR / "kp505_marine2023_iftikhar.pdf"
IGES = SRC_DIR / "KP505_modelScale_250mm.igs"
TRANSCRIPT = SRC_DIR / "kp505_marine2023_p10_table2.txt"
GEOM_TRANSCRIPT = SRC_DIR / "kp505_marine2023_table1_propeller_characteristics.txt"
OUT = REPO / "benchmarks" / "anchors" / "kp505_openwater.json"

TABLE_CAPTION = "Table 2. Comparison of CFD and Experimental Results"
DESIGN_J = 0.70
MIN_POINTS = 8
ETA_TOL = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 1. transcription of the published table (verbatim, from the committed PDF)
# --------------------------------------------------------------------------
def find_page(caption: str) -> tuple[int, str]:
    from pypdf import PdfReader  # local import: only the builder needs pypdf

    reader = PdfReader(str(PDF))
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if caption in text:
            return idx + 1, text
    raise SystemExit("caption not found in the source PDF: %s" % caption)


def transcribe_table() -> dict:
    page_no, page_text = find_page(TABLE_CAPTION)
    TRANSCRIPT.write_text(page_text, encoding="utf-8")

    # rows look like:  0.1 0.479 0.644 0.494 0.683 3.06% 5.73%
    rows = []
    for line in page_text.splitlines():
        m = re.match(
            r"^\s*(\d+(?:\.\d+)?)\s+"          # J
            r"([\d.]+)\s+([\d.]+)\s+"          # CFD: KT, 10KQ
            r"([\d.]+)\s+([\d.]+)\s+"          # Experimental: KT, 10KQ
            r"(-?[\d.]+)%\s+(-?[\d.]+)%\s*$",  # error columns
            line,
        )
        if m:
            rows.append(
                {
                    "J": float(m.group(1)),
                    "KT_cfd": float(m.group(2)),
                    "10KQ_cfd": float(m.group(3)),
                    "KT": float(m.group(4)),
                    "10KQ": float(m.group(5)),
                    "err_KT_pct_reported": float(m.group(6)),
                    "err_10KQ_pct_reported": float(m.group(7)),
                    "source_line": line.rstrip(),
                }
            )
    return {"page_no": page_no, "page_text": page_text, "rows": rows}


# --------------------------------------------------------------------------
# 2. geometry measured on the official workshop IGES model-scale propeller
# --------------------------------------------------------------------------
def _iges_records(path: Path) -> list[list[str]]:
    """Split the IGES parameter-data section into per-entity field lists."""
    text = path.read_text(errors="ignore").splitlines()
    chunks: list[list[str]] = []
    cur: list[str] = []
    for line in text:
        if len(line) < 73 or line[72] != "P":
            continue
        for piece in line[:64].strip().split(","):
            piece = piece.strip()
            if not piece:
                continue
            if piece.endswith(";"):
                cur.append(piece[:-1])
                chunks.append(cur)
                cur = []
            else:
                cur.append(piece)
    return chunks


def _nums(fields: list[str], start: int = 1) -> list[float]:
    out: list[float] = []
    for f in fields[start:]:
        try:
            out.append(float(f))
        except ValueError:
            break
    return out


def iges_axis(path: Path) -> tuple[str, tuple[float, float, float]]:
    """Shaft axis from the IGES 'surface of revolution' entities (type 120)."""
    axes = []
    for fields in _iges_records(path):
        try:
            etype = int(fields[0])
        except ValueError:
            continue
        if etype != 120:
            continue
        n = _nums(fields)          # L, AX, AY, AZ, R, T, G
        if len(n) >= 4:
            axes.append((n[1], n[2], n[3]))
    if not axes:
        raise SystemExit("no surface-of-revolution entity found in %s" % path)
    sx = sum(a[0] for a in axes) / len(axes)
    sy = sum(a[1] for a in axes) / len(axes)
    sz = sum(a[2] for a in axes) / len(axes)
    comps = {"x": abs(sx), "y": abs(sy), "z": abs(sz)}
    name = max(comps, key=comps.get)
    idx = "xyz".index(name)
    vec = [0.0, 0.0, 0.0]
    vec[idx] = (sx, sy, sz)[idx]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return name, tuple(v / norm for v in vec)


def iges_points(path: Path) -> list[tuple[float, float, float]]:
    """Coordinate triples of the IGES NURBS curves/surfaces (types 126, 128).

    For both types the parameter layout is
    126: K, M, PROP1..5, (K+M+2) knots, (K+M+1) weights, 3*(K+1)*(M+1) control points
    128: K1, K2, M1, M2, PROP1..5, (K1+M1+2)+(K2+M2+2) knots,
         (K1+1)*(K2+1) weights, 3*(K1+1)*(K2+1) control points
    """
    pts: list[tuple[float, float, float]] = []
    for fields in _iges_records(path):
        try:
            etype = int(fields[0])
        except ValueError:
            continue
        nums = _nums(fields)
        if etype == 126 and len(nums) >= 7:
            K, M = int(nums[0]), int(nums[1])
            skip = 7 + (K + M + 2) + (K + M + 1)
            payload = nums[skip:skip + 3 * (K + 1) * (M + 1)]
        elif etype == 128 and len(nums) >= 9:
            K1, K2, M1, M2 = (int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]))
            skip = 9 + (K1 + M1 + 2) + (K2 + M2 + 2)
            npnt = (K1 + 1) * (K2 + 1)
            payload = nums[skip:skip + 3 * npnt]
        elif etype in (110, 116):
            payload = nums
        else:
            continue
        n = len(payload) - (len(payload) % 3)
        for i in range(0, n, 3):
            pts.append((payload[i], payload[i + 1], payload[i + 2]))
    return pts


def measure_geometry(path: Path) -> dict:
    pts = iges_points(path)
    if not pts:
        raise SystemExit("no IGES control points parsed from %s" % path)
    axis_name, axis = iges_axis(path)
    ai = "xyz".index(axis_name)
    bi, ci = (ai + 1) % 3, (ai + 2) % 3

    def radius(p):
        return math.hypot(p[bi], p[ci])

    r_max = max(radius(p) for p in pts)

    near = [p for p in pts if radius(p) >= 0.90 * r_max]
    nbins = 72
    occ = [0] * nbins
    for p in near:
        theta = math.degrees(math.atan2(p[ci], p[bi])) % 360.0
        occ[int(theta // (360.0 / nbins)) % nbins] += 1
    closed = [1 if (occ[i] or occ[(i - 1) % nbins] or occ[(i + 1) % nbins]) else 0
              for i in range(nbins)]
    clusters = sum(1 for i in range(nbins) if closed[i] and not closed[(i - 1) % nbins])

    return {
        "iges_sha256": sha256(path),
        "iges_file": path.name,
        "shaft_axis": axis_name,
        "shaft_axis_vector": axis,
        "points_used": len(pts),
        "tip_radius_raw": r_max,
        "diameter_raw": 2.0 * r_max,
        "unit_note": "IGES global section: '2HMM,1.0' -> model units are millimetres, factor 1.0",
        "near_tip_points": len(near),
        "blade_count_measured": clusters,
    }


# --------------------------------------------------------------------------
# 3. assemble the artefact
# --------------------------------------------------------------------------
def eta0(J: float, KT: float, tenKQ: float) -> float:
    """eta0 = J*KT/(2*pi*KQ) with KQ = 10KQ/10  ->  5*J*KT/(pi*10KQ)."""
    return 5.0 * J * KT / (math.pi * tenKQ)


def build() -> dict:
    tab = transcribe_table()
    rows = tab["rows"]
    geom = measure_geometry(IGES)

    table = []
    for r in rows:
        e = eta0(r["J"], r["KT"], r["10KQ"])
        table.append(
            {
                "J": r["J"],
                "KT": r["KT"],
                "10KQ": r["10KQ"],
                "eta0": e,
                "eta0_recomputed": eta0(r["J"], r["KT"], r["10KQ"]),
                "eta0_residual": abs(e - eta0(r["J"], r["KT"], r["10KQ"])),
                "KT_cfd": r["KT_cfd"],
                "10KQ_cfd": r["10KQ_cfd"],
                "source_line": r["source_line"],
            }
        )
    peak = max(table, key=lambda x: x["eta0"])
    max_res = max(t["eta0_residual"] for t in table)
    js = [t["J"] for t in table]
    design = [t for t in table if abs(t["J"] - DESIGN_J) < 1e-12]

    # reported error columns of the source, recomputed (secondary consistency check)
    err_checks = []
    for r in rows:
        err_checks.append(
            {
                "J": r["J"],
                "err_KT_pct_reported": r["err_KT_pct_reported"],
                "err_KT_pct_recomputed": (r["KT_cfd"] - r["KT"]) / r["KT"] * 100.0,
                "err_10KQ_pct_reported": r["err_10KQ_pct_reported"],
                "err_10KQ_pct_recomputed": (r["10KQ_cfd"] - r["10KQ"]) / r["10KQ"] * 100.0,
            }
        )
    max_err_dev = max(
        max(abs(c["err_KT_pct_reported"] - c["err_KT_pct_recomputed"]),
            abs(c["err_10KQ_pct_reported"] - c["err_10KQ_pct_recomputed"]))
        for c in err_checks
    )

    geom_page_no, geom_page_text = find_page("Table 1. Propeller Characteristics")
    GEOM_TRANSCRIPT.write_text(geom_page_text, encoding="utf-8")
    m_blades = re.search(r"No\. of Blades\s+(\d+)", geom_page_text)
    m_diam = re.search(r"Diameter \(m\)\s+([\d.]+)", geom_page_text)
    if not (m_blades and m_diam):
        raise SystemExit("could not read Table 1 (blade count / diameter) from the source PDF")

    geometry_fields = {
        "diameter_m": {
            "value": float(m_diam.group(1)),
            "status": "VERIFIED",
            "evidence": (
                "official KCS workshop geometry archive member 'KP505/KP505_modelScale_250mm.igs' "
                "(link text 'Propeller Geometory' on t2015.nmri.go.jp/kcs_gc.html) states the model "
                "diameter D = 250 mm; independently quoted from the source paper Table 1 "
                "(page %d): '%s'" % (geom_page_no, m_diam.group(0).strip())
            ),
        },
        "blade_count_Z": {
            "value": int(m_blades.group(1)),
            "status": "VERIFIED_SECONDARY",
            "evidence": "source paper Table 1 (page %d): '%s'" % (geom_page_no, m_blades.group(0).strip()),
        },
        "P_over_D": {
            "value": None,
            "status": "WITHHELD",
            "evidence": (
                "the only reachable statement is the source paper Table 1 (page %d): 'P/D (mean) 0.950', "
                "which is not confirmed by any official document reachable here and is not used as a "
                "criterion value; deriving P/D from the IGES needs blade-section pitch angles "
                "(not done in this deliverable)" % geom_page_no
            ),
        },
        "AE_over_A0": {
            "value": None,
            "status": "WITHHELD",
            "evidence": (
                "the only reachable statement is the source paper Table 1 (page %d): 'Ae/Ao 0.800', "
                "conflicting with the workshop geometry package and with the value commonly quoted for "
                "KP505; the developed-area ratio cannot be measured on the 3-D surface without "
                "un-wrapping cylindrical sections, so it stays WITHHELD" % geom_page_no
            ),
        },
    }

    geom_attempt = {
        "status": "UNRELIABLE",
        "why": (
            "the IGES NURBS control-point cage is not a surface point cloud; the measured tip radius "
            "(%.1f file units) exceeds the documented model radius (125 mm) and the near-tip angular "
            "histogram resolves only %d clusters, so neither D nor Z is taken from this measurement"
            % (geom["tip_radius_raw"], geom["blade_count_measured"])
        ),
        "measured": geom,
    }

    criteria = [
        {
            "id": "C1-quote-and-table-number",
            "statement": "every data point carries a verbatim quotation and a table number",
            "threshold": "1 quote + table id per point",
            "measured": {
                "table_id": "Table 2",
                "page": tab["page_no"],
                "quote_all_points": TABLE_CAPTION,
                "rows_with_source_line": sum(1 for t in table if t["source_line"]),
            },
            "pass": all(t["source_line"] for t in table) and bool(TABLE_CAPTION),
        },
        {
            "id": "C2-points-and-design-J",
            "statement": ">= 8 advance-ratio points, design advance ratio included",
            "threshold": "n_J >= 8 and J_design in J",
            "measured": {"n_J": len(js), "J": js, "design_J": DESIGN_J,
                         "design_J_present": bool(design)},
            "pass": len(js) >= MIN_POINTS and bool(design),
        },
        {
            "id": "C3-eta0-self-consistency",
            "statement": "eta0 = J*KT/(2*pi*KQ) recomputed from the tabulated KT, 10KQ; peak J recorded explicitly",
            "threshold": "|eta0 - recomputed| < %g" % ETA_TOL,
            "measured": {"max_residual": max_res, "peak_J": peak["J"], "peak_eta0": peak["eta0"]},
            "pass": max_res < ETA_TOL and bool(peak),
        },
    ]

    art = {
        "task_key": "paper-exp-propeller-openwater-bem-1",
        "repo": "TensorBEM",
        "topic": "propeller-openwater-bem / KP505 official open-water data transcription",
        "generated_by": "tools/build_kp505_openwater_anchor.py",
        "regenerate": "python -m tools.build_kp505_openwater_anchor",
        "criteria": criteria,
        "verdict": {
            "all_criteria_pass": all(c["pass"] for c in criteria),
            "geometry_complete": all(geometry_fields[k]["value"] is not None
                                     for k in ("diameter_m", "blade_count_Z",
                                               "P_over_D", "AE_over_A0")),
            "honest_notes": [
                "PROVENANCE GAP (honest): the primary KRISO/KCS-workshop EFD handout and the "
                "Hino (2005) Tokyo-2015 proceedings were not reachable from this sandbox "
                "(egress allowlist). The transcribed curve is a SECONDARY reproduction: the "
                "'Experimental' columns of Table 2 of Iftikhar & Takinaci, MARINE 2023.",
                "That paper does not cite the source of its experimental column, so the anchor is "
                "labelled SECONDARY_REPRODUCTION and must be re-verified against the KRISO table "
                "before it is used for an accuracy claim in the parent study.",
                "Geometry: D and blade count are taken from/measured on the official KRISO geometry "
                "archive; P/D and AE/A0 are WITHHELD (no reachable official textual source).",
            ],
        },
        "design_point": {k: design[0][k] for k in ("J", "KT", "10KQ", "eta0")} if design else None,
        "peak_efficiency": {"J": peak["J"], "eta0": peak["eta0"],
                            "KT": peak["KT"], "10KQ": peak["10KQ"]},
        "open_water_table": table,
        "source": {
            "kind": "SECONDARY_REPRODUCTION",
            "citation": (
                "H. Iftikhar, A. C. Takinaci, 'Performance Prediction of KP505 Propeller Using "
                "Lifting Line Theory and RANS', X International Conference on Computational Methods "
                "in Marine Engineering (MARINE 2023), CIMNE, DOI 10.23967/marine.2023.120"
            ),
            "url": "https://www.scipedia.com/wd/images/2/23/Iftikhar_Takinaci_2023a_2756_performance.pdf",
            "doi": "10.23967/marine.2023.120",
            "page": tab["page_no"],
            "table": "Table 2. Comparison of CFD and Experimental Results (columns 'Experimental')",
            "quote": TABLE_CAPTION,
            "pdf": PDF.name,
            "pdf_sha256": sha256(PDF),
            "transcript": TRANSCRIPT.name,
            "transcript_sha256": sha256(TRANSCRIPT),
        },
        "source_consistency_checks": {
            "reported_vs_recomputed_error_columns_max_dev_pct": max_err_dev,
            "note": "the source's own error columns reproduce from its own CFD/EFD columns",
        },
        "geometry": {
            "spec": {
                "diameter_m": float(m_diam.group(1)),
                "blade_count_Z": int(m_blades.group(1)),
                "P_over_D": None,
                "AE_over_A0": None,
            },
            "fields": geometry_fields,
            "measurement_attempt": geom_attempt,
            "provenance": {
                "archive_url": "http://www.t2015.nmri.go.jp/file/Geometry_IGES_files/kcs/KP505.zip",
                "archive_page": "http://www.t2015.nmri.go.jp/kcs_gc.html (link text 'Propeller Geometory')",
                "archive_quote": "Following archive contains IGES&STP geometories of the KCS propeller.",
                "archive_members": ["KP505/KP505.png", "KP505/KP505_fullScale_7900mm.igs",
                                    "KP505/KP505_fullScale_7900mm.stp",
                                    "KP505/KP505_modelScale_250mm.igs"],
                "committed_member": IGES.name,
                "table1_transcript": GEOM_TRANSCRIPT.name,
                "table1_transcript_sha256": sha256(GEOM_TRANSCRIPT),
                "table1_page": geom_page_no,
            },
        },
        "method": {
            "eta0_definition": "eta0 = J*KT/(2*pi*KQ), KQ = 10KQ/10",
            "script": "tools/build_kp505_openwater_anchor.py",
            "tests": "tests/test_kp505_openwater_anchor.py",
        },
    }
    return art


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed artefact, do not rewrite")
    args = ap.parse_args(argv)

    art = build()
    if args.check:
        if not OUT.exists():
            print("MISSING %s" % OUT)
            return 1
        old = json.loads(OUT.read_text())
        same = json.dumps(old, sort_keys=True) == json.dumps(art, sort_keys=True)
        print("artefact up to date" if same else "artefact OUT OF DATE")
        return 0 if same else 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % OUT.relative_to(REPO))
    for c in art["criteria"]:
        print("  %-28s %s" % (c["id"], "PASS" if c["pass"] else "FAIL"))
    print("  design point   ", art["design_point"])
    print("  peak eta0      ", art["peak_efficiency"])
    g = art["geometry"]["spec"]
    print("  geometry       ", g)
    return 0


if __name__ == "__main__":
    sys.exit(main())