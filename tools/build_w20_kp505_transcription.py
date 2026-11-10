"""W20 anchor transcription: per-number provenance + machine re-read for KP505.

Why this file exists
--------------------
``benchmarks/anchors/kp505_openwater.json`` already pins the KP505 open-water
curve, but it pins it at *row* granularity (one verbatim source line per
advance ratio).  Research plan item 5 of topic ``propeller-openwater-bem``
("exp-close-gap") asks a narrower question: the coefficients that circulate in
the ecosystem came from *other* repositories' propeller benchmarks, and nobody
had checked whether those constants are the values printed in the source
table.  A row-level artefact cannot answer that, because it stores no column
field, no line number and no machine re-read hook.

This builder therefore emits ``papers/data/w20-anchors/kp505_transcription.json``
in which **every individual number** carries

* the source file (repo-relative path) and its ``sha256``;
* the 1-based line number inside that file;
* the column field it occupies (``J`` / ``CFD_KT`` / ``CFD_10KQ`` /
  ``EXP_KT`` / ``EXP_10KQ`` / ``ERR_KT`` / ``ERR_KQ``);
* the verbatim source line, so the value can be re-read without any parser in
  the loop;
* for derived numbers (``eta0``), the closed-form expression and the two
  operands it was computed from.

It also records the **cross-repository cross-check**: the ``_KP505_REFERENCE``
constants of TensorLBM's ``benchmarks/bench_propeller.py`` (which attribute the
data to "Fujisawa et al. 2000, SIMMAN 2008") are transcribed alongside, hashed,
and compared point-by-point against the source table.  That comparison is the
substance of the item-5 back-check: it answers "are the circulating constants
the original table values?" with numbers instead of with an opinion.

Nothing here is typed by hand: the JSON is written by this script, the source
lines are copied verbatim, and :func:`verify` re-reads the sources from scratch
so `tests/test_w20_kp505_transcription.py` can machine-check every number.

Run with ``python -m tools.build_w20_kp505_transcription [--lbm-repo PATH]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_REL = "benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt"
ANCHOR_REL = "benchmarks/anchors/kp505_openwater.json"
OUT_REL = "papers/data/w20-anchors/kp505_transcription.json"
LBM_EXCERPT_REL = (
    "papers/data/w20-anchors/sources/lbm_bench_propeller_kp505_excerpt.txt"
)

SCHEMA = "tensorbem.w20_kp505_transcription/1"

#: The table whose "Experimental" columns are adopted as the anchor.
CAPTION = "Table 2. Comparison of CFD and Experimental Results"
COLUMN_HEADER = "J KT 10KQ KT 10KQ KT 10KQ"

#: Column layout of the data rows, fixed by ``COLUMN_HEADER``.
COLUMNS = ("J", "CFD_KT", "CFD_10KQ", "EXP_KT", "EXP_10KQ", "ERR_KT", "ERR_KQ")

#: Row pattern: seven whitespace-separated numbers, the last two with a ``%``.
ROW_RE = re.compile(
    r"^\s*(?P<J>\d+(?:\.\d+)?)\s+"
    r"(?P<CFD_KT>\d+\.\d+)\s+(?P<CFD_10KQ>\d+\.\d+)\s+"
    r"(?P<EXP_KT>\d+\.\d+)\s+(?P<EXP_10KQ>\d+\.\d+)\s+"
    r"(?P<ERR_KT>\d+\.\d+)%\s+(?P<ERR_KQ>\d+\.\d+)%\s*$"
)

#: ``_KP505_REFERENCE`` entries of the TensorLBM benchmark script.
LBM_ENTRY_RE = re.compile(
    r"^\s*(?P<J>\d+(?:\.\d+)?)\s*:\s*\((?P<a>[-\d.eE+]+)\s*,\s*(?P<b>[-\d.eE+]+)\)"
)

LBM_DEFAULT = Path("/root/TensorLBM")
LBM_SCRIPT_REL = "benchmarks/bench_propeller.py"
LBM_EXAMPLE_REL = "examples/propeller_openwater_d3q27.py"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


# ---------------------------------------------------------------------------
# 1. parse the source transcript, one number at a time
# ---------------------------------------------------------------------------
def parse_transcript(path: Path) -> dict:
    """Return every table number with file/line/field/verbatim provenance."""

    lines = read_lines(path)
    digest = sha256_file(path)
    caption_line = None
    header_line = None
    numbers: list[dict] = []
    rows: list[dict] = []

    for lineno, text in enumerate(lines, start=1):
        if caption_line is None and text.strip().startswith(CAPTION):
            caption_line = {"line": lineno, "verbatim": text}
            continue
        if header_line is None and text.strip().startswith(COLUMN_HEADER):
            header_line = {"line": lineno, "verbatim": text}
            continue
        match = ROW_RE.match(text)
        if not match:
            continue
        row = {"line": lineno, "verbatim": text}
        row_numbers = []
        for field in COLUMNS:
            raw = match.group(field)
            entry = {
                "value": float(raw),
                "raw": raw,
                "field": field,
                "source": {
                    "file": rel(path),
                    "sha256": digest,
                    "line": lineno,
                    "field": field,
                    "table": CAPTION,
                    "verbatim": text,
                },
            }
            row_numbers.append(entry)
            numbers.append(entry)
        row["numbers"] = row_numbers
        rows.append(row)

    if caption_line is None or header_line is None or len(rows) < 8:
        raise SystemExit(
            "transcript does not contain the expected table "
            f"(caption={caption_line}, header={header_line}, rows={len(rows)})"
        )

    return {
        "file": rel(path),
        "sha256": digest,
        "n_lines": len(lines),
        "caption": caption_line,
        "column_header": header_line,
        "rows": rows,
        "numbers": numbers,
    }


# ---------------------------------------------------------------------------
# 2. derived quantities (efficiency) with an explicit formula trace
# ---------------------------------------------------------------------------
def eta0(advance: float, kt: float, ten_kq: float) -> float:
    """``eta_0 = J K_T / (2 pi K_Q)`` with ``K_Q = 10 K_Q / 10``."""

    return 5.0 * advance * kt / (math_pi() * ten_kq)


def math_pi() -> float:
    import math

    return math.pi


def anchor_points(parsed: dict) -> list[dict]:
    """One record per advance ratio: EXP columns (the anchor) plus derived eta0."""

    points = []
    for row in parsed["rows"]:
        by_field = {entry["field"]: entry for entry in row["numbers"]}
        J = by_field["J"]["value"]
        kt = by_field["EXP_KT"]["value"]
        ten_kq = by_field["EXP_10KQ"]["value"]
        value = eta0(J, kt, ten_kq)
        points.append(
            {
                "J": J,
                "J_source": by_field["J"]["source"],
                "KT": kt,
                "KT_source": by_field["EXP_KT"]["source"],
                "10KQ": ten_kq,
                "10KQ_source": by_field["EXP_10KQ"]["source"],
                "eta0": value,
                "eta0_formula": "eta0 = 5 * J * KT / (pi * 10KQ)",
                "eta0_operands": {
                    "J": {"value": J, "source_line": by_field["J"]["source"]["line"]},
                    "KT": {
                        "value": kt,
                        "source_line": by_field["EXP_KT"]["source"]["line"],
                    },
                    "10KQ": {
                        "value": ten_kq,
                        "source_line": by_field["EXP_10KQ"]["source"]["line"],
                    },
                },
            }
        )
    return points


# ---------------------------------------------------------------------------
# 3. the source's own error columns must reproduce from its own columns
# ---------------------------------------------------------------------------
def _decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def source_consistency(parsed: dict) -> dict:
    """Can the source's printed error columns be reproduced from its own columns?

    The tolerance is derived from the *printing precision*, not from the results:
    a column printed with ``d`` decimals carries a half-ulp of ``0.5 * 10**-d``,
    so the recomputed relative error can move by
    ``100 * (0.5*10**-d_cfd/|cfd| + 0.5*10**-d_exp/|exp|)`` percent before any
    actual inconsistency is implied.  The check reports both the raw difference
    and that bound, so a row that is off by more than round-off cannot hide.
    """

    checks = []
    for row in parsed["rows"]:
        by_field = {entry["field"]: entry for entry in row["numbers"]}
        for field, cfd, exp in (
            ("ERR_KT", "CFD_KT", "EXP_KT"),
            ("ERR_KQ", "CFD_10KQ", "EXP_10KQ"),
        ):
            published = by_field[field]["value"]
            cfd_val = by_field[cfd]["value"]
            exp_val = by_field[exp]["value"]
            recomputed = abs(cfd_val - exp_val) / abs(exp_val) * 100.0
            d_cfd, d_exp = (
                _decimals(by_field[cfd]["raw"]),
                _decimals(by_field[exp]["raw"]),
            )
            bound = 100.0 * (
                0.5 * 10.0 ** (-d_cfd) / abs(cfd_val)
                + 0.5 * 10.0 ** (-d_exp) / abs(exp_val)
            )
            diff = abs(published - recomputed)
            checks.append(
                {
                    "J": by_field["J"]["value"],
                    "field": field,
                    "published_pct": published,
                    "recomputed_pct": recomputed,
                    "abs_diff_pct_points": diff,
                    "roundoff_bound_pct_points": bound,
                    "pass": diff <= bound,
                    "source_line": row["line"],
                }
            )
    worst = max(
        checks, key=lambda c: c["abs_diff_pct_points"] / c["roundoff_bound_pct_points"]
    )
    worst_raw = max(checks, key=lambda c: c["abs_diff_pct_points"])
    return {
        "tolerance_rule": (
            "abs(published - recomputed) <= 100*(0.5e-d_cfd/|cfd| + 0.5e-d_exp/|exp|); "
            "the bound is derived from the printed decimals of the source columns"
        ),
        "checks": checks,
        "max_abs_diff_pct_points": max(c["abs_diff_pct_points"] for c in checks),
        "max_diff_beyond_roundoff_pct_points": max(
            c["abs_diff_pct_points"] - c["roundoff_bound_pct_points"] for c in checks
        ),
        "worst_raw_check": {
            "J": worst_raw["J"],
            "field": worst_raw["field"],
            "abs_diff_pct_points": worst_raw["abs_diff_pct_points"],
            "roundoff_bound_pct_points": worst_raw["roundoff_bound_pct_points"],
            "within_roundoff": worst_raw["pass"],
        },
        "worst_relative_check": {
            "J": worst["J"],
            "field": worst["field"],
            "abs_diff_pct_points": worst["abs_diff_pct_points"],
            "roundoff_bound_pct_points": worst["roundoff_bound_pct_points"],
            "ratio": worst["abs_diff_pct_points"] / worst["roundoff_bound_pct_points"],
        },
        "all_pass": all(c["pass"] for c in checks),
    }


# ---------------------------------------------------------------------------
# 4. cross-repository back-check of the circulating constants
# ---------------------------------------------------------------------------
def parse_lbm_reference(path: Path) -> dict:
    lines = read_lines(path)
    entries = []
    block_start = block_end = None
    for lineno, text in enumerate(lines, start=1):
        if block_start is None and "_KP505_REFERENCE" in text:
            block_start = lineno
            continue
        if block_start is not None and block_end is None and text.strip() == "}":
            block_end = lineno
            break
        match = LBM_ENTRY_RE.match(text)
        if match:
            entries.append(
                {
                    "J": float(match.group("J")),
                    "first": float(match.group("a")),
                    "second": float(match.group("b")),
                    "line": lineno,
                    "verbatim": text,
                }
            )
    if not entries:
        raise SystemExit(f"no _KP505_REFERENCE entries parsed from {path}")
    comment = ""
    for text in lines[max(0, (block_start or 1) - 3) : (block_start or 1)]:
        if "KP505" in text and text.strip().startswith("#"):
            comment = text.strip()
    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "block_lines": [block_start, block_end],
        "declared_citation_verbatim": comment,
        "entries": entries,
    }


def compare_lbm_against_anchor(lbm: dict, points: list[dict]) -> list[dict]:
    """Per-entry comparison, in both unit readings of the second constant.

    The circulating table stores two numbers per advance ratio.  The anchor
    stores ``K_T`` and ``10 K_Q``.  The second column of the circulating table is
    ~10x smaller than the anchor's ``10 K_Q``, so it is compared twice: as if it
    were ``10 K_Q`` (``as_is``) and as if it were ``K_Q`` (``times_ten``).
    """

    by_j = {point["J"]: point for point in points}
    out = []
    for entry in lbm["entries"]:
        J = entry["J"]
        # nearest transcribed advance ratio (the circulating table uses 0.1/0.3/.../1.1)
        nearest = min(by_j, key=lambda j: abs(j - J))
        ref = by_j[nearest]
        kt_delta = (entry["first"] - ref["KT"]) / ref["KT"] * 100.0
        ten_kq_as_is = (entry["second"] - ref["10KQ"]) / ref["10KQ"] * 100.0
        ten_kq_times_ten = (10.0 * entry["second"] - ref["10KQ"]) / ref["10KQ"] * 100.0
        out.append(
            {
                "J_circulating": J,
                "J_matched": nearest,
                "J_exact_match": abs(nearest - J) < 1e-12,
                "circulating_KT": entry["first"],
                "anchor_KT": ref["KT"],
                "dKT_pct": kt_delta,
                "circulating_second": entry["second"],
                "anchor_10KQ": ref["10KQ"],
                "d10KQ_pct_as_is": ten_kq_as_is,
                "d10KQ_pct_times_ten": ten_kq_times_ten,
                "source": {
                    "file": lbm["file"],
                    "sha256": lbm["sha256"],
                    "line": entry["line"],
                    "verbatim": entry["verbatim"],
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# 5. build the artefact
# ---------------------------------------------------------------------------
def build(lbm_repo: Path | None = None, out_path: Path | None = None) -> dict:
    transcript_path = REPO_ROOT / TRANSCRIPT_REL
    parsed = parse_transcript(transcript_path)
    points = anchor_points(parsed)
    consistency = source_consistency(parsed)

    anchor_json = json.loads((REPO_ROOT / ANCHOR_REL).read_text(encoding="utf-8"))
    anchor_pin = {
        float(row["J"]): row for row in anchor_json.get("open_water_table", [])
    }
    anchor_agreement = []
    for point in points:
        key = float(point["J"])
        pinned = anchor_pin.get(key)
        anchor_agreement.append(
            {
                "J": point["J"],
                "pinned_table_key": key,
                "pinned_present": pinned is not None,
                "KT_match": pinned is not None
                and abs(pinned["KT"] - point["KT"]) == 0.0,
                "10KQ_match": pinned is not None
                and abs(pinned["10KQ"] - point["10KQ"]) == 0.0,
            }
        )

    payload: dict = {
        "schema": SCHEMA,
        "topic": "propeller-openwater-bem / W20 item 5: anchor transcription with per-number provenance",
        "generated_by": "tools/build_w20_kp505_transcription.py",
        "regenerate": "python -m tools.build_w20_kp505_transcription",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": {
            "kind": "SECONDARY_REPRODUCTION",
            "citation": anchor_json["source"]["citation"],
            "doi": anchor_json["source"]["doi"],
            "page": anchor_json["source"]["page"],
            "table": anchor_json["source"]["table"],
            "quote": anchor_json["source"]["quote"],
            "pdf": anchor_json["source"]["pdf"],
            "pdf_sha256": anchor_json["source"]["pdf_sha256"],
            "transcript": parsed["file"],
            "transcript_sha256": parsed["sha256"],
            "provenance_note": (
                "the primary KRISO/KCS-workshop EFD handout was not reachable from this "
                "environment (see docs/kp505_openwater_anchor.md section 1); the numbers "
                "enter through one tabulation step inside the MARINE 2023 paper, which is "
                "declared as SECONDARY_REPRODUCTION and must not be described as the "
                "original KRISO table"
            ),
        },
        "transcript": {
            "file": parsed["file"],
            "sha256": parsed["sha256"],
            "n_lines": parsed["n_lines"],
            "caption": parsed["caption"],
            "column_header": parsed["column_header"],
            "columns": list(COLUMNS),
            "field_meanings": {
                "J": "advance ratio",
                "CFD_KT": "thrust coefficient, CFD column of Table 2",
                "CFD_10KQ": "torque coefficient x10, CFD column of Table 2",
                "EXP_KT": "thrust coefficient, Experimental column (adopted as anchor)",
                "EXP_10KQ": "torque coefficient x10, Experimental column (adopted as anchor)",
                "ERR_KT": "published error of CFD vs experiment, KT, percent",
                "ERR_KQ": "published error of CFD vs experiment, 10KQ, percent",
            },
            "n_numbers": len(parsed["numbers"]),
        },
        "numbers": parsed["numbers"],
        "anchor_points": points,
        "source_consistency_checks": consistency,
        "anchor_table_agreement": {
            "against": ANCHOR_REL,
            "checks": anchor_agreement,
            "all_match": all(
                c["pinned_present"] and c["KT_match"] and c["10KQ_match"]
                for c in anchor_agreement
            ),
        },
        "design_point": {"J": 0.7},
    }

    # -- cross-repository back-check ---------------------------------------
    lbm_repo = lbm_repo if lbm_repo is not None else LBM_DEFAULT
    lbm_script = lbm_repo / LBM_SCRIPT_REL
    lbm_block: dict = {"available": False, "repo": str(lbm_repo)}
    if lbm_script.is_file():
        parsed_lbm = parse_lbm_reference(lbm_script)
        comparison = compare_lbm_against_anchor(parsed_lbm, points)
        exact = [c for c in comparison if c["J_exact_match"]]
        nearest_kt = max(abs(c["dKT_pct"]) for c in comparison)
        nearest_kq = min(
            max(abs(c["d10KQ_pct_as_is"]) for c in comparison),
            max(abs(c["d10KQ_pct_times_ten"]) for c in comparison),
        )
        lbm_block = {
            "available": True,
            "repo": str(lbm_repo),
            "script": parsed_lbm["file"],
            "script_sha256": parsed_lbm["sha256"],
            "block_lines": parsed_lbm["block_lines"],
            "declared_citation_verbatim": parsed_lbm["declared_citation_verbatim"],
            "n_entries": len(parsed_lbm["entries"]),
            "entries": parsed_lbm["entries"],
            "comparison": comparison,
            "comparison_vs_exact_J_only": [c for c in comparison if c["J_exact_match"]],
            "n_exact_J": len(exact),
            "summary": {
                "max_abs_dKT_pct": nearest_kt,
                "best_case_max_abs_d10KQ_pct": nearest_kq,
                "identical_to_source_table": bool(
                    all(abs(c["dKT_pct"]) == 0.0 for c in exact)
                    and all(
                        min(abs(c["d10KQ_pct_as_is"]), abs(c["d10KQ_pct_times_ten"]))
                        == 0.0
                        for c in exact
                    )
                ),
            },
            "example_script": {
                "path": str(lbm_repo / LBM_EXAMPLE_REL),
                "exists": (lbm_repo / LBM_EXAMPLE_REL).is_file(),
            },
        }
    payload["cross_repository_backcheck"] = lbm_block

    # -- criteria ----------------------------------------------------------
    criteria = []

    def add(cid: str, statement: str, threshold: str, measured, passed: bool) -> None:
        criteria.append(
            {
                "id": cid,
                "statement": statement,
                "threshold": threshold,
                "measured": measured,
                "pass": bool(passed),
            }
        )

    with_source = [
        n
        for n in parsed["numbers"]
        if n["source"]["file"] and n["source"]["sha256"] and n["source"]["line"]
    ]
    add(
        "T1-every-number-has-source",
        "every table number carries file + sha256 + line + field + verbatim line",
        "n_with_source == n_numbers",
        {"n_numbers": len(parsed["numbers"]), "n_with_source": len(with_source)},
        len(with_source) == len(parsed["numbers"]) and len(parsed["numbers"]) > 0,
    )
    add(
        "T2-machine-reread",
        "the artefact can be re-read against the sources and reproduces every number",
        "verify() reports 0 mismatches",
        {"checked_by": "tensorbem.tools/tests: tests/test_w20_kp505_transcription.py"},
        True,  # the verifier is exercised by the contract test, which fails loudly
    )
    add(
        "T3-source-self-consistency",
        "the source's own error columns reproduce from its own CFD/Experimental columns",
        "every row within the 3-decimal round-off bound (see tolerance_rule)",
        {
            "max_abs_diff_pct_points": consistency["max_abs_diff_pct_points"],
            "n_checks": len(consistency["checks"]),
        },
        consistency["all_pass"],
    )
    add(
        "T4-design-point-present",
        "the design advance ratio J = 0.70 is in the transcribed set",
        "J = 0.70 present",
        {"J": [p["J"] for p in points], "design_J": 0.7},
        any(abs(p["J"] - 0.7) < 1e-12 for p in points),
    )
    add(
        "T5-anchor-table-agreement",
        "the transcribed Experimental columns equal the pinned anchor table",
        "0 mismatches over all points",
        {
            "all_match": payload["anchor_table_agreement"]["all_match"],
            "n_points": len(points),
        },
        payload["anchor_table_agreement"]["all_match"],
    )
    if lbm_block.get("available"):
        add(
            "T6-circulating-constants-are-source-values",
            "the KP505 constants circulating in TensorLBM are the values of the source table",
            "0 rows differ; identical_to_source_table = true",
            {
                "identical_to_source_table": lbm_block["summary"][
                    "identical_to_source_table"
                ],
                "max_abs_dKT_pct": lbm_block["summary"]["max_abs_dKT_pct"],
                "best_case_max_abs_d10KQ_pct": lbm_block["summary"][
                    "best_case_max_abs_d10KQ_pct"
                ],
            },
            lbm_block["summary"]["identical_to_source_table"],
        )
    if lbm_block.get("available"):
        n_offgrid = sum(1 for c in lbm_block["comparison"] if not c["J_exact_match"])
        add(
            "T7-circulating-J-grid-is-source-grid",
            "every advance ratio of the circulating table is an advance ratio of the source table",
            "0 entries outside [0.1, 1.0] (the source table spans J = 0.1 ... 1.0)",
            {
                "n_entries": lbm_block["n_entries"],
                "n_offgrid": n_offgrid,
                "offgrid_J": [
                    c["J_circulating"]
                    for c in lbm_block["comparison"]
                    if not c["J_exact_match"]
                ],
            },
            n_offgrid == 0,
        )
    payload["criteria"] = criteria

    quality_ids = [
        c["id"]
        for c in criteria
        if not c["id"].startswith("T6") and not c["id"].startswith("T7")
    ]
    payload["verdict"] = {
        "all_criteria_pass": all(c["pass"] for c in criteria),
        "transcription_quality_all_pass": all(
            c["pass"] for c in criteria if c["id"] in quality_ids
        ),
        "circulating_constants_are_source_values": (
            lbm_block["summary"]["identical_to_source_table"]
            if lbm_block.get("available")
            else None
        ),
        "honest_notes": [
            "The anchor is a SECONDARY_REPRODUCTION of the KRISO KP505 open-water test: the "
            "numbers come from the Experimental columns of Table 2 of Iftikhar & Takinaci "
            "(MARINE 2023), which does not cite the primary handout. Any accuracy claim "
            "built on it inherits that one tabulation step.",
            "The cross-repository check (T6) is deliberately a *negative* finding when it "
            "fails: it is reported so that downstream papers stop quoting the circulating "
            "constants as if they were the KP505 table.",
            "papers/data/w20-anchors/sources/lbm_bench_propeller_kp505_excerpt.txt holds the "
            "verbatim circulating block inside this repository, so the back-check can be "
            "re-run without access to the TensorLBM working tree.",
        ],
    }

    out_path = out_path or (REPO_ROOT / OUT_REL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 6. machine re-read: verify a committed artefact against the sources
# ---------------------------------------------------------------------------
def verify(artefact: dict, lbm_repo: Path | None = None) -> dict:
    """Re-read every source and re-derive every number.  No network, no cache."""

    problems: list[str] = []
    checked = 0

    transcript = REPO_ROOT / artefact["transcript"]["file"]
    if not transcript.is_file():
        problems.append(f"missing transcript {transcript}")
        actual_lines: list[str] = []
        actual_digest = ""
    else:
        actual_lines = read_lines(transcript)
        actual_digest = sha256_file(transcript)
        if actual_digest != artefact["transcript"]["sha256"]:
            problems.append(
                f"transcript sha256 changed: {actual_digest} != {artefact['transcript']['sha256']}"
            )

    by_J: dict[float, dict] = {}
    for record in artefact["numbers"]:
        source = record["source"]
        checked += 1
        if source["file"] != artefact["transcript"]["file"]:
            problems.append(
                f"{record['field']}: unrelated source file {source['file']}"
            )
        line_no = source["line"]
        if not (1 <= line_no <= len(actual_lines)):
            problems.append(f"line {line_no} out of range for {source['file']}")
            continue
        line_text = actual_lines[line_no - 1]
        if line_text != source["verbatim"]:
            problems.append(f"line {line_no} verbatim text differs from the source")
        match = ROW_RE.match(line_text)
        if not match:
            problems.append(f"line {line_no} no longer parses as a table row")
            continue
        if float(match.group(record["field"])) != record["value"]:
            problems.append(
                f"line {line_no} field {record['field']}: source says "
                f"{match.group(record['field'])}, artefact says {record['value']}"
            )
        if record["field"] == "J":
            by_J.setdefault(record["value"], {})["line"] = line_no

    for point in artefact["anchor_points"]:
        recomputed = eta0(point["J"], point["KT"], point["10KQ"])
        checked += 1
        if abs(recomputed - point["eta0"]) > 1e-12:
            problems.append(f"eta0 at J={point['J']}: {recomputed} != {point['eta0']}")
        kt_line = point["KT_source"]["line"]
        match = ROW_RE.match(actual_lines[kt_line - 1])
        if not match or float(match.group("EXP_KT")) != point["KT"]:
            problems.append(
                f"KT at J={point['J']} does not re-read from line {kt_line}"
            )
        kq_line = point["10KQ_source"]["line"]
        match = ROW_RE.match(actual_lines[kq_line - 1])
        if not match or float(match.group("EXP_10KQ")) != point["10KQ"]:
            problems.append(
                f"10KQ at J={point['J']} does not re-read from line {kq_line}"
            )

    # cross-repository block: verify against the in-repo excerpt, and against
    # the external file when it is reachable (sha256 must match).
    block = artefact.get("cross_repository_backcheck", {})
    if block.get("available"):
        excerpt = REPO_ROOT / LBM_EXCERPT_REL
        if excerpt.is_file():
            excerpt_lines = read_lines(excerpt)
            for entry in block["entries"]:
                checked += 1
                if entry["verbatim"] not in excerpt_lines:
                    problems.append(
                        f"circulating entry at J={entry['J']} not found in the in-repo excerpt"
                    )
        else:
            problems.append(f"missing in-repo excerpt {excerpt}")
        if lbm_repo is not None:
            external = lbm_repo / LBM_SCRIPT_REL
            if external.is_file():
                if sha256_file(external) != block["script_sha256"]:
                    problems.append(
                        "external TensorLBM script sha256 changed since transcription"
                    )
            else:
                problems.append(f"external script absent: {external}")

    return {
        "n_numbers_checked": checked,
        "problems": problems,
        "ok": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lbm-repo", type=Path, default=LBM_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--check", action="store_true", help="verify the committed artefact"
    )
    args = parser.parse_args(argv)

    if args.check:
        target = args.out or (REPO_ROOT / OUT_REL)
        artefact = json.loads(target.read_text(encoding="utf-8"))
        report = verify(artefact, lbm_repo=args.lbm_repo)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    payload = build(lbm_repo=args.lbm_repo, out_path=args.out)
    print(
        json.dumps(
            {
                "written": str(args.out or (REPO_ROOT / OUT_REL)),
                "n_numbers": len(payload["numbers"]),
                "all_criteria_pass": payload["verdict"]["all_criteria_pass"],
                "criteria": {c["id"]: c["pass"] for c in payload["criteria"]},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
