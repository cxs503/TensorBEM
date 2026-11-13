#!/usr/bin/env python3
"""Independent re-verification of m17207 (CASE-BEM-KP505-OPENWATER).

Nothing here imports the subject's driver: every number is recomputed from a
source that the audited branch did not produce for this purpose --

* the reference column is re-extracted **from the PDF itself** (pypdf, p. 10),
  not from the branch's transcription file;
* the measured column is read from the delivered CSV / JSON artefacts;
* the 5 % verdict is recounted arithmetically.

Outputs ``verification.json`` next to this file.  Exit code 1 if any check
contradicts the audited claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pypdf

HERE = Path(__file__).resolve().parent
SNAP = HERE / "snapshot"
SUBJECT_WT = Path("/root/tp-agent/work/m17207")
PDF = SUBJECT_WT / "benchmarks/anchors/sources/kp505_marine2023_iftikhar.pdf"
TOL = 5.0  # case-pool tol_policy for BEM-KP505-OPENWATER: "<5%"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_pdf_table2() -> dict[float, dict[str, float]]:
    """Extract Table 2 (p. 10) rows: J, CFD KT/10KQ, EXP KT/10KQ, printed errors."""
    text = pypdf.PdfReader(str(PDF)).pages[9].extract_text()
    rows = re.findall(
        r"^([01](?:\.\d)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)%\s+([\d.]+)%", text, re.M)
    out = {}
    for j, ckt, ckq, ekt, ekq, rkt, rkq in rows:
        out[float(j)] = {"cfd_KT": float(ckt), "cfd_10KQ": float(ckq),
                         "exp_KT": float(ekt), "exp_10KQ": float(ekq),
                         "printed_err_KT": float(rkt), "printed_err_10KQ": float(rkq)}
    return out


def main() -> int:
    checks: dict[str, object] = {}
    pdf = parse_pdf_table2()
    checks["pdf_rows_parsed"] = len(pdf)

    case = json.loads((SNAP / "kp505_case.json").read_text())
    rows = case["canonical"]["rows"]

    # 1. transcribed anchor vs the PDF, digit by digit
    anchor_diff = [max(abs(r["KT_anchor"] - pdf[r["J"]]["exp_KT"]),
                       abs(r["10KQ_anchor"] - pdf[r["J"]]["exp_10KQ"]))
                   for r in rows]
    checks["anchor_vs_pdf_max_abs_diff"] = max(anchor_diff)
    checks["anchor_matches_pdf_all_10"] = max(anchor_diff) == 0.0

    # 2. eta0 anchor self-consistency (eta0 = J*KT/(2*pi*KQ))
    eta_res = [abs(r["eta0_anchor"] - r["J"] * pdf[r["J"]]["exp_KT"]
                   / (2 * math.pi * pdf[r["J"]]["exp_10KQ"] / 10.0)) for r in rows]
    checks["anchor_eta0_max_residual"] = max(eta_res)

    # 3. the paper's own printed error column vs its own KT/10KQ columns
    paper_err_dev = [max(abs(abs(100 * (pdf[j]["cfd_KT"] - pdf[j]["exp_KT"]) / pdf[j]["exp_KT"])
                             - pdf[j]["printed_err_KT"]),
                         abs(abs(100 * (pdf[j]["cfd_10KQ"] - pdf[j]["exp_10KQ"]) / pdf[j]["exp_10KQ"])
                             - pdf[j]["printed_err_10KQ"])) for j in pdf]
    checks["paper_error_column_max_dev_pp"] = max(paper_err_dev)
    checks["paper_error_column_note"] = (
        "printed CFD-vs-EXP error column recomputes within this many percentage points "
        "from the printed KT/10KQ columns -> the experimental columns were transcribed "
        "consistently with the table's own arithmetic; outlier is J=1.0 KT")

    # 4. recompute the measured side against the anchor, independently
    recomp = {}
    counts = {"KT": 0, "10KQ": 0, "eta0": 0}
    all_three, all_three_adm = [], []
    worst_abs = {"KT_pct": 0.0, "10KQ_pct": 0.0, "eta0_pct": 0.0}
    worst_adm_abs = dict(worst_abs)
    for r in rows:
        j = r["J"]
        e = pdf[j]
        kq_a = e["exp_10KQ"] / 10.0
        eta_a = j * e["exp_KT"] / (2 * math.pi * kq_a)
        pct = {"KT_pct": 100 * (r["KT"] - e["exp_KT"]) / e["exp_KT"],
               "10KQ_pct": 100 * (r["10KQ"] - e["exp_10KQ"]) / e["exp_10KQ"],
               "eta0_pct": 100 * (r["eta0"] - eta_a) / eta_a}
        agree = max(abs(pct[k] - r["error_pct"][k]) for k in pct)
        recomp[j] = {"recomputed_pct": pct, "reported_pct": r["error_pct"],
                     "max_abs_agreement_diff": agree,
                     "pass_recomputed": {k: abs(v) < TOL for k, v in pct.items()}}
        for k, short in (("KT_pct", "KT"), ("10KQ_pct", "10KQ"), ("eta0_pct", "eta0")):
            if abs(pct[k]) < TOL:
                counts[short] += 1
            worst_abs[k] = max(worst_abs[k], abs(pct[k]))
            if r["admissible"]:
                worst_adm_abs[k] = max(worst_adm_abs[k], abs(pct[k]))
        if all(abs(pct[k]) < TOL for k in pct):
            all_three.append(j)
        if r["admissible"] and all(abs(pct[k]) < TOL for k in pct):
            all_three_adm.append(j)

    checks["recomputed_pass_counts_10pts"] = counts
    checks["reported_pass_counts_10pts"] = dict(
        case["canonical"]["summary"]["n_pass_by_quantity"])
    checks["points_all_three_pass"] = all_three
    checks["admissible_points_all_three_pass"] = all_three_adm
    checks["worst_abs_pct_all_10"] = worst_abs
    checks["worst_abs_pct_admissible_only"] = worst_adm_abs
    checks["max_abs_agreement_diff_vs_reported"] = max(
        v["max_abs_agreement_diff"] for v in recomp.values())
    checks["recompute_vs_reported"] = {"rows": recomp}

    # 5. the J=1.0 point: inadmissible everywhere, so it must not carry the verdict alone
    checks["J1_is_inadmissible"] = [r["admissible"] for r in rows if r["J"] == 1.0] == [False]
    checks["mean_abs_pct_excluding_J1"] = {
        k: sum(abs(r["error_pct"][k]) for r in rows if r["admissible"]) / sum(
            1 for r in rows if r["admissible"]) for k in ("KT_pct", "10KQ_pct", "eta0_pct")}

    # 6. artefacts: sizes, hashes, and the figure files being real PNGs
    arts = {}
    for p in sorted(SNAP.rglob("*")):
        if p.is_file():
            arts[str(p.relative_to(SNAP))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    checks["artefacts"] = arts
    checks["png_magic_ok"] = all(
        Path(SNAP, n).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        for n in arts if n.endswith(".png"))

    # 7. does the delivered environment pin a commit that contains the code?
    checks["artefact_git_commit"] = case["environment"]["git_commit"]

    checks["verdict_recomputed"] = (
        "FAIL" if not all_three and not all_three_adm else "PASS")
    checks["verdict_reported"] = case["verdict"]
    checks["verdict_agrees"] = checks["verdict_recomputed"] == case["verdict"]

    out = HERE / "verification.json"
    out.write_text(json.dumps(checks, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in checks.items()
                      if k not in ("artefacts",)}, indent=1, ensure_ascii=False, default=float))

    blockers = []
    if not checks["anchor_matches_pdf_all_10"]:
        blockers.append("anchor != PDF")
    if checks["verdict_recomputed"] != case["verdict"]:
        blockers.append("verdict disagrees")
    if checks["max_abs_agreement_diff_vs_reported"] > 1e-6:
        blockers.append("reported pct != recomputed pct")
    print("\nBLOCKERS:", blockers or "none")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())