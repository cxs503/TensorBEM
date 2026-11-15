#!/usr/bin/env python3
"""Open-water anchor registration, ``eta_0`` identity closure and ``eta_R`` convention audit.

Mission m21514 (wave 30, ``propeller-openwater-bem``).  The wave audit of the
propulsion block reported two defects that this module exists to adjudicate with
numbers instead of prose:

1. **efficiency identity** -- is ``eta_0 = J K_T / (2 pi K_Q)`` actually satisfied
   point by point on the *registered* anchor tables, or only asserted?
2. **``eta_R`` convention** -- 9 of 11 propulsion states reported by the platform
   leave the physically admissible window ``eta_R in [0.90, 1.10]``.  Which
   convention produces that, and is the platform's ``eta_R`` a computed quantity
   at all?

What the module produces
------------------------
``papers/artifacts/w30/propeller-openwater-bem/openwater_efficiency_closure.json``
    * ``anchors`` -- two open-water anchors, each with ``>= 5`` advance ratios,
      a machine-extracted quotation and the sha256 of the in-worktree source
      carrier: (a) Oosterveld & van Oossanen (1975) 39/47-term regression
      evaluated at the B4-70 geometry (``LITERATURE_CLOSED_FORM``), (b) KP505
      Table 2 of Iftikhar & Takinaci (MARINE 2023) transcribed verbatim
      (``LITERATURE_TABLE``, secondary reproduction).
    * ``eta0_identity`` -- every anchor row recomputed through an independent
      code path (``J K_T / (2 pi (10 K_Q / 10))``) and compared with the
      registered efficiency; residual per row, ``<= 0.5 %`` gate.
    * ``solver_vs_anchor`` -- the repository's vortex-lattice open-water kernel
      run at the same advance ratios (B4-70 fine configuration and the KP505
      32-panel configuration), per-point ``dK_T``, ``d(10 K_Q)``, ``d eta_0``
      against the anchor, ``< 10 %`` gate, plus the explicit ``< 3 %`` answer
      for every row.
    * ``etaR_audit`` -- 11 propulsion states driven through the platform's
      ``calculations/propulsion`` and ``calculations/resistance`` endpoints
      (receipts archived verbatim in ``data/w30/propulsion_states/
      api_receipts.json``), with three quantities per state: the returned
      ``relative_rot_eff``, the closure-implied ``eta_D / (eta_H eta_O)``, and
      the cross-module implied ``(P_E^resistance / P_D) / (eta_H eta_O)``; each
      is judged against ``[0.90, 1.10]``.

Honesty rules baked in
----------------------
* No anchor digit is typed here.  The Oosterveld & van Oossanen coefficients are
  read from ``benchmarks/anchors/wageningen_b_series.json`` (39 KT terms, 47 KQ
  terms) and the KP505 rows are parsed out of the in-worktree transcript of the
  paper's Table 2, so the quotation and the numbers come from the same bytes.
* The printed NSMB tables are *not* in this worktree; that anchor is labelled
  ``LITERATURE_CLOSED_FORM`` with ``quote_status =
  PUBLISHED_TABLE_NOT_IN_WORKTREE`` rather than pretending to be a digitised
  table.
* The 11-state grid is this mission's own grid (single hull, speed sweep plus
  rpm sweep plus one P/D point); the wave audit's state list is not held in this
  worktree, so the audit count is reported as *reproduced-in-count*, not as the
  same states.

Run::

    PYTHONPATH=src python -m benchmarks.anchors.build_openwater_efficiency_closure
    # or: PYTHONPATH=src python benchmarks/anchors/build_openwater_efficiency_closure.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
ANCHOR_DIR = REPO / "benchmarks" / "anchors"
SOURCE_DIR = ANCHOR_DIR / "sources"
OUT_DIR = REPO / "papers" / "artifacts" / "w30" / "propeller-openwater-bem"
FIG_DIR = REPO / "figures" / "w30"
RECEIPTS = REPO / "data" / "w30" / "propulsion_states" / "api_receipts.json"

TWO_PI = 2.0 * math.pi

#: gates, fixed before the run
ETA0_RESIDUAL_MAX_PCT = 0.5     # criterion 2
SOLVER_DEVIATION_MAX_PCT = 10.0  # criterion 3
TIGHT_PCT = 3.0                  # criterion 5 ("is it < 3 %?")
ETAR_WINDOW = (0.90, 1.10)       # criterion 4

#: B4-70 geometry of the Oosterveld & van Oossanen closed form
OVO_GEOMETRY = {"Z": 4, "Ae_Ao": 0.70, "P_D": 1.00}
OVO_J = [round(0.1 * k, 10) for k in range(1, 9)]   # 8 points, J = 0.1 .. 0.8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:                                        # pragma: no cover
        return "unknown"


def rel_pct(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference) or abs(reference) < 1e-12:
        return float("nan")
    return 100.0 * (value - reference) / reference


def ovo_kt(k_t: Sequence[Sequence[float]], advance: float, p_d: float, ae_ao: float, z: int) -> float:
    """39-term Oosterveld & van Oossanen thrust regression."""

    return float(
        sum(c * advance**s * p_d**t * ae_ao**u * z**v for c, s, t, u, v in k_t)
    )


def ovo_kq(k_q: Sequence[Sequence[float]], advance: float, p_d: float, ae_ao: float, z: int) -> float:
    """47-term Oosterveld & van Oossanen torque regression."""

    return float(
        sum(c * advance**s * p_d**t * ae_ao**u * z**v for c, s, t, u, v in k_q)
    )


def eta0_from(KT: float, KQ: float, advance: float) -> float:
    """The efficiency identity under test, written once and only once."""

    return advance * KT / (TWO_PI * KQ)


# ---------------------------------------------------------------------------
# anchor 1: Oosterveld & van Oossanen (1975), B4-70 closed form
# ---------------------------------------------------------------------------


def ovo_anchor() -> Dict[str, Any]:
    carrier = ANCHOR_DIR / "wageningen_b_series.json"
    blob = json.loads(carrier.read_text())
    kt_terms, kq_terms = blob["kt_terms"], blob["kq_terms"]
    if len(kt_terms) != 39 or len(kq_terms) != 47:
        raise SystemExit(
            f"term count changed: KT={len(kt_terms)} KQ={len(kq_terms)} (expected 39/47)"
        )

    # independent evaluation of the published closed form
    recomputed = {}
    for advance in OVO_J:
        kt = ovo_kt(kt_terms, advance, OVO_GEOMETRY["P_D"], OVO_GEOMETRY["Ae_Ao"], OVO_GEOMETRY["Z"])
        kq = ovo_kq(kq_terms, advance, OVO_GEOMETRY["P_D"], OVO_GEOMETRY["Ae_Ao"], OVO_GEOMETRY["Z"])
        recomputed[advance] = (kt, kq)

    # the registered values: the reference curve already carried by the anchor file
    curve = {round(row["J"], 10): row for row in blob["b4_70_pd1_curve"]["rows"]}
    rows: List[Dict[str, Any]] = []
    for advance in OVO_J:
        registered = curve[advance]
        kt, kq = recomputed[advance]
        registered_eta = float(registered["eta0"])
        independent_eta = eta0_from(float(registered["KT"]), float(registered["KQ"]), advance)
        rows.append(
            {
                "J": advance,
                "KT_registered": float(registered["KT"]),
                "KQ_registered": float(registered["KQ"]),
                "10KQ_registered": 10.0 * float(registered["KQ"]),
                "eta0_registered": registered_eta,
                "KT_independent": kt,
                "KQ_independent": kq,
                "10KQ_independent": 10.0 * kq,
                "eta0_independent": independent_eta,
                "eta0_residual_pct": rel_pct(independent_eta, registered_eta),
                "KT_reproduction_pct": rel_pct(kt, float(registered["KT"])),
                "KQ_reproduction_pct": rel_pct(kq, float(registered["KQ"])),
                "quote": blob["model"]["eta0"],
                "quote_status": "PUBLISHED_TABLE_NOT_IN_WORKTREE",
                "quote_note": (
                    "the printed NSMB Publication 479 tables are not reachable offline from this "
                    "worktree; the machine-extracted quotation is the closed form recorded with the "
                    "coefficients, and the coefficients themselves enter through the two-step chain "
                    "recorded in this anchor's provenance block"
                ),
            }
        )

    peak = max(rows, key=lambda r: r["eta0_registered"])
    return {
        "anchor_id": "AN-OVO-1975-B4.70-OPENWATER",
        "source_class": "LITERATURE_CLOSED_FORM",
        "citation": blob["citation"],
        "provenance_depth": blob["provenance"]["depth"],
        "provenance_chain": blob["provenance"]["chain"],
        "declared_limit": blob["provenance"]["declared_limit"],
        "geometry": dict(OVO_GEOMETRY),
        "geometry_validity": blob["domain"],
        "carrier": {
            "path": str(carrier.relative_to(REPO)),
            "sha256": sha256_file(carrier),
            "term_counts": {"KT": len(kt_terms), "KQ": len(kq_terms)},
            "declared_term_sha256": blob.get("fingerprints", {}),
        },
        "n_points": len(rows),
        "points": rows,
        "peak": {"J": peak["J"], "eta0": peak["eta0_registered"]},
    }


# ---------------------------------------------------------------------------
# anchor 2: KP505, Table 2 of Iftikhar & Takinaci (MARINE 2023)
# ---------------------------------------------------------------------------


def _parse_kp505_transcript() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse the in-worktree transcript of Table 2 into rows + verbatim lines."""

    path = SOURCE_DIR / "kp505_marine2023_p10_table2.txt"
    rows: List[Dict[str, Any]] = []
    verbatim: List[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) == 7 and all(_is_number(p) for p in parts):
            advance = float(parts[0])
            rows.append(
                {
                    "J": advance,
                    "KT_cfd": float(parts[1]),
                    "10KQ_cfd": float(parts[2]),
                    "KT": float(parts[3]),
                    "10KQ": float(parts[4]),
                    "source_line": stripped,
                }
            )
            verbatim.append(stripped)
        elif stripped.startswith("Table 2."):
            verbatim.append(stripped)
    return rows, verbatim


def _is_number(text: str) -> bool:
    try:
        float(text.rstrip("%"))
        return True
    except ValueError:
        return False


def kp505_anchor() -> Dict[str, Any]:
    carrier = ANCHOR_DIR / "kp505_openwater.json"
    blob = json.loads(carrier.read_text())
    rows_file, verbatim = _parse_kp505_transcript()
    if len(rows_file) < 5:
        raise SystemExit(f"transcript parse yielded {len(rows_file)} rows (need >= 5)")

    # the anchor file's own rows, keyed by J -- the registered efficiency
    registered = {round(row["J"], 10): row for row in blob["open_water_table"]}

    declared = blob["source"]
    transcript_path = SOURCE_DIR / "kp505_marine2023_p10_table2.txt"
    pdf_path = SOURCE_DIR / "kp505_marine2023_iftikhar.pdf"
    sha_actual = {
        "transcript_sha256": sha256_file(transcript_path),
        "pdf_sha256": sha256_file(pdf_path),
    }
    sha_checks = {
        "transcript_sha256_matches_declared": sha_actual["transcript_sha256"] == declared["transcript_sha256"],
        "pdf_sha256_matches_declared": sha_actual["pdf_sha256"] == declared["pdf_sha256"],
    }

    rows: List[Dict[str, Any]] = []
    for row in rows_file:
        key = round(row["J"], 10)
        reg = registered[key]
        if abs(reg["KT"] - row["KT"]) > 1e-12 or abs(reg["10KQ"] - row["10KQ"]) > 1e-12:
            raise SystemExit(
                f"transcript/anchor mismatch at J={key}: {row} vs anchor KT={reg['KT']} 10KQ={reg['10KQ']}"
            )
        registered_eta = float(reg["eta0"])
        independent_eta = eta0_from(float(reg["KT"]), float(reg["10KQ"]) / 10.0, key)
        rows.append(
            {
                "J": key,
                "KT": float(reg["KT"]),
                "10KQ": float(reg["10KQ"]),
                "KQ": float(reg["10KQ"]) / 10.0,
                "KT_cfd": row["KT_cfd"],
                "10KQ_cfd": row["10KQ_cfd"],
                "eta0_registered": registered_eta,
                "eta0_recomputed_registered_path": float(reg["eta0_recomputed"]),
                "eta0_independent": independent_eta,
                "eta0_residual_pct": rel_pct(independent_eta, registered_eta),
                "quote": row["source_line"],
                "quote_status": "VERBATIM_TRANSCRIPT_IN_WORKTREE",
            }
        )

    return {
        "anchor_id": "AN-KP505-2023-TABLE2",
        "source_class": "LITERATURE_TABLE",
        "citation": {
            "authors": "Iftikhar, H. & Takinaci, A.C.",
            "year": 2023,
            "title": "CFD analysis of the KP505 propeller in open water (MARINE 2023)",
            "locator": "Table 2, page 10",
            "columns_used": "Experimental KT, 10KQ",
        },
        "provenance_depth": "SECONDARY_REPRODUCTION",
        "declared_limit": (
            blob["verdict"]["honest_notes"][0]
            if blob.get("verdict", {}).get("honest_notes")
            else "secondary reproduction of a paper's experimental column"
        ),
        "carrier": {
            "anchor_json": str(carrier.relative_to(REPO)),
            "anchor_json_sha256": sha256_file(carrier),
            "transcript": str(transcript_path.relative_to(REPO)),
            **sha_actual,
            "declared_sha256": {
                "transcript_sha256": declared["transcript_sha256"],
                "pdf_sha256": declared["pdf_sha256"],
            },
            "sha256_checks": sha_checks,
        },
        "header_quotes": [line for line in verbatim if not _is_number(line.split()[0])],
        "n_points": len(rows),
        "points": rows,
    }


# ---------------------------------------------------------------------------
# solver-versus-anchor table
# ---------------------------------------------------------------------------


def solver_b4_70(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from tensorbem.benchmarks.b4_70_error_surface import B4_70_FINE, solve  # noqa: WPS433

    rows = []
    for anchor in points:
        advance = anchor["J"]
        result = solve(B4_70_FINE, advance)
        rows.append(
            _comparison_row(
                case="B4-70 lattice (B4_70_FINE)",
                advance=advance,
                KT=result["KT"],
                KQ=result["KQ"],
                eta0=result["eta0"],
                anchor=anchor,
                anchor_KT=anchor["KT_registered"],
                anchor_10KQ=anchor["10KQ_registered"],
                anchor_eta0=anchor["eta0_registered"],
                extra={
                    "n_radial": B4_70_FINE.n_radial,
                    "n_chord": B4_70_FINE.n_chord,
                    "wake_model": B4_70_FINE.wake_model,
                    "wake_length": B4_70_FINE.wake_length,
                    "wake_steps": B4_70_FINE.wake_steps,
                    "rev_per_second": result["rev_per_second"],
                },
            )
        )
    return rows


def solver_kp505(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """KP505 open-water case at the module's canonical (registered) configuration.

    The overrides are deliberately left alone: ``_geometry()`` defaults (16 radial
    x 2 chord = 32 panels per blade) are the configuration whose points are already
    registered under ``benchmarks/kp505_openwater_bem/points.csv``, so this table
    is comparable with the repository's earlier KP505 evidence instead of being a
    new, unexplained mesh.
    """

    from tensorbem.benchmarks.kp505_openwater_bem import _geometry, _point  # noqa: WPS433

    geometry = _geometry()
    rows = []
    for anchor in points:
        advance = anchor["J"]
        result = _point(advance)
        kq = -float(result.torque) if float(result.torque) < 0.0 else float(result.torque)
        eta0 = eta0_from(float(result.thrust), kq, advance)
        rows.append(
            _comparison_row(
                case="KP505 vortex lattice (32 panels/blade)",
                advance=advance,
                KT=float(result.thrust),
                KQ=kq,
                eta0=eta0,
                anchor=anchor,
                anchor_KT=anchor["KT"],
                anchor_10KQ=anchor["10KQ"],
                anchor_eta0=anchor["eta0_registered"],
                extra={
                    "n_radial": geometry.n_radial,
                    "n_chord": geometry.n_chord,
                    "blades": geometry.blades,
                    "pitch_ratio": geometry.pitch_ratio,
                    "expanded_area_ratio": geometry.expanded_area_ratio,
                    "n_panels": int(getattr(result, "n_panels", 0)),
                },
            )
        )
    return rows


def _comparison_row(
    *,
    case: str,
    advance: float,
    KT: float,
    KQ: float,
    eta0: float,
    anchor: Dict[str, Any],
    anchor_KT: float,
    anchor_10KQ: float,
    anchor_eta0: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    dkt = rel_pct(KT, anchor_KT)
    d10kq = rel_pct(10.0 * KQ, anchor_10KQ)
    deta = rel_pct(eta0, anchor_eta0)
    return {
        "case": case,
        "J": advance,
        "KT_solver": KT,
        "KT_anchor": anchor_KT,
        "dKT_pct": dkt,
        "10KQ_solver": 10.0 * KQ,
        "10KQ_anchor": anchor_10KQ,
        "d10KQ_pct": d10kq,
        "eta0_solver": eta0,
        "eta0_anchor": anchor_eta0,
        "deta0_pct": deta,
        "within_10pct": bool(abs(dkt) < SOLVER_DEVIATION_MAX_PCT and abs(d10kq) < SOLVER_DEVIATION_MAX_PCT),
        "within_3pct_dKT": bool(abs(dkt) < TIGHT_PCT),
        "within_3pct_d10KQ": bool(abs(d10kq) < TIGHT_PCT),
        "within_3pct_deta0": bool(abs(deta) < TIGHT_PCT) if math.isfinite(deta) else False,
        "solver": extra,
    }


# ---------------------------------------------------------------------------
# eta_R convention audit
# ---------------------------------------------------------------------------


def etar_audit(receipts_path: Path = RECEIPTS) -> Dict[str, Any]:
    blob = json.loads(receipts_path.read_text())
    resistance = {row["speed_kn"]: row for row in blob["resistance_receipts"]["rows"]}
    rows: List[Dict[str, Any]] = []

    for state in blob["states"]:
        speed = state["inputs"]["speed_kn"]
        response = state["propulsion_response"]
        pe_res = float(resistance[speed]["pe_kW"])
        pd_prop = float(response["delivered_power"])
        eta_h = float(response["hull_eff"])
        eta_r = float(response["relative_rot_eff"])
        eta_o = float(response["open_water_eff"])
        eta_d = float(response["propulsive_eff"])
        advance = float(response["advance_ratio_J"])
        kt = float(response["thrust_coefficient_KT"])
        kq = float(response["torque_coefficient_KQ"])

        eta0_recomputed = eta0_from(kt, kq, advance) if kq > 0.0 else float("nan")
        eta_d_implied = pe_res / pd_prop
        eta_r_identity = eta_d / (eta_h * eta_o) if eta_h * eta_o > 0.0 else float("nan")
        eta_r_cross_module = eta_d_implied / (eta_h * eta_o) if eta_h * eta_o > 0.0 else float("nan")

        low, high = ETAR_WINDOW
        rows.append(
            {
                "state": state["id"],
                "inputs": state["inputs"],
                "J": advance,
                "KT": kt,
                "KQ": kq,
                "eta0_returned": eta_o,
                "eta0_recomputed_from_KT_KQ": eta0_recomputed,
                "eta0_residual_pct": rel_pct(eta0_recomputed, eta_o),
                "etaH_returned": eta_h,
                "etaR_returned": eta_r,
                "etaD_returned": eta_d,
                "etaD_identity_residual_pct": rel_pct(eta_h * eta_r * eta_o, eta_d),
                "pe_resistance_kW": pe_res,
                "pd_propulsion_kW": pd_prop,
                "pe_propulsion_kW": float(response["effective_power_pe"]),
                "pe_module_disagreement_pct": rel_pct(pe_res, float(response["effective_power_pe"])),
                "etaD_implied_pe_over_pd": eta_d_implied,
                "etaR_identity_implied": eta_r_identity,
                "etaR_cross_module_implied": eta_r_cross_module,
                "etaR_returned_in_window": bool(low <= eta_r <= high),
                "etaR_identity_in_window": bool(low <= eta_r_identity <= high),
                "etaR_cross_module_in_window": bool(low <= eta_r_cross_module <= high),
            }
        )

    n = len(rows)
    out_returned = sum(1 for r in rows if not r["etaR_returned_in_window"])
    out_cross = sum(1 for r in rows if not r["etaR_cross_module_in_window"])
    distinct_returned_etar = sorted({round(r["etaR_returned"], 12) for r in rows})
    distinct_returned_etah = sorted({round(r["etaH_returned"], 12) for r in rows})
    worst_eta0 = max(rows, key=lambda r: abs(r["eta0_residual_pct"]) if math.isfinite(r["eta0_residual_pct"]) else -1.0)

    return {
        "grid_definition": {
            "hull": blob["hull"],
            "states": "11 states: S1..S6 speed sweep at 100 rpm (12..17 kn); S7..S10 rpm sweep at 15 kn (85/90/95/105 rpm); S11 P/D 1.05 at 15 kn / 100 rpm",
            "predeclared_before_calls": True,
            "note": (
                "this is THIS mission's grid.  The wave audit's own 11-state list is not held in "
                "this worktree, so the audit count is reported as a reproduction in count, not as "
                "the identification of the same states"
            ),
        },
        "receipts": {
            "path": str(receipts_path.relative_to(REPO)),
            "sha256": sha256_file(receipts_path),
            "endpoint_base": blob["endpoint_base"],
            "resistance_method": blob["resistance_receipts"]["method"],
            "precision_note": blob["transport"],
        },
        "states": rows,
        "summary": {
            "n_states": n,
            "n_etaR_returned_out_of_window": out_returned,
            "n_etaR_cross_module_out_of_window": out_cross,
            "distinct_etaR_returned": distinct_returned_etar,
            "distinct_etaH_returned": distinct_returned_etah,
            "max_abs_eta0_residual_pct": abs(worst_eta0["eta0_residual_pct"]),
            "worst_eta0_state": worst_eta0["state"],
            "verdict": (
                "the platform's relative_rot_eff is a CONSTANT equal to "
                f"{distinct_returned_etar[0]:.4f} on all {n} states (and hull_eff is the "
                f"constant {distinct_returned_etah[0]:.4f}); the multiplicative identity "
                "eta_D = eta_H eta_R eta_O therefore closes to rounding error by construction, "
                "which is exactly why it has no diagnostic power.  Inverting the identity against "
                "the resistance module's independent P_E puts "
                f"{out_cross}/{n} states outside [0.90, 1.10]"
            ),
        },
    }


# ---------------------------------------------------------------------------
# assembled artefact
# ---------------------------------------------------------------------------


def build(outdir: Path = OUT_DIR) -> Dict[str, Any]:
    ovo = ovo_anchor()
    kp505 = kp505_anchor()

    b4_rows = solver_b4_70(ovo["points"])
    kp_rows = solver_kp505(kp505["points"])

    audit = etar_audit()

    eta0_rows = [
        {"anchor": "AN-OVO-1975-B4.70-OPENWATER", **{k: r[k] for k in ("J", "eta0_registered", "eta0_independent", "eta0_residual_pct")}}
        for r in ovo["points"]
    ] + [
        {"anchor": "AN-KP505-2023-TABLE2", **{k: r[k] for k in ("J", "eta0_registered", "eta0_independent", "eta0_residual_pct")}}
        for r in kp505["points"]
    ]
    worst_eta0 = max(eta0_rows, key=lambda r: abs(r["eta0_residual_pct"]))

    criteria = [
        {
            "id": "C1-anchors-registered",
            "statement": ">= 5 advance ratios per anchor with a machine-extracted quotation and a verifiable sha256",
            "measured": {
                "anchors": {
                    ovo["anchor_id"]: {"n_points": ovo["n_points"], "sha256": ovo["carrier"]["sha256"], "quote_status": ovo["points"][0]["quote_status"]},
                    kp505["anchor_id"]: {"n_points": kp505["n_points"], "sha256": kp505["carrier"]["transcript_sha256"], "quote_status": kp505["points"][0]["quote_status"]},
                },
                "kp505_sha256_matches_declared": all(kp505["carrier"]["sha256_checks"].values()),
            },
            "passed": bool(ovo["n_points"] >= 5 and kp505["n_points"] >= 5 and all(kp505["carrier"]["sha256_checks"].values())),
        },
        {
            "id": "C2-eta0-identity-residual",
            "statement": f"|eta0_recomputed - eta0_registered| / eta0_registered <= {ETA0_RESIDUAL_MAX_PCT} % on every anchor row",
            "measured": {
                "n_rows": len(eta0_rows),
                "max_abs_residual_pct": abs(worst_eta0["eta0_residual_pct"]),
                "worst_row": worst_eta0,
            },
            "passed": bool(abs(worst_eta0["eta0_residual_pct"]) <= ETA0_RESIDUAL_MAX_PCT),
        },
        {
            "id": "C3-solver-versus-anchor",
            "statement": f"per-point |dKT| and |d(10KQ)| < {SOLVER_DEVIATION_MAX_PCT} % against the anchor",
            "measured": {
                "b4_70": {
                    "n_points": len(b4_rows),
                    "max_abs_dKT_pct": max(abs(r["dKT_pct"]) for r in b4_rows),
                    "max_abs_d10KQ_pct": max(abs(r["d10KQ_pct"]) for r in b4_rows),
                    "worst_dKT": max(b4_rows, key=lambda r: abs(r["dKT_pct"]))["J"],
                },
                "kp505": {
                    "n_points": len(kp_rows),
                    "max_abs_dKT_pct": max(abs(r["dKT_pct"]) for r in kp_rows),
                    "max_abs_d10KQ_pct": max(abs(r["d10KQ_pct"]) for r in kp_rows),
                    "worst_dKT": max(kp_rows, key=lambda r: abs(r["dKT_pct"]))["J"],
                },
                "n_points_over_10pct": {
                    "b4_70": [r["J"] for r in b4_rows if not r["within_10pct"]],
                    "kp505": [r["J"] for r in kp_rows if not r["within_10pct"]],
                },
            },
            "passed": bool(
                all(r["within_10pct"] for r in b4_rows) and all(r["within_10pct"] for r in kp_rows)
            ),
        },
        {
            "id": "C4-etar-window",
            "statement": "every propulsion state's eta_R judged against [0.90, 1.10]; out-of-window is physically inadmissible and must be reported",
            "measured": audit["summary"],
            "passed": bool(audit["summary"]["n_etaR_cross_module_out_of_window"] == 0),
        },
        {
            "id": "C5-under-3pct",
            "statement": "explicit per-row answer to 'is it < 3 %?'",
            "measured": {
                "b4_70_rows_under_3pct": [
                    {"J": r["J"], "dKT": r["within_3pct_dKT"], "d10KQ": r["within_3pct_d10KQ"], "deta0": r["within_3pct_deta0"]}
                    for r in b4_rows
                ],
                "kp505_rows_under_3pct": [
                    {"J": r["J"], "dKT": r["within_3pct_dKT"], "d10KQ": r["within_3pct_d10KQ"], "deta0": r["within_3pct_deta0"]}
                    for r in kp_rows
                ],
                "eta0_identity_under_3pct_all_rows": bool(all(abs(r["eta0_residual_pct"]) < TIGHT_PCT for r in eta0_rows)),
                "n_solver_rows_under_3pct": {
                    "b4_70": sum(1 for r in b4_rows if r["within_3pct_dKT"] and r["within_3pct_d10KQ"]),
                    "kp505": sum(1 for r in kp_rows if r["within_3pct_dKT"] and r["within_3pct_d10KQ"]),
                },
            },
            "passed": None,
        },
    ]

    artefact = {
        "artifact_id": "openwater_efficiency_closure_w30",
        "mission": 21514,
        "topic": "propeller-openwater-bem: anchor registration, eta0 identity closure, eta_R convention audit",
        "generated_by": "benchmarks/anchors/build_openwater_efficiency_closure.py",
        "regenerate": "PYTHONPATH=src python benchmarks/anchors/build_openwater_efficiency_closure.py",
        "git_commit": git_commit(),
        "gates": {
            "eta0_residual_max_pct": ETA0_RESIDUAL_MAX_PCT,
            "solver_deviation_max_pct": SOLVER_DEVIATION_MAX_PCT,
            "tight_pct": TIGHT_PCT,
            "etar_window": list(ETAR_WINDOW),
            "predeclared": "fixed at the top of the module before the sweep was run; not retuned after the numbers were seen",
        },
        "anchors": [ovo, kp505],
        "eta0_identity": {
            "formula": "eta0 = J * KT / (2 * pi * KQ)  with  KQ = 10KQ / 10",
            "n_rows": len(eta0_rows),
            "max_abs_residual_pct": abs(worst_eta0["eta0_residual_pct"]),
            "rows": eta0_rows,
        },
        "solver_vs_anchor": {"b4_70": b4_rows, "kp505": kp_rows},
        "etaR_audit": audit,
        "criteria": criteria,
        "tensor_reports": [
            {
                "id": 19430,
                "title": "W30 敞水性能锚值核销与效率恒等回归：η0 恒等 ~1e-14 %，ηR 9/11 越物理窗",
                "note": "posted from benchmarks/anchors/publish_openwater_closure_w30.py; every number in it is injected from this artefact",
            }
        ],
        "honest_notes": [
            "The Oosterveld & van Oossanen anchor is a CLOSED-FORM anchor: the printed NSMB tables are not in this worktree, so no table digit is claimed.  What is claimed is that the published 39/47-term regression, fed the machine-chain coefficients already registered in benchmarks/anchors/wageningen_b_series.json, reproduces the registered B4-70 curve and satisfies the efficiency identity.",
            "The KP505 anchor is a LITERATURE_TABLE anchor but a SECONDARY_REPRODUCTION: the digits are the 'Experimental' columns of Table 2 of Iftikhar & Takinaci (MARINE 2023), not the KRISO handout; the paper does not state where its experimental column came from.",
            "The eta_R window judgement on the platform path is possible at all only because the resistance module and the propulsion module are independent code paths; without that second P_E the identity is a tautology (see etaR_audit.summary).",
            "Receipt precision: the archived endpoint fields carry 2 dp (kW) / 4 dp (dimensionless), which is what bounds the identity residuals at ~1e-3, not a solver effect.",
        ],
        "verdict": {
            "all_criteria_pass": all(c["passed"] for c in criteria if c["passed"] is not None),
            "c5_is_informational": True,
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "openwater_efficiency_closure.json"
    target.write_text(json.dumps(artefact, indent=2, ensure_ascii=False) + "\n")

    _write_csv(outdir / "solver_vs_anchor.csv", b4_rows + kp_rows)
    _write_csv(outdir / "etar_audit.csv", audit["states"])
    _write_csv(outdir / "eta0_identity.csv", eta0_rows)
    artefact["artifacts"] = {
        "json": str(target.relative_to(REPO)),
        "csv": [
            str((outdir / name).relative_to(REPO))
            for name in ("solver_vs_anchor.csv", "etar_audit.csv", "eta0_identity.csv")
        ],
    }
    target.write_text(json.dumps(artefact, indent=2, ensure_ascii=False) + "\n")
    return artefact


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in row.items()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    artefact = build(Path(args.outdir))
    print(json.dumps({"artifact": artefact["artifacts"]["json"], "verdict": artefact["verdict"],
                      "criteria": [{c["id"]: c["passed"]} for c in artefact["criteria"]],
                      "etaR": artefact["etaR_audit"]["summary"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())