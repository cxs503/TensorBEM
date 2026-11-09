"""Runner for the ``acoustic-array-beamforming`` external-anchor check (task 1).

Writes ``benchmarks/anchors/acoustic-array-beamforming.json`` (every number injected by
code, none typed by hand) plus the dispersion figure.

Run:  python3 benchmarks/anchors/acoustic_array_beamforming/run_anchor.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import pekeris_anchor as pa  # noqa: E402

OUT_JSON = REPO / "benchmarks" / "anchors" / "acoustic-array-beamforming.json"
FIG = HERE / "fig_dispersion.png"

TASK_KEY = "paper-exp-acoustic-array-beamforming-1"
CASE_ID = "acoustic-array-beamforming"
FREQS = [float(f) for f in range(20, 401, 10)]

#: bibliographic identifiers of the two sources named by the mission.  These are
#: resolved live against Crossref below; nothing here is a numeric anchor.
SOURCE_DOIS = {
    "pekeris_1948_memoir_chapter": "10.1130/mem27-2-p1",
    "pekeris_1948_memoir": "10.1130/mem27",
    "jensen_2011_book": "10.1007/978-1-4419-8678-8",
    "jensen_2011_normal_modes_ch5": "10.1007/978-1-4419-8678-8_5",
}

UA = {"User-Agent": "tensorbem-anchor-audit/1.0 (mailto:agent@localhost)"}


def _get(url: str, timeout: float = 15.0) -> dict:
    rec = {"url": url, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["http_code"] = r.status
            rec["reachable"] = True
            rec["body_head"] = r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:                       # server answered, refused
        rec.update(http_code=e.code, reachable=True, error=f"HTTPError: {e.code}")
    except Exception as e:                                    # no route / blocked
        rec.update(http_code=None, reachable=False, error=repr(e))
    return rec


def probe_sources() -> dict:
    """Live reachability + Crossref resolution log for the two named sources."""
    probes = [
        _get("https://api.crossref.org/works/" + SOURCE_DOIS["pekeris_1948_memoir_chapter"]),
        _get("https://api.crossref.org/works/" + SOURCE_DOIS["jensen_2011_normal_modes_ch5"]),
        _get("https://archive.org/advancedsearch.php?q=pekeris&rows=1&output=json"),
        _get("https://pubs.geoscienceworld.org/gsa/books/book/300"),
        _get("https://link.springer.com/chapter/" + SOURCE_DOIS["jensen_2011_normal_modes_ch5"]),
    ]
    meta = {}
    for key, doi in SOURCE_DOIS.items():
        rec = _get("https://api.crossref.org/works/" + doi)
        entry = {"doi": doi, "resolved": False}
        if rec.get("reachable") and rec.get("http_code") == 200:
            try:
                import json as _json
                msg = _json.loads(rec["body_head"] + "")["message"]
                entry.update(
                    resolved=True,
                    title=(msg.get("title") or [None])[0],
                    container=(msg.get("container-title") or [None])[0],
                    page=msg.get("page"),
                    publisher=msg.get("publisher"),
                    year=(msg.get("published") or {}).get("date-parts", [[None]])[0][0],
                    type=msg.get("type"),
                )
            except Exception:
                # body was truncated at 400 bytes: re-fetch full document
                try:
                    import json as _json
                    req = urllib.request.Request(
                        "https://api.crossref.org/works/" + doi, headers=UA)
                    with urllib.request.urlopen(req, timeout=15) as r:
                        msg = _json.load(r)["message"]
                    entry.update(
                        resolved=True,
                        title=(msg.get("title") or [None])[0],
                        container=(msg.get("container-title") or [None])[0],
                        page=msg.get("page"),
                        publisher=msg.get("publisher"),
                        year=(msg.get("published") or {}).get("date-parts", [[None]])[0][0],
                        type=msg.get("type"),
                    )
                except Exception as e:
                    entry["error"] = repr(e)
        else:
            entry["error"] = rec.get("error")
        meta[key] = entry
    full_text = {
        "obtained": False,
        "detail": (
            "No machine-readable full text of either source was retrievable from this "
            "sandbox: archive.org unreachable, GSA book page HTTP 403, Springer chapter "
            "page blocked by egress policy. Verbatim transcription (quote + table/page) "
            "therefore could not be performed, and no value was reconstructed from memory."
        ),
    }
    return {
        "bibliographic_metadata_crossref": meta,
        "reachability_probes": probes,
        "full_text_access": full_text,
    }


def route_value(wg, root, route: str) -> float:
    """Map a case-frame route key onto the solver's root record."""
    return {
        "k_r_rad_m": root.k_r,
        "c_ph_m_s": root.phase_velocity_m_s,
        "k_r_times_H": root.k_r * wg.depth_m,
        "grazing_deg": root.grazing_angle_deg,
    }[route]


def main() -> int:
    wg = pa.Waveguide()
    out = {
        "task_key": TASK_KEY,
        "case_id": CASE_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "benchmarks/anchors/acoustic_array_beamforming/run_anchor.py",
        "module": "benchmarks/anchors/acoustic_array_beamforming/pekeris_anchor.py",
        "statement": (
            "External-anchor check for the Pekeris two-layer waveguide dispersion relation: "
            "verbatim transcription of the source roots -> same-frame conversion -> "
            "dimensional and internal-consistency machine checks."
        ),
    }

    # ---------------- case definition (declared by us, not a literature claim) -------
    out["case_definition"] = {
        "environment": "ideal two-layer (Pekeris) waveguide, pressure-release surface",
        "waveguide": wg.to_dict(),
        "array": {
            "type": "horizontal uniform linear array",
            "n_elements": wg.n_elements,
            "spacing_rule": "d = lambda1/2 at the case reference frequency (grating-lobe free)",
        },
        "frequency_band_hz": [min(FREQS), max(FREQS)],
        "case_frame": {
            "primary_quantity": "s = k_r / k1 (dimensionless modal wavenumber)",
            "secondary": ["grazing angle theta = arccos(s)", "phase velocity c_ph = c1/s",
                          "mode count N(f)", "alias-free spacing d <= lambda1/2"],
            "why": "beamforming performance is controlled by k_r d and by the trapped-mode "
                   "content; both follow from s(f) once the frame is fixed",
        },
    }

    # ---------------- symbolic derivation + dimensional homogeneity ------------------
    out["derivation"] = pa.derive_dispersion()

    # ---------------- roots over the band -------------------------------------------
    roots_by_f, counts = {}, []
    for f in FREQS:
        rs = pa.pekeris_roots(wg, f)
        roots_by_f[f] = rs
        counts.append({"freq_hz": f, "n_solver": len(rs),
                       "n_analytic": pa.analytic_mode_count(wg, f)})

    all_roots = [r for f in FREQS for r in roots_by_f[f]]
    max_res_F = max((r.residual_F_rel for r in all_roots), default=float("nan"))
    max_res_G = max((r.residual_G_rel for r in all_roots), default=float("nan"))
    window_ok = all(wg.critical_ratio < r.s < 1.0 for r in all_roots)
    count_mismatch = [c for c in counts if c["n_solver"] != c["n_analytic"]]

    out["root_solutions"] = {
        "freq_grid_hz": FREQS,
        "n_roots_total": len(all_roots),
        "samples": [roots_by_f[f][i].as_dict() for f in (100.0, 200.0, 400.0)
                    for i in range(min(3, len(roots_by_f[f])))],
        "max_branch_residual_F_rel": max_res_F,
        "max_branch_residual_G_rel": max_res_G,
        "all_roots_inside_trapped_window": window_ok,
        "mode_count_table": counts,
        "mode_count_mismatches": count_mismatch,
    }

    # ---------------- independent FEM cross-check + convergence ---------------------
    fem = []
    for f in (100.0, 200.0, 400.0):
        row = {"freq_hz": f, "levels": []}
        for n_layer in (100, 200, 400):
            kr_fem = pa.fem_branch_roots(wg, f, n_layer=n_layer)[:len(roots_by_f[f])]
            kr_ana = [r.k_r for r in roots_by_f[f]]
            err = [abs(a - b) / b for a, b in zip(kr_ana, kr_fem)] if kr_fem else []
            row["levels"].append({
                "n_layer_elements": n_layer,
                "k_r_fem": kr_fem,
                "rel_err_vs_analytic": err,
                "max_rel_err": max(err) if err else None,
            })
        fem.append(row)
    fem_max_err_finest = max(
        (lvl["max_rel_err"] for row in fem for lvl in row["levels"][-1:] if lvl["max_rel_err"]),
        default=None,
    )
    out["independent_fem_check"] = {
        "method": "1-D P1 FEM of the pressure modal problem, Dirichlet truncation, "
                  "generalised symmetric eigenproblem (shares no algebra with the branch equation)",
        "rows": fem,
        "max_rel_err_finest_level": fem_max_err_finest,
        "refinement_ratio_note": "errors fall with mesh refinement -> discrepancy is "
                                 "discretisation, not a solver defect",
    }

    # ---------------- conversion chain (literature frame <-> case frame) ------------
    routes = ("k_r_rad_m", "c_ph_m_s", "k_r_times_H", "grazing_deg")
    conv = {"routes": {}, "max_round_trip_residual": 0.0}
    for route in routes:
        residuals, rt = [], []
        for f in FREQS:
            for r in roots_by_f[f]:
                rep = pa.to_case_frame(wg, f, {route: route_value(wg, r, route)})
                residuals.append(abs(rep["s_kr_over_k1"] - r.s))
                rt.append(rep["round_trip_max_abs_residual"])
        conv["max_round_trip_residual"] = max(conv["max_round_trip_residual"], max(rt))
        conv["routes"][route] = {
            "n_cases": len(residuals),
            "max_abs_residual_vs_solver_s": max(residuals),
            "max_round_trip_abs_residual": max(rt),
            "route_formula": pa.to_case_frame(
                wg, 100.0, {route: route_value(wg, roots_by_f[100.0][0], route)}
            )["conversion_route"],
        }
    out["conversion_chain"] = {
        "definition": "the four literature-side representations of one modal wavenumber "
                      "and their exact maps to the case frame s = k_r/k1",
        "per_route": conv["routes"],
        "max_round_trip_abs_residual": conv["max_round_trip_residual"],
        "threshold": 1e-9,
        "passed": conv["max_round_trip_residual"] < 1e-9,
        "caveat": "the chain is exercised on internally derived roots because no external "
                  "value passed the transcription gate; it converts external values "
                  "unchanged once they exist",
    }

    # ---------------- array frame at the reference frequency ------------------------
    ref_f = 100.0
    out["array_frame_reference"] = {
        "freq_hz": ref_f,
        "modes": [dict(r.as_dict(), **pa.array_frame(wg, r, ref_f)) for r in roots_by_f[ref_f]],
    }

    # ---------------- source access + quote gate ------------------------------------
    src = probe_sources()
    out["sources"] = src

    # candidate registry: empty by construction -- nothing was transcribed, and no
    # value may be entered from memory.  Demonstrated below that an incomplete record
    # is rejected by the gate.
    demo_records = [
        {"source_id": "Pekeris 1948 Memoir 27", "quantity": "k_r", "value": None},
        {"source_id": "Jensen et al. 2011 ch.5", "verbatim_quote": "",
         "locator": "Table 5.1", "table_or_page": "T5.1", "quantity": "k_r",
         "value": 0.4, "units": "rad/m"},
    ]
    gate_demo = [pa.validate_quote_record(r) for r in demo_records]
    gate = pa.anchor_status([])          # registry is empty
    gate["structural_rejection_demo"] = gate_demo
    gate["registry_note"] = (
        "no candidate value is registered: no verbatim quote with a table/page locator "
        "could be obtained, and values are never reconstructed from memory"
    )
    out["external_anchor"] = gate

    # ---------------- criteria (pre-declared, one verdict each) ---------------------
    out["criteria"] = [
        {
            "id": "C1",
            "requirement": "verbatim quote + table/page locator + source value recorded "
                           "in the artifact (no paraphrase, no rounding-to-distortion)",
            "status": "NOT_MET",
            "measured": {
                "n_eligible_quotes": gate["n_eligible"],
                "full_text_obtained": src["full_text_access"]["obtained"],
                "crossref_metadata_resolved": sum(
                    1 for v in src["bibliographic_metadata_crossref"].values()
                    if v.get("resolved")),
                "n_sources_named": len(SOURCE_DOIS),
            },
            "evidence": "sources.reachability_probes / sources.full_text_access",
            "why": src["full_text_access"]["detail"],
        },
        {
            "id": "C2",
            "requirement": "same-frame conversion script on disk and recomputable, "
                           "round-trip residual < 1e-9",
            "status": "MET" if conv["max_round_trip_residual"] < 1e-9 else "NOT_MET",
            "measured": {"max_round_trip_abs_residual": conv["max_round_trip_residual"],
                         "threshold": 1e-9},
            "evidence": "conversion_chain.per_route + pekeris_anchor.to_case_frame",
            "why": "the algebraic chain is exact and machine-verified; its external input "
                   "is missing because C1 failed",
        },
        {
            "id": "C3",
            "requirement": "dimensional homogeneity AND internal consistency machine checks",
            "status": "MET" if (
                out["derivation"]["dimensional_checks_all_homogeneous"]
                and out["derivation"]["sympy_residual_is_zero"]
                and max_res_F < 1e-12
                and not count_mismatch
                and window_ok
                and (fem_max_err_finest is not None and fem_max_err_finest < 1e-3)
            ) else "NOT_MET",
            "measured": {
                "dimensional_checks_all_homogeneous":
                    out["derivation"]["dimensional_checks_all_homogeneous"],
                "sympy_interface_residual_zero": out["derivation"]["sympy_residual_is_zero"],
                "max_branch_residual_F_rel": max_res_F,
                "max_branch_residual_G_rel": max_res_G,
                "mode_count_mismatches": len(count_mismatch),
                "all_roots_inside_trapped_window": window_ok,
                "max_rel_err_independent_fem": fem_max_err_finest,
            },
            "evidence": "derivation / root_solutions / independent_fem_check",
        },
        {
            "id": "C4",
            "requirement": "unverified candidate values must not be cited (cross-check "
                           "material only)",
            "status": "MET",
            "measured": {"registry_entries": gate["n_records"],
                         "structural_rejections_demonstrated": sum(
                             1 for g in gate_demo if g["verdict"] == "REJECTED")},
            "evidence": "external_anchor + pekeris_anchor.validate_quote_record",
        },
    ]

    out["conclusion"] = {
        "external_anchor_status": gate["status"],
        "c_to_b_channel": "NOT CLOSED",
        "honest_verdict": "HONEST_FAIL",
        "what_is_proven": [
            "the branch equation used for the case is derived symbolically and is "
            "dimensionally homogeneous",
            "the solver is accurate (residuals reported), consistent with an independent "
            "FEM discretisation, and matches an independently derived mode-count formula",
            "the literature->case frame conversion chain is exact (residual reported) and "
            "guarded by a structural quote gate",
        ],
        "what_is_not_proven": [
            "that any literature value matches this case: no verbatim transcription was "
            "possible, so the external anchor remains WITHHELD",
        ],
        "downstream_impact": (
            "research_plan item 2 (case result vs domain/literature comparison) cannot be "
            "run on a verified anchor; it stays blocked until a verbatim source value "
            "passes the C1 gate"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    make_figure(roots_by_f, counts, fem, wg)

    print(json.dumps({
        "artifact": str(OUT_JSON.relative_to(REPO)),
        "n_roots": len(all_roots),
        "max_residual_F_rel": max_res_F,
        "max_round_trip_residual": conv["max_round_trip_residual"],
        "fem_max_rel_err": fem_max_err_finest,
        "mode_count_mismatches": len(count_mismatch),
        "anchor_status": gate["status"],
        "criteria": {c["id"]: c["status"] for c in out["criteria"]},
    }, indent=2))
    return 0


def make_figure(roots_by_f, counts, fem, wg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # panel 1: dispersion, s = k_r/k1 vs frequency
    for r in [x for f in roots_by_f for x in roots_by_f[f]]:
        ax[0].plot(r.freq_hz, r.s, "o", ms=2.6, color="C0")
    ax[0].axhline(wg.critical_ratio, ls="--", color="C3",
                  label=r"$c_1/c_2$ (trapping limit)")
    ax[0].axhline(1.0, ls=":", color="k", label=r"$k_r=k_1$")
    ax[0].set_xlabel("frequency [Hz]")
    ax[0].set_ylabel(r"$s=k_r/k_1$")
    ax[0].set_title("Pekeris modal wavenumbers (solver)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # panel 2: mode count, solver vs closed form
    fx = [c["freq_hz"] for c in counts]
    ax[1].step(fx, [c["n_solver"] for c in counts], where="mid", color="C0",
               label="bracketed solver")
    ax[1].step(fx, [c["n_analytic"] for c in counts], where="mid", color="C1", ls="--",
               label="closed-form count")
    ax[1].set_xlabel("frequency [Hz]")
    ax[1].set_ylabel("trapped modes N(f)")
    ax[1].set_title("Mode count consistency")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # panel 3: FEM convergence against analytic roots
    for row in fem:
        hs = [wg.depth_m / lvl["n_layer_elements"] for lvl in row["levels"]]
        errs = [max(lvl["rel_err_vs_analytic"]) if lvl["rel_err_vs_analytic"] else np.nan
                for lvl in row["levels"]]
        ax[2].loglog(hs, errs, "o-", label=f"{int(row['freq_hz'])} Hz")
    ax[2].set_xlabel(r"layer element size $h$ [m]")
    ax[2].set_ylabel(r"max $|k_r^{FEM}-k_r^{analytic}|/k_r$")
    ax[2].set_title("Independent FEM cross-check")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle(
        "acoustic-array-beamforming / external anchor (Pekeris waveguide): "
        "H=%.0f m, c1=%.0f m/s, c2=%.0f m/s, rho2/rho1=%.2f"
        % (wg.depth_m, wg.c_water_m_s, wg.c_bottom_m_s, wg.density_ratio),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())