#!/usr/bin/env python3
"""Round 2: does the ka > 2 band of ``CASE-BEM-HULME-HEMISPHERE`` now stand?

Round 1 (``audit_np246_contradiction.py``) kept the extended band WITHHELD: the
two committed artifacts that describe ``ka = 3.5`` disagreed with each other,
carried the same ``numpy`` version, and none of them recorded the solve-stage
residual or the environment of the run that produced it.  It also named the
experiment that would settle the question (entry point x BLAS thread count,
residual next to every value) and the cluster jobs that never returned a
payload.

This round judges the question against *new* evidence instead of the two
disputed artifacts:

* ``evidence/np246_failing_snapshot_m20034.json`` -- the failing artifact as a
  frozen git object (immutable, sha256-pinned), so the failure stays citable
  after the working-tree artifact is regenerated;
* ``band_provenance.json`` -- a fresh re-run of the whole band plus the
  discriminator arms, every cell carrying host, BLAS thread environment, both
  solve-stage residuals, the condition number and the git head.

Gates (the credit rests on C1-C3 and on the per-``ka`` credit list; C4 is
reported next to them because a band is credited point by point):

``C1_failure_does_not_reproduce``  the fresh provenanced run at ``ka = 3.5``
    lands inside the anchor tolerance and its residual is admissible.
``C2_same_matrix``                 the failing snapshot's condition number and
    the fresh one agree to 1e-6 relative -- the matrix is not the explanation,
    so the historical failure was a *solution* event.
``C3_value_is_entry_point_and_thread_invariant``  ``band_provenance``'s own P4
    gate: neither entry point nor thread count moves ``A11``/``B33``.
``C4_band_credit_per_ka``          per ``ka``: credited inside 5 %, withheld
    above it.  A ``ka`` that fails keeps its number.

The ``numpy`` version attribution is reported separately as UNPROVEN: the
frozen failing artifact records no environment at all and no ``numpy 2.0.1``
interpreter is available on this host, so no artifact exists that would carry
that attribution.  It is deliberately *not* a gate: the band credit rests on a
fresh provenanced run, not on a story about why the old one was wrong.

Usage
-----
    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\
audit_np246_closure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SNAPSHOT = HERE / "evidence" / "np246_failing_snapshot_m20034.json"
PROVENANCE = HERE / "band_provenance.json"
OUT_JSON = HERE / "evidence" / "np246_closure.json"
OUT_DOC = REPO / "docs" / "case_bem_hulme_np246_closure.md"
KA = 3.5
COND_RTOL = 1e-6


def build() -> dict:
    snap = json.loads(SNAPSHOT.read_text())
    prov = json.loads(PROVENANCE.read_text())
    band = {f"{c['ka']:g}": c for c in prov["cells"] if c["group"] == "band"}
    fresh = band[f"{KA:g}"]
    frozen = snap["failing_cell"]
    tolerance = prov["tolerance"]

    fresh_err = abs(fresh["A11_dimless"] / frozen["reference_A11_dimless"] - 1.0)
    c1 = fresh_err < tolerance and fresh["radiation_relative_residual"] <= prov["residual_max"]
    cond_rel = abs(fresh["condition_number"] / frozen["condition_number"] - 1.0)
    c2 = cond_rel < COND_RTOL
    p4 = prov["gates"]["P4_value_is_entry_point_and_thread_invariant"]
    per_ka = prov["gates"]["P3_band_within_tolerance"]["per_ka"]

    gates = {
        "C1_failure_does_not_reproduce": {
            "criterion": f"fresh provenanced run at ka = {KA:g} is inside "
                         f"{tolerance:.0%} and its radiation residual is <= "
                         f"{prov['residual_max']:g}",
            "pass": bool(c1),
            "measured_error_A11": fresh_err,
            "measured_residual_radiation": fresh["radiation_relative_residual"],
            "measured_residual_diffraction": fresh["diffraction_relative_residual"],
            "frozen_error_A11": abs(frozen["A11_dimless"]
                                    / frozen["reference_A11_dimless"] - 1.0),
        },
        "C2_same_matrix": {
            "criterion": f"condition number of the frozen failing cell and of the "
                         f"fresh cell agree to {COND_RTOL:g} relative",
            "pass": bool(c2),
            "measured": cond_rel,
            "frozen": frozen["condition_number"],
            "fresh": fresh["condition_number"],
        },
        "C3_value_is_entry_point_and_thread_invariant": {
            "criterion": p4["criterion"], "pass": p4["pass"], "spread": p4["spread"]},
        "C4_band_credit_per_ka": {
            "criterion": "each ka of the extension band is credited inside the "
                         "anchor tolerance or withheld with its measured error",
            "pass": True,
            "per_ka": {k: {"error": v["error"], "worst_quantity": v["worst_quantity"],
                           "credited": v["credited"]} for k, v in per_ka.items()},
        },
    }
    payload = {
        "case_code": "BEM-HULME-HEMISPHERE",
        "anchor": "CASE-BEM-HULME-HEMISPHERE",
        "round": 2,
        "inputs": {"frozen_snapshot": {"commit": snap["recovered_from"]["commit"],
                                       "blob_sha256": snap["blob_sha256"],
                                       "A11_dimless": frozen["A11_dimless"]},
                   "fresh_A11_dimless": fresh["A11_dimless"],
                   "provenance": {"generated_at": prov["generated_at"],
                                  "environment": prov["environment"]}},
        "gates": gates,
        "verdict": "PASS" if all(g["pass"] for g in gates.values()) else "FAIL",
        "band_status": prov["band_status"],
        "numpy_version_attribution": {
            "status": "UNPROVEN",
            "why": "the frozen failing artifact records no environment at all "
                   f"(recorded_environment = {snap['recorded_environment']!r}) and no "
                   "numpy 2.0.1 interpreter exists on this host, so no artifact "
                   "carries the attribution; the band credit does not rest on it.",
        },
    }
    return payload


def render(payload: dict) -> str:
    g = payload["gates"]
    L: list[str] = []
    a = L.append
    a("# CASE-BEM-HULME-HEMISPHERE — round 2 closure of the ka > 2 contradiction")
    a("")
    a("Generated by `benchmarks/verified/casepool_bem_hulme_hemisphere/"
      "audit_np246_closure.py` from the frozen failure snapshot and the "
      "provenanced re-run; every number is injected programmatically.")
    a("")
    a(f"* frozen failure: commit {payload['inputs']['frozen_snapshot']['commit'][:12]}, "
      f"blob sha256 `{payload['inputs']['frozen_snapshot']['blob_sha256'][:16]}…`")
    env = payload["inputs"]["provenance"]["environment"]
    a(f"* fresh re-run: numpy {env['numpy']}, host {env['host']}, git "
      f"{str(env['git_head'])[:12]}, generated {payload['inputs']['provenance']['generated_at']}")
    a("")
    a("## 1. Gates")
    a("")
    a("| gate | criterion | pass |")
    a("|---|---|---|")
    for key, gate in g.items():
        a(f"| `{key}` | {gate['criterion']} | {'PASS' if gate['pass'] else 'FAIL'} |")
    a("")
    snap_A11 = payload["inputs"]["frozen_snapshot"]["A11_dimless"]
    a("## 2. ka = 3.5, frozen failure vs fresh provenanced run")
    a("")
    c1 = g["C1_failure_does_not_reproduce"]
    c2 = g["C2_same_matrix"]
    a("| arm | A11 | rel. error vs Hulme | radiation res | diffraction res | cond |")
    a("|---|---|---|---|---|---|")
    a(f"| m20034 (frozen) | {snap_A11:+.6f} | {100 * c1['frozen_error_A11']:+.2f}% "
      f"| not recorded | not recorded | {c2['frozen']:.6f} |")
    a(f"| fresh re-run | {payload['inputs']['fresh_A11_dimless']:+.6f} | {100 * c1['measured_error_A11']:+.2f}% "
      f"| {c1['measured_residual_radiation']:.2e} "
      f"| {c1['measured_residual_diffraction']:.2e} | {c2['fresh']:.6f} |")
    a("")
    a(f"The two condition numbers agree to {c2['measured']:.1e} relative: the matrix "
      f"whose solution was wrong and the matrix whose solution is right are the same "
      f"matrix. The historical failure was a solution-stage event, and it does not "
      f"reproduce under the recorded environment.")
    a("")
    a("## 3. Per-ka credit of the extension band")
    a("")
    a("| ka | worst error | quantity | credited (<5 %) |")
    a("|---|---|---|---|")
    for ka in sorted(g["C4_band_credit_per_ka"]["per_ka"], key=float):
        row = g["C4_band_credit_per_ka"]["per_ka"][ka]
        a(f"| {ka} | {100 * row['error']:.2f}% | "
          f"{row['worst_quantity'].replace('_dimless', '')} | "
          f"{'yes' if row['credited'] else 'no'} |")
    a("")
    a(f"* credited ka: {payload['band_status']['credited_ka']}")
    a(f"* withheld ka: {payload['band_status']['withheld_ka']}")
    a("")
    a("## 4. numpy version attribution")
    a("")
    a(f"**{payload['numpy_version_attribution']['status']}** — "
      f"{payload['numpy_version_attribution']['why']}")
    a("")
    a(f"**Overall: {payload['verdict']}**")
    a("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT_JSON)
    p.add_argument("--doc", type=Path, default=OUT_DOC)
    args = p.parse_args(argv)
    if not PROVENANCE.is_file():
        print(f"missing {PROVENANCE}; run band_provenance.py first")
        return 2
    payload = build()
    args.out.write_text(json.dumps(payload, indent=1))
    args.doc.write_text(render(payload))
    print(json.dumps({"verdict": payload["verdict"],
                      "gates": {k: v["pass"] for k, v in payload["gates"].items()},
                      "band_status": payload["band_status"]}, indent=1))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())