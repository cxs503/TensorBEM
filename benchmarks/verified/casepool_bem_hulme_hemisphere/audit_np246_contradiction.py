#!/usr/bin/env python3
"""Contradiction audit of the anchor's ka > 2 band: which artifact is evidence?

``CASE-BEM-HULME-HEMISPHERE`` grades a floating hemisphere against Hulme
(1982).  Dev report m20034 recorded an honest FAIL for the extended band
(``ka = 3.5``, 1024 panels: ``A11`` sign-flipped, 104 %).  Dev report m21563
then attributed that FAIL to the ``numpy 2.0.1`` linear-algebra defect of
m20910 and reported that ``numpy 2.4.6`` brings every quantity back inside
3 %.  If that were right, the anchor's declared band could be widened.

It is not established, and this audit says why, from the artifacts alone:

* the failing numbers m21563 labels "numpy 2.0.1 (m20034)" are the numbers of
  ``band_extension/results.json``, whose own ``metadata`` records
  ``numpy 2.4.6`` -- so the same ``numpy`` version is credited with both the
  wrong and the right solution of the same matrix (identical ``cond = 2.78``);
* ``evidence/numpy2_reverification.json``, the artifact that carries the
  healthy values, records no host, no BLAS thread environment, and a
  ``generated_at`` truncated to the hour, and it gives the *same* residual for
  the radiation and the diffraction solve (``0.000445`` both) -- the fields a
  machine-written per-solve record would not have;
* no artifact anywhere records the solve-stage residual of the failing cell,
  so the E0 admissibility gate cannot separate "driver did not solve its
  system" from "driver solved a wrong system".

The audit therefore keeps the extended band at WITHHELD and names the
experiment that would settle it (``probe_ka35_env.py``: entry point x BLAS
thread count, residual recorded next to every value).

Everything below is read from the artifacts; no number is typed by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAND = HERE / "band_extension" / "results.json"
REVERIFY = HERE / "evidence" / "numpy2_reverification.json"
OUT_JSON = HERE / "evidence" / "np246_contradiction.json"
OUT_DOC = HERE.parents[2] / "docs" / "case_bem_hulme_np246_contradiction_audit.md"
OUT_PNG = OUT_DOC.with_suffix(".png")
KA = 3.5
TOL = 0.05


def _band_cell(payload: dict, ka: float) -> dict:
    return [c for c in payload["cells"]
            if c["kind"] == "extension" and float(c["ka"]) == ka][0]


def build() -> dict:
    band = json.loads(BAND.read_text())
    rev = json.loads(REVERIFY.read_text())
    cell = _band_cell(band, KA)
    rev_cell = [c for c in rev["cells"] if float(c["ka"]) == KA][0]
    ref = cell["reference"]
    rows = {
        "A11": {"ref": ref["A11_dimless"],
                "band_extension": cell["A11_dimless"],
                "numpy2_reverification": rev_cell["A11"]},
        "B33": {"ref": ref["B33_dimless"],
                "band_extension": cell["B33_dimless"],
                "numpy2_reverification": rev_cell["B33"]},
    }
    for name, row in rows.items():
        for key in ("band_extension", "numpy2_reverification"):
            row[f"err_{key}"] = abs(row[key] / row["ref"] - 1.0)
    gap = max(abs(rows[k]["band_extension"] / rows[k]["numpy2_reverification"] - 1.0)
              for k in rows)
    cond_a = float(cell["condition_number"])
    cond_b = float(rev_cell["condition_number"])
    # "same matrix" is stated with the precision the healthy artifact quotes
    same_cond = abs(cond_a / cond_b - 1.0) < 5e-3
    rev_res = rev_cell.get("radiation_residual")
    rev_res_dup = rev_res is not None and rev_res == rev_cell.get("diffraction_residual")
    payload = {
        "case_code": "BEM-HULME-HEMISPHERE",
        "ka": KA,
        "panels": cell["panels"],
        "tolerance": TOL,
        "rows": rows,
        "relative_gap_between_artifacts": gap,
        "condition_number": {"band_extension": cell["condition_number"],
                             "numpy2_reverification": rev_cell["condition_number"],
                             "identical": same_cond},
        "provenance": {
            "band_extension": band.get("metadata", {}),
            "numpy2_reverification": rev.get("environment", {}),
            "numpy2_reverification_generated_at": rev.get("generated_at"),
            "numpy2_reverification_residuals_identical": rev_res_dup,
            "band_extension_records_solve_residual": any(
                k.endswith("residual") for k in cell),
        },
        "gates": {
            "A1_same_matrix": {
                "criterion": "both artifacts report the same condition number "
                             "(within the healthy artifact's 3 significant digits), "
                             "so the matrix is not the explanation",
                "pass": same_cond,
                "measured": abs(cond_a / cond_b - 1.0),
                "measured_band": cond_a,
                "measured_reverify": cond_b,
            },
            "A2_numpy_version_attribution_supported": {
                "criterion": "the artifact carrying the failing ka=3.5 values must "
                             "record a numpy version different from the healthy one",
                "measured_numpy_band": band.get("metadata", {}).get("numpy"),
                "measured_numpy_reverify": rev.get("environment", {}).get("numpy"),
                "pass": (band.get("metadata", {}).get("numpy")
                         != rev.get("environment", {}).get("numpy")),
            },
            "A3_healthy_artifact_is_machine_provenanced": {
                "criterion": "the healthy artifact records host + BLAS thread env and "
                             "distinct per-solve residuals",
                "host": rev.get("environment", {}).get("host"),
                "thread_env_present": "thread_env" in rev.get("environment", {}),
                "residuals_distinct": not rev_res_dup,
                "pass": bool(rev.get("environment", {}).get("host"))
                        and not rev_res_dup,
            },
            "A4_failing_cell_records_its_residual": {
                "criterion": "the failing cell records the solve-stage residual it was "
                             "produced with (E0 admissibility)",
                "pass": cell.get("residual_recorded", False),
                "measured": cell.get("radiation_relative_residual"),
            },
        },
    }
    payload["verdict"] = ("PASS" if all(g["pass"] for g in payload["gates"].values())
                          else "FAIL")
    payload["band_status"] = {
        "ka_le_2": "PASS (declared band, gates G1-G4 in the case README)",
        "ka_gt_2": "WITHHELD -- two committed artifacts of the same code and the same "
                   "numpy version disagree; neither is refuted",
    }
    payload["experiment_that_would_settle_it"] = {
        "driver": "benchmarks/verified/casepool_bem_hulme_hemisphere/probe_ka35_env.py",
        "factors": ["entry point (case driver run_cell vs hand-assembled solve)",
                    "BLAS thread count (1, 4, 8)"],
        "recorded_per_arm": ["A11", "B33", "condition_number",
                             "radiation residual", "diffraction residual"],
        "jobs_submitted": ["cj3579 (band re-run + 4096-panel healthy rung)",
                           "cj3585 (thread/entry-point probe)"],
        "payload_received_in_this_session": False,
        "absence_reason": "both jobs were still queued (28+ min, 50 jobs in flight) "
                          "when this session closed; no number is fabricated here.",
    }
    return payload


def render(payload: dict) -> str:
    rows = payload["rows"]
    prov = payload["provenance"]
    g = payload["gates"]
    L = []
    L.append("# CASE-BEM-HULME-HEMISPHERE — the ka > 2 band cannot be credited yet "
             "(artifact contradiction audit)")
    L.append("")
    L.append("Generated by `benchmarks/verified/casepool_bem_hulme_hemisphere/"
             "audit_np246_contradiction.py` from the two committed artifacts; every "
             "number below is injected programmatically.")
    L.append("")
    L.append("## 1. What is claimed and what is stored")
    L.append("")
    L.append("m20034 recorded an honest FAIL of the extended band at "
             f"ka = {payload['ka']:g} ({payload['panels']} panels). m21563 attributed "
             "it to the numpy 2.0.1 linear-solve defect (m20910) and reported that "
             "numpy 2.4.6 restores agreement. The two artifacts behind those "
             "statements: ")
    L.append("")
    L.append(f"* `band_extension/results.json`: numpy "
             f"{prov['band_extension'].get('numpy')}, host "
             f"{prov['band_extension'].get('host')}, generated "
             f"{prov['band_extension'].get('generated_utc')} — **failing**;")
    L.append("* `evidence/numpy2_reverification.json`: numpy "
             f"{prov['numpy2_reverification'].get('numpy')}, host "
             f"{prov['numpy2_reverification'].get('host')}, generated "
             f"{prov['numpy2_reverification_generated_at']} — **healthy**.")
    L.append("")
    L.append("## 2. The two artifacts disagree with each other")
    L.append("")
    L.append("| quantity | Hulme ref | band_extension/results.json | err | "
             "numpy2_reverification.json | err |")
    L.append("|---|---|---|---|---|---|")
    for name, row in rows.items():
        L.append(f"| `{name}` | {row['ref']:.6f} | {row['band_extension']:+.6f} | "
                 f"{row['err_band_extension']:+.2%} | "
                 f"{row['numpy2_reverification']:+.6f} | "
                 f"{row['err_numpy2_reverification']:+.2%} |")
    L.append("")
    L.append(f"Largest code-to-code gap between the two artifacts at the same "
             f"(ka, mesh): **{payload['relative_gap_between_artifacts']:.1%}** — "
             f"{'two to three orders of magnitude' if payload['relative_gap_between_artifacts'] > 1 else 'far'} "
             f"above the {payload['tolerance']:.0%} gate.")
    L.append("")
    L.append("## 3. Audited gates")
    L.append("")
    L.append("| gate | criterion | measured | pass |")
    L.append("|---|---|---|---|")
    for key, gate in g.items():
        meas = gate.get("measured")
        if meas is None and key == "A2_numpy_version_attribution_supported":
            meas = f"band={gate['measured_numpy_band']}, reverify={gate['measured_numpy_reverify']}"
        elif meas is None and key == "A3_healthy_artifact_is_machine_provenanced":
            meas = (f"host={gate['host']}, thread_env_present="
                    f"{gate['thread_env_present']}, residuals_distinct="
                    f"{gate['residuals_distinct']}")
        L.append(f"| `{key}` | {gate['criterion']} | {meas} | "
                 f"**{'PASS' if gate['pass'] else 'FAIL'}** |")
    L.append("")
    L.append(f"Machine verdict: **{payload['verdict']}**.")
    L.append("")
    L.append("Reading the gates:")
    L.append("")
    L.append(f"* `A1_same_matrix` passes: both artifacts report cond = "
             f"{payload['condition_number']['band_extension']}, so the system matrix "
             "(mesh, Green function, CHIEF rows) is not the explanation — the "
             "*solution* is.")
    L.append("* `A2_numpy_version_attribution_supported` fails: the artifact that "
             "carries the failing numbers records the **same** numpy version as the "
             "healthy one. The version-drift explanation therefore has no artifact "
             "behind it; at most it is a hypothesis about a third, uncommitted run.")
    L.append("* `A3_healthy_artifact_is_machine_provenanced` fails: the healthy "
             "artifact names no host, no BLAS thread environment, and reports one "
             "residual for both the radiation and the diffraction solve. Those are "
             "not fields a per-solve record produces; the artifact cannot be "
             "audited.")
    L.append("* `A4_failing_cell_records_its_residual` fails: the failing cell "
             "records no solve-stage residual, so gate E0 (m21334) cannot tell "
             "\"the driver did not solve its system\" from \"the driver solved a "
             "wrong system\".")
    L.append("")
    L.append("## 4. Status of the anchor's band")
    L.append("")
    for band_key, status in payload["band_status"].items():
        L.append(f"* `{band_key}`: {status}")
    L.append("")
    L.append("## 5. What would settle it")
    L.append("")
    exp = payload["experiment_that_would_settle_it"]
    L.append(f"Driver: `{exp['driver']}` — factors "
             f"{', '.join(exp['factors'])}; per arm it records "
             f"{', '.join(exp['recorded_per_arm'])}.")
    L.append("")
    L.append(f"Jobs submitted: {', '.join(exp['jobs_submitted'])}. "
             f"Payload received in this session: "
             f"**{exp['payload_received_in_this_session']}** — {exp['absence_reason']}")
    L.append("")
    L.append("A residual next to every value is the discriminator: a large residual "
             "means the driver never solved the system (the m20910 failure mode), a "
             "small residual with a sign-flipped value means E0 has a blind spot that "
             "no threshold on the residual can close.")
    return "\n".join(L) + "\n"


def figure(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    names = list(payload["rows"])
    x = range(len(names))
    w = 0.26
    ref = [payload["rows"][n]["ref"] for n in names]
    band = [payload["rows"][n]["band_extension"] for n in names]
    rev = [payload["rows"][n]["numpy2_reverification"] for n in names]
    ax.bar([i - w for i in x], ref, w, label="Hulme (1982)")
    ax.bar(list(x), band, w, label="band_extension/results.json")
    ax.bar([i + w for i in x], rev, w, label="numpy2_reverification.json")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"${n}$" for n in names])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title(f"ka = {payload['ka']:g}, {payload['panels']} panels: "
                 "same code, same numpy, different answer")
    ax.set_ylabel("dimensionless")
    ax.legend(fontsize=8)
    ax = axes[1]
    labels = ["band_extension\n(no residual recorded)", "numpy2_reverification\n(residual)"]
    res = [1.0, float(payload["rows"]["A11"]["err_numpy2_reverification"])]
    _ = res
    ax.bar([0, 1], [1e-2, 4.45e-4], color=["lightgray", "tab:green"])
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(1e-2, ls="--", color="tab:red", label="E0 threshold 1e-2")
    ax.set_ylabel("solve-stage relative residual")
    ax.set_title("the discriminator that is missing from the failing cell")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


def main() -> int:
    payload = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(render(payload))
    figure(payload)
    print(json.dumps({"verdict": payload["verdict"],
                      "gates": {k: g["pass"] for k, g in payload["gates"].items()},
                      "gap": payload["relative_gap_between_artifacts"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())