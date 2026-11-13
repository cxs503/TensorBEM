"""Board audit m17264 / subject m17209 — evidence binding check (no new BEM solve).

Re-derives every gate of the audited case from the *subject's own* cached
radiation sweeps and binds the delivery report's numbers to the subject's
artifacts, so that "report looks good" is replaced by "every number has a
pointer".  Runs in seconds; no 3-D BEM solve is repeated here (the sweeps are
the frozen BEM output shipped by the subject).

Checks
  C1 artifacts present and parseable (results.json + sweep_coarse/medium.json)
  C2 gates recomputed from the sweeps == gates in results.json (bit-level)
  C3 verdict / failed_gates consistent with the gate table and the tolerance spec
  C4 geometry gate before any hydrodynamic number: Cb, volume = L*B*T*Cb, L/B, B/T
  C5 report §4 R1 table == results.json cross_solver rows (ref, got, rel err)
  C6 report §4 gate table + free-decay paragraph == results.json
  C7 independent-solver reference is loaded from the committed file, and the
     values in that file are the ones quoted in the report (no hand transcription)
  C8 roll R1 deficit is NOT a mesh-resolution limit (coarse -> medium moves away
     from the independent reference), i.e. a systematic convention/implementation
     effect -> motivates docs/board_audit/m17264/audit_rotation_center.py

    python3 docs/board_audit/m17264/audit_evidence_binding.py
        [--snapshot docs/board_audit/m17264] [--json artifacts/audit_m17264/binding.json]

Exit code 0 only if every check that is expected to hold does hold (C8 is a
diagnostic, it does not gate the exit code).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def load_case_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("subject_case_m17209", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return False
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(HERE))
    ap.add_argument("--json", default="artifacts/audit_m17264/binding.json")
    ap.add_argument("--ref", default="benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json")
    args = ap.parse_args()
    snap = pathlib.Path(args.snapshot)
    checks: dict[str, dict] = {}

    def record(cid, ok, detail):
        checks[cid] = {"ok": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")

    # ---------------- C1 artifacts ----------------
    res_path = snap / "subject_artifacts" / "results.json"
    sweeps = {lv: json.loads((snap / "subject_artifacts" / f"sweep_{lv}.json").read_text())
              for lv in ("coarse", "medium")}
    results = json.loads(res_path.read_text())
    record("C1", res_path.exists() and len(sweeps) == 2,
           f"results.json + 2 sweeps parsed; medium panels={sweeps['medium']['panels']} "
           f"quads ({len(sweeps['medium']['omega'])} freqs, wall {sweeps['medium']['wall_seconds']} s), "
           f"coarse panels={sweeps['coarse']['panels']} (wall {sweeps['coarse']['wall_seconds']} s)")

    # ---------------- C2 gates recomputed from the sweeps ----------------
    case = load_case_module(snap / "subject_snapshot" / "cummins_wigley_kernel_6dof.py")

    def sweep_fn(omega, *, level):
        payload = sweeps[level]
        assert [round(float(x), 6) for x in payload["omega"]] == [round(float(x), 6) for x in omega]
        return payload

    reference = case.load_capytaine_reference(pathlib.Path(args.ref))
    rerun = case.run_case(level="medium", levels=("coarse", "medium"), omegas=tuple(sweeps["medium"]["omega"]),
                          dt=0.02, tau_max=20.0, reference=reference, sweep_fn=sweep_fn,
                          progress=lambda *_: None)
    gate_ok, gate_detail = True, []
    for gid, gate in results["gates"].items():
        other = rerun["gates"][gid]
        same = close(gate["value"], other["value"]) and close(gate["tol"], other["tol"]) and gate["passed"] == other["passed"]
        gate_ok &= same
        gate_detail.append(f"{gid} {gate['value']:.6g}/{other['value']:.6g}"
                           f" tol {gate['tol']}, passed={gate['passed']}")
    record("C2", gate_ok, "gates re-derived from cached sweeps: " + "; ".join(gate_detail))

    # ---------------- C3 verdict consistency ----------------
    expect_failed = [gid for gid, gate in results["gates"].items() if not gate["passed"]]
    verdict = "PASS" if not expect_failed else "FAIL:" + ",".join(expect_failed)
    record("C3", verdict == results["verdict"] and expect_failed == results["failed_gates"],
           f"verdict {results['verdict']} vs recomputed {verdict}; every gate re-evaluated "
           f"against its own tol (no gate is 'passed by narrative')")

    # ---------------- C4 geometry gate ----------------
    geom = results["geometry"]
    vol = geom["L"] * geom["B"] * geom["T"] * geom["Cb"]
    geo_ok = (abs(vol - geom["volume"]) <= 1e-12 and abs(geom["L_B"] - geom["L"] / geom["B"]) <= 1e-12
              and abs(geom["B_T"] - geom["B"] / geom["T"]) <= 1e-12)
    record("C4", geo_ok,
           f"Cb={geom['Cb']}, nabla={geom['volume']:.8f} m3 == L*B*T*Cb={vol:.8f}; "
           f"L/B={geom['L_B']}, B/T={geom['B_T']} (geometry verified before any hydrodynamic number)")

    # ---------------- C5 report R1 table vs results.json ----------------
    report = (snap / "subject_report" / "dev_report_m17209_cummins_wigley_kernel_6dof.md").read_text()
    rows = results["cross_solver"]["rows"]
    table_ok, table_detail = True, []
    for row in rows:
        pat = re.compile(r"\|\s*" + row["dof_label"] + r"\s*\|\s*([-\d.eE+]+)\s*\|\s*([-\d.eE+]+)\s*\|\s*([-+\d.]+)\s*%")
        m = pat.search(report)
        if not m:
            table_ok = False
            table_detail.append(f"{row['dof_label']}: row not found in report")
            continue
        rep_ref, rep_got, rep_err = float(m.group(1)), float(m.group(2)), float(m.group(3)) / 100.0
        ok = (abs(rep_ref - row["added_mass_ref"]) <= 5e-4 * abs(row["added_mass_ref"]) + 5e-4
              and abs(rep_got - row["added_mass_got"]) <= 5e-4 * abs(row["added_mass_got"]) + 5e-4
              and abs(rep_err - row["added_mass_rel_err"]) <= 5e-4)
        table_ok &= ok
        table_detail.append(f"{row['dof_label']} ref/got {rep_ref}/{rep_got} vs "
                            f"{row['added_mass_ref']:.6g}/{row['added_mass_got']:.6g} {'ok' if ok else 'MISMATCH'}")
    record("C5", table_ok, "report §4 R1 table bound to cross_solver rows (3-4 significant digits): "
                           + "; ".join(table_detail))

    # ---------------- C6 gate table + free decay in report ----------------
    gate_tok = re.search(r"\| R1 \| R1_cross_solver \| ([\d.]+) \|", report)
    fd_tok = re.search(r"ζ_decay = ([\d.]+)", report)
    fd = results["free_decay_heave"]
    fd_ok = (gate_tok is not None and abs(float(gate_tok.group(1)) - results["gates"]["R1"]["value"]) <= 1e-3
             and fd_tok is not None and abs(float(fd_tok.group(1)) - fd["zeta_free_decay"]) <= 5e-5)
    record("C6", fd_ok,
           f"report gate R1={gate_tok.group(1) if gate_tok else None} vs {results['gates']['R1']['value']:.6f}; "
           f"report zeta_decay={fd_tok.group(1) if fd_tok else None} vs {fd['zeta_free_decay']:.6f}; "
           f"R4 half-power error field: {fd['half_power']['error']!r}")

    # ---------------- C7 reference is committed, values match ----------------
    ref = json.loads(pathlib.Path(args.ref).read_text())
    lvl = next(e for e in ref["builders"]["wigley3_freq"]
               if e["level"] == results["cross_solver"]["reference"]["level"])
    ref_ok = all(abs(lvl["added_mass"][l.capitalize()] - row["added_mass_ref"]) <= 1e-12 * max(1.0, abs(row["added_mass_ref"]))
                 for l, row in zip(DOFS, rows))
    record("C7", ref_ok,
           f"cross_solver ref loaded from {results['cross_solver']['reference']['path']} "
           f"({ref['metadata']['tool']} {ref['metadata']['version']}, {ref['metadata']['platform']}, "
           f"centre {ref['metadata']['rotation_center']}); per-DOF ref values identical to the frozen "
           f"file -> no hand transcription")

    # ---------------- C8 roll deficit is systematic, not resolution ----------------
    ref_roll = lvl["added_mass"]["Roll"]
    coarse_roll = sweeps["coarse"]["added_mass"][2][3][3]
    medium_roll = sweeps["medium"]["added_mass"][2][3][3]
    diag_ok = abs(medium_roll - ref_roll) / ref_roll > 0.10 and abs(medium_roll - ref_roll) > abs(coarse_roll - ref_roll)
    record("C8", diag_ok,
           f"A44(ref)={ref_roll:.4f}, coarse={coarse_roll:.4f} ({100*(coarse_roll-ref_roll)/ref_roll:+.2f}%), "
           f"medium={medium_roll:.4f} ({100*(medium_roll-ref_roll)/ref_roll:+.2f}%) -> refinement moves "
           f"AWAY from the reference, ratio ref/medium={ref_roll/medium_roll:.3f}: systematic effect "
           f"(rotation-centre convention), not a panel-resolution limit")

    payload = {
        "subject": {"mission": 17209, "anchor": "CASE-BEM-CUMMINS-WIGLEY",
                    "commit": "f8a5442c7a8282ea66bc583c79d42ad8313a9a0b",
                    "branch": "agent/m17209"},
        "checks": checks,
        "all_gating_ok": all(c["ok"] for cid, c in checks.items() if cid != "C8"),
        "diagnostic_c8_ok": checks["C8"]["ok"],
    }
    out = pathlib.Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", out)
    return 0 if payload["all_gating_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())