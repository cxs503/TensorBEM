"""Board audit m17340 / subject m17321 (CASE-BEM-CUMMINS-WIGLEY).

Independent re-derivation of the numbers the subject reports.  Nothing is
copied from the subject's ``results.json``: every block below either

* re-runs the subject's own pipeline from the cached single-frequency solves
  (``artifacts/audit_m17340/case_bem_cummins_wigley/*_omega_*.json``) and diffs
  the outcome against the stored ``results.json`` (reproducibility), or
* re-derives the quantity by a route the subject does not use
  (geometry by a divergence-theorem sum, Kramers-Kronig by a time-domain memory
  kernel instead of the subject's principal-value quadrature, G1 reference by a
  fresh capytaine build instead of the frozen file), so agreement is evidence
  and disagreement is a finding.

    PYTHONPATH=src python3 docs/board_audit/m17340/audit_independent.py

Writes ``artifacts/audit_m17340/audit_checks.json``.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem import time_domain as td  # noqa: E402
from tensorbem.wigley3_freq import wigley3_hydrostatic_reference  # noqa: E402

SUBJECT_DIR = ROOT / "artifacts" / "audit_m17340" / "case_bem_cummins_wigley"
SNAPSHOT = ROOT / "docs" / "board_audit" / "m17340" / "subject_snapshot" / "cummins_wigley_kk_causality.py"
REF_PATH = ROOT / "benchmarks" / "verified" / "wigley3_recalled_ref" / "capytaine_ref.json"
OUT = ROOT / "artifacts" / "audit_m17340" / "audit_checks.json"
DOF_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def load_subject():
    spec = importlib.util.spec_from_file_location("audited_case", SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. reproducibility: rebuild gates from the cached solves, diff vs results.json
# --------------------------------------------------------------------------
def check_reproduction(case, stored: dict) -> dict:
    reference = case.load_capytaine_reference(path=REF_PATH)
    out = {}
    for level in ("coarse", "medium"):
        points = case.load_points(SUBJECT_DIR / "points", level)
        grid = [w for w in case.OMEGA_GRID if round(w, 9) in points]
        sweep = case.sweep_from_points(points, grid)
        analysis = case.gates_from_sweep(sweep, reference=reference)
        was = stored["levels"][level]["gates"]
        rows = {}
        for gid in ("G1_cross_solver", "G2_kernel_at_zero", "G3_kramers_kronig",
                    "G4_free_decay", "G5_round_trip"):
            new = analysis["gates"][gid]
            old = was["detail"][gid]
            a, b = old.get("value"), new.get("value")
            same = (
                (a is None and b is None)
                or (isinstance(a, float) and isinstance(b, float)
                    and (a == b or (not np.isfinite(a) and not np.isfinite(b))
                         or (np.isfinite(a) and np.isfinite(b)
                             and abs(a - b) <= 1e-9 * max(abs(a), abs(b)))))
            )
            rows[gid] = {"stored": a, "recomputed": b, "passed_stored": old.get("passed"),
                         "passed_recomputed": new.get("passed"), "identical": bool(same),
                         "tol": new.get("tol")}
        out[level] = {
            "n_points_cached": len(points),
            "n_points_grid": len(case.OMEGA_GRID),
            "missing_omega": stored["levels"][level].get("missing_omega"),
            "verdict_stored": was["verdict"],
            "verdict_recomputed": analysis["verdict"],
            "gates": rows,
            "all_gate_values_identical": all(r["identical"] for r in rows.values()),
        }
    return out


# --------------------------------------------------------------------------
# 2. geometry: independent displacement from the same wetted mesh
# --------------------------------------------------------------------------
def check_geometry(case, stored: dict) -> dict:
    ref_volume = float(wigley3_hydrostatic_reference()["volume"])
    out = {"analytic_volume": ref_volume,
           "analytic_Cb": ref_volume / (3.0 * 0.3 * 0.1875),
           "declared_Cb": 0.543, "levels": {}}
    for level in ("coarse", "medium"):
        mesh = case.wetted_mesh(level)
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        tri = verts[faces]
        volume = float(np.sum(np.einsum("ij,ij->i", tri[:, 0],
                                       np.cross(tri[:, 1], tri[:, 2]))) / 6.0)
        was = stored["levels"][level]["signature"]
        out["levels"][level] = {
            "panels_triangles": int(faces.shape[0]),
            "mesh_volume_auditor": volume,
            "mesh_volume_subject": was["mesh_volume"],
            "rel_error_auditor": volume / ref_volume - 1.0,
            "stored_rel_error": was["mesh_volume_rel_error"],
            "agrees_with_subject": abs(volume - was["mesh_volume"]) <= 1e-9,
        }
    gate = case.geometry_gate()
    out["subject_geometry_gate_passed"] = gate["passed"]
    out["Cb_from_volume"] = gate["Cb_from_volume"]
    out["Cb_abs_error"] = gate["Cb_abs_error"]
    return out


# --------------------------------------------------------------------------
# 3. G1 reference: fresh capytaine build vs the frozen file
# --------------------------------------------------------------------------
def check_reference(levels=((10, 5), (20, 10))) -> dict:
    import capytaine as cpt

    from tensorbem import wigley3_freq as wf

    frozen = json.loads(REF_PATH.read_text())
    rows = {}
    solver = cpt.BEMSolver()
    for n_sections, n_levels in levels:
        label = f"n_sections={n_sections},n_levels={n_levels}"
        entry = next(e for e in frozen["builders"]["wigley3_freq"]
                     if e["level"] == label)
        hull = wf.wigley3_wetted_hull_mesh(n_sections=n_sections, n_levels=n_levels)
        mesh = cpt.Mesh(vertices=np.asarray(hull.vertices, dtype=float),
                        faces=np.asarray(hull.faces, dtype=np.int64))
        body = cpt.FloatingBody(mesh=mesh)
        body.add_all_rigid_body_dofs(rotation_center=np.array((0.0, 0.0, -0.1875 / 2.0)))
        worst = 0.0
        per_dof = {}
        for dof in (d.capitalize() for d in DOF_LABELS):
            res = solver.solve(cpt.RadiationProblem(
                body=body, omega=1.0, radiating_dof=dof, water_depth=np.inf,
                rho=1025.0, g=9.80665))
            a_fresh = float(res.added_mass[dof])
            b_fresh = float(res.radiation_damping[dof])
            a_frozen = float(entry["added_mass"][dof])
            b_frozen = float(entry["damping"][dof])
            rel_a = abs(a_fresh - a_frozen) / max(abs(a_frozen), 1e-30)
            rel_b = abs(b_fresh - b_frozen) / max(abs(b_frozen), 1e-30)
            worst = max(worst, rel_a, rel_b)
            per_dof[dof] = {"added_mass_frozen": a_frozen, "added_mass_fresh": a_fresh,
                            "damping_frozen": b_frozen, "damping_fresh": b_fresh,
                            "rel_err_max": max(rel_a, rel_b)}
        rows[label] = {
            "panels": int(mesh.nb_faces),
            "panels_frozen": entry["panels"],
            "mesh_volume_fresh": float(mesh.volume),
            "mesh_volume_frozen": entry["mesh_volume"],
            "worst_rel_err_fresh_vs_frozen": worst,
            "per_dof": per_dof,
        }
    return {"tool": f"capytaine {getattr(cpt, '__version__', 'unknown')}",
            "frozen_generated_utc": frozen["metadata"].get("generated_utc"),
            "levels": rows}


# --------------------------------------------------------------------------
# 4. Kramers-Kronig by an independent route (time-domain memory kernel)
# --------------------------------------------------------------------------
def kk_time_domain(omega, damping, omega_eval, a_inf, *, sign=-1.0,
                   tau_max=400.0, dtau=0.01) -> np.ndarray:
    """A(w) - A_inf = sign * (1/w) int K(tau) sin(w tau) dtau, K from the same band.

    ``sign=-1`` is the Cummins/Ogilvie relation; ``sign=+1`` is run only as a
    convention control.  Unlike the subject's route this closes nothing below
    the first tabulated frequency: the band is used exactly as tabulated.
    """
    tau = np.arange(0.0, tau_max + 0.5 * dtau, dtau)
    kernel = td.radiation_memory_kernel(omega, damping, tau,
                                        low_frequency_model="zero")
    k = np.asarray(kernel.values, dtype=float)
    out = []
    for w in np.atleast_1d(np.asarray(omega_eval, dtype=float)):
        integral = np.trapezoid(k * np.sin(w * tau), tau)
        out.append(a_inf + sign * integral / w)
    return np.asarray(out)


def check_kk(case, stored: dict, dofs=("surge", "heave", "roll", "yaw")) -> dict:
    reference = case.load_capytaine_reference(path=REF_PATH)
    out = {}
    for level in ("coarse", "medium"):
        points = case.load_points(SUBJECT_DIR / "points", level)
        grid = [w for w in case.OMEGA_GRID if round(w, 9) in points]
        sweep = case.sweep_from_points(points, grid)
        analysis = case.gates_from_sweep(sweep, reference=reference)
        rows = {}
        for dof in dofs:
            study = analysis["studies"][dof]
            i = DOF_LABELS.index(dof)
            omega = np.asarray(sweep["omega"], dtype=float)
            added = np.asarray(sweep["added_mass"], dtype=float)[:, i, i]
            damping = np.asarray(sweep["damping"], dtype=float)[:, i, i]
            a_inf = float(study["added_mass_inf"])
            w_eval = 0.5 * (omega[:-1] + omega[1:])
            measured = 0.5 * (added[:-1] + added[1:])
            scale = max(float(np.max(np.abs(measured - a_inf))), 1e-30)
            pred_minus = kk_time_domain(omega, damping, w_eval, a_inf, sign=-1.0)
            pred_plus = kk_time_domain(omega, damping, w_eval, a_inf, sign=+1.0)
            r_minus = float(np.max(np.abs(pred_minus - measured)) / scale)
            r_plus = float(np.max(np.abs(pred_plus - measured)) / scale)
            rows[dof] = {
                "added_mass_inf": a_inf,
                "subject_G3_rel_err": float(study["G3"]["rel_err"]),
                "auditor_G3_rel_err_cummins_sign": r_minus,
                "auditor_G3_rel_err_flipped_sign": r_plus,
                "tol": 0.10,
                "both_routes_fail": bool(r_minus > 0.10),
            }
        out[level] = rows
    return out


# --------------------------------------------------------------------------
# 5. what the missing low-frequency band would have to be (diagnosis, not a fix)
# --------------------------------------------------------------------------
def check_band_sensitivity(case, stored: dict, dof="yaw") -> dict:
    """Drop the two lowest nodes of the coarse sweep and watch G3 move.

    If the residual is a band-truncation effect it must *grow* when the band is
    cut from below; if it is a bug in the gate it is indifferent to the band.
    """
    reference = case.load_capytaine_reference(path=REF_PATH)
    points = case.load_points(SUBJECT_DIR / "points", "coarse")
    realised = [w for w in case.OMEGA_GRID if round(w, 9) in points]
    out = {}
    for tag, grid in (("full", realised), ("drop_lowest_2", realised[2:])):
        sweep = case.sweep_from_points(points, grid)
        analysis = case.gates_from_sweep(sweep, reference=reference)
        out[tag] = {
            "omega_min": grid[0],
            "G3_rel_err": float(analysis["studies"][dof]["G3"]["rel_err"]),
            "G3_passed": bool(analysis["studies"][dof]["G3"]["rel_err"] <= 0.10),
            "verdict": analysis["verdict"],
        }
    out["dof"] = dof
    return out


def main() -> int:
    t0 = time.time()
    case = load_subject()
    stored = json.loads((SUBJECT_DIR / "results.json").read_text())
    report = {
        "audit": "m17340 / subject m17321 / CASE-BEM-CUMMINS-WIGLEY",
        "subject_artifacts": str(SUBJECT_DIR.relative_to(ROOT)),
        "reproduction": check_reproduction(case, stored),
        "geometry": check_geometry(case, stored),
        "reference_rebuild": check_reference(),
        "kk_independent_route": check_kk(case, stored),
        "band_sensitivity": check_band_sensitivity(case, stored),
        "wall_seconds": None,
    }
    report["wall_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:6000])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())