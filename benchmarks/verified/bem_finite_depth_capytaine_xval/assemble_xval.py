#!/usr/bin/env python3
"""Assemble the frozen cross-validation artifact from the per-case cache.

Reads ``cases/*.json`` (written by ``run_cases.py`` on the cluster) and writes
``xval_results.json`` in the schema the test suite and ``make_figures.py`` read.
Also evaluates the closed-form infinite-depth multipole reference
(:func:`tensorbem.analytic.submerged_sphere_wave_multipole`) so at least one leg
of the comparison is anchored to an *analytic* value rather than to a second
numerical code.

Run with ``PYTHONPATH=src python .../assemble_xval.py``.  Pure post-processing:
no solver is invoked, so it is safe to run anywhere.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CASE_DIR = Path("benchmarks/verified/bem_finite_depth_capytaine_xval/cases")
RESULT = CASE_DIR.parent / "xval_results.json"
G1, G2, G3, G4, G5 = 0.10, 0.15, 0.05, 0.05, 0.05
KA_GRID = (0.3, 0.6, 1.0, 1.5, 2.0)
STATIONS = (3.0, 5.0, 8.0, 12.0, 20.0, "inf")


def _key(row):
    return (row["depth_ratio"], float(row["ka"]))


def _rel(got, ref):
    denom = abs(ref)
    if denom < 1e-12:
        return 0.0 if abs(got) < 1e-12 else float("inf")
    return abs(got - ref) / denom


def load(prefix):
    out = {}
    for path in sorted(CASE_DIR.glob(f"{prefix}*.json")):
        row = json.loads(path.read_text())
        if row.get("status") == "error" or "error" in row:
            continue
        out.setdefault(path.stem.split("-")[0] if False else prefix, {})
        out[prefix][_key(row)] = row
    return out.get(prefix, {})


def analytic_inf():
    """Closed-form Havelock multipole A33/B33 at infinite depth."""
    try:
        from tensorbem.analytic import submerged_sphere_wave_multipole

        rows = {}
        for ka in KA_GRID:
            res = an = submerged_sphere_wave_multipole(
                ka, radius=1.0, submergence=2.0, density=1025.0
            )
            fields = {}
            for src in (getattr(an, "_asdict", lambda: {})(), vars(an) if hasattr(an, "__dict__") else {}):
                if src:
                    fields.update(src)
            a33 = b33 = None
            vol = float(np.asarray(fields.get("displaced_volume", 4.0 / 3.0 * math.pi)).ravel()[0])
            dimless = fields.get("heave_added_mass_dimless")
            dimful = fields.get("heave_added_mass")
            if dimless is not None:
                a33 = float(np.asarray(dimless).ravel()[0]) * 1025.0 * vol
            elif dimful is not None:
                a33 = float(np.asarray(dimful).ravel()[0]) * 1025.0 * vol
            if fields.get("heave_damping_dimless") is not None:
                b33 = float(np.asarray(fields["heave_damping_dimless"]).ravel()[0]) * 1025.0 * vol
            elif fields.get("heave_damping") is not None:
                b33 = float(np.asarray(fields["heave_damping"]).ravel()[0]) * 1025.0 * vol
            cp_inf = load("cpt-24x48").get(("inf", ka))
            rows[ka] = {"added_mass_heave": a33, "damping_heave": b33,
                        "displaced_volume": vol,
                        "raw_heave_added_mass": fields.get("heave_added_mass"),
                        "raw_heave_added_mass_dimless": fields.get("heave_added_mass_dimless"),
                        "cpt_inf_am_heave": cp_inf["added_mass_heave"] if cp_inf else None,
                        "am_rel_error_dimful_vs_cpt_inf":
                            _rel(a33, cp_inf["added_mass_heave"]) if (cp_inf and a33) else None,
                        "available_fields": sorted(fields)}
            if a33 is None:
                rows[ka] = {"added_mass_heave": None, "damping_heave": None,
                            "available_fields": sorted(fields)}
        return rows
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    tb2 = load("tbem-sub2")
    tb3 = load("tbem-sub3")
    tb2ea = load("tbem-sub2ea")
    cpt = {lvl: load(f"cpt-{lvl}") for lvl in ("8x16", "12x24", "16x32", "24x48")}
    fine, second = cpt["24x48"], cpt["16x32"]

    rows = []
    gates = {
        "G1_tbem_cpt_am_heave": {"threshold": G1, "passed": None, "max_error": 0.0, "n": 0},
        "G2_tbem_cpt_b_heave": {"threshold": G2, "passed": None, "max_error": 0.0, "n": 0},
        "G3_cpt_self_conv_am": {"threshold": G3, "passed": None, "max_error": 0.0, "n": 0},
        "G4_tbem_self_conv_am": {"threshold": G4, "passed": None, "max_error": 0.0, "n": 0},
        "G5_dw_limit_am": {"threshold": G5, "passed": None, "max_error": 0.0, "n": 0},
    }
    order = lambda k: (isinstance(k[0], str), str(k[0]), k[1])
    for key in sorted(set(tb2) | set(fine), key=order):
        station, ka = key
        row = {"depth_ratio": station, "ka": ka}
        tb, cp, cps = tb2.get(key), fine.get(key), second.get(key)
        if tb and cp:
            row.update({
                "tbem_am_heave": tb["added_mass_heave"], "cpt_am_heave": cp["added_mass_heave"],
                "am_rel_error": _rel(tb["added_mass_heave"], cp["added_mass_heave"]),
                "tbem_b_heave": tb["damping_heave"], "cpt_b_heave": cp["damping_heave"],
                "b_rel_error": _rel(tb["damping_heave"], cp["damping_heave"]),
                "tbem_n_panels": tb["n_panels"], "tbem_mode_count": tb.get("mode_count"),
                "cpt_n_faces": cp.get("n_faces"),
                "tbem_excitation_heave_mag": None,
            })
            for name, key_err in (("G1_tbem_cpt_am_heave", "am_rel_error"),
                                  ("G2_tbem_cpt_b_heave", "b_rel_error")):
                g = gates[name]
                g["max_error"] = max(g["max_error"], row[key_err]); g["n"] += 1
            if station == 20.0:
                g = gates["G5_dw_limit_am"]
                g["max_error"] = max(g["max_error"], row["am_rel_error"]); g["n"] += 1
        if cp and cps:
            row["cpt_self_conv_am"] = _rel(cp["added_mass_heave"], cps["added_mass_heave"])
            g = gates["G3_cpt_self_conv_am"]
            g["max_error"] = max(g["max_error"], row["cpt_self_conv_am"]); g["n"] += 1
        if tb and tb3.get(key):
            row["tbem_self_conv_am"] = _rel(tb3[key]["added_mass_heave"], tb["added_mass_heave"])
            g = gates["G4_tbem_self_conv_am"]
            g["max_error"] = max(g["max_error"], row["tbem_self_conv_am"]); g["n"] += 1
        if tb2ea.get(key):
            row["tbem_am_heave_exact_areas"] = tb2ea[key]["added_mass_heave"]
            if cp:
                row["am_rel_error_exact_areas"] = _rel(
                    tb2ea[key]["added_mass_heave"], cp["added_mass_heave"])
        rows.append(row)

    for name, g in gates.items():
        if g["n"] == 0:
            g["passed"] = False
            g["note"] = "not evaluated: no case pair in the cache (see study_status)"
        else:
            g["passed"] = bool(g["max_error"] <= g["threshold"])
            g["note"] = f"evaluated on {g['n']} case(s)"

    ana = analytic_inf()
    payload = {
        "case": "bem_finite_depth_capytaine_xval",
        "geometry": {"type": "submerged_sphere", "radius_m": 1.0, "centre_z_m": -2.0,
                     "bottom_z_m": -3.0,
                     "seabed_clearance_m": {"3.0": 0.0, "5.0": 2.0, "12.0": 9.0, "20.0": 17.0}},
        "parameters": {"ka_grid": list(KA_GRID),
                       "depth_ratios_measured": sorted({r["depth_ratio"] for r in rows}, key=str),
                       "density": 1025.0, "gravity": 9.80665,
                       "tbem_panels": {"2": 320, "3": 1280},
                       "cpt_faces": {"8x16": 128, "12x24": 288, "16x32": 512, "24x48": 1152}},
        "tbem_study": {"rows": [r for r in tb2.values()], "n": len(tb2),
                       "subdivisions_available": sorted({r["subdivisions"] for r in tb2.values()}),
                       "note": "subdivision 2 (320 panels); subdivision 3 (1280 panels) only at ka=1"},
        "cpt_study": {"levels": [{"resolution": lvl.split("x"), "rows": list(v.values())}
                                 for lvl, v in cpt.items()],
                      "n": sum(len(v) for v in cpt.values()),
                      "provenance": {"tool": "capytaine",
                                     "version": next(iter(fine.values())).get("capytaine_version", "unknown")
                                     if fine else "unknown"}},
        "analytic_infinite_depth": ana,
        "comparison": {"comparison": rows, "gates": gates,
                       "all_gates_passed": all(g["passed"] for g in gates.values())},
        "study_status": {
            "state": "partial",
            "measured": sorted({f"{r['depth_ratio']}/ka{r['ka']}" for r in rows})[:8] + ["..."],
            "missing": [
                "h/R = 4, 8 stations (never run; h/R = 3 in the cache has zero seabed clearance)",
                "deep-water TBEM leg (the inf station is unreachable through the finite-depth API "
                "because depth must be finite; run_cases.py routes it to solver free_surface_solver3d)",
                "subdivision 3 beyond ka = 1.0 (1280-panel solves are 630-1500 s each)",
                "capytaine 24x48 levels for h/R = 3.0 stations the crossval grid dropped",
            ],
            "blocking_defects": [
                "eigenseries truncation is not convergent: A33 at h/R=5, ka=1, 1280 panels is 9786 (M=48), "
                "463427 (M=64), -2175 (M=96) on the identical physical case",
                "numpy.linalg.LinAlgError: Eigenvalues did not converge in 20 of 44 cached TensorBEM cases",
            ],
        },
    }
    RESULT.write_text(json.dumps(payload, indent=2, default=str))

    print(f"{'h/R':>6} {'ka':>4} {'A33_tbem':>10} {'A33_cpt':>10} {'err%':>7} {'B33_tbem':>10} {'B33_cpt':>10} {'err%':>7}")
    for r in rows:
        if "am_rel_error" in r:
            print(f"{str(r['depth_ratio']):>6} {r['ka']:>4} {r['tbem_am_heave']:>10.1f} "
                  f"{r['cpt_am_heave']:>10.1f} {100*r['am_rel_error']:>7.2f} "
                  f"{r['tbem_b_heave']:>10.1f} {r['cpt_b_heave']:>10.1f} {100*r['b_rel_error']:>7.2f}")
    print("\ngates:")
    for name, g in gates.items():
        print(f"  {name}: passed={g['passed']} max_error={g['max_error']:.4f} thr={g['threshold']} n={g['n']}")
    print("\nanalytic:", json.dumps(ana, indent=1)[:600])
    print(f"\nwrote {RESULT}")


if __name__ == "__main__":
    main()
