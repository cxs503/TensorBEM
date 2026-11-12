#!/usr/bin/env python3
"""W25 (mission 16712) -- RAO truncation convergence vs the Journee (1992) anchor.

Three stages, each cached under ``runs/w25_rao_truncation/`` so the expensive
parts can be sharded:

``--stage sweep``
    Frequency-domain RAO of the Wigley III hull (the hull form of Journee 1992,
    report 09-V-09) in head seas, from the certified zero-speed 3-D
    free-surface Green-function BEM
    (:func:`tensorbem.rao.compute_rao_at_frequency`), on a fine nested
    frequency grid, for three mesh levels.

``--stage radius``
    Domain-truncation radius sweep of the *truncated-domain* Rankine free-surface
    panel method (:func:`tensorbem.rankine_fs.solve_rankine_free_surface`) on the
    same hull at the two mission Froude numbers.  This is the only solver in the
    package whose fluid domain is explicitly truncated, i.e. the only place where
    a geometric truncation radius exists as a real knob.

``--stage assemble``
    Reads the caches and writes
    ``papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json``
    with the per-condition tables, the criterion verdicts and the provenance
    (command / inputs / sha256).

Honesty rules baked in
----------------------
* Every number in the artifact is produced by one of the stages (or is a
  closed-form geometry value of the Wigley hull); no reference value is typed by
  hand.  The recalled Journee literals that live in the package are *imported*
  from their modules and tagged recalled/UNSOURCED: they are reported for
  diagnosis only and never enter a criterion verdict.
* The zero-speed limitation is declared, not papered over: the certified unsteady
  solver here is the zero-speed Green-function BEM, so the RAO axes are evaluated
  at U = 0 and the two mission Froude numbers are served by the steady
  truncated-domain solver.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks.rao_truncation import (  # noqa: E402
    band_statistics,
    block_coefficient,
    curve_peak,
    doubling_pairs,
    ittc_spectral_peak_omega,
    ittc_two_parameter_spectrum,
    relative_change_pct,
    relative_deviation_pct,
    truncation_radius_omega_min,
    wigley3_body_inertia,
)
from tensorbem.rao import compute_rao_at_frequency  # noqa: E402
from tensorbem.wigley3_hull import (  # noqa: E402
    WIGLEY3_ALPHA,
    WIGLEY3_B,
    WIGLEY3_CB,
    WIGLEY3_L,
    WIGLEY3_T,
    compute_wigley3_properties,
    wigley3_wetted_hull_mesh,
)

RHO = 1025.0
G = 9.80665

MISSION_ID = 16712
TASK_KEY = "paper-exp-seakeeping-truncation-5"

CACHE_DIR = ROOT / "runs" / "w25_rao_truncation"
ARTIFACT_PATH = (
    ROOT / "papers" / "artifacts" / "w25" / "seakeeping-truncation"
    / "journee_rao_truncation.json"
)

# --- predeclared study design (fixed before the production runs) ------------
OMEGA_MAX_LEVELS = (1.5, 3.0, 6.0, 12.0)         # exact doubling ladder [rad/s]
RADIUS_LAMBDA_LEVELS = (5.0, 10.0, 20.0, 40.0)   # lambda_max / L, doublings
SEA_STATES = (("ss_T1_1p00", 1.00), ("ss_T1_1p40", 1.40), ("ss_T1_1p80", 1.80))
HS_SEA = 1.0
MESH_LEVELS = (("coarse", 8, 4), ("production", 12, 6), ("fine", 16, 8))
PRODUCTION_MESH = "production"
GCI_OMEGAS = (1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 11.0)

RANKINE_Y_MAX_LEVELS = (1.0, 1.25, 2.0, 2.5)     # contains 1.0->2.0, 1.25->2.5
RANKINE_X_MAX_FACTOR = 2.5
RANKINE_PPW = 4.0
MISSION_FROUDES = (0.142, 0.20)

GATE_PEAK_RAO_PCT = 15.0
GATE_PEAK_FREQ_PCT = 15.0
GATE_CONVERGENCE_PCT = 3.0


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def git_state() -> dict:
    import subprocess

    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False
            )
            return out.stdout.strip()
        except Exception:  # pragma: no cover
            return ""

    return {
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "base_commit": run("git", "rev-parse", "HEAD"),
        "describe": run("git", "describe", "--always", "--dirty"),
    }


def fine_frequency_grid() -> np.ndarray:
    """Production sweep grid; every truncation edge lands exactly on a node."""
    base = list(np.round(np.arange(0.6, 8.0001, 0.2), 10))
    tail = [8.5, 9.2, 10.0, 11.0, 12.0]
    edges = list(OMEGA_MAX_LEVELS)
    edges += [truncation_radius_omega_min(r, WIGLEY3_L, G) for r in RADIUS_LAMBDA_LEVELS]
    edges += list(GCI_OMEGAS)
    grid = np.unique(np.round(np.array(base + tail + edges, dtype=float), 10))
    return grid[(grid >= 0.6) & (grid <= 12.0)]


# ---------------------------------------------------------------------------
# hull model
# ---------------------------------------------------------------------------

def hull_model(n_x_q: int = 801, n_z_q: int = 401) -> dict:
    """Geometry, mass and inertia of the Journee (1992) Wigley III hull."""
    props = compute_wigley3_properties(L=WIGLEY3_L, B=WIGLEY3_B, T=WIGLEY3_T, rho=RHO)
    inertia = wigley3_body_inertia(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_ALPHA, RHO, n_x=n_x_q, n_z=n_z_q
    )
    volume = inertia["volume"]
    cb_quadrature = WIGLEY3_T + inertia["z_centroid"]
    cg = np.array([0.0, 0.0, -cb_quadrature], dtype=float)
    mass = RHO * volume
    moments = np.diag([inertia["Ixx"], inertia["Iyy"], inertia["Izz"]])
    r2 = float(cg @ cg)
    inertia_origin = moments + mass * (r2 * np.eye(3) - np.outer(cg, cg))
    return {
        "L": WIGLEY3_L,
        "B": WIGLEY3_B,
        "T": WIGLEY3_T,
        "alpha": WIGLEY3_ALPHA,
        "rho": RHO,
        "gravity": G,
        "Cb_closed_form": WIGLEY3_CB,
        "Cb_quadrature": block_coefficient(volume, WIGLEY3_L, WIGLEY3_B, WIGLEY3_T),
        "volume_closed_form": props.volume,
        "volume_quadrature": volume,
        "waterplane_area_closed_form": props.waterplane_area,
        "Ixx_waterplane_closed_form": props.Ixx,
        "Iyy_waterplane_closed_form": props.Iyy,
        "KB_closed_form": props.KB,
        "KB_quadrature": cb_quadrature,
        "mass": mass,
        "cg": cg,
        "inertia_about_cg_diag": [inertia["Ixx"], inertia["Iyy"], inertia["Izz"]],
        "inertia_about_origin": inertia_origin,
        "gyration_radius_over_L_pitch": math.sqrt(inertia["Iyy"] / mass) / WIGLEY3_L,
        "gyration_radius_over_B_roll": math.sqrt(inertia["Ixx"] / mass) / WIGLEY3_B,
    }


def _mesh(level: str):
    for name, nx, nz in MESH_LEVELS:
        if name == level:
            return wigley3_wetted_hull_mesh(nx=nx, nz=nz)
    raise ValueError(f"unknown mesh level {level!r}")


_SHARDS = 1


def shard_indices(n_items: int, shard: int) -> list[int]:
    """Deterministic strided assignment (1-based shard)."""
    return list(range(max(0, int(shard) - 1), n_items, max(1, _SHARDS)))


# ---------------------------------------------------------------------------
# stage: sweep
# ---------------------------------------------------------------------------

def rao_at(model: dict, mesh, omega: float, wave_angle: float) -> dict:
    t0 = time.time()
    res = compute_rao_at_frequency(
        mesh,
        omega,
        wave_angle,
        mass=model["mass"],
        center_of_gravity=model["cg"],
        inertia=model["inertia_about_origin"],
        density=RHO,
        gravity=G,
        chief_points="auto",
    )
    return {
        "omega": float(omega),
        "wave_angle_deg": math.degrees(float(wave_angle)),
        "heave_rao": float(res.rao_amplitude[2]),
        "pitch_rao": float(res.rao_amplitude[4]),
        "impedance_residual": float(res.impedance_residual),
        "seconds": round(time.time() - t0, 3),
    }


def stage_sweep(shard: int = 1, shards: int = 1):
    global _SHARDS
    _SHARDS = int(shards)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = hull_model()
    grid = fine_frequency_grid()
    out = {
        "stage": "sweep",
        "generated_utc": utc_now(),
        "shard": int(shard),
        "shards": int(shards),
        "grid": grid.tolist(),
        "gci_omegas": list(GCI_OMEGAS),
        "head_seas_wave_angle_deg": 180.0,
        "head_seas_beta_invariance_check": [],
        "levels": {},
    }
    for name, nx, nz in MESH_LEVELS:
        mesh = _mesh(name)
        omegas = grid if name == PRODUCTION_MESH else np.array(GCI_OMEGAS, dtype=float)
        rows = [
            rao_at(model, mesh, float(omegas[i]), math.pi)
            for i in shard_indices(len(omegas), shard)
        ]
        out["levels"][name] = {
            "nx": nx,
            "nz": nz,
            "panel_count": int(mesh.panel_count),
            "waterplane_area": float(mesh.waterplane_area),
            "rows": rows,
        }
        if name == PRODUCTION_MESH and int(shard) == 1:
            for om in (1.5, 3.5):
                a = rao_at(model, mesh, om, math.pi)
                b = rao_at(model, mesh, om, 0.0)
                out["head_seas_beta_invariance_check"].append(
                    {
                        "omega": om,
                        "heave_pi": a["heave_rao"],
                        "heave_0": b["heave_rao"],
                        "pitch_pi": a["pitch_rao"],
                        "pitch_0": b["pitch_rao"],
                        "heave_rel_change_pct": relative_change_pct(
                            a["heave_rao"], b["heave_rao"]
                        ),
                        "pitch_rel_change_pct": relative_change_pct(
                            a["pitch_rao"], b["pitch_rao"]
                        ),
                    }
                )
    path = CACHE_DIR / f"sweep_shard{shard}of{shards}.json"
    path.write_text(json.dumps(out, indent=1))
    return path


# ---------------------------------------------------------------------------
# stage: radius (geometric domain truncation, steady Rankine free surface)
# ---------------------------------------------------------------------------

def stage_radius(shard: int = 1, shards: int = 1):
    from tensorbem.rankine_fs import solve_rankine_free_surface

    global _SHARDS
    _SHARDS = int(shards)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cases = [(fn, lv) for fn in MISSION_FROUDES for lv in RANKINE_Y_MAX_LEVELS]
    rows = []
    for idx in shard_indices(len(cases), shard):
        froude, y_max = cases[idx]
        u = float(froude) * math.sqrt(G * WIGLEY3_L)
        t0 = time.time()
        res = solve_rankine_free_surface(
            WIGLEY3_L,
            WIGLEY3_B,
            WIGLEY3_T,
            u,
            rho=RHO,
            g=G,
            panels_per_wavelength=RANKINE_PPW,
            x_max_factor=RANKINE_X_MAX_FACTOR,
            y_max_factor=float(y_max),
        )
        zeta = getattr(res, "zeta", None)
        rows.append(
            {
                "froude": float(froude),
                "speed_m_s": u,
                "y_max_factor": float(y_max),
                "x_max_factor": RANKINE_X_MAX_FACTOR,
                "panels_per_wavelength": RANKINE_PPW,
                "n_unknowns": int(res.n_unknowns),
                "rw_pressure": float(res.rw_pressure),
                "rw_farfield": float(res.rw_farfield),
                "cw_pressure": float(res.cw_pressure),
                "wave_elevation_max": (
                    float(np.max(np.abs(np.asarray(zeta, dtype=float))))
                    if zeta is not None
                    else float("nan")
                ),
                "seconds": round(time.time() - t0, 3),
            }
        )
    out = {
        "stage": "radius",
        "generated_utc": utc_now(),
        "shard": int(shard),
        "shards": int(shards),
        "rows": rows,
    }
    path = CACHE_DIR / f"radius_shard{shard}of{shards}.json"
    path.write_text(json.dumps(out, indent=1))
    return path


# ---------------------------------------------------------------------------
# merge helpers
# ---------------------------------------------------------------------------

def load_cache(prefix: str) -> list:
    files = sorted(CACHE_DIR.glob(f"{prefix}_shard*of*.json"))
    if not files:
        raise SystemExit(f"no cache files for {prefix!r} in {CACHE_DIR}")
    return [json.loads(p.read_text()) for p in files]


def merge_sweep(chunks: list) -> dict:
    merged = {"levels": {}, "grid": [], "gci_omegas": [], "beta_checks": []}
    for ch in chunks:
        merged["grid"] = ch["grid"]
        merged["gci_omegas"] = ch["gci_omegas"]
        merged["beta_checks"].extend(ch.get("head_seas_beta_invariance_check", []))
        for name, blk in ch["levels"].items():
            slot = merged["levels"].setdefault(
                name,
                {
                    "nx": blk["nx"],
                    "nz": blk["nz"],
                    "panel_count": blk["panel_count"],
                    "waterplane_area": blk["waterplane_area"],
                    "rows": [],
                },
            )
            slot["rows"].extend(blk["rows"])
    for slot in merged["levels"].values():
        slot["rows"].sort(key=lambda r: r["omega"])
    return merged


def merge_radius(chunks: list) -> list:
    rows = [r for ch in chunks for r in ch["rows"]]
    rows.sort(key=lambda r: (r["froude"], r["y_max_factor"]))
    return rows


def _rao_arrays(rows: list):
    om = np.array([r["omega"] for r in rows], dtype=float)
    heave = np.array([r["heave_rao"] for r in rows], dtype=float)
    pitch = np.array([r["pitch_rao"] for r in rows], dtype=float)
    return om, heave, pitch


# ---------------------------------------------------------------------------
# truncation axes
# ---------------------------------------------------------------------------

def axis_truncation_frequency(omega, heave, pitch) -> dict:
    rows = []
    for ss_name, t1 in SEA_STATES:
        spec = ittc_two_parameter_spectrum(omega, HS_SEA, t1, G)
        for om_max in OMEGA_MAX_LEVELS:
            sh = band_statistics(omega, heave, spec, 0.0, om_max)
            sp = band_statistics(omega, pitch, spec, 0.0, om_max)
            rows.append(
                {
                    "sea_state": ss_name,
                    "t1": t1,
                    "omega_p_spectrum": ittc_spectral_peak_omega(t1),
                    "omega_max": om_max,
                    "sigma_heave": sh.sigma_band,
                    "sigma_pitch": sp.sigma_band,
                    "area_fraction_heave": sh.area_fraction,
                    "area_fraction_pitch": sp.area_fraction,
                    "peak_heave_in_band": sh.peak_value,
                    "peak_heave_omega": sh.peak_omega,
                    "peak_pitch_in_band": sp.peak_value,
                    "peak_inside_band": bool(sh.peak_inside_band and sp.peak_inside_band),
                    "sigma_heave_full_band": sh.sigma_full,
                    "sigma_pitch_full_band": sp.sigma_full,
                }
            )
    verdicts = {}
    for ss_name, _t1 in SEA_STATES:
        sub = [r for r in rows if r["sea_state"] == ss_name]
        for key in ("sigma_heave", "sigma_pitch"):
            changes = []
            for lo, hi in doubling_pairs(OMEGA_MAX_LEVELS):
                vlo = next(r[key] for r in sub if r["omega_max"] == lo)
                vhi = next(r[key] for r in sub if r["omega_max"] == hi)
                changes.append(
                    {
                        "pair": [lo, hi],
                        "value_lo": vlo,
                        "value_hi": vhi,
                        "rel_change_pct": relative_change_pct(vlo, vhi),
                    }
                )
            verdicts[f"{ss_name}:{key}"] = {
                "doubling_pairs": changes,
                "max_rel_change_pct": max(c["rel_change_pct"] for c in changes),
                "gate_pct": GATE_CONVERGENCE_PCT,
                "pass": bool(
                    all(c["rel_change_pct"] < GATE_CONVERGENCE_PCT for c in changes)
                ),
            }
    return {"rows": rows, "doubling_verdicts": verdicts}


def axis_truncation_radius_longwave(omega, heave, pitch) -> dict:
    rows = []
    for ss_name, t1 in SEA_STATES:
        spec = ittc_two_parameter_spectrum(omega, HS_SEA, t1, G)
        for r_lam in RADIUS_LAMBDA_LEVELS:
            om_min = truncation_radius_omega_min(r_lam, WIGLEY3_L, G)
            sh = band_statistics(omega, heave, spec, om_min, 12.0)
            sp = band_statistics(omega, pitch, spec, om_min, 12.0)
            rows.append(
                {
                    "sea_state": ss_name,
                    "t1": t1,
                    "radius_lambda_over_L": r_lam,
                    "lambda_max_m": r_lam * WIGLEY3_L,
                    "omega_min": om_min,
                    "sigma_heave": sh.sigma_band,
                    "sigma_pitch": sp.sigma_band,
                    "area_fraction_heave": sh.area_fraction,
                    "area_fraction_pitch": sp.area_fraction,
                    "peak_heave_in_band": sh.peak_value,
                    "peak_heave_omega": sh.peak_omega,
                    "peak_pitch_in_band": sp.peak_value,
                    "peak_inside_band": bool(sh.peak_inside_band and sp.peak_inside_band),
                }
            )
    verdicts = {}
    for ss_name, _t1 in SEA_STATES:
        sub = [r for r in rows if r["sea_state"] == ss_name]
        for key in ("sigma_heave", "sigma_pitch"):
            changes = []
            for lo, hi in doubling_pairs(RADIUS_LAMBDA_LEVELS):
                vlo = next(r[key] for r in sub if r["radius_lambda_over_L"] == lo)
                vhi = next(r[key] for r in sub if r["radius_lambda_over_L"] == hi)
                changes.append(
                    {
                        "pair": [lo, hi],
                        "value_lo": vlo,
                        "value_hi": vhi,
                        "rel_change_pct": relative_change_pct(vlo, vhi),
                    }
                )
            verdicts[f"{ss_name}:{key}"] = {
                "doubling_pairs": changes,
                "max_rel_change_pct": max(c["rel_change_pct"] for c in changes),
                "gate_pct": GATE_CONVERGENCE_PCT,
                "pass": bool(
                    all(c["rel_change_pct"] < GATE_CONVERGENCE_PCT for c in changes)
                ),
            }
    return {"rows": rows, "doubling_verdicts": verdicts}


def axis_truncation_radius_geometric(rows: list) -> dict:
    out = {"rows": rows, "per_froude": {}}
    for froude in MISSION_FROUDES:
        sub = [r for r in rows if r["froude"] == froude]
        if not sub:
            continue
        sub.sort(key=lambda r: r["y_max_factor"])
        energy = []
        for r in sub:
            denom = abs(r["rw_pressure"])
            r["energy_gate_pct"] = (
                100.0 * abs(r["rw_pressure"] - r["rw_farfield"]) / denom
                if denom > 0.0
                else float("nan")
            )
            energy.append(r["energy_gate_pct"])
        changes = []
        for lo, hi in doubling_pairs(RANKINE_Y_MAX_LEVELS):
            vlo = next(r["cw_pressure"] for r in sub if r["y_max_factor"] == lo)
            vhi = next(r["cw_pressure"] for r in sub if r["y_max_factor"] == hi)
            changes.append(
                {
                    "pair": [lo, hi],
                    "cw_lo": vlo,
                    "cw_hi": vhi,
                    "rel_change_pct": relative_change_pct(vlo, vhi),
                }
            )
        out["per_froude"][str(froude)] = {
            "speed_m_s": sub[0]["speed_m_s"],
            "doubling_pairs": changes,
            "max_rel_change_pct": max(c["rel_change_pct"] for c in changes),
            "gate_pct": GATE_CONVERGENCE_PCT,
            "pass": bool(
                all(c["rel_change_pct"] < GATE_CONVERGENCE_PCT for c in changes)
            ),
            "energy_gate_pct_range": [min(energy), max(energy)],
            "n_unknowns_range": [
                min(r["n_unknowns"] for r in sub),
                max(r["n_unknowns"] for r in sub),
            ],
        }
    return out


# ---------------------------------------------------------------------------
# anchor ledger / verification / criteria
# ---------------------------------------------------------------------------

def recalled_anchor_ledger() -> dict:
    """Import (never retype) the recalled Journee literals and their provenance."""
    import importlib

    from tensorbem import reference_provenance

    entries = getattr(
        reference_provenance, "ENTRIES", getattr(reference_provenance, "_ENTRIES", {})
    )
    ledger = {
        "source_modules": [],
        "literals": {},
        "provenance_notes": {},
        "registry_size": len(entries) if hasattr(entries, "__len__") else None,
    }
    targets = (
        (
            "tensorbem.benchmarks.wigley3_complete_verification",
            ("_JOURNEE_HEAVE_RAO_PEAK", "_JOURNEE_HEAVE_RAO_PEAK_OMEGA"),
        ),
        ("tensorbem.benchmarks.wigley3_freq_verification", ("_JOURNEE_RAO_PEAKS",)),
    )
    for mod_name, attrs in targets:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # pragma: no cover
            ledger["source_modules"].append(
                {"module": mod_name, "import_error": repr(exc)}
            )
            continue
        entry = {"module": mod_name, "attributes": {}}
        for attr in attrs:
            if not hasattr(mod, attr):
                continue
            value = getattr(mod, attr)
            if isinstance(value, np.generic):
                value = value.item()
            entry["attributes"][attr] = value
            key = f"{mod_name}.{attr}"
            ledger["literals"][key] = value
            note = None
            for probe in (key, attr.lstrip("_")):
                if isinstance(entries, dict) and probe in entries:
                    item = entries[probe]
                    note = {
                        "grade": str(getattr(item, "grade", "")),
                        "note": getattr(item, "note", None),
                    }
                    break
            ledger["provenance_notes"][key] = note
        ledger["source_modules"].append(entry)
    return ledger


def anchor_section() -> dict:
    return {
        "requested_anchor": {
            "citation_as_assigned": "Journee (1992) KVLCC2 model-test RAO (heave/pitch)",
            "verdict": "WITHHELD",
            "reasons": [
                {
                    "kind": "chronology",
                    "statement": (
                        "The KVLCC2 hull form postdates the cited source: the package cites "
                        "Journee (1992) as 'Experiments and Calculations on 4 Wigley Hull "
                        "Forms in Head Waves', Report 09-V-09, DUT (docstring of "
                        "tensorbem.wigley3_hull).  A 1992 report cannot contain KVLCC2 "
                        "experiments, so the assigned anchor is internally inconsistent and "
                        "no such table can be transcribed."
                    ),
                },
                {
                    "kind": "not_shipped",
                    "statement": (
                        "No experimental RAO table is shipped in this repository.  "
                        "tensorbem.reference_provenance registers every Journee literal used "
                        "by the package as recalled/UNSOURCED with the note that the source "
                        "report is not shipped; papers/reports/w20_seakeeping_truncation4.md "
                        "records an earlier attempt to obtain the table as an honest FAIL."
                    ),
                },
                {
                    "kind": "no_network",
                    "statement": (
                        "Outbound network access is blocked in the execution sandbox, so the "
                        "report cannot be retrieved during the run."
                    ),
                },
            ],
            "consequence": (
                "Criteria (1) peak-RAO deviation < 15 % and (2) peak-frequency offset "
                "< 15 % are not adjudicable against the assigned anchor and are reported "
                "WITHHELD; they are not replaced by a recalled number."
            ),
        },
        "substitute_evidence": {
            "analytic": [
                "long-wave Froude-Krylov limit: heave RAO -> 1, pitch RAO -> 0 as omega -> 0",
                "closed-form Wigley geometry: Cb = 4/9 + 4 alpha/45, V = L B T Cb, "
                "A_wp = 2 L B / 3, Ixx_wp = 4 B^3 L / 105, Iyy_wp = B L^3 / 30, "
                "KB = T [1 - (15 + 5 alpha)/(40 + 8 alpha)] (= 5T/8 at alpha = 0)",
                "ITTC (1978) two-parameter spectrum zeroth moment m0 = Hs^2 / 16",
            ],
            "numerical": [
                "three-mesh discretisation study of the same RAO curve",
                "head-seas convention check: |RAO| invariant under beta -> beta + pi",
                "truncated-domain Rankine solver: pressure vs far-field wave-resistance "
                "energy gate",
            ],
        },
        "recalled_literals_are_diagnostic_only": True,
    }


def verification_section(merged: dict, model: dict) -> dict:
    out = {"geometry": {}, "physics": {}, "discretisation": {}}
    props = compute_wigley3_properties()
    out["geometry"] = {
        "L": WIGLEY3_L,
        "B": WIGLEY3_B,
        "T": WIGLEY3_T,
        "alpha": WIGLEY3_ALPHA,
        "Cb_closed_form": WIGLEY3_CB,
        "Cb_quadrature": model["Cb_quadrature"],
        "Cb_quadrature_deviation_pct": relative_deviation_pct(
            model["Cb_quadrature"], WIGLEY3_CB
        ),
        "volume_closed_form_m3": model["volume_closed_form"],
        "volume_quadrature_m3": model["volume_quadrature"],
        "volume_deviation_pct": relative_deviation_pct(
            model["volume_quadrature"], model["volume_closed_form"]
        ),
        "KB_closed_form_m": props.KB,
        "KB_quadrature_m": model["KB_quadrature"],
        "KB_deviation_pct": relative_deviation_pct(model["KB_quadrature"], props.KB),
        "KB_over_T": props.KB / WIGLEY3_T,
        "KB_over_T_at_alpha_zero": compute_wigley3_properties(alpha=0.0).KB / WIGLEY3_T,
        "KB_note": (
            "KB_over_T_at_alpha_zero must be 0.625 (classical 5T/8).  The package formula "
            "returned 0.8125 before this mission's fix of tensorbem.wigley3_hull."
        ),
        "mesh_waterplane_area": {
            name: blk["waterplane_area"] for name, blk in merged["levels"].items()
        },
        "mesh_panel_counts": {
            name: blk["panel_count"] for name, blk in merged["levels"].items()
        },
        "waterplane_area_closed_form": props.waterplane_area,
        "gyration_radius_over_L_pitch": model["gyration_radius_over_L_pitch"],
        "gyration_radius_over_B_roll": model["gyration_radius_over_B_roll"],
        "mass_kg": model["mass"],
        "cg_m": [float(v) for v in model["cg"]],
    }
    prod = merged["levels"][PRODUCTION_MESH]["rows"]
    omega, heave, pitch = _rao_arrays(prod)
    low = int(np.argmin(omega))
    ph = curve_peak(omega, heave)
    pp = curve_peak(omega, pitch)
    out["physics"] = {
        "omega_min_evaluated": float(omega[low]),
        "lambda_over_L_at_omega_min": float(
            2.0 * math.pi * G / omega[low] ** 2 / WIGLEY3_L
        ),
        "heave_rao_at_omega_min": float(heave[low]),
        "heave_long_wave_deviation_pct": relative_deviation_pct(float(heave[low]), 1.0),
        "pitch_rao_at_omega_min": float(pitch[low]),
        "heave_peak": {"omega": ph.omega, "value": ph.value, "refined": ph.refined},
        "pitch_peak": {"omega": pp.omega, "value": pp.value, "refined": pp.refined},
        "heave_peak_omega_sqrt_L_over_g": float(ph.omega * math.sqrt(WIGLEY3_L / G)),
        "pitch_peak_omega_sqrt_L_over_g": float(pp.omega * math.sqrt(WIGLEY3_L / G)),
        "max_abs_impedance_residual": float(
            max(abs(r["impedance_residual"]) for r in prod)
        ),
        "beta_invariance": merged["beta_checks"],
    }
    gci_rows = []
    key_omegas = sorted({float(r["omega"]) for r in merged["levels"]["coarse"]["rows"]})
    for om in key_omegas:
        row = {"omega": om}
        for name in ("coarse", PRODUCTION_MESH, "fine"):
            hit = next(
                (r for r in merged["levels"][name]["rows"] if abs(r["omega"] - om) < 1e-9),
                None,
            )
            row[f"heave_{name}"] = None if hit is None else hit["heave_rao"]
            row[f"pitch_{name}"] = None if hit is None else hit["pitch_rao"]
        complete = all(
            row.get(f"heave_{n}") is not None and row.get(f"pitch_{n}") is not None
            for n in ("coarse", PRODUCTION_MESH, "fine")
        )
        if complete:
            row["heave_coarse_to_production_pct"] = relative_change_pct(
                row["heave_coarse"], row["heave_production"]
            )
            row["heave_production_to_fine_pct"] = relative_change_pct(
                row["heave_production"], row["heave_fine"]
            )
            row["pitch_coarse_to_production_pct"] = relative_change_pct(
                row["pitch_coarse"], row["pitch_production"]
            )
            row["pitch_production_to_fine_pct"] = relative_change_pct(
                row["pitch_production"], row["pitch_fine"]
            )
        gci_rows.append(row)
    out["discretisation"] = {
        "rows": gci_rows,
        "note": (
            "Panel counts are not in a constant refinement ratio, so no formal GCI index is "
            "quoted; the successive-mesh changes bound the discretisation error of the RAO "
            "curve on which the truncation study is built."
        ),
    }
    return out


def assembly_detail(omega, heave, pitch) -> dict:
    """Direct answer to criterion (3) for the RAO curve itself."""
    out = {"peak_definition": "grid maximum with 3-point parabolic refinement"}
    rows = []
    for lo, hi in doubling_pairs(OMEGA_MAX_LEVELS):
        m_lo = omega <= lo
        m_hi = omega <= hi
        ph_lo = curve_peak(omega[m_lo], heave[m_lo])
        ph_hi = curve_peak(omega[m_hi], heave[m_hi])
        pp_lo = curve_peak(omega[m_lo], pitch[m_lo])
        pp_hi = curve_peak(omega[m_hi], pitch[m_hi])
        rows.append(
            {
                "pair_omega_max": [lo, hi],
                "peak_heave_lo": ph_lo.value,
                "peak_heave_hi": ph_hi.value,
                "peak_heave_rel_change_pct": relative_change_pct(ph_lo.value, ph_hi.value),
                "peak_heave_omega_lo": ph_lo.omega,
                "peak_heave_omega_hi": ph_hi.omega,
                "peak_pitch_lo": pp_lo.value,
                "peak_pitch_hi": pp_hi.value,
                "peak_pitch_rel_change_pct": relative_change_pct(pp_lo.value, pp_hi.value),
            }
        )
    ph = curve_peak(omega, heave)
    pp = curve_peak(omega, pitch)
    out["doubling_pairs"] = rows
    out["full_band_peak_heave"] = {"omega": ph.omega, "value": ph.value}
    out["full_band_peak_pitch"] = {"omega": pp.omega, "value": pp.value}
    vals = [r["peak_heave_rel_change_pct"] for r in rows]
    vals += [r["peak_pitch_rel_change_pct"] for r in rows]
    finite = [v for v in vals if v == v]
    out["max_peak_rel_change_pct"] = max(finite) if finite else float("nan")
    out["gate_pct"] = GATE_CONVERGENCE_PCT
    out["pass"] = bool(out["max_peak_rel_change_pct"] < GATE_CONVERGENCE_PCT)
    out["caveat"] = (
        "Pointwise RAO peaks are interior to the band, so they cannot move once the band "
        "edge is above the resonance region; the meaningful truncation target is the "
        "band-limited response variance (axes A and B).  Pairs whose lower level cuts "
        "through the resonance region are reported as FAIL."
    )
    return out


def diagnostic_against_recalled(omega, heave, pitch, ledger: dict) -> dict:
    ph = curve_peak(omega, heave)
    pp = curve_peak(omega, pitch)
    lit = ledger["literals"]
    heave_peak_ref = None
    heave_omega_ref = None
    pitch_peak_ref = None
    for key, value in lit.items():
        if "HEAVE_RAO_PEAK_OMEGA" in key and isinstance(value, (int, float)):
            heave_omega_ref = float(value)
        elif key.endswith("_JOURNEE_HEAVE_RAO_PEAK") and isinstance(value, (int, float)):
            heave_peak_ref = float(value)
        elif key.endswith("_JOURNEE_RAO_PEAKS") and isinstance(value, dict):
            pitch_peak_ref = float(value.get("4", float("nan")))
            if heave_peak_ref is None:
                heave_peak_ref = float(value.get("2", float("nan")))
    return {
        "label": "DIAGNOSTIC ONLY -- recalled literals, UNSOURCED, not an accepted anchor",
        "computed_peak_heave": {"omega": ph.omega, "value": ph.value},
        "computed_peak_pitch": {"omega": pp.omega, "value": pp.value},
        "recalled_heave_peak": heave_peak_ref,
        "recalled_heave_peak_omega": heave_omega_ref,
        "recalled_pitch_peak": pitch_peak_ref,
        "heave_peak_deviation_pct": (
            relative_deviation_pct(ph.value, heave_peak_ref)
            if heave_peak_ref
            else float("nan")
        ),
        "heave_peak_omega_offset_pct": (
            relative_deviation_pct(ph.omega, heave_omega_ref)
            if heave_omega_ref
            else float("nan")
        ),
        "pitch_peak_deviation_pct": (
            relative_deviation_pct(pp.value, pitch_peak_ref)
            if pitch_peak_ref
            else float("nan")
        ),
        "verdict": "NOT ACCEPTED AS CRITERION EVIDENCE (reference is recalled, unsourced)",
    }


def criteria_section(axis_freq, axis_rad, axis_geom, detail, diagnostics) -> dict:
    # Criterion (3) is a statement about the RAO, so only the RAO-side axes enter
    # its verdict.  The geometric-domain axis C is reported separately: it is a
    # steady resistance functional whose own energy gate fails (see the artifact),
    # so folding it into an "RAO convergence" verdict would be misleading.
    rao_axes = {
        "truncation_frequency_upper_band": axis_freq["doubling_verdicts"],
        "truncation_radius_long_wave": axis_rad["doubling_verdicts"],
    }
    geom_axes = {
        k: {
            "max_rel_change_pct": v["max_rel_change_pct"],
            "gate_pct": v["gate_pct"],
            "pass": v["pass"],
            "doubling_pairs": v["doubling_pairs"],
            "energy_gate_pct_range": v.get("energy_gate_pct_range"),
        }
        for k, v in axis_geom["per_froude"].items()
    }
    all_pass = True
    worst = 0.0
    for group in rao_axes.values():
        for entry in group.values():
            val = entry["max_rel_change_pct"]
            if val == val:
                worst = max(worst, val)
            if not entry["pass"]:
                all_pass = False
    axes = dict(rao_axes)
    if detail["max_peak_rel_change_pct"] == detail["max_peak_rel_change_pct"]:
        worst = max(worst, detail["max_peak_rel_change_pct"])
    if not detail["pass"]:
        all_pass = False
    # The lowest truncation level (1.5 rad/s) sits below the pitch resonance, so its
    # doubling pair is expected to fail; the verdict must say which range converged
    # instead of hiding either fact.
    failing_pairs = []
    for group in rao_axes.values():
        for key, entry in group.items():
            for ch in entry["doubling_pairs"]:
                if not ch["rel_change_pct"] < GATE_CONVERGENCE_PCT:
                    failing_pairs.append({"group": key, "pair": ch["pair"],
                                          "rel_change_pct": ch["rel_change_pct"]})
    peak_fails = not detail["pass"]
    n_pairs = sum(len(v["doubling_pairs"]) for g in rao_axes.values() for v in g.values())
    n_pass = n_pairs - len(failing_pairs)
    top_lo = max(lv for lv in OMEGA_MAX_LEVELS if lv * 2.0 <= max(OMEGA_MAX_LEVELS))
    top_fail = [p for p in failing_pairs if p["pair"][0] == top_lo]
    n_top = sum(1 for g in rao_axes.values() for v in g.values()
                for ch in v["doubling_pairs"] if ch["pair"][0] == top_lo)
    if not failing_pairs and not peak_fails:
        verdict = "PASS (every doubling pair and the peak sensitivity < 3 %)"
    else:
        verdict = (
            f"FAIL as an all-pairs criterion ({n_pass}/{n_pairs} doubling pairs < 3 %); the "
            f"highest doubling {top_lo}->{top_lo * 2} rad/s passes in "
            f"{n_top - len(top_fail)}/{n_top} channel-sea-state combinations"
            + (
                f" and fails marginally at {max(p['rel_change_pct'] for p in top_fail):.2f} %"
                if top_fail
                else " (all pass)"
            )
            + "."
        )
    return {
        "criterion_1_peak_rao_vs_anchor": {
            "gate_pct": GATE_PEAK_RAO_PCT,
            "verdict": "WITHHELD",
            "reason": "assigned anchor unavailable (see anchor.requested_anchor.reasons)",
            "recalled_literal_diagnostic": diagnostics,
        },
        "criterion_2_peak_frequency_vs_anchor": {
            "gate_pct": GATE_PEAK_FREQ_PCT,
            "verdict": "WITHHELD",
            "reason": "assigned anchor unavailable (see anchor.requested_anchor.reasons)",
        },
        "criterion_3_doubling_convergence": {
            "gate_pct": GATE_CONVERGENCE_PCT,
            "verdict": verdict,
            "all_pairs_pass": bool(all_pass and not peak_fails),
            "failing_doubling_pairs": failing_pairs,
            "highest_doubling_pair_evaluated": [top_lo, top_lo * 2],
            "highest_pair_failures": len(top_fail),
            "worst_rel_change_pct": worst,
            "rao_peak_sensitivity": detail,
            "axes_rao_side": axes,
            "axis_C_steady_domain_radius": {
                "verdict": "REPORTED SEPARATELY - not an RAO criterion",
                "per_froude": geom_axes,
                "caveat": (
                    "The steady Rankine solution fails its own energy consistency gate "
                    "(pressure vs far-field wave resistance) at these Froude numbers, so its "
                    "domain-radius convergence is a property of that functional, not evidence "
                    "about RAO truncation."
                ),
            },
            "scope_note": (
                "Axes 'truncation_frequency_upper_band' and 'truncation_radius_long_wave' "
                "are evaluated on the certified zero-speed 3-D RAO; the axis "
                "'truncation_radius_geometric_domain' is evaluated on the truncated-domain "
                "Rankine solver (steady wave resistance), because the package has neither an "
                "unsteady truncated-domain solver nor a forward-speed unsteady free-surface "
                "solver."
            ),
        },
    }


def capability_section() -> dict:
    return {
        "used": [
            {
                "key": "rao-frequency-domain-3d",
                "module": "tensorbem.rao + tensorbem.free_surface_solver3d",
                "level": "REAL",
                "why": "zero-speed 3-D free-surface Green-function BEM with CHIEF suppression",
            },
            {
                "key": "rankine-free-surface-truncated-domain",
                "module": "tensorbem.rankine_fs",
                "level": "REAL",
                "why": "only solver whose fluid domain is truncated, hence the only place "
                       "where a geometric truncation radius is a real knob",
            },
        ],
        "gaps": [
            {
                "key": "unsteady-forward-speed-bem",
                "level": "NOT_SUPPORTED",
                "statement": (
                    "tensorbem.rao / tensorbem.free_surface_solver3d evaluate the zero-speed "
                    "Green function only (no encounter-frequency kernel and no "
                    "Havelock-with-speed kernel), so the Fr = 0.142 / 0.20 RAO of the mission "
                    "cannot be produced by the certified unsteady route."
                ),
            },
            {
                "key": "stf-forward-speed-rao",
                "level": "PARTIAL (assembly only)",
                "statement": (
                    "tensorbem.stf_fullspeed supplies the STF assembly, the speed-correction "
                    "terms and circular-section coefficients, but no hull excitation-force "
                    "builder: assemble_stf_coefficients takes excitation_forces as a caller "
                    "input, so an end-to-end hull RAO at forward speed cannot be produced "
                    "without new, unverified physics."
                ),
            },
            {
                "key": "unsteady-truncated-domain-solver",
                "level": "NOT_SUPPORTED",
                "statement": (
                    "The truncated-domain Rankine panel method is steady (Neumann-Kelvin); "
                    "there is no unsteady free-surface patch solver, hence no radius-truncation "
                    "experiment on an RAO."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# stage: assemble
# ---------------------------------------------------------------------------

def stage_assemble():
    sweep_chunks = load_cache("sweep")
    radius_chunks = load_cache("radius")
    merged = merge_sweep(sweep_chunks)
    radius_rows = merge_radius(radius_chunks)
    model = hull_model()

    omega, heave, pitch = _rao_arrays(merged["levels"][PRODUCTION_MESH]["rows"])
    axis_freq = axis_truncation_frequency(omega, heave, pitch)
    axis_rad = axis_truncation_radius_longwave(omega, heave, pitch)
    axis_geom = axis_truncation_radius_geometric(radius_rows)
    detail = assembly_detail(omega, heave, pitch)
    ledger = recalled_anchor_ledger()
    diagnostics = diagnostic_against_recalled(omega, heave, pitch, ledger)
    verification = verification_section(merged, model)

    inputs = {
        "script": {
            "path": str(pathlib.Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(pathlib.Path(__file__).resolve()),
        },
        "code": {},
        "data": {},
    }
    for rel in (
        "src/tensorbem/rao.py",
        "src/tensorbem/free_surface_solver3d.py",
        "src/tensorbem/free_surface3d.py",
        "src/tensorbem/hydrostatic.py",
        "src/tensorbem/rankine_fs.py",
        "src/tensorbem/wigley3_hull.py",
        "src/tensorbem/benchmarks/rao_truncation.py",
        "src/tensorbem/reference_provenance.py",
    ):
        p = ROOT / rel
        if p.is_file():
            inputs["code"][rel] = _sha256(p)
    for p in sorted(CACHE_DIR.glob("*.json")):
        inputs["data"][str(p.relative_to(ROOT))] = _sha256(p)

    commands = []
    for ch in sweep_chunks:
        commands.append(
            f"python run_w25_rao_truncation.py --stage sweep --shard {ch['shard']}/{ch['shards']}"
        )
    for ch in radius_chunks:
        commands.append(
            f"python run_w25_rao_truncation.py --stage radius --shard {ch['shard']}/{ch['shards']}"
        )
    commands.append("python run_w25_rao_truncation.py --stage assemble")

    artifact = {
        "schema": "w25-journee-rao-truncation/v1",
        "task": TASK_KEY,
        "mission_id": MISSION_ID,
        "repo": "TensorBEM",
        "generated_utc": utc_now(),
        "git": git_state(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "provenance": {
            "queue_note": (
                "The sweep was also submitted to the cluster (job cj1341, 8 shards, stage_dir "
                "snapshot of this work tree); the queue reported 500 jobs ahead and the job was "
                "still queued when the identical local sharded run finished, so the local "
                "caches were used and the cluster job was cancelled.  Cache sha256 values below "
                "identify exactly the data behind every number."
            ),
            "driver_copy_at_cluster_submit": (
                "runs/w25_rao_truncation/driver_at_submit_cj1341.py"
            ),
            "post_submit_code_changes": [
                "tensorbem.benchmarks.rao_truncation.band_statistics: band edges now keep a "
                "1e-9 relative rounding slack (not called by the sweep stage)",
                "this driver: criteria_section aggregates only the RAO-side axes for criterion "
                "(3) and reports the steady geometric axis separately (assemble stage only)",
            ],
            "commands": commands,
            "inputs": inputs,
            "zero_hand_copying": (
                "Every numeric leaf of this artifact is produced by the stages above, by a "
                "closed-form expression evaluated at run time, or is imported from a package "
                "module (recalled reference literals, tagged UNSOURCED)."
            ),
        },
        "case": {
            "hull": (
                "Wigley III (Journee 1992, Report 09-V-09 model geometry, L/B = 10, B/T = 1.6)"
            ),
            "reference_frame": {
                "origin": "still-water plane, amidships",
                "z_positive": "up",
                "omega": "circular wave frequency [rad/s]",
                "head_seas": "wave_angle = 180 deg (waves propagate along -x, bow at +x)",
            },
            "model_parameters": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in model.items()
                if k not in ("inertia_about_origin", "cg")
            },
            "inertia_about_origin": model["inertia_about_origin"].tolist(),
            "fluid": {"rho": RHO, "gravity": G},
            "sea_states": [
                {
                    "name": name,
                    "hs_m": HS_SEA,
                    "t1_s": t1,
                    "omega_p": ittc_spectral_peak_omega(t1),
                    "spectrum": "ITTC (1978) two-parameter, m0 = Hs^2/16",
                }
                for name, t1 in SEA_STATES
            ],
            "study_design": {
                "truncation_frequency_levels_rad_s": list(OMEGA_MAX_LEVELS),
                "truncation_radius_levels_lambda_over_L": list(RADIUS_LAMBDA_LEVELS),
                "geometric_radius_levels_y_max_over_L": list(RANKINE_Y_MAX_LEVELS),
                "geometric_radius_x_max_over_L": RANKINE_X_MAX_FACTOR,
                "geometric_radius_panels_per_wavelength": RANKINE_PPW,
                "mission_froudes": list(MISSION_FROUDES),
                "mesh_levels": [
                    {"name": n, "nx": nx, "nz": nz} for n, nx, nz in MESH_LEVELS
                ],
                "frequency_grid_points": int(omega.size),
                "gates_pct": {
                    "peak_rao": GATE_PEAK_RAO_PCT,
                    "peak_frequency": GATE_PEAK_FREQ_PCT,
                    "doubling_change": GATE_CONVERGENCE_PCT,
                },
            },
        },
        "capabilities": capability_section(),
        "anchor": anchor_section(),
        "anchor_recall_ledger": ledger,
        "verification": verification,
        "rao_table": {
            "mesh_level": PRODUCTION_MESH,
            "columns": [
                "omega_rad_s",
                "omega_sqrt_L_over_g",
                "heave_rao_m_per_m",
                "pitch_rao_rad_per_m",
            ],
            "rows": [
                [
                    float(o),
                    float(o * math.sqrt(WIGLEY3_L / G)),
                    float(h),
                    float(p),
                ]
                for o, h, p in zip(omega, heave, pitch)
            ],
        },
        "axis_A_truncation_frequency": axis_freq,
        "axis_B_truncation_radius_long_wave": axis_rad,
        "axis_C_truncation_radius_geometric": axis_geom,
        "criteria": criteria_section(axis_freq, axis_rad, axis_geom, detail, diagnostics),
        "limitations": [
            "The certified unsteady RAO solver is zero-speed; forward speed (Fr = 0.142 / "
            "0.20) is NOT_SUPPORTED for RAO in this package and is exercised only on the "
            "steady truncated-domain solver (wave-resistance coefficient).",
            "No experimental anchor is available in this environment (see anchor section), "
            "so criteria (1) and (2) are WITHHELD rather than estimated.",
            "The 3-D Green-function RAO has no domain truncation; the 'truncation radius' of "
            "the RAO axes is therefore a wavelength-space radius (lambda_max/L), and the "
            "geometric radius is measured on the truncated-domain Rankine solver instead.",
            "ITTC (1978) spectrum with Hs = 1 m: sigma values are transfer-function "
            "properties and scale linearly with Hs.",
        ],
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=1))
    print(f"wrote {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size} bytes)")
    return ARTIFACT_PATH


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sweep", "radius", "assemble"), required=True)
    parser.add_argument("--shard", default="1", help="shard index, or i/n shorthand")
    parser.add_argument("--shards", type=int, default=None)
    args = parser.parse_args(argv)
    shard, shards = args.shard, args.shards
    if isinstance(shard, str) and "/" in shard:
        left, right = shard.split("/", 1)
        shard, shards = int(left), int(right)
    shard = int(shard)
    shards = int(shards) if shards is not None else 1
    if args.stage == "sweep":
        path = stage_sweep(shard=shard, shards=shards)
    elif args.stage == "radius":
        path = stage_radius(shard=shard, shards=shards)
    else:
        path = stage_assemble()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
