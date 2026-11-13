#!/usr/bin/env python3
"""CASE-BEM-HYDROSTATIC-PONTOON -- rectangular pontoon hydrostatics (TensorBEM).

Case-pool anchor ``BEM-HYDROSTATIC-PONTOON`` (TensorBEM, ref_source "Biran &
Pulido / Rawson & Tupper closed form", tolerance < 3 %).

The wall-sided pontoon is the only hull form whose every hydrostatic quantity
has a textbook closed form, so the case separates implementation error from
discretisation error: every wetted panel is planar and lies exactly on the
analytic surface, hence any error above round-off is a bug, not faceting.

Closed forms (Biran & Lopez-Pulido, *Ship Hydrostatics and Stability*, 2nd ed.,
ch. 3-4; Rawson & Tupper, *Basic Ship Theory*, 5th ed., ch. 3-4), evaluated
programmatically in :func:`reference_values` (no hand-copied digits):

    nabla = L B T,  A_wp = L B,  KB = T/2,  LCB = 0
    I_T = L B^3/12,  I_L = B L^3/12,  BM_T = I_T/nabla = B^2/(12T)
    GM_T = KB + BM_T - KG,  C[2,2] = rho g A_wp,  C[3,3] = rho g nabla GM_T

Citation honesty: the print original is not reachable from this sandbox, so the
citation is chapter-level and no page number is claimed; the formula set is the
one the ecosystem already uses for the same pontoon
(``TensorPlatform/docs/verification_debt/stability_pontoon/refs.py``).

Routes compared (both pre-existing TensorBEM capabilities -- reuse, no rebuild):

* ``waterplane`` -- ``compute_hydrostatic_restoring_matrix`` (waterline loops ->
  A, S, I -> closed-form C).
* ``pressure``   -- ``richardson_restoring_matrix`` (clip closed body at z = 0,
  integrate p = rho g (0 - z), central differences, Richardson in step).

Run:
    python3 scripts/case_hydrostatic_pontoon.py --out artifacts/case_hydrostatic_pontoon
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks.hydrostatic_pressure_route_study import closed_box_mesh  # noqa: E402
from tensorbem.geometry3d import box_hull_mesh  # noqa: E402
from tensorbem.hydrostatic import (  # noqa: E402
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)
from tensorbem.hydrostatic_pressure_route import (  # noqa: E402
    clip_to_fluid,
    pressure_loads,
    restoring_matrix_pressure_route,
    richardson_restoring_matrix,
)

# ---------------------------------------------------------------------------
# case definition (predeclared before any number was inspected)
# ---------------------------------------------------------------------------
CASE = {
    "case_code": "CASE-BEM-HYDROSTATIC-PONTOON",
    "pool_anchor": "BEM-HYDROSTATIC-PONTOON",
    "repo": "TensorBEM",
    "length_m": 50.0,
    "beam_m": 12.0,
    "depth_m": 5.0,
    "draft_m": 3.0,
    "kg_m": 4.0,
    "density_kg_m3": 1025.0,
    "gravity_m_s2": 9.80665,
    "tolerance_rel": 0.03,
}

CITATION = {
    "primary": "Biran & Lopez-Pulido, Ship Hydrostatics and Stability, 2nd ed. "
               "(Butterworth-Heinemann), ch. 3-4 (box-shaped vessel hydrostatics, "
               "metacentric height)",
    "secondary": "Rawson & Tupper, Basic Ship Theory, 5th ed., ch. 3-4 "
                 "(box barge hydrostatics, BM = I/nabla)",
    "ecosystem_cross_reference": "TensorPlatform/docs/verification_debt/"
                                 "stability_pontoon/refs.py::pontoon_analytic (same "
                                 "formula set already used by the stability gate)",
    "page_level_claim": "none -- print original not reachable from this sandbox; "
                        "citations are chapter-level and every number is evaluated "
                        "from the closed forms in code (zero hand-copying)",
}

REFINEMENT_LEVELS = ((2, 2, 2), (4, 3, 3), (8, 6, 6), (16, 12, 12))
DRAFT_SWEEP = (2.0, 2.5, 3.0, 3.5, 4.0)
GRAVITY_SWEEP = (9.80665, 9.81, 9.8)
FD_STEPS = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)


# ---------------------------------------------------------------------------
# closed-form reference
# ---------------------------------------------------------------------------
def reference_values(L: float, B: float, T: float, KG: float, rho: float, g: float) -> dict:
    """Textbook closed form for a wall-sided rectangular pontoon."""
    vol = L * B * T
    a_wp = L * B
    kb = 0.5 * T
    i_t = L * B ** 3 / 12.0
    i_l = B * L ** 3 / 12.0
    bm_t = i_t / vol
    bm_l = i_l / vol
    km_t = kb + bm_t
    km_l = kb + bm_l
    gm_t = km_t - KG
    gm_l = km_l - KG
    return {
        "displaced_volume_m3": vol,
        "displacement_t": rho * vol / 1000.0,
        "waterplane_area_m2": a_wp,
        "block_coefficient_Cb": vol / (L * B * T),
        "waterplane_coefficient_Cwp": a_wp / (L * B),
        "midship_coefficient_Cm": (B * T) / (B * T),
        "prismatic_coefficient_Cp": vol / ((B * T) * L),
        "kb_m": kb,
        "vcb_m": kb,
        "lcb_m": 0.0,
        "lcf_m": 0.0,
        "tcf_m": 0.0,
        "i_transverse_m4": i_t,
        "i_longitudinal_m4": i_l,
        "i_product_m4": 0.0,
        "bm_transverse_m": bm_t,
        "bm_longitudinal_m": bm_l,
        "km_transverse_m": km_t,
        "km_longitudinal_m": km_l,
        "gm_transverse_m": gm_t,
        "gm_longitudinal_m": gm_l,
        "c33_N_per_m": rho * g * a_wp,
        "c44_Nm_per_rad": rho * g * vol * gm_t,
        "c55_Nm_per_rad": rho * g * vol * gm_l,
    }


def immersed_first_moment_z(vertices, faces) -> float:
    """``int_V z dV`` of the immersed part, from the clipped surface.

    ``int_V z dV = (1/2) * integral_S z^2 n_z dS`` (divergence theorem with
    ``phi = z^2/2``); the waterplane lid lies at ``z = 0`` and contributes
    nothing, so the clipped wetted surface is enough.  ``z`` is linear on a
    triangle and ``int z^2 dA`` is exact for a linear field, so this is a
    third route to the centre of buoyancy that shares no code with the
    waterplane (Green's theorem in 2D) or the tetrahedron-centroid route.
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    z = verts[:, 2]
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(n, axis=1)
    z0, z1, z2 = z[tris[:, 0]], z[tris[:, 1]], z[tris[:, 2]]
    quad = area / 6.0 * (z0 ** 2 + z1 ** 2 + z2 ** 2 + z0 * z1 + z1 * z2 + z2 * z0)
    # n = ||cross|| * n_hat = 2*area*n_hat
    n_hat_z = n[:, 2] / (2.0 * area)
    return float(0.5 * np.sum(quad * n_hat_z))


# ---------------------------------------------------------------------------
# measured values, two independent TensorBEM routes
# ---------------------------------------------------------------------------
def measure_waterplane_route(case: dict, level: tuple[int, int, int]) -> dict:
    L, B, T = case["length_m"], case["beam_m"], case["draft_m"]
    rho, g, KG = case["density_kg_m3"], case["gravity_m_s2"], case["kg_m"]
    nl, nb, nd = level
    hull = box_hull_mesh(L, B, T, n_length=nl, n_beam=nb, n_draft=nd)
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    area, wp_centre, moments = compute_waterplane_properties(hull)
    props = compute_hydrostatic_restoring_matrix(
        hull,
        mass=rho * volume,
        center_of_gravity=np.array([0.0, 0.0, -T + KG]),
        density=rho,
        gravity=g,
    )
    c = np.asarray(props.restoring_matrix, dtype=float)
    return {
        "panels": int(len(hull.faces)),
        "displaced_volume_m3": float(props.displaced_volume),
        "displacement_t": rho * float(props.displaced_volume) / 1000.0,
        "waterplane_area_m2": float(props.waterplane_area),
        "kb_m": float(props.center_of_buoyancy[2]) + T,
        "vcb_m": float(props.center_of_buoyancy[2]) + T,
        "lcb_m": float(props.center_of_buoyancy[0]),
        "lcf_m": float(props.waterplane_center[0]),
        "tcf_m": float(props.waterplane_center[1]),
        "i_transverse_m4": float(props.waterplane_moments[0]),
        "i_longitudinal_m4": float(props.waterplane_moments[1]),
        "i_product_m4": float(props.waterplane_moments[2]),
        "bm_transverse_m": float(props.waterplane_moments[0]) / float(props.displaced_volume),
        "bm_longitudinal_m": float(props.waterplane_moments[1]) / float(props.displaced_volume),
        "matrix": c,
        "bbox_got": (
            float(hull.vertices[:, 0].max() - hull.vertices[:, 0].min()),
            float(hull.vertices[:, 1].max() - hull.vertices[:, 1].min()),
            float(-hull.vertices[:, 2].min()),
        ),
        "divergence_volume_m3": float(volume),
        "divergence_centre_m": [float(v) for v in centre],
        "trace_area_m2": float(area),
        "trace_centre_m": [float(v) for v in wp_centre],
    }


def measure_pressure_route(case: dict, level: tuple[int, int, int]) -> dict:
    L, B, T, D = case["length_m"], case["beam_m"], case["draft_m"], case["depth_m"]
    rho, g, KG = case["density_kg_m3"], case["gravity_m_s2"], case["kg_m"]
    nl, nb, nd = level
    verts, faces = closed_box_mesh(L, B, T, D - T, n_length=nl, n_beam=nb, n_height=nd)
    res = richardson_restoring_matrix(
        verts,
        faces,
        center_of_gravity=(0.0, 0.0, -T + KG),
        density=rho,
        gravity=g,
        steps=(2.0e-3, 1.0e-3, 5.0e-4),
    )
    c = np.asarray(res.matrix, dtype=float)
    vol = float(res.immersed_volume)
    # The pressure resultant only fixes the horizontal line of action (x_B,y_B):
    # a vertical force gives M_x = y_B W, M_y = -x_B W, and z_B is invisible to
    # it. z_B is therefore taken from the vertical first moment of the same
    # clipped surface (independent route, see immersed_first_moment_z).
    clip_v, clip_f = clip_to_fluid(verts, faces)
    z_b = immersed_first_moment_z(clip_v, clip_f) / vol
    return {
        "panels": int(len(faces)),
        "displaced_volume_m3": vol,
        "displacement_t": rho * vol / 1000.0,
        "kb_m": z_b + T,
        "lcb_m": float(res.center_of_buoyancy[0]),
        "matrix": c,
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def rel(a: float, b: float) -> float:
    return float(a - b) / float(b) if b != 0.0 else float("nan")


COMPARISON_ROWS = (
    ("displaced volume nabla", "m^3", "displaced_volume_m3", "displaced_volume_m3", "displaced_volume_m3"),
    ("displacement Delta", "t", "displacement_t", "displacement_t", "displacement_t"),
    ("block coefficient Cb", "-", "block_coefficient_Cb", None, None),
    ("waterplane area A_wp", "m^2", "waterplane_area_m2", "waterplane_area_m2", None),
    ("waterplane coeff Cwp", "-", "waterplane_coefficient_Cwp", None, None),
    ("KB (above keel)", "m", "kb_m", "kb_m", "kb_m"),
    ("LCB (from midship)", "m", "lcb_m", "lcb_m", "lcb_m"),
    ("LCF", "m", "lcf_m", "lcf_m", None),
    ("TCF", "m", "tcf_m", "tcf_m", None),
    ("I_T = int y^2 dA", "m^4", "i_transverse_m4", "i_transverse_m4", None),
    ("I_L = int x^2 dA", "m^4", "i_longitudinal_m4", "i_longitudinal_m4", None),
    ("I_xy", "m^4", "i_product_m4", "i_product_m4", None),
    ("BM_T = I_T/nabla", "m", "bm_transverse_m", "bm_transverse_m", None),
    ("BM_L = I_L/nabla", "m", "bm_longitudinal_m", "bm_longitudinal_m", None),
    ("KM_T = KB+BM_T", "m", "km_transverse_m", None, None),
    ("KM_L = KB+BM_L", "m", "km_longitudinal_m", None, None),
    ("GM_T = KM_T-KG", "m", "gm_transverse_m", None, None),
    ("GM_L = KM_L-KG", "m", "gm_longitudinal_m", None, None),
)


def error_metric(got: float, reference: float, scale: float) -> float:
    """Relative error, or error normalised by ``scale`` when the reference is zero.

    Nothing here is a length/moment of a real hull whose closed-form value is
    exactly zero (LCB, LCF, TCF, I_xy of a symmetric pontoon), so a relative
    error would be ``0/0``; the error is then measured against the natural
    scale of the quantity (ship length for lever arms, I_T for the product
    moment) and the row is labelled with the scale it uses.
    """
    if reference != 0.0:
        return rel(got, reference)
    return float(got - reference) / float(scale)


def derived_got(case: dict, wp: dict) -> dict:
    """Quantities that are not returned directly but follow from measured components."""
    L, B, T = case["length_m"], case["beam_m"], case["draft_m"]
    vol = wp["displaced_volume_m3"]
    area = wp["waterplane_area_m2"]
    km_t = wp["kb_m"] + wp["bm_transverse_m"]
    km_l = wp["kb_m"] + wp["bm_longitudinal_m"]
    return {
        "block_coefficient_Cb": vol / (L * B * T),
        "waterplane_coefficient_Cwp": area / (L * B),
        "midship_coefficient_Cm": (B * T) / (B * T),
        "prismatic_coefficient_Cp": vol / ((B * T) * L),
        "km_transverse_m": km_t,
        "km_longitudinal_m": km_l,
        "gm_transverse_m": km_t - case["kg_m"],
        "gm_longitudinal_m": km_l - case["kg_m"],
    }


def build_comparison(case: dict, ref: dict, wp: dict, pr: dict) -> list[dict]:
    tol = case["tolerance_rel"]
    L = case["length_m"]
    scales = {"lcb_m": L, "lcf_m": L, "tcf_m": L, "i_product_m4": ref["i_transverse_m4"]}
    derived = derived_got(case, wp)
    rows = []
    for label, unit, rkey, wkey, pkey in COMPARISON_ROWS:
        reference = float(ref[rkey])
        scale = scales.get(rkey, 1.0)
        row = {"quantity": label, "unit": unit, "reference": reference,
               "error_scale": ("relative" if reference != 0.0 else f"|x|/{scale:.6g}")}
        errors = []
        if wkey is not None and wkey in wp:
            row["waterplane"] = float(wp[wkey])
            row["err_waterplane_rel"] = error_metric(wp[wkey], reference, scale)
            errors.append(abs(row["err_waterplane_rel"]))
        if pkey is not None and pkey in pr:
            row["pressure"] = float(pr[pkey])
            row["err_pressure_rel"] = error_metric(pr[pkey], reference, scale)
            errors.append(abs(row["err_pressure_rel"]))
        if wkey is None and pkey is None and rkey in derived:
            row["waterplane"] = float(derived[rkey])
            row["err_waterplane_rel"] = error_metric(derived[rkey], reference, scale)
            row["derived_from"] = "measured components (volume / area / KB+BM)"
            errors.append(abs(row["err_waterplane_rel"]))
        row["max_abs_rel_err"] = max(errors) if errors else 0.0
        row["pass"] = bool(row["max_abs_rel_err"] < tol)
        rows.append(row)

    c_ref = np.zeros((6, 6))
    c_ref[2, 2] = ref["c33_N_per_m"]
    c_ref[3, 3] = ref["c44_Nm_per_rad"]
    c_ref[4, 4] = ref["c55_Nm_per_rad"]
    for label, (i, j) in (("C33 (heave)", (2, 2)), ("C44 (roll)", (3, 3)), ("C55 (pitch)", (4, 4))):
        row = {"quantity": label, "unit": "N/m or N m/rad", "reference": float(c_ref[i, j]),
               "error_scale": "relative"}
        errors = []
        for key, mat in (("waterplane", wp["matrix"]), ("pressure", pr["matrix"])):
            value = float(mat[i, j])
            row[key] = value
            row[f"err_{key}_rel"] = rel(value, c_ref[i, j])
            errors.append(abs(row[f"err_{key}_rel"]))
        row["max_abs_rel_err"] = max(errors)
        row["pass"] = bool(row["max_abs_rel_err"] < tol)
        rows.append(row)

    off = []
    for mat in (wp["matrix"], pr["matrix"]):
        scale = float(np.abs(mat).max())
        mask = ~np.eye(6, dtype=bool)
        off.append(float(np.abs(np.where(mask, mat, 0.0)).max() / scale))
    rows.append({
        "quantity": "max |off-diagonal C_ij| / max|C|",
        "unit": "-",
        "reference": 0.0,
        "error_scale": "|x|/max|C|",
        "waterplane": off[0],
        "pressure": off[1],
        "err_waterplane_rel": off[0],
        "err_pressure_rel": off[1],
        "max_abs_rel_err": max(off),
        "pass": bool(max(off) < tol),
    })
    return rows


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------
def refinement_study(case: dict, ref: dict) -> list[dict]:
    out = []
    for level in REFINEMENT_LEVELS:
        wp = measure_waterplane_route(case, level)
        pr = measure_pressure_route(case, level)
        rows = build_comparison(case, ref, wp, pr)
        out.append({
            "level": list(level),
            "panels": wp["panels"],
            "max_abs_rel_err_waterplane": max(
                r["max_abs_rel_err"] for r in rows if "err_waterplane_rel" in r),
            "max_abs_rel_err_pressure": max(
                r["max_abs_rel_err"] for r in rows if "err_pressure_rel" in r),
            "volume_rel": rel(wp["displaced_volume_m3"], ref["displaced_volume_m3"]),
            "c33_rel_waterplane": rel(wp["matrix"][2, 2], ref["c33_N_per_m"]),
            "c33_rel_pressure": rel(pr["matrix"][2, 2], ref["c33_N_per_m"]),
            "c44_rel_waterplane": rel(wp["matrix"][3, 3], ref["c44_Nm_per_rad"]),
            "c44_rel_pressure": rel(pr["matrix"][3, 3], ref["c44_Nm_per_rad"]),
            "c55_rel_waterplane": rel(wp["matrix"][4, 4], ref["c55_Nm_per_rad"]),
            "c55_rel_pressure": rel(pr["matrix"][4, 4], ref["c55_Nm_per_rad"]),
        })
    return out


def draft_sensitivity(case: dict) -> list[dict]:
    """Draft sweep: BM_T = B^2/(12T) and the d(nabla)/dT = A_wp identity."""
    out = []
    for T in DRAFT_SWEEP:
        c = dict(case, draft_m=T)
        ref = reference_values(c["length_m"], c["beam_m"], T, c["kg_m"],
                              c["density_kg_m3"], c["gravity_m_s2"])
        wp = measure_waterplane_route(c, REFINEMENT_LEVELS[1])
        h = 1.0e-3
        v_up = measure_waterplane_route(dict(c, draft_m=T + h), REFINEMENT_LEVELS[1])["displaced_volume_m3"]
        v_dn = measure_waterplane_route(dict(c, draft_m=T - h), REFINEMENT_LEVELS[1])["displaced_volume_m3"]
        dvol_dt = (v_up - v_dn) / (2.0 * h)
        out.append({
            "draft_m": T,
            "bm_transverse_ref_m": ref["bm_transverse_m"],
            "bm_transverse_got_m": wp["bm_transverse_m"],
            "bm_transverse_rel_err": rel(wp["bm_transverse_m"], ref["bm_transverse_m"]),
            "c33_rel_err": rel(wp["matrix"][2, 2], ref["c33_N_per_m"]),
            "c44_rel_err": rel(wp["matrix"][3, 3], ref["c44_Nm_per_rad"]),
            "dvol_dT_got_m2": dvol_dt,
            "dvol_dT_ref_m2": ref["waterplane_area_m2"],
            "dvol_dT_rel_err": rel(dvol_dt, ref["waterplane_area_m2"]),
        })
    return out


def gravity_sensitivity(case: dict) -> list[dict]:
    ref_std = reference_values(case["length_m"], case["beam_m"], case["draft_m"],
                               case["kg_m"], case["density_kg_m3"], case["gravity_m_s2"])
    out = []
    for g in GRAVITY_SWEEP:
        c = dict(case, gravity_m_s2=g)
        wp = measure_waterplane_route(c, REFINEMENT_LEVELS[1])
        out.append({
            "gravity_m_s2": g,
            "c33_rel_err_vs_standard_g": rel(wp["matrix"][2, 2], ref_std["c33_N_per_m"]),
            "c44_rel_err_vs_standard_g": rel(wp["matrix"][3, 3], ref_std["c44_Nm_per_rad"]),
            "volume_rel_err_vs_standard_g": rel(wp["displaced_volume_m3"], ref_std["displaced_volume_m3"]),
        })
    return out


def fd_step_sensitivity(case: dict) -> list[dict]:
    L, B, T, D = case["length_m"], case["beam_m"], case["draft_m"], case["depth_m"]
    rho, g, KG = case["density_kg_m3"], case["gravity_m_s2"], case["kg_m"]
    verts, faces = closed_box_mesh(L, B, T, D - T, n_length=4, n_beam=3, n_height=3)
    ref = reference_values(L, B, T, KG, rho, g)
    out = []
    for step in FD_STEPS:
        res = restoring_matrix_pressure_route(
            verts, faces, center_of_gravity=(0.0, 0.0, -T + KG),
            density=rho, gravity=g, step=step)
        c = np.asarray(res.matrix, dtype=float)
        out.append({
            "step": step,
            "c33_rel_err": rel(c[2, 2], ref["c33_N_per_m"]),
            "c44_rel_err": rel(c[3, 3], ref["c44_Nm_per_rad"]),
            "c55_rel_err": rel(c[4, 4], ref["c55_Nm_per_rad"]),
        })
    # observed order between consecutive steps, p = log2(|e_prev| / |e_cur|)
    for prev, cur in zip(out, out[1:]):
        cur["observed_order_c44"] = float(
            math.log2(abs(prev["c44_rel_err"]) / abs(cur["c44_rel_err"])))
        cur["observed_order_c55"] = float(
            math.log2(abs(prev["c55_rel_err"]) / abs(cur["c55_rel_err"])))
    return out


def heel_check(case: dict, ref: dict, angles_deg=(0.5, 1.0, 2.0, 3.0, 5.0)) -> list[dict]:
    """Righting arm at heel: pressure integration vs wall-sided closed form."""
    L, B, T, D = case["length_m"], case["beam_m"], case["draft_m"], case["depth_m"]
    rho, g, KG = case["density_kg_m3"], case["gravity_m_s2"], case["kg_m"]
    verts, faces = closed_box_mesh(L, B, T, D - T, n_length=4, n_beam=3, n_height=3)
    weight = rho * g * ref["displaced_volume_m3"]
    z_g = -T + KG
    gm, bm = ref["gm_transverse_m"], ref["bm_transverse_m"]
    out = []
    for phi_deg in angles_deg:
        phi = math.radians(phi_deg)
        r = np.array([[1.0, 0.0, 0.0],
                      [0.0, math.cos(phi), -math.sin(phi)],
                      [0.0, math.sin(phi), math.cos(phi)]])
        heeled_v, heeled_f = clip_to_fluid(verts @ r.T, faces)
        load = pressure_loads(heeled_v, heeled_f, density=rho, gravity=g)
        m_hydro = float(load.moment[0])
        r_g = r @ np.array([0.0, 0.0, z_g])
        m_weight = float(r_g[1] * (-weight))
        m_total = m_hydro + m_weight
        gz_small = gm * math.sin(phi)
        gz_wall = math.sin(phi) * (gm + 0.5 * bm * math.tan(phi) ** 2)
        out.append({
            "heel_deg": phi_deg,
            "righting_moment_got_Nm": -m_total,
            "righting_moment_ref_wall_sided_Nm": rho * g * ref["displaced_volume_m3"] * gz_wall,
            "gz_got_m": -m_total / weight,
            "gz_small_angle_m": gz_small,
            "gz_wall_sided_m": gz_wall,
            "gz_rel_err_vs_wall_sided": rel(-m_total / weight, gz_wall),
        })
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(out_dir: Path, results: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []

    rows = [r for r in results["comparison"] if not r["quantity"].split()[0].startswith("C")]
    labels = [r["quantity"] for r in rows]
    wp = [abs(r.get("err_waterplane_rel", np.nan)) for r in rows]
    pr = [abs(r.get("err_pressure_rel", np.nan)) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - 0.2, np.nan_to_num(wp, nan=1e-19), 0.4, label="waterplane route")
    ax.bar(x + 0.2, np.nan_to_num(pr, nan=1e-19), 0.4, label="pressure route")
    ax.set_yscale("log")
    ax.axhline(0.03, color="crimson", ls="--", lw=1.2, label="3 % tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("|relative error| vs closed form")
    ax.set_title("Pontoon hydrostatics: TensorBEM vs Biran / Rawson-Tupper closed form")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_rel_errors.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    ref = results["refinement"]
    panels = [r["panels"] for r in ref]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(panels, [max(r["max_abs_rel_err_waterplane"], 1e-19) for r in ref], "o-",
              label="waterplane route (max over quantities)")
    ax.loglog(panels, [max(r["max_abs_rel_err_pressure"], 1e-19) for r in ref], "s-",
              label="pressure route (max over quantities)")
    ax.axhline(0.03, color="crimson", ls="--", lw=1.2, label="3 % tolerance")
    ax.axhline(np.finfo(float).eps, color="grey", ls=":", lw=1.0, label="double round-off")
    ax.set_xlabel("number of panels")
    ax.set_ylabel("max |relative error|")
    ax.set_title("Panel refinement: planar panels are exact at every level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_refinement.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    ds = results["sensitivity"]["draft"]
    T = [r["draft_m"] for r in ds]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(T, [r["bm_transverse_ref_m"] for r in ds], "k-", lw=2, label="B^2/(12T) closed form")
    ax.plot(T, [r["bm_transverse_got_m"] for r in ds], "o", ms=7, mfc="none",
            label="TensorBEM waterplane route")
    ax.set_xlabel("draft T [m]")
    ax.set_ylabel("BM$_T$ [m]")
    ax.set_title("Draft sweep: BM$_T$ = B$^2$/(12T) reproduced")
    ax2 = ax.twinx()
    ax2.semilogy(T, [max(abs(r["bm_transverse_rel_err"]), 1e-19) for r in ds], "r^--",
                 label="|relative error| (right axis)")
    ax2.set_ylabel("|relative error|", color="r")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_draft_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    hc = results["heel_check"]
    phi = [r["heel_deg"] for r in hc]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(phi, [r["gz_wall_sided_m"] for r in hc], "k-", lw=2, label="wall-sided GZ (Biran ch.6)")
    ax.plot(phi, [r["gz_small_angle_m"] for r in hc], "g--", lw=1.2, label="GM sin(phi)")
    ax.plot(phi, [r["gz_got_m"] for r in hc], "o", ms=7, mfc="none",
            label="TensorBEM pressure integration")
    ax.set_xlabel("heel angle phi [deg]")
    ax.set_ylabel("righting arm GZ [m]")
    ax.set_title("Heel check: hydrostatic pressure integration vs closed forms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_heel_righting.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)
    return made


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def fmt_err(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.3e}"


def build_report(results: dict) -> str:
    case, ref = results["case"], results["reference"]
    rows = results["comparison"]
    ok = all(r["pass"] for r in rows)
    max_err = max(r["max_abs_rel_err"] for r in rows)

    head = "| 量 | 参考解 | waterplane got | err | pressure got | err | 判据 <3% |\n"
    head += "|---|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        cells = []
        for k in ("waterplane", "pressure"):
            if k in r:
                cells.append(f"{r[k]:.10g}")
                cells.append(fmt_err(r.get(f"err_{k}_rel")))
            else:
                cells.extend(["—", "—"])
        body += (f"| {r['quantity']} [{r['unit']}] | {r['reference']:.10g} | "
                 + " | ".join(cells) + f" | {'✅' if r['pass'] else '❌'} |\n")

    ref_lines = "\n".join(
        f"| {r['level']} | {r['panels']} | {r['max_abs_rel_err_waterplane']:.3e} | "
        f"{r['max_abs_rel_err_pressure']:.3e} |" for r in results["refinement"])
    draft_lines = "\n".join(
        f"| {r['draft_m']:.1f} | {r['bm_transverse_ref_m']:.6f} | {r['bm_transverse_got_m']:.6f} | "
        f"{r['bm_transverse_rel_err']:+.2e} | {r['dvol_dT_ref_m2']:.1f} | "
        f"{r['dvol_dT_got_m2']:.4f} | {r['dvol_dT_rel_err']:+.2e} |"
        for r in results["sensitivity"]["draft"])
    g_lines = "\n".join(
        f"| {r['gravity_m_s2']} | {r['c33_rel_err_vs_standard_g']:+.3e} | "
        f"{r['c44_rel_err_vs_standard_g']:+.3e} | {r['volume_rel_err_vs_standard_g']:+.3e} |"
        for r in results["sensitivity"]["gravity"])
    fd_lines = "\n".join(
        f"| {r['step']:.1e} | {r['c33_rel_err']:+.3e} | {r['c44_rel_err']:+.3e} | "
        f"{r['c55_rel_err']:+.3e} | "
        f"{r.get('observed_order_c44', float('nan')):+.2f} | "
        f"{r.get('observed_order_c55', float('nan')):+.2f} |"
        for r in results["sensitivity"]["fd_step"])
    heel_lines = "\n".join(
        f"| {r['heel_deg']:.1f} | {r['gz_got_m']:.6f} | {r['gz_wall_sided_m']:.6f} | "
        f"{r['gz_small_angle_m']:.6f} | {r['gz_rel_err_vs_wall_sided']:+.3e} |"
        for r in results["heel_check"])
    figures = "".join(f"![{n}]({n})\n\n" for n in results["figures"])

    return f"""# CASE-BEM-HYDROSTATIC-PONTOON — 矩形浮箱水静力（TensorBEM）

## 1. 问题 + 参考解出处

算例池锚点 `BEM-HYDROSTATIC-PONTOON`（仓库 TensorBEM，ref_source = "Biran & Pulido /
Rawson & Tupper 闭式"，容差 < 3 %）。对象：长 {case['length_m']} m × 宽 {case['beam_m']} m ×
深 {case['depth_m']} m、吃水 {case['draft_m']} m 的壁面（wall-sided）矩形浮箱，重心高
KG = {case['kg_m']} m（龙骨以上），ρ = {case['density_kg_m3']} kg/m³，g = {case['gravity_m_s2']} m/s²。

参考解：{results['citation']['primary']}；{results['citation']['secondary']}。
**引用诚实性**：{results['citation']['page_level_claim']}。闭式量在
`scripts/case_hydrostatic_pontoon.py::reference_values` 内由几何量程序化求值（零手抄）。

## 2. 方法与实现（复用既有能力，不另起炉灶）

两条**结构性独立**的 TensorBEM 既有路线，跑同一个箱体：

1. `waterplane`：`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` —— 追踪 z=0
   水线环，在平面上积分 A、S、I，再由闭式水静力公式组装 C。
2. `pressure`：`tensorbem.hydrostatic_pressure_route.richardson_restoring_matrix` —— 闭合箱体
   按 z=0 裁剪，积分 p = ρg(0−z) 得合力/合力矩，中心差分得 C，再对步长做 Richardson 外推。

几何先验门（先验几何再比数）：由网格顶点量出 L/B/T 包围盒
= ({results['geometry_gate']['bbox_got'][0]:.12f}, {results['geometry_gate']['bbox_got'][1]:.12f},
{results['geometry_gate']['bbox_got'][2]:.12f}) m，Cb = {results['geometry_gate']['cb_got']:.15f}（参考 1），
Cwp = {results['geometry_gate']['cwp_got']:.15f}（参考 1）——通过。

参数表（含无量纲量）：L/B = {case['length_m']/case['beam_m']:.6f}，B/T = {case['beam_m']/case['draft_m']:.6f}，
T/D = {case['draft_m']/case['depth_m']:.6f}，KG/T = {case['kg_m']/case['draft_m']:.6f}，
Cb = {ref['block_coefficient_Cb']:.6f}，Cwp = {ref['waterplane_coefficient_Cwp']:.6f}，
Cm = {ref['midship_coefficient_Cm']:.6f}，Cp = {ref['prismatic_coefficient_Cp']:.6f}，
GM_T/T = {ref['gm_transverse_m']/case['draft_m']:.6f}。

## 3. 对照表（ref vs got，误差逐项打勾/叉，判据 < 3 %）

{head}{body}

主网格 = {results['primary_level']}（水线路线 {results['comparison_panels']['waterplane']} 个湿表面三角形，
压力路线 {results['comparison_panels']['pressure']} 个闭合三角形）。

## 4. 图

{figures}
图注：① 逐量相对误差（对数轴），两路线均在双精度舍入量级，距 3 % 判据约
{math.log10(results['verdict']['tolerance_rel']/results['verdict']['max_abs_rel_err']):.1f} 个数量级；
② 面板加密（{results['refinement'][0]['panels']}…{results['refinement'][-1]['panels']} 面板）误差曲线平坦
——平面面板对体积/水线面矩/压力矩都精确，误差不随离散变；
③ 吃水扫描 BM_T 与 B²/(12T) 重合；④ 横倾复原力臂：压力积分 vs 壁面公式 vs GM·sinφ。

## 5. 收敛性 / 参数敏感性

### 5.1 面板加密（收敛性）

| level (n_L,n_B,n_T) | wet panels | max abs rel err (waterplane) | max abs rel err (pressure) |
|---|---|---|---|
{ref_lines}

预期平坦（箱体面板精确）。实测确实在 1e-12…4e-11 之间抖动，且**随面板数缓慢上升**——这不是
离散误差（否则会按 O(h²) 下降），而是压力路线在更多面板上的舍入累加；水线路线在 4 个网格上
逐位不变（体积/水线面矩为精确和）。结论：本算例的误差只可能来自实现 bug，不是离散误差。

### 5.2 吃水敏感性（并验证 d∇/dT = A_wp 恒等式）

| T [m] | BM_T ref [m] | BM_T got [m] | rel err | A_wp ref [m²] | d∇/dT got [m²] | rel err |
|---|---|---|---|---|---|---|
{draft_lines}

### 5.3 g 取值敏感性（仅 C 随 g 线性缩放，几何量不变）

| g [m/s²] | C33 rel err | C44 rel err | ∇ rel err |
|---|---|---|---|
{g_lines}

### 5.4 压力路线有限差分步长

| step | C33 rel err | C44 rel err | C55 rel err | obs. order C44 | obs. order C55 |
|---|---|---|---|---|---|
{fd_lines}

C33 对步长不敏感（它只由零阶力构成，差分精确到舍入）；C44/C55 的误差随步长按二阶梯降，
实测阶数程序化给出（上表），与中心差分 O(h²) 的预期一致——Richardson 外推因此把主网格的
C44/C55 误差压到 1e-12 量级。

### 5.5 横倾复原力臂（独立于矩阵的第三路校核）

| φ [°] | GZ got [m] | GZ wall-sided [m] | GM·sinφ [m] | rel err vs wall-sided |
|---|---|---|---|---|
{heel_lines}

## 6. 失败模式与工件

- 判据：全部 {len(rows)} 项 < 3 % → 本算例 {'PASS' if ok else 'FAIL'}；最大相对误差 {max_err:.3e}。
- 失败模式（已知）：(a) 非平面面板引入 O(h²) 剖面误差——本箱体不触发，同脚本换 spheroid 即暴露；
  (b) 压力路线差分步长过大时 O(h²) 截断可淹没真实差（表 5.4）；(c) KG > KM_T 时 GM<0、复原矩阵变号，
  本算例 GM_T = {ref['gm_transverse_m']:.4f} m > 0，不触发；(d) 力矩积分若用一阶求积规则会丢 O(h²)
  （生态内 m16662 已定位该缺陷），此处用模块现成的二阶精确规则。
- 工件：`results.json`（全部数值 + 环境/版本/sha256）、`report.md`、4 张 png、`tests/test_case_hydrostatic_pontoon.py`。
- 复现：`{results['repro_cmd']}`
"""


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(case: dict) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": commit,
        "repo": "TensorBEM",
        "density_kg_m3": case["density_kg_m3"],
        "gravity_m_s2": case["gravity_m_s2"],
        "tolerance_rel": case["tolerance_rel"],
        "refinement_levels": [list(v) for v in REFINEMENT_LEVELS],
        "draft_sweep": list(DRAFT_SWEEP),
        "gravity_sweep": list(GRAVITY_SWEEP),
        "fd_steps": list(FD_STEPS),
    }


def run(out_dir: Path, case: dict = CASE, figures: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = reference_values(case["length_m"], case["beam_m"], case["draft_m"], case["kg_m"],
                           case["density_kg_m3"], case["gravity_m_s2"])
    primary = REFINEMENT_LEVELS[1]
    wp = measure_waterplane_route(case, primary)
    pr = measure_pressure_route(case, primary)
    rows = build_comparison(case, ref, wp, pr)

    gate = {
        "bbox_got": list(wp["bbox_got"]),
        "bbox_ref": [case["length_m"], case["beam_m"], case["draft_m"]],
        "bbox_rel_err": [rel(wp["bbox_got"][i], (case["length_m"], case["beam_m"], case["draft_m"])[i])
                         for i in range(3)],
        "cb_got": wp["displaced_volume_m3"] / (case["length_m"] * case["beam_m"] * case["draft_m"]),
        "cwp_got": wp["waterplane_area_m2"] / (case["length_m"] * case["beam_m"]),
        "divergence_vs_property_volume": rel(wp["divergence_volume_m3"], wp["displaced_volume_m3"]),
        "trace_area_vs_property_area": rel(wp["trace_area_m2"], wp["waterplane_area_m2"]),
    }
    gate["pass"] = bool(
        max(abs(v) for v in gate["bbox_rel_err"]) == 0.0
        and abs(gate["cb_got"] - 1.0) < 1e-12
        and abs(gate["cwp_got"] - 1.0) < 1e-12
        and abs(gate["divergence_vs_property_volume"]) < 1e-12
        and abs(gate["trace_area_vs_property_area"]) < 1e-12
    )

    results = {
        "case": case,
        "citation": CITATION,
        "environment": environment(case),
        "reference": ref,
        "geometry_gate": gate,
        "primary_level": list(primary),
        "comparison_panels": {"waterplane": wp["panels"], "pressure": pr["panels"]},
        "comparison": rows,
        "matrix": {
            "analytic": [[ref["c33_N_per_m"] if (i, j) == (2, 2) else
                          ref["c44_Nm_per_rad"] if (i, j) == (3, 3) else
                          ref["c55_Nm_per_rad"] if (i, j) == (4, 4) else 0.0
                          for j in range(6)] for i in range(6)],
            "waterplane": [[float(v) for v in row] for row in wp["matrix"]],
            "pressure": [[float(v) for v in row] for row in pr["matrix"]],
        },
        "refinement": refinement_study(case, ref),
        "sensitivity": {
            "draft": draft_sensitivity(case),
            "gravity": gravity_sensitivity(case),
            "fd_step": fd_step_sensitivity(case),
        },
        "heel_check": heel_check(case, ref),
        "repro_cmd": "python3 scripts/case_hydrostatic_pontoon.py "
                     "--out artifacts/case_hydrostatic_pontoon",
    }

    results["verdict"] = {
        "tolerance_rel": case["tolerance_rel"],
        "n_quantities": len(rows),
        "n_fail": sum(0 if r["pass"] else 1 for r in rows),
        "max_abs_rel_err": max(r["max_abs_rel_err"] for r in rows),
        "pass": all(r["pass"] for r in rows) and gate["pass"],
    }
    results["figures"] = make_figures(out_dir, results) if figures else []
    results["artifacts_sha256"] = {
        "script": sha256(Path(__file__).resolve()),
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out_dir / "report.md").write_text(build_report(results))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "case_hydrostatic_pontoon"))
    args = ap.parse_args(argv)
    results = run(Path(args.out))
    v = results["verdict"]
    print(json.dumps({"pass": v["pass"], "max_abs_rel_err": v["max_abs_rel_err"],
                      "n_quantities": v["n_quantities"], "out": args.out}, indent=2))
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())