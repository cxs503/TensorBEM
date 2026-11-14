"""Independent audit cross-check for m20029 / CASE-BEM-LAPLACE-CYLINDER.

Deliberately does NOT import tensorbem: the reference table stored in the
mission's case.json is re-derived from the complex potential
W(z) = U (z + a^2/z) with plain numpy, and the *continuum* Dirichlet problem of
the inscribed polygon is solved a second, integral-equation-free way (exterior
multipole least squares) to bound the faceting floor that any panel BEM on that
polygon must sit above.

Outputs /tmp/audit_m20100_indep.json (also copied next to this file).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

M20029 = "/root/tp-agent/work/m20029"
CASE = os.path.join(M20029, "benchmarks/verified/case_bem_laplace_cylinder")
U = 1.0
A = 1.0

case = json.load(open(os.path.join(CASE, "case.json")))
res = json.load(open(os.path.join(CASE, "results.json")))
tab = res["reference"]["probe_table"]
probes = np.array([[p["x_over_a"] * A, p["y_over_a"] * A] for p in tab])
ref_tab = np.array(
    [[p["u_over_U"], p["v_over_U"], p["perturbation_potential_over_Ua"]] for p in tab]
)

# ---------------------------------------------------------------- reference --
z = probes[:, 0] + 1j * probes[:, 1]
w = U * (z + A ** 2 / z)                     # complex potential
dw = U * (1.0 - A ** 2 / z ** 2)             # dw/dz = u - i v
u_exact = dw.real
v_exact = -dw.imag
phi_pert_exact = (w - U * z).real            # phi' = U a^2 cos(theta)/r
mine = np.column_stack([u_exact / U, v_exact / U, phi_pert_exact / (U * A)])
ref_err = np.abs(mine - ref_tab)
ref_check = {
    "points": int(probes.shape[0]),
    "max_abs_diff_vs_case_json_table": float(ref_err.max()),
    "max_abs_diff_u": float(ref_err[:, 0].max()),
    "max_abs_diff_v": float(ref_err[:, 1].max()),
    "max_abs_diff_phi": float(ref_err[:, 2].max()),
    "surface_speed_check_2U_sin": {
        "note": "|v|=2U|sin th| and Cp=1-4 sin^2 th re-derived on the circle",
        "max_abs_diff": float(
            np.max(
                np.abs(
                    np.abs(U * (1.0 - np.exp(-2j * np.linspace(0, 2 * np.pi, 360)))) - 2 * U * np.abs(np.sin(np.linspace(0, 2 * np.pi, 360)))
                )
            )
        ),
    },
}

# --------------------------------------- continuum polygon Dirichlet (2nd way) --
# exterior harmonic, decaying:  phi'(r,th) = sum_n r^-n (A_n cos n th + B_n sin n th)
def basis(theta, r, nmax):
    cols = []
    for n in range(1, nmax + 1):
        cols.append(np.cos(n * theta) / r ** n)
        cols.append(np.sin(n * theta) / r ** n)
    return np.column_stack(cols)


def grad_basis(x, y, nmax):
    """d/dx, d/dy of r^-n cos(n th) and r^-n sin(n th), analytic in x,y."""
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    gx, gy = [], []
    for n in range(1, nmax + 1):
        for fn, dth in ((np.cos, lambda t: -np.sin(t)), (np.sin, np.cos)):
            # f = r^-n fn(n th)
            fr = -n * r ** (-n - 1) * fn(n * theta)
            fth = r ** (-n) * n * dth(n * theta)
            gx.append(fr * np.cos(theta) - fth * np.sin(theta) / r)
            gy.append(fr * np.sin(theta) + fth * np.cos(theta) / r)
    return np.column_stack(gx), np.column_stack(gy)


def polygon(N):
    k = np.arange(N)
    return np.column_stack([A * np.cos(2 * np.pi * k / N), A * np.sin(2 * np.pi * k / N)])


def sample_boundary(verts, per_edge=8):
    pts = []
    N = len(verts)
    for i in range(N):
        p, q = verts[i], verts[(i + 1) % N]
        for s in np.linspace(0.0, 1.0, per_edge, endpoint=False):
            pts.append(p + s * (q - p))
    return np.array(pts)


poly_results = {}
NMAX = 40
for N in (32, 64, 128):
    verts = polygon(N)
    bp = sample_boundary(verts, per_edge=8)
    r = np.hypot(bp[:, 0], bp[:, 1])
    th = np.arctan2(bp[:, 1], bp[:, 0])
    data = U * A ** 2 * np.cos(th) / r          # analytic cylinder perturbation potential
    M = basis(th, r, NMAX)
    coef, *_ = np.linalg.lstsq(M, data / (U * A), rcond=None)
    fitted = basis(th, r, NMAX) @ coef
    rp = np.hypot(probes[:, 0], probes[:, 1])
    thp = np.arctan2(probes[:, 1], probes[:, 0])
    phi_poly = (basis(thp, rp, NMAX) @ coef) * U * A
    gx, gy = grad_basis(probes[:, 0], probes[:, 1], NMAX)
    up = (gx @ coef) * U
    vp = (gy @ coef) * U
    e_u = np.abs(up - (u_exact - U)) / U
    e_v = np.abs(vp - v_exact) / U
    e_tot = np.hypot(up - (u_exact - U), vp - v_exact) / np.hypot(u_exact, v_exact)
    poly_results[N] = {
        "bc_fit_rel_residual_max": float(np.max(np.abs(fitted - data / (U * A))) / np.max(np.abs(data / (U * A)))),
        "coef_n1": float(coef[0]),
        "velocity_relative_l2_vs_cylinder": float(np.sqrt(np.mean(e_tot ** 2))),
        "velocity_max_abs_error_over_U": float(max(e_u.max(), e_v.max())),
        "perturbation_potential_relative_l2": float(
            np.sqrt(np.mean(((phi_poly - phi_pert_exact) / (U * A)) ** 2))
        ),
    }

# ------------------------------------------------- reference_length invariance --
sys.path.insert(0, os.path.join(M20029, "src"))
inv = {}
try:
    from tensorbem.geometry import regular_polygon  # noqa: E402
    from tensorbem.solver import (  # noqa: E402
        evaluate_exterior_field,
        solve_laplace_exterior_dirichlet,
        solve_laplace_exterior_neumann,
    )

    mesh = regular_polygon(panel_count=128, radius=A)
    mids = np.asarray(mesh.centers, dtype=float)
    rad = np.hypot(mids[:, 0], mids[:, 1])
    ang = np.arctan2(mids[:, 1], mids[:, 0])
    phi_bc = U * A ** 2 * np.cos(ang) / rad
    qe = U * A ** 2 * np.cos(ang) / rad ** 2

    def rel_l2(field):
        up = U + field.gradient[:, 0]
        vp = field.gradient[:, 1]
        return float(np.sqrt(np.mean((up - u_exact) ** 2 + (vp - v_exact) ** 2)) / np.sqrt(np.mean(u_exact ** 2 + v_exact ** 2)))

    for L in (2.0, 3.0, 5.0):
        rd = solve_laplace_exterior_dirichlet(mesh, phi_bc, reference_length=L)
        rn = solve_laplace_exterior_neumann(mesh, qe, reference_length=L)
        inv[f"L={L}"] = {
            "dirichlet_cond": float(rd.condition_number),
            "dirichlet_velocity_relative_l2": rel_l2(evaluate_exterior_field(mesh, rd, probes)),
            "neumann_cond": float(rn.condition_number),
            "neumann_velocity_relative_l2": rel_l2(evaluate_exterior_field(mesh, rn, probes)),
        }
except Exception as exc:  # pragma: no cover - audit reporting only
    inv = {"error": f"{type(exc).__name__}: {exc}"}

out = {
    "mission_audited": 20029,
    "case": "CASE-BEM-LAPLACE-CYLINDER",
    "reference_table_check": ref_check,
    "continuum_polygon_dirichlet_multipole": {str(k): v for k, v in poly_results.items()},
    "reference_length_invariance": inv,
    "claimed": {
        "dirichlet_velocity_relative_l2_N128": res["routes"]["dirichlet"]["finest"]["velocity_relative_l2"],
        "neumann_velocity_relative_l2_N128": res["routes"]["neumann"]["finest"]["velocity_relative_l2"],
        "neumann_surface_velocity_relative_l2_N128": res["routes"]["neumann"]["finest"]["surface_velocity_relative_l2"],
    },
    "overall": res["overall"],
    "n_criteria": len(res["criteria"]),
    "criteria_all_ok": bool(all(c["ok"] for c in res["criteria"])),
}
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/audit_m20100_indep.json", "w"), indent=1)
# ------------------- continuum polygon Neumann (multipole LS, independent) ---
def grad_phi_pert(x, y):
    r2 = x * x + y * y
    r4 = r2 * r2
    return U * A ** 2 * (1.0 / r2 - 2.0 * x * x / r4), U * A ** 2 * (-2.0 * x * y / r4)


neumann_indep = {}
for N in (32, 64, 128):
    verts = polygon(N)
    centers = 0.5 * (verts + np.roll(verts, -1, axis=0))
    tang = np.roll(verts, -1, axis=0) - verts
    tang = tang / np.linalg.norm(tang, axis=1)[:, None]
    n_body = np.column_stack([tang[:, 1], -tang[:, 0]])   # outward
    n_fluid = -n_body
    bp = sample_boundary(verts, per_edge=8)
    # per-sample edge index and normal
    edge = (np.arange(len(bp)) // 8) % N
    nn = n_fluid[edge]
    gx0, gy0 = grad_phi_pert(bp[:, 0], bp[:, 1])
    q = gx0 * nn[:, 0] + gy0 * nn[:, 1]
    r = np.hypot(bp[:, 0], bp[:, 1])
    th = np.arctan2(bp[:, 1], bp[:, 0])
    M = basis(th, r, NMAX)
    Gx, Gy = grad_basis(bp[:, 0], bp[:, 1], NMAX)
    A_mat = Gx * nn[:, 0:1] + Gy * nn[:, 1:2]
    coef, *_ = np.linalg.lstsq(A_mat, q / U, rcond=None)
    # surface speed at midpoints
    rp = np.hypot(centers[:, 0], centers[:, 1])
    thp = np.arctan2(centers[:, 1], centers[:, 0])
    Gxm, Gym = grad_basis(centers[:, 0], centers[:, 1], NMAX)
    up = U + (Gxm @ coef) * U
    vp = (Gym @ coef) * U
    speed = np.abs(up * tang[:, 0] + vp * tang[:, 1])   # tangential speed (magnitude; CCW tangent makes the signed value negative)
    exact_speed = 2 * U * np.abs(np.sin(thp))
    neumann_indep[N] = {
        "bc_fit_rel_residual_max": float(np.max(np.abs((Gx * nn[:, 0:1] + Gy * nn[:, 1:2]) @ coef - q / U)) / np.max(np.abs(q / U))),
        "surface_speed_relative_l2_vs_2U_sin": float(np.sqrt(np.mean((speed - exact_speed) ** 2)) / np.sqrt(np.mean(exact_speed ** 2))),
        "surface_speed_max_error_over_U": float(np.max(np.abs(speed - exact_speed)) / U),
        "signed_tangential_velocity_vs_minus_2U_sin_max_error_over_U": float(
            np.max(np.abs(up * tang[:, 0] + vp * tang[:, 1] + 2 * U * np.sin(thp))) / U
        ),
        "normal_velocity_residual_over_U": float(
            np.max(np.abs((Gx @ coef) * 0.0 + (Gx * nn[:, 0:1] + Gy * nn[:, 1:2]) @ coef - q / U))
        ),
    }

out["continuum_polygon_neumann_multipole"] = {str(k): v for k, v in neumann_indep.items()}
out["discrete_neumann_claimed"] = {
    str(lev["panels"]): {
        "surface_velocity_relative_l2": lev["surface_velocity_relative_l2"],
        "surface_speed_max_error_over_speed": lev["surface_speed_max_error_over_speed"],
    }
    for lev in res["routes"]["neumann"]["levels"]
}
print(json.dumps({k: out[k] for k in ("continuum_polygon_neumann_multipole", "discrete_neumann_claimed")}, indent=1))
json.dump(out, open("/tmp/audit_m20100_indep.json", "w"), indent=1)

# ---- continuum Neumann with the *discrete data* (panel-constant q, panel normal) --
# This isolates "does the reported O(h^2) surface-speed error belong to the
# prescribed polygonal data (faceting) or to the BEM solver?"  The BEM solve is
# the collocation approximation of exactly this continuum problem.
disc_data = {}
for N in (32, 64, 128):
    verts = polygon(N)
    centers = 0.5 * (verts + np.roll(verts, -1, axis=0))
    tang = np.roll(verts, -1, axis=0) - verts
    tang = tang / np.linalg.norm(tang, axis=1)[:, None]
    n_fluid = -np.column_stack([tang[:, 1], -tang[:, 0]])
    r_c = np.hypot(centers[:, 0], centers[:, 1])
    th_c = np.arctan2(centers[:, 1], centers[:, 0])
    q_panel = U * A ** 2 * np.cos(th_c) / r_c ** 2       # exactly the mission's q_e
    rows, rhs = [], []
    per_edge = 6
    for j in range(N):
        p, d = verts[j], verts[(j + 1) % N] - verts[j]
        for s in np.linspace(0.0, 1.0, per_edge, endpoint=False):
            pt = p + s * d
            Gx, Gy = grad_basis(np.array([pt[0]]), np.array([pt[1]]), NMAX)
            rows.append(Gx[0] * n_fluid[j, 0] + Gy[0] * n_fluid[j, 1])
            rhs.append(q_panel[j] / U)
    coef, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    Gxm, Gym = grad_basis(centers[:, 0], centers[:, 1], NMAX)
    up = U + (Gxm @ coef) * U
    vp = (Gym @ coef) * U
    speed = np.abs(up * tang[:, 0] + vp * tang[:, 1])
    exact_speed = 2 * U * np.abs(np.sin(th_c))
    disc_data[N] = {
        "surface_speed_relative_l2_vs_2U_sin": float(np.sqrt(np.mean((speed - exact_speed) ** 2)) / np.sqrt(np.mean(exact_speed ** 2))),
        "surface_speed_max_error_over_U": float(np.max(np.abs(speed - exact_speed)) / U),
        "normal_velocity_residual_over_U": float(np.max(np.abs((np.array(rows) @ coef) - np.array(rhs)))),
    }

out["continuum_neumann_with_discrete_data"] = {str(k): v for k, v in disc_data.items()}
json.dump(out, open("/tmp/audit_m20100_indep.json", "w"), indent=1)
print(json.dumps(out["continuum_neumann_with_discrete_data"], indent=1))
