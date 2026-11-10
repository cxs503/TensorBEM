"""Study driver for the steady Neumann-Kelvin panel module (M10003 / p315w1-n01).

Produces ``artifacts/nk_panel3d_study.json`` plus the field data needed by the
figures.  Every number in the report is written by this script (no hand copy).

Run:  PYTHONPATH=src python scripts/nk_panel3d_study.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from tensorbem import nk_panel3d as nk
from tensorbem.wigley import michell_wave_resistance, wigley_hull_properties

OUT = Path(__file__).resolve().parents[1] / "artifacts"
L, B, T = 10.0, 1.0, 0.625  # Wigley III, L/B = 10, B/T = 1.6
G = 9.80665
RHO = 1025.0


def kernel_verification(quick: bool) -> dict:
    kappa = 1.0  # lengths scaled by 1/kappa, kappa-dependent quantities are invariant
    out: dict = {"kappa": kappa}
    out["free_surface_condition"] = {
        mode: nk.verify_free_surface_condition(kappa, 64, mode, -1.0, h=2e-3)
        for mode in ("none", "halfplane", "downstream")
    }
    out["laplace"] = nk.verify_laplace(kappa, 64, "halfplane", -1.0)
    out["radiation"] = {
        f"{mode}_{sign:+.0f}": nk.verify_radiation_direction(kappa, 64, mode, sign)
        for mode in ("none", "halfplane")
        for sign in (1.0, -1.0)
    }
    out["wedge"] = {
        f"{mode}_{sign:+.0f}": nk.kelvin_wake_extent(
            kappa, 96, mode, sign, span=16.0, n=181, threshold=0.1
        )
        for mode in ("halfplane",)
        for sign in (1.0, -1.0)
    }
    # theta-quadrature convergence of the kernel value at a sample point
    src = np.array([[0.0, 0.0, -0.3]])
    p = np.array([[1.1, 0.4, -0.2]])
    ref = None
    conv = []
    for nt in (16, 24, 32, 48, 64, 96, 128):
        val = float(nk.nk_wave_kernel(p, src, kappa, nt, "halfplane", -1.0)[0].ravel()[0])
        ref = val if nt == 128 else ref
        conv.append({"n_theta": nt, "value": val})
    if ref:
        for row in conv:
            row["rel_error"] = abs(row["value"] - ref) / max(abs(ref), 1e-30)
    out["theta_convergence"] = conv
    return out


def kelvin_wake_field(quick: bool) -> dict:
    """Free-surface elevation of a point source (Kelvin wake) on a grid."""
    kappa = 1.0
    n = 121 if quick else 201
    span = 18.0
    src = np.array([[0.0, 0.0, -0.25]])
    xs = np.linspace(-span, 1.0, n)
    ys = np.linspace(-span, span, 2 * n - 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    Gk = nk.nk_wave_kernel(pts, src, kappa, 96, "halfplane", -1.0)[0].reshape(X.shape)
    zeta = Gk.copy()
    np.savez_compressed(
        OUT / "kelvin_wake_field.npz", x=xs, y=ys, zeta=zeta, span=span
    )
    # wedge boundary: outermost |zeta| above 8% of the local maximum
    thresh_frac = 0.1
    thr = thresh_frac * np.abs(zeta).max()
    angs = []
    for i in range(len(xs) // 2, len(xs)):
        row = np.abs(zeta[i])
        idx = np.where(row > thr)[0]
        if idx.size == 0:
            continue
        y_edge = np.max(np.abs(ys[idx]))
        angs.append(np.degrees(math.atan2(y_edge, abs(xs[i]))))
    return {
        "grid": [int(n), int(2 * n - 1)],
        "span_over_kappa": span,
        "wedge_boundary_angle_deg": float(np.mean(angs[-40:])) if angs else float("nan"),
        "theoretical_deg": float(np.degrees(math.asin(1.0 / 3.0))),
        "max_abs_elevation": float(np.abs(zeta).max()),
    }


def geometry_validation() -> dict:
    props = wigley_hull_properties(L, B, T)
    rows = []
    for nx, nz in ((10, 5), (16, 8), (24, 12), (40, 20)):
        mesh = nk.wigley_half_panels(L, B, T, nx, nz)
        vol = mesh.total_area * 0.0  # half-area only; volume comes from the closed form
        rows.append(
            {
                "nx": nx,
                "nz": nz,
                "n_panels": mesh.n_panels,
                "wet_area_half": mesh.total_area,
                "mean_edge": mesh.mean_edge,
            }
        )
    # analytic wetted area of the half hull
    def integrand(x, z):
        fx = -(4.0 * B / L**2) * x * (1.0 - (z / T) ** 2)
        fz = -(B / T**2) * (1.0 - (2.0 * x / L) ** 2) * z
        return np.sqrt(1.0 + fx * fx + fz * fz)

    n = 400
    xs = (np.arange(n) + 0.5) / n * L - 0.5 * L
    zs = (np.arange(n) + 0.5) / n * T - T
    XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
    area = float(np.sum(integrand(XX, ZZ)) * (L / n) * (T / n))
    for row in rows:
        row["wet_area_analytic"] = area
        row["rel_error"] = abs(row["wet_area_half"] - area) / area
    return {
        "L": L,
        "B": B,
        "T": T,
        "Cb_analytic": props["Cb"],
        "Cb_metadata_claim": 0.5625,
        "Cb_note": "Wigley III with y=(B/2)(1-(2x/L)^2)(1-(z/T)^2) has Cb=4/9=0.4444; "
        "the 0.5625 quoted in the node metadata does not correspond to this hullform",
        "panels": rows,
    }


def mesh_study(quick: bool) -> dict:
    levels = [(10, 5), (16, 8), (24, 12)] if quick else [(10, 5), (16, 8), (24, 12), (36, 18)]
    out: dict = {"levels": [], "froude_numbers": [0.2, 0.3, 0.4]}
    for Fr in (0.2, 0.3, 0.4):
        U = Fr * math.sqrt(G * L)
        rw_m = michell_wave_resistance(L, B, T, U, rho=RHO, g=G)
        cw_m = rw_m / (0.5 * RHO * U**2 * L * T)
        case = {"froude": Fr, "speed": U, "michell_rw": rw_m, "michell_cw": cw_m, "meshes": []}
        for nx, nz in levels:
            t0 = time.time()
            mesh = nk.wigley_half_panels(L, B, T, nx, nz)
            res = nk.solve_nk_panel(
                mesh,
                U,
                rho=RHO,
                g=G,
                n_theta=48,
                radiation="halfplane",
                sign=-1.0,
                s_ref=L * T,
                L_ref=L,
            )
            case["meshes"].append(
                {
                    "nx": nx,
                    "nz": nz,
                    "n_panels": mesh.n_panels,
                    "mean_edge": mesh.mean_edge,
                    "rw_near": res.wave_resistance,
                    "cw_near": res.cw,
                    "bc_residual": res.bc_residual,
                    "condition_number": res.extra["condition_number"],
                    "rel_error_vs_michell": (res.wave_resistance - rw_m) / rw_m,
                    "wall_time_s": time.time() - t0,
                }
            )
        out["levels"].append(case)
    return out


def wave_cuts(quick: bool) -> dict:
    """Free-surface elevation along three longitudinal cuts, Fr = 0.3."""
    Fr = 0.3
    U = Fr * math.sqrt(G * L)
    nx, nz = (16, 8) if quick else (24, 12)
    mesh = nk.wigley_half_panels(L, B, T, nx, nz)
    res = nk.solve_nk_panel(
        mesh, U, rho=RHO, g=G, n_theta=48, radiation="halfplane", sign=-1.0, s_ref=L * T
    )
    xs = np.linspace(-3.0 * L, 1.5 * L, 91)
    cuts = {}
    for y in (0.0, 0.5 * B, 1.0 * B):
        pts = np.column_stack([xs, np.full(xs.size, y)])
        zeta = nk.free_surface_elevation(
            pts, mesh, res.sigma, U, g=G, n_theta=48, radiation="halfplane", sign=-1.0
        )
        cuts[f"y={y:.3f}"] = {"x": xs.tolist(), "zeta": zeta.tolist()}
    # thin-ship reference elevation with the same kernel (centreplane sheet)
    ts = nk.thin_ship_reference(L, B, T, U, nx=40, nz=20)
    ref = {}
    for y in (0.0, 0.5 * B, 1.0 * B):
        pts = np.column_stack([xs, np.full(xs.size, y)])
        zeta = nk.free_surface_elevation(
            pts,
            ts["mesh"],
            ts["sigma"],
            U,
            g=G,
            n_theta=48,
            radiation="halfplane",
            sign=-1.0,
        )
        ref[f"y={y:.3f}"] = {"x": xs.tolist(), "zeta": zeta.tolist()}
    errs = []
    for key in cuts:
        a = np.asarray(cuts[key]["zeta"])
        b = np.asarray(ref[key]["zeta"])
        scale = max(np.abs(b).max(), 1e-30)
        errs.append(float(np.abs(a - b).max() / scale))
    return {
        "froude": Fr,
        "mesh": [nx, nz],
        "cuts_panel": cuts,
        "cuts_thin_ship": ref,
        "max_rel_error_vs_thin_ship": float(max(errs)),
        "note": "thin-ship reference uses the same NK kernel on a centreplane sheet; "
        "the deviation is the finite-beam effect plus panel discretisation",
    }


def thin_ship_farfield_diagnostic() -> dict:
    rows = []
    for Fr in (0.2, 0.3, 0.4):
        U = Fr * math.sqrt(G * L)
        ts = nk.thin_ship_reference(L, B, T, U, nx=90, nz=45, prefactor=1.0)
        rw_m = michell_wave_resistance(L, B, T, U, rho=RHO, g=G)
        rows.append(
            {
                "froude": Fr,
                "michell_rw": rw_m,
                "farfield_raw": ts["rw_far"],
                "implied_prefactor": rw_m / ts["rw_far"],
            }
        )
    return {
        "rows": rows,
        "conclusion": "the implied prefactor drifts with Froude number, so the "
        "simple |H|^2 sec^3 theta normalisation used here is NOT a valid far-field "
        "operator; the far-field route is therefore reported as unvalidated and the "
        "near-field pressure integral remains the only resistance path in this module",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "mission": "m10003",
        "node": "p315w1-n01-bem-potential-flow",
        "repo": "TensorBEM",
        "module": "tensorbem.nk_panel3d",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "condition_number_note": "single-layer NK system, half body + mirror",
    }
    report["kernel_verification"] = kernel_verification(args.quick)
    report["kelvin_wake"] = kelvin_wake_field(args.quick)
    report["geometry"] = geometry_validation()
    report["thin_ship_farfield"] = thin_ship_farfield_diagnostic()
    report["mesh_study"] = mesh_study(args.quick)
    report["wave_cuts"] = wave_cuts(args.quick)

    path = OUT / "nk_panel3d_study.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}")
    kv = report["kernel_verification"]
    print("FS max residual:", {k: round(v["max_relative_residual"], 8) for k, v in kv["free_surface_condition"].items()})
    print("wedge:", kv["wedge"])
    print("kelvin wake:", report["kelvin_wake"])
    for case in report["mesh_study"]["levels"]:
        print(
            f"Fr={case['froude']}: Michell Cw={case['michell_cw']:.6f} | "
            + " ".join(
                f"{m['nx']}x{m['nz']}:Cw={m['cw_near']:.6f}" for m in case["meshes"]
            )
        )


if __name__ == "__main__":
    main()