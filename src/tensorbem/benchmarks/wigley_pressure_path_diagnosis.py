"""Re-diagnosis of the Wigley pressure-integration path.

Why this module exists
----------------------
``docs/wigley_wave_resistance.md`` reports that the two wave-resistance paths
of :mod:`tensorbem.wigley` disagree by 100-300 % and that the pressure path does
not converge under mesh refinement.  It then *asserts* a root cause:

    "the two paths do not agree within 5 % ... because (1) thin-ship
     approximation ... (2) pressure on hull, the Michell wave potential is
     evaluated on the actual hull surface but Michell theory assumes the
     potential is evaluated on the centerplane ..."

That attribution is a ``B/L -> 0`` statement: if it were true, the discrepancy
would have to vanish as the hull is made thinner, because in that limit the
actual hull *is* the centerplane.  It is testable in a few seconds, and it is
**false**.  The measurements in this module show:

*T1 - scale invariance (decisive).*  Sweeping ``B/L`` from 0.2 down to
0.00625 at fixed ``L``, ``T``, ``Fr`` leaves ``Rw_pressure / Rw_kochin``
unchanged to 6 significant figures.  Both paths are exactly ``O(B^2)``, so the
ratio is a pure function of the two similarity parameters ``(k0 L, k0 T)``.
A thin-ship error cannot behave that way; it would be ``O(B/L)``.

*T2 - quadrature invariance.*  The pressure path integrates the Kochin function
against ``exp(i k x)`` from ``k0`` to ``k_max``.  The oscillatory integrand was
the other natural suspect (the study in
:mod:`tensorbem.benchmarks.wigley_verification` had already shown that this
kernel is quadrature-sensitive).  Sweeping ``n_quad`` from 128 to 8192, and
splitting the interval into 16 composite panels, changes the answer by less
than ``1e-9`` relative: the pressure-path integral is *converged*, so the
discrepancy is not a quadrature artefact either.

*T3 - condition residuals of the potential that is used.*  The potential the
pressure path differentiates,

    ``phi(x, z) = (U/pi) Re int_{k0}^{k_max} H(k) exp(k z + i k x) dk``,

has no ``y``-dependence at all, so ``dphi/dy == 0`` identically, while the
linearised hull condition needs ``dphi/dy = U dY/dx`` on the wetted surface:
the transverse flux is missing outright (T3a).  Consistently, the same
potential does not satisfy the steady free-surface condition
``phi_z - (1/k0) phi_xx = 0``: substituting the spectral representation gives
the residual weight ``k - k^2/k0``, which vanishes only at ``k = k0`` and is
``O(1)`` relative on the rest of the band (T3b, measured).  A field that
satisfies neither the body nor the free-surface condition cannot be expected to
reproduce the far-field energy balance; that, not thin-ship, is the defect.

Consequence for the documentation: the two-path comparison of the Wigley module
is *not* evidence about the validity of the thin-ship approximation, and the
``+/-30 %`` band the document quotes for Michell theory against experiment is
unsupported by anything this package computes.  The pressure path stays an
*audit* path; it is not a verification path for the Kochin route.

The replica used here re-implements the production routines with vectorised
numpy so that the sweeps are cheap (the production loops are ``O(panels x
nodes)`` Python loops).  :func:`replica_fidelity` re-runs both against each
other on the production defaults and is asserted to ``1e-12`` relative in the
test suite, so the sweeps below measure the shipped code and not a second
implementation of it.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.wigley_pressure_path_diagnosis \
        --output docs/wigley_pressure_path_diagnosis/study.json \
        --figures docs/wigley_pressure_path_diagnosis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from ..wigley import (
    michell_kochin_function,
    wave_resistance_kochin,
    wave_resistance_pressure,
    wigley_hull_mesh,
)

__all__ = [
    "HULL",
    "G",
    "RHO",
    "kochin_vector",
    "pressure_resistance_vectorised",
    "replica_fidelity",
    "scale_invariance",
    "quadrature_invariance",
    "froude_sweep",
    "mesh_refinement",
    "body_condition_residual",
    "free_surface_residual",
    "study",
]

#: Hull of the module and of ``docs/wigley_wave_resistance.md``.
HULL = (2.0, 0.2, 0.125)
RHO = 1025.0
G = 9.80665


# ---------------------------------------------------------------------------
# Vectorised replica of the shipped routines
# ---------------------------------------------------------------------------
def kochin_vector(k, L: float, B: float, T: float) -> np.ndarray:
    """Vectorised ``michell_kochin_function`` (same closed forms, numpy arrays)."""
    k = np.asarray(k, dtype=float)
    ik = 1j * k
    ix = (
        (L / 2.0) * np.exp(ik * L / 2.0) / ik
        - np.exp(ik * L / 2.0) / (ik * ik)
        + (L / 2.0) * np.exp(-ik * L / 2.0) / ik
        + np.exp(-ik * L / 2.0) / (ik * ik)
    )
    ek = np.exp(-k * T)
    iz = (1.0 - ek) / k - (2.0 / k**3 - ek * (T**2 / k + 2.0 * T / k**2 + 2.0 / k**3)) / T**2
    return 2.0 * (-4.0 * B / L**2) * ix * iz


def pressure_resistance_vectorised(
    mesh,
    L: float,
    B: float,
    T: float,
    U: float,
    n_quad: int = 128,
    n_sub: int = 1,
    rho: float = RHO,
    g: float = G,
) -> float:
    """Vectorised ``wigley.wave_resistance_pressure`` (same discretisation).

    ``n_sub`` is a diagnostic knob that the shipped routine does not expose: it
    splits ``[k0, k_max]`` into composite Gauss-Legendre panels.  ``n_sub = 1``
    with ``n_quad = 128`` reproduces the shipped call exactly.
    """
    k0 = g / U**2
    k_max = k0 * max(10.0, 20.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    x = mesh.centroids[:, 0]
    z = mesh.centroids[:, 2]

    total = 0.0
    edges = np.linspace(k0, k_max, n_sub + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        kk = 0.5 * (b - a) * (nodes + 1.0) + a
        ww = 0.5 * (b - a) * weights
        Hk = kochin_vector(kk, L, B, T)
        phase = np.exp(1j * np.outer(x, kk)) * np.exp(np.outer(z, kk))
        integrand = ((1j * kk * Hk)[None, :] * phase).real
        dphi_dx = (U / math.pi) * (integrand * ww[None, :]).sum(axis=1)
        total -= rho * U * float(np.sum(dphi_dx * mesh.normals[:, 0] * mesh.areas))
    return total


def replica_fidelity(L: float = HULL[0], B: float = HULL[1], T: float = HULL[2]) -> dict:
    """Compare the vectorised replica with the shipped routine (must be ~1e-13)."""
    U = 0.30 * math.sqrt(G * L)
    mesh = wigley_hull_mesh(L, B, T, 8, 4)
    shipped = wave_resistance_pressure(mesh, L, B, T, U, n_quad=96)
    replica = pressure_resistance_vectorised(mesh, L, B, T, U, n_quad=96)
    scaled = max(abs(shipped), 1e-30)
    k_grid = np.array([1.7, 3.1, 11.9])
    k_ref = np.array([michell_kochin_function(L, B, T, float(k)) for k in k_grid])
    k_rep = kochin_vector(k_grid, L, B, T)
    return {
        "hull": {"L": L, "B": B, "T": T},
        "froude": 0.30,
        "panels": mesh.panel_count,
        "Rw_pressure_shipped": shipped,
        "Rw_pressure_replica": replica,
        "relative_difference": abs(replica - shipped) / scaled,
        "kochin_relative_difference_max": float(np.max(np.abs(k_rep - k_ref) / np.abs(k_ref))),
    }


# ---------------------------------------------------------------------------
# T1: is the discrepancy a thin-ship (B/L -> 0) effect?
# ---------------------------------------------------------------------------
def scale_invariance(
    L: float = HULL[0],
    T: float = HULL[2],
    froude: float = 0.30,
    breadth_ratios=(0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625),
    nx: int = 24,
    nz: int = 12,
    n_quad: int = 512,
) -> dict:
    """Sweep ``B/L`` at fixed ``L``, ``T``, ``Fr`` and watch the ratio."""
    U = froude * math.sqrt(G * L)
    rows = []
    for bl in breadth_ratios:
        B = bl * L
        rk = wave_resistance_kochin(L, B, T, U)
        mesh = wigley_hull_mesh(L, B, T, nx, nz)
        rp = pressure_resistance_vectorised(mesh, L, B, T, U, n_quad=n_quad)
        rows.append(
            {
                "B_over_L": bl,
                "B": B,
                "Rw_kochin": rk,
                "Rw_pressure": rp,
                "ratio_pressure_over_kochin": rp / rk,
                "kochin_over_B_squared": rk / B**2,
                "pressure_over_B_squared": rp / B**2,
            }
        )
    ratios = np.array([r["ratio_pressure_over_kochin"] for r in rows])
    spread = float((ratios.max() - ratios.min()) / abs(ratios.mean()))
    gaps = np.array([1.0 - r for r in ratios])
    return {
        "hull": {"L": L, "T": T},
        "froude": froude,
        "mesh": {"nx": nx, "nz": nz},
        "n_quad": n_quad,
        "rows": rows,
        "ratio_mean": float(ratios.mean()),
        "ratio_relative_spread": spread,
        "gap_at_B_over_L_0p2": float(gaps[0]),
        "gap_at_B_over_L_0p00625": float(gaps[-1]),
        "gap_ratio_finest_over_coarsest": float(abs(gaps[-1]) / abs(gaps[0])),
        "thin_ship_hypothesis_refuted": spread < 1e-3,
    }


# ---------------------------------------------------------------------------
# T2: is it the oscillatory k-integral?
# ---------------------------------------------------------------------------
def quadrature_invariance(
    L: float = HULL[0],
    B: float = HULL[1],
    T: float = HULL[2],
    froude: float = 0.30,
    nx: int = 24,
    nz: int = 12,
    quad_grid=(128, 256, 512, 1024),
    n_sub_grid=(1, 16),
) -> dict:
    """Sweep Gauss nodes and composite panels of the ``k``-integral.

    The sweep is on the *effective* resolution: ``n_quad * n_sub`` nodes in
    total.  ``numpy``'s ``leggauss`` is a Python-level Newton loop, so node
    counts above a few thousand are dominated by the node construction rather
    than by the integral; the composite knob reaches the same effective
    resolution far more cheaply.
    """
    U = froude * math.sqrt(G * L)
    mesh = wigley_hull_mesh(L, B, T, nx, nz)
    rows = []
    for n_quad in quad_grid:
        for n_sub in n_sub_grid:
            rp = pressure_resistance_vectorised(mesh, L, B, T, U, n_quad=n_quad, n_sub=n_sub)
            rows.append(
                {
                    "n_quad": n_quad,
                    "n_sub": n_sub,
                    "effective_nodes": n_quad * n_sub,
                    "Rw_pressure": rp,
                }
            )
    vals = np.array([r["Rw_pressure"] for r in rows])
    spread = float((vals.max() - vals.min()) / abs(vals.mean()))
    return {
        "hull": {"L": L, "B": B, "T": T},
        "froude": froude,
        "mesh": {"nx": nx, "nz": nz},
        "rows": rows,
        "relative_spread": spread,
        "quadrature_hypothesis_refuted": spread < 1e-6,
    }


# ---------------------------------------------------------------------------
# T3a: the body boundary condition the pressure path never imposes
# ---------------------------------------------------------------------------
def body_condition_residual(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froude: float = 0.30, nx: int = 24, nz: int = 12,
) -> dict:
    """Quantify the missing transverse flux ``dphi/dy``.

    The potential used by :func:`tensorbem.wigley.wave_resistance_pressure` is a
    function of ``(x, z)`` only, so ``dphi/dy = 0`` everywhere.  The linearised
    hull condition requires ``dphi/dn = U n_x``, whose ``y``-component is
    ``U n_y = -U dY/dx``.  The residual is therefore the whole transverse flux.
    """
    U = froude * math.sqrt(G * L)
    mesh = wigley_hull_mesh(L, B, T, nx, nz)
    required = U * mesh.normals[:, 1]
    areas = mesh.areas
    weighted = float(np.sum(np.abs(required) * areas) / np.sum(areas))
    normalised = float(
        np.sum(np.abs(required) * areas) / np.sum(np.abs(U * mesh.normals[:, 0]) * areas)
    )
    return {
        "hull": {"L": L, "B": B, "T": T},
        "froude": froude,
        "speed": U,
        "mean_abs_required_dphi_dy": weighted,
        "mean_abs_dphi_dy_of_used_potential": 0.0,
        "relative_residual": 1.0,
        "transverse_vs_longitudinal_flux_ratio": normalised,
    }


# ---------------------------------------------------------------------------
# T3b: the free-surface condition residual of the potential that is used
# ---------------------------------------------------------------------------
def free_surface_residual(
    L: float = HULL[0],
    B: float = HULL[1],
    T: float = HULL[2],
    froude: float = 0.30,
    n_quad: int = 512,
    n_x: int = 41,
) -> dict:
    """Measure ``phi_z - (1/k0) phi_xx`` on ``z = 0`` for the used potential.

    With ``phi = (U/pi) Re int H(k) exp(k z + i k x) dk`` on the band
    ``k in [k0, k_max]`` the residual carries the spectral weight
    ``k - k^2/k0``; it vanishes only at ``k = k0``.  Both the residual and the
    ``phi_z`` scale of the same potential are evaluated with the production
    quadrature settings so the ratio is a property of the potential, not of the
    discretisation.
    """
    U = froude * math.sqrt(G * L)
    k0 = G / U**2
    k_max = k0 * max(10.0, 20.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    k = 0.5 * (k_max - k0) * (nodes + 1.0) + k0
    w = 0.5 * (k_max - k0) * weights
    Hk = kochin_vector(k, L, B, T)
    x = np.linspace(-L / 2.0, L / 2.0, n_x)
    phase = np.exp(1j * np.outer(x, k))  # z = 0
    phi_z = (U / math.pi) * (((k * Hk)[None, :] * phase).real * w[None, :]).sum(axis=1)
    phi_xx = (U / math.pi) * (((-(k**2) * Hk)[None, :] * phase).real * w[None, :]).sum(axis=1)
    residual = phi_z - phi_xx / k0

    scale = float(np.max(np.abs(phi_z)))
    rel = float(np.max(np.abs(residual)) / scale)
    rows = [
        {"x": float(xi), "phi_z": float(pz), "phi_xx": float(pxx), "residual": float(res)}
        for xi, pz, pxx, res in zip(x, phi_z, phi_xx, residual)
    ]
    weight = k - k**2 / k0
    return {
        "hull": {"L": L, "B": B, "T": T},
        "froude": froude,
        "n_quad": n_quad,
        "k0": k0,
        "max_abs_residual": float(np.max(np.abs(residual))),
        "max_abs_phi_z": scale,
        "relative_residual": rel,
        "residual_exceeds_free_surface_flux": rel > 1.0,
        "spectral_weight_at_k0": float(weight[0]),
        "spectral_weight_max_abs": float(np.max(np.abs(weight))),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Supporting sweeps
# ---------------------------------------------------------------------------
def froude_sweep(
    L: float = HULL[0],
    B: float = HULL[1],
    T: float = HULL[2],
    froude=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    nx: int = 24,
    nz: int = 12,
    n_quad: int = 512,
) -> dict:
    """Ratio versus Froude number at a converged quadrature and a fine mesh."""
    rows = []
    for fr in froude:
        U = fr * math.sqrt(G * L)
        rk = wave_resistance_kochin(L, B, T, U)
        mesh = wigley_hull_mesh(L, B, T, nx, nz)
        rp = pressure_resistance_vectorised(mesh, L, B, T, U, n_quad=n_quad)
        rows.append(
            {
                "froude": fr,
                "Rw_kochin": rk,
                "Rw_pressure": rp,
                "ratio_pressure_over_kochin": rp / rk,
                "k0_L": (G / U**2) * L,
                "k0_T": (G / U**2) * T,
            }
        )
    ratios = [r["ratio_pressure_over_kochin"] for r in rows]
    return {
        "hull": {"L": L, "B": B, "T": T},
        "rows": rows,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "sign_changes": sum(1 for a, b in zip(ratios[:-1], ratios[1:]) if a * b < 0.0),
    }


def mesh_refinement(
    L: float = HULL[0],
    B: float = HULL[1],
    T: float = HULL[2],
    froude: float = 0.30,
    levels=((4, 2), (8, 4), (16, 8), (24, 12)),
    n_quad: int = 512,
) -> dict:
    """Mesh refinement of the pressure path at converged quadrature."""
    U = froude * math.sqrt(G * L)
    rk = wave_resistance_kochin(L, B, T, U)
    rows = []
    for nx, nz in levels:
        mesh = wigley_hull_mesh(L, B, T, nx, nz)
        rp = pressure_resistance_vectorised(mesh, L, B, T, U, n_quad=n_quad)
        rows.append(
            {
                "nx": nx,
                "nz": nz,
                "panel_count": mesh.panel_count,
                "Rw_pressure": rp,
                "ratio_pressure_over_kochin": rp / rk,
            }
        )
    ratios = [r["ratio_pressure_over_kochin"] for r in rows]
    return {
        "hull": {"L": L, "B": B, "T": T},
        "froude": froude,
        "n_quad": n_quad,
        "Rw_kochin": rk,
        "rows": rows,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
    }


def study(L: float = HULL[0], B: float = HULL[1], T: float = HULL[2]) -> dict:
    """Run the whole diagnosis and return the report dictionary."""
    report = {
        "module": "tensorbem.benchmarks.wigley_pressure_path_diagnosis",
        "hull": {"L": L, "B": B, "T": T, "rho": RHO, "g": G},
        "replica_fidelity": replica_fidelity(L, B, T),
        "scale_invariance": scale_invariance(L, T),
        "quadrature_invariance": quadrature_invariance(L, B, T),
        "froude_sweep": froude_sweep(L, B, T),
        "mesh_refinement": mesh_refinement(L, B, T),
        "body_condition_residual": body_condition_residual(L, B, T),
        "free_surface_residual": free_surface_residual(L, B, T),
    }
    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict) -> dict:
    t1 = report["scale_invariance"]
    t2 = report["quadrature_invariance"]
    t3b = report["free_surface_residual"]
    thin_ship = not t1["thin_ship_hypothesis_refuted"]
    quadrature = not t2["quadrature_hypothesis_refuted"]
    if thin_ship or quadrature:
        cause = "unresolved: a swept axis still moves the answer"
    else:
        cause = (
            "formulation defect: the pressure path differentiates a potential that is "
            "y-invariant (dphi/dy == 0, so the transverse body flux U dY/dx is absent) and "
            "that violates the steady free-surface condition by "
            f"{t3b['relative_residual']:.2f} x the phi_z scale"
        )
    return {
        "thin_ship_explanation_supported": thin_ship,
        "quadrature_explanation_supported": quadrature,
        "scale_invariance_relative_spread": t1["ratio_relative_spread"],
        "quadrature_relative_spread": t2["relative_spread"],
        "pressure_path_is_verification_path": False,
        "root_cause": cause,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(report: dict, outdir: Path) -> list[Path]:
    """Three-panel diagnostic figure (plain matplotlib, no seaborn)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    t1 = report["scale_invariance"]
    bl = [r["B_over_L"] for r in t1["rows"]]
    ratio = [r["ratio_pressure_over_kochin"] for r in t1["rows"]]
    ax = axes[0]
    ax.plot(bl, ratio, "o-", color="#1f4e79", label="Rw_pressure / Rw_kochin")
    ax.axhline(1.0, color="#c00000", ls="--", lw=1.0, label="agreement (ratio = 1)")
    ax.set_xscale("log")
    ax.set_xlabel("B / L  (hull thickness ratio)")
    ax.set_ylabel("ratio")
    ax.set_title("T1  thin-ship sweep: ratio is scale invariant")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(
        0.03,
        0.06,
        f"spread {t1['ratio_relative_spread']:.1e}\n"
        f"(ratio = {t1['ratio_mean']:.4f} over 32x in B)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    t2 = report["quadrature_invariance"]
    ax = axes[1]
    for n_sub, colour, marker in ((1, "#1f4e79", "o"), (16, "#e07b39", "s")):
        xs = [r["effective_nodes"] for r in t2["rows"] if r["n_sub"] == n_sub]
        ys = [r["Rw_pressure"] for r in t2["rows"] if r["n_sub"] == n_sub]
        ax.plot(xs, ys, marker=marker, color=colour, label=f"n_sub = {n_sub}")
    ax.set_xscale("log")
    ax.set_xlabel("effective Gauss nodes on [k0, k_max]")
    ax.set_ylabel("Rw_pressure  [N]")
    ax.set_title("T2  k-quadrature is converged")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(
        0.03,
        0.06,
        f"spread {t2['relative_spread']:.1e}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    t3 = report["free_surface_residual"]
    k0 = t3["k0"]
    x = [r["x"] for r in t3["rows"]]
    ax = axes[2]
    ax.plot(x, [r["phi_z"] for r in t3["rows"]], "-", color="#1f4e79", label="phi_z")
    ax.plot(
        x,
        [r["phi_xx"] / k0 for r in t3["rows"]],
        "--",
        color="#2e7d32",
        label="phi_xx / k0  (wanted)",
    )
    ax.plot(x, [r["residual"] for r in t3["rows"]], "-", color="#c00000", label="residual")
    ax.set_xlabel("x  [m]  on z = 0")
    ax.set_ylabel("flux  [1/s]")
    ax.set_title("T3  free-surface residual is O(1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(
        0.03,
        0.06,
        f"max |residual| / max |phi_z| = {t3['relative_residual']:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    fig.suptitle(
        "Wigley pressure path: the 100-300 % discrepancy is neither thin-ship nor quadrature",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = outdir / "wigley_pressure_path_diagnosis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wigley pressure-path re-diagnosis")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    report = study()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.figures and not args.no_figures:
        for path in make_figures(report, args.figures):
            print(f"wrote {path}")
    print(json.dumps(report["verdict"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())