"""Audit of the Wigley wave-resistance "pressure-integration" path (mission m3100).

The repository carries two ways of computing the wave resistance of the Wigley
parabolic hull: a Kochin far-field (Michell) integral and a "pressure
integration on the hull" path.  ``docs/wigley_wave_resistance.md`` recorded a
103-284 % disagreement between them as an *honest FAIL* and attributed it to the
thin-ship approximation (B/L not small enough).  This script tests that
attribution.

Three independent measurements are produced, all of them cheap (2-D hull, 1-D
k-quadrature):

1. ``parity``  -- the near-field integrand that the shipped pressure path sums
   is ``dphi_dx(x, z) * n_x dS``.  For a fore-aft symmetric hull
   ``dphi_dx`` is even in x (``sigma`` odd -> ``phi`` odd -> ``phi_x`` even)
   while ``n_x dS = -b_x dx dz`` is odd in x, so the *exact* integral over the
   symmetric x-interval is zero.  Evaluated on an x-symmetric quadrature grid
   it must therefore come out at roundoff.
2. ``ladder`` -- the shipped API value versus mesh refinement.  A convergent
   discretisation of a non-zero integral tends to that integral; this one
   must tend to zero, and its sign must be an artefact too.
3. ``thin``   -- the thin-ship limit (B/L -> 0 with L, Fr fixed).  If the
   disagreement were a finite-B/L physical effect, Rw_pressure/Rw_kochin would
   tend to 1 in that limit.

Run:  PYTHONPATH=src python3 docs/m3100/wigley_pressure_path_audit.py
Writes: docs/m3100/outputs/wigley_pressure_path_audit.json
        docs/m3100/figures/wigley_pressure_path_audit.png
        docs/m3100/figures/wigley_pressure_field.png
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from matplotlib.figure import Figure
from numpy.polynomial.legendre import leggauss

from tensorbem.wigley import (
    michell_kochin_function,
    wave_resistance_kochin,
    wave_resistance_pressure,
    wigley_hull_mesh,
)

G = 9.80665
RHO = 1025.0
L, B, T = 2.0, 0.2, 0.125
FR = (0.20, 0.25, 0.30, 0.35, 0.40)
OUT = pathlib.Path(__file__).with_name("outputs")
FIG = pathlib.Path(__file__).with_name("figures")


# ----------------------------------------------------------------------------
# the near-field integrand, exactly as the shipped pressure path builds it
# ----------------------------------------------------------------------------

def _half_breadth(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return 0.5 * B * (1.0 - (2.0 * x / L) ** 2) * (1.0 - (z / T) ** 2)


def _k_nodes(U: float, n_quad: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k nodes/weights and the Kochin function, as used by the shipped path."""
    k0 = G / U**2
    k_max = k0 * max(10.0, 20.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    k = 0.5 * (k_max - k0) * (nodes + 1.0) + k0
    w = 0.5 * (k_max - k0) * weights
    H = np.array([michell_kochin_function(L, B, T, kk) for kk in k], dtype=complex)
    return k, w, H


def dphi_dx(x: np.ndarray, z: np.ndarray, U: float, n_quad: int = 256) -> np.ndarray:
    """Wave-part x-velocity on the centreplane (the shipped formula)."""
    k, w, H = _k_nodes(U, n_quad)
    ph = (1j * k * H * np.exp(np.outer(z, k)) * np.exp(1j * np.outer(x, k))).real
    return (U / math.pi) * (ph * w).sum(axis=1)


def near_field_trapezoid(U: float, n_x: int, n_z: int, n_quad: int = 256) -> float:
    """Rw = -rho U int int dphi_dx n_x dS on an x-symmetric trapezoid grid.

    ``n_x dS = -b_x dx dz`` on each side, and both sides contribute equally,
    so Rw = 2 rho U int int phi_x (-b_x) dx dz.
    """
    xs = np.linspace(-L / 2, L / 2, n_x)
    zs = np.linspace(-T, 0.0, n_z)
    w_x = np.full(n_x, L / (n_x - 1))
    w_x[[0, -1]] = 0.5 * L / (n_x - 1)
    w_z = np.full(n_z, T / (n_z - 1))
    w_z[[0, -1]] = 0.5 * T / (n_z - 1)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    px = dphi_dx(X.ravel(), Z.ravel(), U, n_quad).reshape(X.shape)
    b_x = -8.0 * B * X / L**3 * (1.0 - (Z / T) ** 2)
    integral = float((w_x[:, None] * w_z[None, :] * px * b_x).sum())
    return 2.0 * RHO * U * integral


def shipped_split(U: float, n_x: int, n_z: int, n_quad: int = 128) -> dict:
    """Per-panel contributions of the shipped path + mirror-pairing census.

    ``n_x`` panels per half give panel centroid abscissae
    ``x_c = (x_i + 2 x_{i+1})/3 = -L/2 + h(3i+2)/3``.  A mirror partner exists
    iff some ``j`` satisfies ``3(i+j)+4 = 3 L/h``, which has no integer
    solution when ``L/h = n_x`` is an integer.  So *no* panel has an exact
    mirror partner and the odd integrand cannot cancel pairwise: the whole
    residual reported by the shipped path is that aliasing, not a physical
    resistance.
    """
    mesh = wigley_hull_mesh(L, B, T, n_x, n_z)
    k0 = G / U**2
    k_max = k0 * max(10.0, 20.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    k = 0.5 * (k_max - k0) * (nodes + 1.0) + k0
    w = 0.5 * (k_max - k0) * weights
    H = np.array([michell_kochin_function(L, B, T, kk) for kk in k], dtype=complex)
    c = mesh.centroids
    dphi = np.empty(mesh.panel_count)
    for i in range(mesh.panel_count):
        integ = (1j * k * H * np.exp(k * c[i, 2]) * np.exp(1j * k * c[i, 0])).real
        dphi[i] = (U / math.pi) * float(np.sum(w * integ))
    contrib = dphi * mesh.normals[:, 0] * mesh.areas
    raw = float(contrib.sum())

    # mirror-pairing census (tolerance 1e-9 L)
    tol = 1e-9 * L
    paired_panels = 0
    seen: set[int] = set()
    for i in range(mesh.panel_count):
        if i in seen:
            continue
        d = np.hypot(c[:, 0] + c[i, 0], c[:, 2] - c[i, 2])
        cand = [j for j in np.where(d < tol)[0] if j != i and j not in seen]
        if cand:
            seen.update((i, int(cand[0])))
            paired_panels += 2
        else:
            seen.add(i)
    return {
        "panels": mesh.panel_count,
        "h_mean": mesh.h_mean,
        "raw_pressure_sum": raw,
        "rw_raw": -RHO * U * raw,
        "paired_panel_fraction": paired_panels / mesh.panel_count,
        "mirror_pairs_found": paired_panels // 2,
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "hull": {"L": L, "B": B, "T": T, "Cb": 4.0 / 9.0, "B_over_T": B / T, "rho": RHO, "g": G},
        "parity": [],
        "ladder": [],
        "thin_ship_limit": [],
        "froude_sweep": [],
    }

    # 1. parity: exact value of the shipped integrand on a symmetric grid
    for Fn in (0.25, 0.30, 0.35):
        U = Fn * math.sqrt(G * L)
        rw_k = wave_resistance_kochin(L, B, T, U, RHO, G)
        for n_x, n_z, n_quad in ((32, 16, 256), (64, 32, 256), (128, 64, 512), (256, 128, 512)):
            rw_near = near_field_trapezoid(U, n_x, n_z, n_quad)
            payload["parity"].append({
                "froude": Fn, "n_x": n_x, "n_z": n_z, "n_quad": n_quad,
                "rw_near_exact_grid": rw_near,
                "rw_kochin": rw_k,
                "rel": abs(rw_near) / rw_k,
            })

    # 2. mesh ladder of the shipped path, plus the parity-paired total
    U = 0.30 * math.sqrt(G * L)
    rw_k30 = wave_resistance_kochin(L, B, T, U, RHO, G)
    for n_x, n_z in ((8, 4), (16, 8), (32, 16), (64, 32), (128, 64)):
        split = shipped_split(U, n_x, n_z)
        split["froude"] = 0.30
        split["rw_kochin"] = rw_k30
        split["rel_raw"] = abs(split["rw_raw"]) / rw_k30
        payload["ladder"].append(split)

    # 3. thin-ship limit: B/L -> 0 with L, Fr fixed (B/T = 1.6 kept)
    for bl in (0.10, 0.05, 0.025, 0.0125):
        b = bl * L
        t = b / 1.6
        rw_k = wave_resistance_kochin(L, b, t, U, RHO, G)
        row = {"B_over_L": bl, "rw_kochin": rw_k}
        for n_x, n_z in ((16, 8), (32, 16)):
            mesh = wigley_hull_mesh(L, b, t, n_x, n_z)
            rw_p = wave_resistance_pressure(mesh, L, b, t, U, RHO, G)
            row[f"rw_pressure_{n_x}x{n_z}"] = rw_p
            row[f"ratio_{n_x}x{n_z}"] = rw_p / rw_k
        payload["thin_ship_limit"].append(row)

    # 4. Froude sweep at a fixed mesh: sign of the artefact
    for Fn in FR:
        Uf = Fn * math.sqrt(G * L)
        rw_k = wave_resistance_kochin(L, B, T, Uf, RHO, G)
        rw_p = wave_resistance_pressure(wigley_hull_mesh(L, B, T, 16, 8), L, B, T, Uf, RHO, G)
        payload["froude_sweep"].append({
            "froude": Fn, "rw_kochin": rw_k, "rw_pressure": rw_p,
            "ratio": rw_p / rw_k, "rel": abs(rw_p - rw_k) / abs(rw_k),
        })

    # ------------------------------------------------------------------ figure 1
    fig = Figure(figsize=(11.0, 4.2), dpi=140)
    ax1 = fig.add_subplot(1, 3, 1)
    h = [row["h_mean"] for row in payload["ladder"]]
    rel_raw = [row["rel_raw"] for row in payload["ladder"]]
    par = [row for row in payload["parity"] if abs(row["froude"] - 0.30) < 1e-12]
    ax1.loglog(h, rel_raw, "o-", label=r"shipped sum $|R_p|/R_K$")
    ax1.loglog([L / r["n_x"] for r in par], [max(r["rel"], 1e-17) for r in par], "s--",
               label=r"exact formula, symmetric grid")
    ax1.loglog(h, [0.6 * hh / h[0] * rel_raw[0] for hh in h], ":", color="grey", label=r"$\propto h$")
    ax1.set_xlabel(r"mean panel size $h$ [m]")
    ax1.set_ylabel(r"$|R_{\rm pressure}| / R_{\rm Kochin}$")
    ax1.set_title(r"Fr = 0.30: artefact shrinks as $h\to 0$")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(1, 3, 2)
    bl = [row["B_over_L"] for row in payload["thin_ship_limit"]]
    for tag, style in (("16x8", "o-"), ("32x16", "s--")):
        ax2.semilogx(bl, [row[f"ratio_{tag}"] for row in payload["thin_ship_limit"]], style,
                     label=f"mesh {tag}")
    ax2.axhline(1.0, color="k", lw=1.0, ls=":", label="thin-ship limit target = 1")
    ax2.axhline(0.0, color="grey", lw=0.8)
    ax2.set_xlabel(r"$B/L$  (thin-ship limit $\to 0$)")
    ax2.set_ylabel(r"$R_{\rm pressure}/R_{\rm Kochin}$")
    ax2.set_title("no convergence in the thin-ship limit")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(1, 3, 3)
    xs = np.linspace(-L / 2, L / 2, 241)
    zs = np.array([-0.6 * T])
    px = dphi_dx(xs, zs, U)
    bx = -8.0 * B * xs / L**3 * (1.0 - (zs[0] / T) ** 2)
    scale = np.max(np.abs(px * bx))
    ax3.plot(xs, px * bx / scale, "-", label=r"$\varphi_x\,n_x\mathrm{d}S$ (integrand)")
    ax3.plot(xs, px / np.max(np.abs(px)), "--", label=r"$\varphi_x$ (even in $x$)")
    ax3.plot(xs, bx / np.max(np.abs(bx)), ":", label=r"$n_x\mathrm{d}S=-b_x$ (odd in $x$)")
    ax3.axhline(0.0, color="k", lw=0.6)
    ax3.set_xlabel(r"$x$ [m]")
    ax3.set_title("odd integrand $\\Rightarrow$ integral $\\equiv 0$")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "wigley_pressure_path_audit.png")

    # ------------------------------------------------------------------ figure 2
    fig2 = Figure(figsize=(6.4, 3.6), dpi=140)
    ax = fig2.add_subplot(1, 1, 1)
    X, Z = np.meshgrid(np.linspace(-L / 2, L / 2, 121), np.linspace(-T, 0.0, 61), indexing="ij")
    field = dphi_dx(X.ravel(), Z.ravel(), U).reshape(X.shape)
    lim = float(np.max(np.abs(field)))
    cs = ax.contourf(X, Z, field, levels=21, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig2.colorbar(cs, ax=ax, label=r"$\partial\varphi/\partial x$ [m/s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(r"Wave-part $\varphi_x$ on the centreplane, Fr = 0.30 (even in $x$)")
    fig2.tight_layout()
    fig2.savefig(FIG / "wigley_pressure_field.png")

    (OUT / "wigley_pressure_path_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    data = main()
    print(json.dumps({k: data[k][:2] if isinstance(data[k], list) else data[k] for k in data}, indent=2)[:4000])