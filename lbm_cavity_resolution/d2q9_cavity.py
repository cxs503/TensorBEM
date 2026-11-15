"""Self-contained D2Q9 lid-driven cavity solver (NumPy), D2Q9 + RLBM/TRT/BGK.

Purpose
-------
Resolution-convergence study for the JOUR-DIFFLBM cavity anchor
(feature_ledger slug ``GAP-DIFFLBM-CAV-RES``): the LBM lid-driven cavity at
Re = 100 is compared with the Ghia et al. (1982) Table I/II centreline data at
n = 65 / 97 / 129 and the observed order of accuracy ``p`` is measured from the
max-point error vs ``h`` curve.

This module re-implements, in NumPy and without any dependency on torch or on
the ``tensorlbm`` package, the *same* discretisation that the TensorLBM
verification solver uses
(``tensorlbm/verification/lid_driven_cavity/solver.py`` at TensorLBM main):

  * D2Q9, lattice speed of sound ``cs^2 = 1/3``, equilibrium truncated at
    second order in the Mach number;
  * half-way (link) bounce-back on the three stationary walls, with the Ladd
    (1994) momentum-transfer correction ``6 w_i rho (c_i . u_w)`` for the
    moving lid; wall nodes carry the wall equilibrium (rho = 1, u = u_w);
  * three collision models: BGK, TRT with the "magic" parameter
    ``Lambda = (tau-1/2)(tau_- -1/2) = 3/16``, and RLBM (Latt & Chopard 2006
    regularised non-equilibrium projection).  TensorLBM defaults to RLBM;
  * macroscopic velocity from the populations, ``u = sum f_i c_i / rho``.

Being an independent re-implementation, agreement with the torch solver in
TensorLBM is itself a verification result (see ``crosscheck_tensorlbm.py``).

Conventions
-----------
Arrays are indexed ``[y, x]``; ``y`` increases bottom -> top with the lid at
``y = n-1``.  Lengths are normalised by the cavity side ``L = n - 1`` lattice
units, so ``h = 1/L``.  Velocities are reported in units of the lid speed
``u_lid``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 lattice
# ---------------------------------------------------------------------------
C = np.array(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]],
    dtype=np.int64,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=np.int64)
CS2 = 1.0 / 3.0
MAGIC = 3.0 / 16.0


@dataclass
class CavityConfig:
    n: int = 129
    re: float = 100.0
    u_lid: float = 0.1
    collision: str = "rlbm"          # "rlbm" | "trt" | "bgk"
    max_steps: int = 200_000
    check_every: int = 500
    resid_tol: float = 1e-9          # max |du| over fluid nodes (lattice units)
    dtype: np.dtype = np.float64
    ramp_steps: int = 0

    def __post_init__(self):
        if self.n < 17:
            raise ValueError(f"grid too small: n={self.n}")
        if self.u_lid >= 0.3:
            raise ValueError(f"u_lid={self.u_lid} too high for incompressible LBM")
        self.L = self.n - 1
        self.h = 1.0 / self.L
        self.nu = self.u_lid * self.L / self.re
        self.tau = self.nu / CS2 + 0.5
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} <= 0.5: unstable")
        if self.collision not in ("rlbm", "trt", "bgk"):
            raise ValueError(f"unknown collision {self.collision!r}")

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def tau_minus(self) -> float:
        """Ghost-mode relaxation time (TRT), magic parameter Lambda = 3/16."""
        return 0.5 + MAGIC / (self.tau - 0.5)


@dataclass
class CavityResult:
    n: int
    re: float
    L: int
    h: float
    tau: float
    u_lid: float
    collision: str
    steps: int
    resid: float
    converged: bool
    y: np.ndarray = field(repr=False)
    u: np.ndarray = field(repr=False)   # vertical centreline u(y) / u_lid
    x: np.ndarray = field(repr=False)
    v: np.ndarray = field(repr=False)   # horizontal centreline v(x) / u_lid
    ux: Optional[np.ndarray] = field(default=None, repr=False)
    uy: Optional[np.ndarray] = field(default=None, repr=False)


class CavitySolver:
    """D2Q9 lid-driven cavity, half-way bounce-back walls + Ladd lid."""

    def __init__(self, cfg: CavityConfig):
        self.cfg = cfg
        n = cfg.n
        dt = cfg.dtype
        self.f = np.zeros((9, n, n), dtype=dt)
        for i in range(9):
            self.f[i] = W[i]

        # fluid = interior; wall = boundary ring
        self.fluid = np.zeros((n, n), dtype=bool)
        self.fluid[1:-1, 1:-1] = True
        self.wall = ~self.fluid

        # wall velocity field (only the lid moves)
        self.uw_x = np.zeros((n, n), dtype=dt)
        self.uw_y = np.zeros((n, n), dtype=dt)
        self.uw_x[-1, :] = cfg.u_lid

        self.ox = np.arange(n)[None, :]
        self.oy = np.arange(n)[:, None]

        # pre-compute the bounce-back links: for every direction i, the fluid
        # nodes whose neighbour in direction i is a wall node, and the Ladd
        # coefficient 6 w_i (c_i . u_wall) evaluated with the local wall speed
        self.links = []
        for i in range(1, 9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            yy = np.arange(1, n - 1)
            xx = np.arange(1, n - 1)
            Y, X = np.meshgrid(yy, xx, indexing="ij")
            Wy = Y + cy
            Wx = X + cx
            # neighbour outside the fluid interior == wall node
            m = (Wy < 1) | (Wy > n - 2) | (Wx < 1) | (Wx > n - 2)
            if not m.any():
                continue
            Py = Y[m]
            Px = X[m]
            # wall node coordinate (clamped, only used for the wall velocity)
            Wyy = np.clip(Wy[m], 0, n - 1)
            Wxx = np.clip(Wx[m], 0, n - 1)
            cu = cx * self.uw_x[Wyy, Wxx] + cy * self.uw_y[Wyy, Wxx]
            # Ladd (1994) momentum-transfer correction.  Sign: the reflected
            # population gains -6 w_i (c_i . u_w) with c_i pointing from the
            # fluid node into the wall (verified against the physical lid
            # drag direction and against the TensorLBM solver).
            coef = -6.0 * W[i] * cu
            self.links.append((i, int(OPP[i]), cx, cy, Py, Px, coef.astype(dt)))
        self._t = 0

    # -- macroscopic ------------------------------------------------------
    def macroscopic(self):
        f = self.f
        rho = f.sum(axis=0)
        rho_safe = np.where(rho > 1e-12, rho, 1e-12)
        ux = np.tensordot(C[:, 0], f, axes=(0, 0)) / rho_safe
        uy = np.tensordot(C[:, 1], f, axes=(0, 0)) / rho_safe
        return rho, ux, uy

    def _equilibrium(self, rho, ux, uy):
        u2 = ux * ux + uy * uy
        cu = C[:, 0].reshape(9, 1, 1) * ux + C[:, 1].reshape(9, 1, 1) * uy
        return W.reshape(9, 1, 1) * rho[None] * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[None]
        )

    # -- collision --------------------------------------------------------
    def collide(self):
        cfg = self.cfg
        rho, ux, uy = self.macroscopic()
        feq = self._equilibrium(rho, ux, uy)
        if cfg.collision == "bgk":
            return self.f + cfg.omega * (feq - self.f)
        if cfg.collision == "trt":
            f_plus = 0.5 * (self.f + self.f[OPP])
            f_minus = 0.5 * (self.f - self.f[OPP])
            feq_plus = 0.5 * (feq + feq[OPP])
            feq_minus = 0.5 * (feq - feq[OPP])
            return (
                self.f
                - cfg.omega * (f_plus - feq_plus)
                - (1.0 / cfg.tau_minus) * (f_minus - feq_minus)
            )
        # RLBM (Latt & Chopard 2006, eq. 15)
        fneq = self.f - feq
        cix = C[:, 0].reshape(9, 1, 1)
        ciy = C[:, 1].reshape(9, 1, 1)
        Pi_xx = (fneq * cix * cix).sum(axis=0)
        Pi_yy = (fneq * ciy * ciy).sum(axis=0)
        Pi_xy = (fneq * cix * ciy).sum(axis=0)
        Pi_tr = Pi_xx + Pi_yy
        fneq_reg = W.reshape(9, 1, 1) / (2.0 * CS2 * CS2) * (
            cix * cix * Pi_xx + ciy * ciy * Pi_yy + 2.0 * cix * ciy * Pi_xy
            - CS2 * Pi_tr
        )
        return feq + (1.0 - cfg.omega) * fneq_reg

    def stream(self, f_post):
        """Periodic roll, then link bounce-back + wall equilibrium."""
        n = self.cfg.n
        out = np.empty_like(f_post)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            out[i] = np.roll(np.roll(f_post[i], cy, axis=0), cx, axis=1)
        rho_post = f_post.sum(axis=0)
        for i, o, cx, cy, Py, Px, coef in self.links:
            out[o][Py, Px] = f_post[i][Py, Px] + coef * rho_post[Py, Px]
        feq_wall = self._equilibrium(
            np.ones((n, n), dtype=f_post.dtype), self.uw_x, self.uw_y
        )
        out[:, self.wall] = feq_wall[:, self.wall]
        return out

    def step(self):
        self._t += 1
        if self.cfg.ramp_steps > 0:
            frac = min(1.0, self._t / float(self.cfg.ramp_steps))
            self.uw_x[-1, :] = self.cfg.u_lid * frac
        f_post = self.collide()
        self.f = self.stream(f_post)

    # -- driving ----------------------------------------------------------
    def run(self, verbose: bool = False, callback=None):
        cfg = self.cfg
        _, ux_prev, uy_prev = self.macroscopic()
        resid = np.inf
        converged = False
        while self._t < cfg.max_steps:
            for _ in range(cfg.check_every):
                self.step()
            _, ux, uy = self.macroscopic()
            du = max(
                np.abs(ux[self.fluid] - ux_prev[self.fluid]).max(),
                np.abs(uy[self.fluid] - uy_prev[self.fluid]).max(),
            )
            resid = float(du)
            ux_prev, uy_prev = ux, uy
            if verbose:
                print(f"  step {self._t:7d}  resid {resid:.3e}", flush=True)
            if callback is not None:
                callback(self._t, self, resid)
            if resid < cfg.resid_tol:
                converged = True
                break
        return converged, resid

    # -- profiles ---------------------------------------------------------
    def profiles(self):
        """Centreline profiles, normalised exactly as the TensorLBM solver.

        ``u_center = ux[:, mid]`` (vertical line), ``v_center = uy[mid, :]``
        (horizontal line), abscissae ``linspace(0, 1, n)``, velocities divided
        by ``u_lid``.
        """
        rho, ux, uy = self.macroscopic()
        mid = self.cfg.n // 2
        lin = np.linspace(0.0, 1.0, self.cfg.n)
        return (
            lin.copy(), ux[:, mid] / self.cfg.u_lid,
            lin.copy(), uy[mid, :] / self.cfg.u_lid,
        )


# ---------------------------------------------------------------------------
# Ghia et al. (1982), Table I (u, vertical centreline) and Table II (v,
# horizontal centreline), Re = 100
# ---------------------------------------------------------------------------
# Transcribed from the provenance-bearing anchor of the ecosystem,
# ``benchmarks/anchors/ghia1982_provenance.json`` (TensorLBM), which is bound
# to the OCR witness ``ghia1982_tables_I_II.ocr.txt``
# sha256 3637ef9ef72a76bb090f4e31765ed2870d18db67f0f99611730e61927e060f5b
# extracted from the paper PDF sha256 1d76e463... (Ghia, Ghia & Shin 1982,
# JCP 48, 387-411, Tables I/II, pp. 398-399).
#
# NOTE (defect found by this study): TensorLBM's shipped
# ``reference_data.py`` stores Table II on **Table I's y-stations** with a
# corrupted value list, so its v-profile gate is not comparing against Ghia at
# all.  Both variants are kept here (``GHIA100_V_LEGACY_TENSORLBM``) so the
# impact on the previously reported accuracy numbers can be quantified.
# Table I stations (y, bottom -> top) and Table II stations (x) are *different*.
GHIA100_Y = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                      0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0])
GHIA100_U = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                      -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                      0.68717, 0.73722, 0.78871, 0.84123, 1.0])
GHIA100_X = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                      0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0])
GHIA100_V = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
                      0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313,
                      -0.08864, -0.07391, -0.05906, 0.0])

#: exactly what TensorLBM ``reference_data.py`` currently ships for Table II
#: (x taken from Table I; values a mixture) -- kept only for the defect study.
GHIA100_X_LEGACY_TENSORLBM = GHIA100_Y.copy()
GHIA100_V_LEGACY_TENSORLBM = np.array(
    [0.0, -0.04192, -0.04775, -0.05602, -0.06434, -0.10150, 0.15662, 0.21090,
     0.20581, 0.13641, 0.00332, -0.23151, -0.37451, -0.41057, -0.44056,
     -0.44656, 0.0])


def errors_vs_ghia(y, u, x, v, legacy_v: bool = False) -> Dict[str, float]:
    """Error metrics vs Ghia (1982) Re = 100.

    Definition identical to TensorLBM ``compute_errors``: the reference is
    interpolated (``np.interp``) onto the simulation grid, so all ``n`` grid
    nodes contribute (wall nodes included); velocities are already normalised
    by ``u_lid`` so errors are fractions of the lid speed.

    ``legacy_v=True`` reproduces the (defective) Table II currently shipped in
    TensorLBM, for impact quantification only.
    """
    u_ref = np.interp(y, GHIA100_Y, GHIA100_U)
    if legacy_v:
        v_ref = np.interp(x, GHIA100_X_LEGACY_TENSORLBM, GHIA100_V_LEGACY_TENSORLBM)
    else:
        v_ref = np.interp(x, GHIA100_X, GHIA100_V)
    du = np.abs(u - u_ref)
    dv = np.abs(v - v_ref)
    return {
        "l2_u": float(np.sqrt(np.mean(du ** 2))),
        "l2_v": float(np.sqrt(np.mean(dv ** 2))),
        "max_u": float(du.max()),
        "max_v": float(dv.max()),
        "max_uv": float(max(du.max(), dv.max())),
        "max_u_at_y": float(y[int(np.argmax(du))]),
        "max_v_at_x": float(x[int(np.argmax(dv))]),
    }


def error_at_ghia_points(y, u, x, v) -> Dict[str, object]:
    """Error sampled *at* the 17 Ghia stations (interpolate the simulation).

    Complementary metric: ``errors_vs_ghia`` can be biased by linear
    interpolation of the coarse station list between stations; here the
    simulation is interpolated onto the station abscissae instead.
    """
    keep = (y > 0.0) & (y < 1.0)
    u_s = np.interp(GHIA100_Y[1:-1], y[keep], u[keep])
    v_s = np.interp(GHIA100_X[1:-1], x[keep], v[keep])
    du = np.abs(u_s - GHIA100_U[1:-1])
    dv = np.abs(v_s - GHIA100_V[1:-1])
    return {
        "max_u_pts": float(du.max()),
        "max_v_pts": float(dv.max()),
        "max_uv_pts": float(max(du.max(), dv.max())),
        "l2_u_pts": float(np.sqrt(np.mean(du ** 2))),
        "l2_v_pts": float(np.sqrt(np.mean(dv ** 2))),
        "max_u_pt_y": float(GHIA100_Y[1:-1][int(np.argmax(du))]),
        "max_v_pt_x": float(GHIA100_X[1:-1][int(np.argmax(dv))]),
    }


def run_case(n: int, re: float = 100.0, collision: str = "rlbm",
             u_lid: float = 0.1, max_steps: int = 200_000,
             resid_tol: float = 1e-10, check_every: int = 1000,
             verbose: bool = True) -> CavityResult:
    cfg = CavityConfig(n=n, re=re, u_lid=u_lid, collision=collision,
                       max_steps=max_steps, resid_tol=resid_tol,
                       check_every=check_every)
    solver = CavitySolver(cfg)
    converged, resid = solver.run(verbose=verbose)
    y, u, x, v = solver.profiles()
    rho, ux, uy = solver.macroscopic()
    res = CavityResult(
        n=n, re=re, L=cfg.L, h=cfg.h, tau=cfg.tau, u_lid=u_lid,
        collision=collision, steps=solver._t, resid=resid, converged=converged,
        y=y, u=u, x=x, v=v, ux=ux / u_lid, uy=uy / u_lid,
    )
    res.errors = errors_vs_ghia(y, u, x, v)                     # type: ignore[attr-defined]
    res.errors_legacy = errors_vs_ghia(y, u, x, v, legacy_v=True)  # type: ignore[attr-defined]
    res.errors_pts = error_at_ghia_points(y, u, x, v)           # type: ignore[attr-defined]
    return res


def observed_order(h: List[float], e: List[float]) -> Dict[str, float]:
    """Least-squares slope of log(e) vs log(h) plus successive-pair orders."""
    h = np.asarray(h, dtype=float)
    e = np.asarray(e, dtype=float)
    slope, intercept = np.polyfit(np.log(h), np.log(e), 1)
    pairs = []
    for k in range(len(h) - 1):
        p = np.log(e[k + 1] / e[k]) / np.log(h[k + 1] / h[k])
        hmean = float(np.sqrt(h[k] * h[k + 1]))
        pairs.append({"h_mean": hmean, "p": float(p)})
    return {"slope": float(slope), "intercept": float(intercept), "pairs": pairs}
