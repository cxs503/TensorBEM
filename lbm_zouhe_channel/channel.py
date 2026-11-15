"""D2Q9 channel solver used to cross-validate the Zou-He inlet/outlet
reconstruction (anchor XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI).

Two drivers are provided:

``run_zouhe_channel``
    The anchor configuration: Zou-He velocity inlet (parabolic channel profile)
    + Zou-He pressure outlet (``rho_out = 1``), bounce-back walls, BGK
    collision, step order ``setup -> collide -> stream`` exactly as in
    ``LatFlow.Domain.Solve``.

``run_forced_channel``
    Periodic, body-force-driven plane Poiseuille channel with Guo forcing.
    It has no inlet/outlet at all, so it isolates the *lattice* discretisation
    error from the *boundary-condition* error.

Measurement conventions (important):
  * The reconstruction is applied to the pre-collision state (anchor order).
    After the subsequent collision + streaming the boundary *column* is
    refilled from the interior, so macroscopic values measured on the boundary
    column are NOT the imposed ones and are never used in the error metrics;
    profiles are taken at the channel centre column and fluxes on faces
    between interior columns.
  * Mass flux through the face right of column ``k`` is the conservative
    lattice flux ``Q(k) = sum_y (f1 + f5 + f8 - f3 - f6 - f7)(y, k)``.
"""

from __future__ import annotations

import numpy as np

from zou_he import (
    CX,
    CY,
    OPP,
    W,
    apply_inlet_velocity,
    apply_outlet_pressure,
    equilibrium,
    macroscopic,
    poiseuille_profile,
)


def _shift(a, cy, cx):
    """Return ``a`` advected by (cy, cx): dst[y, x] = src[y-cy, x-cx]."""
    out = np.zeros_like(a)
    ny, nx = a.shape[0], a.shape[1]
    sy = slice(0, ny - cy) if cy >= 0 else slice(-cy, ny)
    dy = slice(cy, ny) if cy >= 0 else slice(0, ny + cy)
    sx = slice(0, nx - cx) if cx >= 0 else slice(-cx, nx)
    dx = slice(cx, nx) if cx >= 0 else slice(0, nx + cx)
    out[dy, dx] = a[sy, sx]
    return out


def stream(f):
    """Pull streaming with zero-fill outside the domain (cf. ``tf.pad``)."""
    out = np.empty_like(f)
    for i in range(9):
        out[:, :, i] = _shift(f[:, :, i], CY[i], CX[i])
    return out


def stream_periodic_x(f):
    """Pull streaming, periodic in x and bounce-back free (walls handled by
    the collision stage)."""
    out = np.empty_like(f)
    for i in range(9):
        out[:, :, i] = np.roll(_shift(f[:, :, i], CY[i], 0), CX[i], axis=1)
    return out


def wall_mask(ny, nx):
    m = np.zeros((ny, nx), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    return m


def _bounce_back_walls(f, wall):
    f[wall] = f[wall][:, OPP]
    return f


def _collide_bgk(f, tau, wall):
    """BGK collision; wall nodes instead take the reflection of their
    *pre-collision* populations (the anchor's ``Op * F`` bounce term)."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    collided = f - (f - feq) / tau
    collided[wall] = f[:, :, OPP][wall]
    return collided


def face_flux(f, k):
    """Conservative streamwise mass flux through the face between k and k+1."""
    pos = f[:, k, 1] + f[:, k, 5] + f[:, k, 8]
    neg = f[:, k, 3] + f[:, k, 6] + f[:, k, 7]
    return float((pos - neg)[1:-1].sum())


def run_zouhe_channel(
    ny=32,
    nx=None,
    u_peak=0.08,
    nu=0.05,
    profile="reference",
    rows="all",
    rho_out=1.0,
    transverse=True,
    max_steps=40000,
    tol=1e-9,
    check_every=500,
):
    """Run the anchor inlet/outlet channel to steady state.

    Parameters
    ----------
    profile : {"reference", "corrected"}
        ``"reference"`` = the anchor's ``yp = i - 1.5`` inlet parabola;
        ``"corrected"`` = ``yp = i - 0.5`` (vanishes on both wall mid-planes).
    rows : {"all", "interior"}
        Apply the reconstruction to the whole inlet/outlet column (literal
        anchor behaviour) or to the fluid rows only (walls keep bounce-back).
    """
    nx = nx or 2 * ny
    tau = 3.0 * nu + 0.5
    wall = wall_mask(ny, nx)
    rsel = slice(0, ny) if rows == "all" else slice(1, ny - 1)

    u_in = poiseuille_profile(ny, u_peak, offset=profile)

    f = equilibrium(
        np.ones((ny, nx)),
        np.repeat(u_in[:, None], nx, axis=1),
        np.zeros((ny, nx)),
    )

    hist, steps, ux_prev = [], 0, None
    for steps in range(1, max_steps + 1):
        apply_inlet_velocity(f, u_in, rows=rsel, transverse=transverse)
        apply_outlet_pressure(f, rho_out=rho_out, rows=rsel, transverse=transverse)
        f = stream(_collide_bgk(f, tau, wall))
        if steps % check_every == 0:
            _, ux_now, _ = macroscopic(f)
            mid = ux_now[1:-1, nx // 2]
            if ux_prev is not None:
                d = float(np.max(np.abs(mid - ux_prev)))
                hist.append((steps, d))
                if d < tol * max(u_peak, 1e-12):
                    break
            ux_prev = mid

    rho, ux, uy = macroscopic(f)
    return _metrics(
        f=f, rho=rho, ux=ux, uy=uy, ny=ny, nx=nx, u_peak=u_peak, nu=nu, tau=tau,
        steps=steps, converged=(steps < max_steps), hist=hist,
        config=dict(driver="zouhe_channel", ny=ny, nx=nx, u_peak=u_peak, nu=nu,
                    profile=profile, rows=rows, rho_out=rho_out,
                    transverse=bool(transverse), u_in=u_in.tolist()),
    )


def run_forced_channel(
    ny=32, nx=None, u_peak=0.08, nu=0.05, max_steps=40000, tol=1e-9, check_every=500
):
    """Periodic body-force-driven plane Poiseuille channel (no inlet/outlet).

    Guo forcing with ``g = 8*nu*u_peak/H**2`` so the analytic peak velocity is
    ``u_peak`` by construction; any deviation is D2Q9 discretisation error.
    """
    nx = nx or 2 * ny
    tau = 3.0 * nu + 0.5
    h = float(ny - 2)
    g = 8.0 * nu * u_peak / (h * h)
    wall = wall_mask(ny, nx)

    u_in = poiseuille_profile(ny, u_peak, offset="corrected")
    f = equilibrium(
        np.ones((ny, nx)), np.repeat(u_in[:, None], nx, axis=1), np.zeros((ny, nx))
    )
    # full-momentum body-force term (sum_i fx_i c_ix = g; the 1-1/(2tau)
    # half-step split of Guo forcing is not applied, so no velocity correction
    # is needed in the equilibrium -- residual bias is O(u*g) ~ 1e-6)
    fx = 3.0 * W[None, None, :] * CX[None, None, :] * g

    hist, steps, ux_prev = [], 0, None
    for steps in range(1, max_steps + 1):
        f = stream_periodic_x(_collide_bgk(f, tau, wall) + fx)
        if steps % check_every == 0:
            _, ux_now, _ = macroscopic(f)
            mid = ux_now[1:-1, nx // 2]
            if ux_prev is not None:
                d = float(np.max(np.abs(mid - ux_prev)))
                hist.append((steps, d))
                if d < tol * u_peak:
                    break
            ux_prev = mid

    rho, ux, uy = macroscopic(f)
    return _metrics(
        f=f, rho=rho, ux=ux, uy=uy, ny=ny, nx=nx, u_peak=u_peak, nu=nu, tau=tau,
        steps=steps, converged=(steps < max_steps), hist=hist,
        config=dict(driver="forced_channel(periodic)", ny=ny, nx=nx,
                    u_peak=u_peak, nu=nu, body_force=g),
    )


def _metrics(*, f, rho, ux, uy, ny, nx, u_peak, nu, tau, steps, converged, hist, config):
    """One run's error bundle; every number below is computed, none typed."""
    xm = nx // 2
    inner = slice(1, ny - 1)
    u_mid = ux[inner, xm]
    u_an = poiseuille_profile(ny, u_peak, offset="corrected")[inner]

    a = float(np.dot(u_mid, u_an) / np.dot(u_an, u_an))
    return dict(
        config=config,
        steps=int(steps),
        converged=bool(converged),
        last_delta=float(hist[-1][1]) if hist else None,
        # ---- accuracy vs the analytic plane-Poiseuille solution
        u_peak_target=u_peak,
        u_mid_profile=u_mid.tolist(),
        u_analytic_profile=u_an.tolist(),
        u_peak_measured=float(u_mid.max()),
        err_peak=float(abs(u_mid.max() - u_peak) / u_peak),
        err_l2_profile=float(np.linalg.norm(u_mid - u_an) / np.linalg.norm(u_an)),
        err_shape=float(np.linalg.norm(u_mid - a * u_an) / np.linalg.norm(a * u_an)),
        fit_amplitude=a,
        u_wall_slip_low=float(u_mid[0]),
        u_wall_slip_high=float(u_mid[-1]),
        u_wall_slip_rel=float(max(abs(u_mid[0]), abs(u_mid[-1])) / u_peak),
        # ---- mass conservation along the channel (conservative face fluxes)
        Q_face_2=face_flux(f, 2),
        Q_face_mid=face_flux(f, xm),
        Q_face_nx3=face_flux(f, nx - 4),
        Q_rel_drift=float(
            abs(face_flux(f, nx - 4) - face_flux(f, 2)) / max(abs(face_flux(f, 2)), 1e-30)
        ),
        # ---- boundary-column artefacts (post-streaming; diagnostics only)
        rho_mean_interior=float(rho[inner, 2:-2].mean()),
        rho_boundary_inlet_col=float(rho[inner, 0].mean()),
        rho_boundary_outlet_col=float(rho[inner, -1].mean()),
        ux_boundary_inlet_col=float(ux[inner, 0].mean()),
        ux_boundary_outlet_col=float(ux[inner, -1].mean()),
        spurious_uy_inlet=float(np.max(np.abs(uy[inner, 1]))),
        spurious_uy_outlet=float(np.max(np.abs(uy[inner, -2]))),
        u_inlet_imposed_min=float(config["u_in"][1]) if config.get("u_in") else None,
        u_inlet_imposed_wall_low=float(config["u_in"][0]) if config.get("u_in") else None,
        u_inlet_imposed_wall_high=float(config["u_in"][-1]) if config.get("u_in") else None,
    )