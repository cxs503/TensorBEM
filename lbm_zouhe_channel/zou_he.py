"""Zou-He velocity-inlet / pressure-outlet reconstruction, D2Q9.

This module is a NumPy reconstruction of the inlet/outlet treatment used by
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
(commit ``af07744``, file ``car.py``, function ``car_setup_step``), which the
feature ledger records as the anchor
``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI``:

  inlet (first column, velocity prescribed, density reconstructed)::

      rho = (f0 + f2 + f4 + 2*(f3 + f6 + f7)) / (1 - u)
      f1  = f3 + (2/3)*rho*u
      f5  = f7 + (1/6)*rho*u - 0.5*(f2 - f4)
      f8  = f6 + (1/6)*rho*u + 0.5*(f2 - f4)

  outlet (last column, density = 1 prescribed, velocity reconstructed)::

      vx  = -1 + (f0 + f2 + f4 + 2*(f1 + f5 + f8))     # rho_out = 1 assumed
      f3  = f1 - (2/3)*vx
      f7  = f5 - (1/6)*vx + 0.5*(f2 - f4)
      f6  = f8 - (1/6)*vx - 0.5*(f2 - f4)

Both are the textbook Zou & He (1997) reconstruction formulas for a plane
boundary with normal +/- x, specialised to ``u_y = 0`` (the reference drops the
transverse-momentum terms) and, at the outlet, to ``rho_out = 1``.

Design notes (deliberate, documented deviations from the reference):
  * the reference applies these formulas to the *entire* inlet/outlet column,
    including the two bounce-back wall nodes; ``apply_inlet_velocity`` takes an
    explicit ``rows`` slice so that both the literal ("all rows") and the
    wall-excluded variants can be exercised;
  * ``rho_out`` is a parameter here so that the literal ``rho_out = 1`` can be
    compared against a general pressure outlet;
  * collision in :mod:`channel` is plain BGK; the reference adds a
    Smagorinsky-LES relaxation, which is irrelevant for the laminar channel
    cross-validation but is recorded in the report.

Reference: Zou, Q. & He, X. (1997), "On pressure and velocity boundary
conditions for the lattice Boltzmann BGK model", Phys. Fluids 9, 1591.
"""

from __future__ import annotations

import numpy as np

# D2Q9 lattice, identical ordering to LatFlow/D2Q9.py
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def macroscopic(f):
    """Return (rho, ux, uy) for a population array of shape ``(ny, nx, 9)``."""
    rho = f.sum(axis=-1)
    ux = (f * CX).sum(axis=-1) / rho
    uy = (f * CY).sum(axis=-1) / rho
    return rho, ux, uy


def equilibrium(rho, ux, uy):
    """D2Q9 second-order equilibrium, shape ``(ny, nx, 9)``."""
    cu = CX[None, None, :] * ux[..., None] + CY[None, None, :] * uy[..., None]
    usq = ux * ux + uy * uy
    return W[None, None, :] * rho[..., None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[..., None]
    )


def apply_inlet_velocity(f, u_profile, rows=None, transverse=True):
    """Zou-He velocity inlet on column ``x = 0``, in place.

    Parameters
    ----------
    f : ndarray, shape (ny, nx, 9)
        Populations; the ``x = 0`` column is overwritten.
    u_profile : ndarray, shape (ny,)
        Prescribed streamwise velocity for every row of the inlet column.
    rows : slice, optional
        Which rows of the inlet column receive the condition.  ``None`` means
        all rows, i.e. the literal reference behaviour (wall nodes included).
    transverse : bool
        If True (textbook Zou-He) keep the ``-(1/2)(f2-f4)`` transverse
        non-equilibrium terms, as the reference does.  If False drop them
        (kept only for the sensitivity check in the tests).

    Returns
    -------
    rho_in : ndarray, shape (ny,)
        Density reconstructed on the inlet column (NaN outside ``rows``).
    """
    r = slice(0, f.shape[0]) if rows is None else rows
    col = f[r, 0, :]
    u = np.asarray(u_profile, dtype=float)[r]
    rho = (col[:, 0] + col[:, 2] + col[:, 4] + 2.0 * (col[:, 3] + col[:, 6] + col[:, 7])) / (
        1.0 - u
    )
    half = 0.5 * (col[:, 2] - col[:, 4]) if transverse else 0.0
    new = col.copy()
    new[:, 1] = col[:, 3] + (2.0 / 3.0) * rho * u
    new[:, 5] = col[:, 7] + (1.0 / 6.0) * rho * u - half
    new[:, 8] = col[:, 6] + (1.0 / 6.0) * rho * u + half
    f[r, 0, :] = new
    rho_full = np.full(f.shape[0], np.nan)
    rho_full[r] = rho
    return rho_full


def apply_outlet_pressure(f, rho_out=1.0, rows=None, transverse=True):
    """Zou-He pressure outlet on column ``x = nx-1``, in place.

    ``rho_out=1.0`` reproduces the reference implementation; the velocity is
    reconstructed from the known populations.  Returns ``ux`` of the outlet
    column (NaN outside ``rows``).
    """
    r = slice(0, f.shape[0]) if rows is None else rows
    col = f[r, -1, :]
    ux = (col[:, 0] + col[:, 2] + col[:, 4] + 2.0 * (col[:, 1] + col[:, 5] + col[:, 8])) / rho_out - 1.0
    half = 0.5 * (col[:, 2] - col[:, 4]) if transverse else 0.0
    new = col.copy()
    new[:, 3] = col[:, 1] - (2.0 / 3.0) * rho_out * ux
    new[:, 7] = col[:, 5] - (1.0 / 6.0) * rho_out * ux + half
    new[:, 6] = col[:, 8] - (1.0 / 6.0) * rho_out * ux - half
    f[r, -1, :] = new
    ux_full = np.full(f.shape[0], np.nan)
    ux_full[r] = ux
    return ux_full


def poiseuille_profile(ny, u_peak, offset="corrected"):
    """Parabolic channel-inlet profile over ``ny`` rows.

    ``H = ny - 2`` is the distance between the two bounce-back wall mid-planes,
    and the parabola vanishes at both wall mid-planes.  With the wall nodes at
    rows ``0`` and ``ny-1`` (their surfaces at ``y = 0.5`` and ``y = ny-1.5``)
    the wall-distance of row ``i`` is ``yp = i - 0.5``.

    ``offset="reference"`` reproduces the anchor implementation instead, which
    uses ``yp = i - 1.5`` (see ``car.py``: ``yp = i - 1.5``), i.e. the parabola
    is shifted one lattice unit towards the bottom wall.
    """
    i = np.arange(ny, dtype=float)
    yp = i - 0.5 if offset == "corrected" else i - 1.5
    h = float(ny - 2)
    return u_peak * 4.0 / (h * h) * (h * yp - yp * yp)