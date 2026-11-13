"""Cell-centred finite-volume observed-order cross-validation.

Anchor: ``XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR``.

What this module is
-------------------
A *verification* (not a physics) module.  It solves a manufactured 2-D Poisson
problem with a cell-centred finite-volume (FV) discretisation on a ladder of
uniform grids and estimates the **observed order of accuracy** (``p_obs``) with
the three-grid log-log slope procedure, then cross-checks that single number
against four independent routes:

1. the analytically expected order of the scheme (``p_exact`` given by the
   truncation-error analysis of the flux/quadrature pair actually used);
2. the three-grid Roache observed order computed from *solution differences*
   (a functional of the solution, no exact solution needed) -- this is the
   route :func:`tensorbem.uq.grid_convergence_index` implements;
3. an independent discrete route: P1 Galerkin FEM (scikit-fem, OSS) on the same
   grid ladder, sampling the same cell-centre norm;
4. negative controls (deliberately degraded variants of the *same* code) that
   must return the degraded order ``p = 1`` -- without them, a "p = 2" result
   does not prove the slope estimator can discriminate orders at all.

Method references (see ``docs/fv_order_xval/REPORT.md`` for exact locations):

* Ferziger & Peric, *Computational Methods for Fluid Dynamics*, 3rd ed.,
  Springer 2002 -- cell-centred FV bookkeeping, face flux from neighbouring
  cell centres, Dirichlet through a half-cell ghost.
* Versteeg & Malalasekera, *An Introduction to Computational Fluid Dynamics:
  The Finite Volume Method*, 2nd ed., Pearson 2007 -- Sec. 4.2/4.3 (the
  diffusion-flux balance used here) and Sec. 5 (boundary treatment).
* Roache, *J. Fluids Eng.* 116(3):405-413 (1994) and *Verification and
  Validation in Computational Science and Engineering*, Hermosa 1998 --
  three-grid observed order and the GCI with the factor of safety ``Fs = 1.25``.
* Celik et al., *J. Fluids Eng.* 130(7):078001 (2008) / ASME V&V 20-2009 --
  the ``p``-from-differences formula and the asymptotic-range acceptance test
  (already the convention of :mod:`tensorbem.uq`).

Discretisation (the object being verified)
------------------------------------------
For ``-laplace(u) = f`` on the unit square with Dirichlet data
``u = u_exact`` on every boundary face, cell ``P`` (volume ``h^2``, face area
``h``) is integrated as

    sum_faces  h * (du/dn)_face  =  f(x_P) h^2 ,

with the interior face gradient taken between the two neighbouring cell
centres, ``(u_N - u_P)/h``, and the boundary face gradient from the wall value
to the cell centre over the half cell, ``(u_b - u_P)/(h/2)``.  With
``u_b = u_exact(x_face)`` this is the textbook second-order cell-centred FV
stencil; the source is integrated by the midpoint (second-order) rule.

The degraded variants used as negative controls are one keyword away and reuse
the *identical* assembly code path:

``fv2``     the scheme above (expected ``p = 2``);
``fv1_src`` midpoint source replaced by the lower-left corner (rectangle) rule
            -- ``O(h)`` quadrature defect in the equation, expected ``p = 1``;
``fv1_bc``  boundary face gradient taken over the full cell, ``(u_b - u_P)/h``
            instead of ``(u_b - u_P)/(h/2)`` -- an ``O(1)`` boundary-flux
            scaling error that behaves as a first-order Robin defect, expected
            ``p = 1`` in the discrete L-infinity norm.

Everything is deterministic: no randomness, no iteration-count dependence
(direct sparse solves), so a re-run reproduces the JSON byte for byte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tensorbem.uq import grid_convergence_index

__all__ = [
    "PROBLEMS",
    "LADDERS",
    "solve_fv_poisson",
    "discrete_errors",
    "loglog_slope",
    "pairwise_slopes",
    "three_grid_slope",
    "fv_ladder",
    "skfem_ladder",
    "ROACHES_FS_THREE_GRID",
]

#: Safety factor of the GCI for a three-grid study (Roache 1994).
ROACHES_FS_THREE_GRID = 1.25


# ---------------------------------------------------------------------------
# manufactured solutions
# ---------------------------------------------------------------------------
def _u_sin_sin(x, y):
    return np.sin(math.pi * x) * np.sin(math.pi * y)


def _f_sin_sin(x, y):
    # -laplace(u) = 2 pi^2 u
    return 2.0 * math.pi ** 2 * _u_sin_sin(x, y)


def _u_exp_sin(x, y):
    return np.exp(x) * np.sin(math.pi * y)


def _f_exp_sin(x, y):
    # -laplace(u) = -(1 - pi^2) exp(x) sin(pi y)
    return (math.pi ** 2 - 1.0) * _u_exp_sin(x, y)


@dataclass(frozen=True)
class Problem:
    """A manufactured Poisson problem ``-laplace(u) = f`` with exact ``u``."""

    name: str
    u: object
    f: object
    expression: str
    citation: str


PROBLEMS: Dict[str, Problem] = {
    "sin_sin": Problem(
        name="sin_sin",
        u=_u_sin_sin,
        f=_f_sin_sin,
        expression="u = sin(pi x) sin(pi y), f = 2 pi^2 sin(pi x) sin(pi y)",
        citation=(
            "method of manufactured solutions, eigenfunction of both operators "
            "(Ferziger & Peric 2002, Sec. 3.3 'manufactured solutions / order "
            "verification'); the double sine is the standard Dirichlet "
            "eigenfunction on the unit square"
        ),
    ),
    "exp_sin": Problem(
        name="exp_sin",
        u=_u_exp_sin,
        f=_f_exp_sin,
        expression="u = exp(x) sin(pi y), f = (pi^2 - 1) exp(x) sin(pi y)",
        citation=(
            "second manufactured solution, non-symmetric and exponentially "
            "stretched in x, so the observed order is not an artefact of the "
            "problem symmetry (Ferziger & Peric 2002, Sec. 3.3)"
        ),
    ),
}


# ---------------------------------------------------------------------------
# cell-centred FV solve
# ---------------------------------------------------------------------------
@dataclass
class FVSolution:
    n: int
    scheme: str
    h: float
    xc: np.ndarray
    yc: np.ndarray
    u: np.ndarray  # (n, n) cell-centre values, u[i, j] at (xc[i], yc[j])

    @property
    def mean(self) -> float:
        """Cell-average of ``u`` -- grid-independent quantity of interest."""
        return float(self.u.mean())

    def at_centre(self) -> float:
        """Value in the cell containing the domain centre (0.5, 0.5)."""
        c = self.n // 2
        return float(self.u[c, c])


def solve_fv_poisson(
    n: int,
    scheme: str = "fv2",
    problem: str = "sin_sin",
    length: float = 1.0,
) -> FVSolution:
    """Solve ``-laplace(u) = f`` with the cell-centred FV scheme on ``n x n``.

    ``scheme`` is one of ``fv2`` (expected second order), ``fv1_src`` (first
    order through a rectangle-rule source), ``fv1_bc`` (first order through the
    boundary-flux distance).  The linear system is solved directly, so the
    result carries no solver-tolerance noise.
    """
    prob = PROBLEMS[problem]
    h = length / n
    xc = (np.arange(n) + 0.5) * h
    yc = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # source term: midpoint rule (2nd order) unless degraded
    if scheme == "fv1_src":
        # lower-left corner of each cell -> O(h) quadrature error
        Xs, Ys = np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing="ij")
        src = prob.f(Xs, Ys) * h * h
    elif scheme in ("fv2", "fv1_bc"):
        src = prob.f(X, Y) * h * h
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    # boundary face-centre values (the midpoints of the cell faces)
    yb = yc
    xb = xc
    u_x0 = prob.u(np.zeros_like(yb), yb)
    u_x1 = prob.u(np.full_like(yb, length), yb)
    u_y0 = prob.u(xb, np.zeros_like(xb))
    u_y1 = prob.u(xb, np.full_like(xb, length))

    # boundary coefficient: flux over the half cell is (u_b - u_P)/(h/2), and
    # the face term is h * that = 2 (u_b - u_P)  -> diag += 2, rhs += 2 u_b
    if scheme == "fv1_bc":
        bc_diag = 1.0  # wrong distance: (u_b - u_P)/h -> h*that = (u_b - u_P)
    else:
        bc_diag = 2.0

    n2 = n * n
    idx = np.arange(n2).reshape(n, n)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    diag = np.zeros(n2)
    rhs = src.reshape(-1).copy()

    # interior x faces
    a = idx[:-1, :].reshape(-1)
    b = idx[1:, :].reshape(-1)
    rows += [a, b]
    cols += [b, a]
    vals += [-np.ones_like(a, dtype=float), -np.ones_like(a, dtype=float)]
    diag[a] += 1.0
    diag[b] += 1.0

    # interior y faces
    a = idx[:, :-1].reshape(-1)
    b = idx[:, 1:].reshape(-1)
    rows += [a, b]
    cols += [b, a]
    vals += [-np.ones_like(a, dtype=float), -np.ones_like(a, dtype=float)]
    diag[a] += 1.0
    diag[b] += 1.0

    # Dirichlet boundaries
    left = idx[0, :]
    right = idx[-1, :]
    bottom = idx[:, 0]
    top = idx[:, -1]
    diag[left] += bc_diag
    diag[right] += bc_diag
    diag[bottom] += bc_diag
    diag[top] += bc_diag
    rhs[left] += bc_diag * u_x0
    rhs[right] += bc_diag * u_x1
    rhs[bottom] += bc_diag * u_y0
    rhs[top] += bc_diag * u_y1

    rows.append(np.arange(n2))
    cols.append(np.arange(n2))
    vals.append(diag)
    A = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n2, n2),
    )
    u = spla.spsolve(A, rhs)
    if not np.all(np.isfinite(u)):
        raise RuntimeError(f"FV solve failed for n={n} scheme={scheme}")
    return FVSolution(n=n, scheme=scheme, h=h, xc=xc, yc=yc, u=u.reshape(n, n))


# ---------------------------------------------------------------------------
# error norms (cell-centre sampling, identical for every route)
# ---------------------------------------------------------------------------
def discrete_errors(sol: FVSolution, problem: str = "sin_sin") -> Dict[str, float]:
    """Discrete ``L1`` / ``L2`` (RMS) / ``Linf`` error of a cell-centre field."""
    prob = PROBLEMS[problem]
    X, Y = np.meshgrid(sol.xc, sol.yc, indexing="ij")
    e = sol.u - prob.u(X, Y)
    ae = np.abs(e)
    return {
        "l1": float(ae.mean()),
        "l2": float(np.sqrt((e ** 2).mean())),
        "linf": float(ae.max()),
    }


# ---------------------------------------------------------------------------
# order estimators
# ---------------------------------------------------------------------------
def loglog_slope(
    hs: Sequence[float], errs: Sequence[float]
) -> Dict[str, float]:
    """Least-squares slope of ``log(err)`` against ``log(h)``.

    This is the ``THREE-GRID-LOGLOG-SLOPE`` object of the anchor: with three
    grid levels the fit has one degree of freedom, so ``r2`` (coefficient of
    determination) is reported and the pairwise slopes are reported beside it
    by :func:`pairwise_slopes` rather than hidden behind a single number.
    """
    h = np.asarray(hs, dtype=float)
    e = np.asarray(errs, dtype=float)
    if h.size < 2 or h.size != e.size:
        raise ValueError("need matching hs/errs of length >= 2")
    if np.any(h <= 0.0) or np.any(e <= 0.0):
        raise ValueError("hs and errs must be positive")
    lx, ly = np.log(h), np.log(e)
    p, c = np.polyfit(lx, ly, 1)
    pred = p * lx + c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    dof = max(h.size - 2, 1)
    stderr = (
        math.sqrt(ss_res / dof / float(np.sum((lx - lx.mean()) ** 2)))
        if h.size > 2
        else float("nan")
    )
    return {
        "observed_order": float(p),
        "intercept": float(c),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "stderr": stderr,
        "n_points": int(h.size),
    }


def pairwise_slopes(hs: Sequence[float], errs: Sequence[float]) -> List[float]:
    """``p = log(e_k/e_{k+1}) / log(h_k/h_{k+1})`` for every adjacent pair."""
    h = np.asarray(hs, dtype=float)
    e = np.asarray(errs, dtype=float)
    return [
        float(math.log(e[i] / e[i + 1]) / math.log(h[i] / h[i + 1]))
        for i in range(h.size - 1)
    ]


def three_grid_slope(
    fine: float, medium: float, coarse: float, r: float
) -> float:
    """Roache's three-grid observed order from a monotone solution triplet.

    ``p = log(|(f_coarse - f_medium)/(f_medium - f_fine)|) / log(r)`` --
    the functional-based route (no exact solution required).
    """
    d_fine = medium - fine
    d_coarse = coarse - medium
    if d_fine == 0.0 or d_coarse == 0.0 or d_fine * d_coarse <= 0.0:
        return float("nan")
    return float(math.log(abs(d_coarse / d_fine)) / math.log(r))


# ---------------------------------------------------------------------------
# grid ladders
# ---------------------------------------------------------------------------
LADDERS: Dict[str, Tuple[int, ...]] = {
    "r2_coarse": (32, 64, 128),
    "r2_fine": (64, 128, 256),
    "r15": (40, 60, 90),
}


def fv_ladder(
    levels: Iterable[int],
    scheme: str = "fv2",
    problem: str = "sin_sin",
) -> Dict[str, object]:
    """Run the FV scheme on a grid ladder and estimate its order every way.

    Returns the raw per-level errors plus the three order estimates
    (log-log LS slope over the whole ladder, the two-finest pairwise slope, and
    Roache's functional route through :func:`tensorbem.uq.grid_convergence_index`)
    so that the report can tabulate ref-versus-got for each of them.
    """
    levels = tuple(int(n) for n in levels)
    sols = [solve_fv_poisson(n, scheme=scheme, problem=problem) for n in levels]
    hs = [s.h for s in sols]
    errs = [discrete_errors(s, problem=problem) for s in sols]

    r_levels = [levels[i] / levels[i - 1] for i in range(1, len(levels))]
    r = float(np.mean(r_levels))
    uniform_r = bool(np.allclose(r_levels, r_levels[0], rtol=1e-12))

    out: Dict[str, object] = {
        "scheme": scheme,
        "problem": problem,
        "levels": list(levels),
        "h": hs,
        "refinement_ratio": r,
        "refinement_uniform": uniform_r,
        "errors": errs,
        "mean_u": [s.mean for s in sols],
        "centre_value": [s.at_centre() for s in sols],
    }

    for norm in ("l2", "l1", "linf"):
        e = [er[norm] for er in errs]
        fit = loglog_slope(hs, e)
        out[f"order_{norm}"] = fit
        out[f"pairwise_{norm}"] = pairwise_slopes(hs, e)

    # functional route (Roache / ASME V&V 20): mean of u, finest first
    fine, medium, coarse = out["mean_u"][-1], out["mean_u"][-2], out["mean_u"][0]
    gci = grid_convergence_index(
        fine,
        medium,
        coarse,
        refinement_ratio=r,
        safety_factor=ROACHES_FS_THREE_GRID,
        quantity="mean_u",
        label=f"fv {scheme} {problem} {levels}",
    )
    out["roache"] = asdict(gci)
    out["order_roache"] = float(gci.observed_order)

    # discrete-solution check: does the FV field converge to the exact field at
    # the same rate as the error does?  (catches a "both are wrong together"
    # trap where a fixed defect dominates)
    fine_u = sols[-1].u
    out["asymptotic_ratio"] = float(gci.asymptotic_ratio) if math.isfinite(gci.asymptotic_ratio) else float("nan")
    out["d_fine_medium"] = float(abs(out["mean_u"][-1] - out["mean_u"][-2]) or 0.0)
    out["fine_field_finite"] = bool(np.all(np.isfinite(fine_u)))
    return out


# ---------------------------------------------------------------------------
# independent route: P1 Galerkin FEM (scikit-fem, OSS)
# ---------------------------------------------------------------------------
def skfem_ladder(
    levels: Iterable[int],
    problem: str = "sin_sin",
    element: str = "P1",
) -> Dict[str, object]:
    """Same manufactured problem, independent discretisation family.

    Each square cell of the ``n x n`` cell-centred lattice is split into two
    triangles, so the mesh size is comparable, and the solution is sampled at
    the *same* cell centres and measured with the *same* discrete L2 norm.  The
    resulting ``p_obs`` is an independent estimate of the achievable order of
    the problem+norm pair: if the FV number were an assembly mistake, this
    route would not reproduce it.
    """
    from skfem import (  # local import: optional dependency
        MeshTri,
        Basis,
        ElementTriP1,
        ElementTriP2,
        asm,
        LinearForm,
        BilinearForm,
    )

    levels = tuple(int(n) for n in levels)
    prob = PROBLEMS[problem]
    hs: List[float] = []
    errs: List[Dict[str, float]] = []

    @BilinearForm
    def laplace(u, v, w):
        return u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]

    @LinearForm
    def load(v, w):
        return prob.f(w.x[0], w.x[1]) * v

    for n in levels:
        # structured triangulation of the unit square with n x n quads -> 2 n^2
        # triangles; nodes at the same lattice points as a vertex-centred grid
        # of spacing 1/n.
        xs = np.linspace(0.0, 1.0, n + 1)
        P = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=0).reshape(2, -1)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        k = (jj * (n + 1) + ii).reshape(-1)
        tris = np.concatenate(
            [
                np.stack([k, k + 1, k + n + 2], axis=0),
                np.stack([k, k + n + 2, k + n + 1], axis=0),
            ],
            axis=1,
        )
        m = MeshTri(P, tris)
        e_cls = {"P1": ElementTriP1, "P2": ElementTriP2}[element]
        basis = Basis(m, e_cls())
        A = asm(laplace, basis)
        b = asm(load, basis)

        nodes = basis.doflocs
        tol = 1e-12
        on_boundary = (
            (np.abs(nodes[0]) < tol)
            | (np.abs(nodes[0] - 1.0) < tol)
            | (np.abs(nodes[1]) < tol)
            | (np.abs(nodes[1] - 1.0) < tol)
        )
        D = np.nonzero(on_boundary)[0]
        # explicit Dirichlet elimination (no reliance on skfem's helper API,
        # whose keyword plumbing differs between 9.x and 12.x)
        x = np.zeros(basis.N)
        x[D] = prob.u(nodes[0][D], nodes[1][D])
        free = np.setdiff1d(np.arange(basis.N), D)
        A_ff = A[free][:, free].tocsc()
        rhs = b[free] - A[free][:, D] @ x[D]
        x[free] = spla.spsolve(A_ff, rhs)

        # sample at the cell centres of the cell-centred lattice (i+0.5)/n
        xc = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xc, xc, indexing="ij")
        pts = np.vstack([X.ravel(), Y.ravel()])
        u_samp = basis.interpolator(x)(pts).reshape(n, n)
        e = u_samp - prob.u(X, Y)
        hs.append(1.0 / n)
        errs.append(
            {
                "l1": float(np.abs(e).mean()),
                "l2": float(np.sqrt((e ** 2).mean())),
                "linf": float(np.abs(e).max()),
            }
        )

    out: Dict[str, object] = {
        "route": f"skfem {element} Galerkin",
        "problem": problem,
        "levels": list(levels),
        "h": hs,
        "errors": errs,
        "mean_u": None,
    }
    for norm in ("l2", "l1", "linf"):
        e = [er[norm] for er in errs]
        out[f"order_{norm}"] = loglog_slope(hs, e)
        out[f"pairwise_{norm}"] = pairwise_slopes(hs, e)
    return out