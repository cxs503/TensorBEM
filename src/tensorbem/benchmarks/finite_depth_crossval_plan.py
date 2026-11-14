"""The finite-depth code-to-code cross-validation plan (TensorBEM vs capytaine).

Why a *plan* and not a single case
----------------------------------
``docs/capytaine_crosscheck.md`` cross-checks the 3-D deep-water solver against
capytaine, but every case in that reference was produced with
``water_depth = inf``.  The finite-depth solver
(:mod:`tensorbem.finite_depth_solver3d`) therefore still had only *internal*
anchors: the analytic multipole solution at ``h/a = 20`` (i.e. essentially deep
water), energy conservation, reciprocity, and the deep-water limit.  None of
those can see an error that lives in the finite-depth part of the kernel and
that the internal routes share -- exactly the failure mode a second,
independently written panel code does see.

capytaine's own approach to finite depth is what this plan borrows:

1. **Formulate the comparison in the similarity parameters of the problem.**
   A finite-depth radiation problem of a body of size ``a`` has two independent
   non-dimensional groups, ``k0*a`` (wave length vs body) and ``h/a`` (depth vs
   body); ``k0*h = (k0*a)(h/a)`` follows.  capytaine's finite-depth tests sweep
   ``h/a`` at fixed ``k0*a`` (and vice versa) rather than varying only ``omega``,
   because a frequency sweep at fixed dimensional depth mixes the two groups.
   Here the plan is the full ``h/a x k0*a`` grid, with ``nu = omega^2/g``
   *back-computed* per cell from ``k0 tanh(k0 h) = nu`` so that ``k0*a`` is exact
   for every depth (a naive fixed ``nu`` grid would let ``k0*a`` drift by the
   finite-depth dispersion, which is precisely the effect under test).
2. **Give the external reference its own uncertainty.**  capytaine ships two
   independent finite-depth Green-function formulations
   (``finite_depth_method = "newer"`` and ``"legacy"``); the spread between them
   is a free, code-internal band that says how well the reference itself is
   determined.  A TensorBEM/capytaine difference smaller than that band carries
   no information.
3. **Refine both codes, do not trust one.**  Both are low-order constant-strength
   panel methods, so the raw difference at one mesh size is dominated by two
   unrelated discretisation errors.  The plan evaluates a panel ladder on each
   side and compares *Richardson-extrapolated* values, reporting the observed
   order and the extrapolation residual, so a disagreement can be attributed
   either to the solver or to insufficient refinement -- instead of being
   reported as "solver disagreement" when it is mesh noise.
4. **Keep a case that must move.**  The deep-water solver (depth-blind by
   construction) is run on the same cells as a negative control: at small
   ``k0*h`` the plan must separate the two, otherwise it is not measuring depth
   at all.

Geometry
--------
A fully submerged sphere, radius ``a = 1 m``, centre ``z = -1.5 m`` (lowest point
``z = -2.5 m``).  A sphere is the only body with an independent semi-analytic
anchor in deep water (Thorne/Wang/Linton multipoles, already in
:mod:`tensorbem.analytic`), it is smooth (a low-order panel method converges, no
irregular frequencies for a submerged closed body), and its depth clearance
stays positive for every ``h/a >= 3`` in the grid.  Heave and surge are the two
radiating modes kept: the remaining four follow from axisymmetry, which is
already gated in ``tests/test_finite_depth_solver3d.py``.

Grid
----
``h/a in (3, 5, 10, 20)`` x ``k0*a in (0.5, 1.0, 2.0)`` -> 12 cells,
``k0*h in {1.5 ... 40}``, i.e. from strongly shallow-water (bottom at 1.5 times the
free-surface wavelength scale) to effectively deep water.

Tolerances are fixed *before* the run (see the constants below); the owner
acceptance limit for a literature/reference comparison is 15 %.
"""

from __future__ import annotations

import math
from typing import Iterator

#: Submerged-sphere geometry (metres).
SPHERE_RADIUS = 1.0
SPHERE_SUBMERGENCE = 1.5

#: ``h/a`` grid.  ``h = 3`` leaves 0.5 m of water under the sphere.
DEPTH_RATIOS: tuple[float, ...] = (3.0, 5.0, 10.0, 20.0)

#: ``k0*a`` grid.  ``k0*h = (k0*a)(h/a)`` follows from the grid.
KA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)

#: Icosphere subdivisions on the TensorBEM side: 20 / 80 / 320 panels.  The
#: coarse member is deliberately measurably wrong so that the observed order is
#: a measurement and not an assumption.
TENSORBEM_LEVELS: tuple[int, ...] = (0, 1, 2)

#: Cells where the ladder is extended to subdivision 3 (1280 panels) to test
#: that the order measured on 20/80/320 still holds one level further out.
EXTRA_LEVEL_CELLS: tuple[tuple[float, float], ...] = (
    (3.0, 0.5),
    (3.0, 2.0),
    (20.0, 0.5),
    (20.0, 2.0),
)
EXTRA_TENSORBEM_LEVEL = 3

#: capytaine sphere resolutions ``(n_theta, n_phi)`` -> 288 / 1152 / 4608 panels.
CAPYTAINE_RESOLUTIONS: tuple[tuple[int, int], ...] = ((12, 24), (24, 48), (48, 96))

#: Resolution used for the second (``"legacy"``) capytaine finite-depth
#: formulation; the spread against ``"newer"`` is the reference's own band.
CAPYTAINE_LEGACY_RESOLUTION: tuple[int, int] = (24, 48)
CAPYTAINE_METHOD = "newer"
CAPYTAINE_LEGACY_METHOD = "legacy"

DENSITY = 1025.0
GRAVITY = 9.80665

#: Pre-registered criteria (owner limit on any reference comparison: 15 %).
TOL_ADDED_MASS = 0.03
TOL_DAMPING = 0.05
OWNER_LIMIT = 0.15

#: Minimum sensible observed order; below this the Richardson estimate is not
#: extrapolated (the ladder is not in its asymptotic range) and the cell is
#: reported through its finest-level difference only.
MIN_OBSERVED_ORDER = 0.3

#: The negative control must separate the finite-depth solver from the
#: depth-blind deep-water solver by at least this much at ``k0*h = 1.5``.
NEGATIVE_CONTROL_MIN_GAP = 0.02


def depth_for(ratio: float) -> float:
    """Dimensional depth for a depth ratio ``h/a``."""
    return float(ratio) * SPHERE_RADIUS


def wavenumber_for(ka: float) -> float:
    """Propagating wavenumber ``k0`` for a target ``k0*a``."""
    return float(ka) / SPHERE_RADIUS


def nu_for(ka: float, depth: float) -> float:
    """``nu = omega^2/g`` whose *finite-depth* wavenumber equals ``k0 = ka/a``.

    Solves the dispersion relation ``k0 tanh(k0 h) = nu`` for ``nu`` given
    ``k0`` -- the inverse of the usual direction, which is what pins ``k0*a``
    exactly across the depth ladder.
    """
    k0 = wavenumber_for(ka)
    return k0 * math.tanh(k0 * float(depth))


def cells() -> Iterator[dict[str, float]]:
    """Iterate the ``h/a x k0*a`` grid with every derived quantity filled in."""
    for ratio in DEPTH_RATIOS:
        depth = depth_for(ratio)
        for ka in KA_GRID:
            k0 = wavenumber_for(ka)
            yield {
                "h_over_a": float(ratio),
                "ka": float(ka),
                "depth": depth,
                "k0": k0,
                "k0h": k0 * depth,
                "nu": nu_for(ka, depth),
                "omega": math.sqrt(nu_for(ka, depth) * GRAVITY),
            }


def cell_id(ratio: float, ka: float) -> str:
    """Stable string key for one grid cell."""
    return f"h_over_a={float(ratio):.1f}|ka={float(ka):.2f}"


def relaxation(delta: float) -> float:
    """A tight relative tolerance for float comparisons on the same inputs."""
    return float(delta)


__all__ = [
    "SPHERE_RADIUS",
    "SPHERE_SUBMERGENCE",
    "DEPTH_RATIOS",
    "KA_GRID",
    "TENSORBEM_LEVELS",
    "EXTRA_LEVEL_CELLS",
    "EXTRA_TENSORBEM_LEVEL",
    "CAPYTAINE_RESOLUTIONS",
    "CAPYTAINE_LEGACY_RESOLUTION",
    "CAPYTAINE_METHOD",
    "CAPYTAINE_LEGACY_METHOD",
    "DENSITY",
    "GRAVITY",
    "TOL_ADDED_MASS",
    "TOL_DAMPING",
    "OWNER_LIMIT",
    "MIN_OBSERVED_ORDER",
    "NEGATIVE_CONTROL_MIN_GAP",
    "depth_for",
    "wavenumber_for",
    "nu_for",
    "cells",
    "cell_id",
]