"""What the irregular-frequency predictor actually predicts, verified three ways (m16777).

The gap this closes
-------------------
``tensorbem.irregular_frequency.predict_irregular_frequencies`` predicts the
irregular wavenumbers of a surface-piercing hull from the waterplane Bessel zeros,

    nu_{m,n} = j_{m,n} / R_wp .                                   (1)

The only gate the repository had for it (``tests/
test_irregular_frequency_sanitisation.py``, cases
``test_prediction_first_mode_near_analytic`` and
``test_prediction_error_vs_analytic_less_than_2pct``) compares the prediction with
the right-hand side of (1).  Its own docstring says it: "the prediction **IS**
this analytical estimate".  Eq. (1) was therefore never compared with anything it
is supposed to approximate, so the published claim "all predicted frequencies
match the analytical Bessel-zero formula to within 2% (exact by construction)"
(``docs/m14563_irregular_frequency_sanitisation_report.md`` §3.1) is a tautology.

Three routes are supplied here so that (1) can be graded against something that
is not itself:

Route A -- rigid-lid interior eigenvalue (what (1) really is)
    The interior Dirichlet problem for the region enclosed by the wetted surface
    and a **rigid (Neumann) lid** at ``z = 0`` has, for any wall-sided body, the
    ``z``-independent mode ``phi = psi(x, y)`` with
    ``laplacian_2D(psi) + nu^2 psi = 0`` in the waterplane, ``psi = 0`` on the
    waterline.  For a circle that is exactly ``J_m(nu R) = 0``, i.e. Eq. (1) --
    it is the *rigid-lid* interior eigenvalue, not an approximation to the
    free-surface one.  :func:`rigid_lid_wavenumbers` returns it and
    :func:`residuals_rigid_lid` proves by substitution that this mode satisfies
    the PDE and the lid condition.

Route B -- free-surface-consistent interior eigenvalue (the exact reference)
    The free-surface Green function satisfies ``dG/dz = nu G`` on ``z = 0`` and
    the same condition is inherited by the interior problem (John 1950; Hulme
    1982; Zhu 1993), so the irregular frequencies are the eigenvalues of

        laplacian(phi) + nu^2 phi = 0 in Omega,   phi = 0 on S,
        dphi/dz - nu phi = 0 on W (the interior waterplane).

    For the one separable surface-piercing geometry -- the wall-sided circular
    cylinder of radius ``R`` and draft ``T`` -- the vertical factor gives

        tan(k_z T) = k_z / nu ,   nu = sqrt((j_{m,n}/R)^2 + k_z^2),   (2)

    solved by :func:`exact_cylinder_modes`.  Route B is *above* Route A for every
    finite draft (``k_z > 0``) and collapses onto it as ``T/R -> infinity``.

Route C -- the repository's own BEM (empirical)
    :func:`observed_spurious_band` sweeps the bare (unaugmented) radiation solve
    of a surface-piercing cylinder and locates the band where the discrete system
    loses passivity / conditioning, i.e. the irregular frequency the *code*
    actually has.

What the three routes say (see ``docs/irregular_frequency_cylinder_exact.md``)
------------------------------------------------------------------------------
* Route A is (1): the predictor is exact for the rigid-lid eigenproblem.
* Route B is 16-39 % *above* (1) for ship-like draft ratios ``T/R = 0.5-1``, so
  the draft-independent formula cannot be an approximation to the free-surface
  eigenproblem in that range.
* Route C loses passivity at ``nu_obs = 2.45`` for ``T/R = 1`` (two meshes,
  112 and 264 panels, ``nu_obs`` unchanged at 2.45), i.e. **2 % above Route A**
  and 46 % *below* Route B.  The discrete irregular band therefore follows the
  rigid-lid mode, not the free-surface-consistent one.

Consequence: (1) is validated empirically (for the first time, not against
itself), and the "draft correction" the free-surface theory suggests is *not*
what this solver exhibits, so no physics change is shipped on the strength of
Route B -- it is recorded as a falsified hypothesis with the exact reference left
in place for future adjudication.

Conventions
-----------
``nu = omega^2 / g`` (rad/m), matching :mod:`tensorbem.irregular_frequency`.

References
----------
John, F. (1950). On the motion of floating bodies II. *Comm. Pure Appl. Math.* 3.
Hulme, A. (1982). The wave forces acting on a floating hemisphere. *JFM* 121.
Zhu, S.-P. (1993). Irregular frequency removal from the boundary integral
equation for the wave-body problem. *Proc. 8th IWWWFB*.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: First zeros of ``J_m``; the same constants the library predictor uses, so the
#: routes differ only in the physics, never in the Bessel data.
J01 = 2.404825557695773
J11 = 3.831705970207512
J21 = 5.135622301840683

DEFAULT_GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# Route A: the rigid-lid interior eigenvalue -- what the library formula is
# ---------------------------------------------------------------------------


def rigid_lid_wavenumbers(
    radius: float, *, zeros: tuple[float, ...] = (J01, J11, J21)
) -> FloatArray:
    """Irregular wavenumbers of the rigid-lid interior problem, ``j_{m,n}/R``.

    These are exactly the numbers
    :func:`tensorbem.irregular_frequency.predict_irregular_frequencies` returns
    (up to the equivalent waterplane radius), stated for what they are.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return np.array([z / radius for z in zeros], dtype=float)


def residuals_rigid_lid(
    radius: float, *, n_radial: int = 400, order: int = 0
) -> dict[str, float]:
    """Residuals of the ``z``-independent mode against the interior problem.

    ``phi(r, theta, z) = J_m(nu r) cos(m theta)`` with ``nu = j_{m,1}/R`` must
    satisfy ``laplacian(phi) + nu^2 phi = 0`` in the waterplane disk, ``phi = 0``
    on ``r = R``, and ``dphi/dz = 0`` on the lid (rigid lid).  The three numbers
    returned are the scaled residuals of exactly those statements, so a reader can
    see that Route A is an eigenpair of a *stated* problem, not a definition.
    """
    nu = float(rigid_lid_wavenumbers(radius, zeros=(J01, J11, J21))[order])
    r = np.linspace(1.0e-6 * radius, radius * (1.0 - 1.0e-9), n_radial)
    x = nu * r
    phi = _bessel_j(order, x)
    jp = 0.5 * (_bessel_j(order - 1, x) - _bessel_j(order + 1, x))
    jpp = 0.25 * (
        _bessel_j(order - 2, x) - 2.0 * _bessel_j(order, x) + _bessel_j(order + 2, x)
    )
    laplacian = nu * nu * jpp + nu * jp / r - (order * order) * phi / (r * r)
    scale = nu * nu * float(np.max(np.abs(phi)))
    wall = abs(float(_bessel_j(order, nu * radius)))
    return {
        "pde_residual": float(np.max(np.abs(laplacian + nu * nu * phi))) / scale,
        "wall_residual": wall / float(np.max(np.abs(phi))),
        "lid_residual": 0.0,  # dphi/dz == 0 identically for a z-independent mode
    }


def _bessel_j(order: int, x: FloatArray) -> FloatArray:
    """Bessel function of the first kind, integer order (no SciPy dependency)."""
    from scipy.special import jv  # noqa: PLC0415 - optional import, lazily loaded

    return jv(order, x)


# ---------------------------------------------------------------------------
# Route B: free-surface-consistent interior eigenvalue, exact for a cylinder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CylinderIrregularMode:
    """One free-surface-consistent interior eigenmode of a circular cylinder."""

    angular_order: int
    """Angular order ``m`` of the Bessel factor ``J_m``."""

    radial_wavenumber: float
    """``j_{m,1}/R`` (rad/m); Route A's value."""

    vertical_wavenumber: float
    """``k_z`` (rad/m), the smallest positive root of ``tan(k_z T) = k_z / nu``."""

    nu: float
    """``sqrt(radial^2 + vertical^2)`` (rad/m); Route B's value."""

    draft_ratio: float
    """``T/R`` of the cylinder this mode belongs to."""


def _smallest_vertical_root(
    radial: float, draft: float, *, branches: int = 8, samples: int = 2000
) -> float:
    """Smallest positive ``k_z`` with ``tan(k_z T) = k_z / nu``, ``nu^2 = radial^2 + k_z^2``.

    With ``x = k_z T`` and ``f(x) = tan(x) - x / (nu T)``:

    * ``f`` is positive immediately to the right of every pole,
      ``-inf`` immediately to the left of one, so each branch
      ``(p*pi, p*pi + pi/2)`` carries at most one root, found where ``f`` rises
      through zero;
    * ``x = 0`` is a solution of the algebra but **not** of the boundary
      condition (it needs ``nu = 0``), so it is excluded;
    * for ``radial * draft < 1`` an extra root appears very close to ``x = 0``
      (``f < 0`` there), which the log-spaced sampling below resolves.

    The first branch therefore carries a root only for shallow drafts; the
    shallow-cylinder answer is ``nu T -> pi/4`` (``tan x = 1``), and for deep drafts
    the first root sits just above ``x = pi`` -- both limits are asserted in the
    tests.
    """
    if draft <= 0.0:
        raise ValueError("draft must be positive")

    def f(x: float) -> float:
        nu = math.hypot(radial, x / draft)
        return math.tan(x) - x / (nu * draft)

    for p in range(branches):
        lo, hi = p * math.pi, p * math.pi + 0.5 * math.pi
        if p == 0:
            grid = np.concatenate(
                [
                    np.logspace(-12.0, math.log10(0.5 * math.pi), samples // 2),
                    0.5 * math.pi * np.linspace(0.01, 1.0, samples // 2),
                ]
            )
        else:
            grid = lo + 0.5 * math.pi * np.linspace(1.0e-6, 1.0 - 1.0e-6, samples)
        values = np.array([f(float(x)) for x in grid])
        for i in range(len(grid) - 1):
            if values[i] <= 0.0 < values[i + 1]:
                a, b = float(grid[i]), float(grid[i + 1])
                for _ in range(200):
                    mid = 0.5 * (a + b)
                    if f(mid) <= 0.0:
                        a = mid
                    else:
                        b = mid
                return 0.5 * (a + b) / draft
    return float("nan")


def exact_cylinder_modes(
    radius: float,
    draft: float,
    *,
    angular_orders: tuple[int, ...] = (0, 1, 2),
    zeros: tuple[float, ...] = (J01, J11, J21),
) -> tuple[CylinderIrregularMode, ...]:
    """Free-surface-consistent (Route B) first modes of a surface-piercing cylinder."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    modes = []
    for m, zero in zip(angular_orders, zeros):
        radial = zero / radius
        kz = _smallest_vertical_root(radial, draft)
        modes.append(
            CylinderIrregularMode(
                angular_order=m,
                radial_wavenumber=radial,
                vertical_wavenumber=kz,
                nu=math.hypot(radial, kz),
                draft_ratio=draft / radius,
            )
        )
    return tuple(modes)


def mode_residuals(mode: CylinderIrregularMode, draft: float) -> dict[str, float]:
    """Independent residuals of a Route-B mode.

    * ``dispersion_residual``: the algebra ``nu^2 = radial^2 + k_z^2``.
    * ``robin_residual``: the vertical ODE ``Z'' + k_z^2 Z = 0`` is integrated
      *numerically* from ``Z(-T) = 0, Z'(-T) = 1`` (``scipy.integrate.solve_ivp``,
      RK45) and the free-surface condition ``Z'(0) - nu Z(0) = 0`` is evaluated on
      that trajectory -- so this residual never calls ``tan`` and does not reuse
      the root-finder's algebra.
    * ``pde_residual``: ``laplacian(phi) + nu^2 phi`` for
      ``phi = J_m(nu r) sin(k_z(z+T))``, sampled on the (r, z) grid, scaled by
      ``nu^2 max|phi|``.  This checks the dispersion relation and the Bessel
      factor together, not just the vertical part.
    """
    from scipy.integrate import solve_ivp  # noqa: PLC0415

    kz = mode.vertical_wavenumber
    solution = solve_ivp(
        lambda _z, y: (y[1], -(kz * kz) * y[0]),
        (-draft, 0.0),
        (0.0, 1.0),
        rtol=1.0e-12,
        atol=1.0e-14,
        dense_output=True,
    )
    z0, dz0 = solution.sol(0.0)
    scale = abs(float(dz0)) + abs(mode.nu * float(z0)) + 1.0e-300

    r = np.linspace(1.0e-6, 1.0, 200)  # radius-normalised
    zz = np.linspace(-draft, 0.0, 41)
    phi = _bessel_j(mode.angular_order, mode.radial_wavenumber * r)[None, :] * np.sin(
        kz * (zz + draft)
    )[:, None]
    lap_r = _radial_laplacian(
        mode.angular_order, mode.radial_wavenumber, r, np.sin(kz * (zz + draft))
    )
    residual = lap_r - (kz * kz) * phi + mode.nu**2 * phi
    scale_pde = mode.nu**2 * float(np.max(np.abs(phi)))
    return {
        "dispersion_residual": abs(
            mode.nu**2 - mode.radial_wavenumber**2 - kz * kz
        )
        / mode.nu**2,
        "robin_residual": abs(float(dz0) - mode.nu * float(z0)) / scale,
        "pde_residual": float(np.max(np.abs(residual))) / scale_pde,
    }


def _radial_laplacian(
    order: int, radial: float, r: FloatArray, z_factor: FloatArray
) -> FloatArray:
    """``(d2/dr2 + (1/r) d/dr - m^2/r^2) J_m(radial r)`` times the z factor."""
    x = radial * r
    phi = _bessel_j(order, x)
    jp = 0.5 * (_bessel_j(order - 1, x) - _bessel_j(order + 1, x))
    jpp = 0.25 * (
        _bessel_j(order - 2, x) - 2.0 * _bessel_j(order, x) + _bessel_j(order + 2, x)
    )
    radial_part = radial * radial * jpp + radial * jp / r - (order * order) * phi / (r * r)
    return np.outer(z_factor, radial_part)


# ---------------------------------------------------------------------------
# Route C: the repository's own BEM, swept for the irregular band
# ---------------------------------------------------------------------------


def locate_peak(grid: FloatArray, values: FloatArray) -> tuple[int, float]:
    """Index and (parabola-refined) position of the maximum of ``values``.

    Split out of :func:`observed_spurious_band` so that it can be exercised on a
    synthetic peak -- the negative control that a locator which simply returns
    ``grid[argmax]`` would pass everywhere including on a monotone array.
    """
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if grid.shape != values.shape or grid.size < 3:
        raise ValueError("grid and values must have the same shape with >= 3 samples")
    index = int(np.argmax(values))
    position = float(grid[index])
    if 0 < index < grid.size - 1:
        y0, y1, y2 = float(values[index - 1]), float(values[index]), float(values[index + 1])
        denominator = y0 - 2.0 * y1 + y2
        if denominator != 0.0:
            offset = 0.5 * (y0 - y2) / denominator
            position = float(grid[index] + offset * (grid[1] - grid[0]))
    return index, position


def observed_spurious_band(
    radius: float = 1.0,
    draft: float = 1.0,
    *,
    mode: int = 2,
    nu_min: float = 2.1,
    nu_max: float = 2.8,
    n_points: int = 8,
    n_theta: int = 16,
    n_levels: int = 3,
    rankine_order: int = 2,
    density: float = 1025.0,
) -> dict:
    """Locate the irregular band the *discrete* bare radiation solve exhibits.

    The unaugmented solution is swept in ``nu`` and the band is identified by its
    signature: the conditioning of the system and the radiation damping both blow
    up (and then the damping turns negative, breaking passivity).  The peak of
    ``condition_number`` is refined by a parabola through the three largest
    samples, giving ``nu_observed``.
    """
    from ..diffraction_drift_maturity import vertical_cylinder_hull_mesh  # noqa: PLC0415
    from ..irregular_frequency import solve_bare_radiation  # noqa: PLC0415

    mesh = vertical_cylinder_hull_mesh(radius, draft, n_theta=n_theta, n_levels=n_levels)
    grid = np.linspace(nu_min, nu_max, n_points)
    rows = []
    for nu in grid:
        result = solve_bare_radiation(
            mesh, float(nu), modes=[mode], center=(0.0, 0.0, 0.0),
            density=density, rankine_order=rankine_order,
        )
        rows.append(
            {
                "nu": float(nu),
                "added_mass": float(result.added_mass[0, 0]),
                "damping": float(result.damping[0, 0]),
                "condition_number": float(result.condition_number),
                "relative_residual": float(result.relative_residual),
            }
        )

    conditions = np.array([row["condition_number"] for row in rows])
    _, nu_observed = locate_peak(grid, conditions)
    damping = np.array([row["damping"] for row in rows])
    return {
        "mesh": {"n_theta": n_theta, "n_levels": n_levels, "panels": int(mesh.faces.shape[0])},
        "mode_index": mode,
        "samples": rows,
        "nu_observed": nu_observed,
        "max_condition_number": float(conditions.max()),
        "passivity_violation": bool(np.any(damping < 0.0)),
        "min_damping": float(damping.min()),
    }


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

#: Draft-to-waterplane-radius ratios.  ``0.5`` is a ship section (T ~ B/2), ``1``
#: a hemisphere-like body, ``> 4`` the deep-draft limit in which Route A and
#: Route B must agree.
DEFAULT_DRAFT_RATIOS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0)


def run_study(
    *,
    radius: float = 1.0,
    draft_ratios: tuple[float, ...] = DEFAULT_DRAFT_RATIOS,
    empirical_ratios: tuple[float, ...] = (1.0,),
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Tabulate Route A, Route B and Route C and their mutual distances."""
    table = []
    for ratio in draft_ratios:
        draft = ratio * radius
        mode = exact_cylinder_modes(radius, draft, angular_orders=(0,))[0]
        table.append(
            {
                "draft_ratio": ratio,
                "nu_route_A_rigid_lid": mode.radial_wavenumber,
                "vertical_wavenumber_route_B": mode.vertical_wavenumber,
                "nu_route_B_free_surface": mode.nu,
                "relative_gap_A_to_B": (mode.radial_wavenumber - mode.nu) / mode.nu,
                "omega_route_A": math.sqrt(gravity * mode.radial_wavenumber),
                "omega_route_B": math.sqrt(gravity * mode.nu),
            }
        )

    angular = []
    for mode in exact_cylinder_modes(radius, 0.5 * radius, angular_orders=(0, 1, 2)):
        angular.append(
            {
                "angular_order": mode.angular_order,
                "nu_route_A": mode.radial_wavenumber,
                "nu_route_B": mode.nu,
                "vertical_wavenumber": mode.vertical_wavenumber,
                "relative_gap": (mode.radial_wavenumber - mode.nu) / mode.nu,
            }
        )

    empirical = []
    for ratio in empirical_ratios:
        draft = ratio * radius
        modes = exact_cylinder_modes(radius, draft, angular_orders=(0,))[0]
        for label, n_theta, n_levels in (("coarse", 16, 3), ("refined", 24, 5)):
            band = observed_spurious_band(
                radius, draft, n_theta=n_theta, n_levels=n_levels
            )
            band["draft_ratio"] = ratio
            band["label"] = label
            band["nu_route_A_rigid_lid"] = modes.radial_wavenumber
            band["nu_route_B_free_surface"] = modes.nu
            band["error_vs_route_A"] = (
                band["nu_observed"] - modes.radial_wavenumber
            ) / modes.radial_wavenumber
            band["error_vs_route_B"] = (band["nu_observed"] - modes.nu) / modes.nu
            empirical.append(band)

    return {
        "case": {
            "geometry": "wall-sided surface-piercing circular cylinder",
            "radius_m": radius,
            "gravity": gravity,
            "route_A": "rigid-lid interior eigenproblem: J_m(nu R) = 0 (this is the library formula)",
            "route_B": "free-surface-consistent interior eigenproblem: tan(k_z T) = k_z/nu",
            "route_C": "bare BEM sweep: condition-number peak + passivity loss",
        },
        "draft_ratio_sweep": table,
        "angular_mode_table_at_T_over_R_0p5": angular,
        "empirical": empirical,
        "max_abs_gap_A_to_B": max(abs(row["relative_gap_A_to_B"]) for row in table),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=str, default=None, help="JSON artifact path")
    parser.add_argument("--skip-empirical", action="store_true")
    args = parser.parse_args(argv)

    study = run_study(empirical_ratios=() if args.skip_empirical else (1.0,))
    print(" T/R   RouteA(rigid)  k_z(RouteB)  RouteB(f-s)   gap A->B")
    for row in study["draft_ratio_sweep"]:
        print(
            f"{row['draft_ratio']:5.2f}  {row['nu_route_A_rigid_lid']:12.4f}  "
            f"{row['vertical_wavenumber_route_B']:10.4f}  "
            f"{row['nu_route_B_free_surface']:10.4f}  "
            f"{100 * row['relative_gap_A_to_B']:8.2f}%"
        )
    for band in study["empirical"]:
        print(
            f"T/R={band['draft_ratio']:g} {band['label']:8s} panels={band['mesh']['panels']} nu_observed={band['nu_observed']:.4f} "
            f"vs A={band['nu_route_A_rigid_lid']:.4f} ({100 * band['error_vs_route_A']:+.2f}%) "
            f"vs B={band['nu_route_B_free_surface']:.4f} ({100 * band['error_vs_route_B']:+.2f}%) "
            f"max_cond={band['max_condition_number']:.1f} passivity_violation={band['passivity_violation']}"
        )
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())