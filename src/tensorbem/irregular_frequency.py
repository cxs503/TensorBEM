"""Irregular-frequency prediction and sanitisation for 3-D frequency-domain potential-flow BEM.

Theory
------
For a surface-piercing hull the exterior Neumann boundary-integral equation

    (1/2 I - H_nu) phi = S_nu q_e

loses uniqueness at discrete wavenumbers ``nu_k`` that coincide with the
eigenvalues of the *interior* Dirichlet problem on the hull interior (John 1950;
Hulme 1982).  At those irregular frequencies the system matrix becomes nearly
singular, radiation damping can change sign, and a frequency sweep shows
unphysical spikes.

Two classical remedies are implemented here:

1. **CHIEF-point method** (Jones 1974): select M interior points and enforce
   the null-field equation ``-H_nu(x_m) phi = S_nu(x_m) q_e`` at each point.
   The over-determined system is solved by least squares.

2. **Interior-surface (lid) method** (Ohmatsu 1994; Zhu 1993): discretise the
   interior waterplane as a set of collocation points and stack the null-field
   rows.  This is a distributed version of CHIEF that damps several interior
   modes simultaneously.

Irregular-frequency prediction
-------------------------------
The irregular frequencies of a surface-piercing hull are the eigenvalues of the
interior Dirichlet problem on the waterplane lid.  For a hull with waterplane
area A_wp and characteristic length L_wp, the lowest irregular wavenumber is
approximately

    nu_0 ~ j_{0,1} / R_wp

where ``j_{0,1} = 2.4048`` is the first zero of J_0 and ``R_wp`` is the
equivalent waterplane radius.  A more accurate prediction solves the 2-D
Helmholtz eigenvalue problem on the waterplane polygon.

This module provides:
- :func:`predict_irregular_frequencies` — analytical estimate from waterplane geometry
- :func:`sanitise_with_chief` — CHIEF-point augmented solve
- :func:`sanitise_with_interior_surface` — interior-lid augmented solve
- :func:`frequency_sweep` — scan omega in [omega_min, omega_max] and compare
  bare vs sanitised added mass A(omega) and damping B(omega)
- :func:`irregular_frequency_sanitisation` — full pipeline returning sanitised
  hydrodynamic coefficients

References
----------
- John, F. (1950). "On the motion of floating bodies II". Comm. Pure Appl. Math.
- Jones, D.S. (1974). "An eigenvalue problem for the Helmholtz equation". Q. J. Mech. Appl. Math.
- Hulme, A. (1982). "The wave forces acting on a floating hemisphere". J. Fluid Mech.
- Ohmatsu, S. (1994). "Irregular frequencies in the panel method". Ship Tech. Res.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .free_surface_solver3d import (
    FreeSurface3DOperators,
    Radiation3DResult,
    assemble_free_surface_operators_3d,
    rigid_body_normals_3d,
    solve_radiation_3d,
    _DEFAULT_DENSITY,
    _DEFAULT_GRAVITY,
)
from .geometry3d import SurfaceMesh3D, WettedHullMesh
from .interior_panels import InteriorPanelSet, interior_panel_points

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

#: First zero of J_0 (axisymmetric interior mode).
_BESSEL_J01 = 2.404825557695773
#: First zero of J_1 (dipole interior mode).
_BESSEL_J11 = 3.831705970207512
#: First zero of J_2 (quadrupole interior mode).
_BESSEL_J21 = 5.135622301840683


@dataclass(frozen=True)
class IrregularFrequencyPrediction:
    """Predicted irregular frequencies for a hull."""

    waterplane_radius: float
    """Equivalent waterplane radius (m)."""

    predicted_nu: FloatArray
    """Predicted irregular wavenumbers (rad/m), sorted ascending."""

    predicted_omega: FloatArray
    """Predicted irregular frequencies (rad/s), sorted ascending."""

    mode_labels: list[str]
    """Labels for each predicted mode (e.g. 'j01_axisymmetric')."""

    gravity: float
    """Gravity used for omega conversion (m/s^2)."""


@dataclass(frozen=True)
class SanitisationResult:
    """Result of irregular-frequency sanitisation at one wavenumber."""

    wavenumber: float
    omega: float
    method: str
    """'chief' | 'interior_surface' | 'bare'."""

    added_mass: FloatArray
    """Added mass matrix (6, 6) or subset."""

    damping: FloatArray
    """Radiation damping matrix (6, 6) or subset."""

    condition_number: float
    relative_residual: float
    chief_points_used: int
    interior_lid_points: int
    modes: IntArray


@dataclass(frozen=True)
class FrequencySweepResult:
    """Full frequency sweep comparing bare and sanitised solutions."""

    omega_grid: FloatArray
    """Frequency grid (rad/s)."""

    bare_added_mass: FloatArray
    """Bare added mass array (N_omega, N_modes, N_modes)."""

    bare_damping: FloatArray
    """Bare damping array (N_omega, N_modes, N_modes)."""

    chief_added_mass: FloatArray
    """CHIEF-sanitised added mass array (N_omega, N_modes, N_modes)."""

    chief_damping: FloatArray
    """CHIEF-sanitised damping array (N_omega, N_modes, N_modes)."""

    interior_added_mass: FloatArray
    """Interior-surface-sanitised added mass array (N_omega, N_modes, N_modes)."""

    interior_damping: FloatArray
    """Interior-surface-sanitised damping array (N_omega, N_modes, N_modes)."""

    predicted_irregular_omega: FloatArray
    """Predicted irregular frequencies (rad/s)."""

    modes: IntArray
    gravity: float
    density: float


def _waterplane_equivalent_radius(mesh: SurfaceMesh3D) -> float:
    """Compute equivalent radius from waterplane area."""
    if isinstance(mesh, WettedHullMesh) and mesh.waterline_edges.shape[0] > 0:
        edges = mesh.waterline_edges
        idx = np.unique(edges.ravel())
        xy = mesh.vertices[idx][:, :2]
        # Shoelace formula for polygon area
        x = xy[:, 0]
        y = xy[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
        if area > 0.0:
            return math.sqrt(area / math.pi)
    # Fallback: use inscribed radius
    if isinstance(mesh, WettedHullMesh) and mesh.waterline_edges.shape[0] > 0:
        edges = mesh.waterline_edges
        idx = np.unique(edges.ravel())
        xy = mesh.vertices[idx][:, :2]
        centre = xy.mean(axis=0)
        radius = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
        return float(np.min(radius))
    # Fallback for closed mesh: use characteristic dimension
    centroids = mesh.centroids
    extent = np.max(centroids, axis=0) - np.min(centroids, axis=0)
    return float(np.mean(extent[:2]) / 2.0)


def predict_irregular_frequencies(
    mesh: SurfaceMesh3D,
    *,
    gravity: float = _DEFAULT_GRAVITY,
    n_modes: int = 3,
) -> IrregularFrequencyPrediction:
    """Predict irregular frequencies from waterplane geometry.

    Uses the analytical approximation that irregular wavenumbers correspond to
    the zeros of Bessel functions scaled by the equivalent waterplane radius:

        nu_{m,n} = j_{m,n} / R_wp

    where j_{m,n} is the n-th zero of J_m.

    Parameters
    ----------
    mesh :
        Wetted hull mesh (must pierce z = 0 for meaningful prediction).
    gravity :
        Gravity for omega conversion (m/s^2).
    n_modes :
        Number of mode families to predict (1 = axisymmetric only,
        2 = axisymmetric + dipole, 3 = + quadrupole).

    Returns
    -------
    IrregularFrequencyPrediction
    """
    r_wp = _waterplane_equivalent_radius(mesh)
    if r_wp <= 0.0:
        raise ValueError("could not determine waterplane radius")

    bessel_zeros: list[tuple[str, float]] = [
        ("j01_axisymmetric", _BESSEL_J01),
        ("j11_dipole", _BESSEL_J11),
        ("j21_quadrupole", _BESSEL_J21),
    ]
    zeros_to_use = bessel_zeros[:n_modes]

    nu_pred = np.array([z / r_wp for _, z in zeros_to_use], dtype=float)
    omega_pred = np.sqrt(gravity * nu_pred)

    return IrregularFrequencyPrediction(
        waterplane_radius=r_wp,
        predicted_nu=nu_pred,
        predicted_omega=omega_pred,
        mode_labels=[label for label, _ in zeros_to_use],
        gravity=gravity,
    )


def _auto_chief_points_for_hull(
    mesh: SurfaceMesh3D, n_points: int = 5
) -> FloatArray:
    """Generate asymmetric interior CHIEF points for a hull.

    Points are placed at various depths and radial offsets inside the hull,
    ensuring they are strictly below z = 0 and away from the hull surface.
    """
    if not isinstance(mesh, WettedHullMesh):
        raise ValueError("CHIEF points require a WettedHullMesh")

    areas = mesh.areas
    centroids = mesh.centroids
    hull_mean = np.sum(areas[:, None] * centroids, axis=0) / np.sum(areas)

    if mesh.waterline_edges.shape[0] > 0:
        wl_verts = np.unique(mesh.waterline_edges.ravel())
        xy_center = np.mean(mesh.vertices[wl_verts, :2], axis=0)
        base = np.array([xy_center[0], xy_center[1], 0.6 * hull_mean[2]], dtype=float)
    else:
        base = hull_mean.copy()

    min_panel_dist = float(np.min(np.linalg.norm(centroids - base[None, :], axis=1)))
    depth_margin = abs(float(base[2]))
    radius = 0.3 * min(min_panel_dist, 0.7 * depth_margin)

    # Deterministic asymmetric offsets
    if n_points == 1:
        offsets = np.array([[0.0, 0.0, 0.0]], dtype=float)
    elif n_points == 2:
        offsets = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.2]], dtype=float)
    elif n_points == 3:
        offsets = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.3, 0.2], [-0.4, 0.5, -0.15]], dtype=float
        )
    elif n_points == 4:
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.3, 0.2],
                [-0.4, 0.5, -0.15],
                [0.3, -0.55, 0.25],
            ],
            dtype=float,
        )
    else:
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.55, 0.31, 0.18],
                [-0.49, 0.53, -0.14],
                [0.34, -0.58, 0.22],
                [-0.57, -0.36, -0.25],
            ],
            dtype=float,
        )
        if n_points > 5:
            # Add more points on a spiral
            extra = []
            for k in range(5, n_points):
                angle = 2.0 * math.pi * k / n_points
                r = 0.4 * (1.0 + 0.3 * (k - 5) / max(1, n_points - 5))
                z = -0.15 + 0.1 * math.sin(angle)
                extra.append([r * math.cos(angle), r * math.sin(angle), z])
            offsets = np.vstack([offsets, extra])

    points = base[None, :] + radius * offsets
    # Ensure all points are below z = 0
    points[:, 2] = np.minimum(points[:, 2], -1.0e-10)
    return points


def sanitise_with_chief(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    n_chief: int = 5,
    chief_points: ArrayLike | None = None,
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    rankine_order: int = 5,
) -> SanitisationResult:
    """Solve radiation problem with CHIEF-point augmentation.

    Parameters
    ----------
    mesh :
        Wetted hull mesh.
    nu :
        Wavenumber (rad/m).
    n_chief :
        Number of CHIEF points (used only if chief_points is None).
    chief_points :
        Explicit CHIEF point coordinates (K, 3). If None, auto-generated.
    modes :
        Mode indices to solve (default: all 6 DOF).
    center :
        Reference point for moment calculations.
    density :
        Fluid density (kg/m^3).
    gravity :
        Gravity (m/s^2).
    rankine_order :
        Quadrature order for Rankine kernel.

    Returns
    -------
    SanitisationResult
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")

    if chief_points is None:
        chief_pts = _auto_chief_points_for_hull(mesh, n_chief)
    else:
        chief_pts = np.asarray(chief_points, dtype=float)
        if chief_pts.ndim != 2 or chief_pts.shape[1] != 3:
            raise ValueError("chief_points must have shape (K, 3)")

    result = solve_radiation_3d(
        mesh,
        nu,
        modes=modes,
        center=center,
        density=density,
        gravity=gravity,
        chief_points=chief_pts,
        interior_panels=None,
        rankine_order=rankine_order,
    )

    return SanitisationResult(
        wavenumber=nu,
        omega=result.omega,
        method="chief",
        added_mass=result.added_mass,
        damping=result.damping,
        condition_number=result.condition_number,
        relative_residual=result.relative_residual,
        chief_points_used=int(chief_pts.shape[0]),
        interior_lid_points=0,
        modes=result.modes,
    )


def sanitise_with_interior_surface(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    interior_panels: InteriorPanelSet | Literal["auto"] | None = "auto",
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    rankine_order: int = 5,
    lid_rings: int = 3,
    lid_azimuth: int = 8,
    lid_shrink: float = 0.75,
) -> SanitisationResult:
    """Solve radiation problem with interior-surface (lid) augmentation.

    Parameters
    ----------
    mesh :
        Wetted hull mesh.
    nu :
        Wavenumber (rad/m).
    interior_panels :
        Interior panel set. "auto" generates one from mesh geometry.
    modes :
        Mode indices to solve (default: all 6 DOF).
    center :
        Reference point for moment calculations.
    density :
        Fluid density (kg/m^3).
    gravity :
        Gravity (m/s^2).
    rankine_order :
        Quadrature order for Rankine kernel.
    lid_rings :
        Number of concentric rings for auto-generated lid.
    lid_azimuth :
        Number of panels on outermost ring.
    lid_shrink :
        Radial shrink factor for lid.

    Returns
    -------
    SanitisationResult
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")

    if interior_panels == "auto" or interior_panels is None:
        lid = interior_panel_points(
            mesh,
            rings=lid_rings,
            azimuth=lid_azimuth,
            shrink=lid_shrink,
        )
    elif isinstance(interior_panels, InteriorPanelSet):
        lid = interior_panels
    else:
        raise ValueError("interior_panels must be 'auto', None, or InteriorPanelSet")

    result = solve_radiation_3d(
        mesh,
        nu,
        modes=modes,
        center=center,
        density=density,
        gravity=gravity,
        chief_points=None,
        interior_panels=lid,
        rankine_order=rankine_order,
    )

    return SanitisationResult(
        wavenumber=nu,
        omega=result.omega,
        method="interior_surface",
        added_mass=result.added_mass,
        damping=result.damping,
        condition_number=result.condition_number,
        relative_residual=result.relative_residual,
        chief_points_used=0,
        interior_lid_points=int(lid.points.shape[0]),
        modes=result.modes,
    )


def solve_bare_radiation(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    rankine_order: int = 5,
) -> SanitisationResult:
    """Solve radiation problem without any irregular-frequency suppression.

    This is the baseline (unaugmented) solution that will exhibit pseudo-
    resonance peaks at irregular frequencies.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")

    result = solve_radiation_3d(
        mesh,
        nu,
        modes=modes,
        center=center,
        density=density,
        gravity=gravity,
        chief_points=None,
        interior_panels=None,
        rankine_order=rankine_order,
    )

    return SanitisationResult(
        wavenumber=nu,
        omega=result.omega,
        method="bare",
        added_mass=result.added_mass,
        damping=result.damping,
        condition_number=result.condition_number,
        relative_residual=result.relative_residual,
        chief_points_used=0,
        interior_lid_points=0,
        modes=result.modes,
    )


def frequency_sweep(
    mesh: SurfaceMesh3D,
    omega_min: float = 0.1,
    omega_max: float = 3.0,
    n_points: int = 30,
    *,
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_chief: int = 5,
    lid_rings: int = 3,
    lid_azimuth: int = 8,
    rankine_order: int = 5,
) -> FrequencySweepResult:
    """Sweep frequency band comparing bare, CHIEF, and interior-surface methods.

    Parameters
    ----------
    mesh :
        Wetted hull mesh.
    omega_min, omega_max :
        Frequency range (rad/s).
    n_points :
        Number of frequency points.
    modes :
        Mode indices (default: all 6 DOF).
    center :
        Reference point for moment calculations.
    density :
        Fluid density (kg/m^3).
    gravity :
        Gravity (m/s^2).
    n_chief :
        Number of CHIEF points.
    lid_rings, lid_azimuth :
        Interior lid parameters.
    rankine_order :
        Quadrature order.

    Returns
    -------
    FrequencySweepResult
    """
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("require 0 < omega_min < omega_max")

    omega_grid = np.linspace(omega_min, omega_max, n_points)
    g = gravity
    n_modes = 6 if modes is None else len(modes)

    bare_am = np.empty((n_points, n_modes, n_modes), dtype=float)
    bare_bm = np.empty((n_points, n_modes, n_modes), dtype=float)
    chief_am = np.empty((n_points, n_modes, n_modes), dtype=float)
    chief_bm = np.empty((n_points, n_modes, n_modes), dtype=float)
    interior_am = np.empty((n_points, n_modes, n_modes), dtype=float)
    interior_bm = np.empty((n_points, n_modes, n_modes), dtype=float)

    # Predict irregular frequencies
    pred = predict_irregular_frequencies(mesh, gravity=g)

    for i, omega in enumerate(omega_grid):
        nu = omega**2 / g

        # Bare
        bare = solve_bare_radiation(
            mesh, nu, modes=modes, center=center, density=density, gravity=g,
            rankine_order=rankine_order,
        )
        bare_am[i] = bare.added_mass
        bare_bm[i] = bare.damping

        # CHIEF
        chief = sanitise_with_chief(
            mesh, nu, n_chief=n_chief, modes=modes, center=center,
            density=density, gravity=g, rankine_order=rankine_order,
        )
        chief_am[i] = chief.added_mass
        chief_bm[i] = chief.damping

        # Interior surface
        interior = sanitise_with_interior_surface(
            mesh, nu, modes=modes, center=center, density=density, gravity=g,
            rankine_order=rankine_order, lid_rings=lid_rings, lid_azimuth=lid_azimuth,
        )
        interior_am[i] = interior.added_mass
        interior_bm[i] = interior.damping

    return FrequencySweepResult(
        omega_grid=omega_grid,
        bare_added_mass=bare_am,
        bare_damping=bare_bm,
        chief_added_mass=chief_am,
        chief_damping=chief_bm,
        interior_added_mass=interior_am,
        interior_damping=interior_bm,
        predicted_irregular_omega=pred.predicted_omega,
        modes=np.arange(n_modes, dtype=np.int64) if modes is None else np.asarray(modes, dtype=np.int64),
        gravity=g,
        density=density,
    )


def irregular_frequency_sanitisation(
    mesh: SurfaceMesh3D,
    omega: float | FloatArray,
    *,
    method: Literal["chief", "interior_surface", "both"] = "both",
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_chief: int = 5,
    lid_rings: int = 3,
    lid_azimuth: int = 8,
    rankine_order: int = 5,
) -> dict[str, Any]:
    """Full irregular-frequency sanitisation pipeline.

    This is the main entry point for the API endpoint. It predicts irregular
    frequencies, solves the radiation problem with the specified sanitisation
    method(s), and returns the sanitised hydrodynamic coefficients.

    Parameters
    ----------
    mesh :
        Wetted hull mesh.
    omega :
        Frequency or frequency array (rad/s).
    method :
        Sanitisation method: 'chief', 'interior_surface', or 'both'.
    modes :
        Mode indices (default: all 6 DOF).
    center :
        Reference point for moment calculations.
    density :
        Fluid density (kg/m^3).
    gravity :
        Gravity (m/s^2).
    n_chief :
        Number of CHIEF points.
    lid_rings, lid_azimuth :
        Interior lid parameters.
    rankine_order :
        Quadrature order.

    Returns
    -------
    dict
        Contains:
        - prediction: IrregularFrequencyPrediction data
        - results: list of SanitisationResult data for each omega
        - energy_check: bool (B(omega) >= 0 for all omega)
    """
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    if np.any(omega_arr <= 0.0):
        raise ValueError("omega must be positive")

    # Predict irregular frequencies
    prediction = predict_irregular_frequencies(mesh, gravity=gravity)

    results: list[dict[str, Any]] = []
    all_damping_nonnegative = True

    for om in omega_arr:
        nu = om**2 / gravity

        if method == "chief":
            res = sanitise_with_chief(
                mesh, nu, n_chief=n_chief, modes=modes, center=center,
                density=density, gravity=gravity, rankine_order=rankine_order,
            )
        elif method == "interior_surface":
            res = sanitise_with_interior_surface(
                mesh, nu, modes=modes, center=center, density=density,
                gravity=gravity, rankine_order=rankine_order,
                lid_rings=lid_rings, lid_azimuth=lid_azimuth,
            )
        elif method == "both":
            # Use interior surface as primary (more robust)
            res = sanitise_with_interior_surface(
                mesh, nu, modes=modes, center=center, density=density,
                gravity=gravity, rankine_order=rankine_order,
                lid_rings=lid_rings, lid_azimuth=lid_azimuth,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        # Energy conservation check: B(omega) >= 0
        diag_damping = np.diag(res.damping)
        if np.any(diag_damping < -1.0e-10):
            all_damping_nonnegative = False

        results.append({
            "omega": float(om),
            "wavenumber": float(nu),
            "method": res.method,
            "added_mass": res.added_mass.tolist(),
            "damping": res.damping.tolist(),
            "condition_number": res.condition_number,
            "relative_residual": res.relative_residual,
            "chief_points_used": res.chief_points_used,
            "interior_lid_points": res.interior_lid_points,
            "modes": res.modes.tolist(),
        })

    return {
        "prediction": {
            "waterplane_radius": float(prediction.waterplane_radius),
            "predicted_nu": prediction.predicted_nu.tolist(),
            "predicted_omega": prediction.predicted_omega.tolist(),
            "mode_labels": prediction.mode_labels,
            "gravity": float(prediction.gravity),
        },
        "results": results,
        "energy_check": all_damping_nonnegative,
        "n_frequencies": len(omega_arr),
    }
