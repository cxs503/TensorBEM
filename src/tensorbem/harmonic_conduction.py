"""Interior two-dimensional harmonic conduction with mixed thermal boundary conditions."""
from dataclasses import dataclass

import numpy as np
from scipy.special import kv

from .geometry import PanelMesh
from .kernels import integrate_double_layer_panel_many, integrate_single_layer_panel_many


def _operators(mesh, targets, wave, collocation=False):
    # Constant densities on straight panels. Integrate Laplace singularities
    # analytically on close panels and use Gauss quadrature for the remainder.
    nodes, weights = np.polynomial.legendre.leggauss(24)
    single = np.empty((len(targets), mesh.panel_count), dtype=complex)
    double = np.empty_like(single)
    for j, (start, segment, normal, length) in enumerate(zip(mesh.vertices, mesh.vectors, mesh.normals, mesh.lengths)):
        samples = start+.5*(nodes[:, None]+1)*segment
        delta = targets[:, None, :]-samples
        radius = np.linalg.norm(delta, axis=2)
        z = wave*radius
        projection = delta @ normal
        g = kv(0, z)
        dg = wave*kv(1, z)*projection/radius
        close = np.linalg.norm(targets-(start+.5*segment), axis=1) < 2*length
        g[close] += np.log(radius[close])
        dg[close] -= projection[close]/radius[close]**2
        scale = length/(4*np.pi)
        single[:, j] = scale*(g @ weights)
        double[:, j] = scale*(dg @ weights)
        if np.any(close):
            tangent = segment/length
            single[close, j] += integrate_single_layer_panel_many(targets[close], start, tangent, length, 1.)
            double[close, j] += integrate_double_layer_panel_many(targets[close], start, tangent, normal, length)
        if collocation:
            double[j, j] = 0.  # principal value on a straight self-panel
    return single, double


@dataclass(frozen=True)
class HarmonicConduction:
    mesh: PanelMesh
    wave: complex
    conductivity: float
    wall_temperature: np.ndarray
    inward_flux: np.ndarray

    def temperature(self, points):
        """Evaluate temperature phasors at interior targets of shape (N, 2)."""
        targets = np.asarray(points, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
            raise ValueError("supply finite interior coordinates shaped (N, 2)")
        if not self.mesh.contains_point(targets).all() or np.any(self.mesh.boundary_clearance(targets) == 0):
            raise ValueError("temperature targets must be strictly inside the mesh")
        single, double = _operators(self.mesh, targets, self.wave)
        return single @ (self.inward_flux/self.conductivity)-double @ self.wall_temperature


def solve_harmonic_conduction(mesh, wall_temperature, angular_frequency, *, conductivity, volumetric_heat_capacity):
    """Solve k Laplacian(T) = i omega C T on a closed CCW polygon.

    Inputs use SI units and the exp(+i omega t) phasor convention. Supply a
    uniform wall temperature or one complex value per panel midpoint. The
    returned wall flux is positive into the material, in W/m². This constant
    panel scheme needs spatial resolution of the thermal penetration depth.
    """
    return solve_mixed_harmonic_conduction(
        mesh, "temperature", wall_temperature, angular_frequency, conductivity=conductivity,
        volumetric_heat_capacity=volumetric_heat_capacity)


def solve_mixed_harmonic_conduction(mesh, boundary_types, boundary_values, angular_frequency, *,
                                    conductivity, volumetric_heat_capacity, film_coefficients=0.):
    """Solve harmonic diffusion with one condition per panel, in exp(+i omega t).

    Use "temperature" for prescribed K, "flux" for prescribed inward W/m²,
    and "film" for ambient K with q_in = h*(ambient - wall). Types, values
    and film coefficients may each be scalar or panel-sized. A zero film
    coefficient insulates. Frequency must be positive, including for pure
    flux data. The mesh describes a single bounded homogeneous material.
    """
    for value in (angular_frequency, conductivity, volumetric_heat_capacity):
        if np.ndim(value) or not np.isrealobj(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("frequency, conductivity and volumetric capacity must be finite and positive")
    n = mesh.panel_count
    kinds = np.asarray(boundary_types)
    if kinds.shape not in ((), (n,)) or not np.isin(kinds, ["temperature", "flux", "film"]).all():
        raise ValueError("use temperature, flux or film per panel")
    kinds = np.broadcast_to(kinds, (n,))
    values = np.asarray(boundary_values, dtype=complex)
    if values.shape not in ((), (n,)) or not np.isfinite(values).all():
        raise ValueError("supply finite scalar or panel boundary values")
    values = np.broadcast_to(values, (n,))
    film = np.asarray(film_coefficients)
    if (film.shape not in ((), (n,)) or not np.isrealobj(film)
            or not np.isfinite(film).all() or np.any(film < 0)):
        raise ValueError("supply finite nonnegative real film coefficients")
    film = np.where(kinds == "film", film, 0.)
    fixed = kinds == "temperature"
    wall = np.where(fixed, values, 0.)
    flux = np.where(kinds == "flux", values, film*values)
    wave = np.sqrt(1j*angular_frequency*volumetric_heat_capacity/conductivity)
    single, double = _operators(mesh, mesh.centers, wave, collocation=True)
    single /= conductivity
    double.flat[::n+1] += .5
    # H*T = S*q. Unknowns are inward flux on fixed-temperature panels and
    # temperature elsewhere. On film panels the latter also changes the flux.
    matrix = np.where(fixed, single, -double-single*film)
    unknown = np.linalg.solve(matrix, double @ wall-single @ flux)
    wall[~fixed] = unknown[~fixed]
    flux[fixed] = unknown[fixed]
    flux -= film*wall
    return HarmonicConduction(mesh, wave, conductivity, wall, flux)
