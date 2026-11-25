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


_GAUSS_ROOT = np.sqrt(3/5)


def harmonic_boundary_points(mesh, *, quadratic=False):
    """Return panel sample locations in panel-major order, shape (N, 2)."""
    if not quadratic:
        return mesh.centers
    offsets = np.array([-_GAUSS_ROOT, 0., _GAUSS_ROOT])/2
    return (mesh.centers[:, None, :]+mesh.vectors[:, None, :]*offsets[None, :, None]).reshape(-1, 2)


def _quadratic_operators(mesh, targets, wave, collocation=False):
    # Discontinuous quadratic Lagrange traces at three Gauss points per panel.
    # Work in panel coordinates so self interactions have exactly zero height.
    nodes, weights = np.polynomial.legendre.leggauss(24)
    t = (nodes+1)/2
    single = np.empty((len(targets), mesh.panel_count*3), dtype=complex)
    double = np.empty_like(single)
    for j, (center, tangent, normal, length) in enumerate(zip(
            mesh.centers, mesh.vectors/mesh.lengths[:, None], mesh.normals, mesh.lengths)):
        delta = targets-center
        xi = (delta @ tangent)*2/length
        height = delta @ normal
        if collocation:
            height[3*j:3*j+3] = 0.
        near = (abs(xi) < 2) & (abs(height) < length)
        cols = slice(3*j, 3*j+3)
        for close, mask in ((True, near), (False, ~near)):
            if not np.any(mask):
                continue
            u, h = xi[mask, None], height[mask, None]
            if close:
                # Split at the projected target. A cubic change of variable
                # resolves the weak logarithmic singularity on self panels.
                foot = u.clip(-1, 1)
                spans = np.concatenate((-1-foot, 1-foot), axis=1)
                x = (foot[:, :, None]+spans[:, :, None]*t**3).reshape(len(u), -1)
                w = (abs(spans[:, :, None])*1.5*t**2*weights).reshape(len(u), -1)
            else:
                x, w = nodes[None, :], weights
            r = np.hypot((u-x)*length/2, h)
            # At an endpoint projection one of the two split intervals is empty.
            r = np.where(w == 0., 1., r)
            v = x/_GAUSS_ROOT
            basis = np.stack((v*(v-1)/2, 1-v*v, v*(v+1)/2), axis=-1)
            factor = length/(4*np.pi)*w
            single[mask, cols] = np.einsum("ij,ijk->ik", factor*kv(0, wave*r), basis)
            derivative = wave*kv(1, wave*r)*h/r
            if close:
                # Integrate the sharp Laplace normal kernel times the value at
                # the projection analytically, leaving a regular difference.
                v0 = foot[:, 0]/_GAUSS_ROOT
                at_foot = np.stack((v0*(v0-1)/2, 1-v0*v0, v0*(v0+1)/2), axis=1)
                laplace = h/r**2
                regular = derivative[:, :, None]*basis-laplace[:, :, None]*at_foot[:, None, :]
                exact = integrate_double_layer_panel_many(targets[mask], mesh.vertices[j], tangent, normal, length)
                if collocation:
                    exact[np.flatnonzero(mask)//3 == j] = 0.
                double[mask, cols] = np.einsum("ij,ijk->ik", factor, regular)+exact[:, None]*at_foot
            else:
                double[mask, cols] = np.einsum("ij,ijk->ik", factor*derivative, basis)
    return single, double


@dataclass(frozen=True)
class HarmonicConduction:
    mesh: PanelMesh
    wave: complex
    conductivity: float
    wall_temperature: np.ndarray
    inward_flux: np.ndarray
    quadratic: bool = False

    def temperature(self, points):
        """Evaluate temperature phasors at interior targets of shape (N, 2)."""
        targets = np.asarray(points, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
            raise ValueError("supply finite interior coordinates shaped (N, 2)")
        if not self.mesh.contains_point(targets).all() or np.any(self.mesh.boundary_clearance(targets) == 0):
            raise ValueError("temperature targets must be strictly inside the mesh")
        operators = _quadratic_operators if self.quadratic else _operators
        single, double = operators(self.mesh, targets, self.wave)
        return single @ (self.inward_flux/self.conductivity)-double @ self.wall_temperature


def solve_harmonic_conduction(mesh, wall_temperature, angular_frequency, *, conductivity, volumetric_heat_capacity, quadratic=False):
    """Solve k Laplacian(T) = i omega C T on a closed CCW polygon.

    Inputs use SI units and the exp(+i omega t) phasor convention. Supply a
    uniform wall temperature or values at harmonic_boundary_points for the
    chosen interpolation (midpoint by default, three nodes for quadratic). The
    returned wall flux is positive into the material, in W/m². The
    mesh needs spatial resolution of the thermal penetration depth.
    """
    return solve_mixed_harmonic_conduction(
        mesh, "temperature", wall_temperature, angular_frequency, conductivity=conductivity,
        volumetric_heat_capacity=volumetric_heat_capacity, quadratic=quadratic)


def solve_mixed_harmonic_conduction(mesh, boundary_types, boundary_values, angular_frequency, *,
                                    conductivity, volumetric_heat_capacity, film_coefficients=0., quadratic=False):
    """Solve harmonic diffusion with one condition per panel, in exp(+i omega t).

    Use "temperature" for prescribed K, "flux" for prescribed inward W/m²,
    and "film" for ambient K with q_in = h*(ambient - wall). Types, values
    and film coefficients may each be scalar or panel-sized. A zero film
    coefficient insulates. Frequency must be positive, including for pure
    flux data. With quadratic=True, each panel has three independent Gauss
    samples, returned in harmonic_boundary_points order. Inputs may be scalar,
    per panel, or flat arrays in this sample order. The mesh describes a single bounded homogeneous material.
    """
    for value in (angular_frequency, conductivity, volumetric_heat_capacity):
        if np.ndim(value) or not np.isrealobj(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("frequency, conductivity and volumetric capacity must be finite and positive")
    points = harmonic_boundary_points(mesh, quadratic=quadratic)
    n = len(points)

    def samples(value):
        array = np.asarray(value)
        if quadratic and array.shape == (mesh.panel_count,):
            return np.repeat(array, 3)
        return array

    kinds = samples(boundary_types)
    if kinds.shape not in ((), (n,)) or not np.isin(kinds, ["temperature", "flux", "film"]).all():
        raise ValueError("use temperature, flux or film per panel")
    kinds = np.broadcast_to(kinds, (n,))
    values = samples(boundary_values).astype(complex)
    if values.shape not in ((), (n,)) or not np.isfinite(values).all():
        raise ValueError("supply finite scalar or panel boundary values")
    values = np.broadcast_to(values, (n,))
    film = samples(film_coefficients)
    if (film.shape not in ((), (n,)) or not np.isrealobj(film)
            or not np.isfinite(film).all() or np.any(film < 0)):
        raise ValueError("supply finite nonnegative real film coefficients")
    film = np.where(kinds == "film", film, 0.)
    fixed = kinds == "temperature"
    wall = np.where(fixed, values, 0.)
    flux = np.where(kinds == "flux", values, film*values)
    wave = np.sqrt(1j*angular_frequency*volumetric_heat_capacity/conductivity)
    operators = _quadratic_operators if quadratic else _operators
    single, double = operators(mesh, points, wave, collocation=True)
    single /= conductivity
    double.flat[::n+1] += .5
    # H*T = S*q. Unknowns are inward flux on fixed-temperature panels and
    # temperature elsewhere. On film panels the latter also changes the flux.
    matrix = np.where(fixed, single, -double-single*film)
    unknown = np.linalg.solve(matrix, double @ wall-single @ flux)
    wall[~fixed] = unknown[~fixed]
    flux[fixed] = unknown[fixed]
    flux -= film*wall
    return HarmonicConduction(mesh, wave, conductivity, wall, flux, quadratic)
