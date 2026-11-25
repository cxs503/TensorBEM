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
                # Split at the projected target. A quartic change of variable
                # resolves the weak logarithmic singularity on self panels.
                foot = u.clip(-1, 1)
                spans = np.concatenate((-1-foot, 1-foot), axis=1)
                x = (foot[:, :, None]+spans[:, :, None]*t**4).reshape(len(u), -1)
                w = (abs(spans[:, :, None])*2*t**3*weights).reshape(len(u), -1)
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


def _particular_field(points, constant, gradient, quadratic):
    x, y = points.T
    xx, xy, yy = quadratic
    return constant + points @ gradient + xx*x*x + xy*x*y + yy*y*y


def _particular_gradient(points, gradient, quadratic):
    x, y = points.T
    xx, xy, yy = quadratic
    return gradient + np.column_stack((2*xx*x+xy*y, xy*x+2*yy*y))


@dataclass(frozen=True)
class HarmonicConduction:
    mesh: PanelMesh
    wave: complex
    conductivity: float
    wall_temperature: np.ndarray
    inward_flux: np.ndarray
    quadratic: bool = False
    particular_temperature: complex = 0j
    particular_gradient: tuple[complex, complex] = (0j, 0j)
    particular_quadratic: tuple[complex, complex, complex] = (0j, 0j, 0j)

    surface_admittance: complex | np.ndarray = 0j

    @property
    def external_inward_flux(self):
        """External heat input including local wall storage, in W/m²."""
        return self.inward_flux + self.surface_admittance*self.wall_temperature

    def _interior_targets(self, points):
        targets = np.asarray(points, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
            raise ValueError("supply finite interior coordinates shaped (N, 2)")
        if not self.mesh.contains_point(targets).all() or np.any(self.mesh.boundary_clearance(targets) == 0):
            raise ValueError("temperature targets must be strictly inside the mesh")
        return targets

    def _homogeneous_traces(self):
        points = harmonic_boundary_points(self.mesh, quadratic=self.quadratic)
        normals = np.repeat(self.mesh.normals, len(points)//self.mesh.panel_count, axis=0)
        coefficients = (self.particular_temperature, self.particular_gradient, self.particular_quadratic)
        boundary = self.wall_temperature-_particular_field(points, *coefficients)
        gradient = _particular_gradient(points, self.particular_gradient, self.particular_quadratic)
        derivative = self.inward_flux/self.conductivity-np.sum(normals*gradient, axis=1)
        return boundary, derivative

    def temperature(self, points):
        """Evaluate temperature phasors at interior targets of shape (N, 2)."""
        targets = self._interior_targets(points)
        operators = _quadratic_operators if self.quadratic else _operators
        single, double = operators(self.mesh, targets, self.wave)
        boundary, derivative = self._homogeneous_traces()
        particular = _particular_field(targets, self.particular_temperature, self.particular_gradient, self.particular_quadratic)
        return particular + single @ derivative - double @ boundary

    def heat_flux(self, points):
        """Evaluate Cartesian Fourier heat-flux phasors -k grad(T) in W/m².

        Return shape (N, 2) for strictly interior targets. Integrate the
        differentiated Green kernels on straight panels, bisecting near a
        target until the subpanel length resolves the target distance.
        """
        targets = self._interior_targets(points)
        gradient = _particular_gradient(targets, self.particular_gradient, self.particular_quadratic)
        gradient = gradient.astype(complex)
        boundary, derivative = self._homogeneous_traces()
        width = 3 if self.quadratic else 1
        boundary, derivative = boundary.reshape(-1, width), derivative.reshape(-1, width)
        nodes, weights = np.polynomial.legendre.leggauss(16)
        for j, (start, edge, normal, length) in enumerate(zip(
                self.mesh.vertices, self.mesh.vectors, self.mesh.normals, self.mesh.lengths)):
            pending = [(0., 1., np.arange(len(targets)))]
            while pending:
                lo, hi, ids = pending.pop()
                mid = (lo+hi)/2
                # Targets within one subpanel length of its midpoint need
                # finer quadrature; distant targets use this interval once.
                close = np.linalg.norm(targets[ids]-(start+mid*edge), axis=1) < (hi-lo)*length
                if np.any(close) and lo < mid < hi:
                    pending.extend(((lo, mid, ids[close]), (mid, hi, ids[close])))
                    ids = ids[~close]
                if not len(ids):
                    continue
                s = mid+(hi-lo)*nodes/2
                delta = targets[ids, None, :]-(start+s[:, None]*edge)
                radius = np.linalg.norm(delta, axis=2)
                direction = delta/radius[:, :, None]
                z = self.wave*radius
                radial = self.wave*kv(1, z)
                single_grad = -radial[:, :, None]*direction
                double_grad = ((radial/radius)[:, :, None]*normal
                    - ((self.wave**2*kv(0, z)+2*radial/radius)*(direction @ normal))[:, :, None]*direction)
                if self.quadratic:
                    v = (2*s-1)/_GAUSS_ROOT
                    basis = np.column_stack((v*(v-1)/2, 1-v*v, v*(v+1)/2))
                    temp, grad_n = basis @ boundary[j], basis @ derivative[j]
                else:
                    temp, grad_n = boundary[j, 0], derivative[j, 0]
                factor = weights*(hi-lo)*length/(4*np.pi)
                gradient[ids] += np.einsum("iqc,q->ic", single_grad, factor*grad_n)
                gradient[ids] -= np.einsum("iqc,q->ic", double_grad, factor*temp)
        return -self.conductivity*gradient


def solve_harmonic_conduction(mesh, wall_temperature, angular_frequency, *, conductivity, volumetric_heat_capacity, quadratic=False,
                              volumetric_source=0., source_gradient=(0., 0.), source_quadratic=(0., 0., 0.),
                              heat_loss_coefficient=0., bulk_ambient_temperature=0.):
    """Solve k Laplacian(T) + Q + H*(Tamb - T) = i omega C T on a closed CCW polygon.

    Inputs use SI units and the exp(+i omega t) phasor convention. Supply a
    uniform wall temperature or values at harmonic_boundary_points for the
    chosen interpolation (midpoint by default, three nodes for quadratic). The
    returned wall flux is positive into the material, in W/m². The
    source in W/m³ is volumetric_source + (x,y) dot source_gradient plus
    (x², xy, y²) dot source_quadratic. The
    distributed exchange uses a uniform heat_loss_coefficient H in W/(m³ K)
    and a uniform bulk_ambient_temperature phasor Tamb in K. The
    mesh needs spatial resolution of the thermal penetration depth.
    """
    return solve_mixed_harmonic_conduction(
        mesh, "temperature", wall_temperature, angular_frequency, conductivity=conductivity,
        volumetric_heat_capacity=volumetric_heat_capacity, quadratic=quadratic, volumetric_source=volumetric_source,
        source_gradient=source_gradient, source_quadratic=source_quadratic,
        heat_loss_coefficient=heat_loss_coefficient, bulk_ambient_temperature=bulk_ambient_temperature)


def solve_mixed_harmonic_conduction(mesh, boundary_types, boundary_values, angular_frequency, *,
                                    conductivity, volumetric_heat_capacity, film_coefficients=0., wall_areal_heat_capacity=0.,
                                    quadratic=False, volumetric_source=0., source_gradient=(0., 0.), source_quadratic=(0., 0., 0.),
                                    heat_loss_coefficient=0., bulk_ambient_temperature=0.):
    """Solve harmonic diffusion with one condition per panel, in exp(+i omega t).

    Use "temperature" for prescribed K, "flux" for prescribed external inward W/m²,
    and "film" for ambient K with external inward flux h*(ambient - wall). Types, values
    and film coefficients may each be scalar or panel-sized. A zero film
    coefficient gives zero external input. A local wall_areal_heat_capacity
    in J/(m² K) stores i omega C_wall T_wall per area, so inward_flux into
    the bulk equals external_inward_flux minus this storage rate. Frequency must be positive, including for pure
    flux data. volumetric_source is the generation phasor at the coordinate origin
    in W/m³. source_gradient gives its two Cartesian derivatives at the origin in W/m⁴.
    source_quadratic gives coefficients of x², xy and y² in W/m⁵.
    A uniform heat_loss_coefficient H in W/(m³ K) adds H*(Tamb-T) to
    the volumetric heating, with uniform bulk_ambient_temperature Tamb in K.
    With quadratic=True, each panel has three independent Gauss
    samples, returned in harmonic_boundary_points order. Inputs may be scalar,
    per panel, or flat arrays in this sample order. The mesh describes a single bounded homogeneous material.
    """
    for value in (angular_frequency, conductivity, volumetric_heat_capacity):
        if np.ndim(value) or not np.isrealobj(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("frequency, conductivity and volumetric capacity must be finite and positive")
    source = np.asarray(volumetric_source, dtype=complex)
    if source.ndim or not np.isfinite(source):
        raise ValueError("volumetric_source must be a finite uniform phasor")
    source_gradient = np.asarray(source_gradient, dtype=complex)
    if source_gradient.shape != (2,) or not np.isfinite(source_gradient).all():
        raise ValueError("source_gradient must have two finite Cartesian components")
    if (np.ndim(heat_loss_coefficient) or not np.isrealobj(heat_loss_coefficient)
            or not np.isfinite(heat_loss_coefficient) or heat_loss_coefficient < 0):
        raise ValueError("heat_loss_coefficient must be finite, real and nonnegative")
    ambient = np.asarray(bulk_ambient_temperature, dtype=complex)
    if ambient.ndim or not np.isfinite(ambient):
        raise ValueError("bulk_ambient_temperature must be a finite uniform phasor")
    admittance = heat_loss_coefficient+1j*angular_frequency*volumetric_heat_capacity
    source_quadratic = np.asarray(source_quadratic, dtype=complex)
    if source_quadratic.shape != (3,) or not np.isfinite(source_quadratic).all():
        raise ValueError("source_quadratic must have three finite coefficients for x², xy and y²")
    curvature = source_quadratic/admittance
    # The Laplacian of the quadratic particular field is spatially constant.
    particular = (complex(source)+heat_loss_coefficient*ambient+2*conductivity*(curvature[0]+curvature[2]))/admittance
    gradient = source_gradient/admittance
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
    capacity = samples(wall_areal_heat_capacity)
    if (capacity.shape not in ((), (n,)) or not np.isrealobj(capacity)
            or not np.isfinite(capacity).all() or np.any(capacity < 0)):
        raise ValueError("supply finite nonnegative real wall areal heat capacities")
    surface_admittance = 1j*angular_frequency*capacity
    fixed = kinds == "temperature"
    # Subtract the polynomial particular field from both thermal traces.
    boundary = _particular_field(points, particular, gradient, curvature)
    normals = np.repeat(mesh.normals, n//mesh.panel_count, axis=0)
    normal_flux = conductivity*np.sum(normals*_particular_gradient(points, gradient, curvature), axis=1)
    shifted = values-boundary
    wall = np.where(fixed, shifted, 0.)
    exchange = np.where(fixed, 0., film+surface_admittance)
    external = np.where(kinds == "flux", values, film*shifted)
    flux = np.where(fixed, 0., external-surface_admittance*boundary-normal_flux)
    wave = np.sqrt(admittance/conductivity)
    operators = _quadratic_operators if quadratic else _operators
    single, double = operators(mesh, points, wave, collocation=True)
    single /= conductivity
    double.flat[::n+1] += .5
    # H*T = S*q. Unknowns are inward flux on fixed-temperature panels and
    # temperature elsewhere. Film exchange and wall storage both depend on temperature.
    matrix = np.where(fixed, single, -double-single*exchange)
    unknown = np.linalg.solve(matrix, double @ wall-single @ flux)
    wall[~fixed] = unknown[~fixed]
    flux[fixed] = unknown[fixed]
    flux -= exchange*wall
    return HarmonicConduction(mesh, wave, conductivity, wall+boundary, flux+normal_flux, quadratic, particular, tuple(gradient), tuple(curvature), surface_admittance)
