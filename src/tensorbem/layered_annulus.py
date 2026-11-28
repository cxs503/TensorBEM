"""Steady conduction in two concentric annuli with a thermal contact."""
from dataclasses import dataclass

import numpy as np

from .geometry import _validate_panel_count
from .multiloop import MultiLoopMesh, _evaluate_traces, annulus_mesh
from .kernels import integrate_single_layer_panel_many, integrate_double_layer_panel_many
from .solver import _MAX_CONDITION


def _particular(xy, conductivity, source, source_gradient, radial_coefficient, quadratic):
    """Particular temperature and gradient for a Cartesian quadratic heat source."""
    g = np.asarray(source_gradient)
    r2 = np.sum(xy*xy, axis=1)
    linear = xy @ g
    value = -r2*(source/4 + linear/8 + radial_coefficient*r2/16)/conductivity
    gradient = -(xy*(source/2 + linear/4 + radial_coefficient*r2/4)[:, None] + r2[:, None]*g/8)/conductivity
    if np.any(quadratic):
        xx, cross, yy = quadratic
        x, y = xy.T
        value -= (xx*x**4 + yy*y**4 + 2*cross*x**3*y)/(12*conductivity)
        gradient[:, 0] -= (xx*x**3/3 + cross*x**2*y/2)/conductivity
        gradient[:, 1] -= (yy*y**3/3 + cross*x**3/6)/conductivity
    return value, gradient


def _source_traces(mesh, conductivity, source, circular, source_gradient, radial_coefficient, quadratic):
    xy = mesh.centers
    if circular:
        radii = np.concatenate([np.full(loop.panel_count,np.linalg.norm(loop.vertices[0]))
                                for loop in mesh.loops])
        xy = xy*(radii/np.linalg.norm(xy,axis=1))[:,None]
    value, gradient = _particular(xy, conductivity, source, source_gradient, radial_coefficient, quadratic)
    return value, np.sum(gradient*mesh.normals, axis=1)


def _polygonal_sheet_cut(loop, angles):
    """Resolve ray angles into full turns, panel indices and chord fractions."""
    angles = np.asarray(angles, dtype=float)
    if not np.isfinite(angles).all():
        raise ValueError("sheet current angles must be finite")
    first = loop.vertices[0]
    turns, phase = np.divmod(angles-np.arctan2(first[1], first[0]), 2*np.pi)
    width = 2*np.pi/loop.panel_count
    position = phase/width
    panel = np.minimum(position.astype(int), loop.panel_count-1)
    fraction = .5*(1+np.tan((position-panel-.5)*width)/np.tan(.5*width))
    return turns, panel, fraction


def _integrate_polygonal_arc(loop, flux, start_angle, end_angle):
    start, end = np.broadcast_arrays(start_angle, end_angle)
    turns, panels, fractions = _polygonal_sheet_cut(loop, np.stack((start, end)))
    if flux is None:
        return np.zeros_like(fractions[0])
    heat = flux*loop.lengths
    prefix = np.concatenate(([0.], np.cumsum(heat)))
    return ((turns[1]-turns[0])*prefix[-1] + (prefix[panels[1]]-prefix[panels[0]])
            + (fractions[1]*heat[panels[1]]-fractions[0]*heat[panels[0]]))


def _circular_arc_mean(loop, samples, start_angle, end_angle):
    """Mean of the real periodic interpolant, including the pointwise limit."""
    start, end = np.broadcast_arrays(np.asarray(start_angle, dtype=float),
                                     np.asarray(end_angle, dtype=float))
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("arc angles must be finite")
    n = loop.panel_count
    coeff = np.fft.rfft(samples) / n
    width = end - start
    phase = start + width/2 - np.arctan2(loop.centers[0, 1], loop.centers[0, 0])
    mean = np.full_like(width, coeff[0].real)
    for mode in range(1, len(coeff)):
        copies = 1 if 2*mode == n else 2
        average = np.sinc(mode*width/(2*np.pi))
        mean += copies * average * (coeff[mode]*np.exp(1j*mode*phase)).real
    return mean


def _integrate_circular_arc(loop, samples, start_angle, end_angle):
    mean = _circular_arc_mean(loop, samples, start_angle, end_angle)
    width = np.subtract(end_angle, start_angle, dtype=float)
    return np.linalg.norm(loop.vertices[0]) * width * mean


@dataclass(frozen=True)
class AnnulusLayerField:
    """Solved traces for one polygonal material region; targets must be interior."""
    mesh: MultiLoopMesh
    boundary_values: np.ndarray
    normal_derivative: np.ndarray
    conductivity: float
    reference_length: float
    circular_geometry: bool = False
    volumetric_source: float = 0.
    source_gradient: tuple = (0., 0.)

    radial_source_coefficient: float = 0.
    source_quadratic: tuple = (0., 0., 0.)

    def _polygonal_extension(self, points, gradient):
        has_source = (self.volumetric_source or np.any(self.source_gradient)
                      or self.radial_source_coefficient or np.any(self.source_quadratic))
        values,derivative = self.boundary_values,self.normal_derivative
        if has_source:
            p,dn = _source_traces(self.mesh,self.conductivity,self.volumetric_source,False,self.source_gradient, self.radial_source_coefficient, self.source_quadratic)
            values,derivative = values-p,derivative-dn
        field = _evaluate_traces(self.mesh,values,derivative,self.reference_length,points,gradient=gradient)
        if has_source:
            xy = np.asarray(points)
            value, slope = _particular(xy, self.conductivity, self.volumetric_source, self.source_gradient, self.radial_source_coefficient, self.source_quadratic)
            field += slope if gradient else value
        return field

    def temperature(self, points):
        if self.circular_geometry:
            return self.circular_field(points)[0]
        return self._polygonal_extension(points,False)

    def heat_flux(self, points):
        """Cartesian Fourier heat flux in W/m², shape (number of targets, 2)."""
        if self.circular_geometry:
            return self.circular_field(points)[1]
        return -self.conductivity*self._polygonal_extension(points,True)


    def circular_mean_and_heat_rate(self, radii):
        """Circumferential mean T (K) and outward radial heat rate (W/m of axis).

        Accept scalar or array radii on the closed interval of this circular
        material. At a shared boundary, each layer gives its one-sided limit.
        Both outputs have the input shape. Positive heat flows toward larger r.
        """
        if not self.circular_geometry:
            raise ValueError("circle averages require the circular formulation")
        r = np.asarray(radii, dtype=float)
        outer, inner = (np.linalg.norm(loop.vertices[0]) for loop in self.mesh.loops)
        if not np.isfinite(r).all() or np.any((r < inner) | (r > outer)):
            raise ValueError("radii must be finite and inside the closed material interval")

        # Only the logarithmic zero mode of the harmonic remainder survives.
        wall_particular, _ = _source_traces(
            self.mesh, self.conductivity, self.volumetric_source, True, self.source_gradient,
            self.radial_source_coefficient, self.source_quadratic)
        harmonic = self.boundary_values - wall_particular
        outer_mean, inner_mean = (harmonic[sl].mean() for sl in self.mesh.loop_slices)
        log_slope = (outer_mean - inner_mean) / np.log(outer / inner)

        # Azimuthal mean of the volume source is source + beta*r².
        xx, _, yy = self.source_quadratic
        beta = self.radial_source_coefficient + (xx + yy) / 2
        temperature = (inner_mean + log_slope * np.log(r / inner)
                       - (self.volumetric_source * r**2 / 4 + beta * r**4 / 16) / self.conductivity)
        heat_rate = 2 * np.pi * (-self.conductivity * log_slope
                                  + self.volumetric_source * r**2 / 2 + beta * r**4 / 4)
        return temperature, heat_rate

    def circular_field(self, points):
        """Temperature and Cartesian flux from harmonic extension of both traces.

        For the regular concentric rings produced by solve_annulus_contact,
        interpret each midpoint temperature as a sample on the nominal circle
        at the same angle. Targets are strictly between those two circles.
        """
        xy = np.asarray(points, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("points must be a finite array of shape (M, 2)")
        radius = np.hypot(xy[:, 0], xy[:, 1])
        outer, inner = (np.linalg.norm(loop.vertices[0]) for loop in self.mesh.loops)
        if np.any((radius <= inner) | (radius >= outer)):
            raise ValueError("targets must be inside the selected circular material")
        return self._circular_extension(xy, radius, inner, outer)

    def circular_boundary_field(self, boundary, angles):
        """One-sided temperature (K) and Cartesian heat flux (W/m²) on a circle.

        Select "inner" or "outer" relative to this material layer. Supply a
        vector of azimuths in radians measured counterclockwise from +x.
        """
        if not self.circular_geometry:
            raise ValueError("boundary fields require the circular formulation")
        if boundary not in ("inner", "outer"):
            raise ValueError("boundary must be 'inner' or 'outer'")
        angles = np.asarray(angles, dtype=float)
        if angles.ndim != 1 or not np.isfinite(angles).all():
            raise ValueError("angles must be a finite vector in radians")
        outer, inner = (np.linalg.norm(loop.vertices[0]) for loop in self.mesh.loops)
        radius = np.full(angles.shape, inner if boundary == "inner" else outer)
        xy = radius[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        return self._circular_extension(xy, radius, inner, outer)

    def polygonal_arc_heat_rate(self, boundary, start_angle, end_angle):
        """Integrate straight-panel normal heat flux over oriented ray cuts (W/m).

        Select the layer's inner or outer polygon. Positive flux points out
        of the enclosed polygon on either boundary. Unwrapped angles in radians
        broadcast, with reversed limits changing the sign of the integral.
        """
        if self.circular_geometry:
            raise ValueError("polygonal arc heat rates require the straight-panel formulation")
        if boundary not in ("inner", "outer"):
            raise ValueError("boundary must be 'inner' or 'outer'")
        index = 1 if boundary == "inner" else 0
        normal = self.normal_derivative[self.mesh.loop_slices[index]]
        flux = self.conductivity * normal * (1 if index == 1 else -1)
        return _integrate_polygonal_arc(self.mesh.loops[index], flux, start_angle, end_angle)

    def circular_arc_mean_temperature(self, boundary, start_angle, end_angle):
        """One-sided mean boundary temperature (K) on a nominal circle.

        The unwrapped angles in radians broadcast together. Reversing the
        endpoints preserves the mean. Equal endpoints give the local trace.
        """
        if not self.circular_geometry:
            raise ValueError("circular arc temperatures require the circular formulation")
        if boundary not in ("inner", "outer"):
            raise ValueError("boundary must be 'inner' or 'outer'")
        index = 1 if boundary == "inner" else 0
        samples = self.boundary_values[self.mesh.loop_slices[index]]
        return _circular_arc_mean(self.mesh.loops[index], samples, start_angle, end_angle)

    def circular_arc_heat_rate(self, boundary, start_angle, end_angle):
        """Integrate the sampled radial boundary flux over oriented arcs (W/m).

        Select this layer's "inner" or "outer" nominal circle. Angles in
        radians broadcast together. For increasing angles the sign is positive
        toward larger radius. Use an unwrapped end angle to cross 2π.
        The integral uses the real trigonometric interpolant of the solved
        normal derivatives, including the cosine Nyquist mode for even grids.
        """
        if not self.circular_geometry:
            raise ValueError("arc heat rates require the circular formulation")
        if boundary not in ("inner", "outer"):
            raise ValueError("boundary must be 'inner' or 'outer'")
        index = 1 if boundary == "inner" else 0
        loop = self.mesh.loops[index]
        normal = self.normal_derivative[self.mesh.loop_slices[index]]
        radial_flux = self.conductivity * normal * (1 if index == 1 else -1)
        return _integrate_circular_arc(loop, radial_flux, start_angle, end_angle)

    def circular_cut_heat_rate(self, angles):
        """Heat crossing each full radial cut toward increasing angle, W/m of axis.

        Angles are in radians. Each cut extends from this material's inner
        circle to its outer circle. The result has the input shape.
        """
        if not self.circular_geometry:
            raise ValueError("cut heat rates require the circular formulation")
        angles = np.asarray(angles, dtype=float)
        if not np.isfinite(angles).all():
            raise ValueError("cut angles must be finite")
        theta = angles.ravel()
        outer, inner = (np.linalg.norm(loop.vertices[0]) for loop in self.mesh.loops)
        span = np.log(outer/inner)
        wall_particular, _ = _source_traces(
            self.mesh, self.conductivity, self.volumetric_source, True, self.source_gradient,
            self.radial_source_coefficient, self.source_quadratic)
        harmonic = self.boundary_values - wall_particular
        rate = np.zeros_like(theta)
        for loop, sl in zip(self.mesh.loops, self.mesh.loop_slices):
            n = loop.panel_count
            coeff = np.fft.rfft(harmonic[sl]) / n
            phase = theta - np.arctan2(loop.centers[0, 1], loop.centers[0, 0])
            for m in range(1, len(coeff)):
                # Integrating either radial basis against dr/r yields tanh(m*span/2)/m.
                wave = coeff[m] * np.exp(1j*m*phase)
                copies = 1 if 2*m == n else 2
                rate += self.conductivity * copies * np.tanh(m*span/2) * wave.imag

        # The particular heat flux is at most cubic along a straight radial cut.
        direction = np.column_stack((np.cos(theta), np.sin(theta)))
        tangent = np.column_stack((-direction[:, 1], direction[:, 0]))
        half_width = (outer-inner)/2
        for sign in (-1, 1):
            radius = (outer+inner)/2 + sign*half_width/np.sqrt(3)
            _, gradient = _particular(radius*direction, self.conductivity, self.volumetric_source,
                                      self.source_gradient, self.radial_source_coefficient, self.source_quadratic)
            rate -= self.conductivity * half_width * np.sum(gradient*tangent, axis=1)
        return rate.reshape(angles.shape)

    def circular_sector_generation(self, start_angle, end_angle):
        """Integrated volumetric heating (W/m of axis) in a full-thickness sector.

        Use unwrapped, counterclockwise angles in radians as for sector outflow.
        This integral uses the material source coefficients only.
        """
        if not self.circular_geometry:
            raise ValueError("sector heat generation requires the circular formulation")
        start, end = np.broadcast_arrays(np.asarray(start_angle, dtype=float),
                                         np.asarray(end_angle, dtype=float))
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError("sector angles must be finite")
        outer, inner = (np.linalg.norm(loop.vertices[0]) for loop in self.mesh.loops)
        width = end - start
        mid = start + width/2
        # Angular averages of the linear and quadratic Cartesian source terms.
        gx, gy = self.source_gradient
        xx, xy, yy = self.source_quadratic
        linear = np.sinc(width/(2*np.pi)) * (gx*np.cos(mid) + gy*np.sin(mid))
        quadratic = (self.radial_source_coefficient + (xx+yy)/2
                     + np.sinc(width/np.pi) * ((xx-yy)*np.cos(2*mid) + xy*np.sin(2*mid))/2)
        return width * (self.volumetric_source*(outer**2-inner**2)/2
                        + linear*(outer**3-inner**3)/3 + quadratic*(outer**4-inner**4)/4)

    def circular_sector_outflow(self, start_angle, end_angle):
        """Net outward heat rate (W/m of axis) for a full-thickness sector.

        The sector runs counterclockwise from start to end in unwrapped
        radians. Exchanging the angles reverses the sign. Arrays broadcast.
        All four boundary contributions are evaluated from the solved field.
        """
        inner = self.circular_arc_heat_rate("inner", start_angle, end_angle)
        outer = self.circular_arc_heat_rate("outer", start_angle, end_angle)
        cuts = np.stack(np.broadcast_arrays(start_angle, end_angle))
        start_cut, end_cut = self.circular_cut_heat_rate(cuts)
        return outer - inner + (end_cut - start_cut)

    def _circular_extension(self, xy, radius, inner, outer):
        span = np.log(outer/inner)
        log_r = np.log(radius/inner)
        angle = np.arctan2(xy[:, 1], xy[:, 0])
        temperature, slope_xy = _particular(xy, self.conductivity, self.volumetric_source, self.source_gradient, self.radial_source_coefficient, self.source_quadratic)
        wall_value, _ = _source_traces(self.mesh, self.conductivity, self.volumetric_source, True, self.source_gradient, self.radial_source_coefficient, self.source_quadratic)
        radial = np.zeros_like(radius)  # derivative with respect to log(r)
        angular = np.zeros_like(radius)
        for loop, sl, distance, sign in zip(self.mesh.loops, self.mesh.loop_slices,
                                            (log_r, span-log_r), (1, -1)):
            n = loop.panel_count
            coeff = np.fft.rfft(self.boundary_values[sl]-wall_value[sl]) / n
            temperature += coeff[0].real * distance/span
            radial += sign * coeff[0].real/span
            phase = angle - np.arctan2(loop.centers[0, 1], loop.centers[0, 0])
            for m in range(1, len(coeff)):
                # sinh(m*distance)/sinh(m*span) in bounded exponentials.
                decay = np.exp(-m*(span-distance)) / -np.expm1(-2*m*span)
                weight = decay * -np.expm1(-2*m*distance)
                slope = m * decay * (1 + np.exp(-2*m*distance))
                multiplicity = 1 if 2*m == n else 2
                wave = multiplicity * coeff[m] * np.exp(1j*m*phase)
                temperature += weight * wave.real
                radial += sign * slope * wave.real
                angular -= m * weight * wave.imag
        direction = xy / radius[:, None]
        tangent = np.column_stack((-direction[:, 1], direction[:, 0]))
        flux = -(self.conductivity/radius[:, None]) * (radial[:, None]*direction + angular[:, None]*tangent)
        return temperature, flux - self.conductivity*slope_xy


@dataclass(frozen=True)
class AnnulusContactResult:
    inner_trace: np.ndarray
    outer_trace: np.ndarray
    interface_flux: np.ndarray  # W/m², positive toward increasing radius
    heat_rates: tuple[float, float, float]  # bore, contact, exterior, W/m of axis
    condition_number: float  # coupled interface system
    inner_field: AnnulusLayerField | None = None
    outer_field: AnnulusLayerField | None = None
    outer_interface_flux: np.ndarray | None = None
    sheet_temperature: np.ndarray | None = None
    sheet_tangential_heat_rate: np.ndarray | None = None
    sheet_reservoir_flux: np.ndarray | None = None

    sheet_source_flux: np.ndarray | None = None

    def sheet_sector_heat_balance(self, start_angle, end_angle):
        """Return the steady heat budget of oriented sheet sectors, in W/m of axis.

        Angles in radians broadcast. For increasing angles, radial_in is heat
        from the inner material into the sheet and radial_out is heat from the
        sheet into the outer material. Tangential_out is the end current minus
        the start current. Reservoir_out is loss to the reservoir and generation
        is heat produced in the sheet. Residual is net outflow minus generation.
        """
        if self.inner_field is None or self.outer_field is None:
            raise ValueError("sheet sector balance requires both material fields")
        if self.inner_field.circular_geometry:
            inner_arc = self.inner_field.circular_arc_heat_rate
            outer_arc = self.outer_field.circular_arc_heat_rate
            current = self.circular_sheet_current
            reservoir = self.circular_reservoir_heat_rate
        else:
            inner_arc = self.inner_field.polygonal_arc_heat_rate
            outer_arc = self.outer_field.polygonal_arc_heat_rate
            current = self.polygonal_sheet_current
            reservoir = self.polygonal_reservoir_heat_rate
        radial_in = inner_arc("outer", start_angle, end_angle)
        radial_out = outer_arc("inner", start_angle, end_angle)
        tangential_out = current(end_angle)-current(start_angle)
        reservoir_out = reservoir(start_angle, end_angle)
        generation = self.sheet_generation_heat_rate(start_angle, end_angle)
        residual = radial_out-radial_in+tangential_out+reservoir_out-generation
        return dict(radial_in=radial_in, radial_out=radial_out, tangential_out=tangential_out,
                    reservoir_out=reservoir_out, generation=generation, residual=residual)

    def sheet_generation_heat_rate(self, start_angle, end_angle):
        """Signed sheet generation on an oriented angular sector (W/m of axis).

        Polygonal solutions integrate the applied constant source on each chord.
        Circular solutions integrate its periodic interpolant on the nominal circle.
        Angles in radians broadcast; a positive source adds heat to the sheet.
        """
        if self.inner_field is None or self.sheet_source_flux is None:
            raise ValueError("sheet generation requires stored geometry and interface source")
        loop = self.inner_field.mesh.loops[0]
        integrate = _integrate_circular_arc if self.inner_field.circular_geometry else _integrate_polygonal_arc
        return integrate(loop, self.sheet_source_flux, start_angle, end_angle)

    def polygonal_sheet_current(self, angles):
        """Tangential heat rate (W/m of axis) at ray cuts of the straight-panel sheet.

        Angles are in radians, positive in the direction of increasing azimuth.
        Interpolate the vertex currents linearly in distance along the chord,
        consistent with a constant net distributed load on each panel.
        """
        if self.inner_field is None or self.inner_field.circular_geometry:
            raise ValueError("polygonal sheet current requires the straight-panel formulation")
        _, panel, fraction = _polygonal_sheet_cut(self.inner_field.mesh.loops[0], angles)
        current = self.sheet_tangential_heat_rate
        if current is None:
            return np.zeros_like(fraction)
        return (1-fraction)*current[panel-1] + fraction*current[panel]

    def polygonal_reservoir_heat_rate(self, start_angle, end_angle):
        """Heat lost to the reservoir over an oriented polygonal sheet sector (W/m).

        Angles in radians broadcast. Reversed limits negate the exchange and
        extra full turns count repeatedly. Panel flux is constant per unit
        chord length, including on the two partially intersected panels.
        """
        if self.inner_field is None or self.inner_field.circular_geometry:
            raise ValueError("polygonal reservoir heat rates require the straight-panel formulation")
        loop = self.inner_field.mesh.loops[0]
        return _integrate_polygonal_arc(loop, self.sheet_reservoir_flux, start_angle, end_angle)

    def circular_sheet_current(self, angles):
        """Sheet heat rate toward increasing azimuth (W/m of axis) at angles in radians.

        Reconstruct the periodic trigonometric interpolant of the solved currents
        at panel ends. A sheet without tangential conduction carries zero current.
        """
        if self.inner_field is None or not self.inner_field.circular_geometry:
            raise ValueError("sheet current interpolation requires the circular formulation")
        angles = np.asarray(angles, dtype=float)
        if not np.isfinite(angles).all():
            raise ValueError("sheet current angles must be finite")
        samples = self.sheet_tangential_heat_rate
        if samples is None:
            return np.zeros_like(angles)
        n = len(samples)
        # Current sample zero lies at the downstream vertex of panel zero.
        vertex = self.inner_field.mesh.loops[0].vertices[1]
        phase = angles - np.arctan2(vertex[1], vertex[0])
        coeff = np.fft.rfft(samples) / n
        current = np.full_like(angles, coeff[0].real)
        for mode in range(1, len(coeff)):
            copies = 1 if 2*mode == n else 2
            current += copies * (coeff[mode]*np.exp(1j*mode*phase)).real
        return current

    def circular_reservoir_heat_rate(self, start_angle, end_angle):
        """Integrated heat from the sheet to its reservoir (W/m of axis).

        Angles follow the oriented arc convention of circular_arc_heat_rate.
        With no reservoir the exchanged heat is zero.
        """
        if self.inner_field is None or not self.inner_field.circular_geometry:
            raise ValueError("reservoir arc heat rates require the circular formulation")
        loop = self.inner_field.mesh.loops[0]
        samples = self.sheet_reservoir_flux
        if samples is None:
            samples = np.zeros(loop.panel_count)
        return _integrate_circular_arc(loop, samples, start_angle, end_angle)


def _polygonal_ring_matrices(mesh,reference):
    """Constant-panel collocation for two equally divided, concentric regular rings."""
    n = mesh.loops[0].panel_count
    # Simultaneously rotating source and target by one panel preserves both
    # kernels. The first column determines each block, including its diagonal.
    shifts = (np.arange(n)[:,None]-np.arange(n)) % n
    single = np.empty((2*n,2*n))
    double = np.empty_like(single)
    normals = mesh.normals
    for source,cols in zip(mesh.loops,mesh.loop_slices):
        start = source.vertices[0]
        length = source.lengths[0]
        tangent = source.vectors[0]/length
        normal = normals[cols.start]
        for target,rows in zip(mesh.loops,mesh.loop_slices):
            centers = target.centers
            single[rows,cols] = integrate_single_layer_panel_many(centers,start,tangent,length,reference)[shifts]
            double[rows,cols] = integrate_double_layer_panel_many(centers,start,tangent,normal,length)[shifts]
    return single,double


def _circular_normal_map(outer, inner, n):
    """Map temperature samples to outward normal derivatives on exact circles."""
    span = np.log(outer/inner)
    modes = np.arange(1, n//2 + 1)
    z = modes*span
    # Harmonic radial basis in log(r). Zero mode is linear in log(r).
    diagonal = np.r_[1/span, modes / np.tanh(z)]
    coupling = np.r_[1/span, 2*modes*np.exp(-z) / -np.expm1(-2*z)]
    shifts = (np.arange(n)[:, None] - np.arange(n)) % n
    own = np.fft.irfft(diagonal, n=n)[shifts]
    other = np.fft.irfft(coupling, n=n)[shifts]
    return np.block([[own/outer, -other/outer], [-other/inner, own/inner]])


def _wall_response(normal_map, interface, wall, ambient, film, inward_flux, offset):
    """Eliminate the wall temperature; return its constant and linear parts."""
    n = normal_map.shape[0]//2
    if inward_flux is not None:
        if ambient is not None or film is not None:
            raise ValueError("a prescribed flux requires None for wall temperature and film")
        h = np.zeros(n)
        load = np.broadcast_to(np.asarray(inward_flux, dtype=float), (n,))
        if not np.isfinite(load).all():
            raise ValueError("prescribed wall flux must be finite")
    else:
        ambient = np.broadcast_to(np.asarray(ambient, dtype=float), (n,))
        if not np.isfinite(ambient).all():
            raise ValueError("wall temperatures must be finite")
        if film is None:
            return ambient, None
        h = np.broadcast_to(np.asarray(film, dtype=float), (n,))
        if not np.isfinite(h).all() or np.any(h < 0):
            raise ValueError("film coefficients must be finite and nonnegative")
        load = h * ambient
    # k*dT/dn + h*Twall = h*Tambient, with the material-outward normal.
    loads = np.column_stack((load-offset[wall], -normal_map[wall, interface]))
    response = np.linalg.solve(normal_map[wall, wall] + np.diag(h), loads)
    return response[:, 0], response[:, 1:]


def solve_annulus_contact(radii, conductivities, wall_temperatures, resistance=0., panels=32, *,
                          formulation="polygonal", film_coefficients=(None, None), wall_fluxes=(None, None),
                          mean_bore_temperature=None, interface_source=0., inner_resistance_fraction=.5, volumetric_source=0.,
                          source_gradient=(0., 0.), radial_source_coefficient=0., source_quadratic=(0., 0., 0.),
                          sheet_conductivity=0., sheet_film=None):
    """Coupled two-material conduction on concentric annuli.

    Supply radii (bore, contact, exterior), two positive conductivities, and
    inner/outer Dirichlet data (scalars or panel vectors in azimuthal order).
    film_coefficients is a (bore, exterior) pair. None prescribes the wall
    temperature. A scalar or panel vector h in W/(m² K) instead uses that
    wall's temperature data as the ambient fluid temperature. Zero h insulates.
    A wall_fluxes entry prescribes inward W/m² (scalar or azimuthal samples);
    set the corresponding wall temperature and film to None. At least one
    wall must have a prescribed temperature or a nonzero film anchor unless
    both circular walls have prescribed fluxes. For that case supply
    mean_bore_temperature in K and compatible integrated wall inputs.
    The areal resistance is in m² K/W and may vary by interface panel.
    interface_source is sheet generation in W/m². inner_resistance_fraction
    places a fraction of the contact resistance between inner material and
    sheet. Both accept a scalar or a panel vector. interface_flux is the
    increasing-radius flux arriving from the inner material, and
    outer_interface_flux leaves into the outer material. heat_rates reports
    increasing-radius rates at the bore, inner contact trace and exterior.
    sheet_conductivity (W/K) is the circumferential conductivity times
    thickness of the interface, scalar or one value per angular cut (cut j
    follows sample j). On nominal circles with constant conductivity the local outward flux
    increase equals interface_source + sheet_conductivity*T_sheet,θθ/contact².
    Straight panels use a conservative difference of tangential heat currents
    with the panel length as the distance between sheet samples. For a conducting
    sheet, sheet_tangential_heat_rate[j] gives heat per unit axial length
    (W/m) toward increasing angle at the cut after sample j.
    sheet_film=(h,T_ambient) exchanges heat with an external reservoir. Both
    entries may be scalar or sampled at sheet panel centres. The returned
    sheet_reservoir_flux is positive from sheet to reservoir in W/m². A
    positive h anchors the absolute temperature for prescribed-flux walls.
    volumetric_source is a constant volumetric heating rate (W/m³), scalar
    or one per material layer; negative values describe heat removal.
    source_gradient adds gx*x + gy*y to the heating rate, using one Cartesian
    pair in W/m⁴ for both layers or a separate pair per layer.
    radial_source_coefficient adds a multiple of x²+y² to the heating rate,
    supplied in W/m⁵ as a scalar or one coefficient per layer.
    source_quadratic adds A*x² + B*x*y + C*y² in W/m³, with a shared
    (A, B, C) triple in W/m⁵ or one triple per material.
    The default polygonal formulation inscribes straight constant panels.
    polygonal_equal_area scales each regular polygon so its enclosed area
    equals that of the nominal circle. Boundary data are still ordered by
    azimuth; integrated rates use the actual chord lengths. The
    circular formulation collocates the contact law at the same angles and
    uses the Fourier Dirichlet-to-Neumann map of each nominal annulus.
    """
    if formulation not in ("polygonal", "polygonal_equal_area", "circular"):
        raise ValueError("formulation must be polygonal, polygonal_equal_area or circular")
    circular = formulation == "circular"
    bore, contact, exterior = radii
    ki, ko = conductivities
    if not np.isfinite([ki, ko]).all() or min(ki, ko) <= 0:
        raise ValueError("conductivities must be finite and positive")
    n = _validate_panel_count(panels)
    # Area of a regular polygon of circumradius R: n R² sin(2π/n)/2.
    scale = 1/np.sqrt(np.sinc(2/n)) if formulation == "polygonal_equal_area" else 1.
    inner = annulus_mesh(n,n,contact*scale,bore*scale)
    outer = annulus_mesh(n,n,exterior*scale,contact*scale)
    ks = np.broadcast_to(np.asarray(sheet_conductivity,dtype=float),(n,))
    if not np.isfinite(ks).all() or np.any(ks<0):
        raise ValueError("sheet_conductivity must be finite and nonnegative")
    conducting = np.any(ks>0)
    film,ambient = (0.,0.) if sheet_film is None else sheet_film
    film,ambient = (np.broadcast_to(np.asarray(v,dtype=float),(n,)) for v in (film,ambient))
    if not np.isfinite(film).all() or np.any(film<0) or not np.isfinite(ambient).all():
        raise ValueError("sheet film needs finite nonnegative coefficients and finite ambient temperatures")
    reservoir_anchored = np.any(film>0)
    coupled_sheet = conducting or reservoir_anchored
    hot, cold = wall_temperatures
    bore_flux, exterior_flux = wall_fluxes
    r = np.broadcast_to(np.asarray(resistance, dtype=float), (n,))
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("resistance must be finite and nonnegative")

    source = np.broadcast_to(np.asarray(interface_source,dtype=float),(n,))
    fraction = np.broadcast_to(np.asarray(inner_resistance_fraction,dtype=float),(n,))
    if not np.isfinite(source).all() or not np.isfinite(fraction).all() or np.any((fraction<0)|(fraction>1)):
        raise ValueError("interface source must be finite and inner resistance fraction in [0,1]")
    # q_outer=q_inner+source; the outer part of R carries the generated heat.
    jump_offset = (1-fraction)*r*source

    def normal_map(mesh):
        if circular:
            return _circular_normal_map(*(np.linalg.norm(loop.vertices[0]) for loop in mesh.loops), n)
        single, double = _polygonal_ring_matrices(mesh,2*exterior)
        return np.linalg.solve(single, double + .5 * np.eye(mesh.panel_count))

    # Each map takes [outer trace, inner trace] to domain-outward derivatives.
    di = ki * normal_map(inner)
    do = ko * normal_map(outer)
    generation = np.broadcast_to(np.asarray(volumetric_source,dtype=float),(2,))
    if not np.isfinite(generation).all():
        raise ValueError("volumetric sources must be finite")
    gradients = np.asarray(source_gradient, dtype=float)
    if gradients.shape not in ((2,), (2, 2)) or not np.isfinite(gradients).all():
        raise ValueError("source_gradient requires one finite Cartesian pair or one pair per layer")
    gradients = np.broadcast_to(gradients, (2, 2))
    radial = np.broadcast_to(np.asarray(radial_source_coefficient, dtype=float), (2,))
    if not np.isfinite(radial).all():
        raise ValueError("radial_source_coefficient must be finite")

    quadratic = np.asarray(source_quadratic, dtype=float)
    if quadratic.shape not in ((3,), (2, 3)) or not np.isfinite(quadratic).all():
        raise ValueError("source_quadratic requires one finite triple or one triple per layer")
    quadratic = np.broadcast_to(quadratic, (2, 3))

    def source_offset(mesh,k,q,matrix,g,beta,coefficients):
        p,dn = _source_traces(mesh,k,q,circular,g,beta,coefficients)
        return k*dn-matrix@p
    oi = source_offset(inner,ki,generation[0],di,gradients[0],radial[0],quadratic[0])
    oo = source_offset(outer,ko,generation[1],do,gradients[1],radial[1],quadratic[1])
    # Unknowns: outer material's contact trace and outward radial heat flux.
    # Inner trace = outer trace + R*q; Fourier flux reverses sign on the hole.
    first, second = slice(0, n), slice(n, 2*n)
    bore_film, exterior_film = film_coefficients
    if mean_bore_temperature is not None:
        if reservoir_anchored:
            raise ValueError("a positive sheet film fixes the temperature; omit mean_bore_temperature")
        if not circular or bore_flux is None or exterior_flux is None:
            raise ValueError("mean_bore_temperature requires two prescribed fluxes on circular walls")
        if np.ndim(mean_bore_temperature) != 0 or not np.isfinite(mean_bore_temperature):
            raise ValueError("mean_bore_temperature must be a finite scalar")
    elif not reservoir_anchored and all(q is not None or (h is not None and np.all(np.asarray(h) == 0))
           for q, h in zip(wall_fluxes, film_coefficients)):
        raise ValueError("at least one wall must anchor the temperature")
    bore_base, bore_response = _wall_response(di, first, second, hot, bore_film, bore_flux,oi)
    ext_base, ext_response = _wall_response(do, second, first, cold, exterior_film, exterior_flux,oo)
    inner_map = di[first, first]
    outer_map = do[second, second]
    if bore_response is not None:
        inner_map = inner_map + di[first, second] @ bore_response
    if ext_response is not None:
        outer_map = outer_map + do[second, first] @ ext_response
    eye = np.eye(n)
    rhs = -np.concatenate((di[first, second] @ bore_base, do[second, first] @ ext_base))
    rhs[:n] -= oi[first]
    rhs[n:] -= oo[second]
    if coupled_sheet:
        # Keep both radial fluxes as unknowns so the bulk normal maps
        # never multiply the surface second derivative.
        sheet_operator = np.diag(film)
        if conducting:
            if circular:
                modes = np.arange(n//2+1)
                # Differentiate the periodic interpolant at the staggered cuts.
                symbol = 1j*modes*np.exp(1j*modes*np.pi/n)/contact
                kernel = np.fft.irfft(symbol,n=n)
                gradient_at_cut = kernel[(np.arange(n)[:,None]-np.arange(n))%n]
            else:
                spacing = inner.loops[0].lengths[0]
                gradient_at_cut = (np.roll(eye,-1,axis=0)-eye)/spacing
            # The negative adjoint gives the conservative divergence on equally
            # spaced samples; the conductive operator is negative semidefinite.
            sheet_operator += gradient_at_cut.T@(ks[:,None]*gradient_at_cut)
        ri,ro = fraction*r,(1-fraction)*r
        zero = np.zeros_like(eye)
        system = np.block([[inner_map,inner_map*ri+eye,zero],
                           [outer_map,zero,-outer_map*ro-eye],
                           [sheet_operator,-eye,eye]])
        rhs = np.r_[rhs,source+film*ambient]
    else:
        system = np.block([[inner_map, inner_map * r + eye],
                           [outer_map, -eye]])
        rhs[:n] -= inner_map @ jump_offset
        rhs[n:] += source
    if mean_bore_temperature is not None:
        # Uniform angular panels have equal arc lengths on each nominal circle.
        inputs = [radius*np.broadcast_to(np.asarray(value, dtype=float), (n,))
                  for radius,value in ((bore,bore_flux),(exterior,exterior_flux))]
        inputs.append(contact*source)
        inputs.append(.5*np.diff(np.square(radii))*generation*n)
        mean_quadratic = radial + .5*(quadratic[:, 0] + quadratic[:, 2])
        inputs.append(.25*np.diff(np.asarray(radii, dtype=float)**4)*mean_quadratic*n)
        imbalance = sum(np.sum(q) for q in inputs)
        throughput = sum(np.abs(q).sum() for q in inputs)
        if abs(imbalance) > 1e-12*throughput:
            raise ValueError("prescribed fluxes have incompatible integrated wall inputs")
        # T_bore = bore_base + bore_response @ (outer_trace + R*flux).
        mean_response = bore_response.mean(axis=0)
        resistance_to_inner = ri if conducting else r
        offset = 0. if conducting else mean_response@jump_offset
        gauge_row = np.r_[mean_response,mean_response*resistance_to_inner]
        if conducting:
            gauge_row = np.r_[gauge_row,np.zeros(n)]
        system = np.vstack((system,gauge_row))
        rhs = np.r_[rhs, mean_bore_temperature-bore_base.mean()-offset]
    condition = float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition > _MAX_CONDITION:
        raise ValueError(f"ill-conditioned interface system ({condition:.3e})")
    if mean_bore_temperature is None:
        state = np.linalg.solve(system, rhs)
    else:
        state = np.linalg.lstsq(system, rhs, rcond=None)[0]
    tangential_rate = None
    if coupled_sheet:
        sheet_temperature,flux,outer_flux = np.split(state,3)
        inner_trace = sheet_temperature+ri*flux
        outer_trace = sheet_temperature-ro*outer_flux
        if conducting:
            tangential_rate = -ks*(gradient_at_cut@sheet_temperature)
    else:
        outer_trace, flux = np.split(state, 2)
        inner_trace = outer_trace + r * flux + jump_offset
        outer_flux = flux+source
        sheet_temperature = outer_trace+(1-fraction)*r*outer_flux
    bore_temp = bore_base if bore_response is None else bore_base + bore_response @ inner_trace
    ext_temp = ext_base if ext_response is None else ext_base + ext_response @ outer_trace
    ti = np.concatenate((inner_trace, bore_temp))
    to = np.concatenate((ext_temp, outer_trace))
    dni = (di @ ti + oi) / ki
    dno = (do @ to + oo) / ko
    rates = (float(ki * dni[n:] @ inner.loops[1].lengths), float(flux @ inner.loops[0].lengths),
             float(-ko * dno[:n] @ outer.loops[0].lengths))
    if circular:
        # Equal angular panels on the nominal circles use arc length.
        arc_to_chord = (np.pi/n) / np.sin(np.pi/n)
        rates = tuple(float(q * arc_to_chord) for q in rates)
    return AnnulusContactResult(inner_trace, outer_trace, flux, rates, condition,
                                AnnulusLayerField(inner, ti, dni, ki, 2 * exterior, circular,generation[0],tuple(gradients[0]),radial[0],tuple(quadratic[0])),
                                AnnulusLayerField(outer, to, dno, ko, 2 * exterior, circular,generation[1],tuple(gradients[1]),radial[1],tuple(quadratic[1])),
                                outer_flux,sheet_temperature,tangential_rate,
                                None if sheet_film is None else film*(sheet_temperature-ambient),
                                sheet_source_flux=source.copy())
