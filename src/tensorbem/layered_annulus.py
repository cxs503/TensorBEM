"""Steady conduction in two concentric annuli with a thermal contact."""
from dataclasses import dataclass

import numpy as np

from .multiloop import MultiLoopMesh, _evaluate_traces, annulus_mesh
from .solver import assemble_matrices, _MAX_CONDITION


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
                          sheet_conductivity=0.):
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
    thickness of the interface. On nominal circles the local outward flux
    increase equals interface_source + sheet_conductivity*T_sheet,θθ/contact².
    Straight panels use a conservative difference of tangential heat currents
    with the panel length as the distance between sheet samples.
    volumetric_source is a constant volumetric heating rate (W/m³), scalar
    or one per material layer; negative values describe heat removal.
    source_gradient adds gx*x + gy*y to the heating rate, using one Cartesian
    pair in W/m⁴ for both layers or a separate pair per layer.
    radial_source_coefficient adds a multiple of x²+y² to the heating rate,
    supplied in W/m⁵ as a scalar or one coefficient per layer.
    source_quadratic adds A*x² + B*x*y + C*y² in W/m³, with a shared
    (A, B, C) triple in W/m⁵ or one triple per material.
    The default polygonal formulation uses straight constant panels. The
    circular formulation collocates the contact law at the same angles and
    uses the Fourier Dirichlet-to-Neumann map of each nominal annulus.
    """
    if formulation not in ("polygonal", "circular"):
        raise ValueError("formulation must be polygonal or circular")
    circular = formulation == "circular"
    if np.ndim(sheet_conductivity) or not np.isfinite(sheet_conductivity) or sheet_conductivity<0:
        raise ValueError("sheet_conductivity must be finite and nonnegative")
    bore, contact, exterior = radii
    ki, ko = conductivities
    if not np.isfinite([ki, ko]).all() or min(ki, ko) <= 0:
        raise ValueError("conductivities must be finite and positive")
    inner = annulus_mesh(panels, panels, contact, bore)
    outer = annulus_mesh(panels, panels, exterior, contact)
    n = inner.loops[0].panel_count
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
        single, double = assemble_matrices(mesh, reference_length=2 * exterior)
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
        if not circular or bore_flux is None or exterior_flux is None:
            raise ValueError("mean_bore_temperature requires two prescribed fluxes on circular walls")
        if np.ndim(mean_bore_temperature) != 0 or not np.isfinite(mean_bore_temperature):
            raise ValueError("mean_bore_temperature must be a finite scalar")
    elif all(q is not None or (h is not None and np.all(np.asarray(h) == 0))
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
    if sheet_conductivity:
        # Keep both radial fluxes as unknowns so the bulk normal maps
        # never multiply the surface second derivative.
        if circular:
            modes = np.arange(n//2+1)
            kernel = np.fft.irfft(-sheet_conductivity*(modes/contact)**2,n=n)
            surface_laplacian = kernel[(np.arange(n)[:,None]-np.arange(n))%n]
        else:
            spacing = inner.loops[0].lengths[0]
            surface_laplacian = (sheet_conductivity/spacing**2)*(
                    np.roll(eye,1,axis=1)+np.roll(eye,-1,axis=1)-2*eye)
        ri,ro = fraction*r,(1-fraction)*r
        zero = np.zeros_like(eye)
        system = np.block([[inner_map,inner_map*ri+eye,zero],
                           [outer_map,zero,-outer_map*ro-eye],
                           [-surface_laplacian,-eye,eye]])
        rhs = np.r_[rhs,source]
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
        resistance_to_inner = ri if sheet_conductivity else r
        offset = 0. if sheet_conductivity else mean_response@jump_offset
        gauge_row = np.r_[mean_response,mean_response*resistance_to_inner]
        if sheet_conductivity:
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
    if sheet_conductivity:
        sheet_temperature,flux,outer_flux = np.split(state,3)
        inner_trace = sheet_temperature+ri*flux
        outer_trace = sheet_temperature-ro*outer_flux
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
                                outer_flux,sheet_temperature)
