"""Run python -m tensorbem.benchmarks.layered_annulus for radial validation."""
import json
import math

import numpy as np

from ..layered_annulus import solve_annulus_contact


def measure(n, resistance=.2, reverse=False):
    a, b, c = .25, .6, 1.
    ki, ko = 4., 1.
    hot, cold = (290., 310.) if reverse else (310., 290.)
    rate = 2 * math.pi * (hot-cold) / (math.log(b/a)/ki + resistance/b + math.log(c/b)/ko)
    trace_i = hot - rate * math.log(b/a) / (2*math.pi*ki)
    trace_o = cold + rate * math.log(c/b) / (2*math.pi*ko)
    result = solve_annulus_contact((a,b,c), (ki,ko), (hot,cold), resistance, panels=n)
    angles = 2 * math.pi * (np.arange(4*n) + .25)/(4*n)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    ring_rates = []
    for radius, field in ((.42, result.inner_field), (.79, result.outer_field)):
        local_flux = field.heat_flux(radius * directions)
        ring_rates.append(float(2*math.pi*radius*np.sum(local_flux*directions, axis=1).mean()))
    return {"panels_per_circle": n, "resistance": resistance, "reversed": reverse,
            "interior_ring_rates_W_m": ring_rates, "exact_rate_W_m": rate, "rates_W_m": result.heat_rates,
            "max_rate_relative_error": max(abs(q/rate - 1) for q in result.heat_rates),
            "max_trace_error_K": float(max(np.max(abs(result.inner_trace-trace_i)),
                                           np.max(abs(result.outer_trace-trace_o)))),
            "condition_number": result.condition_number}


def _angular_solution(n):
    """First azimuthal harmonic with sign-changing local heat transfer."""
    a, b, c = .25, .6, 1.
    ki, ko, resistance = 4., 1., .2
    cosine = np.cos(2 * math.pi * (np.arange(n) + .5) / n)
    # Inner radial factor f=r. Choose outer f=C*r+D/r to enforce both
    # k_i f_i'=k_o f_o' and f_i-f_o=-R*k_i*f_i' at the contact.
    value = b + resistance * ki
    linear = .5 * (value/b + ki/ko)
    inverse = b * (value - linear*b)
    walls = (a*cosine, (linear*c + inverse/c)*cosine)
    result = solve_annulus_contact((a,b,c), (ki,ko), walls, resistance, panels=n)
    return result, b, ki, ko, linear, inverse, cosine, value


def angular_mode(n):
    result, b, ki, ko, linear, inverse, cosine, value = _angular_solution(n)
    # Probe both layers over full circles. The Cartesian analytic field is
    # T=x in the inner material and T=C*x+D*x/(x²+y²) in the outer one.
    theta = 2 * math.pi * (np.arange(73) + .37)/73
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    field_errors = {}
    for name, field, radius, k, C, D in (
        ("inner", result.inner_field, .42, ki, 1., 0.),
        ("outer", result.outer_field, .79, ko, linear, inverse),
    ):
        points = radius * directions
        x, y = points.T
        exact = C*x + D*x/radius**2
        grad = np.column_stack((C + D*(y*y-x*x)/radius**4, -2*D*x*y/radius**4))
        flux_error = field.heat_flux(points) + k*grad
        field_errors[name] = {
            "temperature_max_error_K": float(np.max(np.abs(field.temperature(points)-exact))),
            "flux_vector_max_error_W_m2": float(np.linalg.norm(flux_error, axis=1).max()),
        }
    return {"panels_per_circle": n, "interior_fields": field_errors,
            "flux_max_error_over_amplitude": float(np.max(abs(result.interface_flux + ki*cosine))/ki),
            "max_trace_error_K": float(max(np.max(abs(result.inner_trace - b*cosine)),
                                           np.max(abs(result.outer_trace - value*cosine)))),
            "max_abs_integrated_rate_W_m": max(abs(q) for q in result.heat_rates)}


def varying_contact(n):
    """Uniform radial flux at the interface with a cosine-varying jump."""
    a, b, c = .25, .6, 1.
    ki, ko = 4., 1.
    cosine = np.cos(2 * math.pi * (np.arange(n) + .5) / n)
    # (r/b+b/r)*cos(theta) is harmonic and has zero radial slope at r=b.
    # Adding it only to the exterior layer leaves interface flux at 1 W/m².
    walls = (2 - b/ki * math.log(a/b),
             1 - b/ko * math.log(c/b) + .1 * (c/b+b/c) * cosine)
    resistance = 1 - .2 * cosine
    result = solve_annulus_contact((a,b,c), (ki,ko), walls, resistance, panels=n)
    return {"panels_per_circle": n, "max_flux_error_W_m2": float(np.max(abs(result.interface_flux-1))),
            "max_inner_trace_error_K": float(np.max(abs(result.inner_trace-2))),
            "max_outer_trace_error_K": float(np.max(abs(result.outer_trace-(1+.2*cosine))))}


def near_contact(n):
    """Compare on rings at prescribed radial distances from the true circle."""
    result, b, ki, ko, linear, inverse, _, _ = _angular_solution(n)
    h = 2 * b * math.sin(math.pi/n)
    theta = 2 * math.pi * np.arange(8*n) / (8*n)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    rings = []
    for name, field, sign, k, C, D in (
        ("inner", result.inner_field, -1, ki, 1., 0.),
        ("outer", result.outer_field, 1, ko, linear, inverse),
    ):
        for fraction in (.125, .25, .5, 1.):
            radius = b + sign * fraction * h
            points = radius * directions
            x, y = points.T
            exact = -k * np.column_stack((C + D*(y*y-x*x)/radius**4, -2*D*x*y/radius**4))
            scale = np.linalg.norm(exact, axis=1).max()
            row = {"layer": name, "clearance_m": fraction*h, "clearance_over_panel_length": fraction}
            direct = field.heat_flux(points)
            row["panel_relative_linf"] = float(np.linalg.norm(direct-exact, axis=1).max()/scale)
            temperature, reconstructed = field.circular_field(points)
            row["circular_relative_linf"] = float(np.linalg.norm(reconstructed-exact, axis=1).max()/scale)
            row["circular_temperature_max_error_K"] = float(np.abs(temperature-(C*x+D*x/radius**2)).max())
            rings.append(row)
    return {"panels_per_circle": n, "rings": rings}


if __name__ == "__main__":
    print(json.dumps({"radial": [measure(n,r,rev) for r in (0., .2) for rev in (False, True)
                      for n in (16, 32, 64, 128)],
                      "near_contact": [near_contact(n) for n in (16, 32, 64, 128)],
                      "angular": [angular_mode(n) for n in (16, 32, 64, 128)],
                      "varying_contact": [varying_contact(n) for n in (16, 32, 64, 128)]}, indent=2))
