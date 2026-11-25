"""Steady Robin and thermal contact benchmark with independent cylindrical references."""
import json
import math

import numpy as np

from ..layered_annulus import solve_annulus_contact


def radial(n, formulation, hot=310., cold=290.):
    a, b, c = .25, .6, 1.
    ki, ko, resistance = 4., 1., .2
    hi, ho = 12., 3.
    rate = 2*math.pi*(hot-cold) / (
        1/(a*hi) + math.log(b/a)/ki + resistance/b + math.log(c/b)/ko + 1/(c*ho))
    result = solve_annulus_contact((a,b,c), (ki,ko), (hot,cold), resistance, panels=n,
                                   formulation=formulation, film_coefficients=(hi,ho))
    bore = hot - rate/(2*math.pi*a*hi)
    exterior = cold + rate/(2*math.pi*c*ho)
    return {"formulation": formulation, "panels": n, "exact_W_m": rate,
            "max_rate_relative_error": max(abs(q/rate-1) for q in result.heat_rates),
            "max_wall_temperature_error_K": float(max(
                np.max(abs(result.inner_field.boundary_values[n:] - bore)),
                np.max(abs(result.outer_field.boundary_values[:n] - exterior))))}


def angular(n, formulation):
    a, b, c = .25, .6, 1.
    ki, ko, resistance = 4., 1., .2
    theta = 2*math.pi*(np.arange(n)+.5)/n
    wave = np.sin(theta)
    # Inner T=r*sin(theta), outer T=(P*r+D/r)*sin(theta).
    value = b + resistance*ki
    p, d = .5*(value/b+ki/ko), .5*b*(value-b*ki/ko)
    hi, ho = 3+.5*np.cos(theta), 4+.7*np.sin(theta)
    bore_wall, ext_wall = a*wave, (p*c+d/c)*wave
    ambient = (bore_wall - ki*wave/hi, ext_wall + ko*(p-d/c**2)*wave/ho)
    result = solve_annulus_contact((a,b,c), (ki,ko), ambient, resistance, panels=n,
                                   formulation=formulation, film_coefficients=(hi,ho))
    return {"formulation": formulation, "panels": n,
            "interface_flux_max_error_W_m2": float(np.max(abs(result.interface_flux+ki*wave))),
            "max_wall_temperature_error_K": float(max(
                np.max(abs(result.inner_field.boundary_values[n:] - bore_wall)),
                np.max(abs(result.outer_field.boundary_values[:n] - ext_wall))))}


if __name__ == "__main__":
    print(json.dumps({fn.__name__: [fn(n, f) for f in ("polygonal", "circular")
                                    for n in (16, 32, 64, 128)] for fn in (radial, angular)}, indent=2))
