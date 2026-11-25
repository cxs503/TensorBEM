"""Angular resolution of near-contact Fourier heat flux in concentric materials."""
import json
import math
from dataclasses import replace

import numpy as np

from ..layered_annulus import solve_annulus_contact


def measure(n, m):
    a, b, c = .25, .6, 1.
    ki, ko, resistance = 4., 1., .2
    phase = math.pi / 5
    angles = 2*np.pi*(np.arange(n)+.5)/n
    samples = np.cos(m*angles + phase)
    # f_i(b)=1, f_i'(b)=m/b. Continuity of heat flux and the contact
    # law determine the coefficients of f_o=P*(r/b)^m+Q*(r/b)^(-m).
    outer_value = 1 + resistance*ki*m/b
    p, q = .5*(outer_value + ki/ko), .5*(outer_value - ki/ko)
    walls = ((a/b)**m * samples, (p*(c/b)**m + q*(b/c)**m) * samples)
    solution = solve_annulus_contact((a,b,c), (ki,ko), walls, resistance, panels=n)
    spectral = solve_annulus_contact((a,b,c), (ki,ko), walls, resistance, panels=n, formulation="circular")
    theta = 2*np.pi*(np.arange(16*n)+.125)/(16*n)
    direction = np.column_stack((np.cos(theta), np.sin(theta)))
    tangent = np.column_stack((-direction[:, 1], direction[:, 0]))
    distance = .125 * 2*b*np.sin(np.pi/n)
    errors = {}
    for name, field, radius, k, plus, minus, traces in (
        ("inner", solution.inner_field, b-distance, ki, 1., 0., (samples, walls[0])),
        ("outer", solution.outer_field, b+distance, ko, p, q, (walls[1], outer_value*samples)),
    ):
        grow, fall = plus*(radius/b)**m, minus*(b/radius)**m
        radial = (grow-fall)*np.cos(m*theta+phase)
        angular = -(grow+fall)*np.sin(m*theta+phase)
        exact = -k*m/radius * (radial[:, None]*direction + angular[:, None]*tangent)
        scale = np.linalg.norm(exact, axis=1).max()
        targets = radius * direction
        control = replace(field, boundary_values=np.concatenate(traces))
        nominal = spectral.inner_field if name == "inner" else spectral.outer_field
        errors[name] = {}
        for key, data in (("solved", field), ("exact_trace_control", control), ("circular_solve", nominal)):
            _, flux = data.circular_field(targets)
            errors[name][key] = float(np.linalg.norm(flux-exact, axis=1).max()/scale)
    return {"panels": n, "mode": m, "panels_per_wavelength": n/m, "flux_relative_linf": errors}


if __name__ == "__main__":
    print(json.dumps([measure(n, m) for m in (2, 4, 8) for n in (16, 32, 64, 128)], indent=2))
