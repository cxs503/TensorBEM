"""Havelock (1955) frequency-domain verification for submerged sphere radiation.

P190-N03: BEM潜球附加质量Havelock频率域解析解验证面

Uses icosphere geodesic meshes (subdivisions 1,2,3 = 80,320,1280 panels)
bracketing the target 200/400/800 range, with superior panel quality.
"""
from __future__ import annotations
import argparse, json, math, platform
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import assemble_free_surface_operators_3d, solve_radiation_3d
from tensorbem.geometry3d import submerged_sphere_mesh


def _observed_order(h_values, errors):
    if len(h_values)<3 or len(errors)<3:
        return None
    h0,h1 = h_values[-2], h_values[-1]
    e0,e1 = errors[-2], errors[-1]
    if e0<=0 or e1<=0 or h0<=h1:
        return None
    return math.log(e0/e1)/math.log(h0/h1)


def run_convergence_study(ka=0.8, radius=1.0, submergence=2.0, density=1025.0, gravity=9.80665,
                          subdivisions=(1, 2, 3)):
    """Panel convergence over ``subdivisions`` (default 1,2,3 -> 80,320,1280 panels)."""
    subdivisions = [int(s) for s in subdivisions]
    ref = submerged_sphere_wave_multipole(ka, radius=radius, submergence=submergence, density=density, gravity=gravity)
    vol = ref.displaced_volume
    omega = ref.omega
    nu = ka / radius
    center = (0.0, 0.0, -submergence)
    levels = []
    h_list = []
    err_a33 = []
    for sub in subdivisions:
        mesh = submerged_sphere_mesh(sub, radius=radius, submergence=submergence)
        ops = assemble_free_surface_operators_3d(mesh, nu)
        rad = solve_radiation_3d(mesh, nu, center=center, density=density, gravity=gravity, operators=ops)
        a33 = float(rad.added_mass[2,2]/(density*vol))
        b33 = float(rad.damping[2,2]/(density*omega*vol))
        a11 = float(rad.added_mass[0,0]/(density*vol))
        b11 = float(rad.damping[0,0]/(density*omega*vol))
        e_a33 = abs(a33 - ref.heave_added_mass_dimless)/ref.heave_added_mass_dimless
        e_b33 = abs(b33 - ref.heave_damping_dimless)/max(ref.heave_damping_dimless, 1e-30)
        e_a11 = abs(a11 - ref.surge_added_mass_dimless)/ref.surge_added_mass_dimless
        e_b11 = abs(b11 - ref.surge_damping_dimless)/max(ref.surge_damping_dimless, 1e-30)
        h_list.append(mesh.h_mean)
        err_a33.append(e_a33)
        levels.append({"subdivision":sub,"n_panels":mesh.panel_count,
            "h_mean":mesh.h_mean,"a33_bem":a33,"a33_exact":ref.heave_added_mass_dimless,"a33_rel_error":e_a33,
            "b33_bem":b33,"b33_exact":ref.heave_damping_dimless,"b33_rel_error":e_b33,
            "a11_bem":a11,"a11_exact":ref.surge_added_mass_dimless,"a11_rel_error":e_a11,
            "b11_bem":b11,"b11_exact":ref.surge_damping_dimless,"b11_rel_error":e_b11,
            "condition_number":rad.condition_number,"relative_residual":rad.relative_residual})
    return {"ka":ka,"submergence_ratio":submergence/radius,
        "exact_reference":{"a33":ref.heave_added_mass_dimless,"b33":ref.heave_damping_dimless,
            "a11":ref.surge_added_mass_dimless,"b11":ref.surge_damping_dimless},
        "levels":levels,"observed_order_a33":_observed_order(h_list, err_a33)}


def run_frequency_sweep(ka_values=(0.1,0.5,1.0,2.0,3.0,4.0,5.0), radius=1.0, submergence=2.0,
                        subdivision=2, density=1025.0, gravity=9.80665):
    """Frequency sweep ka=0.1~5.0 with subdivision=2 (320 panels)."""
    mesh = submerged_sphere_mesh(subdivision, radius=radius, submergence=submergence)
    center = (0.0, 0.0, -submergence)
    rows = []
    for ka in ka_values:
        nu = float(ka)/radius
        ref = submerged_sphere_wave_multipole(ka, radius=radius, submergence=submergence, density=density, gravity=gravity)
        ops = assemble_free_surface_operators_3d(mesh, nu)
        rad = solve_radiation_3d(mesh, nu, center=center, density=density, gravity=gravity, operators=ops)
        vol = ref.displaced_volume
        omega = ref.omega
        a33 = float(rad.added_mass[2,2]/(density*vol))
        b33 = float(rad.damping[2,2]/(density*omega*vol))
        a11 = float(rad.added_mass[0,0]/(density*vol))
        b11 = float(rad.damping[0,0]/(density*omega*vol))
        e_a33 = abs(a33-ref.heave_added_mass_dimless)/ref.heave_added_mass_dimless
        e_b33 = abs(b33-ref.heave_damping_dimless)/max(ref.heave_damping_dimless,1e-30)
        e_a11 = abs(a11-ref.surge_added_mass_dimless)/ref.surge_added_mass_dimless
        e_b11 = abs(b11-ref.surge_damping_dimless)/max(ref.surge_damping_dimless,1e-30)
        rows.append({"ka":float(ka),"n_panels":mesh.panel_count,
            "a33_bem":a33,"a33_exact":ref.heave_added_mass_dimless,"a33_rel_error":e_a33,
            "b33_bem":b33,"b33_exact":ref.heave_damping_dimless,"b33_rel_error":e_b33,
            "a11_bem":a11,"a11_exact":ref.surge_added_mass_dimless,"a11_rel_error":e_a11,
            "b11_bem":b11,"b11_exact":ref.surge_damping_dimless,"b11_rel_error":e_b11,
            "condition_number":rad.condition_number})
    return rows


def run_low_frequency_limit(ka_values=(0.01,0.02,0.05,0.1,0.2), radius=1.0, submergence=2.0,
                            subdivision=2, density=1025.0, gravity=9.80665):
    """Low-frequency limit: ka->0 extrapolation of A_33."""
    mesh = submerged_sphere_mesh(subdivision, radius=radius, submergence=submergence)
    center = (0.0, 0.0, -submergence)
    rows = []
    for ka in ka_values:
        nu = float(ka)/radius
        ref = submerged_sphere_wave_multipole(ka, radius=radius, submergence=submergence, density=density, gravity=gravity)
        ops = assemble_free_surface_operators_3d(mesh, nu)
        rad = solve_radiation_3d(mesh, nu, center=center, density=density, gravity=gravity, operators=ops)
        vol = ref.displaced_volume
        a33 = float(rad.added_mass[2,2]/(density*vol))
        rows.append({"ka":float(ka),"a33_bem":a33,"a33_exact":ref.heave_added_mass_dimless,
            "a33_rel_error":abs(a33-ref.heave_added_mass_dimless)/ref.heave_added_mass_dimless})
    ka_arr = np.array([r["ka"] for r in rows])
    a33_arr = np.array([r["a33_bem"] for r in rows])
    coeffs = np.polyfit(ka_arr**2, a33_arr, 1)
    a33_limit_bem = float(coeffs[1])
    ref_ka0 = submerged_sphere_wave_multipole(0.001, radius=radius, submergence=submergence, density=density, gravity=gravity)
    a33_limit_exact = ref_ka0.heave_added_mass_dimless
    limit_error = abs(a33_limit_bem - a33_limit_exact)/a33_limit_exact
    return {"bem_extrapolated_limit":a33_limit_bem,"exact_limit":a33_limit_exact,
        "limit_relative_error":limit_error,"data_points":rows}


def run(*, quick=False, radius=1.0, submergence=2.0, density=1025.0, gravity=9.80665):
    ka_sweep = (0.1,0.5,1.0,2.0,3.0,4.0,5.0) if not quick else (0.5,1.0,2.0)
    ka_low = (0.01,0.02,0.05,0.1,0.2) if not quick else (0.05,0.1)
    return {"benchmark":"submerged_sphere_havelock_radiation",
        "geometry":{"radius":radius,"submergence":submergence,"submergence_ratio":submergence/radius,
            "density":density,"gravity":gravity},
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},
        "convergence_study":run_convergence_study(ka=0.8,radius=radius,submergence=submergence,density=density,gravity=gravity),
        "frequency_sweep":run_frequency_sweep(ka_values=ka_sweep,radius=radius,submergence=submergence,density=density,gravity=gravity),
        "low_frequency_limit":run_low_frequency_limit(ka_values=ka_low,radius=radius,submergence=submergence,density=density,gravity=gravity)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--submergence", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(quick=args.quick, radius=args.radius, submergence=args.submergence)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
