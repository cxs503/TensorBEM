"""Verify annular Neumann/Dirichlet contact with a log term and a second angular mode."""
import argparse
import json

import numpy as np

from ..layered_annulus import solve_annulus_contact


def measure(n, formulation, driven):
    a, b, c = .25, .6, 1.
    ki, ko, resistance, amplitude = 2., 1.3, .15, .2
    theta = 2*np.pi*(np.arange(n)+.5)/n
    # At r=b, continuity of radial power fixes the outer slope; the
    # resistive law fixes the outer trace. The two radial powers are r²,r⁻².
    v = amplitude*(1+2*resistance*ki/b)
    p, d = .5*(v+amplitude*ki/ko), .5*(v-amplitude*ki/ko)

    def field(radius, angle, inner):
        ratio = radius/b
        if inner:
            base, base_slope = np.log(ratio), 1.
            mode, slope = amplitude*ratio**2, 2*amplitude*ratio**2
        else:
            base_slope = ki/ko
            base = resistance*ki/b + base_slope*np.log(ratio)
            mode, slope = p*ratio**2+d/ratio**2, 2*(p*ratio**2-d/ratio**2)
        phase = 2*angle+.3
        temp = base + mode*np.cos(phase)
        grad_r = (base_slope + slope*np.cos(phase))/radius
        grad_theta = -2*mode*np.sin(phase)/radius
        return temp, grad_r, grad_theta

    bore, slope_bore, _ = field(a, theta, True)
    ext, slope_ext, _ = field(c, theta, False)
    wall_data = [bore, ext]
    inward = [-ki*slope_bore, ko*slope_ext]
    fluxes = [None, None]
    mean_bore = None
    if driven == 2:
        wall_data, fluxes = (None,None), inward
        mean_bore = np.log(a/b)
    else:
        wall_data[driven], fluxes[driven] = None, inward[driven]
    sol = solve_annulus_contact((a,b,c), (ki,ko), wall_data, resistance, n,
                                formulation=formulation, wall_fluxes=fluxes, mean_bore_temperature=mean_bore)
    targets = np.linspace(.13, 2*np.pi+.13, 31, endpoint=False)
    temp_err, vector_err = [], []
    unit = np.column_stack((np.cos(targets), np.sin(targets)))
    tangent = np.column_stack((-np.sin(targets), np.cos(targets)))
    for rad, region, k, inner in ((.43,sol.inner_field,ki,True), (.78,sol.outer_field,ko,False)):
        expected, gr, gt = field(rad, targets, inner)
        vector = -k*(gr[:,None]*unit + gt[:,None]*tangent)
        temp_err.append(np.max(abs(region.temperature(rad*unit)-expected)))
        vector_err.append(np.max(np.linalg.norm(region.heat_flux(rad*unit)-vector, axis=1)))
    _, slope, _ = field(b, theta, True)
    measured = (ki*sol.inner_field.normal_derivative[n:], ko*sol.outer_field.normal_derivative[:n])
    return dict(panels=n, formulation=formulation, flux_wall=("bore","exterior","both")[driven],
                temperature_error_K=float(max(temp_err)), interior_flux_error_W_m2=float(max(vector_err)),
                wall_flux_error_W_m2=float(max(np.max(abs(m-q)) for m,q in zip(measured,inward))),
                contact_flux_error_W_m2=float(np.max(abs(sol.interface_flux+ki*slope))),
                rate_error_W_m=float(np.max(abs(np.asarray(sol.heat_rates)+2*np.pi*ki))),
                condition_number=sol.condition_number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equal-area", action="store_true", help="use equal-area polygonal rings")
    args = parser.parse_args()
    polygon = "polygonal_equal_area" if args.equal_area else "polygonal"
    rows = []
    for form in (polygon,"circular"):
        for driven in ((0,1,2) if form == "circular" else (0,1)):
            previous = None
            for n in (16,32,64,128):
                row = measure(n, form, driven)
                errors = np.array([row[k] for k in ("temperature_error_K", "interior_flux_error_W_m2",
                                  "wall_flux_error_W_m2", "contact_flux_error_W_m2", "rate_error_W_m")])
                passed = np.isfinite(errors).all()
                if form == "circular":
                    passed &= max(errors) < 1e-10
                elif previous is not None:
                    passed &= (errors < previous/3).all()
                row["passed"] = bool(passed)
                rows.append(row)
                previous = errors
    print(json.dumps(rows, indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
