"""Generated interface heat in concentric annuli: cylindrical resistances and angular harmonics."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from ..layered_annulus import solve_annulus_contact


def radial(n, formulation):
    a,b,c = .3,.6,1.
    ki,ko = 2.,.8
    film_i,film_o = 3.,5.
    r,f = .35,.27
    generation = 4.
    # Two parallel escape paths from a sheet to isothermal ambient fluids.
    resistance_i = (1/(a*film_i)+math.log(b/a)/ki+f*r/b)/(2*math.pi)
    resistance_o = (1/(c*film_o)+math.log(c/b)/ko+(1-f)*r/b)/(2*math.pi)
    power = 2*math.pi*b*generation
    sheet = 2.+power/(1/resistance_i+1/resistance_o)
    qi,qo = (2.-sheet)/resistance_i,(sheet-2.)/resistance_o
    result = solve_annulus_contact((a,b,c),(ki,ko),(2.,2.),r,n,formulation=formulation,
                           film_coefficients=(film_i,film_o),interface_source=generation,inner_resistance_fraction=f)
    return dict(sheet_error_K=float(abs(result.sheet_temperature-sheet).max()),
                rate_error_W_m=float(abs(np.asarray(result.heat_rates)-[qi,qi,qo]).max()),
                condition=result.condition_number)


def angular(n,formulation,bulk=(0.,0.)):
    a,b,c = .3,.6,1.
    ki,ko = 2.,.8
    angle = 2*np.pi*(np.arange(n)+.5)/n
    def field(radius,theta,inner):
        # Independently specified Cauchy data at r=b extended by harmonic
        # functions. q(r=b) is positive in the increasing-radius direction.
        k,temp,amp,q0,q1,m,phase = ((ki,2.,.2,1.,.2,2,.3) if inner else (ko,1.6,.1,2.,1.4,3,-.2))
        z = np.log(radius/b)
        wave = np.cos(m*theta+phase)
        axial = amp*np.cosh(m*z)-(b*q1/(k*m))*np.sinh(m*z)
        value = temp-b*q0*z/k+axial*wave
        flux = (b*q0+(b*q1*np.cosh(m*z)-k*m*amp*np.sinh(m*z))*wave)/radius
        # A radial Poisson solution with zero value and slope at the contact.
        generation = bulk[0 if inner else 1]
        value += generation*(b*b*z/2-(radius*radius-b*b)/4)/k
        flux += generation*(radius-b*b/radius)/2
        return value,flux

    ti,qi = field(b,angle,True)
    to,qo = field(b,angle,False)
    fraction = .5+.25*np.sin(angle)
    r = (ti-to)/(fraction*qi+(1-fraction)*qo)
    tb,_ = field(a,angle,True)
    te,_ = field(c,angle,False)
    result = solve_annulus_contact((a,b,c),(ki,ko),(tb,te),r,n,formulation=formulation,
                       interface_source=qo-qi,inner_resistance_fraction=fraction,volumetric_source=bulk)
    sheet = ti-fraction*r*qi
    targets = angle+.07
    xy = np.column_stack((np.cos(targets),np.sin(targets)))
    interior,grad_errors = [],[]
    for rad,layer,is_inner in ((.42,result.inner_field,True),(.81,result.outer_field,False)):
        expected,radial_flux = field(rad,targets,is_inner)
        grad_errors.append(abs(np.sum(layer.heat_flux(rad*xy)*xy,axis=1)-radial_flux).max())
        interior.append(np.max(abs(layer.temperature(rad*xy)-expected)))
    return dict(sheet_error_K=float(abs(result.sheet_temperature-sheet).max()),
                inner_flux_error_W_m2=float(abs(result.interface_flux-qi).max()),
                outer_flux_error_W_m2=float(abs(result.outer_interface_flux-qo).max()),
                interior_error_K=float(max(interior)),interior_flux_error_W_m2=float(max(grad_errors)),
                condition=result.condition_number)


def angular_bulk(n,formulation):
    return angular(n,formulation,bulk=(6.,-2.))


if __name__=="__main__":
    rows = []
    for formulation in ("polygonal","circular"):
        for case in (radial,angular,angular_bulk):
            previous = None
            for n in (16,32,64,128):
                row = case(n,formulation)
                errors = {k:v for k,v in row.items() if "_error_" in k}
                rates = {} if previous is None or formulation=="circular" else {
                    k:math.log2(previous[k]/v) for k,v in errors.items()}
                passed = max(errors.values())<1e-10 if formulation=="circular" else all(v>1.5 for v in rates.values())
                row.update(case=case.__name__,panels=n,formulation=formulation,orders=rates,passed=passed)
                rows.append(row)
                previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
