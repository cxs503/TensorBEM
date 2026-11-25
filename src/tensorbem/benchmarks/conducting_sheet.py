"""Third azimuthal conduction mode with two contact jumps and a conducting sheet."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from ..layered_annulus import solve_annulus_contact


def measure(panels,formulation="circular",prescribe_flux=False,variable=False,reservoir=False):
    bore,contact,exterior = .4,.7,1.1
    ki,ko,resistance,fraction,conductance = 1.5,2.3,.4,.3,.18
    order = 3
    theta = 2*np.pi*(np.arange(panels)+.5)/panels
    sheet, incoming, outgoing = .7+.2j, .4-.5j, -.2+.6j
    cuts = theta+np.pi/panels
    def conductivity(angle):
        return conductance*(1+.35*np.cos(2*angle)) if variable else conductance

    def sample(amplitude,angle):
        return (amplitude*np.exp(1j*order*angle)).real

    def material(radius,k,trace,flux,angle):
        z = order*np.log(radius/contact)
        potential = trace*np.cosh(z)-contact/(order*k)*flux*np.sinh(z)
        power = (contact*flux*np.cosh(z)-order*k*trace*np.sinh(z))/radius
        return 1.2+sample(potential,angle),sample(power,angle),sample(-1j*order*k*potential/radius,angle)

    # q_out-q_in = generation + (K*T_theta)_theta / radius².
    ks_prime = -.7*conductance*np.sin(2*theta) if variable else 0.
    source = (sample(outgoing-incoming,theta) + (conductivity(theta)*order**2*sample(sheet,theta)
              - ks_prime*sample(1j*order*sheet,theta))/contact**2)
    hot = sheet+fraction*resistance*incoming
    cold = sheet-(1-fraction)*resistance*outgoing
    tw,qw,_ = material(bore,ki,hot,incoming,theta)
    te,qe,_ = material(exterior,ko,cold,outgoing,theta)
    options = dict(wall_fluxes=(qw,-qe),mean_bore_temperature=1.2) if prescribe_flux else {}
    if reservoir:
        film = 1.1+.35*np.cos(theta)
        ambient = .8+.9*np.sin(2*theta)
        absorbed = film*(1.2+sample(sheet,theta)-ambient)
        source += absorbed
        options.pop("mean_bore_temperature",None)
        options["sheet_film"] = film,ambient
    result = solve_annulus_contact((bore,contact,exterior),(ki,ko),(None,None) if prescribe_flux else (tw,te),
              resistance,panels,formulation=formulation,sheet_conductivity=conductivity(cuts),
              interface_source=source,inner_resistance_fraction=fraction,**options)
    errors = {}
    for name,value,target in (("sheet_K",result.sheet_temperature,1.2+sample(sheet,theta)),
                  ("incoming_W_m2",result.interface_flux,sample(incoming,theta)),
                  ("outgoing_W_m2",result.outer_interface_flux,sample(outgoing,theta))):
        errors[name] = np.max(abs(value-target))
    if reservoir:
        errors["reservoir_flux_W_m2"] = np.max(abs(result.sheet_reservoir_flux-absorbed))
    expected_rate = conductivity(cuts)*sample(-1j*order*sheet/contact,cuts)
    errors["tangential_heat_rate_W_m"] = np.max(abs(result.sheet_tangential_heat_rate-expected_rate))
    theta = 2*np.pi*(np.arange(37)+.29)/37
    direction = np.column_stack((np.cos(theta),np.sin(theta)))
    tangent = np.column_stack((-np.sin(theta),np.cos(theta)))
    for field,k,r,trace,flux in ((result.inner_field,ki,.54,hot,incoming),(result.outer_field,ko,.87,cold,outgoing)):
        u,radial,angular = material(r,k,trace,flux,theta)
        value,vector = field.temperature(r*direction),field.heat_flux(r*direction)
        errors[f"field_{r}_K"] = np.max(abs(value-u))
        errors[f"flux_{r}_W_m2"] = np.max(abs(vector-radial[:,None]*direction-angular[:,None]*tangent))
    # Radial wall rates need not vanish discretely when the reservoir
    # exchanges heat. Test the sheet balance with its own panel measure.
    length = (2*np.pi*contact/panels if formulation=="circular" else 2*contact*np.sin(np.pi/panels))
    external = result.sheet_reservoir_flux if reservoir else 0.
    balance = length*np.sum(result.outer_interface_flux-result.interface_flux-source+external)
    return dict(errors=errors,condition_number=result.condition_number,sheet_balance_W_m=float(abs(balance)),
                net_heat_rate_W_m=max(abs(v) for v in result.heat_rates))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservoir",action="store_true",help="include azimuthally varying heat exchange with a reservoir")
    args = parser.parse_args()
    rows = []
    for formulation,flux,variable in (("circular",False,False),("circular",True,False),("polygonal",False,False),
                                      ("circular",True,True),("polygonal",False,True)):
        previous = None
        grids = (16,32,64,128,256) if formulation=="polygonal" else (16,32,64,128)
        for n in grids:
            row = measure(n,formulation,flux,variable,args.reservoir)
            errors = row["errors"]
            orders = {k:float(np.log2(previous[k]/v)) for k,v in errors.items()} if previous and formulation=="polygonal" else {}
            passes = max(errors.values())<2e-10 if formulation=="circular" else (n<128 or min(orders.values())>1.8)
            row.update(variable=variable,panels=n,formulation=formulation,prescribed_flux=flux,orders=orders,
                       passed=bool(passes and row["sheet_balance_W_m"]<1e-10 and
                                   (args.reservoir or row["net_heat_rate_W_m"]<1e-10)))
            rows.append(row)
            previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,reservoir=args.reservoir,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
