"""Probe 2: how many poles does the roll damping spectrum need before the SS route agrees?"""
import json
import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem import state_space as ssm
from tensorbem.causality import kk_added_mass_closure

s = case.load_sweep("benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz")
dof = 3
w = s.omega
B = s.damping[:, dof, dof]
A = s.added_mass[:, dof, dof]
ainf = float(kk_added_mass_closure(w, A, B).added_mass_inf)
osc = case.oscillator_properties(dof, label=s.label)
z_cv, p_cv, n_cv, _ = case._decay_zeta_for(s, dof, damping=B, omega=w, added_mass_inf=ainf)
rows = []
for max_poles in (6, 12, 16, 20, 24, 30):
    fit = ssm.adaptive_order_fit(w, B, target_error=0.01, max_poles=max_poles, min_poles=2)
    real = ssm.state_space_from_fit(fit)
    ss_res = ssm.cummins_state_space(
        real, osc["inertia"], ainf, osc["stiffness"],
        initial_displacement=1.0, initial_velocity=0.0, dt=case.DT, duration=case.DURATION,
    )
    z_ss, p_ss, n_ss = case._decay_zeta(np.asarray(ss_res.time), np.asarray(ss_res.displacement))
    rows.append(dict(max_poles=max_poles, used=int(len(fit.poles)),
                     fit_err=float(fit.relative_error_max),
                     z_ss=z_ss, dz=abs(z_ss - z_cv) / z_cv,
                     p_ss=p_ss, dp=abs(p_ss - p_cv) / p_cv, substeps=ss_res.substeps))
print(json.dumps(dict(z_conv=z_cv, p_conv=p_cv, rows=rows), indent=1))