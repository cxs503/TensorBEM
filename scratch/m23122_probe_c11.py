"""Probe: can the state-space route (case gate C11) be scored under the case's own conventions?"""
import json
import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem import state_space as ssm
from tensorbem.causality import kk_added_mass_closure

s = case.load_sweep("benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz")
out = {}
for dof in (2, 3, 4):
    w = s.omega
    B = s.damping[:, dof, dof]
    A = s.added_mass[:, dof, dof]
    ainf = float(kk_added_mass_closure(w, A, B).added_mass_inf)
    fit = ssm.adaptive_order_fit(w, B, target_error=0.03, max_poles=12, min_poles=2)
    real = ssm.state_space_from_fit(fit)
    osc = case.oscillator_properties(dof, label=s.label)
    ss_res = ssm.cummins_state_space(
        real, osc["inertia"], ainf, osc["stiffness"],
        initial_displacement=1.0, initial_velocity=0.0,
        dt=case.DT, duration=case.DURATION,
    )
    z_ss, p_ss, n_ss = case._decay_zeta(np.asarray(ss_res.time), np.asarray(ss_res.displacement))
    z_cv, p_cv, n_cv, _decay = case._decay_zeta_for(
        s, dof, damping=B, omega=w, added_mass_inf=ainf
    )
    out[case.DOF_NAMES[dof]] = dict(
        ainf=ainf, poles=int(len(fit.poles)), fit_err=float(fit.relative_error_max),
        z_ss=z_ss, z_conv=z_cv, dz=abs(z_ss - z_cv) / z_cv,
        p_ss=p_ss, p_conv=p_cv, dp=abs(p_ss - p_cv) / p_cv,
        n_ss=n_ss, n_conv=n_cv, substeps=ss_res.substeps,
        min_eig=float(np.min(real.eigenvalues().real)),
    )
print(json.dumps(out, indent=1))