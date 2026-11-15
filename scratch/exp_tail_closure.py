"""Is the roll C5 excess caused by truncating the damping spectrum at 12 rad/s?

Experiment: rebuild the memory kernel from the tabulated damping *plus* a
power-law tail beyond the band top (measured local exponent), which restores the
missing high-frequency memory.  If the decay then approaches the frequency-domain
estimate, the excess is a band-truncation effect, not a decay-model artefact.
"""
import math
import sys

import numpy as np

sys.path.insert(0, "src")
from tensorbem import time_domain as td
from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.causality import local_tail_exponent

L, B, T = case.LENGTH, case.BEAM, case.DRAFT
moments = case._uniform_body_second_moments()
vol = case.VOLUME_ANALYTIC
m = case.DENSITY * vol

sweep = case.load_sweep("benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz")


def tail_power(omega, damping, *, head=4):
    """Fit B ~ c * omega**(-p) on the last ``head`` samples."""
    w = omega[-head:]
    b = damping[-head:]
    keep = b > 0
    p, logc = np.polyfit(np.log(w[keep]), np.log(b[keep]), 1)
    return -float(p), float(math.exp(logc))


def run(dof, inertia, stiffness, *, tail=None, n_tail_ratio=1.0, duration=None):
    omega = sweep.omega
    damping = sweep.damping[:, dof, dof]
    added = sweep.added_mass[:, dof, dof]
    if tail is not None:
        p, c = tail
        w_tail = np.geomspace(omega[-1], omega[-1] * n_tail_ratio, 120)[1:]
        omega = np.concatenate([omega, w_tail])
        damping = np.concatenate([damping, c * w_tail ** (-p)])
        added = np.concatenate([added, np.full(w_tail.size, added[-1])])
    closure = case.kk_added_mass_closure(sweep.omega, sweep.added_mass[:, dof, dof], sweep.damping[:, dof, dof])
    a_inf = float(closure.added_mass_inf)
    wn = math.sqrt(stiffness / (inertia + a_inf))
    for _ in range(80):
        wn = math.sqrt(stiffness / (inertia + float(np.interp(wn, sweep.omega, sweep.added_mass[:, dof, dof]))))
    a_at = float(np.interp(wn, sweep.omega, sweep.added_mass[:, dof, dof]))
    b_at = float(np.interp(wn, sweep.omega, sweep.damping[:, dof, dof]))
    zeta_fd = b_at / (2.0 * (inertia + a_at) * wn)
    tau = np.arange(0.0, case.TAU_MAX + 0.5 * case.DT, case.DT)
    kernel = td.radiation_memory_kernel(omega, damping, tau)
    decay = td.cummins_free_decay(
        kernel, mass=inertia, restoring=stiffness, added_mass_inf=a_inf,
        initial_displacement=1.0, dt=case.DT, duration=duration or case.DURATION,
    )
    zeta, period, npeaks = case._decay_zeta(np.asarray(decay.time), np.asarray(decay.displacement))
    return dict(zeta=zeta, zeta_fd=zeta_fd, rel=abs(zeta - zeta_fd) / zeta_fd, period=period,
                peaks=npeaks, wn=wn, a_at=a_at, b_at=b_at, a_inf=a_inf,
                b_over_peak=float(sweep.damping[-1, dof, dof] / sweep.damping[:, dof, dof].max()))


for dof, inertia, name in ((2, m, "heave"), (3, moments["roll"], "roll"), (4, moments["pitch"], "pitch")):
    osc = case.oscillator_properties(dof)
    stiffness = osc["stiffness"]
    p, c = tail_power(sweep.omega, sweep.damping[:, dof, dof])
    print(f"\n=== {name}: tail exponent on last 4 nodes p={p:.2f}, Bmax_share_at_top={sweep.damping[-1, dof, dof]/sweep.damping[:, dof, dof].max():.2f}")
    for label, kw in (
        ("band only", {}),
        ("tail x2", {"tail": (p, c), "n_tail_ratio": 2.0}),
        ("tail x10", {"tail": (p, c), "n_tail_ratio": 10.0}),
        ("tail x50", {"tail": (p, c), "n_tail_ratio": 50.0}),
    ):
        r = run(dof, inertia, stiffness, **kw)
        print(f"{label:10s} zeta_decay={r['zeta']:.5f} zeta_fd={r['zeta_fd']:.5f} rel={r['rel']:7.3%} "
              f"period={r['period']:.4f} peaks={r['peaks']} A(wn)={r['a_at']:.4f} A_inf={r['a_inf']:.4f}")