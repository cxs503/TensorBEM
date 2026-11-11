"""m14767 study driver: state-space fit / realization evidence, zero hand-copied numbers.

Writes docs/m14767/evidence.json, docs/m14767/numbers.md and two PNGs.
Run: PYTHONPATH=src python scripts/m14767_state_space_study.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorbem.state_space import (
    vector_fit_damping, state_space_from_fit, cummins_state_space,
    adaptive_order_fit, validate_state_space, radiation_response,
    _harmonic_memory_response,
)
from tensorbem.time_domain import radiation_memory_kernel, cummins_free_decay, decay_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "m14767"
OUT.mkdir(parents=True, exist_ok=True)
rec = json.loads((ROOT / "tests" / "data" / "time_domain_hemisphere_sweep.json").read_text())
omega = np.asarray(rec["omega"], float)
damping = np.asarray(rec["damping"], float)
added = np.asarray(rec["added_mass"], float)
body = rec["body"]

# ---------------------------------------------------------------- 1. fit ladder
ladder = {}
for n in (1, 3, 5, 8, 10, 12, 15):
    fit = vector_fit_damping(omega, damping, n_poles=n)
    ss = state_space_from_fit(fit)
    ladder[n] = dict(
        peak_error=fit.relative_error_max,
        pointwise_error=fit.relative_error_max_pointwise,
        constant=fit.constant,
        alpha_min=float(np.min(fit.poles)), alpha_max=float(np.max(fit.poles)),
        b_at_lowest_omega_data=float(damping[0]),
        b_at_lowest_omega_fit=float(fit.B_reconstructed[0]),
        stable=ss.is_stable(), n_iter=fit.n_iter, converged=fit.converged,
    )
fit10 = vector_fit_damping(omega, damping, n_poles=10)
ss10 = state_space_from_fit(fit10)
fit15 = vector_fit_damping(omega, damping, n_poles=15)
ss15 = state_space_from_fit(fit15)

# ------------------------------------------------- 2. realization dual-path (C4)
c4 = {}
for om in (0.3, 1.0, 3.0, 10.0, 25.0):
    ana = radiation_response(ss15, om)
    num = _harmonic_memory_response(ss15, om)
    c4[f"{om:g}"] = dict(analytic_abs=abs(ana), numeric_abs=abs(num),
                         amp_rel_error=abs(abs(num) - abs(ana)) / abs(ana),
                         phase_error=abs(float(np.angle(num / ana))),
                         b_fit=float(np.interp(om, omega, damping)) if om <= omega[-1] else None)
# feedthrough audit.  A fit of the Hullme curve returns d ~ 0, so the silent
# drop of the constant term is invisible there; use a synthetic curve with a
# known positive asymptote (delta-correlated high-frequency damping, d = 50).
om_syn = np.linspace(0.1, 6.0, 160)
d_true, coefs = 50.0, ((900.0, 0.6), (450.0, 1.8), (220.0, 3.4))
B_syn = d_true + sum(c * a / (a * a + om_syn**2) for c, a in coefs)
fit_syn = vector_fit_damping(om_syn, B_syn, n_poles=3)
ss_syn = state_space_from_fit(fit_syn)
ss_nofeed = state_space_from_fit(fit_syn)
ss_nofeed.D_ss = 0.0
feed_audit = dict(fitted_constant=float(fit_syn.constant), true_constant=d_true,
                  peak_error=float(fit_syn.relative_error_max), cases={})
for om in (0.5, 2.0, 4.0):
    ana = radiation_response(ss_syn, om)
    num_with = _harmonic_memory_response(ss_syn, om)
    num_without = _harmonic_memory_response(ss_nofeed, om)
    feed_audit["cases"][f"{om:g}"] = dict(
        amp_rel_error_with_D=abs(abs(num_with) - abs(ana)) / abs(ana),
        amp_rel_error_without_D=abs(abs(num_without) - abs(ana)) / abs(ana),
    )

# --------------------------------------------- 3. free decay: SS vs independent IRF
kw = dict(mass=body["mass"], restoring=body["restoring"],
          added_mass_inf=body["added_mass_inf"], initial_displacement=0.05, dt=0.01, duration=30.0)
dec_ss = cummins_state_space(ss10, kw["mass"], kw["added_mass_inf"], kw["restoring"],
                             initial_displacement=0.05, dt=0.01, duration=30.0)
K = radiation_memory_kernel(omega, damping, np.arange(0.0, 30.0 + 1e-9, 0.01))
dec_irf = cummins_free_decay(K, mass=kw["mass"], restoring=kw["restoring"],
                             added_mass_inf=kw["added_mass_inf"],
                             initial_displacement=0.05, dt=0.01, duration=30.0)
m_ss = decay_metrics(dec_ss.time, dec_ss.displacement, amplitude_fraction=1e-4)
m_irf = decay_metrics(dec_irf.time, dec_irf.displacement, amplitude_fraction=1e-4)
decay = dict(
    ss_period=m_ss["period"], irf_period=m_irf["period"],
    period_rel_error=abs(m_ss["period"] - m_irf["period"]) / m_irf["period"],
    ss_decay_rate=m_ss["decay_rate"], irf_decay_rate=m_irf["decay_rate"],
    decay_rate_rel_error=abs(m_ss["decay_rate"] - m_irf["decay_rate"]) / m_irf["decay_rate"],
    final_amplitude_fraction=float(abs(dec_ss.displacement[-1]) / 0.05),
    substeps=dec_ss.substeps,
    max_abs_difference=float(np.max(np.abs(dec_ss.displacement - dec_irf.displacement))),
)

# --------------------------------------------------------- 4. criteria C1-C6
val = validate_state_space(omega, added, damping, body["added_mass_inf"], body["mass"],
                           body["restoring"], fit10, ss10)
validation = {k: float(v) if isinstance(v, (int, float, np.floating)) else bool(v)
              for k, v in val.__dict__.items()}

evidence = dict(case="Hulme floating hemisphere, heave", n_frequencies=int(omega.size),
                omega_range=[float(omega[0]), float(omega[-1])],
                damping_range=[float(damping.min()), float(damping.max())], fit_ladder=ladder,
                c4_realization=c4, c4_without_feedthrough=feed_audit,
                free_decay=decay, validation=validation)
(OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))

def pct(x):
    return f"{100.0 * x:.3f} %"

rows = ["| n_poles | peak-normalised max err | pointwise floor err | d (feedthrough) | alpha range | B(w_min) data | B(w_min) fit |",
        "|---|---|---|---|---|---|---|"]
for n, d in ladder.items():
    rows.append(f"| {n} | {pct(d['peak_error'])} | {pct(d['pointwise_error'])} | {d['constant']:.4g} | "
                f"{d['alpha_min']:.4g} … {d['alpha_max']:.4g} | {d['b_at_lowest_omega_data']:.4g} | {d['b_at_lowest_omega_fit']:.4g} |")
numbers = "\n".join(rows)

c4rows = ["| omega (rad/s) | |G| analytic | |G| integrated | amp rel err | phase err |",
          "|---|---|---|---|---|"]
for om, d in c4.items():
    c4rows.append(f"| {om} | {d['analytic_abs']:.6g} | {d['numeric_abs']:.6g} | {pct(d['amp_rel_error'])} | {d['phase_error']:.2e} rad |")
c4rows.append("")
c4rows.append("### feedthrough D 项审计（合成曲线，真值 d = 50）")
c4rows.append("")
c4rows.append(f"拟合得到 d = {feed_audit['fitted_constant']:.6g}，peak err = {pct(feed_audit['peak_error'])}")
c4rows.append("")
c4rows.append("| omega (rad/s) | amp rel err (含 D) | amp rel err (丢掉 D = m14767 之前的行为) |")
c4rows.append("|---|---|---|")
for om, d in feed_audit["cases"].items():
    c4rows.append(f"| {om} | {pct(d['amp_rel_error_with_D'])} | {pct(d['amp_rel_error_without_D'])} |")

vrows = ["| 判据 | 量 | 值 | pass |", "|---|---|---|---|",
         f"| C1 fitting | peak-normalised max err | {pct(val.c1_B_relative_error_max)} | {val.c1_pass} |",
         f"| C2 stability | min pole | {val.c2_min_pole_real_part:.4g} | {val.c2_pass} |",
         f"| C3 free decay | period rel err | {pct(val.c3_period_rel_error)} | {val.c3_pass} |",
         f"| C3 free decay | decay-rate rel err | {pct(val.c3_decay_rate_rel_error)} | {val.c3_pass} |",
         f"| C4 dual path | memory amp rel err | {pct(val.c4_memory_amp_rel_error_max)} | {val.c4_pass} |",
         f"| C4 dual path | memory phase err | {val.c4_memory_phase_error_max:.2e} rad | {val.c4_pass} |",
         f"| C5 Kramers-Kronig | max rel A residual | {pct(val.c5_kk_residual)} | {val.c5_pass} |",
         f"| C6 negative control | order-1 peak err | {pct(val.c6_order1_error)} | {val.c6_pass} |",
         "", f"all_pass = **{val.all_pass}**", ""]

decay_md = "\n".join([
    "| 量 | SS 辅助 ODE | IRF 卷积 (K(t) 由列表 B 积分) | 相对差 |", "|---|---|---|---|",
    f"| 固有周期 (s) | {m_ss['period']:.5f} | {m_irf['period']:.5f} | {pct(decay['period_rel_error'])} |",
    f"| 衰减率 (1/s) | {m_ss['decay_rate']:.5f} | {m_irf['decay_rate']:.5f} | {pct(decay['decay_rate_rel_error'])} |",
    f"| 30 s 后幅值 / 初值 | — | — | SS {pct(decay['final_amplitude_fraction'])} |",
    f"| 两条路径最大位移差 (m) | — | — | {decay['max_abs_difference']:.4g} (峰值 0.05) |",
    f"| RK4 内部子步数 | {dec_ss.substeps} | 1 | — |", ""])

(OUT / "numbers.md").write_text(
    "# m14767 numbers (程序化生成，勿手改)\n\n## 1. 拟合阶梯\n\n" + numbers +
    "\n\n## 2. 实现（realization）双路径 C4\n\n" + "\n".join(c4rows) +
    "\n\n## 3. 自由衰减 SS vs IRF\n\n" + decay_md +
    "\n## 4. 判据 C1–C6（10 极点，Hulme 半球）\n\n" + "\n".join(vrows))

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
ax.loglog(omega, damping, "k.", ms=3, label="BEM sweep (data)")
ax.loglog(omega, fit10.B_reconstructed, "-", lw=1.6, label="fit, 10 poles")
ax.loglog(omega, fit15.B_reconstructed, "--", lw=1.2, label="fit, 15 poles")
ax.set_ylabel("B (kg/s)")
ax.set_title("Hulme hemisphere heave: radiation damping fit (m14767 separable weighted LS)")
ax.grid(True, which="both", alpha=0.3); ax.legend()
ax2 = axes[1]
ax2.semilogx(omega, 100 * np.abs(fit10.B_reconstructed - damping) / damping.max(), lw=1.4, label="10 poles")
ax2.semilogx(omega, 100 * np.abs(fit15.B_reconstructed - damping) / damping.max(), "--", lw=1.2, label="15 poles")
ax2.axhline(100 * 0.15, color="r", ls=":", lw=1.2, label="test gate 15 %")
ax2.set_xlabel("omega (rad/s)"); ax2.set_ylabel("|dB| / max B (%)"); ax2.grid(True, which="both", alpha=0.3); ax2.legend()
fig.tight_layout(); fig.savefig(OUT / "fig1_fit.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(dec_ss.time, dec_ss.displacement, "-", lw=1.8, label="state space (10 poles)")
ax.plot(dec_irf.time[::5], dec_irf.displacement[::5], "--", lw=1.2, label="IRF convolution (independent)")
ax.set_xlabel("t (s)"); ax.set_ylabel("heave z (m)")
ax.set_title("Free decay, Hulme hemisphere: two independent paths "
             f"(period diff {100*decay['period_rel_error']:.2f} %, decay-rate diff "
             f"{100*decay['decay_rate_rel_error']:.2f} %)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig2_decay.png", dpi=140); plt.close(fig)

print(numbers); print("\n".join(c4rows)); print(decay_md); print("\n".join(vrows))
print("converged/all finite:", bool(np.all(np.isfinite(dec_ss.displacement))))
