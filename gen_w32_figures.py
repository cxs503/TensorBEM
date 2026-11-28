#!/usr/bin/env python3
"""Generate W32 figures from truncation_convergence.json."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pathlib

ROOT = pathlib.Path("/root/tp-agent/work/m28259")
ARTIFACT = ROOT / "papers/data/w32-2026-11-16/seakeeping/truncation_convergence.json"
FIG_DIR = ROOT / "papers/figures/w32"
FIG_DIR.mkdir(parents=True, exist_ok=True)

d = json.loads(ARTIFACT.read_text())
sea_states = ["ss_T1_1p00", "ss_T1_1p40", "ss_T1_1p80"]
labels = {"ss_T1_1p00": "T1=1.0s", "ss_T1_1p40": "T1=1.4s", "ss_T1_1p80": "T1=1.8s"}

# Figure 1: RAO curves (from W25 artifact)
W25_ARTIFACT = ROOT / "papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json"
w25 = json.loads(W25_ARTIFACT.read_text())
rt = w25["rao_table"]
om = np.array([r[0] for r in rt["rows"]])
heave = np.array([r[2] for r in rt["rows"]])
pitch = np.array([r[3] for r in rt["rows"]])

plt.figure(figsize=(8, 6))
plt.plot(om, heave, label='Heave RAO')
plt.plot(om, pitch, label='Pitch RAO')
for om_max in [3.0, 6.0, 8.0, 12.0]:
    plt.axvline(om_max, color='gray', linestyle='--', alpha=0.5)
plt.axvline(4.236856543337184, color='red', linestyle='--', alpha=0.5, label='Pitch peak')
plt.xlabel('ω [rad/s]')
plt.ylabel('RAO')
plt.legend()
plt.grid(True)
plt.title('Wigley III RAO (U=0)')
plt.savefig(FIG_DIR / "fig1_rao_curves.png")
plt.close()

# Figure 2: m0 convergence
plt.figure(figsize=(8, 6))
for ss in sea_states:
    rows = [r for r in d["axis_A_truncation_frequency"]["rows"] if r["sea_state"] == ss]
    rows.sort(key=lambda r: r["omega_max"])
    om_max = [r["omega_max"] for r in rows]
    m0_err_h = [r["m0_heave_trunc_error_pct"] for r in rows]
    m0_err_p = [r["m0_pitch_trunc_error_pct"] for r in rows]
    plt.semilogy(om_max, m0_err_h, 'o-', label=f'{labels[ss]} Heave')
    plt.semilogy(om_max, m0_err_p, 's-', label=f'{labels[ss]} Pitch')
plt.axhline(0.5, color='red', linestyle='--', label='0.5% gate')
plt.xlabel('ω_max [rad/s]')
plt.ylabel('m0 truncation error [%]')
plt.legend()
plt.grid(True)
plt.title('m0 Convergence vs ω_max')
plt.savefig(FIG_DIR / "fig2_m0_convergence.png")
plt.close()

# Figure 3: Peak convergence
plt.figure(figsize=(8, 6))
for ss in sea_states:
    rows = [r for r in d["axis_A_truncation_frequency"]["rows"] if r["sea_state"] == ss]
    rows.sort(key=lambda r: r["omega_max"])
    om_max = [r["omega_max"] for r in rows]
    pk_err_h = [r["peak_heave_trunc_error_pct"] for r in rows]
    pk_err_p = [r["peak_pitch_trunc_error_pct"] for r in rows]
    plt.semilogy(om_max, pk_err_h, 'o-', label=f'{labels[ss]} Heave')
    plt.semilogy(om_max, pk_err_p, 's-', label=f'{labels[ss]} Pitch')
plt.axhline(2.0, color='red', linestyle='--', label='2% gate')
plt.xlabel('ω_max [rad/s]')
plt.ylabel('Peak truncation error [%]')
plt.legend()
plt.grid(True)
plt.title('Peak Convergence vs ω_max')
plt.savefig(FIG_DIR / "fig3_peak_convergence.png")
plt.close()

print("Figures generated.")
