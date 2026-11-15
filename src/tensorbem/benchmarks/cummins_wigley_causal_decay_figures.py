"""Figures for the C5 attribution audit (``docs/case_bem_cummins_wigley_c5_attribution``).

Four panels, each one a claim of the audit:

1. ``fig_kernel_tails.png`` -- the truncated kernel does not decay, the causal
   one does (log |K(tau)|).
2. ``fig_band_fragility.png`` -- legacy damping ratio vs the truncated band top,
   with the causal exact value as a horizontal reference.
3. ``fig_decay_records.png`` -- the two time-domain free-decay records.
4. ``fig_estimators.png`` -- every estimator of the heave damping ratio with its
   order-scan spread.
"""

from __future__ import annotations

import pathlib

import numpy as np

try:  # pragma: no cover - plotting only
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from tensorbem.benchmarks import cummins_wigley_causal_decay as audit


def _guard():
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required for the figures")


def fig_kernel_tails(result: dict, path: pathlib.Path) -> pathlib.Path:
    _guard()
    omega, _added, damping, restoring, mass, a_inf = audit.load_heave(result["level"])
    legacy = audit.legacy_free_decay(omega, damping, mass=mass, restoring=restoring,
                                     added_mass_inf=a_inf)
    fit, realization, _n = audit.causal_fit(omega, damping)
    tau = np.asarray(legacy["kernel_tau"], dtype=float)
    causal = audit.causal_kernel(realization, tau)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(tau, np.maximum(np.abs(legacy["kernel_values"]), 1e-12), label="band-truncated kernel (legacy route)")
    ax.semilogy(tau, np.maximum(np.abs(causal), 1e-12), label="causal (rational) kernel")
    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(r"$|K(\tau)|$ [N s/m]")
    ax.set_title("Heave memory kernel: hard cutoff oscillates, causal model decays")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_band_fragility(result: dict, path: pathlib.Path) -> pathlib.Path:
    _guard()
    rows = [r for r in result["band_top_scan"] if "zeta_decay" in r]
    tops = [r["band_top"] for r in rows]
    zeta = [r["zeta_decay"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(tops, zeta, "o-", label="legacy route (truncated kernel)")
    ax.axhline(result["causal_model"]["zeta_exact"], color="k", ls="--",
               label=r"causal exact modal $\zeta$")
    ax.axhline(result["legacy_reference"]["zeta_frequency_domain"], color="tab:red", ls=":",
               label="legacy frequency-domain reference")
    ax.set_xlabel(r"truncated band top $\omega_{\max}$ [rad/s]")
    ax.set_ylabel(r"heave damping ratio $\zeta$ from free decay")
    ax.set_title("Legacy decay route is a function of the cutoff, not of the hull")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_decay_records(level: str, path: pathlib.Path) -> pathlib.Path:
    _guard()
    omega, _added, damping, restoring, mass, a_inf = audit.load_heave(level)
    legacy = audit.legacy_free_decay(omega, damping, mass=mass, restoring=restoring,
                                     added_mass_inf=a_inf)
    fit, realization, _n = audit.causal_fit(omega, damping)
    causal = audit.causal_free_decay(realization, mass=mass, restoring=restoring,
                                     added_mass_inf=a_inf)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tmax = 12.0
    mask_l = np.asarray(legacy["time"]) <= tmax
    mask_c = np.asarray(causal["time"]) <= tmax
    ax.plot(np.asarray(legacy["time"])[mask_l], np.asarray(legacy["displacement"])[mask_l],
            lw=1.4, label=f"legacy kernel, $\\zeta$={legacy['zeta_decay']:.4f}")
    ax.plot(np.asarray(causal["time"])[mask_c], np.asarray(causal["displacement"])[mask_c],
            lw=1.4, ls="--", label=f"causal kernel, $\\zeta$={causal['zeta_decay']:.4f}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("heave displacement [m]")
    ax.set_title("Cummins free decay: same estimator, two kernels")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_estimators(result: dict, path: pathlib.Path) -> pathlib.Path:
    _guard()
    causal = result["causal_model"]
    names = [
        "legacy\ndecay\n(truncated)",
        "legacy\nreference\n(interp. energy)",
        "causal\nexact root",
        "causal\ndecay\n(time domain)",
    ]
    values = [
        result["legacy_route"]["zeta_decay"],
        result["legacy_reference"]["zeta_frequency_domain"],
        causal["zeta_exact"],
        causal["zeta_decay_time_domain"],
    ]
    errors = [0.0, 0.0, causal["zeta_exact_order_spread"], 0.0]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(names, values, yerr=errors, capsize=4,
                  color=["tab:red", "tab:olive", "tab:blue", "tab:green"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(r"heave damping ratio $\zeta$")
    ax.set_title("Four estimators; error bar = admissible order-scan spread")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all(target: pathlib.Path, level: str = "medium") -> dict[str, pathlib.Path]:
    target = pathlib.Path(target)
    target.mkdir(parents=True, exist_ok=True)
    result = audit.run_audit(level)
    out = {
        "kernel_tails": fig_kernel_tails(result, target / "fig_kernel_tails.png"),
        "band_fragility": fig_band_fragility(result, target / "fig_band_fragility.png"),
        "decay_records": fig_decay_records(level, target / "fig_decay_records.png"),
        "estimators": fig_estimators(result, target / "fig_estimators.png"),
    }
    return out
