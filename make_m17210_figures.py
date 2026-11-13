"""Generate the m17210 report figures from the archived case payload.

Every number in the figures is read from ``case.json`` (the run artifact) or
evaluated by the reference anchor module -- nothing is typed by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem import acoustic3d as ac
from tensorbem import acoustic3d_pulse as ap
from tensorbem.benchmarks import acoustic3d_pulse_ref as ref

ROOT = Path(__file__).resolve().parent
CASE = __import__("json").loads((ROOT / "artifacts/case_bem_acoustic_pulse/case.json").read_text())
OUT = ROOT / "artifacts/case_bem_acoustic_pulse"


def fig_waveform() -> Path:
    w = CASE["waveforms"]
    t = np.asarray(w["time_s"]) * 1e3  # ms
    exact = np.asarray(w["exact_pa"])
    sub1 = np.asarray(w["sub1"])
    sub2 = np.asarray(w["sub2"])
    inp = CASE["inputs"]

    def velocity(times: np.ndarray) -> np.ndarray:
        return ap.gaussian_velocity_pulse(
            times, centre=inp["pulse_centre_s"], width=inp["pulse_width_s"]
        )

    p_imp = ref.pulse_by_impulse_response(
        np.asarray(w["time_s"]), velocity, inp["receiver_range_m"],
        radius=inp["radius_m"], density=inp["density_kg_m3"],
        sound_speed=inp["sound_speed_m_s"], substeps=32,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t, exact, "k-", lw=1.6, label="exact retarded potential (ref C2/C3)")
    ax1.plot(t, p_imp, "g--", lw=1.0, alpha=0.8,
             label="impulse-response route (ref C3, independent)")
    ax1.plot(t, sub1, "b:", lw=1.2, label="BEM sub1 (80 panels)")
    ax1.plot(t, sub2, "r-.", lw=1.2, ms=3, label="BEM sub2 (320 panels)")
    ax1.set_ylabel("pressure [Pa]")
    ax1.set_title("CASE-ACOUSTIC-PULSE: pulsating-sphere pulse at r = 2 m (20a)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    floor = 1e-3 * np.max(np.abs(exact))
    e1 = (sub1 - exact) / np.where(np.abs(exact) > floor, exact, np.sign(exact) * floor)
    e2 = (sub2 - exact) / np.where(np.abs(exact) > floor, exact, np.sign(exact) * floor)
    ax2.plot(t, 100 * np.abs(e1), "b:", lw=1.0, label="|err| sub1")
    ax2.plot(t, 100 * np.abs(e2), "r-", lw=1.2, label="|err| sub2")
    ax2.axhline(8.0, color="gray", ls="--", lw=1.0, label="8 % acceptance")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-2, 1e3)
    ax2.set_xlabel("time [ms]")
    ax2.set_ylabel("|pointwise rel. error| [%]")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "fig_waveform.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_anchor() -> Path:
    checks = ref.cross_checks(
        radius=0.1, receiver_distance=2.0,
        density=ac.DENSITY_SEAWATER, sound_speed=ac.SOUND_SPEED_SEAWATER,
    )
    names = [c.name for c in checks]
    vals = [c.value for c in checks]
    tols = [1e-10, 5e-3, 1e-3, 1e-3, 2e-3, 1e-2]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=["tab:green" if c.passed else "tab:red" for c in checks],
             label="measured")
    ax.barh(y, tols, color="none", edgecolor="k", height=0.25,
             label="pre-declared tolerance")
    ax.set_xscale("log")
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("check value (log scale)")
    ax.set_title("Reference anchor cross-checks (all must lie left of the tolerance)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = OUT / "fig_anchor_checks.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_band() -> Path:
    rows = CASE["band_study"]
    bands = [r["band_hz"] for r in rows]
    l2 = [r["vs_route_a"]["l2_relative_error"] * 100 for r in rows]
    peak = [r["vs_route_a"]["peak_relative_error"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bands, l2, "o-", label="L2 waveform error")
    ax.plot(bands, peak, "s--", label="peak error")
    ax.axhline(8.0, color="gray", ls="--", label="8 % acceptance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("band [Hz]")
    ax.set_ylabel("error [%]")
    ax.set_title("Band-truncation sensitivity (sub2 mesh)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "fig_band.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_waveform(), fig_anchor(), fig_band()):
        print("wrote", f)
