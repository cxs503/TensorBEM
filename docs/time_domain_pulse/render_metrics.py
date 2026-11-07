"""Render the time-domain impulse-response metrics table from the JSON artifact.

The numbers quoted in ``docs/time_domain_pulse.md`` are not typed by hand: this
script reads ``artifacts/time_domain_pulse.json`` (produced by
``python -m tensorbem.benchmarks.time_domain_pulse``) and writes
``docs/time_domain_pulse/metrics.md``. Re-running the benchmark and this script
refreshes the document.

    python docs/time_domain_pulse/render_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "time_domain_pulse.json"
PINNED = ROOT / "docs" / "time_domain_pulse" / "time_domain_pulse.json"
OUTPUT = ROOT / "docs" / "time_domain_pulse" / "metrics.md"


def _fmt(value: float, digits: int = 6) -> str:
    if value != value:  # NaN
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.{digits}g}"


def render(record: dict) -> str:
    case = record["case"]
    criteria = record["criteria"]
    c1 = criteria["C1_decay_and_parity"]
    c2 = criteria["C2_fourier_round_trip"]
    c3 = criteria["C3_time_vs_frequency"]
    decay = c3["free_decay"]
    response = c3["resonance"]
    body = case["body"]
    sweep = case["sweep"]
    lines: list[str] = []
    add = lines.append

    add("| quantity | value | source |")
    add("|---|---|---|")
    add(f"| displaced volume nabla [m^3] | {_fmt(body['displaced_volume'])} | analytic (2/3 pi a^3) |")
    add(f"| body mass M = rho nabla [kg] | {_fmt(body['mass'])} | analytic |")
    add(f"| heave restoring C33 = rho g A_wp [N/m] | {_fmt(body['restoring'])} | analytic (A_wp = pi a^2) |")
    add(f"| A_inf used [kg] | {_fmt(body['added_mass_inf'])} | 0.5 rho nabla, sweep asymptote |")
    add(f"| sweep points / omega range [rad/s] | {sweep['points']} / {_fmt(sweep['omega_range'][0])}-{_fmt(sweep['omega_range'][1])} | Hulme (1982) |")
    add(f"| damping peak B33 [kg/s] at omega [rad/s] | {_fmt(sweep['damping_peak'])} at {_fmt(sweep['omega_at_damping_peak'])} | Hulme (1982) |")
    add(f"| K(0) = (2/pi) int B domega [kg/s^2] | {_fmt(c1['value_at_zero'])} | eq. (1) |")
    add(f"| first zero of K [s] | {_fmt(c1['first_zero'])} | eq. (1) |")
    add(f"| envelope decay rate of K [1/s] | {_fmt(c1['decay_rate'])} | log fit, 50%-1% of peak |")
    add(f"| memory time scale 1/rate [s] | {_fmt(c1['memory_time_scale'])} | derived |")
    add(f"| kernel tail beyond tau_max / peak | {_fmt(c1['tail_relative'])} | eq. (1) |")
    add(f"| even-extension residual (two-sided) | {_fmt(c1['even_residual'])} | roundoff |")
    add(f"| imaginary part of two-sided spectrum / real | {_fmt(c1['two_sided_parity']['relative_imaginary'])} | parity |")
    add(f"| parity-closure residual / kernel peak | {_fmt(c1['parity_closure']['relative_residual_max'])} | even/odd reconstruction |")
    add(f"| round-trip band [rad/s] | {_fmt(c2['band'][0])}-{_fmt(c2['band'][1])} | sampling-limited |")
    add(f"| kernel tau_max / dt [s] | {_fmt(c2['tau_max'])} / {_fmt(c2['kernel_dt'])} | round-trip grid |")
    add(f"| round-trip B error / peak B | {_fmt(c2['damping_relative_error_max'])} | eq. (2) |")
    add(f"| round-trip A error / peak A | {_fmt(c2['added_mass_relative_error_max'])} | eq. (3) |")
    add(f"| A_inf implied by the round trip [kg] | {_fmt(c2['implied_added_mass_inf']['mean'])} | eq. (3) inverted |")
    add(f"| implied A_inf error / A_inf | {_fmt(c2['relative_error_of_implied_added_mass_inf'])} | derived |")
    add(f"| implied A_inf spread / A_inf | {_fmt(c2['implied_added_mass_inf']['relative_spread'])} | derived |")
    add(f"| free-decay period [s] | {_fmt(decay['period'])} | Cummins integration |")
    add(f"| free-decay rate [1/s] | {_fmt(decay['decay_rate'])} | envelope fit, R^2 = {_fmt(decay['fit_r_squared'])} |")
    add(f"| extrema used / total | {decay['extrema_used']} / {decay['extrema_total']} | decay_metrics |")
    add(f"| resonance omega_n [rad/s] | {_fmt(response['omega_n'])} | Re kappa = 0 |")
    add(f"| zeta (energy identity) | {_fmt(response['zeta'])} | B/(2(M+A)omega_n) |")
    add(f"| zeta (half-power bandwidth) | {_fmt(response['zeta_bandwidth'])} | abs(1/kappa) bandwidth |")
    add(f"| response period 2 pi/(omega_n sqrt(1-zeta^2)) [s] | {_fmt(response['damped_period'])} | derived |")
    add(f"| half-power band [rad/s] | {_fmt(response['half_power_omega_low'])}-{_fmt(response['half_power_omega_high'])} | abs(1/kappa) |")
    add(f"| decay rate from bandwidth [1/s] | {_fmt(c3['decay_rate_from_bandwidth'])} | zeta_bw * omega_n |")
    add(f"| period error vs response | {_fmt(c3['period_relative_error_vs_resonance'])} | criterion < 1% |")
    add(f"| decay-rate error vs bandwidth | {_fmt(c3['decay_rate_relative_error_vs_bandwidth'])} | criterion < 5% |")
    add(f"| decay-rate error vs energy identity | {_fmt(c3['decay_rate_relative_error_vs_energy_identity'])} | reported, not a criterion |")
    for label, entry in sorted(c3["driven_response"].items()):
        add(
            f"| driven {label}: amplitude / phase error | {_fmt(entry['amplitude_relative_error'])} / "
            f"{_fmt(entry['phase_error'])} rad | time domain vs 1/kappa |"
        )
    add("")
    add("## criterion verdicts")
    add("")
    add("| criterion | requirement | measured | verdict |")
    add("|---|---|---|---|")
    add(
        f"| C1 decay | decay rate > 0, tail < 5% | {_fmt(c1['decay_rate'])} 1/s, {_fmt(c1['tail_relative'])} | "
        f"{'PASS' if c1['criterion']['decay_rate_positive'] and c1['criterion']['tail_below_5_percent'] else 'FAIL'} |"
    )
    add(
        f"| C1 symmetry | even at roundoff, parity closure < 2% | {_fmt(c1['even_residual'])}, "
        f"{_fmt(c1['parity_closure']['relative_residual_max'])} | "
        f"{'PASS' if c1['criterion']['even_residual_at_roundoff'] and c1['criterion']['parity_closure_below_2_percent'] else 'FAIL'} |"
    )
    add(
        f"| C2 round trip | worst in-band error < 5% | {_fmt(c2['worst_relative_error'])} | "
        f"{'PASS' if c2['passed'] else 'FAIL'} |"
    )
    add(
        f"| C3 free decay | period < 1%, rate < 5% | {_fmt(c3['period_relative_error_vs_resonance'])}, "
        f"{_fmt(c3['decay_rate_relative_error_vs_bandwidth'])} | {'PASS' if c3['passed'] else 'FAIL'} |"
    )
    add(
        f"| C3 driven | amplitude < 2%, phase < 0.05 rad | "
        f"{_fmt(max(entry['amplitude_relative_error'] for entry in c3['driven_response'].values()))}, "
        f"{_fmt(max(entry['phase_error'] for entry in c3['driven_response'].values()))} | "
        f"{'PASS' if c3['criterion']['driven_amplitude_within_2_percent'] and c3['criterion']['driven_phase_within_0p05_rad'] else 'FAIL'} |"
    )
    add("")
    add(f"All criteria: **{'PASS' if record['all_passed'] else 'FAIL'}**")
    return "\n".join(lines) + "\n"


def main() -> int:
    source = ARTIFACT if ARTIFACT.exists() else PINNED
    record = json.loads(source.read_text())
    OUTPUT.write_text(render(record))
    print(f"wrote {OUTPUT} from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())