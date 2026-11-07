| quantity | value | source |
|---|---|---|
| displaced volume nabla [m^3] | 2.0944 | analytic (2/3 pi a^3) |
| body mass M = rho nabla [kg] | 2146.75 | analytic |
| heave restoring C33 = rho g A_wp [N/m] | 31578.7 | analytic (A_wp = pi a^2) |
| A_inf used [kg] | 1073.38 | 0.5 rho nabla, sweep asymptote |
| sweep points / omega range [rad/s] | 501 / 0.14-29.7 | Hulme (1982) |
| damping peak B33 [kg/s] at omega [rad/s] | 1746.73 at 2.74128 | Hulme (1982) |
| K(0) = (2/pi) int B domega [kg/s^2] | 3949.76 | eq. (1) |
| first zero of K [s] | 0.48 | eq. (1) |
| envelope decay rate of K [1/s] | 1.58363 | log fit, 50%-1% of peak |
| memory time scale 1/rate [s] | 0.631461 | derived |
| kernel tail beyond tau_max / peak | 1.13351e-06 | eq. (1) |
| even-extension residual (two-sided) | 0 | roundoff |
| imaginary part of two-sided spectrum / real | 4.06851e-18 | parity |
| parity-closure residual / kernel peak | 0.00963166 | even/odd reconstruction |
| round-trip band [rad/s] | 0.23648-17.82 | sampling-limited |
| kernel tau_max / dt [s] | 53.1393 / 0.01 | round-trip grid |
| round-trip B error / peak B | 0.000126735 | eq. (2) |
| round-trip A error / peak A | 0.000237488 | eq. (3) |
| A_inf implied by the round trip [kg] | 1073.38 | eq. (3) inverted |
| implied A_inf error / A_inf | 6.59366e-06 | derived |
| implied A_inf spread / A_inf | 0.000799631 | derived |
| free-decay period [s] | 1.94548 | Cummins integration |
| free-decay rate [1/s] | 0.306029 | envelope fit, R^2 = 0.999988 |
| extrema used / total | 18 / 20 | decay_metrics |
| resonance omega_n [rad/s] | 3.21814 | Re kappa = 0 |
| zeta (energy identity) | 0.083476 | B/(2(M+A)omega_n) |
| zeta (half-power bandwidth) | 0.0947055 | abs(1/kappa) bandwidth |
| response period 2 pi/(omega_n sqrt(1-zeta^2)) [s] | 1.95927 | derived |
| half-power band [rad/s] | 2.88682-3.49637 | abs(1/kappa) |
| decay rate from bandwidth [1/s] | 0.304776 | zeta_bw * omega_n |
| period error vs response | 0.0070359 | criterion < 1% |
| decay-rate error vs bandwidth | 0.00411412 | criterion < 5% |
| decay-rate error vs energy identity | 0.139191 | reported, not a criterion |
| driven above_resonance: amplitude / phase error | 0.000226794 / 0.000912786 rad | time domain vs 1/kappa |
| driven at_resonance: amplitude / phase error | 0.0030622 / 7.52812e-05 rad | time domain vs 1/kappa |
| driven below_resonance: amplitude / phase error | 0.000234725 / 0.000720779 rad | time domain vs 1/kappa |

## criterion verdicts

| criterion | requirement | measured | verdict |
|---|---|---|---|
| C1 decay | decay rate > 0, tail < 5% | 1.58363 1/s, 1.13351e-06 | PASS |
| C1 symmetry | even at roundoff, parity closure < 2% | 0, 0.00963166 | PASS |
| C2 round trip | worst in-band error < 5% | 0.000237488 | PASS |
| C3 free decay | period < 1%, rate < 5% | 0.0070359, 0.00411412 | PASS |
| C3 driven | amplitude < 2%, phase < 0.05 rad | 0.0030622, 0.000912786 | PASS |

All criteria: **PASS**
