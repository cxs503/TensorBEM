# Extended-band probe: CASE-BEM-CUMMINS-WIGLEY

Auto-rendered by `tensorbem/benchmarks/cummins_wigley_extended_probe.py` from `extended_probe_report.json`; every number below is programmatic.

## 1. What was asked

Gate `C3a` scores a Kramers-Kronig sample only when the omitted band `s > s_max` cannot produce the residual on its own, i.e. when the measured damping slope at the band top exceeds `p > 2.05`. On the committed 12 rad/s band surge, sway and yaw fail that test, so three of six DOF are uncertified. The declared remedy is to widen the band to 20 rad/s on a `C2`-compliant mesh.

## 2. Why this document is a probe and not a gate

| mesh | sections x levels | panels | max panel scale [m] | C2 band limit [rad/s] | C2 over 0.2..20 |
| --- | --- | --- | --- | --- | --- |
| coarse | 12 x 4 | 192 | 0.1004 | 9.88 | NO |
| medium | 20 x 6 | 480 | 0.0645 | 12.33 | NO |

probe: the probe meshes exceed gate C2 (k*sqrt(A_panel) > 1) above the frequency where the certified mesh starts to matter, so these numbers bound the physics but do not certify it.

Shared-band reproduction of the committed sweep on 1 overlapping frequencies: worst relative difference of `B` = 0.000000 (pitch).

## 3. Does the widening reach algebraic decay?

| DOF | p at 12 (committed) | p at 20 (coarse/medium) | budget admissible at 20 (coarse/medium) | verdict |
| --- | --- | --- | --- | --- |
| surge | 1.09 | 6.53 / 4.54 | yes / yes | robust: both meshes put the tail above the algebraic-decay bar |
| sway | 1.53 | 4.79 / 2.55 | yes / yes | robust: both meshes put the tail above the algebraic-decay bar |
| heave | 2.87 | 4.87 / 1.99 | yes / no | inconclusive: the two meshes straddle the bar, so the mesh error is as large as the margin |
| roll | 3.20 | 5.49 / 7.52 | yes / yes | robust: both meshes put the tail above the algebraic-decay bar |
| pitch | 3.07 | -2.03 / 5.93 | no / yes | inconclusive: the two meshes straddle the bar, so the mesh error is as large as the margin |
| yaw | 1.62 | 1.97 / 5.58 | no / yes | inconclusive: the two meshes straddle the bar, so the mesh error is as large as the margin |

Robust conclusions: surge, sway, roll. Mesh-bracketed (inconclusive): heave, pitch, yaw.

## 4. What the certified sweep would cost (measured, then extrapolated)

| sections x levels | panels | wall time of one node [s] |
| --- | --- | --- |
| 12 x 4 | 192 | 0.59 |
| 20 x 6 | 480 | 3.46 |

Fitted cost law `t ~ N^1.93`; at the certified mesh (3840 panels) that is 3.2 min per node, i.e. **1 CPU-hours** for the 10-node band, against the 2-4 h the plan assumed.

## 5. Disposition

Gates `C8`/`C9` remain **not_run**: the probe bounds the physics but its meshes violate `C2`, and a probe cannot promote itself to a gate. The declared certified sweep stays executable (`scripts/case_cummins_wigley_extended_band.py`) and stays out of the verdict until it is actually run.
