# m17243 — Rigid-lid hemisphere added mass (ka -> 0)

Reference (even-parity double-body series): `A33(0)/(rho*nabla) = 0.830954148` (31 even modes, tail = 4.63e-09).

## Mesh ladder (ka = 1e-3)

| sub | panels | A33/(rho*nabla) | rel err vs ref | cond |
|---|---|---|---|---|
| 2 | 64 | 0.807388 | -2.836 % | 2.18 |
| 3 | 256 | 0.828152 | -0.337 % | 2.1 |
| 4 | 1024 | 0.833304 | +0.283 % | 2.06 |

## ka ladder (256 panels)

| ka | A33/(rho*nabla) | rel err vs ref | Hulme | rel err vs Hulme |
|---|---|---|---|---|
| 0.001 | 0.828152 | -0.337 % | 0.835066 | -0.828 % |
| 0.01 | 0.846565 | +1.879 % | 0.853644 | -0.829 % |
| 0.03 | 0.864358 | +4.020 % | 0.871489 | -0.818 % |
| 0.1 | 0.856361 | +3.058 % | 0.862792 | -0.745 % |

## Criteria

| id | measured | criterion | verdict |
|---|---|---|---|
| C1_reference_below_rho_nabla | 0.830954 | ref < 0.9 (refutes the 1.0 rho*nabla anchor) | PASS |
| C2_headline_within_5pct | 0.282785 | |rel err| < 5 % at ka=1e-3, finest mesh (sub=4) | PASS |
| C3_mesh_convergence_monotone | [0.028360520670282607, 0.003372581600669222, 0.002827846418132385] | error strictly decreases 64 -> 256 -> 1024 panels | PASS |
| C2b_256panel_within_5pct | -0.337258 | |rel err| < 5 % at ka=1e-3, 256 panels | PASS |
| C4_ka_ladder_bounded_5pct_at_ka_le_0.01 | 0.0187872 | max |rel err| < 5 % for ka <= 0.01 (256 panels) | PASS |
| C5_series_converged | 4.63474e-09 | last two even partial sums differ < 1e-6 | PASS |

**Verdict: ALL PASS**  (wall 89.7 s)