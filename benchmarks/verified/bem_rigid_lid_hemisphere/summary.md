# Floating hemisphere heave added mass: rigid-lid reference corrected

Reference used by report #15809 was `A33(0) = rho*nabla` (1.0); the even-parity double-body series gives `A33(0)/(rho*nabla) = 0.830954148` (31 even modes, degree_max = 60).

## Solver vs both references

| ka | panels | A33/(rho*nabla) | even-mode ref | rel err | report#15809 ref 1.0 | rel err | Hulme | rel err |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 64 | 0.807388 | 0.830954 | -2.836 % | 1.000000 | -19.261 % | 0.835066 | -3.314 % |
| 0.01 | 64 | 0.825313 | 0.830954 | -0.679 % | 1.000000 | -17.469 % | 0.853644 | -3.319 % |
| 0.03 | 64 | 0.842934 | 0.830954 | +1.442 % | 1.000000 | -15.707 % | 0.871489 | -3.277 % |
| 0.1 | 64 | 0.836912 | 0.830954 | +0.717 % | 1.000000 | -16.309 % | 0.862792 | -3.000 % |
| 0.2 | 64 | 0.773478 | 0.830954 | -6.917 % | 1.000000 | -22.652 % | 0.793885 | -2.571 % |
| 0.3 | 64 | 0.699847 | 0.830954 | -15.778 % | 1.000000 | -30.015 % | 0.715726 | -2.219 % |
| 0.6 | 64 | 0.528117 | 0.830954 | -36.444 % | 1.000000 | -47.188 % | 0.538141 | -1.863 % |
| 1 | 64 | 0.415987 | 0.830954 | -49.939 % | 1.000000 | -58.401 % | 0.428462 | -2.912 % |
| 0.001 | 256 | 0.828152 | 0.830954 | -0.337 % | 1.000000 | -17.185 % | 0.835066 | -0.828 % |
| 0.01 | 256 | 0.846565 | 0.830954 | +1.879 % | 1.000000 | -15.343 % | 0.853644 | -0.829 % |
| 0.03 | 256 | 0.864358 | 0.830954 | +4.020 % | 1.000000 | -13.564 % | 0.871489 | -0.818 % |
| 0.1 | 256 | 0.856361 | 0.830954 | +3.058 % | 1.000000 | -14.364 % | 0.862792 | -0.745 % |
| 0.2 | 256 | 0.788861 | 0.830954 | -5.066 % | 1.000000 | -21.114 % | 0.793885 | -0.633 % |
| 0.3 | 256 | 0.711853 | 0.830954 | -14.333 % | 1.000000 | -28.815 % | 0.715726 | -0.541 % |
| 0.6 | 256 | 0.535706 | 0.830954 | -35.531 % | 1.000000 | -46.429 % | 0.538141 | -0.452 % |
| 1 | 256 | 0.425312 | 0.830954 | -48.816 % | 1.000000 | -57.469 % | 0.428462 | -0.735 % |

## Criteria

| id | measured | criterion | verdict |
|---|---|---|---|
| R1_even_mode_reference_refutes_rho_nabla | 0.830954 | rigid-lid value < 0.9 rho*nabla | PASS |
| R2_pressure_release_endpoint_is_half_rho_nabla | 0.5 | == 0.5 (anchor A2) to 1e-12 | PASS |
| R3_even_series_is_converged | 4.63474e-09 | last two partial sums differ by < 1e-6 | PASS |
| R4_solver_zero_frequency_matches_even_mode_reference | -0.00337258 | |rel err| < 1 % at ka = 1e-03, finest mesh | PASS |
| R5_solver_matches_hulme_over_whole_band | 0.00829211 | |rel err| < 3 % for every ka <= 1.0, finest mesh | PASS |
| R6_mesh_ladder_reduces_the_endpoint_gap | 0.118918 | < 1 (refinement reduces the gap to the even-mode value) | PASS |
| R7_no_mesh_or_ka_reaches_the_report_15809_reference | 0.864358 | every computed A33/(rho*nabla) with ka <= 0.1 stays < 0.92 at all mesh levels (so the 1.0 reference is outside the solution envelope) | PASS |

**Verdict: ALL PASS**

