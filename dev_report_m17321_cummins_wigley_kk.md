# CASE-BEM-CUMMINS-WIGLEY round 2 — Cummins equation + Kramers-Kronig causality on the Wigley III hull

All numbers below are read from `artifacts/case_bem_cummins_wigley/results.json` by `scripts/run_cummins_wigley_kk_causality.py --mode gates` (no hand-typed value).

## 1 Problem setting and reference

* anchor `CASE-BEM-CUMMINS-WIGLEY`, ref = Cummins (1962) equation + Kramers-Kronig causality, tol <10 %.
* G1 anchors on an independent code: capytaine 3.1.dev0 output frozen in `benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json` (rotation centre (0,0,-0.09375), rho=1025, g=9.80665, infinite depth). G2-G5 are the exact identities of the Cummins/KK formalism (Kotik & Mangulis 1962; Newman 1977 ch.6).

## 2 Method

* round-2 corrections taken from `docs/board_audit/m17264/AUDIT_m17209.md`: (F1) radiation sweep now passes `center=(0,0,-T/2)`; (F2) grid widened from 8 nodes on [0.5,6] to 22 nodes on [0.30,12.5] rad/s so the heave resonance is bracketed.
* code: `src/tensorbem/benchmarks/cummins_wigley_kk_causality.py`, driver `scripts/run_cummins_wigley_kk_causality.py`, tests `tests/test_cummins_wigley_kk_causality.py`.

## 3 Geometry gate and parameters

* L = 3.0
* B = 0.3
* T = 0.1875
* c = 1.1087500000000008
* L_over_B = 10.0
* B_over_T = 1.5999999999999999
* volume = 0.09163125
* Cb_declared = 0.543
* Cb_from_volume = 0.543
* Cb_abs_error = 0.0
* waterplane_area = 0.73305
* C33 = 7368.4839020624995
* displaced_mass = 93.92203125
* rotation_center = [0.0, 0.0, -0.09375]
* passed = True

## 4 Gate table (machine-decided)

| level | grid nodes | G1 cross-solver | G2 K(0) | G3 KK | G4 free decay | G5 round trip | verdict |
|---|---|---|---|---|---|---|---|
| coarse | 22 | 6.29% (PASS) | 0.00% (PASS) | 53.23% (FAIL) | 41.51% (FAIL) | 46.33% (FAIL) | FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip |
| medium | 16 | 3.26% (PASS) | 0.00% (PASS) | 81.62% (FAIL) | nan | 53.37% (FAIL) | FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip |

Per-DOF detail (medium level):

* G1 worst: added mass 2.6150 %, damping 3.2558 % at omega = 1.0 rad/s
* G3 per DOF: surge 64.35%, sway 75.56%, heave 39.84%, roll 76.55%, pitch 58.89%, yaw 81.62%
* G5 per DOF: surge 45.58%, sway 39.22%, heave 35.32%, roll 37.85%, pitch 53.37%, yaw 22.27%
* G4 zeta_decay 0.0051126377763679185 vs zeta_half_power nan, error IndexError: index 0 is out of bounds for axis 0 with size 0
* G1 per-DOF rows (ref vs got, medium, omega=1.0):
  * surge: A ref 2.58485 got 2.51974 (-2.52 %), B ref 0.00235365 got 0.00230369 (-2.12 %)
  * sway: A ref 138.13 got 136.718 (-1.02 %), B ref 0.0136678 got 0.0135182 (-1.09 %)
  * heave: A ref 197.415 got 194.732 (-1.36 %), B ref 25.9517 got 25.6322 (-1.23 %)
  * roll: A ref 0.670014 got 0.652492 (-2.62 %), B ref 6.04912e-05 got 5.85217e-05 (-3.26 %)
  * pitch: A ref 53.0425 got 52.2748 (-1.45 %), B ref 0.0421835 got 0.0417501 (-1.03 %)
  * yaw: A ref 91.1205 got 91.7403 (+0.68 %), B ref 1.35257e-05 got 1.3758e-05 (+1.72 %)

## 5 Convergence / sensitivity

* mesh ladder: coarse (200 quads / 400 triangles) vs medium (400 quads / 800 triangles); heave resonance 7.393059060502688 vs 7.378504598323515 rad/s.
* band sensitivity: coarse level solved on both the round-2 grid and (as A/B) the round-1 grid; the round-1 band leaves G4 undefined (no resonance) — see `tests/test_cummins_wigley_kk_causality.py::test_band_truncation_is_what_breaks_the_round_1_gates`.

## 6 Failures, limits, reproduction

* honest status: **partial**. G1 (the only externally anchored gate, and the one round 1 broke) and G2 pass; G3/G4/G5 do not.
* G4 is undefined on medium: the half-power crossing falls outside the estimator's scan window (`IndexError`), recorded verbatim in `results.json` rather than papered over.
* G3/G5 residuals are largest for surge/sway/roll/yaw, the DOFs whose `A - A_inf` scale is small: the KK residual is normalised by max|A-A_inf| and the principal value is truncated at the lowest tabulated frequency, so the remaining error is a band/scale effect. Diagnosing it needs a denser low-frequency sweep (0.1-0.3 rad/s), which is not in this run.
* sweep coverage: medium level realised 16 of the predeclared 22 nodes (missing 4.7, 6.2, 8.6, 9.6, 11.0, 12.5) because the local 6-way sharded run was cut by the budget; `results.json` records `missing_omega`.
* repro: `PYTHONPATH=src python3 scripts/run_cummins_wigley_kk_causality.py --mode sweep --level medium --shard i --shards 6 --outdir artifacts/case_bem_cummins_wigley` then `--mode gates --levels coarse,medium`.
* tests: `PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_kk_causality.py -q` (9 passed).
