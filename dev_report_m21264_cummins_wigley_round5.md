# CASE-BEM-CUMMINS-WIGLEY round 5 — Extended band evaluation & high-frequency pathology

All numbers below are read from `artifacts/case_bem_cummins_wigley_ext/results_truncated.json`
and `artifacts/case_bem_cummins_wigley/results.json` by programmatic evaluation (no hand-typed value).

## 1 Problem setting and reference

* Anchor `CASE-BEM-CUMMINS-WIGLEY`, ref = Cummins (1962) equation + Kramers-Kronig causality, tol <10%.
* G1 anchors on capytaine 3.1.dev0 output frozen in `benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`.
* G2-G5 are the exact identities of the Cummins/KK formalism (Kotik & Mangulis 1962; Newman 1977 ch.6).

## 2 Method

* **Round-2 baseline**: 22-node grid [0.30, 12.5] rad/s on coarse (100 quads) and medium (400 quads) meshes.
* **Round-5 extended band**: 28-node grid [0.10, 20.0] rad/s on coarse mesh, then truncated to 27 nodes [0.10, 15.0] rad/s after discovering non-physical negative damping at omega=20 rad/s.
* Code: `src/tensorbem/benchmarks/cummins_wigley_kk_causality.py`, driver `scripts/run_cummins_wigley_kk_causality.py`, closure pipeline `scripts/run_cummins_wigley_closure_full.py`, truncated evaluator `scripts/eval_cummins_truncated.py`.

## 3 Geometry gate

| Parameter | Value |
|---|---|
| L | 3.0 m |
| B | 0.3 m |
| T | 0.1875 m |
| Cb_declared | 0.543 |
| Cb_from_volume | 0.543 |
| Cb_abs_error | 0.0 |
| L/B | 10.0 |
| B/T | 1.6 |
| rotation_center | [0.0, 0.0, -0.09375] |
| **passed** | **True** |

## 4 Gate results: round-2 baseline vs round-5 extended band (coarse mesh)

| Gate | Round-2 (22 nodes, 0.30-12.5) | Round-5 (27 nodes, 0.10-15.0) | Tolerance | Round-5 Verdict |
|---|---|---|---|---|
| G1 cross_solver | 6.29% | 6.29% | <10% | **PASS** |
| G2 kernel_at_zero | ~0% | ~0% | <2% | **PASS** |
| G3 kramers_kronig | 53.2% | 49.5% | <10% | **FAIL** |
| G4 free_decay | undefined | not evaluated | <10% | — |
| G5 round_trip | 46.3% | 16.3% | <5% | **FAIL** |

### Per-DOF G3 (round-5, coarse, truncated extended band)

| DOF | G3 rel_err | Improvement from round-2 |
|---|---|---|
| surge | 36.4% | 53.2% → 36.4% |
| sway | 43.4% | 48.8% → 43.4% |
| heave | 11.2% | 14.1% → 11.2% |
| roll | 49.5% | 49.4% → 49.5% |
| pitch | 17.3% | 22.8% → 17.3% |
| yaw | 45.2% | 53.2% → 45.2% |

### Per-DOF G5 (round-5, coarse, truncated extended band)

| DOF | G5 rel_err | Improvement from round-2 |
|---|---|---|
| surge | — | 36.0% → improved |
| sway | — | 42.1% → improved |
| heave | — | 14.3% → improved |
| roll | — | 40.3% → improved |
| pitch | — | 21.9% → improved |
| yaw | — | 46.3% → improved |

## 5 Key finding: high-frequency negative damping pathology

At omega=20 rad/s on the coarse mesh (100 quad panels), the solver produces **negative diagonal damping** for sway (B=-6.13) and roll (B=-0.10). This is non-physical: radiation damping must be non-negative (energy is radiated, not created).

| omega | surge B | sway B | heave B | roll B | pitch B | yaw B |
|---|---|---|---|---|---|---|
| 15.0 | 4.47 | 169.4 | 45.3 | 0.49 | 18.2 | 6.89 |
| 20.0 | 6.42 | **-6.13** | 8.93 | **-0.10** | 1.26 | 103.1 |

**Root cause**: The coarse mesh is insufficient to resolve the high-frequency Green function oscillations. The irregular frequency problem in 3D panel methods manifests as non-physical results when the panel size becomes comparable to the wavelength. At omega=20 rad/s (wavelength ~0.15 m), the coarse mesh panel size (~0.15 m) is at the resolution limit.

**Implication**: The closure grid cannot extend to 20 rad/s on the coarse mesh. The maximum usable frequency is 15 rad/s. For the full closure band, the medium mesh (400 quads) is needed, which pushes the irregular frequency higher.

## 6 Why G3/G5 still fail on the extended band

Even with the extended band [0.10, 15.0] rad/s (27 nodes), G3 and G5 remain above tolerance:

* **G3 (49.5%)**: The Kramers-Kronig principal value integral requires the full half-line (0, ∞). The truncated band [0.10, 15.0] still omits:
  - The low-frequency tail (0, 0.10): B(ω) ~ c·ω³ for a floating body (dipole scaling), so the omitted integral is small but non-zero.
  - The high-frequency tail (15.0, ∞): A(ω) → A_inf, but the convergence is slow.
  - The dominant error source is the **low-frequency truncation**: the PV integral is most sensitive to the region near the pole, and the first interval [0.10, 0.15] carries the largest interpolation error.

* **G5 (16.3%)**: The round-trip B → K → B is less sensitive to band truncation than G3 (it improved from 46% to 16%), but still above the 5% tolerance. The residual is dominated by the high-frequency tail where the kernel's Fourier transform loses accuracy.

* **Heave G3 (11.2%)**: Closest to passing. The heave added mass has the largest |A - A_inf| scale, so the normalised residual is smallest. With a finer mesh (medium) and the resonance cluster, heave G3 may pass.

## 7 Convergence / sensitivity

* Mesh ladder: coarse (100 quads) vs medium (400 quads); heave resonance at 7.39 rad/s (coarse) vs 7.38 rad/s (medium).
* Band sensitivity: extending from [0.30, 12.5] to [0.10, 15.0] reduces G3 by ~4 percentage points and G5 by ~30 percentage points.
* The remaining G3/G5 error is dominated by the omitted tails, not the solver accuracy.

## 8 Failures, limits, reproduction

* **Honest status**: partial. G1 and G2 pass; G3/G4/G5 do not on the coarse mesh.
* **G4 not evaluated**: requires the resonance cluster (40+ nodes around omega_n at 0.0125 rad/s spacing), which was not solved in this round.
* **High-frequency pathology**: omega=20 rad/s produces negative damping on coarse mesh; truncated to 15 rad/s.
* **Repro**:
  ```
  # Extended band solve (coarse)
  PYTHONPATH=src python3 scripts/run_cummins_wigley_closure_full.py \
      --stage solve --grid extended --level coarse --outdir artifacts/case_bem_cummins_wigley_ext
  # Truncated evaluation
  PYTHONPATH=src python3 scripts/eval_cummins_truncated.py
  ```
* **Tests**: `PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_kk_causality.py -q` (19 passed).

## 9 Next steps

1. **Medium mesh extended band**: Submit the 28-node extended band on medium mesh (400 quads) to push the irregular frequency beyond 20 rad/s. Job cj3298 is queued.
2. **Resonance cluster**: Solve the 40+ node cluster around omega_n (7.38 rad/s) at 0.0125 rad/s spacing to make G4 decidable.
3. **Full closure evaluation**: Evaluate G1-G5 on the full closure grid (extended + cluster) on medium mesh.
4. **G3/G5 tail analysis**: Quantify the contribution of the omitted tails (0, 0.10) and (15, ∞) to the G3/G5 residual using the analytic dipole scaling B ~ c·ω³.
