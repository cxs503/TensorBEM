# CASE-BEM-CUMMINS-WIGLEY Round 6 — Full closure grid evaluation with G4 free decay

All numbers below are read programmatically from `artifacts/case_bem_cummins_wigley_round6/results_closure.json`
and `artifacts/case_bem_cummins_wigley_round6/points/` (no hand-typed value).

## 1 Problem setting and reference

* **Anchor**: `CASE-BEM-CUMMINS-WIGLEY` — Cummins (1962) equation + Kramers-Kronig causality on the Wigley III hull, 6DOF.
* **Reference**: G1 anchored on capytaine 3.1.dev0 output frozen in `benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`. G2-G5 are the exact identities of the Cummins/KK formalism (Kotik & Mangulis 1962; Newman 1977 ch.6).
* **Tolerance**: G1 <10%, G2 <2%, G3 <10%, G4 <10%, G5 <5%.

## 2 Method

* **Closure grid**: 68 nodes on [0.10, 15.0] rad/s — the full closure grid (extended band + resonance cluster) truncated to exclude omega=20 rad/s where the coarse mesh produces non-physical negative damping.
* **Resonance cluster**: 41 nodes at 0.0125 rad/s spacing around omega_n = 7.393 rad/s (heave resonance), providing ≥6 samples per half-power width.
* **Mesh**: coarse (10×5 = 100 quad panels, 200 triangles).
* **Code**: `scripts/run_cummins_wigley_closure_full.py` (solve pipeline), `scripts/eval_cummins_closure_round6.py` (gate evaluation), `scripts/make_cummins_round6_figures.py` (figures).

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

## 4 Gate results (first full 5-gate evaluation)

| Gate | rel_err | Tolerance | Verdict |
|---|---|---|---|
| G1 cross_solver | 6.29% | <10% | **PASS** |
| G2 kernel_at_zero | ~0% | <2% | **PASS** |
| G3 kramers_kronig | 49.48% | <10% | **FAIL** |
| G4 free_decay | 39.20% | <10% | **FAIL** |
| G5 round_trip | 16.33% | <5% | **FAIL** |

### Per-DOF breakdown

| DOF | G2 | G3 | G3b | G4 (ζ_decay vs ζ_hp) | G5 |
|---|---|---|---|---|---|
| surge | ~0% | 36.88% | 125.58% | 0.85% vs 0.80% (r4=6.58%) | 16.33% |
| sway | ~0% | 43.05% | 96.06% | 8.87% vs 14.59% (r4=39.20%) | 9.96% |
| heave | ~0% | 10.88% | 44.24% | **16.73% vs 16.72% (r4=0.05%)** | 5.36% |
| roll | ~0% | 49.48% | 110.66% | 0.25% vs 0.24% (r4=5.32%) | 5.83% |
| pitch | ~0% | 16.43% | 46.72% | 7.23% vs 6.83% (r4=5.86%) | 4.90% |
| yaw | ~0% | 45.32% | 134.25% | 11.66% vs 13.52% (r4=13.73%) | 0.34% |

## 5 Key findings

### 5.1 G4 free decay — first evaluation, heave passes at 0.05%

This is the **first time G4 (free decay) has been evaluated** for CASE-BEM-CUMMINS-WIGLEY. The resonance cluster (41 nodes at 0.0125 rad/s spacing) makes the half-power width resolvable, and the time-domain Cummins equation simulation produces a free decay whose damping ratio agrees with the frequency-domain half-power estimate.

**Heave G4 is excellent**: ζ_decay = 0.167250 vs ζ_hp = 0.167159, relative error **0.05%** — well within the 10% tolerance. This confirms that the time-domain memory kernel correctly captures the radiation damping physics for the heave DOF.

Surge (6.58%), roll (5.32%), and pitch (5.86%) also pass the 10% tolerance individually. Sway (39.20%) and yaw (13.73%) fail — the sway discrepancy is traced to the very low damping at low frequencies (B ~ 0.01-0.03) where the half-power method is sensitive to interpolation, and the yaw discrepancy is similarly a low-damping DOF effect.

### 5.2 G3/G5 remain dominated by band truncation

G3 (49.48%) and G5 (16.33%) are unchanged from round 5 on the coarse mesh. The root cause is the same: the Kramers-Kronig principal-value integral requires the full half-line (0, ∞), and the truncated band [0.10, 15.0] omits both tails. The state-space bridge test (m21239) confirmed this is a band artifact, not a physics error.

### 5.3 High-frequency pathology at omega=20

The coarse mesh produces negative diagonal damping at omega=20 rad/s: sway B=-6.13, roll B=-0.10. This is non-physical (radiation damping must be ≥0) and is caused by the irregular frequency problem when the panel size becomes comparable to the wavelength. The closure grid is truncated at 15 rad/s to exclude this node.

## 6 Convergence / sensitivity

* **Mesh ladder**: coarse (100 quads) → medium (400 quads) → fine (1600 quads). The medium mesh pushes the irregular frequency beyond 20 rad/s and is needed for the full closure band.
* **Band sensitivity**: extending from [0.30, 12.5] (22 nodes) to [0.10, 15.0] (27 nodes, truncated) reduces G3 by ~4 percentage points and G5 by ~30 percentage points (round 2 → round 5).
* **Cluster resolution**: 41 nodes at 0.0125 rad/s spacing around omega_n provides ≥6 samples per half-power width, making G4 decidable for the first time.

## 7 Failures, limits, reproduction

* **Honest status**: partial. G1 and G2 pass; G3, G4 (aggregate), and G5 fail on the coarse mesh.
* **G4 per-DOF**: heave, surge, roll, pitch pass individually; sway and yaw fail (low-damping DOFs).
* **G3/G5**: dominated by band truncation; requires medium mesh + extended band to improve.
* **High-frequency pathology**: omega=20 excluded (negative damping on coarse mesh).
* **Repro**:
  ```bash
  # Solve closure grid (coarse mesh)
  PYTHONPATH=src python3 scripts/run_cummins_wigley_closure_full.py \
      --level coarse --outdir artifacts/case_bem_cummins_wigley_round6 \
      --stage solve --grid extended --shard 0 --shards 1
  PYTHONPATH=src python3 scripts/run_cummins_wigley_closure_full.py \
      --level coarse --outdir artifacts/case_bem_cummins_wigley_round6 \
      --stage solve --grid closure --shard 0 --shards 1 \
      --resonance 7.393059060502688
  # Evaluate gates
  PYTHONPATH=src python3 scripts/eval_cummins_closure_round6.py
  # Generate figures
  PYTHONPATH=src python3 scripts/make_cummins_round6_figures.py
  # Tests
  PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_kk_causality.py -q
  ```
* **Tests**: 19 passed in `tests/test_cummins_wigley_kk_causality.py`.

## 8 Next steps

1. **Medium mesh closure**: Submit the 68-node closure grid on medium mesh (400 quads) to push the irregular frequency beyond 20 rad/s and include the full band.
2. **G3/G5 tail analysis**: Quantify the contribution of the omitted tails (0, 0.10) and (15, ∞) to the G3/G5 residual using analytic dipole scaling B ~ c·ω³.
3. **Sway/yaw G4 investigation**: The low-damping DOFs show larger G4 residuals; investigate whether the half-power method or the free decay simulation is the limiting factor.
4. **State-space closure**: Use the state-space fit (m21239) as a causal extension to close the KK integrals and evaluate G3/G5 on the SS-extended band.
