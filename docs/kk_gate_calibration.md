# KK gate calibration: what the Cummins/Wigley causality gate can actually detect

**Case**: `CASE-BEM-CUMMINS-WIGLEY` (Wigley hull, 6 DOF radiation, band 0.2–12 rad/s)
**Module**: `src/tensorbem/benchmarks/kk_gate_calibration.py`
**Generator**: `scripts/calibrate_kk_gate.py`
**Artifacts**: `benchmarks/verified/casepool_bem_cummins_wigley/kk_gate_calibration.{json,md}`,
`kk_gate_scale_audit.json`, figure `docs/figures/kk_gate_calibration.png`
**Tests**: `tests/test_kk_gate_calibration.py` (12 tests)

## 1 The open debt this closes

The case publishes 16 gates. Gate **C3** judges causality by reconstructing the
added mass from the damping sweep (`kk_added_mass_closure`) and requiring
`max|A_measured − A_KK| / max|A| ≤ 10 %`; gate **C6b** is the negative control for
that gate — the damping shape is corrupted (`B × 1.3` above 1 rad/s) and the
closure must then report a residual *above* the bar. C6b passes on the committed
sweep (surge, 20.6 %), but the artifact carries the label

> `"pass, not certified: no sample has a truncation budget below 5% …"`

That label is about the *data*, not the detector: at 12 rad/s no DOF's damping has
entered algebraic decay (`measured tail exponent 0.28–1.62`), so the omitted
`s > s_max` integral is unbounded and the corrupted residual cannot be attributed
to the corruption from band-limited data alone. The extended-band (0.2–20 rad/s)
sweep that would fix it is still in the cluster queue.

**The detector's detection power is a separate question and does not need a wider
band.** This work measures it by error injection on a pair whose answer is known
exactly.

## 2 Method

Reference pair (exact, closed form):

```
K(s) = -8 e^{-2s} + 32 e^{-8s}                 (causal memory kernel)
B(ω) = 240 ω² / ((ω²+4)(ω²+64))                > 0, hump at ω = 4 rad/s, B ~ ω² as ω→0
A(ω) = A_inf + (384 - 24 ω²)/((ω²+4)(ω²+64))   A_inf = 3
```

The KK relations hold for this pair *identically* (`tests/test_kk_gate_calibration.py`
re-derives `B` from `∫K(s)cos(ωs)ds` numerically), so any residual the estimator
reports on it is the estimator's own. The grid is the case's grid
(0.2–12 rad/s, 41 log nodes), and the power-tail fit is rejected exactly as it is
on the case sweep (`p = 2 < 2.05`), so the estimator runs in the same
configuration.

Four injection channels, `δ` swept, residual recorded in two normalisations:

| channel | injection | what it tests |
| --- | --- | --- |
| `damping_step` | `B(1+δ)` above 1 rad/s | **verbatim the C6b corruption** |
| `damping_bump` | `B(1+δw)`, localised in log ω | blind spot to *local* errors |
| `added_mass_bump` | `A + δ·scale·w` (one-signed) | the channel the gate measures; exposes `A_inf` absorption |
| `added_mass_bump_zero_mean` | same, zero-mean `w` | the best case (no absorption) |

## 3 Results (generated, zero hand-typed numbers)

### 3.1 Estimator bias and sensitivity

| quantity | reported statistic `|ΔA|/max|A|` | amplitude statistic `|ΔA|/max|A−A_inf|` |
| --- | --- | --- |
| floor (exact pair) | 0.57 % | 1.75 % |
| damping-step slope | 0.174 | 0.402 |
| `δ` that crosses the 10 % bar, damping step | **0.565** | **0.192** |
| `δ` that crosses the bar, added-mass bump | 0.511 | 0.158 |
| `δ` that crosses the bar, zero-mean bump | 0.313 | 0.104 |
| `δ` that crosses the bar, local damping bump | 0.799 | 0.282 |
| residual at the step C6b injects (`δ = 0.30`) | **5.4 %** | **14.8 %** |

Read: the gate **under-reports** an injected shape error by roughly 2.6–5.7×
depending on channel; a 30 % band-wide damping corruption produces 5.4 % of
`max|A|` on the reference pair — *below* the bar — while clearing it (14.8 %) on
the invariant statistic. So the fact that the case's corrupted sweep reported
20.6 % on surge is not a marginal detection: it is far above what this
calibration predicts for the same injection on a reference body.

### 3.2 The blind spot (the strongest negative result here)

`kk_gate_calibration.json → certificate.localisation`:

| log-width | δ = 0.3 | δ = 0.5 | δ = 1.0 | δ = 2.0 |
| --- | --- | --- | --- | --- |
| 0.15 | 2.48 % | 4.22 % | **8.55 % (passes)** | 17.22 % (fails) |
| 0.25 | 2.81 % | 4.77 % | **9.65 % (passes)** | 19.42 % (fails) |
| 0.40 | 3.08 % | 5.17 % | 10.41 % (fails) | 20.89 % (fails) |
| 0.60 | 3.71 % | 6.24 % | 12.54 % (fails) | 25.15 % (fails) |
| 1.00 | 4.44 % | 7.39 % | 14.85 % (fails) | 29.76 % (fails) |

The largest corruption that still **passes** C3 is a **×2.0 damping error confined
to `d(log ω) = 0.15`**. C3 therefore certifies "the damping is not wrong over a
wide band", not "the damping is right": a localised error — an irregular-frequency
spike at one ω, a single-frequency outlier — is integrated away. This is the
honest reading of the case's own remark that "the corruption has to be in shape".

### 3.3 Normalisation audit of the committed verdict (read-only)

`kk_gate_scale_audit.json`, computed from the **unmodified** `case_result.json`:

| gate | DOF | dilution `max|A−A_inf|/max|A|` | residual as declared | residual (amplitude) |
| --- | --- | --- | --- | --- |
| C3 | surge | 0.716 | 2.62 % | 3.66 % |
| C3 | roll | 0.331 | 0.72 % | 2.19 % |
| C3 | yaw | 0.593 | 3.60 % | 6.07 % |
| C3a | sway | 0.615 | 7.62 % | **12.40 %** |
| C3a | yaw | 0.646 | 8.85 % | **13.71 %** |
| C3a | roll | 0.348 | 2.45 % | 7.04 % |
| C6b | surge | 0.838 | 20.64 % | 24.62 % |

The declared bar is a fraction of `max|A|`, which contains the arbitrary
instantaneous offset `A_inf`; the same 10 % is therefore **12–30 % of the
added-mass amplitude**, depending on DOF. On the invariant statistic the anchored
closure **C3a** exceeds 10 % on sway (12.4 %) and yaw (13.7 %) although it passes
as declared. Nothing here changes the pre-declared criterion or the published
verdict — the point is that C3a's "PASS at 8.85 %, margin 1.15 pp" is
*convention-dependent*, and the margin is negative under the scale-free reading.

## 4 What this certifies and what it does not

* **Certifies (gate `C6b_cal`, `passed = true`)**: on an exact pair the estimator's
  bias is 0.57 % of `max|A|` (5.7 % of the bar); the damping-step response is
  strictly increasing for `δ ≥ 0.05` with positive slope; the injected size that
  crosses the bar on the invariant statistic (0.192) is below the 0.30 step C6b
  injects, i.e. the control fails for a reason with a measurable margin.
* **Does not certify the sweep.** surge/sway/yaw keep `n_decidable = 0` at 12 rad/s.
  That is a property of the un-solved high band; gates C8/C9 and `C3a_extended`
  stay un-evaluated until the 20 rad/s artifact lands.
* **Does not change** any pre-declared criterion, the 16-gate verdict, or
  `case_result.json`. Adopting `C6b_cal` as gate 17 requires regenerating
  `case_result.json` **and** `CASE.md` atomically (the invariant locked by
  `test_published_case_md_is_the_render_of_the_published_json`), which is the
  recommended follow-up, not something to slip into an audit.

## 5 Follow-ups suggested by these numbers

1. Add `C6b_cal` to the case as gate 17 (atomic JSON + CASE.md regeneration).
2. Re-state C3/C3a in the amplitude normalisation, or publish both — the verdict
   currently depends on an arbitrary offset convention.
3. A *localised*-corruption negative control (`damping_bump`, width 0.15, δ = 1.0)
   belongs next to C6b: it is a corruption the current gate provably accepts.
4. If the extended-band artifact lands, re-run this calibration on it: the
   detection thresholds should tighten once the tail model becomes admissible.