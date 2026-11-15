# m21275 — CASE-BEM-CUMMINS-WIGLEY: attributing the C5 free-decay failure

Anchor: `CASE-BEM-CUMMINS-WIGLEY` (算例池 · Wigley 船 Cummins 记忆核, 6DOF), ref = Cummins
equation + Kramers–Kronig causality, case tolerance < 10 %. Status at entry: `open`;
the case scores 9/10 gates with **C5 FAIL** (`zeta_decay = 0.13787` vs
`zeta_fd = 0.12875`, 7.09 % against the module's internal 5 % gate).

This round does not re-run the case. It answers the one question the case card
cannot: **which of the two routes in C5 is wrong?** All numbers below are written
by `scripts/run_cummins_wigley_causal_decay.py` to
`benchmarks/verified/casepool_bem_cummins_wigley/causal_decay_audit.json`
(machine-generated, nothing typed here by hand).

## 1. Setting

| item | value (from the frozen case artifact) |
|---|---|
| hull | mathematical Wigley, L = 3.0 m, B = 0.3 m, T = 0.1875 m |
| mesh | coarse 192 panels / medium 480 panels |
| sweep | 41 nodes, ω ∈ [0.2, 12.0] rad/s, ⟨A(ω), B(ω)⟩ 6×6 |
| body | displaced mass 76.875 kg, C₃₃ = 6016.01 N/m, A∞(KK closure) = 50.930 kg |
| heave ω_n (legacy decay record) | 7.8077 rad/s |

Heave is the only DOF the case scores C5 on; roll/pitch would need the C₄₄/C₅₅
restoring terms the case module does not build — recorded below as open debt.

## 2. Method — an estimator neither hypothesis uses

Two explanations fit the 7.09 %:

* **H-reference** — the frequency-domain surrogate `B(ω_n)/(2(M+A(ω_n))ω_n)` is
  biased (linear interpolation of a 41-node curve at ω_n, plus the single-DOF
  energy identity applied to a memory system);
* **H-route** — the time-domain route is biased, because
  `time_domain.radiation_memory_kernel` hard-truncates the sweep at
  ω_max = 12 rad/s, so the kernel it builds is not the kernel of any causal
  system.

The arbiter is the **exact modal damping of the causal model**: continue `B(ω)`
off the band with a rational (state-space) fit — causal by construction, because
`K(τ) = Σ_k c_k e^{−α_k τ}` is a sum of decaying exponentials — and read ζ from
the complex root of

```
(M + A∞) s² + s K̂(s) + C = 0,      K̂(s) = D_ss + Σ_k c_k/(s + α_k)
```

The root is exact for the model that the time-domain route integrates, so the
comparison has no approximation left in the *reference*.

Estimator order is not chosen by hand: the fit order is scanned 4…10 and the
ladder takes the smallest order meeting both an in-band fit budget (≤ 3 %) and a
Kramers–Kronig added-mass budget (≤ 5 %). Order 12 was measured to reproduce the
band to 1.1 % but to break the added-mass budget (17.6 %), so 10 is the cap.

## 3. Gates (predeclared, machine-decided)

| gate | quantity | tolerance | medium | coarse |
|---|---|---|---|---|
| D0 estimator validation | complex root vs `B/(2√((M+A)C))`, constant damping | ≤ 1e-6 | 2.4e-16 PASS | 2.4e-16 PASS |
| D0b cubic memory root | Newton vs `numpy.roots`, 1-exponential kernel | ≤ 1e-8 | 2.3e-15 PASS | 2.3e-15 PASS |
| D1 fit in band | max\|B_fit−B_bem\|/max\|B\| | ≤ 3 % | 1.70 % PASS | 2.86 % PASS |
| D1 KK added mass | max\|A_fit−A_bem\|/max\|A\| | ≤ 5 % | 3.63 % PASS | **6.58 % FAIL** |
| D2 kernel completeness | \|K(25s)\|/\|K(0)\|, causal kernel | ≤ 1e-3 | 2.8e-08 PASS | PASS |
| D3 causal decay (replacement for C5) | \|ζ_decay−ζ_exact\|/ζ_exact | ≤ 5 % | 3.63 % PASS | 3.43 % PASS |
| D4 legacy decay bias | \|ζ_decay_legacy−ζ_exact\|/ζ_exact | ≤ 5 % | **6.39 % FAIL** | 0.15 % PASS |
| D5 band robustness (defect demo, ≥) | max/min ζ over band tops 8…12 | ≥ 1 (10×) | 65.6 PASS | 65.3 PASS |

## 4. Results

### 4.1 Four estimators of the heave damping ratio (medium mesh)

| estimator | ζ | distance to the exact root |
|---|---|---|
| legacy free decay (truncated kernel, `τ_max` = 25 s) | 0.13787 | +6.39 % |
| legacy frequency-domain reference (interp. A, B at ω_n) | 0.12875 | −0.65 % |
| **causal exact modal root** (10 poles, residual 1.8e-13) | **0.12959** | — |
| causal free decay (rational kernel, same estimator) | 0.12489 | −3.63 % |

Order scan of the exact root over the admissible orders 6…10:
0.13098, 0.13023, 0.13027, 0.13071, 0.12959 → spread 0.0014 (1.1 %), i.e. the
causal answer is **ζ = 0.1296 ± 0.0007** on the medium mesh, 0.1333 on the coarse.

### 4.2 The truncated kernel is not a kernel (D2)

| τ [s] | 0 | 1 | 12 | 25 |
|---|---|---|---|---|
| legacy `K(τ)` [N s/m] | 1379.16 | −122.53 | −2.17 | **−1.95** |
| causal `K(τ)` [N s/m] | −1.7e+04 (short-time, unconstrained) | — | — | −4.8e-04 |

The legacy kernel does not decay: at 25 s it still oscillates with 1.4e-3 of its
t = 0 magnitude, and its sign is wrong (`K(25) = −1.95`).  A hard cutoff leaves a
`sin(ω_max τ)/τ` residue that never dies, so **every** decay-rate estimator run on
that kernel measures the cutoff as much as the hull.

### 4.3 Band fragility (D5) — the decisive measurement

Damping ratio of the *same hull* from the *same estimator*, varying only the top
of the truncated sweep:

| ω_max [rad/s] | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|
| ζ (legacy decay) | 0.00210 | 0.00901 | 0.11360 | 0.13251 | 0.13787 |
| cusps in the record | 51 | 56 | 5 | 5 | 5 |

A factor **65.6** between ω_max = 8 and 12 rad/s, with the record degenerating to
a nearly undamped oscillation (51 peaks, ζ ≈ 0.002) when the cutoff sits close to
the resonance. The production value 0.13787 is therefore a point on a
cutoff-driven curve, not a converged physical damping ratio. Cosine-tapering the
top 10/20/30 % of the band (a common palliative) does **not** fix it: 0.1345 /
0.1392 / 0.1407 — still 4–9 % above the causal root, because a taper removes the
discontinuity but not the missing high-frequency tail.

### 4.4 Verdict, including where it does not hold

* On the **medium** mesh the attribution is **H-route**: the truncated-kernel
  decay is 6.39 % off the exact root while the frequency-domain reference is
  0.65 % off. The 7.09 % C5 gap is carried by the time-domain route.
* On the **coarse** mesh it **flips** to **H-reference**: legacy decay 0.15 % off,
  the interpolated reference 2.70 % off.
* Therefore the honest reading is not "route X is wrong" but **"a 5 % agreement
  between these two estimators is not decidable on this case"**: both carry an
  error of the order of the tolerance itself, with a sign that follows the mesh.
  What *is* decidable is (a) the cutoff dependence of the legacy route (D5,
  factor 65, both meshes) and (b) the internal consistency of the causal route
  (D3, 3.63 % / 3.43 %).

### 4.5 Recommended remediation (implemented, ready to wire)

Replace gate C5's *route*, not its tolerance:

1. build the memory kernel from the rational continuation
   (`K(τ) = Σ_k c_k e^{−α_k τ}`) instead of the hard-truncated sweep;
2. score the decay ζ against the **exact complex root of the same causal model**
   (gate D3 here) rather than against an interpolated energy identity;
3. keep the legacy C5 exactly as it is, as the recorded comparison.

`cummins_wigley_causal_decay.causal_free_decay` + `complex_modal_root` already
reproduce the C5 JSON schema; wiring them into
`case_bem_cummins_wigley.free_decay_gate` needs a case re-run (the case driver
re-solves the BEM sweep), which is deliberately out of scope here so that
`case_result.json` in the tree never disagrees with its own run.

## 5. Error judgement (vs the analytic reference inside the model)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| estimator on constant-damping oscillator | closed form | 2.4e-16 | yes |
| estimator on cubic-memory oscillator | `numpy.roots` | 2.3e-15 | yes |
| causal decay ζ (medium) | exact modal root | 3.63 % | **no** |
| causal decay ζ (coarse) | exact modal root | 3.43 % | **no** |
| legacy reference ζ (medium) | exact modal root | 0.65 % | yes |
| legacy reference ζ (coarse) | exact modal root | 2.70 % | yes |
| legacy decay ζ (medium) | exact modal root | 6.39 % | no |
| in-band B reproduction (medium) | BEM sweep | 1.70 % | yes |

The anchor's own < 10 % tolerance is met by every route except the legacy decay
on medium (6.39 % — still inside 10 %). No independent physics anchor is added by
this round: the causal continuation is fitted to the same BEM sweep, so it can
attribute error between routes but cannot certify the sweep itself. R1
(capytaine) remains the only physics anchor of the case.

## 6. Pitfalls met

* `radiation_memory_kernel` enforces Nyquist (`dt ≤ π/ω_max`) and uniform τ; a
  τ-grid laid out at 1 s spacing raises instead of silently aliasing — good, and
  it shaped the kernel-tail experiment.
* The exact-root solver needs *damped* Newton: a bare step diverges when the
  initial guess sits on a steep part of `K̂(s)`.
* Fit order must be capped: order 12 gives the best in-band fit (1.06 %) and the
  worst added-mass consistency (17.6 %) — the pole ladder needs **both** budgets.
* `K(0)` of a rational fit is meaningless (values −1.7e+04 … −5.7e+04 across
  orders): the short-time kernel is unconstrained by a sweep that starts at
  0.2 rad/s. Only in-band quantities are quotable from the fit.

## 7. Open debt (not done here)

* roll and pitch have no restoring terms in the case module, so C5 stays heave-only;
* the high-frequency tail needed to make the truncated route causal-complete is
  not measured above 12 rad/s (the case's own KK closure already reports
  `tail valid = no` for all six DOF);
* wiring D3 into `free_decay_gate` needs a full case re-run (BEM sweep) so that
  `case_result.json` and `CASE.md` stay consistent.

## 8. Reproduction

```bash
PYTHONPATH=src python scripts/run_cummins_wigley_causal_decay.py \
  --levels coarse,medium \
  --out benchmarks/verified/casepool_bem_cummins_wigley/causal_decay_audit.json \
  --figures docs/case_bem_cummins_wigley_c5_attribution
PYTHONPATH=src python -m pytest tests/test_cummins_wigley_causal_decay.py -q
```

Figures (`docs/case_bem_cummins_wigley_c5_attribution/`):
`fig_kernel_tails.png` (kernel decay), `fig_band_fragility.png` (ζ vs cutoff),
`fig_decay_records.png` (both free-decay records), `fig_estimators.png`
(four estimators with the order-scan spread).