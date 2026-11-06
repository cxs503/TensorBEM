# 3-D finite-depth kernel: long-wave collapse, deep-water collapse, and a `k0*h` regime map

**Mission** m3409 (self-start, TensorBEM) · branch `agent/m3409` · kernel `tensorbem.finite_depth3d`
**Deliverables** `src/tensorbem/finite_depth3d.py` (fix), `src/tensorbem/benchmarks/finite_depth3d_regime.py`,
`tests/test_finite_depth3d_regime.py` (11 gates), `docs/m3409/outputs/finite_depth3d_regime.json` (ledger),
`docs/m3409/figures/fig_finite_depth3d_regime.png`.

EDIT_WHITELIST: src/tensorbem/finite_depth3d.py, src/tensorbem/benchmarks/finite_depth3d_regime.py,
tests/test_finite_depth3d_regime.py, docs/m3409/

---

## 1. Case (physics and why this check exists)

`green_function_3d(targets, sources, nu, depth)` is the John (1950) eigenseries of the
three-dimensional free-surface Green function in constant depth `h`,

```
G(x,xi) = (1/2pi) sum_{n>=1} phi_n(z) phi_n(zeta)/N_n K_0(mu_n R)
        + (i/4) phi_0(z) phi_0(zeta)/N_0 H_0^{(1)}(k_0 R),
phi_0 = cosh(k_0(z+h)),  k_0 tanh(k_0 h) = nu = omega^2/g,
N_0 = (h/2)(1 + sinh(2 k_0 h)/(2 k_0 h)).
```

The module documents two limits: `k0*h -> inf` must reproduce the deep-water Kelvin kernel of
`tensorbem.free_surface3d`, and `k0*h -> 0` must degenerate to long-wave (shallow-water) theory.
The shipped gate only walked **one** path to the first limit (fixed `nu`, growing `h`) and stopped at
`k0*h ~ 77`, and **no** test touched the second. A verification pass over the merged kernel found:

* **defect**: the wave term was formed as `cosh(k0*(z+h))*cosh(k0*(zeta+h))/N_0` directly.
  For `k0*h >~ 710` `cosh` overflows and `N_0` is `inf`, so the kernel returned **`nan`** (and
  `dG/dz` with it) — exactly in the deep-water regime the docstring promises to reproduce, and
  silently poisonable inside a panel solver. The two-dimensional sibling
  `tensorbem.finite_depth` already avoided this via `_scaled_mode_factors`
  ("this is what makes depth = 800 testable"); the 3-D module had no such factorisation.
* **missing verification**: the long-wave collapse had no gate, and no document stated where the
  deep-water surrogate stops being admissible.

Test geometry: source and target at mid-depth `z = zeta = -h/2`, horizontal separation
`R = {5,10} h` (similarity form: every ratio below depends on `k0*h` alone, confirmed across
`h = 1, 5, 50` m).

## 2. Method

1. Added `_propagating_factors_3d(k0, h)` returning `(u, v, w)` with
   `cosh(k0(z+h)) = exp(k0 h) u(z)`, `u' = k0 v`, `N_0 = exp(2 k0 h) w`,
   `w = -expm1(-4k0 h)/(8 k0) + (h/2) exp(-2 k0 h)` (cancellation-free). The wave term uses
   `u/sqrt(w)` products, the `z`-gradient uses `k0 v u / w`, and `wave_amplitude_coefficient`
   became `u(z) exp(-k0 h)/w`. In the safe regime the new expressions are algebraically identical.
2. `benchmarks/finite_depth3d_regime.py` recomputes, from the library, (i) the long-wave deviation
   against `(i/(4h)) H_0^{(1)}(k0 R)` and against its closed-form leading error `(k0 h)^2/12`,
   (ii) the deep-water deviation on a fixed-`nu` depth ladder up to `k0*h = 4e4`,
   (iii) a `k0*h` similarity map for surrogate selection, and writes a JSON ledger with a verdict.
3. `tests/test_finite_depth3d_regime.py` pins the ledger: order of convergence, overflow regression,
   gradient/finite-difference consistency at `k0*h = 1e4`, geometry independence, verdict.

## 3. Results (all numbers from the ledger, none transcribed)

**Long-wave collapse** (`h = 1 m`, mid-depth, `R/h = 5`; `R/h = 10` identical to 4 digits):

| `k0 h` | measured \|G/G_lw − 1\| | `(k0 h)²/12` | ratio |
|---|---|---|---|
| 0.02 | 3.334e-05 | 3.333e-05 | 1.0001 |
| 0.05 | 2.084e-04 | 2.083e-04 | 1.0005 |
| 0.10 | 8.351e-04 | 8.333e-04 | 1.0022 |
| 0.20 | 3.362e-03 | 3.333e-03 | 1.0086 |
| 0.40 | 1.377e-02 | 1.333e-02 | 1.0329 |
| 0.80 | 5.928e-02 | 5.333e-02 | 1.1115 |

Observed order: 2.000, 2.002, 2.009, 2.034, 2.106 (quadratic to within 0.5 % up to `k0 h = 0.4`).
Max relative error of the closed form over `k0 h <= 0.2`: **8.55e-03 (0.86 %)**.

**Deep-water collapse** (fixed `nu = 50 1/m`, `R = 1 m`, mid-depth):

| `h` (m) | `k0 h` | \|G/G_deep − 1\| | finite? |
|---|---|---|---|
| 1 | 50 | 3.653e-01 | yes |
| 2 | 100 | 8.225e-02 | yes |
| 5 | 250 | 5.900e-03 | yes |
| 20 | 1 000 | 9.334e-05 | yes (pre-fix: **nan**) |
| 100 | 5 000 | 7.459e-07 | yes (pre-fix: **nan**) |
| 800 | 40 000 | 1.455e-09 | yes (pre-fix: **nan**) |

1 %-crossover for the deep-water surrogate on this ladder: `k0 h >= 250` (0.59 %); monotic above.

**`k0*h` similarity map** (identical for `h = 1, 5, 50` m):

| `k0 h` | 0.05 | 0.1 | 0.2 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| dev vs long wave | 2.08e-04 | 8.35e-04 | 3.36e-03 | 2.19e-02 | 9.61e-02 | 3.91e-01 | 6.80e-01 |
| dev vs deep water | 9.50e+00 | 6.64e+00 | 4.18e+00 | 2.04e+00 | 1.19e+00 | 3.76e-01 | 9.69e-02 |

Long-wave surrogate admissible within 1 % for `k0 h <= 0.2`; deep-water surrogate needs
`k0 h >= 250` on the fixed-`nu` ladder (see §5). Between those the modal kernel must be used: at
`k0 h = 1` both cheap surrogates are off by 12 %–119 %.

## 4. Figures

![regime](figures/fig_finite_depth3d_regime.png)

*Left*: measured long-wave deviation vs `k0 h` for `R/h = 5` and 10, collapsing onto the closed-form
`(k0 h)^2/12` line (dashed). *Right*: deep-water deviation vs `k0 h` on the fixed-`nu` ladder; the
shaded band is the `k0 h > 700` range where the pre-fix kernel returned `nan`.

## 5. Attention points / limitations

* The **defect was real but benign for the documented numbers**: at `k0*h = 77` (the old gate) no
  overflow occurs, which is why the suite was green. Any caller pushing beyond `k0*h ~ 700` — a
  deep-water site, or `h = 800 m` in a validation study — got `nan`, not an exception.
* `wave_amplitude_coefficient` for `k0 h >~ 700` now returns **0.0**, not a small positive number:
  `phi_0/N_0 ~ 4 k0 exp(k0(z-h))` is genuinely below the double range there. That is an
  underflow, documented in the test.
* The deep-water collapse closes slowly (`k0 h >= 250` for 1 %) because with fixed `nu` and growing
  `h` the source sinks many wavelengths deep, so the finite-depth wave term itself decays like
  `exp(k0 z)`. This is a **geometry statement**, not a defect: the surrogate error is a function of
  `k0*h` alone in similarity form (§3 map).
* The similarity map stops at `k0 h = 3`. Beyond it at fixed `h` and `k0 R >~ 50` the deep-water
  kernel's oscillatory quadrature departs from the converged finite-depth series by `O(1)`
  (observed `|G/G_deep-1| = 8.5e-1` at `k0 h = 10`, `k0 R = 50`), while the *correct* deep-water
  collapse is confirmed by the fixed-`nu` ladder (1.5e-09 at `k0 h = 4e4`). Which of the two
  kernels is losing accuracy at large `k0 R` is **now resolved** by m4776
  (`docs/finite_depth3d_deepwater_arbitration.md`): the deep-water kernel reproduces an independent
  principal-value evaluation of its own Havelock integral to `3.4e-6`, and the eigenseries is
  mode-converged to `4e-17`, so the 85 % is a genuine finite-depth effect amplified by the
  cancellation number `kappa = 142`; the same difference is 0.60 % of the local Rankine scale.
  The `k0 R` sweep is implemented in `benchmarks/finite_depth3d_deepwater_arbitration.py`.
* Constant depth, linearised potential flow: no sloping bottom, current, or nonlinearity
  (unchanged from the module contract).

## 6. Error verdict

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| long-wave deviation at `k0 h = 0.1` | `(k0 h)²/12` = 8.333e-04 | 8.351e-04 (+0.22 %) | **yes** |
| closed form `(k0 h)²/12` over `k0 h <= 0.2` | measured deviation | max 0.86 % | **yes** |
| convergence order, `k0 h` in [0.02, 0.4] | 2 | 2.000–2.034 | **yes** |
| deep-water kernel at `k0 h = 250` | `free_surface3d` | 0.59 % | **yes** |
| deep-water kernel at `k0 h >= 1000` | `free_surface3d` | 9.3e-05 … 1.5e-09 | **yes** |
| kernel vs analytic `dG/dz` at `k0 h = 1e4` | central difference | 1e-6 gate, passed | **yes** |
| deep-water kernel at `k0 h = 10`, `k0 R = 50` | finite-depth series | 85 % of `|G|` = 0.60 % of `1/(4 pi r)`; kernel itself certified to 3.4e-6 | **resolved (m4776)** |

Regression status: `tests/test_finite_depth3d_regime.py` 11 passed;
`tests/test_finite_depth3d.py` + `tests/test_finite_depth.py` 40 passed (pre-existing gates unchanged).