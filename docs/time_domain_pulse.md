# Time-domain impulse response of the radiation memory force (Cummins equation)

**Repository scope.** TensorBEM owns boundary-element formulations and their
verification. The frequency-domain legs already verify *coefficients* (added
mass, damping, excitation) against closed-form and published references. A
seakeeping code does not consume coefficients: it consumes the radiation memory
kernel of the Cummins equation. This document verifies that bridge — the map
from the tabulated `(omega, A, B)` of a BEM solve to the kernel `K(tau)` that a
time-domain simulation integrates — on one documented case, with the numbers
written to a machine-readable artifact.

* module: [`src/tensorbem/time_domain.py`](../src/tensorbem/time_domain.py)
* benchmark: [`src/tensorbem/benchmarks/time_domain_pulse.py`](../src/tensorbem/benchmarks/time_domain_pulse.py)
* tests: [`tests/test_time_domain.py`](../tests/test_time_domain.py) (14 tests, 5 negative)
* numbers: [`metrics.md`](metrics.md) rendered from
  [`time_domain_pulse.json`](time_domain_pulse.json) by
  [`render_metrics.py`](render_metrics.py) — nothing in the tables below is typed
  by hand
* figures: `fig_time_domain_round_trip.png`, `fig_time_domain_kernel.png`,
  `fig_time_domain_decay.png`

---

## 0. Reproduce

```
python -m tensorbem.benchmarks.time_domain_pulse --figure \
    --output artifacts/time_domain_pulse.json    # criteria + figures (~3 min)
python -m pytest tests/test_time_domain.py -q    # 14 tests, ~31 s
python docs/time_domain_pulse/render_metrics.py  # metrics.md from the JSON
```

The committed copy of the artifact and the figures lives next to this document
(`time_domain_pulse.json`, `fig_time_domain_*.png`); `artifacts/` is generated
and git-ignored.

---

## 1. Case, formulation and parameters

### 1.1 Physical set-up

A freely floating hemisphere of radius `a = 1 m` in deep water, zero forward
speed, single degree of freedom (heave). The hemisphere is the only geometry in
this repository for which the frequency-domain radiation coefficients are
available *without a panel mesh* (Hulme 1982, J. Fluid Mech. 121:443–463,
implemented as `tensorbem.analytic.hulme_hemisphere_radiation`). Using a
reference solution instead of a mesh is deliberate: it removes panel
discretisation error from the chain, so every residual below is a property of
the frequency-to-time mapping alone.

| parameter | value | how obtained |
|---|---|---|
| radius `a` | 1.0 m | given |
| density `rho` | 1025 kg/m³ | given |
| gravity `g` | 9.80665 m/s² | given |
| displaced volume `nabla` | 2.0944 m³ | `2/3 pi a^3` (closed form) |
| body mass `M` | 2146.75 kg | `rho nabla` (flotation) |
| waterplane area | 3.14159 m² | `pi a^2` (closed form) |
| heave restoring `C33` | 31578.7 N/m | `rho g A_wp` (closed form) |
| infinite-frequency added mass `A_inf` | 1073.38 kg | `0.5 rho nabla`; confirmed below by two independent routes |
| frequency sweep | 501 points, `omega` ∈ [0.14, 29.70] rad/s (`ka` ∈ [0.002, 89.95]) | Hulme, uniform in `omega` |
| damping peak | 1746.7 kg/s at `omega` = 2.741 rad/s | Hulme |

Two details of the set-up are themselves results, not choices:

* `A_inf = 0.5 rho nabla` is the `ka -> 90` asymptote of Hulme's added mass
  (`a33' = 0.49787` at `ka = 90`, rising as `1/ka`). It is *independently*
  recovered in §3 by inverting eq. (3) for `A_inf` at every frequency: the
  implied value is 1073.385 kg with a spread of 8.0e-4 relative. A wrong `A_inf`
  would show up as a slope in that spread.
* The sweep is uniform in **`omega`, not in `ka`**. `ka` is the natural
  parameter of Hulme's expansion, but eq. (1) integrates over `omega`: a sweep
  uniform in `ka` (the form used in the repository's earlier radiation studies)
  leaves the first `omega` interval spanning `0.14 -> 2.35` rad/s, the
  piecewise-linear interpolant of `B` is meaningless there, and the
  reconstructed added mass acquires a spurious low-frequency error of ~25% that
  has nothing to do with the transform. This was found during this work by
  running the round trip on the `ka`-uniform sweep and is now pinned by
  `test_hemisphere_round_trip_within_five_percent`.

### 1.2 Formulation

Cummins (1962) equation for one mode,

```
(M + A_inf) x''(t) + integral_0^t K(t - s) x'(s) ds + C x(t) = F(t),
```

with the memory kernel and its two inverse relations

```
K(tau)           = (2/pi) integral_0^inf B(omega) cos(omega tau) domega        (1)
B(omega)         = integral_0^inf K(tau) cos(omega tau) dtau                   (2)
A(omega) - A_inf = -(1/omega) integral_0^inf K(tau) sin(omega tau) dtau        (3)
```

Eqs. (1)–(3) are one statement in three places, and the verification is
organised around that: a kernel that reproduces `B` but not `A` (or whose
`A_inf` is inconsistent) makes the time-domain and frequency-domain descriptions
of the same body disagree — and that disagreement is invisible in either
description taken alone. The odd partner `omega (A - A_inf)` in eq. (3) is the
parity content: for a causal kernel, `B` is even and `omega (A - A_inf)` is odd
and they are Hilbert pairs (Kotik & Mangulis 1962; the same pairing that
`tensorbem.causality` tests as Kramers–Kronig consistency).

### 1.3 Numerical method (no shared parts between the legs)

| leg | method | grid |
|---|---|---|
| `B -> K` | exact per-interval integration of the piecewise-linear interpolant of `B` (closed form with `sinc`-stabilised cosine moments) | sweep grid, 501 nodes |
| `K -> B`, `K -> A` | composite Simpson | uniform `tau`, `dtau` = 0.01 s, `tau_max` = 53.14 s |
| time integration | classical RK4, trapezoidal memory convolution on the step grid, kernel pre-sampled on a `dt/2` grid | `dt` = 0.0025 s, 20 s |
| frequency-domain response | `kappa(omega) = C - (M + A(omega)) omega^2 + i omega B(omega)` | sweep grid |

The forward and inverse legs share no quadrature rule and no sample set, so a
residual between them measures the transform rather than the algebra. The
memory convolution is evaluated as written (`O(N^2)`), not replaced by a fitted
state-space model, so an error in `K` cannot be hidden inside a rational fit.

Band edges are handled explicitly and their size is reported with every number:

* below `omega_min` = 0.14 rad/s the sweep is extended with the long-wave model
  `B ~ c omega^3` (fitted on the lowest decade; for this body `c = 431.4`
  kg·s²/m³), contributing at most 6.7e-6 of the kernel peak;
* above `omega_max` = 29.7 rad/s the sweep is truncated, contributing at most
  1.0e-4 of the kernel peak (`B(omega_max)` = 0.021 kg/s).

Sampling contract (enforced, not advised): a kernel step must satisfy
`dtau <= pi/omega_max`, and a reconstruction must satisfy
`omega_query <= pi/dtau`; violations raise `ValueError`. A sweep with negative
damping raises `ValueError` — `B >= 0` is the passivity of radiation, and a
negative value would make `K` grow.

---

## 2. Criterion C1 — decay and symmetry of `K(tau)`

![kernel](fig_time_domain_kernel.png)

`K(0) = (2/pi) integral B domega = 3949.8 kg/s²` (matches the tabulated
integral to 1e-5 relative), and the kernel crosses zero at `tau = 0.48 s`. Its
envelope, fitted between 50% and 1% of the peak, decays at `1.584 /s`
(memory time scale 0.63 s); the maximum `|K|` beyond the far half of the record
is 1.1e-6 of the peak, i.e. the truncation at `tau_max` = 53.14 s is far beyond
the memory.

Symmetry is checked two ways:

* the even two-sided extension `K(-tau) = K(tau)` reconstructs a **purely real**
  spectrum: the imaginary part of `integral K exp(-i omega tau) dtau` is
  4.1e-18 relative to the real part (roundoff), and the extension residual is
  exactly zero;
* the **parity closure**: rebuilding the kernel from the even/odd spectrum pair,
  `(1/pi) integral [ B cos(omega tau) - omega (A - A_inf) sin(omega tau) ] domega`,
  must return the cosine transform of `B` alone. Measured residual: 0.96% of the
  kernel peak. That residual is dominated by the truncation of the odd partner,
  which decays only as `1/omega` (its tail is elementary,
  `int_W^inf (c/omega) sin(omega tau) domega = c [pi/2 - Si(W tau)]`, and is
  included). On an exact causal pair (`K = sum_j c_j exp(-a_j tau)`) the same
  closure is 0.38%, which is what makes the 0.96% readable as a truncation
  artefact rather than a transform error.

Also reported: the relative slope at the origin, `-7.8e-4`, within the
grid-scale bound `(omega_max dtau)^2 = 0.088` that a cosine transform must obey
(a sine transform would give a value of order `omega_max dtau`).

| check | value | criterion |
|---|---|---|
| decay rate positive | 1.5836 1/s | > 0 |
| kernel tail beyond `tau_max` | 1.13e-6 of peak | < 5% |
| even-extension residual | 0 (exact) | < 1e-12 |
| two-sided spectrum, Im/Re | 4.07e-18 | < 1e-6 |
| slope at origin (grid-scaled) | -7.81e-4 | < `(omega_max dtau)^2` = 0.088 |
| parity-closure residual | 9.63e-3 of peak | < 2% |

**C1 verdict: PASS.**

---

## 3. Criterion C2 — Fourier reciprocity inside the sampled band

![round trip](fig_time_domain_round_trip.png)

`B -> K -> (B, A)` with independent quadratures, over the band where the sweep
carries information. The band is set by the sampling, not by taste:
`[max(omega_min, 4 pi/tau_max), min(0.6 omega_max, pi/(2 dtau))]` =
`[0.236, 17.82]` rad/s. The low edge keeps at least two memory periods inside
the window (below it, eq. (3) divides by `omega` and the sine quadrature is cut
mid-period); the upper edge stays clear of the truncated band.

| quantity | value |
|---|---|
| round-trip `B` error / peak `B` | 1.27e-4 |
| round-trip `A` error / peak `A` | 2.37e-4 |
| `A_inf` implied by eq. (3), mean over the sweep | 1073.3846 kg |
| implied `A_inf` vs `0.5 rho nabla` = 1073.3775 kg | 6.6e-6 relative |
| implied `A_inf` spread across the sweep | 8.0e-4 relative |

The implied `A_inf` is the sharp part of this criterion: it is obtained by
inverting eq. (3) *without* assuming `A_inf`, so it is an independent
determination of the infinite-frequency added mass from the damping sweep alone.
Two independent routes (`ka -> 90` asymptote of Hulme's `A33`; inversion of the
kernel transform) agree to 6.6e-6, and the spread across the sweep — which would
tilt if `A_inf` were wrong — is 8.0e-4. The acceptance threshold is 5%; the
measured margin is a factor 200.

**C2 verdict: PASS (5% required, 2.37e-4 achieved).**

---

## 4. Criterion C3 — time domain versus frequency domain

![decay](fig_time_domain_decay.png)

The Cummins equation is integrated from rest (`z(0) = 0.05 m`, `zdot(0) = 0`)
with the same kernel, and the resulting free decay is compared with the
mechanical response function `1/kappa(omega)` built from the sweep alone.

| quantity | time domain | frequency domain | difference |
|---|---|---|---|
| period | 1.94548 s | 1.95927 s | 0.704% |
| decay rate | 0.30603 1/s | 0.30478 1/s (bandwidth `zeta_bw omega_n`) | 0.411% |
| amplitude (driven, at `omega_n`) | 1.90e-4 m/N | 1.90e-4 m/N | 0.306% |
| phase (driven, at `omega_n`) | -1.57087 rad | -1.57080 rad | 7.5e-5 rad |

Envelope fit quality: `R² = 0.999988` over 18 extrema spanning an amplitude
ratio of 160; the decay is exponential to the eye and to the fit.

**The interesting number is the one that does not agree.** The energy identity
`zeta = B(omega_n)/(2 (M + A(omega_n)) omega_n)` gives `zeta = 0.08348` and a
decay rate of 0.2686 1/s, which is **13.9% below** the measured 0.30603 1/s —
outside any acceptable band. The half-power bandwidth of the same response
function gives `zeta_bw = 0.09471`, i.e. 0.30478 1/s, which matches to 0.4%.
This is not a defect of the integration; it is the memory effect. The energy
identity evaluates `B` at a single frequency, whereas a decay test measures the
envelope of a response that occupies the whole half-power band (2.887–3.496
rad/s here), and `B` varies by a factor 1.6 across that band. A seakeeping code
that prints `zeta` from the single-frequency identity and a model test that
measures the decay envelope will disagree by ~14% on this body even when both
are right. The two estimates are therefore reported side by side rather than
collapsed into one, and only the bandwidth comparison is used as a criterion.

**C3 verdict: PASS (period < 1%, rate < 5%, driven amplitude < 2%, phase <
0.05 rad).**

---

## 5. Tests

`python -m pytest tests/test_time_domain.py -q` → **14 passed in 31 s**
(no network, no cluster, no panel solves; the pinned Hulme sweep is re-verified
against a fresh evaluation of the reference at three points).

| test | what it establishes |
|---|---|
| `test_kernel_reproduces_exponential_closed_form` | `B -> K` against the band-truncated closed form of `K = sum c exp(-a tau)` (adaptive quadrature), 1e-3; calibrates the transform before it judges a sweep |
| `test_round_trip_returns_both_coefficients_for_exact_pair` | `B -> K -> (B, A)` on a pair that is causal by construction |
| `test_parity_closure_closes_on_exact_pair` | the even/odd reconstruction tool itself, 0.38% |
| `test_cummins_decay_matches_exact_auxiliary_ode` | RK4 + trapezoidal memory vs the exact 3-state ODE of an exponential kernel (`y' = c x' - a y`) and vs the cubic characteristic roots: period and rate within 0.5%, state within 1% of the initial amplitude |
| `test_pinned_sweep_still_matches_the_reference` | the pinned sweep is not stale (re-evaluated against Hulme at three points, `rel = 1e-9`) |
| `test_hemisphere_kernel_decays_and_is_even` | decay, tail, even extension, `K(0) = (2/pi) int B`, slope bound, causality (`K = 0` for `tau < 0`) |
| `test_hemisphere_round_trip_within_five_percent` | C2 on the hemisphere; measured margin 5e-3 |
| `test_hemisphere_free_decay_matches_response_bandwidth` | C3 period/rate |
| `test_driven_response_matches_frequency_domain_impedance` | driven steady state vs `1/kappa` |
| `test_negative_damping_is_rejected` | **negative case**: `B < 0` → `ValueError` (passivity) |
| `test_kernel_sampling_violating_nyquist_is_rejected` | **negative case**: `dtau > pi/omega_max` → `ValueError` |
| `test_reconstruction_frequency_above_kernel_nyquist_is_rejected` | **negative case**: `omega > pi/dtau` → `ValueError`, while an in-grid query matches the tabulated `B` |
| `test_malformed_sweeps_are_rejected` | **negative case**: too few points, non-monotone `omega`, non-positive `omega`, non-finite `B` |
| `test_growing_signal_is_reported_as_negative_decay` | **negative case**: a negative-damping synthetic signal is reported with a negative decay rate, not silently as decaying |

---

## 6. Errors against benchmarks, item by item

| key quantity | benchmark | error | below 3%? |
|---|---|---|---|
| kernel `K(0)` | `(2/pi) int B domega` from the same sweep | 1e-5 | yes |
| `K(tau)` (exponential pair) | band-truncated closed form | 1e-3 | yes |
| round-trip `B` | tabulated `B` | 1.27e-4 | yes |
| round-trip `A` | tabulated `A` | 2.37e-4 | yes |
| `A_inf` | `0.5 rho nabla` = `ka -> 90` asymptote of Hulme | 6.6e-6 | yes |
| free-decay period | `2 pi/(omega_n sqrt(1-zeta²))` from the response function | 0.70% | yes |
| free-decay rate | bandwidth `zeta_bw omega_n` | 0.41% | yes |
| free-decay rate | energy identity `zeta omega_n` | **13.9%** | **no — and it must not** (single-frequency vs broadband damping; see §4) |
| driven amplitude/phase | `1/kappa(omega)` | 0.02–0.31% / ≤ 9e-4 rad | yes |
| Cummins free decay | exact auxiliary ODE (exponential kernel) | 0.26% of initial amplitude | yes |

The one line that is not below 3% is reported as such: it is a physics
statement about equivalent linearisation, not an accuracy failure of the
transform, and the criterion list deliberately does not include it. No other
quantity in this document has a larger error than 0.7% against its benchmark.

---

## 7. Pitfalls, limitations, applicability

Things that were actually wrong on the way, kept as warnings:

1. **Sweep parameterisation.** A sweep uniform in `ka` produces a `omega` grid
   whose first interval spans a decade; the piecewise-linear `B` is then wrong
   at low frequency and the reconstructed `A` is off by 25% — while `B`'s round
   trip still looks perfect (it uses the same wrong data twice). Uniform in
   `omega` is required, and the residual to watch is the *added mass*, not the
   damping.
2. **`tau_max` is set by the sweep spacing, `dtau` by the highest frequency.**
   Choosing `tau_max` too short truncates the sine integral of eq. (3)
   mid-period and the error grows like `1/omega`; the natural choice is
   `tau_max = pi/domega` from the sweep itself.
3. **The odd partner converges conditionally.** `omega (A - A_inf)` decays only
   as `1/omega`, so computing the parity closure without its analytic tail
   (`c [pi/2 - Si(W tau)]`) leaves a residual of order 1 of the kernel peak. The
   sign matters too: it is `B cos - omega(A-A_inf) sin`, not `+`.
4. **The complex pole is not a clean single exponential.** An attempt to make
   the sharper check — solving `s²(M + A_inf) + s Khat(s) + C = 0` by analytic
   continuation of the memory transform — was removed from the module rather
   than shipped unverified: the continuation has a branch cut on the imaginary
   axis, the free decay is therefore not a pure exponential, and a
   naive search finds either no root or a spurious point on the imaginary axis.
   The bandwidth comparison in §4 is the well-posed version of that idea and is
   what a model test measures anyway.
5. **Fitting a decay envelope in the noise.** `|K|` decays by six decades; a
   log-envelope fit over its far tail measures the interpolation error of the
   sweep, not the decay (it even returned a *negative* decay rate on one grid).
   The fit window is restricted to 50%–1% of the peak, and the tail is reported
   separately.

Limitations, stated plainly:

* Linear, single degree of freedom, deep water, zero forward speed. Surge/roll
  coupling, forward-speed memory kernels (which need `omega_e` rather than
  `omega`) and finite depth are outside this surface.
* The case is a hemisphere for which an independent semi-analytical reference
  exists. A panel-mesh sweep would add discretisation error to every residual
  and would need its own conditioning evidence; nothing here claims that a
  panel-solved `(A, B)` pair is causal, only that the transform of a given pair
  is faithful.
* Nonlinear roll damping and parametric roll are **not** modelled: they are the
  declared roadmap next step, and they require a different (amplitude-dependent)
  memory formulation rather than a better transform.
* The kernel is only as good as the sweep band: below 0.14 rad/s and above
  29.7 rad/s the numbers come from the stated models (cubic long-wave extension,
  truncation bound 1.0e-4 of the peak).