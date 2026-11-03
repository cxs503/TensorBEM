# 3-D free-surface Green function in water of finite depth

Status: **implemented and verified** (`src/tensorbem/finite_depth3d.py`,
`tests/test_finite_depth3d.py`, generator
`src/tensorbem/benchmarks/finite_depth3d_verification.py`).
This closes the first of the three items that README roadmap step 6 listed as
"next, needs new physics".

## 1. Why a separate kernel

`tensorbem.free_surface3d` is the deep-water Kelvin/Delhommeau kernel: Rankine
singularity + free-surface image + an `E1` wave integral with a pole at
`nu = omega**2/g`. A flat bottom at `z = -h` destroys that structure — the
vertical spectrum stops being a continuum, the single pole at `nu` is replaced
by the transcendental dispersion relation, and the correct representation is a
*modal expansion of the strip*. The two-dimensional case is
`tensorbem.finite_depth`; this module is its three-dimensional counterpart.

## 2. Formulation

With `z` up, mean free surface `z = 0`, bottom `z = -h`, time factor
`exp(-i*omega*t)`: `dphi/dz = nu*phi` at `z = 0`, `dphi/dz = 0` at `z = -h`. The
vertical eigenfunctions and norms are

| mode | eigenfunction | dispersion | norm `N` |
|---|---|---|---|
| propagating | `phi_0 = cosh(k0 (z+h))` | `k0 tanh(k0 h) = nu` | `(h/2)(1 + sinh(2 k0 h)/(2 k0 h))` |
| evanescent `n>=1` | `phi_n = cos(mu_n (z+h))` | `mu_n tan(mu_n h) = -nu` | `(h/2)(1 + sin(2 mu_n h)/(2 mu_n h))` |

Expanding `delta(z - z0)` in this complete set and solving the horizontal
problem for each mode gives

```
G(x, x0) = sum_{n>=1} phi_n(z) phi_n(z0)/N_n * K0(mu_n R) / (2 pi)
         + phi_0(z) phi_0(z0)/N_0 * (i/4) H0^(1)(k0 R),     R = |x_h - x0_h|
```

because `(mu^2 - Delta_2) K0(mu R) = 2 pi delta_2` and
`(Delta_2 + k0^2) (i/4) H0^(1)(k0 R) = -delta_2`. The overall sign is *pinned by
the `h -> inf` limit against the independent deep-water kernel*, exactly as in
the 2-D module: it is a test, not an assumption.

Convergence is **exponential** in the mode count (every term carries
`exp(-mu_n R)`), so the sum is truncated at `mu_n R >= 36`; no Kummer
acceleration is needed (contrast the 2-D case, whose evanescent sum decays like
`1/n`). The cost is that the required mode count grows like `h/R`; `R = 0` is
rejected because the sum there is only conditionally convergent.

Only `numpy` is used: `K0`, `J0`, `Y0` are implemented from factorial-ratio
series (A&S 9.8.5 / 9.4.1-9.4.6 style coefficients, generated in code, not
transcribed), with SciPy used *only* in the tests as an external reference.

## 3. Verification

`python -m tensorbem.benchmarks.finite_depth3d_verification` regenerates every
number below into `outputs/finite_depth3d/metrics.json` (zero hand copy).

### 3.1 Deep-water limit against the Kelvin kernel (`nu = 1`, fixed geometry)

| `nu h` | max rel. diff. (total) | rel. diff. Im part | observed order |
|---:|---:|---:|---:|
| 25 | 2.432e-05 | 0.0 | 3.09 |
| 50 | 2.857e-06 | 0.0 | 3.05 |
| 100 | 3.457e-07 | 0.0 | 3.02 |
| 200 | 4.270e-08 | 0.0 | 3.01 |
| 400 | 5.300e-09 | 0.0 | — |

The radiating (imaginary) part agrees to machine precision at every depth,
i.e. the propagating-mode normalisation is exact; the residual sits in the real
part and decays with a clean observed order 3.00 ± 0.09, reaching **5.3e-9** at
`nu h = 400`. Two independent code paths (discrete modal sum vs Rankine + image
+ `E1` integral) therefore describe the same kernel in the deep-water limit.

### 3.2 Defining boundary conditions (independent of any other kernel)

| `nu h` | FS residual, step `1e-3 h` | FS residual, step `1e-4 h` | `h |dG/dz| / |G|` at `z = -h` |
|---:|---:|---:|---:|
| 1 | 3.428e-07 | 3.390e-09 | 9.93e-07 |
| 5 | 4.349e-06 | 4.376e-08 | 6.47e-06 |
| 20 | 9.474e-06 | 9.523e-08 | 6.29e-06 |

The free-surface residual falls by exactly the factor expected from the
second-order one-sided stencil (`~step^2`), so `dG/dz = nu G` is satisfied to the
discretisation floor; the bottom normal derivative vanishes to the
finite-difference floor.

### 3.3 Structural and special-function checks

| check | result |
|---|---|
| reciprocity `G(P,Q) - G(Q,P)` | exactly 0 (bit-identical) |
| incoming = conjugate of outgoing | exactly 0 |
| `K0` vs `scipy.special.k0` (1e-3 .. 200) | 6.8e-10 max rel. |
| `J0`, `Y0` vs SciPy (1e-3 .. 200) | 1.2e-10, 8.2e-11 max rel. |
| mode-count sensitivity (auto vs 20 000 modes) | 0.0 |
| evanescent dispersion residual | 3.3e-13 |
| far-field phase per unit length vs `k0` (`nu h = 3`) | 1.00510 vs 1.00483 (`k0`) |

The far-field check confirms the kernel radiates at the *finite-depth*
wavenumber `k0`, not at `nu`: at `nu h = 3` the two differ by 0.48 %, which is
why the deep-water kernel cannot be reused at that depth.

### 3.4 When the deep-water kernel is not good enough

Source at fixed *relative* submergence `z0 = -0.4 h`, five field points, error
of the deep-water kernel relative to the finite-depth one:

| `nu h` | 0.5 | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| `k0 / nu` | 1.543 | 1.200 | 1.033 | 1.00009 | 1.00000 |
| deep-water kernel error | 44.9 % | 20.5 % | 4.76 % | 1.86 % | 1.64 % |

Read this as a scenario table, not as a convergence study: the geometry changes
with `h` (the source submerges with the depth), so the large-`nu h` entries do
not tend to zero. The clean convergence statement is §3.1.

## 4. Limitations (honest)

1. **`R = 0` is rejected.** On the source axis the modal sum converges only
   conditionally (like `sum log(mu_n) cos(...)/n`); the 2-D module's closed-form
   log-sums would be needed. Offset the target horizontally.
2. **Mode count grows like `h/R`.** `R = h/100` already needs ~10^3 modes;
   the auto-selection caps at 4e5 and raises instead of hanging.
3. **Real-part residual in the deep-water limit is `O(h^-3)`, not exponential.**
   Physically the two kernels differ by `exp(-2 nu h)`, so the observed `h^-3`
   is the difference between two *representations* (discrete modal sum vs wave
   integral). Both kernels independently satisfy their own free-surface and
   bottom conditions, so this is an accuracy floor of the comparison, not of the
   kernel; separating the two would need a third, independent representation.
4. **`nu h -> 0` (omega -> 0 at fixed `h`) is degenerate.** The propagating mode
   coalesces with the lowest evanescent mode and the truncated sum converges
   slowly: we measured a 36 % deviation from the rigid-lid strip Green function
   at `nu = 1e-4`, `h = 4`. The kernel is validated for `nu h >= 0.5`
   (which is the regime ships care about: `omega^2 h / g = O(1)`).
5. **Normal derivatives / gradients are not provided yet**, only the kernel
   itself and the `z`-derivative used by the boundary-condition test. A BEM
   solver needs `grad G`; that is the next increment, and the natural target is
   Garrett's (1970) truncated vertical cylinder / gap resonance, which is why
   this kernel was written first.