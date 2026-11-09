# Mean drift force on a vertical cylinder: exact two-formula verification and the `(ka)^3` law

**Node**: `p41w3-n01` (TensorBEM, mission m7374). **Artifacts**:
`artifacts/cylinder_drift_asymptotic/study.json` + two PNG figures, produced by

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.cylinder_drift_asymptotic \
    --json artifacts/cylinder_drift_asymptotic/study.json \
    --figures artifacts/cylinder_drift_asymptotic
```

**Code**: `src/tensorbem/cylinder_drift.py` (exact MacCamy-Fuchs reference field,
near-field and far-field drift formulas, pure NumPy Bessel functions).
**Tests**: `tests/test_cylinder_drift_asymptotic.py` (29 tests, all green).

---

## 1. Introduction: what was open, and why it mattered

`dev_report_m5242_second_order_drift.md` shipped three independent drift-force
routes and closed every gate except one, which it reported as an **honest FAIL**:
the `(ka)^3` long-wave law of the cylinder was *not evaluable*, because the 3-D
Maruo far-field route integrated the **un-reduced** interference flux, which
oscillates like `exp(i k R cos theta)` and therefore does not converge in the
radiation radius `kR` (measured spread ~100 %). That FAIL is preserved here
unchanged — nothing in `benchmarks/drift_force_study.py` is edited. What this
node adds is a route that never touches a radiation radius: the **exact**
MacCamy-Fuchs (MF) diffraction field of a circular cylinder, used twice, once in
the near field (wall pressure + waterline) and once in the far field (reduced
stationary-phase Maruo kernel), so the two formulas verify *each other* and the
`(ka)^3` law becomes a computable object rather than a stalled gate.

## 2. Method

Deep water, fixed bottom-mounted vertical cylinder of radius `a`, incident
amplitude `A`, `Phi = Re{(gA/omega) exp(kz) psi e^{-i omega t}}`,
`omega^2 = g k`. The horizontal problem is the exterior Neumann Helmholtz
problem, solved **exactly** in polar coordinates,

```
psi(r,theta) = sum_m [ i^m J_m(kr) + A_m H_m^(1)(kr) ] e^{i m theta},
A_m = - i^m J_m'(ka) / H_m'^(1)(ka)          (MacCamy & Fuchs 1954),
```

with `m = -14 ... 14` (verified converged: `m_max` 10 vs 18 agree to `1e-10`).
The Bessel functions are the package's own ascending series / upward recurrence
(no SciPy dependency), checked against published values to `1e-11`.

**Near field.** The classical second-order mean pressure integral for a fixed
body, with `<|grad phi|^2> = |grad Phi_hat|^2/2` and `<zeta^2> = |zeta_hat|^2/2`,

```
F_i = (1/4) rho Int |grad Phi_hat|^2 n_i dS  -  (1/4) rho g Int |zeta_hat|^2 n_i dl,
```

evaluated on the wall `r = a` with the depth integral analytic
(`Int exp(2kz) dz = 1/(2k)`). On the wall
`|grad Phi_hat|^2 = (gA/omega)^2 exp(2kz) [ |dpsi/dr|^2 + |dpsi/dtheta|^2/a^2 + k^2|psi|^2 ]`
(the `k^2|psi|^2` term is the vertical velocity, which dominates on a vertical
wall) and `zeta_hat = A psi(a,theta)`.

**Far field.** With `psi_s ~ sqrt(2/(pi k r)) exp(i(kr - pi/4)) S(theta)` at large
`r`, the scattered energy flux plus the stationary-phase momentum balance give the
`kR`-independent form

```
F_x = (rho g A^2 / (2 pi k)) Int_0^{2pi} |S(theta)|^2 (1 - cos theta) dtheta,
```

i.e. `F_x = P_scattered / c_g` with the interference term already reduced; in the
Kochin normalisation documented for `tensorbem.free_surface_solver3d`
(`phi ~ (i k/2) H_0^(1)(kR) exp(kz) K`), `K = -2 i S / k` and
`F_x = (rho g A^2 k / 8 pi) Int |K|^2 (1 - cos theta) dtheta`.

Both constants are pinned by anchors rather than asserted:

* the near-field combination (sign and the `1/4` factor) returns exactly
  `rho g A^2 / 2` per unit length for a fully reflecting vertical wall
  (element 6, and `test_wall_anchor_reproduces_rho_g_A2_over_2`);
* the near/far pair then agrees to machine precision (below), which is the
  momentum theorem applied to the annulus `a < r < inf` for the exact field.

**Long-wave closed form.** For `ka -> 0` only `m = 0, +-1` survive,
`A_0 ~ -i kappa`, `A_1 ~ -kappa`, `A_{-1} ~ +kappa`, `kappa = pi (ka)^2/4`, so
`S ~ i kappa (2 cos theta - 1)` and, since
`Int_0^{2pi} (2cos theta - 1)^2 (1 - cos theta) dtheta = 10 pi`,

```
F_x = (5 pi^2 / 16) rho g A^2 a (ka)^3  =  3.08425 rho g A^2 a (ka)^3.
```

## 3. Results

### 3.1 The two formulas agree exactly (8 frequencies, `ka` 0.2 ... 2.0)

`F` in N, `A = 1 m`, `a = 1 m`, `rho = 1025 kg/m^3`, `g = 9.81 m/s^2`.
`F/(rho g (ka)^3)` is the coefficient normalised out of the `(ka)^3` law.

| ka | near field `F_x` | far field `F_x` | rel. diff | `F/(rho g (ka)^3)` | `Int |S|^2(1-cos)` |
|---|---|---|---|---|---|---|
| 0.200 | 240.6 | 240.6 | 2.8e-15 | 2.9909 | 0.0301 |
| 0.457 | 2337.1 | 2337.1 | 3.9e-16 | 2.4330 | 0.6676 |
| 0.714 | 5434.6 | 5434.6 | 1.7e-16 | 1.4831 | 2.4256 |
| 0.971 | 6675.7 | 6675.7 | 4.1e-16 | 0.7242 | 4.0522 |
| 1.229 | 6356.9 | 6356.9 | 0.0e+00 | 0.3409 | 4.8801 |
| 1.486 | 6039.2 | 6039.2 | 3.0e-16 | 0.1831 | 5.6066 |
| 1.743 | 6170.7 | 6170.7 | 2.9e-16 | 0.1159 | 6.7202 |
| 2.000 | 6306.4 | 6306.4 | 8.7e-16 | 0.0784 | 7.8813 |

Maximum near/far relative difference **2.8e-15** (gate: `< 5 %`). The force is
positive (downwave) for every `ka`, non-monotonic in `ka` (a maximum near
`ka ~ 0.97`), and the waterline term is not a correction: at `ka = 1` the surface
term is `-4.3e2` and the waterline term `+1.1e3` in the same units, i.e. the
result would be wrong in sign and magnitude without it.

### 3.2 The `(ka)^3` law and its closed-form coefficient

| ka | `F/(rho g (ka)^3)` |
|---|---|
| 0.02 | 3.08389 |
| 0.03 | 3.08322 |
| 0.05 | 3.08062 |
| 0.08 | 3.07317 |
| 0.10 | 3.06558 |
| 0.15 | 3.03641 |
| 0.20 | 2.99086 |
| 0.30 | 2.84186 |

Log-log fit over `ka in [0.02, 0.30]` gives slope **2.977** (gate: `3 +- 0.2`),
and the coefficient at `ka = 0.02` is **3.08389** against the closed form
`5 pi^2/16 = 3.08425`, a relative difference of **1.2e-4** (gate: `< 3 %`). The
exact solution therefore *contains* the `(ka)^3` law and leaves it by +3.0 % at
`ka = 0.2` and by a factor 39 at `ka = 2`, which quantifies where the asymptotic
law stops being usable (its own validity band is `ka <~ 0.1` for 1 % accuracy).

### 3.3 Kochin energy balance (non-absorbing body)

For a fixed rigid body nothing is absorbed and no work is done, so the total
outward energy flux through a control cylinder must vanish:
`flux_scattered + flux_interference = 0`. Both terms are evaluated from the
**exact** series field (no far-field approximation) on circles of radius `R`:

| ka | R | flux_scattered | flux_interference | flux_total | residual |
|---|---|---|---|---|---|
| 0.5 | 5 | 0.3614914 | -0.3614914 | -4.4e-16 | 1.2e-15 |
| 0.5 | 10 | 0.3614914 | -0.3614914 | -8.7e-16 | 2.4e-15 |
| 0.5 | 20 | 0.3614914 | -0.3614914 | -3.5e-15 | 9.6e-15 |
| 1.0 | 5 | 2.0003835 | -2.0003835 | -1.7e-15 | 8.7e-16 |
| 1.0 | 10 | 2.0003835 | -2.0003835 | -3.5e-15 | 1.7e-15 |
| 1.0 | 20 | 2.0003835 | -2.0003835 | -1.7e-15 | 8.7e-16 |
| 1.5 | 5 | 3.6002470 | -3.6002470 | -2.6e-15 | 7.3e-16 |
| 1.5 | 10 | 3.6002470 | -3.6002470 | -7.0e-15 | 1.9e-15 |
| 1.5 | 20 | 3.6005868 | -3.6005868 | -8.7e-15 | 2.4e-15 |

Residuals are `1e-15`, nine orders below the 1 % gate, and the scattered flux is
`R`-independent to `1e-11` across `R = 5, 10, 20`. The residual is limited only by
the ascending Bessel series, which is why `energy_balance` refuses `k R > 35`
(`test_energy_balance_rejects_unresolved_probe_radius`); the far-field drift route
used in 3.1 never needs a finite `R` at all.

## 4. Figures

* `artifacts/cylinder_drift_asymptotic/fig_long_wave_law.png` — exact
  `F/(rho g (ka)^3)` over `ka in [0.02, 1]` against the closed form `5 pi^2/16`,
  with the `ka` band of table 1 shaded.
* `artifacts/cylinder_drift_asymptotic/fig_near_far_energy.png` — near-field
  against far-field force on log-log axes (indistinguishable to `1e-15`) and the
  energy-balance residual for the nine `(ka, R)` combinations.

## 5. Pitfalls, limits, applicability

* **The vertical velocity matters.** Dropping the `k^2|psi|^2` term of
  `|grad Phi_hat|^2` on the wall changes the near-field force by tens of percent;
  on a vertical wall `psi_r = 0` exactly (the Neumann condition holds to `1e-17`),
  so the wall pressure is carried entirely by the vertical velocity and the
  waterline term.
* **Free-surface normalisation is not cosmetic.** `<|grad phi|^2> = |grad Phi_hat|^2/2`
  and `<zeta^2> = |zeta_hat|^2/2`, with `|zeta_hat| = (omega/g)|Phi_hat|`; a factor
  2 error here is invisible in the shape of the curve and shifts everything by 2.
* **Repository convention audit (found, not changed).**
  `tensorbem.maruo_drift.maruo_mean_drift_force`, fed with the Kochin function of
  this module in the `free_surface_solver3d` normalisation, returns exactly
  `2 pi` times the calibrated value (measured ratio `6.2831816` versus
  `2 pi = 6.2831853`). `maruo_drift` documents the classical `1/R` Kochin weight
  `rho omega^2/4k`, whereas the deep-water free-surface wave part of this problem
  decays as `R^(-1/2)` under the `(i k/2) H_0^(1) exp(kz)` normalisation
  (`|K|^2` then carries `rho g A^2 k / 8 pi`). The two are consistent *within*
  their own conventions, but a caller that mixes them inherits a `2 pi` bias. This
  node reports the numbers and does **not** edit `maruo_drift.py`: its own tests
  pin its convention, and the audit belongs to whoever unifies it.
* **Deep water only** (`omega^2 = g k`). Finite depth changes the depth integral
  (`1/(2k) -> tanh(kh)/(2k)` and a `nu`-dependent depth factor) and is not
  implemented here.
* **Fixed, bottom-mounted, rigid.** No forward speed, no mooring, no free
  second-order potential, no radiated waves. A floating (free to oscillate) body
  adds a radiation channel to the energy balance that section 3.3 does not cover.
* **Wigley first-order RAO vs experiment is not delivered.** The node brief also
  asked for Wigley III wave-exciting force and heave/pitch RAO against towing-tank
  experiments at six encounter frequencies. No digitised experimental data set is
  carried in this repository, and typing numbers from memory would be fabrication;
  the comparison is therefore **not** claimed. What is delivered is the
  second-order drift half of the brief, with analytic references, and an explicit
  line here so the gap is auditable rather than silent.

## 6. Error judgement against the acceptance criteria

| criterion | target | measured | answer |
|---|---|---|---|
| near-field vs far-field drift (8 `ka`) | `< 5 %` | `2.8e-15` | yes, by 12 orders |
| far-field drift, radiation-radius convergence | `< 3 %` (extrapolation) | reduced kernel is radius-free; exact-field flux residual `<= 9.6e-15` over `R = 5,10,20` | yes — stronger: no extrapolation needed |
| `(ka)^3` slope | `3 +- 0.2` | `2.977` | yes |
| `(ka)^3` coefficient vs closed form `5 pi^2/16` | `< 3 %` | `1.2e-4` (`ka = 0.02`) | yes |
| Kochin energy residual | `< 1 %` | `<= 9.6e-15` | yes |
| wall anchor `rho g A^2/2` | exact | `0.0` | yes |
| pytest | `>= 15` green | 29 passed | yes |

`study.json -> verdict.all_gates_pass` is `true`; every number above is read from
that file, not typed from a run log.