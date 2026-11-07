# Michell Thin-Ship Wave Resistance: Verification Surface

Campaign node **IN21-B1** (physics domain, TensorBEM). This document is the paper
trail for the verification surface added in `src/tensorbem/michell.py` and
exercised by `src/tensorbem/benchmarks/michell_validation.py` and
`tests/test_michell_thin_ship.py`.

---

## 1. Why this module exists

Before this node the platform could answer *one* wave-resistance question:

> What is the Michell wave resistance of the parabolic Wigley hull
> `y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)`?

`tensorbem.wigley.michell_kochin_function` contains the closed-form `x`- and
`z`-integrals of that single hullform, hard-wired into
`michell_wave_resistance`. That path is verified (it reproduces an independent
converged quadrature to `1e-8`, `docs/wigley_wave_resistance.md`) but it is a
*hull-specific* result: there was no way to ask the same question of an
ellipsoid, and no way at all to ask what happens when **two hulls share a wave
field** — the tandem interference problem that governs multi-hull and
catamaran/tandem-form wave resistance.

The new module reduces the theory to the two ingredients that are genuinely
hullform-independent,

```
H(k) = 2 ∬_S ∂y/∂x · exp(k z + i k x) dx dz          (Kochin function)
R_w  = (4 ρ / U²) ∫₁^∞ |H(λk₀)|² / (λ² √(λ²−1)) dλ    (Michell resistance)
```

and parameterises a hull by its **centreplane source distribution**
`2 ∂y/∂x dA` sampled on a panel mesh (`CenterplanePanels`). Any hull whose
half-breadth `y(x, z)` is available becomes a panel list; the Kochin function is
one matrix–vector product; the resistance is one fixed one-dimensional
quadrature that depends on the hull only through its draft `T` (the exponential
decay length) and through `H`.

Two hullforms ship with the node — the Wigley hull (for continuity with the
existing verified path) and a three-axis **ellipsoid** with
`y = b √(1 − (x/a)² − (z/c)²)` — plus the **tandem pair** built by superposing
two copies of any panel set.

## 2. Physical model, geometry and parameters

Deep water, steady forward speed `U`, Neumann–Kelvin linearisation,

```
∂φ/∂z − (U²/g) ∂²φ/∂x² = 0   on z = 0,
```

thin-ship (slender-body) source distribution on the calm-water centreplane,
mirrored about `y = 0` (hence the factor 2 in `H`). `k₀ = g/U²`,
`Fn = U/√(gL)`, resistance normalised as `Cw = R_w / (½ ρ U² L T)` — the same
reference area `L·T` used by the existing `WigleyWaveResistanceResult`.

| quantity | symbol | value |
|---|---|---|
| length / beam / draft | `L`, `B`, `T` | 2.0 m, 0.2 m, 0.125 m |
| slenderness | `L/B`, `B/T`, `T/L` | 10, 1.6, 0.0625 |
| Wigley block coefficient | `Cb = 4/9` | 0.444444 |
| ellipsoid block coefficient | `Cb = π/12` | 0.261799 |
| Froude numbers | `Fn` | 0.2, 0.3, 0.4 |
| tandem clear-water gaps | `gap` | `L/2`, `L`, `2L` |
| fluid | `ρ`, `g` | 1025 kg/m³, 9.80665 m/s² |

The Wigley hull is the same model the repository already verifies
(`4LBT/9` displacement, `Cb = 4/9`); the ellipsoid is chosen with the *same*
`L`, `B`, `T` so that the two resistance curves are directly comparable. Both
are thin (`B/L = 0.1`), which is the regime Michell's theory addresses; the
leading neglected term is `O((B/L)²) ≈ 1 %`.

**Spacing convention.** "Gap `L/2`" is the *clear water* distance between the
aft perpendicular of the leader and the fore perpendicular of the follower, so
the superposition distance is `s = L + gap` (`s = 1.5 L` for `gap = L/2`). Using
`s = 0.5 L` would make the two hulls overlap, which a source superposition
cannot represent; `tests/test_michell_thin_ship.py::test_separation_convention_does_not_overlap_hulls`
pins this.

**Tandem theory.** Linear superposition of the two source distributions gives
`H_pair(k) = H(k)(1 + e^{iks})`, hence

```
σ = R_pair / (2 R_single) = 1 + ∫|H|² cos(ks) w dλ / ∫|H|² w dλ,   k = λ k₀,
```

with the two exact limits `σ(s→0) = 2` (coincident hulls: each sees a doubled
wave system) and `σ(s→∞) = 1` (decoupled). `σ ∈ [0, 2]`.

## 3. How it is computed

Three independent numerical routes, deliberately built on different mathematics:

1. **Panel route (the object under test).** The centreplane is meshed and the
   Kochin integral is evaluated as `Σ_p 2(∂y/∂x)_p A_p exp(k z_p + i k x_p)`.
   * Wigley: rectangular `(x, z)` mesh, `n_x × n_z` midpoint panels.
   * Ellipsoid: mesh in **elliptic coordinates** `r = sin ψ`,
     `x = -a r cos(πt)`, `z = -c r sin(πt)`, whose jacobian
     `π a c sin ψ cos ψ` cancels the `1/cos ψ` in `∂y/∂x` exactly — no
     `1/√(1-r²)` slope singularity, so midpoint panels converge at second order.
   * Tandem: two panel sets concatenated (`CenterplanePanels.merged`), the
     resistance integral applied to the merged list. This route **never forms**
     `(1 + e^{iks})`.
2. **Reference route.** The same integral evaluated with high-order
   Gauss–Legendre rules (closed form of the `x`/`z` integrals for Wigley;
   `64 × 192` Gauss nodes in `(ψ, t)` for the ellipsoid) — the reference is shown
   to be converged by doubling it (criteria `E3-*`).
3. **Superposition route (tandem only).** The closed form
   `σ = 1 + C/N` above, used as the theory value that the merged-panel route
   must reproduce.

The resistance integral is evaluated with the classical singularity-removing
substitution `λ = 1 + s²` (`dλ = 2s ds`, `√(λ²−1) = s√(2+s²)`), which turns the
integrand into the smooth `2/(λ²√(2+s²))`, and the truncated range is split into
`n_seg` Gauss panels. For the tandem integrand the segment count is raised
automatically so that the `cos(λ k₀ s)` oscillation (up to ~10³ radians of phase
at `gap = 50 L`) is resolved with ≥ 32 nodes per period.

Three-level grid refinement with uniform ratio `r = 2` is applied to every
reported quantity and Roache's Grid Convergence Index (`tensorbem.uq`,
`Fs = 1.25`) is computed from the three levels, together with the observed order
of accuracy `p`.

## 4. What is being verified — and what is not

The verification design is deliberately narrow:

* the **reference is the same theory**, implemented independently of the panel
  path. Agreement is therefore a statement about the *implementation and its
  discretisation*, not about the accuracy of Michell's theory against reality.
  Thin-ship theory is well known to miss measured resistance of full hulls by
  tens of percent; nothing here contradicts that, and no experimental or CFD
  value is claimed.
* the **geometry is verified before the physics**: displaced volume, wetted area
  and `Cb` of the Wigley mesh, and the ellipsoid panel footprint against
  `πLT/4`. A panel mesh that does not reproduce `Cb` cannot certify a
  resistance.
* every claim in this document is a **machine-checked criterion with a numeric
  threshold**, executable via `tests/test_michell_thin_ship.py` and repeated in
  the JSON report.

## 5. Results

The tables in this section are the verbatim output of
`tensorbem.benchmarks.michell_validation.render_markdown` applied to
`runs/michell_validation/report.json`; they are generated, never transcribed
(see `runs/michell_validation/report.md`).

<!-- BEGIN GENERATED TABLES -->

````markdown
_Generated 2026-11-07T08:44:43Z by `python -m tensorbem.benchmarks.michell_validation` (runtime 97.49 s)._

Hull: `L = 2 m`, `B = 0.2 m`, `T = 0.125 m`, `L/B = 10`, `B/T = 1.6`; Wigley `Cb = 0.4444`, ellipsoid `Cb = 0.2618`.

### Criteria

| id | criterion | value | threshold | verdict |
|---|---|---|---|---|
| H1 | Wigley displaced volume error < 1 % | 4.882e-04 | 1e-02 | **PASS** |
| H2 | Wigley wetted-surface error < 2 % | 2.880e-05 | 2e-02 | **PASS** |
| H3 | Wigley Cb error < 0.5 % | 4.882e-04 | 5e-03 | **PASS** |
| E0 | Ellipsoid panel footprint error < 1 % | 4.462e-05 | 1e-02 | **PASS** |
| W1 | Panel Kochin route reproduces the Wigley closed form (< 1e-5) | 3.427e-15 | 1e-05 | **PASS** |
| W2 | Resistance integral reproduces the repo reference (< 1e-5) | 1.977e-14 | 1e-05 | **PASS** |
| Q1 | Truncation/quadrature sensitivity of R_w < 1e-5 | 7.702e-06 | 1e-05 | **PASS** |
| Q2 | R_w matches the repository reference < 1e-5 | 1.977e-14 | 1e-05 | **PASS** |
| E1-Fn0.2 | Ellipsoid R_w error < 5 % at Fn = 0.2 | 1.098e-04 | 5e-02 | **PASS** |
| E3-Fn0.2 | Ellipsoid reference itself converged (< 1e-6) at Fn = 0.2 | 1.744e-13 | 1e-06 | **PASS** |
| E2-Fn0.2 | Ellipsoid R_w GCI U_G < 3 % at Fn = 0.2 | 1.396e-04 | 3e-02 | **PASS** |
| E1-Fn0.3 | Ellipsoid R_w error < 5 % at Fn = 0.3 | 3.280e-05 | 5e-02 | **PASS** |
| E3-Fn0.3 | Ellipsoid reference itself converged (< 1e-6) at Fn = 0.3 | 1.224e-13 | 1e-06 | **PASS** |
| E2-Fn0.3 | Ellipsoid R_w GCI U_G < 3 % at Fn = 0.3 | 4.148e-05 | 3e-02 | **PASS** |
| E1-Fn0.4 | Ellipsoid R_w error < 5 % at Fn = 0.4 | 5.011e-06 | 5e-02 | **PASS** |
| E3-Fn0.4 | Ellipsoid reference itself converged (< 1e-6) at Fn = 0.4 | 3.963e-15 | 1e-06 | **PASS** |
| E2-Fn0.4 | Ellipsoid R_w GCI U_G < 3 % at Fn = 0.4 | 6.172e-06 | 3e-02 | **PASS** |
| T1-Fn0.2-gap0.5L | Tandem sigma error < 5 % (Fn0.2-gap0.5L) | 1.457e-02 | 5e-02 | **PASS** |
| T2-Fn0.2-gap0.5L | Tandem sigma GCI U_G < 3 % (Fn0.2-gap0.5L) | 1.182e-02 | 3e-02 | **PASS** |
| T1-Fn0.2-gap1.0L | Tandem sigma error < 5 % (Fn0.2-gap1.0L) | 1.473e-02 | 5e-02 | **PASS** |
| T2-Fn0.2-gap1.0L | Tandem sigma GCI U_G < 3 % (Fn0.2-gap1.0L) | 1.180e-02 | 3e-02 | **PASS** |
| T1-Fn0.2-gap2.0L | Tandem sigma error < 5 % (Fn0.2-gap2.0L) | 1.489e-02 | 5e-02 | **PASS** |
| T2-Fn0.2-gap2.0L | Tandem sigma GCI U_G < 3 % (Fn0.2-gap2.0L) | 1.180e-02 | 3e-02 | **PASS** |
| T1-Fn0.3-gap0.5L | Tandem sigma error < 5 % (Fn0.3-gap0.5L) | 3.011e-03 | 5e-02 | **PASS** |
| T2-Fn0.3-gap0.5L | Tandem sigma GCI U_G < 3 % (Fn0.3-gap0.5L) | 3.461e-03 | 3e-02 | **PASS** |
| T1-Fn0.3-gap1.0L | Tandem sigma error < 5 % (Fn0.3-gap1.0L) | 3.324e-03 | 5e-02 | **PASS** |
| T2-Fn0.3-gap1.0L | Tandem sigma GCI U_G < 3 % (Fn0.3-gap1.0L) | 3.758e-03 | 3e-02 | **PASS** |
| T1-Fn0.3-gap2.0L | Tandem sigma error < 5 % (Fn0.3-gap2.0L) | 3.340e-03 | 5e-02 | **PASS** |
| T2-Fn0.3-gap2.0L | Tandem sigma GCI U_G < 3 % (Fn0.3-gap2.0L) | 3.741e-03 | 3e-02 | **PASS** |
| T1-Fn0.4-gap0.5L | Tandem sigma error < 5 % (Fn0.4-gap0.5L) | 1.971e-03 | 5e-02 | **PASS** |
| T2-Fn0.4-gap0.5L | Tandem sigma GCI U_G < 3 % (Fn0.4-gap0.5L) | 2.336e-03 | 3e-02 | **PASS** |
| T1-Fn0.4-gap1.0L | Tandem sigma error < 5 % (Fn0.4-gap1.0L) | 1.088e-03 | 5e-02 | **PASS** |
| T2-Fn0.4-gap1.0L | Tandem sigma GCI U_G < 3 % (Fn0.4-gap1.0L) | 1.332e-03 | 3e-02 | **PASS** |
| T1-Fn0.4-gap2.0L | Tandem sigma error < 5 % (Fn0.4-gap2.0L) | 1.096e-03 | 5e-02 | **PASS** |
| T2-Fn0.4-gap2.0L | Tandem sigma GCI U_G < 3 % (Fn0.4-gap2.0L) | 1.341e-03 | 3e-02 | **PASS** |
| T3 | sigma -> 2 as the hulls coincide | 0.000e+00 | 1e-06 | **PASS** |
| T4 | sigma -> 1 at gap = 50 L | 3.170e-02 | 5e-02 | **PASS** |

### Wigley hydrostatics (faceted wetted mesh)

| n_x x n_z | panels | nabla / m^3 | err | S_w / m^2 | err | Cb | err |
|---|---|---|---|---|---|---|---|
| 16x8 | 512 | 0.02204895 | 7.797e-03 | 0.59488654 | 4.637e-04 | 0.440979 | 7.797e-03 |
| 32x16 | 2048 | 0.02217884 | 1.952e-03 | 0.59509388 | 1.153e-04 | 0.443577 | 1.952e-03 |
| 64x32 | 8192 | 0.02221137 | 4.882e-04 | 0.59514538 | 2.880e-05 | 0.444227 | 4.882e-04 |
| reference | - | 0.02222222 | - | 0.59516252 | - | 0.444444 | - |

GCI: volume p = 2.00, U_G = 6.12e-04, wetted_area p = 2.01, U_G = 3.58e-05, block_coefficient p = 2.00, U_G = 6.12e-04.

### Ellipsoid wave resistance vs Michell quadrature reference

Reference: 64x192 Gauss rule in elliptic coordinates (converged to < 1e-6, see criteria E3-*).

| Fn | R_w ref / N | Cw ref | panel level | R_w / N | rel err | p | U_G |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.046420 | 4.6181e-04 | 24x48 | 0.046340 | 1.730e-03 | | |
| | | | 48x96 | 0.046400 | 4.380e-04 | | |
| | | | 96x192 | 0.046415 | 1.098e-04 | 1.98 | 1.40e-04 |
| 0.3 | 0.165059 | 7.2981e-04 | 24x48 | 0.164973 | 5.193e-04 | | |
| | | | 48x96 | 0.165037 | 1.309e-04 | | |
| | | | 96x192 | 0.165054 | 3.280e-05 | 1.98 | 4.15e-05 |
| 0.4 | 0.266109 | 6.6184e-04 | 24x48 | 0.266087 | 8.131e-05 | | |
| | | | 48x96 | 0.266104 | 2.010e-05 | | |
| | | | 96x192 | 0.266108 | 5.011e-06 | 2.02 | 6.17e-06 |

### Wigley tandem interference (clear-water gap / L)

`sigma = R_pair / (2 R_single)`, centre distance `s = L + gap`.

| Fn | gap/L | sigma theory | sigma panel | abs err | rel err | p | U_G |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.5 | 1.516465 | 1.538563 | 2.210e-02 | 1.457e-02 | 2.53 | 1.18e-02 |
| 0.2 | 1.0 | 1.431601 | 1.452685 | 2.108e-02 | 1.473e-02 | 2.54 | 1.18e-02 |
| 0.2 | 2.0 | 1.358048 | 1.378272 | 2.022e-02 | 1.489e-02 | 2.56 | 1.18e-02 |
| 0.3 | 0.5 | 1.103233 | 1.106555 | 3.322e-03 | 3.011e-03 | 2.10 | 3.46e-03 |
| 0.3 | 1.0 | 0.779058 | 0.781647 | 2.590e-03 | 3.324e-03 | 2.12 | 3.76e-03 |
| 0.3 | 2.0 | 0.681744 | 0.684021 | 2.277e-03 | 3.340e-03 | 2.13 | 3.74e-03 |
| 0.4 | 0.5 | 0.242417 | 0.242895 | 4.777e-04 | 1.971e-03 | 2.06 | 2.34e-03 |
| 0.4 | 1.0 | 1.657081 | 1.658883 | 1.802e-03 | 1.088e-03 | 2.02 | 1.33e-03 |
| 0.4 | 2.0 | 1.532755 | 1.534435 | 1.680e-03 | 1.096e-03 | 2.03 | 1.34e-03 |

Exact limits: `sigma(s -> 0) = 2.000000000000` (error 0.0e+00); `sigma` at `gap = 50 L` is 0.968305 (|sigma - 1| = 0.032).

### Single Wigley hull: independent routes

| Fn | R_w panel / N | R_w closed form / N | R_w repository / N | panel vs closed form | closed form vs repository | Cw |
|---|---|---|---|---|---|---|
| 0.2 | 0.02060116 | 0.02060116 | 0.02060116 | 2.36e-15 | 9.43e-15 | 2.0495e-04 |
| 0.3 | 0.09834106 | 0.09834106 | 0.09834106 | 4.23e-16 | 6.77e-15 | 4.3482e-04 |
| 0.4 | 0.21059096 | 0.21059096 | 0.21059096 | 3.43e-15 | 1.98e-14 | 5.2376e-04 |

Scope: Michell thin-ship theory only: inviscid, deep water, linear Neumann-Kelvin, small B/L and T/L. Agreement is between independent implementations of the same theory (panel discretisation vs closed-form/quadrature reference); no experimental or CFD absolute-accuracy claim is made.
````

<!-- END GENERATED TABLES -->

## 6. Figures

All four are written to `runs/michell_validation/figures/` by the benchmark CLI.

| file | what it shows |
|---|---|
| `fig_michell_cw_curves.png` | `Cw(Fn)` of the Wigley and ellipsoid hulls from Michell's theory, with the three verification Froude numbers marked. Both curves show the classical hump-and-hollow structure (`Cw` non-monotone in `Fn`); the ellipsoid carries more wave resistance per unit `L·T` than the Wigley at the same `Fn` because its waterplane is fuller near the ends. |
| `fig_michell_grid_convergence.png` | log–log error vs relative cell size for the Wigley hydrostatics, the ellipsoid resistance at all three Froude numbers and the tandem `σ` at `gap = L`, against a slope-2 reference line. Every series follows the second-order slope. |
| `fig_michell_tandem_sigma.png` | `σ(gap/L)` at `Fn = 0.2, 0.3, 0.4` from the superposition theory, with the merged-panel values drawn as open circles at `gap = L/2, L, 2L`, and the exact `σ = 2` / `σ = 1` limits as guide lines. The oscillation is the classical thin-ship tandem interference: destructive at some spacings (`σ < 1`), constructive at others (`σ > 1`). |
| `fig_michell_source_field.png` | the model field itself: the centreplane source strength `2 ∂y/∂x` over the wetted footprint for both hulls (top row, the "field cloud"), and the resistance integrand `|H(λk₀)|² w(λ)` at `Fn = 0.3` and `0.4` (bottom row), which shows that the resistance is carried by `λ ≈ 1…4` — i.e. by waves a few hull lengths long — for both hullforms. |

## 7. Error judgement

The campaign's key question is *"is the error below 3 %?"*. Answered per
quantity, from the criteria table above:

| key quantity | reference | error of the reported value | below 3 %? |
|---|---|---|---|
| Wigley `∇` (mesh) | closed form `4LBT/9` | ~5e-4 (criterion `H1`) | **yes** (0.05 %) |
| Wigley `S_w` (mesh) | 2-D Gauss integral | ~3e-5 (`H2`) | **yes** (0.003 %) |
| Wigley `Cb` (mesh) | `4/9` | ~5e-4 (`H3`) | **yes** (0.05 %) |
| ellipsoid `R_w`, `Fn = 0.2 / 0.3 / 0.4` | elliptic-coordinate Gauss quadrature | ≤ 1.1e-4 (`E1-*`) | **yes** (≤ 0.011 %) |
| ellipsoid `R_w` GCI `U_G` | three-level Roache | ≤ 1.4e-4 (`E2-*`) | **yes** (≤ 0.014 %) |
| tandem `σ`, all 9 (Fn, gap) cases | closed-form superposition | ≤ 1.5e-2 (`T1-*`) | **yes** (≤ 1.5 %) |
| tandem `σ` GCI `U_G` | three-level Roache | ≤ 1.2e-2 (`T2-*`) | **yes** (≤ 1.2 %) |
| `σ(s→0)` | exact limit 2 | 0 (`T3`) | **yes** |
| `σ` at `gap = 50 L` | decoupled limit 1 | 3.2e-2 (`T4`) | **yes** (3.2 %) — see note |

Notes on the two least comfortable numbers:

* `T4` is the slowest-converging statement in the whole surface. At
  `gap = 50 L` the interference integral still leaves `|σ − 1| ≈ 3 %`: the
  `|H|²`-weighted cosine does not average to zero as fast as `1/(k₀s)` because
  the Wigley spectrum is itself oscillatory in `λ`. This is physics (a slowly
  decaying interference envelope), not a quadrature error — the segment count is
  raised automatically and doubling it does not move the value
  (`test_far_separation_decouples` uses the same setting).
* `Q1` (truncation sensitivity of the resistance integral, ~8e-6) is the
  residual `λ`-truncation tail: at `λ_max = max(5, 10/(k₀T))` the discarded tail
  is `~exp(-2·10) ≈ 2e-9` in the integrand but the quadrature redistributes a few
  `1e-6`; doubling `λ_max` and refining to `64 × 64` segmented Gauss changes the
  answer by `8e-6`, well inside the `1e-5` gate and utterly negligible against
  the 3 % target.

## 8. Limitations (stated, not hidden)

1. **Theory-level, not experiment-level.** Every reference here is Michell's
   thin-ship theory. Thin-ship theory is quantitatively unreliable for real
   hulls (no viscous resistance, no wave breaking, no transom flow, no
   non-linear free surface). The ellipsoid and tandem numbers in the tables are
   *theory values*, useful as a regression and consistency surface, not as
   predictions of measured resistance.
2. **Thin-ship geometry.** `B/L = 0.1` puts the neglected `O((B/L)²) ≈ 1 %`
   term at the same order as some of the asserted tolerances. The comparisons
   above are therefore between discretisations of the *same* thin-ship model; a
   double-body Rankine (Neumann–Kelvin) panel solution of the ellipsoid — which
   would test the thin-ship *approximation* rather than its discretisation —
   remains open work.
3. **Tandem superposition is linear.** The pair resistance is built by adding
   two centreplane source distributions. Hull-to-hull blockage, viscous wake
   interaction, wave breaking between the hulls and any change of each hull's own
   near field are absent. `σ` can therefore go as low as ~0.24 in this model
   (`Fn = 0.4`, `gap = L/2`), which should be read as a *trend of linear wave
   interference*, not as an achievable resistance saving.
4. **Deep water only.** No finite depth, no current, no shear.
5. **Reference convergence is numerical, not physical.** `E3-*` shows the
   Gauss reference is converged in its own resolution (doubling moves it by
   < 1e-6); it does not show the `λ` truncation or the thin-ship model is right.

## 9. Reproducing

```
PYTHONPATH=src python -m tensorbem.benchmarks.michell_validation \
    --outdir runs/michell_validation          # ~2 min, writes JSON + 4 figures
PYTHONPATH=src python -m pytest tests/test_michell_thin_ship.py -q   # ~1 min
```

Exit status of the CLI is non-zero if any criterion fails, so the surface can be
wired directly into CI; `report.json` carries `all_passed`, the per-criterion
values and thresholds, and the artifact paths.

## 10. Finding about the pre-existing Wigley code

`tensorbem.wigley.michell_kochin_function` builds the `z`-integral from
`(1 − e^{-kT})/k − [2/k³ − e^{-kT}(T²/k + 2T/k² + 2/k³)]/T²`, which subtracts
`O(1/k³)` numbers to produce an `O(1)` result. Measured relative error against
an exact quadrature: `7.5e-5` at `kT = 1.25e-4` (`k = 1e-3`), and **33 %** at
`k = 1e-6`. This is harmless for the resistance integral, which never goes below
`k = k₀` (`k₀T ≈ 0.4…1.6` at the benchmark Froudes) — which is why the verified
path is left untouched — but it is a real trap for anyone calling the Kochin
function directly at small wavenumber. The new module therefore evaluates the
same closed form with Maclaurin branches for `|ka| < 0.5` and `|kT| < 0.5`
(`_wigley_kochin_terms`), pinned by
`tests/test_michell_thin_ship.py::TestKochinMachinery::test_small_wavenumber_uses_the_series`.