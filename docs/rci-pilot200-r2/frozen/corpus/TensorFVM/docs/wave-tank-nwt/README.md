# Time-domain numerical wave tank (NWT) — initial implementation and honest status

Campaign node **p106w1-n03** (TensorFVM, seakeeping). This directory records what
was built, how it is verified, and — most importantly — which acceptance
criteria are **met** and which are **not**.

## 1. What the module is

`src/tensorfvm/wave_tank/` (new domain directory in TensorFVM):

| file | content |
|---|---|
| `theory.py` | finite-depth dispersion `ω²=gk·tanh(kh)` (bracketed Newton), Airy kinematics, Stokes-II bound harmonic, least-squares harmonic fit, three-probe wavenumber, Goda two-point incident/reflected separation, period-mean added resistance (Gerritsma–Meijers), RAO reduction |
| `geometry.py` | Wigley (parabolic) and *simplified* KCS hulls, the `Cb / L/B / B/T / ∇` pre-gate, the 2-D barge strip, vectorised polygon mask |
| `solver.py` | 2-D staggered MAC free-surface tank: single-phase water / empty-air formulation, relaxation-zone generation, momentum predictor (central + upwind blend), masked 5-point pressure Poisson (Jacobi-preconditioned CG, warm start), THINC flux-form VOF, damping-zone sponge, heave+pitch rigid body with added-mass-informed coupling |
| `cases.py` | reproducible case drivers writing evidence JSON under `outputs/wave-tank/` |

## 2. Method (what is actually solved)

* Grid: staggered MAC, `u` on x-faces, `v` on y-faces, `α` cell-centred.
* Momentum: `u* = u + Δt(−adv + ν∇²u − ∇p/ρ)`, `v* = v + Δt(−adv + ν∇²v − ∇p/ρ − g)`
  with a central/upwind blend (`advection_blend = 0.25`).
* Projection: **incremental** (Chorin) — `∇²φ = −(ρ/Δt)·∇·u*`,
  `u = u* − Δt∇φ/ρ`, `p ← p + φ`, `p = 0` in air, homogeneous Neumann on walls and hull.
  The free-surface Dirichlet uses the **sub-cell** distance
  `d = clamp(η − y_cell, 0.15Δy, 1.5Δy)`; without it the wave amplitude is quantised
  to `Δy`.
* A pool at rest must be an exact fixed point, which it is only if `p` starts
  hydrostatic and the projection is incremental (a uniform free-fall field is
  divergence-free, so a one-shot Chorin projection never “sees” gravity). This is
  enforced by `initialize_hydrostatic()` and covered by
  `tests/test_wave_tank_solver.py::test_hydrostatic_pool_stays_at_rest`.
* Generation: relaxation zone of one wavelength (`α` and face velocities blended
  toward the Stokes-II target with a smoothstep `w(x)`), i.e. the waves2Foam
  recipe. Absorption: a ≥1.5λ sponge with a quadratic ramp on the velocities plus
  an interface relaxation toward still water.
* Probes: `η_i = y_bottom + Δy·Σ_j α_ij` — the surface is recovered **from mass**,
  so interface diffusion does not corrupt the measured amplitude.
* Body: stair-step mask rebuilt every step from the rotated section polygon,
  rigid-face velocities from `V + Ω×r`, wetted-face pressure integral (with the
  hydrostatic part extrapolated from the cell centre to the face) for heave force
  and pitch moment, added-mass-informed left-hand side for stable loose coupling.

## 3. Verification status

`python -m pytest tests/test_wave_tank_theory.py tests/test_wave_tank_solver.py -q`
→ **26 passed, 1 failed** (the failing one is the documented buoyancy-bias
tolerance, see §5).

| check | result |
|---|---|
| dispersion solver vs analytic deep/shallow limits | pass (≤1e-6) |
| Airy/Stokes-II kinematics (deep-water decay, no penetration at the bed, 2nd harmonic) | pass |
| harmonic fit / Goda separation / three-probe inversion / added resistance on synthetic data | pass (≤1e-10) |
| geometry pre-gate: Wigley `Cb = 0.44442` (ref 4/9, err 6e-5), KCS-simplified `Cb = 0.65062` (ref 0.651, err 5.9e-4), `∇ = Cb·L·B·T` to round-off | pass |
| hydrostatic pool at rest (12 steps): `|u|,|v| < 1e-6`, `η < 1e-6` | pass |
| hydrostatic pressure profile `p/(ρg(η−y))` | pass (1e-3) |
| relaxation zone imposes the target at the wall (`w = 1`) | pass (2e-3 m) |
| closed-tank volume conservation over 20 steps | pass (<5e-3) |
| body float near hydrostatic equilibrium | **8 % buoyancy bias** (stair-step, `Δy/(2T)`) |

## 4. Acceptance criteria — honest verdict

Case `outputs/wave-tank/smoke.json` (nx=100, ny=40, L=7 m, h=0.9 m, `a=0.03 m`,
`T=1.0 s`, λ=1.56 m, 22 points/λ, `a/Δy = 0.92`, 4 periods, 100 steps/period,
damped case, 6 s of wall clock):

| criterion (campaign node) | measured | verdict |
|---|---|---|
| wave-height error ≤ 3 % | probe at 0.35λ after the relaxation zone: `a/a_target = 0.21` (error 79 %); next probes 0.037 and 0.121 | **FAIL** |
| `k` from three-probe inversion vs `ω²=gk·tanh(kh)`, ≤1 % | 69 % (the field is not a clean travelling wave after the decay) | **FAIL** |
| reflection `C_R ≤ 5 %` (Goda two-point) | 0.37 — but the incident amplitude at the gauges is only 10 % of the target, so this number measures the decayed field, not the absorber | **FAIL / not meaningful yet** |
| RAO vs BEM within 10 % outside resonance, GCI ≤ 5 % | not delivered (no potential-flow reference implemented yet) | **NOT DELIVERED** |
| added resistance vs BEM 2nd-order drift ≤ 15 % | not delivered | **NOT DELIVERED** |
| negative control: no damping zone → `C_R ≥ 20 %` | not delivered | **NOT DELIVERED** |

**The blocking defect is a strong spurious damping of the wave field**: the wave
loses ~80 % of its first-harmonic amplitude within one wavelength even though the
relaxation zone re-imposes the target at the wall. Candidate causes, in the order
they should be attacked:

1. **THINC fluxes at the free surface** with `C ≈ 0.3` smear/re-steepen the
   interface every step; the resulting `α` redistribution is a mass-neutral but
   momentum-dissipative process. Test: replace THINC by a volume-conserving
   height-function update in the surface cells and compare the decay rate.
2. **Cell-centred free-surface Dirichlet**: `p = 0` is applied at the *air cell
   centre* for horizontal faces (`Δx`, not the sub-cell distance), which biases
   the surface pressure gradient in the direction of propagation.
3. **Advection blend 0.25** adds `ν_num ≈ 0.25·uΔx/2 ≈ 3e-3 m²/s`, whose decay
   over 1λ is `exp(−k²νT) ≈ 0.6` — significant, and the blend exists only for
   stability. A TVD-limited reconstruction (or an energy-conserving central
   scheme with a kinetic-energy limiter) should be tried.
4. Vertical resolution: `a/Δy < 1` in this run; a `Δy ≤ a/3` grid is required
   before blaming the scheme alone (the reported number is from the coarse grid
   that fits the time budget of this node).

The structural parts (geometry gate, hydrostatics, mass conservation, theory
utilities, body force/moment bookkeeping) are verified; the *wave propagation
accuracy* is not, and no claim of a usable NWT is made.

## 5. Known defects / limitations (recorded, not hidden)

* **Stair-step buoyancy bias** of `Δy/(2T)` (8 % for `Δy/T = 1/6`) in the wetted
  pressure integral; the face pressure is extrapolated hydrostatically from the
  cell centre, which removes the leading bias but leaves ±`Δy/(2T)` depending on
  which side of the face the liquid cell is. Remedy: cut-cell areas or a
  ghost-fluid interface on the hull.
* The hull mask is rebuilt every step and whole cells switch between water and
  solid, so the water volume jumps by up to one cell per switch (visible in
  `volume_drift`); a body-volume ledger must be carried in production runs.
* Surge is fixed, pitch is clipped to ±0.30 rad; no viscous force on the hull.
* The tank is 2-D per unit length: the body is the ship's centreplane strip
  (rectangle `L×T` for the Wigley centreplane), not a 3-D hull.

## 6. Reproduce

```bash
python -m pytest tests/test_wave_tank_theory.py tests/test_wave_tank_solver.py -q
python scripts/wave_tank/run_cases.py geometry
python scripts/wave_tank/run_cases.py wave    --amp 0.03 --period 1.0 --nx 200 --ny 60
python scripts/wave_tank/run_cases.py control --amp 0.03 --period 1.0 --nx 200 --ny 60
python scripts/wave_tank/run_cases.py body    --amp 0.03 --period 1.1 --nx 200 --ny 60
```

Artifacts: `outputs/wave-tank/geometry_gate.json`, `smoke.json`,
`fig_wave_decay.png`. Every number quoted above comes from those files.