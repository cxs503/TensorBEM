# CASE-ACOUSTIC-PULSE — transient acoustic pulse of a pulsating sphere

> Ledger anchor: **CASE-ACOUSTIC-PULSE** (case pool; reference = closed-form
> literature solution; tolerance **< 8 %**).
> Modules: `src/tensorbem/acoustic3d_pulse.py` (new),
> `src/tensorbem/benchmarks/acoustic3d_pulse_case.py` (driver).
> Tests: `tests/test_acoustic3d_pulse.py`.
> Archived numbers: `docs/acoustic3d_pulse/case.json`,
> `docs/acoustic3d_pulse/results_tables.md` (both **generated**, never retyped).

## 1. What problem this closes

`tensorbem.acoustic3d` is an exterior **Helmholtz** BEM: it answers *"what is the
steady-state pressure of one harmonic?"*. The underwater-noise and sonar
questions that matter are **broadband**: a machinery knock, a cavitation
collapse, a ping. For those the deliverable is a *waveform* — peak level and
arrival time — and before this case the repository had neither a bridge from the
harmonic solver to the time domain nor a transient reference to check such a
bridge against. The acoustic capability was therefore verified only
harmonic-by-harmonic (`docs/acoustic3d_helmholtz.md`), which cannot detect a
systematic phase/dispersion error that is invisible at a single frequency.

## 2. Why a pulsating sphere is the right reference body

The sphere is the only exterior acoustic geometry with a closed-form transient
solution, and — crucially — the closed form needs **no** far-field, no
compact-source and no `ka << 1` approximation, so the reference is valid for the
whole band used here (`ka` up to ~1.7).

For the spherically symmetric wave equation the pressure in the whole exterior
domain is

```
p(r, t) = (1/r) g(t - r/c),      r > 0,
```

i.e. an exact outgoing spherical wave, not an asymptotic one. Writing the
velocity potential as `phi(r, t) = F(t - r/c)/r` and imposing the prescribed
outward radial velocity `u_a(t) = d(phi)/dr|_{r=a}` gives the first-order ODE

```
F'(tau) + (c/a) F(tau) = -c a u_a(tau + a/c)                        (1)
```

whose exact solution is the exponentially weighted convolution

```
F(tau) = -c a * int_{-inf}^{tau} u_a(s + a/c) exp(-(c/a)(tau - s)) ds,
p(r, t) = -rho F'(t - r/c) / r.                                     (2)
```

The sign in (2) is the one of `tensorbem.acoustic3d`: the library's pressure
amplitude is `p = +i*omega*rho*phi` while the physical pressure of a velocity
potential with time factor `exp(-i*omega*t)` is `rho*d(phi)/dt =
-i*omega*rho*phi`. The global sign of a pressure field is immaterial for levels,
but it must be fixed once so that the harmonic and transient routes can be
compared term by term; `exact_monopole_pulse_pressure` returns the library
convention.

Consistency with the harmonic module: for a harmonic drive `u_a = U e^{-i omega t}`
equation (1) gives `F = -U a^2 e^{-i k a}/(1 + i k a)`, hence the closed-form
amplitude

```
p_hat(r) = i omega rho a^2 U exp(i k (r - a)) / ((1 - i k a) r),
```

which is exactly `tensorbem.acoustic3d.pulsating_sphere_pressure` (asserted in
`tests/test_acoustic3d_pulse.py::test_exact_pulse_amplitude_matches_library_analytic`
to `1e-12`).

## 3. Method: from harmonics to a waveform

A pulse that fits inside an analysis window `T` has a **line spectrum** on the
DFT bins `omega_n = 2*pi*n/T`, and for a band-limited signal the line sum is the
exact signal — no information is lost by sampling the frequency axis on the
bins instead of a continuum. That is what makes the frequency-domain BEM usable
for a transient at all:

```
U_n = int_0^T u_a(t) exp(+i omega_n t) dt        (trapezoid, spectrum)
p(t) = (2/T) Re[ sum_n P_n exp(-i omega_n t) ],  P_n = T(omega_n) U_n
```

Three routes are run on the same pulse:

| route | what it is | what it measures |
| --- | --- | --- |
| **A** | equation (2), integrated with the exact integrating factor on a `substeps`-refined grid | the reference itself (convergence checked against substeps and against a refined output grid) |
| **B** | exact closed-form amplitude on the DFT bins, synthesised | the *synthesis machinery* only (no BEM) |
| **C** | `acoustic3d.solve_exterior_neumann` + `field_potential` per bin, synthesised | the capability under test |

Route B separates discretisation from discretisation *of the boundary*. If A and
B agree, then any A-vs-C residual is BEM error; if they did not agree the test
would be measuring the wrong thing.

### 3.1 Case definition

| quantity | value |
| --- | --- |
| sphere radius `a` | 0.1 m (keeps `ka <= 1.7` over the band) |
| sound speed `c` / density `rho` | 1500 m/s / 1025 kg/m³ (seawater) |
| receiver range `r` | 2 m = 20 radii (outside the 1/r near field) |
| pulse | Gaussian radial velocity, `sigma = 0.25 ms`, centre `0.75 ms` |
| window / samples | `T = 6 ms`, 2048 samples (`dt = 2.93 µs`) |
| band | 4 kHz → 24 bins; pulse spectrum there is `exp(-(pi*sigma*f)^2) ≈ -86 dB` |
| mesh | `icosphere` subdivisions 1 and 2 (80 and 320 panels) |

## 4. How to reproduce

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.acoustic3d_pulse_case \
    --subdivisions 1,2 --band-check \
    --json docs/acoustic3d_pulse/case.json \
    --markdown docs/acoustic3d_pulse/results_tables.md
PYTHONPATH=src python -m pytest tests/test_acoustic3d_pulse.py -q
```

The driver prints the route table and returns exit status 0 only when the
acceptance criterion is met; all tables in `results_tables.md` are rendered from
the JSON payload of the same run.

## 5. Measured results

See `docs/acoustic3d_pulse/results_tables.md` (generated). Summary of what the
tables contain:

1. route A self-convergence (substeps 16 vs 32, output-grid refinement),
2. route B vs route A — the synthesis error floor,
3. route C vs route A per mesh: peak pressure, arrival time, per-bin harmonic
   error and the L2 waveform error against the 8 % acceptance,
4. band-truncation study on the finest mesh (1/2/4/8 kHz),
5. per-bin harmonic error tables keyed by bin frequency.

## 6. Caveats / scope

* The reference body is a **sphere**, so the case verifies the radiation path
  (kernel, quadrature, radiation condition, synthesis) but **not** scattering
  from a non-spherical hull: a blunt body's transient response additionally
  exercises CHIEF points and shadowing, which `docs/acoustic3d_helmholtz.md`
  covers only harmonically. A rigid-sphere transient (Mie series) is the natural
  next anchor.
* Route C's error is dominated by the *panel* approximation of the sphere
  (`icosphere` panels are flat chords inside the true sphere), which is why the
  error falls at roughly the observed convergence order rather than at the
  Helmholtz quadrature order.
* The band is truncated at 4 kHz. The band study shows where that truncation
  starts to bite; the pulse spectrum itself is ~-86 dB there, so the truncation
  is not the error source at the reported mesh levels.
* The synthesis assumes the whole pulse fits in the window (periodic extension
  is then legitimate). Pulses that wrap the window need a longer `T`, not a
  larger band.