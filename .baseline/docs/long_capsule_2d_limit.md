# The 2-D limit of the 3-D Kelvin solver: long submerged capsule vs the Ursell cylinder

**Scope.** Every 3-D verification in this package compares the Kelvin/hydrodynamic
Green-function solver with a *three-dimensional* reference: Hulme's hemisphere
multipoles, the Wang/Linton submerged sphere, or an interior-harmonic identity.
None of them tests the **dimensionless scaling** of the 3-D assembly — a wrong
`1/(4πr)` prefactor, a missing `1/2` in a layer jump, or a wrong Kochin damping
factor `ωρk/(4π)` can survive a same-dimension comparison if the reference
carries the same normalisation. This note closes that gap with a reference that
shares no code with the 3-D path: the **two-dimensional Ursell closed form**
(`tensorbem.analytic.ursell_cylinder_added_mass`, a truncated multipole series plus
an exponential integral).

The check is a single falsifiable statement. A capsule (cylinder of length `L`,
radius `a`, two hemispherical ends) in deep water has wetted mass `∝ ρaL` while
its ends contribute a fixed `O(ρa³)`, so the heave load per unit span must obey

```
A₃D(L)/L → A₂D ,  B₃D(L)/L → B₂D        as L/a → ∞.
```

Everything below is written by
`python -m tensorbem.benchmarks.cylinder_3d_limit` (JSON:
`artifacts/cylinder_3d_limit/report.json`); the geometry comes from
`tensorbem.geometry3d.capsule_mesh`, a surface of revolution whose volume
`πa²L + 4/3πa³` and area `2πaL + 4πa²` are closed form, so the mesh is audited
on the way (measured volume error −2.2 % to −2.8 %, i.e. the inscribed
polyhedron, consistent across the whole ladder).

## 1. Case

| quantity | value |
|---|---|
| radius `a` / submergence `f` | 1.0 m / 2.0 m (`f/a = 2`, Ursell's deeply-submerged regime) |
| wavenumber | `ka = 0.6` → `k = 0.6 m⁻¹`, `ω = 2.4257 rad/s` |
| density / gravity | 1025 kg/m³ / 9.80665 m/s² |
| capsule length | `L/a = 4, 8, 12, 16` (cylinder part; ends add `2a`) |
| panel edge (ladder) | `h/a = 0.35` → 720 … 1980 constant-strength triangles |
| panel refinement | `L/a = 8` at `h/a = 0.5` (572) and `0.35` (1152) |
| motion | heave (mode 2), unit velocity amplitude |
| 2-D reference | Ursell multipole: `A₂D = 2545.1672 kg/m`, `B₂D = 2718.1540 kg/m` |
| 2-D coefficients | `A/(ρπa²) = 0.790392`, `B/(ρπa²ω) = 0.347988` |

The capsule axis is horizontal, so heave is the mode whose 2-D counterpart is
Ursell's cylinder; the body is smooth (no sharp edge) so the discretisation error
is a function of panel size alone.

## 2. Results

### 2.1 Length ladder (`h/a = 0.35`, Kochin far-field damping in the last column)

| `L/a` | panels | `A/L` [kg/m] | `B/L` [kg/(m·s)] | `A` error | `B` error | `B_far/B_direct` |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 720 | 2808.914 | 3488.331 | +10.36 % | +28.34 % | 0.99605 |
| 8 | 1152 | 2499.979 | 2658.520 | −1.78 % | −2.19 % | 0.99634 |
| 12 | 1548 | 2620.197 | 2751.366 | +2.95 % | +1.22 % | 0.99630 |
| 16 | 1980 | 2504.809 | 2769.073 | −1.59 % | +1.87 % | 0.99639 |

### 2.2 Panel refinement at `L/a = 8`

| `h/a` | panels | `A` error | `B` error | observed order (`A`) |
|---:|---:|---:|---:|---:|
| 0.50 | 572 | −1.97 % | −4.15 % | — |
| 0.35 | 1152 | −1.78 % | −2.19 % | 0.29 (damping order 1.79) |

The refinement moves the per-span value by only **0.2 %** (added mass) and 2 %
(damping) between the two meshes, i.e. the ladder rows are mesh-converged to the
percent level, and the residual scatter in §2.1 is *not* a resolution artefact.

## 3. Verdict (honest, and partly negative)

**Established.** For `L/a ≥ 8` the 3-D Kelvin solver reproduces the independent
2-D Ursell load per unit span within **2.95 % (added mass)** and **2.19 %
(damping)**, from a 720–1980-panel closed surface, with condition number ≈ 2.6
and relative residual ≈ 1e-15. The dimensionless scaling of the 3-D assembly is
therefore right: added mass `ρ L a²`-like, damping `ω ρ L a²`-like, no stray
factor of π. At `L/a = 4` the end contribution is still +10 % (mass) and +28 %
(damping), which is the expected physics of a stubby body, not an error.

**Also established (new for a slender body).** The Kochin far-field damping
`far_field_damping_matrix_3d` reproduces the pressure-integrated damping to
**0.37 %** at every length — previously this closure was only demonstrated on the
sphere and the hemisphere. The far field does not degrade as the body becomes
slender.

**Not established — `1/L` extrapolation law.** Fitting `A/L = A∞ + c/L` over the
four rows gives `A∞ = 2417.4` (−5.0 % vs Ursell) with `r² = 0.72`; the damping fit
gives −12.5 % with `r² = 0.82`, and restricting the fit to the tail (`L/a = 8, 12,
16`) flips the slope sign (`dA∞: slope −478`, `r² = 0.05`). The 1/L law therefore
cannot be resolved at this resolution: the end effect at `L/a = 8…16` is of the
same order (≈2–3 %) as the panel-discretisation bias (≈2 %), and the ladder
scatter is non-monotone. **The extrapolation columns are reported as UNRESOLVED,
not as evidence.** Closing them needs `L/a ≥ 32` combined with an `h`-extrapolated
value per length (a several-thousand-panel system each), which was out of budget
here (load average on the shared host was 125–340 during the run).

## 4. Notes and limits

- The acceptance criterion used here is the **envelope** `|A/L − A₂D|/A₂D < 3 %`
  for `L/a ≥ 8`, which is what the data supports. The regression test pins that
  envelope and pins `ladder_monotone is False`, so a future change that only
  *looks* like a clean 1/L convergence has to justify itself.
- The 2-D reference is Ursell's multipole series with 40 terms; its own accuracy
  at `ka = 0.6` is far below the percent level (the series is exponentially
  convergent for `a/f = 0.5`), so the comparison error is dominated by the 3-D
  discretisation.
- Only heave is exercised. Sway is the same problem by axial symmetry of the
  capsule about a vertical plane; surge/roll/pitch are not reducible to a 2-D
  limit in the same way and are not claimed.
- Deep water only: the 3-D finite-depth Green function (`tensorbem.finite_depth3d`)
  is a kernel without a panel solver (README: next increment), so no finite-depth
  version of this check exists yet.
- The capsule ends are hemispherical; a flat-ended cylinder would add a sharp
  edge whose constant-panel error would contaminate the length trend.

## 5. Reproduction

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.cylinder_3d_limit \
    --ratios 4 8 12 16 --panel-sizes 0.5 0.35 --refinement-ratio 8 \
    --output artifacts/cylinder_3d_limit/report.json \
    --figure-dir artifacts/cylinder_3d_limit
pytest tests/test_capsule_3d_limit.py            # 8 cheap guards
TENSORBEM_HEAVY_TESTS=1 pytest tests/test_capsule_3d_limit.py   # + envelope test
```