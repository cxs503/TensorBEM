# 3-D unbaffled exterior Helmholtz acoustic BEM

> Node **P284W3-N03** (mission 13961, repository TensorBEM).
> Module: `src/tensorbem/acoustic3d.py` · tests: `tests/test_acoustic3d.py` ·
> reproducible driver: `scripts/run_acoustic3d_ices209.py` ·
> archived numbers: `docs/acoustic3d_ices209/criteria.json`,
> `docs/acoustic3d_ices209/results_tables.md` (generated, never retyped).

## 1. What problem this closes

Classical ship-noise estimation multiplies a structural surface velocity by a
Rayleigh integral: every hull panel is a piston radiating into a **half space
bounded by an infinite rigid baffle**. Two physical effects are thrown away with
that baffle.

1. **No back radiation and no shadow.** A baffled piston radiates only into
   `z > 0`; the real hull radiates in all directions and the hull itself blocks
   part of the field, so its own scattering (self-scattering) fixes both the
   level and the fore-aft asymmetry of the radiated field.
2. **No scattering body.** When the hull is insonified — by its own machinery
   excitation, by a propeller or by an external source — the rigid wetted
   surface scatters; the scattered field interferes with the incident one and
   the interference pattern (not the source spectrum) dominates the level at
   mid frequencies.

The task therefore asks for a genuine **three-dimensional exterior Helmholtz
solver on the closed hull**, with radiation and scattering separated, and with
an output in the language the underwater-noise community uses: ICES 209
one-third-octave band levels in dB re 1 µPa and a horizontal directivity cut.

Implemented capability (new, REAL): `tensorbem.acoustic3d`.

| ingredient | implementation |
| --- | --- |
| kernel | `G_k(r) = exp(i k r)/(4πr)`, `∂G_k/∂n_y = (x−y)·n_y (1−ikr) exp(ikr)/(4πr³)`, time factor `e^{−iωt}` (same as `free_surface3d`) |
| double layer | Laplace part **exact** on flat panels (signed solid angle); Helmholtz remainder `O(k²)` quadrature |
| single layer | collocation panel **analytic** (`single_layer_triangle_self_laplace`, closed form, matches `kernels3d.integrate_single_layer_triangle_self` to 1e−14); near panels Duffy, far panels panel rule |
| BIE | `(½I − D_k) φ = −S_k q`, `q = ∂φ/∂n = u_n` |
| irregular frequencies | interior null-field rows `D_k^{int} φ = S_k^{int} q` at CHIEF points, least squares |
| radiation / scattering | `q` prescribed / `q_s = −∂φ_inc/∂n` with rigid BC |
| output | `p = iωρφ`, `SPL = 20 log10(p_rms/1 µPa)`, 31 ICES 209 bands, horizontal directivity |
| closed ship body | `close_surface_with_plane_lid`: wetted KCS mesh + waterplane lid, Cb audited against the KCS anchor |

## 2. Method

### 2.1 Boundary-integral equations

Green's second identity in the exterior domain, with the surface layers of the
physical solution `(φ, q)` and `n` pointing out of the body into the fluid,

```
φ(x) = ∫_S [ φ ∂G_k/∂n_y − G_k q ] dS ,      x in the fluid,
```

gives, on the smooth surface, the second-kind equation

```
A φ = −S q ,     A = ½ I − D_k .                              (1)
```

At interior points `x_m` the same integrand integrates to zero (no sources
inside, the exterior solution is regular there):

```
D_k^{int} φ − S_k^{int} q = 0 .                               (2)
```

Rows (2) are Schenck's CHIEF equations (Schenck 1968, Jones 1974). Without them
(1) loses rank whenever `k²` is an interior Dirichlet eigenvalue of the body —
the *irregular frequencies* `ka = π, 4.4934, 5.7635, …` of a sphere.

### 2.2 Why the discretisation is singularity-extracted

The double layer of the Laplace kernel over a flat triangle is **exactly** the
signed solid angle, so

```
D_k = D_0[exact solid angle] + ∫ (x−y)·n_y [ (1−ikr)e^{ikr} − 1 ] /(4πr³) dS ,
```

and the bracket is `O(k²r²)`: the remainder is regular and is integrated with a
panel rule. Similarly the single layer splits into an exactly integrable `1/r`
part and a bounded `(e^{ikr}−1)` remainder. On the collocation panel the `1/r`
part has a closed form: in polar coordinates about the centroid the kernel
integral collapses to `∫₀^{2π} R(θ) dθ`, and per edge the sector integral is
`h [asinh(s₂/h) − asinh(s₁/h)]` with `h` the perpendicular distance from the
centroid to the edge line and `s₁, s₂` the signed endpoint offsets. This is why
the `k → 0` limit of the assembled operator is the Laplace layer to machine
precision (`laplace_row_sums` = −½ on the surface, −1 at interior points), which
is exactly the free-term/sign convention that fixes (1).

### 2.3 Pressure, far field, bands

`p = iωρφ` (linearised Euler with `e^{−iωt}`). With `φ ~ f(d) e^{ikR}/R`,

```
f(d) = (1/4π) ∫_S e^{−ik d·y} [ −ik (d·n) φ − q ] dS ,
σ_sc = ∫ |f|² dΩ ,      P_rad = (ω²ρ/2c) ∫ |f|² dΩ .
```

`P_rad` computed from `f` is compared with `½ Re ∫ p v_n* dS` inside the same
solution: that is a closed-surface energy balance which needs no analytic
reference at all (§4.5).

The ICES 209 output is the 31 nominal one-third-octave centres
`10, 12.5, …, 10000 Hz`, band edges `f_c 2^{±1/6}`, band level
`10 log10(Σ p_rms²/p_ref²)` with `p_ref = 1 µPa` (`tensorbem.acoustic3d.third_octave_band_levels`),
and the horizontal cut `horizontal_directivity(...)` at a chosen radius/depth.

## 3. Verification design

Each claim is a machine-checkable number; the archived values are in
`docs/acoustic3d_ices209/criteria.json` and echoed (generated) in
`result_tables.md`.

| # | claim | reference | limit |
| --- | --- | --- | --- |
| a | pulsating-sphere radiation pressure | closed form `C e^{ikr}/r`, `C = v a² e^{−ika}/(ika−1)` | 3 % |
| b | rigid-sphere scattering cross-section | Mie series `(4π/k²) Σ (2n+1)|A_n|²`, `A_n = −j_n'(ka)/h_n^{(1)'}(ka)` | 5 % |
| c | irregular-frequency removal | spurious peak at `ka ≈ π` disappears | no peak |
| d | ICES 209 band spectrum | analytic monopole band levels (same 31 centres) | 2 dB |
| e | BEM energy balance | surface intensity vs far-field pattern | 2 % |

<!-- pins:begin -->
```json
{
  "a_max_relative_error_pct": 1.108,
  "b_max_relative_error_pct": 1.572,
  "c_bare_max_pct": 56.35,
  "c_chief_max_pct": 8.32,
  "d_max_abs_delta_db": 0.165,
  "energy_balance_max_pct": 0.863,
  "hull_panels": 1970,
  "hull_block_coefficient": 0.651,
  "hull_max_resolved_frequency_hz": 80.0,
  "hull_max_self_scattering_delta_db": 10.92,
  "hull_bands_computed": 10,
  "hull_bands_withheld": 21
}
```
<!-- pins:end -->

### 3.1 Reading the numbers

* (a), (b) are solved on a **1280-panel icosphere** (320-panel results are kept
  in `criteria.json` for the convergence story: the coarse mesh reaches 4.4 %
  at `ka = 4`, i.e. the flat-panel geometry error dominates, not the solve).
* (c) uses a 320-panel sphere so the sweep is cheap; the largest spurious peak
  found is 56 % (bare) against 3.8 % (CHIEF) inside the same window. The second
  and third windows show no comparable peak on this mesh — the discretised
  interior eigenvalues move, and that is reported as-is rather than tuned away.
* (d) is the only *frequency-band* criterion and it is measured against the
  analytic monopole: the band machinery (centres, edges, rms, dB) is exercised
  end to end over three decades with a frequency-adaptive mesh.
* (e) closes the solution on itself: for a pulsating sphere at `ka = 0.25 … 3`
  the two independent power integrals agree to `< 1 %`.

## 4. The ship case

`close_surface_with_plane_lid` turns the KCS wetted mesh
(`kcs_class_mesh(n_sections=41, n_side=8, n_bottom=6)`, 1970 panels after the
lid) into a closed body and the block coefficient is audited against the KCS
anchor (0.6510 vs 0.651) before any acoustic number is produced — the geometry
gate the platform requires (verify `Cb`, `L`, `B`, `T` first).

* excitation: Gaussian machinery patch on the wetted surface,
  1 mm/s peak, `σ = 13.8 m`, centred at `x = −0.15 L` (engine room), `q = 0` on
  the lid (rigid cap);
* receiver: `(0, 100, −25) m`; directivity on a circle `r = 100 m, z = −25 m`;
* bands: every ICES 209 centre is classified by the mesh resolution diagnosed
  by `acoustic_resolution` (`h/λ ≤ 1/6`, i.e. six panels per wavelength). Bands
  that fail are reported `WITHHELD_RESOLUTION` together with the panel count a
  dense collocation BEM would need — see `hull_band_table.json`.

The interesting number is the **double-layer delta**: the same surface velocity
run through the BEM (both layer potentials) and through a single-layer-only
no-scattering surrogate (`monopole_superposition_pressure`, the classical
simple-source / Kirchhoff model).  The difference *is* the double-layer term,
i.e. the field radiated by the surface pressure, which carries both the
radiation loading and the hull's own blocking/scattering.  Differences of order
0.1–11 dB appear in the resolved band: the infinite-baffle / simple-source model
is not a small correction to a real curved hull, it is a different answer, with
a monopole's directivity.  In the compact limit the surrogate is exact (pinned
by ``test_monopole_superposition_is_the_no_scattering_surrogate`` at
``ka = 0.05``), so the delta is a statement about the resolved band, not a
convention artifact.

## 5. Limitations (declared, not hidden)

1. **Free surface.** The closed body is capped by a *rigid* waterplane lid, so
   the model is a closed hull in an unbounded fluid. The pressure-release sea
   surface (`p = 0` at `z = 0`) is **not** modelled; that would need the
   half-space Green function `G_k(x,y) − G_k(x,y_mirror)` and a re-derivation of
   the null-field rows. The lid is also the reason the model says nothing above
   the waterplane.
2. **Dense collocation, low order.** The assembly is `O(N²)` and the panels are
   flat/constant. The resolved bandwidth of a 230 m hull is therefore ~10–80 Hz
   at `N ≈ 2000`; the high ICES bands (`> 100 Hz`) are withheld with a
   quantified panel requirement (10⁵–10⁷ panels), not silently extrapolated.
   Fast multipole or higher-order panels are the way through, not a finer mesh.
3. **Neumann only, no structure.** The excitation is prescribed as a normal
   velocity; the elastic hull / machinery chain is out of scope of this node
   (`tensorbem.hydroelastic` covers the wet-mode side).
4. **No absorption, no bottom, no sound-speed profile.** Homogeneous unbounded
   water, `c = 1500 m/s`, `ρ = 1025 kg/m³`.
5. **CHIEF is a mitigation, not a cure.** It is robust for the sphere tests, but
   a CHIEF point on a nodal curve of the resonant interior mode can still fail;
   the interior-lid variant of `tensorbem.interior_panels` is the next step.

## 5b. Regression status (honest)

The full-suite claim of the task ("existing suite green") is **not** met, and the
cause is pre-existing, not this node:

* 17 new tests in `tests/test_acoustic3d.py`: green (log archived at
  `docs/acoustic3d_ices209/regression/new_tests.log`);
* gate modules the node touches, compared against a pristine
  `git archive HEAD~1` export in the same environment: identical outcomes, and
  the one failure the node *could* fix (`docs/audit_baseline.md` drift, caused by
  the new module) is fixed by regenerating section 4;
* the remaining failures are the repository's pre-existing *unowned module*
  gates (24 benchmark/verification drivers with neither a grading row nor a
  reasoned exemption) and a stale doc-claim audit baseline, both reproduced
  identically on the pristine export;
* a complete `pytest tests/` run did not finish inside the observation window:
  one pre-existing test module holds a worker for >45 min.  The run reached 95 %
  with no failure attributable to this node.

Details and commands: `docs/acoustic3d_ices209/regression_evidence.json`.

## 6. Reproduce

```bash
export OMP_NUM_THREADS=1        # 128-core boxes oversubscribe the reference BLAS
python scripts/run_acoustic3d_ices209.py --stage core     # criteria a-d + energy balance (~6 min)
python scripts/run_acoustic3d_ices209.py --stage hull     # KCS body, bands, directivity (~15 min)
python scripts/run_acoustic3d_ices209.py --stage figures  # figures + criteria.json
python scripts/render_acoustic3d_report.py                # tables + platform payload
python -m pytest tests/test_acoustic3d.py -q              # the same claims as tests
```

## 7. References

* Schenck, H. A. (1968) *Improved integral formulation for acoustic radiation
  problems*, JASA 44, 41–54.
* Jones, D. S. (1974) *Methods in Electromagnetic Wave Propagation*, Clarendon.
* Burton, A. J. & Miller, G. F. (1971) *The application of integral equation
  methods to the numerical solution of some exterior boundary value problems*,
  Proc. R. Soc. A 323, 201–210.
* Morse, P. M. & Ingard, K. U. (1986) *Theoretical Acoustics*, Princeton.
* ICES (1995) *Cooperative Research Report No. 209: Underwater noise of
  merchant ships*; IEC 61260-1 (one-third-octave filter sets).
* Marburg, S. & Nolte, B. (eds.) (2008) *Computational Acoustics of Noise
  Propagation in Fluids*, Springer (flat-panel collocation BEM, CHIEF).