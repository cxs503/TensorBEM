# Linear cathodic-protection BEM on a 200 m bulk carrier — verification report

Module: `tensorbem.cathodic_protection` (`solve_cathodic_protection`,
`verify_conducting_sphere`, `run_verification_study`)
Benchmark: `python -m tensorbem.benchmarks.cathodic_protection_sphere --subdivisions 1 2 3`
Figures/numbers: `PYTHONPATH=src python docs/cathodic_protection_linear/make_figures.py`

Every number below is read from the JSON files produced by that script
(`results.json`, `sphere-spreading-resistance.json`, `hull_field.json`); nothing
is transcribed by hand.

---

## 1. Case introduction

**Physics.** Seawater is a homogeneous conductor (σ = 3.5 S/m at 15 °C). In the
steady state the electrolyte potential satisfies ∇²φ = 0 in the exterior
domain, with three boundary conditions on the wetted hull:

| surface | condition | DNV-RP-B401 role |
|---|---|---|
| intact coating (Neumann) | q_n = ∂φ/∂n = 0 | insulating |
| coating holiday (Dirichlet) | φ = −0.80 V vs Ag/AgCl | minimum protection |
| sacrificial anode (Dirichlet) | φ = −1.05 V vs Ag/AgCl | Al/Zn open-circuit |

The formulation is the *linear* precursor of the nonlinear module
`tensorbem.cp_galvanic` (which adds Butler–Volmer cathode kinetics and an
oxygen-transport limit). It prescribes potentials instead of kinetics, so its
value is as a numerically exact, cheaply verifiable Laplace/BEM engine for
current-distribution screening — and, as shown in §6, it is *not* a protection
predictor.

**Geometry.** Parametric 200 m bulk carrier (`BulkCarrierHull`, `IceClassHull`
family): L = 200 m, B = 32 m, T = 12 m, waterline exponent 6, section exponent
1.5. The open wetted surface is closed with an artificial planar fan in the
waterplane z = 0 (the *lid*) so that the boundary integral equation applies on a
closed surface.

| quantity | value |
|---|---|
| wetted steel area | 9606.3 m² |
| waterline lid area | 5372.2 m² |
| closed-mesh area | 14978.5 m² (lid = 35.9 %) |
| panels (n_sections = 11) | 222 (2 anode, 10 holiday at 5 % damage) |

**Parameters.** σ = 3.5 S/m; φ_prot = −0.80 V; φ_anode = −1.05 V; bare-steel
current density 0.150 A/m²; coating-breakdown current density 0.005 A/m²;
Gauss quadrature order 8; damage ratios 2/5/10 % of wetted area; one random
coating-damage draw at seed 42 (same draw reused across damage ratios).

## 2. Method

1. **Mixed BEM assembly.** From `assemble_surface_matrices` (Van Oosterom–
   Strackee solid angles + Duffy quadrature) build S and H; solve
   `S q = (1/2 I − H) φ` with q unknown on Dirichlet panels (holidays + anodes),
   φ unknown on Neumann panels (coating + lid), q_n = 0 prescribed there.
   Scipy-free dense LU (`numpy.linalg.solve`), condition number and residual
   recorded per case.
2. **Physical current (Ohm's law).** The BIE unknown q has units V/m; the
   physical seawater current density is `j_n = σ q`, positive where current
   leaves the metal (anodic). The conductivity-scaled field is
   `CathodicProtectionResult.current_density`; the raw q is kept as
   `single_layer_density` because only the raw field may be fed to the exterior
   representation formula.
3. **Exact anchor.** An equipotential sphere of radius a at φ_a in an unbounded
   electrolyte has φ(r) = φ_a a/r and total current I = 4πσaφ_a, i.e. the
   spreading resistance R = 1/(4πσa). Prescribing φ = φ_a on every panel of a
   nested icosphere family turns `solve_cathodic_protection` into exactly this
   problem, with the exact refinement ratio r = (N_fine/N_coarse)^0.5 = 2.
4. **Wetted-area accounting.** Lid panels are identified geometrically (all
   three vertices in z = 0), excluded from the damage/anode draw and from all
   wetted-area statistics; the DNV demand is evaluated on the wetted area of the
   *same* mesh as the BEM solve.
5. **Post-processing.** Damage-ratio sweep 2/5/10 %, protection coverage over
   wetted area, BEM-vs-DNV current comparison, JSON artefacts.

## 3. Results

### 3.1 Exact conducting-sphere anchor (C8)

a = 1 m, φ_a = −1 V, σ = 3.5 S/m ⇒ I_exact = 4πσaφ_a = −43.9823 A,
R = 0.0227364 Ω.

| icosphere level | panels | I_numeric [A] | error vs exact | max exterior φ error | linear residual |
|---|---|---|---|---|---|
| 1 | 80 | −41.8202 | 4.916 % | 4.947 % | 2.4e−16 |
| 2 | 320 | −43.4102 | 1.301 % | 1.309 % | 4.4e−16 |
| 3 | 1280 | −43.8364 | **0.332 %** | **0.334 %** | 7.5e−16 |

Observed order from the nested family: p = 1.918 (80→320), p = **1.971**
(320→1280) — second-order, as expected for a piecewise-constant single-layer
representation.

![sphere convergence](cathodic_protection_linear/figures/sphere_convergence.png)

### 3.2 Bulk-carrier damage sweep (n_sections = 11, 222 panels, seed 42)

| damage ratio | I_BEM [A] | I_DNV [A] | I_BEM/I_DNV | J_BEM [A/m²] | J_DNV [A/m²] | wetted mean φ [V] | max φ [V] | coverage | cond | residual |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 % | −116.66 | 75.89 | 1.537 | 0.01214 | 0.00790 | −0.1892 | −0.0683 | 4.56 % | 29.8 | 1.8e−15 |
| 5 % | −105.50 | 117.68 | 0.897 | 0.01098 | 0.01225 | −0.2998 | −0.1391 | 7.29 % | 70.2 | 1.6e−15 |
| 10 % | −94.40 | 187.32 | 0.504 | 0.00983 | 0.01950 | −0.4064 | −0.1986 | 10.56 % | 79.2 | 1.4e−15 |

J = current magnitude per wetted m². Sign: negative I means the net current
enters the anode panels in this prescribed-potential model (see §5).

![DNV comparison](cathodic_protection_linear/figures/dnv_comparison.png)
![hull potential field](cathodic_protection_linear/figures/hull_potential_field.png)

### 3.3 Criterion verdicts (`results.json → verification`)

| criterion | verdict | evidence |
|---|---|---|
| C2 coated/lid insulating | **PASS** | max |j_n| on coated panels = 0.0 A/m² (prescribed) |
| C7 JSON artefacts | **PASS** | potential, current, DNV demand, coverage, sphere anchor all present |
| C8 sphere current/field | **PASS** | 0.332 % / 0.334 % at 1280 panels, tolerance 3 % |
| C4 coverage ≥ 95 % | **FAIL** (honest) | 4.6 / 7.3 / 10.6 % |
| C5 anode demand grows with damage | **FAIL** (honest) | slope −272.5 A per unit damage, r² = 0.979 |

The evidence column is the study's *own* verification block, and `make_figures.py`
runs the C8 anchor on the same three-level icosphere ladder as §3.1
(`sphere_levels = (1, 2, 3)`), so `results.json → verification → C8` and
`sphere-spreading-resistance.json → levels[-1]` are the same 1280-panel solve.
Earlier the C8 block was evaluated on a truncated two-level ladder (1.301 % at
320 panels) while this table quoted the 1280-panel value; both are now pinned
to each other by `tests/test_cathodic_protection.py::
TestReportIsPinnedToArtifacts::test_c8_evidence_row_matches_verification_block`.

## 4. Points to watch

1. **The linear model cannot predict protection.** Coated steel is never
   polarized: only the prescribed panels sit at or below −0.80 V, so coverage
   can never exceed the holiday+anode area fraction. C4 must fail; it is now
   reported as a measured FAIL instead of a number nobody checks.
2. **Anode current is resistance-limited, not demand-driven.** The 0.25 V
   anode–holiday difference drives a current set by the (fixed) anode area and
   the spreading resistance of the holidays. Spreading more holidays lowers the
   resistance, so |I| *decreases* with damage — the opposite of the DNV demand
   trend (whose crossover with the BEM curve is at ≈5 % damage, ratio 0.897).
   The C5 gate used to accept this as a "monotonic increase" because it only
   tested r² > 0.95; it now also requires a positive slope, and honestly FAILs.
   The damage-driven physics needs the kinetic boundary condition of
   `tensorbem.cp_galvanic`.
3. **Sign convention.** With potentials *prescribed*, the model has no
   electrochemical direction: the anode panels are 0.25 V more negative than the
   holidays, so the Laplace field drives current *into* the anodes (I < 0).
   Only a kinetic formulation can produce current leaving the anode. Magnitudes
   and the σ-scaling remain meaningful; the sign does not.
4. **The waterline lid is not steel.** 35.9 % of the closed-mesh area is the
   artificial lid. Earlier revisions assigned holidays and anodes to it and used
   the closed area in the DNV comparison, inflating the demand by 56 %
   (14978.5 vs 9606.3 m²).
5. **Conductivity factor.** Reporting q as a current density without the σ of
   Ohm's law underestimated every current by exactly 1/σ = 28.57 % (a 71.43 %
   error against the exact sphere). The anchor of §3.1 is what makes this class
   of defect impossible to reintroduce silently.
6. **Hull mesh family is not a systematic refinement.** Halving the section
   spacing is not achieved by n_sections = 9, 11, 13, and the discrete anode
   count (1 % of wetted panels → 2, 2, 2 panels) changes with it, so successive
   differences of J do not decay (0.001397 → 0.001384 A/m²). No order of
   accuracy or GCI is claimed for the hull; the refinement evidence rests on the
   nested icosphere. Hull mesh convergence needs a geometry-preserving
   refinement (anode patch fixed in physical area) — open item.
7. **Scope.** Planar triangular panels, piecewise-constant single layer, dense
   LU: O(N³) solve and O(N²) memory. The parameter loop in
   `solve_cathodic_protection` is pure Python, so ≳2000 panels is slow; the
   1280-panel case takes ~10 s. No free-surface/symmetry plane, no seawater
   depth variation, no coating capacitance/transient, no anode consumption.

## 5. Error judgement (key quantities vs benchmark)

| quantity | benchmark | BEM | relative error | < 3 %? |
|---|---|---|---|---|
| total current, conducting sphere (1280 panels) | −43.9823 A (exact 4πσaφ_a) | −43.8364 A | **0.332 %** | **yes** |
| exterior potential φ_a a/r (r = 1.5…5 m) | −0.6667 … −0.2000 V | — | **0.334 %** max | **yes** |
| sphere spreading resistance 1/(4πσa) | 0.0227364 Ω | 0.0228121 Ω (φ_a/I, 1280 panels) | 0.332 % | yes |
| linear-system residual (all cases) | 0 | ≤ 1.9e−15 | — | yes |
| anode current vs DNV-RP-B401 demand | 75.9 / 117.7 / 187.3 A | 116.7 / 105.5 / 94.4 A | **+53.7 % / −10.4 % / −49.6 %** | **no** |
| protection coverage vs DNV ≥ 95 % | 95 % | 4.6 / 7.3 / 10.6 % | −90 … −85 pp | **no** |

The two "no" rows are **honest FAILs of model physics, not of the solver**: the
prescribed-potential formulation omits the polarization kinetics that produce
both the damage-driven anode demand and the cathodic polarization of coated
steel. They are reported as FAIL by `run_verification_study` and pinned by
`tests/test_cathodic_protection.py::TestStudyHonesty::test_criteria_are_measured`.
The quantities the linear model *can* represent (the Laplace solution, Ohm's law
scaling, current density per wetted area, the exact sphere limit) are inside 3 %
— the sphere anchor is inside 0.4 %.

## Reproduce

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.cathodic_protection_sphere \
    --subdivisions 1 2 3 --output artifacts/cathodic_protection/sphere-spreading-resistance.json
PYTHONPATH=src python docs/cathodic_protection_linear/make_figures.py     # figures + results.json
PYTHONPATH=src python -m pytest tests/test_cathodic_protection.py -q      # 25 tests
```