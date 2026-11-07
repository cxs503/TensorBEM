# BEM verification anchors: Wigley III hydrostatics, ellipsoid wave resistance, tandem interference

*Campaign node **P19-N02**, TensorBEM.  Every number below is rendered from the
committed artifacts `benchmarks/verified/*/results.json` by
`tensorbem.benchmarks.bem_verified_anchors.render_document`; the test
`tests/test_bem_verified_anchors.py` re-runs the three studies and fails if the
code, the artifacts and this document disagree.*

Verdict of the three anchors: **ALL PASS**
(`bem_wigley_hydrostatics` = PASS (strict all-levels: FAIL), `bem_ellipsoid_wave` = PASS, `bem_multihull_interference` = PASS).

## 1. What was studied, and why

TensorBEM already carried the ingredients of a panel-method verification — a
wetted-surface triangulator (`geometry3d`), divergence-theorem hydrostatics
(`hydrostatic`), the Michell/Kochin thin-ship machinery for the parabolic Wigley
hull (`wigley`), the Havelock/Bessel ellipsoid anchor (`ellipsoid_kochin`) and a
Roache grid-convergence helper (`uq`) — but no artifact tied them together at
*prescribed mesh densities*, and the multi-hull (tandem) case had no anchor at
all.  The three studies below close that gap on the classical hullforms:

| anchor | hull, parameters | reference | panel route under test |
|---|---|---|---|
| `bem_wigley_hydrostatics` | Wigley III, L = 3.8 m, B = 0.5 m, T = 0.225 m | closed-form parabola integrals: ∇ = 4LBT/9 = 0.190000000 m³, Cb = 4/9, A_wp = 2LB/3, S_w by Gauss–Legendre | 200 / 800 / 3200 wetted triangles (10x5, 20x10, 40x20), divergence theorem |
| `bem_ellipsoid_wave` | half-submerged ellipsoid a = 1.0, b = 0.2, c = 0.15, Fr = 0.20/0.30/0.40 on L = 2a = 2.0 m | Havelock/Michell closed form (Bessel J1 reduction) | 128 / 512 / 2048 constant-strength Rankine panels on the centreplane |
| `bem_multihull_interference` | two identical Wigley III hulls in line, d = L/2, L, 2L, Fr = 0.30 | Michell Kochin superposition σ = R_W,pair/(2R_W,single) | same three panel densities (200 / 800 / 3200) |

**Geometry gate.**  Following the acceptance rules, geometry is checked before
any error is quoted.  The meshed Wigley hull reproduces L = 3.800000 m,
B = 0.500000 m, T = 0.225000 m and
Cb = 0.444097 (closed form 0.444444); the
ellipsoid quadrature reproduces ∇ = 2πabc/3 to
-0.000000 % with Cb = π/6 = 0.523599.

*Brief audit.*  The mission brief quotes Cb = 0.667 for this hullform. 0.667 is its prismatic coefficient Cp = Cb/Cm (Cm = 2/3); the displaced-volume coefficient of the parabolic Wigley hull is Cb = 4/9 = 0.4444. Using 0.667 in V = Cb L B T over-predicts displacement by 50 %.

## 2. How it was done

**Hydrostatics (3-D panels).**  `wigley_wetted_hull_mesh` generates the closed
wetted surface (port + starboard sheets sharing the post meridians and the keel,
waterline open) for a given `(n_sections, n_levels)`; the displaced volume is the
divergence-theorem integral over the triangulation, the waterplane area comes
from the traced waterline loop, and the wetted area is the faceted triangle sum.
All four quantities are compared with the *closed-form* parabola integrals, and
a three-level Roache GCI (refinement ratio r = 2.0, safety factor Fs = 1.25) is
reported for ∇ and S_w.

**Wave resistance (thin-ship Rankine panels).**  The Michell Kochin function
H(k) = 2∫∫ y_x e^{kz+ikx} dx dz is evaluated three ways: the Bessel closed form
(`kochin_analytic`, the reference), a definition quadrature, and a *constant-strength
panel sum* on the centreplane projection (`kochin_panels`, panels clustered
towards the elliptical rim).  The Kelvin-wake integral
R_W = (4ρ/U²)∫₁^∞ |H(k₀λ)|²/(λ²√(λ²−1)) dλ is evaluated with the λ = 1 + s²
substitution that removes the transverse-wave cusp; the same routine serves the
reference and the panel route, so the comparison measures the *discretisation*
and not the quadrature.  For the Wigley hull the analogous centreplane panel
discretisation lives in `tensorbem.multihull` (new in this node) and is checked
against `michell_kochin_function`.

**Tandem interference.**  Linear superposition in the Kochin function gives
H_pair(k) = H(k)(1 + e^{ikd}) for two identical hulls separated by d, hence
|H_pair|² = 2|H|²(1 + cos kd) and

  σ(d) = R_W,pair/(2 R_W,single) = [∫ w(λ)|H|²(1 + cos k₀λd) dλ] / [∫ w(λ)|H|² dλ].

σ reduces to 2 for d → 0 (two coincident hulls) and oscillates around 1 with a
slowly decaying envelope as d grows.  The panel route replaces H by the panel
sum; the superposition itself is verified *without* using the factorisation, by
concatenating the panels of both hulls (`explicit_pair_kochin`), which must agree
with H(1 + e^{ikd}) to round-off (measured worst relative deviation below
2.06e-15).

**Tool chain.** NumPy/SciPy + matplotlib only (`python -m
tensorbem.benchmarks.bem_verified_anchors --out-root benchmarks/verified --write-doc
docs/bem_verified_anchors.md`).

## 3. Results

### 3.1 Wigley III hydrostatics (panel mesh vs closed form)

| triangles | V (m^3) | dV/V | Cb | dCb/Cb | Sw (m^2) | dSw/Sw |
|---|---|---|---|---|---|---|
| 200 | 0.187644 | -1.240 % | 0.438933 | -1.240 % | 2.268059 | -0.164 % |
| 800 | 0.189407 | -0.312 % | 0.443058 | -0.312 % | 2.270875 | -0.040 % |
| 3200 | 0.189852 | -0.078 % | 0.444097 | -0.078 % | 2.271563 | -0.010 % |

GCI on ∇: observed order p = 1.989, extrapolated value
0.190001203 m³ (closed form 0.190000000), Richardson uncertainty
1.50e-04 m³, **U_G = 0.098 %**, asymptotic ratio
1.0023, converged = True.
GCI on S_w: p = 2.034, **U_G = 0.012 %**, converged = True.
Coarse levels outside the 1 % ∇ band (recorded, not hidden):

* 200 triangles: dV/V = -1.240 %, dCb/Cb = -1.240 %

### 3.2 Ellipsoid wave resistance vs the Havelock closed form

| Fr | U (m/s) | R_W Havelock (N) | R_W panel (N) | error | p | U_G |
|---|---|---|---|---|---|---|
| 0.20 | 0.8857 | 0.236043 | 0.236988 | +0.400 % | 2.15 | 0.431 % |
| 0.30 | 1.3286 | 0.885790 | 0.885389 | -0.045 % | 2.26 | 0.048 % |
| 0.40 | 1.7715 | 1.418160 | 1.422740 | +0.323 % | 2.01 | 0.399 % |

Panel-count detail (relative error of R_W at each level):

| Fr | panels | R_W panel (N) | error |
|---|---|---|---|
| 0.20 | 128 | 0.252305 | +6.890 % |
| 0.20 | 512 | 0.239802 | +1.592 % |
| 0.20 | 2048 | 0.236988 | +0.400 % |
| 0.30 | 128 | 0.877969 | -0.883 % |
| 0.30 | 512 | 0.884109 | -0.190 % |
| 0.30 | 2048 | 0.885389 | -0.045 % |
| 0.40 | 128 | 1.491963 | +5.204 % |
| 0.40 | 512 | 1.436505 | +1.294 % |
| 0.40 | 2048 | 1.422740 | +0.323 % |

Kochin-function convergence (|H| from panels vs Bessel closed form):

| k | |H| analytic | err 8x16 | err 16x32 | err 32x64 |
|---|---|---|---|---|
| 0.5 | 0.029789 | 1.215 % | 0.305 % | 0.076 % |
| 1.0 | 0.053648 | 1.045 % | 0.262 % | 0.065 % |
| 2.0 | 0.072884 | 0.287 % | 0.069 % | 0.017 % |
| 4.0 | 0.013609 | 12.107 % | 3.040 % | 0.761 % |
| 8.0 | 0.008075 | 15.876 % | 3.904 % | 0.972 % |

### 3.3 Tandem interference factor σ = R_W,pair/(2R_W,single)

| d/L | sigma Michell | sigma panel | relative difference | R_pair (N) | R_pair error |
|---|---|---|---|---|---|
| 0.50 | 1.788099 | 1.788116 | +0.0010 % | 3.8501 | -0.028 % |
| 1.00 | 1.520493 | 1.520534 | +0.0027 % | 3.2739 | -0.027 % |
| 2.00 | 0.779779 | 0.779863 | +0.0108 % | 1.6790 | -0.019 % |

Asymptotic behaviour (the envelope decays algebraically, so σ approaches 1 only
slowly — this is physics, not a numerical artefact):

| d/L | sigma | |sigma - 1| |
|---|---|---|
| 8 | 0.965307 | 0.0347 |
| 16 | 0.875065 | 0.1249 |
| 32 | 1.044657 | 0.0447 |
| 64 | 0.857476 | 0.1425 |

σ at zero spacing: 2.000000 (exact limit 2).
Panel-discretisation error of the Kochin function used for the pair (relative to
the closed-form Michell value):

| panels | count | err k=1 | err k=2 | err k=5 |
|---|---|---|---|---|
| 20x10 | 200 | 0.2088 % | 0.2588 % | 0.0405 % |
| 40x20 | 800 | 0.0522 % | 0.0646 % | 0.0103 % |
| 80x40 | 3200 | 0.0130 % | 0.0161 % | 0.0026 % |

## 4. Figures

![bem_wigley_hydrostatics](../benchmarks/verified/bem_wigley_hydrostatics/figures/wigley_hydrostatics.png)  _(bem_wigley_hydrostatics)_
![bem_ellipsoid_wave](../benchmarks/verified/bem_ellipsoid_wave/figures/ellipsoid_wave.png)  _(bem_ellipsoid_wave)_
![bem_multihull_interference](../benchmarks/verified/bem_multihull_interference/figures/multihull_interference.png)  _(bem_multihull_interference)_

## 5. Points to watch (what bit, and what is limited)

1. **The brief's Cb = 0.667 is Cp, not Cb.**  For the parabolic Wigley hull
   Cb = 4/9 = 0.4444 and Cp = Cm = 2/3; feeding 0.667 into ∇ = Cb·L·B·T
   over-predicts displacement by 50 %.  Geometry is therefore gated on the
   *meshed* Cb against the closed-form 4/9, never against the brief's number.
2. **A 200-triangle mesh cannot hold the 1 % volume band.**  With 50 cells the
   draft is resolved by 5 panels (h_z = T/5), and the O(h²) volume deficit is
   about −1.2 %; the band is met from 800 triangles up.  The coarse level is
   kept as the third GCI member and its miss is recorded in
   `coarse_level_misses` rather than silently dropped.
3. **σ is insensitive to the panel discretisation of |H|².**  Because σ is a
   *ratio* of two Kelvin integrals over the same |H|², the discretisation error
   largely cancels (the σ deviation is two orders of magnitude smaller than the
   Kochin-function error).  σ is therefore a weak test of mesh refinement and a
   strong test of the superposition algebra; the pair *resistance* itself is
   reported alongside, and that one does carry the panel error.
4. **No ITTC/towing-tank comparison is made.**  No sourced experimental table is
   available offline in this work tree; `docs/wigley_wave_resistance.md` already
   registers the experimental comparison as `WITHHELD`, and inventing a table
   would be worse than the gap.
5. **The wave-resistance route is thin-ship, not a free-surface Rankine BEM.**
   Constant-strength sources on the centreplane projection is the classical
   Michell discretisation; a Neumann-Kelvin 3-D free-surface Rankine solve
   (unknown source/dipole strengths on the wetted hull *and* on the free
   surface, with the Kelvin condition iterated) is **not implemented in
   TensorBEM**.  Nothing in this document should be read as evidence for that
   solver: this is a verified *thin-ship* panel route plus a verified 3-D
   hydrostatic panel route.
6. **Tandem only, identical hulls only, longitudinal offset only.**  Side-by-side
   (catamaran) interference needs the transverse wave-number phase and is not
   implemented; different hulls are a trivial extension that is untested and
   therefore not offered.

## 6. Error verdict, item by item

Targets from the campaign table; "decidable" means a closed-form or
semi-analytic reference exists in the repository.

### 6.1 Hydrostatics (Wigley III)

| quantity | reference | measured | target | verdict |
|---|---|---|---|---|
| ∇ (Wigley III, 3200 triangles) | 4LBT/9 = 0.190000000 m³ | -0.078 % | < 1 % | PASS |
| ∇ (800 triangles) | idem | -0.312 % | < 1 % | PASS |
| ∇ (200 triangles) | idem | -1.240 % | < 1 % | COARSE-LEVEL MISS (recorded) |
| S_w (3200 triangles) | quadrature 2.271790 m² | -0.010 % | < 2 % | PASS |
| Cb (3200 triangles) | 4/9 = 0.444444444 | -0.078 % | < 0.5 % | PASS |
| U_G(∇) | Roache GCI, three levels | 0.098 % | < 3 % | PASS |


### 6.2 Wave resistance (ellipsoid, three Froude numbers)

| Fr | R_W Havelock (N) | R_W panel (N) | error | U_G | verdict (< 5 %, U_G < 3 %, converged) |
|---|---|---|---|---|---|
| 0.20 | 0.236043 | 0.236988 | +0.400 % | 0.431 % | PASS |
| 0.30 | 0.885790 | 0.885389 | -0.045 % | 0.048 % | PASS |
| 0.40 | 1.418160 | 1.422740 | +0.323 % | 0.399 % | PASS |

### 6.3 Tandem interference factor

| d/L | sigma Michell | sigma panel | abs difference | R_pair error | verdict (pair resistance < 5 %) |
|---|---|---|---|---|---|
| 0.50 | 1.788099 | 1.788116 | 1.75e-05 | -0.028 % | PASS |
| 1.00 | 1.520493 | 1.520534 | 4.10e-05 | -0.027 % | PASS |
| 2.00 | 0.779779 | 0.779863 | 8.38e-05 | -0.019 % | PASS |

Spacing sweep d/L = 0.20 … 4.00 (20 points): worst |sigma_panel - sigma_theory| = 1.39e-04; sign of (sigma - 1) agrees with the Michell trend at 20/20 points.

**Answer to the standing question "is every key quantity within 3 %?"**  No single 3 % band covers all three anchors, so the campaign's own bands are used and answered one by one:

* displaced volume: -0.078 % at 3200 triangles (-0.312 % at 800, -1.240 % at 200) against the 1 % band — **yes** from 800 triangles up, **no** at the coarsest level (recorded as a coarse-level miss, see §5.2);
* wetted area: -0.010 % against the 2 % band — **yes**;
* block coefficient: -0.078 % against the 0.5 % band — **yes**;
* grid-convergence index: U_G(nabla) = 0.098 %, worst U_G(R_W) = 0.431 % against the 3 % band — **yes**;
* ellipsoid wave resistance: worst error +0.400 % against the 5 % band — **yes**;
* interference factor: worst |sigma_panel - sigma_theory| = 8.38e-05 against a 0.02 band — **yes**.
