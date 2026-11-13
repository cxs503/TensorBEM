# m17621 — exact surface measure for the flat-panel Kelvin BEM (round 2, CASE-BEM-SUBMERGED-SPHERE-HAVELOCK)

## 1  Problem setting and benchmark
Submerged sphere (a = 1 m, f/a = 2, rho = 1025, g = 9.80665), radiation coefficients
A33', B33', A11', B11' vs ka; reference = Havelock (1955) frequency-domain multipole solution
(Proc. R. Soc. Lond. A 231, 1-13; submerged form Thorne 1953 / Wang 1986 / Linton & McIver,
Ocean Eng. 18 (1991) 61-74), evaluated by `tensorbem.analytic.submerged_sphere_wave_multipole`
(20 terms), certified by truncation independence, Haskind reciprocity and Lamb's limit.
Pool gate: every item < 5 %.

## 2  Method
The committed artefact showed the 320-panel economy mesh (mission m17242's production mesh)
failing the gate on damping (5.59 % band-max) while the 1280-panel mesh passes (1.43 %). The
committed `area_deficit_diagnostic` shows the error scaling with the polyhedral area deficit of
the inscribed icosphere (fig 1). Fix: keep the flat panels, their normals, the collocation
points and every quadrature rule unchanged, and replace only the *integration measure* over the
body surface by the exact spherical-patch area of each panel
(`tensorbem.geometry3d.spherical_patch_areas`, Van Oosterom-Strackee spherical-triangle area),
threaded as the new `panel_areas` keyword through `assemble_free_surface_operators_3d`,
`solve_radiation_3d` and `solve_diffraction_3d`. Control: the same rescaling applied uniformly
(k = A_exact/A_mesh), to test whether the per-panel distribution matters.

## 3  Results (band-max relative error, 4 coefficients x band)
| tier | mesh | flat measure | exact surface measure | uniform-measure control |
|---|---|---|---|---|
| T1 economy, ka = 0.1..5.0 | 320 panels | 5.59 % (7/28 over gate) -> FAIL | 2.25 % (0/28) -> PASS | 1.91 % (0/28) -> PASS |
| T2 coarsest, ka = 0.1..3.0 | 80 panels | 20.24 % (20/20) -> FAIL | 5.59 % (9/20) -> FAIL | 5.59 % |

Worst items: T1 flat at ka = 0.5 (B33'),
T1 exact at ka = 5 (B11).

## 4  Figures
![deficit](measure_correction/fig1_deficit.png)
![tiers](measure_correction/fig2_tiers.png)

## 5  Caveats / limits
* The uniform-measure control reaches the same accuracy as the exact patch measure
  (1.91 % vs 2.25 %): the leading bias of an
  inscribed mesh is a **global** measure deficit, so a single scalar rescaling by the true wetted
  area captures it. The exact patch measure is the better recipe (no free parameter, correct for
  graded meshes) but it is not what makes the difference at this order.
* The O(h^2) facet-normal error is untouched: at 80 panels the corrected tier still misses 5 %
  (B33' improves ~3.6x, A33' barely moves -- see the unit test).
* `spherical_patch_areas` needs an analytic surface (sphere here); for a numerically defined hull
  the uniform rescaling by the measured wetted area plays the same role.
* NOT RUN: the 1280-panel case mesh with the new measure (wall clock), and the per-ka rows of the
  study (the run was interrupted before `write_artifacts`); the aggregate tier verdicts above are
  the measured ones, parsed from the run log.

## 6  Verdict
With the exact surface measure the 320-panel production mesh meets the 5 % pool gate
(0 of 28 items over gate, band-max
2.25 %), versus 7 of 28
and 5.59 % on the untouched flat measure (honest FAIL kept on record).
