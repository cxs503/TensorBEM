# m3217 (reopened as m4062): 3-D lifting surfaces by the horseshoe vortex-lattice method

**Repository** TensorBEM · **branch** `agent/m4062` (cut from `main` at `79bd2cd`) · **date** 2026-11-05
**Code** `src/tensorbem/lifting3d.py` (520 lines) · `src/tensorbem/benchmarks/lifting3d_verification.py`
**Tests** `tests/test_lifting3d.py` (15 test functions, 16 parametrised cases)
**Report** `docs/lifting3d.md` (generated) · **Raw data** `docs/m3217/outputs/lifting3d/study.json`
**Figures** `docs/m3217/figures/fig1..fig4_*.png` · **Evidence level** REAL (verified against closed-form theory)

## 1. Why this reopened

PR #132 (branch `agent/m3217`) was closed by the owner on 2026-11-05. The lifting code itself was
sound, but the change could not be merged cleanly: adding `tensorbem.lifting3d` closes a capability
that `src/tensorbem/benchmarks/doc_inventory.py` pinned as an **absence claim**
(`key="lifting_3d"`, `status=ABSENT`, `modules=("tensorbem.lifting3d",)`, phrase
"three-dimensional lifting wakes"), and its README sentence, its `docs/readme_claim_inventory.md`
table, and the generated ETI artifact `docs/m3203/outputs/verif_m3203/readme_claim_inventory.json`
all said the gap was open. The original PR rewrote only two README lines, so the executable gate
would have gone red the moment the module imported. The study data also sat at a non-convention
path (`outputs/lifting3d/`) instead of `docs/m3217/outputs/lifting3d/`.

This delivery keeps the physics and fixes the delivery: the module, the tests, the report, the
README rewrite and the inventory flip move **together**, and every number in the report is
regenerated from the run's own JSON.

## 2. What the module does

Steady, incompressible, inviscid flow past a thin lifting surface, solved as an indirect
boundary-element problem in which the unknowns are vortex strengths, not surface potentials:

* **Biot-Savart primitives** — `segment_velocity` (exact finite straight segment) and
  `semi_infinite_velocity` (closed-form trailing leg), vectorised over field points.
* **Lattice** — `build_lattice` puts one horseshoe per panel: bound segment on the quarter-chord
  line, two semi-infinite legs downstream, control point at the three-quarter chord; cosine
  clustering in span, cosine spacing in chord; elliptic / rectangular / tapered planforms with
  parabolic camber (`LiftingWing`).
* **Circulation** — `solve_circulation` enforces zero normal flow, `A gamma = -U n_x`; this is the
  discrete counterpart of the Kutta condition (the 2-D module fixes `Gamma` by the cusp closure,
  here it is fixed by the wake).
* **Forces** — `panel_forces` (Kutta-Joukowski at the bound vortex, integrated lift) and
  `induced_drag` (Trefftz-plane momentum balance over the far wake).
* **Closed-form references** — `lifting_line_lift_slope` (Prandtl), `elliptic_induced_drag` (Munk),
  `thin_airfoil_lift` (parabolic camber), and `lifting_line_solve`, a Glauert-Fourier solve that is
  an *independent* discretisation of the same lifting-line model for arbitrary planforms.

## 3. Results (recomputed this run, from `docs/m3217/outputs/lifting3d/study.json`)

Elliptic planform, span 8 m, `alpha = 4 deg`, `U = 1`:

| quantity | reference | measured error | < 3%? |
|---|---|---|---|
| lift slope, AR=40, 64 span panels | Prandtl `2*pi*AR/(AR+2)` | 0.327% | yes |
| lift slope, AR=40, 128 span panels (gate probe) | Prandtl `2*pi*AR/(AR+2)` | 0.403% | yes |
| induced drag, AR=8 (128 panels) | Munk `CL^2/(pi*AR)` | 0.572% | yes |
| induced drag, AR=40 (128 panels) | Munk `CL^2/(pi*AR)` | 0.908% | yes |
| cambered lift, AR=20..40, `z_m/c=0.02` | `2*pi*AR/(AR+2)(alpha+2 z_m/c)` | worst 0.896% | yes |
| non-elliptic lift, AR~8 (rectangular, tapered) | Glauert-Fourier lifting line | worst 2.916% | yes |
| non-elliptic induced drag, AR~8 | Glauert-Fourier lifting line | -10.2% (rectangular), -8.8% (tapered) | **no — open limitation** |

The lift-slope error at moderate aspect ratio (AR=8: -4.28%) is the physical lifting-surface
correction (lifting line ignores the chordwise distribution of bound vorticity); it vanishes as AR
grows and is reported rather than tuned away. The non-elliptic induced drag is the one number
outside the 3% band: the far-wake sum is only first-order accurate for non-elliptic loading and is
non-monotone in the panel count, so it is documented as the next gap with its measured sequence
instead of being smoothed.

Two pitfalls are recorded in `docs/lifting3d.md` because both cost a debugging cycle:

1. **Trailing-leg sign** — with the legs convecting upstream the lift diverges instead of converging;
   the convention (`x` downstream, `+y` starboard, `+z` up, `Gamma>0` along `+y` lifts `+z`) is
   pinned by the primitives' tests.
2. **Quarter-chord self-wash** — the two legs of a panel induce a `1/dy` self-wash at their own bound
   vortex (measured `w_z/Gamma = -66` at 64 panels), so quarter-chord Kutta-Joukowski drag is a
   difference of large cancelling terms and changes sign under refinement. Induced drag therefore
   comes from the far wake, which is stationary in the downstream station (identical at 50 and 200
   spans).

## 4. Gate work that makes it mergeable

* `src/tensorbem/benchmarks/doc_inventory.py` — claim `lifting_3d` flipped `ABSENT -> SHIPPED` with
  module, test file, report and an **executable probe** (`lifting3d_lift_slope_ar40`, limit 3%,
  measured 4.031e-03), plus `stale_phrases` so the retired gap sentence cannot come back.
* `README.md` — milestone-6 paragraph and roadmap item 6 rewritten (steady lifting shipped in two
  layers: 2-D Kutta closure and 3-D vortex lattice), and the 3-D module/report added to the code
  index. The retired sentence is now a failing condition.
* `docs/readme_claim_inventory.md` + `docs/m3203/outputs/verif_m3203/` — regenerated by
  `build_report.py` (prose counts and the absence-module list are now computed, not typed):
  17 shipped / 2 absent, **19/19 PASS**, with a dated addendum recording the status transition.
* `docs/m3217/figures/` and `docs/m3217/outputs/lifting3d/study.json` — benchmark outputs moved to
  the repository's mission-artifact convention; reruns are byte-stable (a second run of the
  benchmark reproduced the previous `study.json` to 0 differing values across every entry).

## 5. Verification performed

```
PYTHONPATH=src python -m tensorbem.benchmarks.lifting3d_verification   # writes study.json, 4 figures, docs/lifting3d.md
PYTHONPATH=src python -m tensorbem.benchmarks.doc_inventory --strict   # 19/19 PASS, 0 DRIFT after the flip
PYTHONPATH=src python -m pytest tests/test_lifting3d.py tests/test_doc_inventory.py -q   # 35 passed
PYTHONPATH=src python -m pytest tests -q -n 32                        # full suite (see CI log)
```

`tests/test_lifting3d.py` gates the physics, not the implementation: the finite-segment kernel
reduces to `Gamma/(2*pi*R)`, the closed-form leg matches a 1e6-long segment to 1e-8, the elliptic
lift slope and induced drag hit Prandtl/Munk, the loading shape matches `sqrt(1-(2y/b)^2)` in L2,
span efficiency approaches 1 from above, camber adds the thin-aerofoil increment, non-elliptic
planforms agree with the Fourier lifting line, and the zero-incidence flat wing returns exactly zero.

## 6. Not done / applicability

* No free surface: the model is a deeply submerged foil, keel, rudder or propeller blade section
  without the wavy-wall correction; no forward-speed free-surface term.
* Zero thickness, linearised tangency, fixed straight wake, inviscid — no profile drag, no
  separation, no roll-up.
* Non-elliptic induced drag is not yet within 3% (see above); a desingularised wake quadrature is
  the identified fix.