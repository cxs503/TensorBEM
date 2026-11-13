# m18249 — CASE-BEM-CUMMINS-WIGLEY: the two failing gates were numerical, not physical

**Anchor**: feature ledger `CASE-BEM-CUMMINS-WIGLEY` (算例池·Wigley 船 Cummins 记忆核（6DOF）),
ref = Cummins equation + Kramers-Kronig causality, tolerance < 10 %.
**Predecessor**: m17375 shipped `benchmarks/verified/casepool_bem_cummins_wigley/` with
**5/7 gates passed** — C3 (KK closure) and C5 (free decay vs frequency-domain damping) failed.

## 1. What was actually wrong

Both failures came from the *measurement apparatus*, not from the BEM data:

1. **C3 (KK closure)** closed the Kramers-Kronig integral with the **piecewise-linear
   interpolant** of the tabulated damping, but the sweep had `13` nodes over
   `0.15 … 8 rad/s` (≈ 7.5 nodes/decade) while the damping curves peak between
   2 and 12 rad/s. The residual (3.5–16 %) was interpolation error, not a causality
   violation. The same run also used **fewer nodes than the module's own pre-declared
   default** (`N_OMEGA = 21`).
2. **C5 (free decay)**: the model's heave natural frequency is ≈ 7.8 rad/s, i.e. it sat
   **outside** the sweep (`0.15 … 8` was only nominally enough — the recorded run
   computed 8.147 rad/s from a contaminated period). Second, the memory kernel is
   **truncated at ω_max**, which leaves a small ripple at ~ω_max in the decay record;
   the peak detector fitted the logarithmic decrement across that residue (69 peaks over
   40 s, spacing 0.57 s) and reported `ζ = 0.0268`, i.e. the *residue* decay, while the
   hull mode gives `ζ ≈ 0.133`.

## 2. Changes (all in `src/tensorbem/benchmarks/case_bem_cummins_wigley.py`)

| change | why |
| --- | --- |
| `BAND = (0.2, 12.0)` (was `(0.2, 6.0)`) | the band must contain every rigid-body natural frequency of the 3 m model (heave ≈ 7.8 rad/s); otherwise C5 is **undecidable**, not failed |
| `N_OMEGA = 41` (was `21`) | resolve the damping peaks: the KK residual is measured, not asserted |
| `sweep_grid()` | log grid with both band ends exactly included (unit-tested) |
| `_refined_grid()` / `--refine-peaks K` | optional geometric refinement of the intervals with the largest damping jump (default off) |
| `grid_convergence_study()` / `--grid-study` | KK residual **as a function of node count** — turns the tolerance into a measurement |
| `free_decay_gate()` status field | separates "undecidable: natural frequency outside the band" from "the two routes disagree"; the recorded FAIL was the former and was silently reported as `nan` |
| `DECAY_AMPLITUDE_FLOOR = 1e-2` in `_decay_zeta()` | fit the decrement only over the physical part of the record (the ripple outlives the mode) |

The negative control C6b (`×1.3` shape corruption) and the zero-damping control C6a are
unchanged and still pass, so the gates did not get easier.

## 3. Results — coarse mesh, 192 panels, band 0.2–12 rad/s, 41 nodes

| DOF | KK residual m17375 [%] | KK residual m18249 [%] | A_inf (closure) [kg] | worst ω [rad/s] |
| --- | --- | --- | --- | --- |
| surge | 10.08 | 2.58 | 0.5814 | 9.291 |
| sway | 16.11 | 2.70 | 77.31 | 9.291 |
| heave | 3.53 | 0.28 | 50.38 | 9.291 |
| roll | 9.66 | 0.75 | 0.4062 | 9.291 |
| pitch | 6.16 | 0.50 | 14.64 | 9.291 |
| yaw | 14.52 | 3.40 | 58.21 | 9.291 |

* worst DOF: **yaw 3.40 %** vs the 10 % criterion;
  the δDof-independent scale is `max|A_ii|` per DOF, so each row is a relative error on the
  coefficient itself.
* C5: `ζ_decay = 0.1335`, `ζ_fd = 0.1296`,
  relative error **2.93 %** (criterion 5 %),
  ω_n = 7.827 rad/s, 4 peaks used,
  A(ω_n) = 28.55 kg, B(ω_n) = 214.0 N·s/m.
* Gate tally on this single-mesh run: 6/7
  (failed: ['C2']). **C2 fails by design on the coarse mesh**
  (`k·√A_panel = 1.47` at 12 rad/s); C2 is a property of the *production* mesh and is
  evaluated on the 20×6 mesh, which is the remaining open item (cluster job `cj1863`).

## 4. Honest limits

* The coarse-mesh run above is complete and reproducible; the **medium-mesh production
  run was submitted to the cluster (`cj1863`) and had not returned** when this report was
  written, so the `benchmarks/verified/casepool_bem_cummins_wigley/` artifact still holds
  the m17375 result. The new artifacts are under `docs/m18249_case_bem_cummins_wigley/`.
* Only the **diagonal** (heave) free-decay comparison is made; the 6-DOF coupled decay
  (off-diagonal kernels, roll-yaw coupling) is **not** covered here — 6DOF *kernels* are
  covered by C3 for all six diagonal DOFs and by the matrix closure artifacts.
* `k·√A_panel ≤ 1` at 12 rad/s on the medium mesh is expected to be ≈ 0.93, i.e. the top
  of the band is *just* resolved; if the production run fails C2 the honest fallback is
  `BAND[1] = 11 rad/s` (still above ω_n = 7.83 rad/s), not a relaxed tolerance.

## 5. Reproduction

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_bem_cummins_wigley \
    --out docs/m18249_case_bem_cummins_wigley --meshes coarse --jobs 4 --no-figure
PYTHONPATH=src python -m pytest tests/test_case_bem_cummins_wigley.py -q   # 19 passed
```

Figure: `docs/m18249/fig_m18249_cummins_wigley.png`.
