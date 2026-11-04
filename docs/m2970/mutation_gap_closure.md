# Closing the last four mutation survivors (m2970)

**Mission**: m2970 (self-directed, `data` seat) · **Repo**: TensorBEM · **Branch**: `agent/m2970`
**Branch point**: `29236f0` (main, PR #76 merged) · **Date**: 2026-11-03
**Artifacts**: `docs/m2970/outputs/{targeted_mutation.json, figure_data.json, mutation_gap_closure.png}`
**Tools**: `docs/m2970/tools/{targeted_mutation.py, make_figure.py}`

---

## 1. Case: what debt this closes

The mutation audit of mission m2582 (`docs/mutation_audit_tensorbem.md` §5) infected the
numerical core with single-line, syntactically valid defects and asked whether the test
suite notices. Seven mutants survived; three were classified *equivalent* (correctly), and
**four were classified as real or suspected coverage gaps and left as a TODO list**:

| # | survivor | audit verdict |
|---|---|---|
| 1 | `hydrostatic:236` `x_moment`: `... / 6.0` → `... * 6.0` | real gap: every waterplane case in the suite is centred on the origin, so its first moment ≐ 0 and the swap is invisible |
| 2 | `hydrostatic:255` no-waterplane return `0.0` → `1.0` | real gap: no case ever enters the degenerate branch |
| 3 | `hydrostatic:314` zero-displacement return `np.zeros(3)` → `np.zeros(4)` | real gap: same class, branch never entered |
| 4 | `free_surface:243` `df_d_depth`: `2.0 * (...)` → `2.0 / (...)` | "either dead code or a missing derivative guard" |

This mission closes all four, and answers the open question in #4 (it *was* dead code)
without hiding the part a test alone cannot fix.

**Physics context** (why these branches matter, not just for coverage): the hydrostatic
module supplies the only restoring term of the frequency-domain RAO system
\([-\omega^2 (M + A) + \mathrm{i}\omega B + C]\xi = F\). Waterplane area enters
\(C_{33} = \rho g A_{wp}\), the first moments place the waterplane centroid that the
parallel-axis shift and every \(BM = I/\nabla\) use, and a wrong-length centre-of-buoyancy
vector would propagate into `HydrostaticProperties.center_of_buoyancy`. The free-surface
kernel derivative \(dG/ds_z\) is the double-layer kernel of the 2-D radiation solver
(Ursell / Dean benchmarks).

## 2. Method

1. **Reproduce first, then close.** `tools/targeted_mutation.py` re-runs the four *named*
   mutants (not a fresh random sample) as minimal textual edits at their exact AST node,
   in a private copy of `src/tensorbem` — the worktree is never mutated. Every case is run
   twice: against the pre-existing test files (*baseline*) and against those files plus the
   new `tests/test_mutation_gaps_hydrostatic.py`.
2. **Legacy vs current source for #4.** The mutant is also applied to the *pre-fix* source
   (`git show HEAD:src/tensorbem/free_surface.py`) because the source itself is part of the
   fix there:
   * legacy source + new tests → still **survived** (the swap is in an intermediate that
     nothing reads; no test can see it);
   * current source + old tests → **survived** (the existing suite never checks
     \(dG/ds_z\) by finite difference — it only checks \(dG/ds_x\));
   * current source + new test → **killed**.
   Closing #4 therefore needed **both** a one-line composition fix and a new guard.
3. **The source fix** (`src/tensorbem/free_surface.py`, `green_gradient`): the vertical
   source derivative now *composes* the depth derivative instead of re-deriving the same
   bracket inline. With \(A = -(z + z_0)\), \(\partial A/\partial z_0 = -1\), so
   \(dG/dz_0 = -(z - z_0)/(2\pi r^2) - dG/dA\). The two formulations agree to
   2.9×10⁻¹⁰ against a central difference of `green_function` at three
   \((ka, \text{depth})\) working points (§3.3), i.e. the refactor is numerically
   transparent, and it removes the dead intermediate.
4. **Guards are written against independent references**: closed-form rectangle
   integrals, exactly degenerate inputs, and a finite difference of `G` — never against
   the formula being policed.

## 3. Results

### 3.1 Mutant verdicts (programmatic, `outputs/targeted_mutation.json`)

| mutant | source | pre-existing suite | + new guards | killing test |
|---|---|---|---|---|
| `hydrostatic:236:20:swap_binop` | current | **survived** | **killed** | `test_eccentric_waterplane_pins_the_centroid_away_from_the_origin` |
| `hydrostatic:255:15:shift_constant` | current | **survived** | **killed** | `test_waterline_with_cancelling_signed_area_reports_no_waterplane` |
| `hydrostatic:161:15:shift_constant` | current | **survived** | **killed** | `test_submerged_hull_without_waterline_reports_exactly_zero_area` |
| `hydrostatic:314:29:shift_constant` | current | **survived** | **killed** | `test_zero_displacement_branch_returns_a_three_component_centre` |
| `free_surface:243:37:swap_binop` | **legacy** | survived | survived | — (dead intermediate: no test can see it) |
| `free_surface:243:37:swap_binop` | **current** | survived | **killed** | `test_source_depth_derivative_matches_finite_difference_of_green_function` |

Note the extra row `hydrostatic:161`: the empty-waterline return is a *second* degenerate
return of the same function, not on the audit's list. It is closed here as well because the
same missing case covered both.

### 3.2 Eccentric waterplane (the decisive case for #1)

Box \(2\,\mathrm{m}\times1\,\mathrm{m}\times1\,\mathrm{m}\) translated so that the
waterplane is \(x\in[0,2]\), \(y\in[0,1]\); reference values are exact for a faceted box.

| quantity | closed form | computed | relative error | < 3 %? |
|---|---|---|---|---|
| \(A_{wp}\) [m²] | 2.0 | 2.000000000000 | 0 | yes |
| \(x_c\) [m] | 1.0 | 1.000000000000 | 0 | yes |
| \(y_c\) [m] | 0.5 | 0.500000000000 | 0 | yes |
| \(I_{xx}\) [m⁴] | \(LB^3/12 = 1/6\) | 0.166666666667 | 3.3×10⁻¹⁶ | yes |
| \(I_{yy}\) [m⁴] | \(BL^3/12 = 2/3\) | 0.666666666667 | 1.1×10⁻¹⁶ | yes |
| \(I_{xy}\) [m⁴] | 0 | 0.0 | 0 | yes |

The mutant multiplies the moment sum by 6 instead of dividing, i.e. it reports
\(x_c = 36\) m instead of 1 m — a 3500 % error that a centred hull hides completely.

### 3.3 \(dG/ds_z\) against a central difference of \(G\)

| \(\nu = ka\) | best step [m] | min relative error | guard tolerance | margin |
|---|---|---|---|---|
| 0.6 | 1.0×10⁻⁵ | 2.63×10⁻¹¹ | 1×10⁻⁷ | 3.8×10³ |
| 1.2 | 3.2×10⁻⁶ | 4.40×10⁻¹¹ | 1×10⁻⁷ | 2.3×10³ |
| 0.35 | 3.2×10⁻⁶ | 2.88×10⁻¹¹ | 1×10⁻⁷ | 3.5×10³ |

Panel C of the figure shows the classic U-curve (truncation \(O(h^2)\) on the left,
round-off \(O(\epsilon/h)\) on the right), so the 1×10⁻⁷ tolerance sits three orders above
the achievable floor and cannot be tripped by round-off at the chosen step \(10^{-6}\).

### 3.4 Regression evidence

| pytest invocation | result |
|---|---|
| `tests/test_mutation_gaps_hydrostatic.py` | 7 passed (0.34 s) |
| `tests/test_hydrostatic_analytic_benchmark.py test_waterplane_multi_loop.py test_hydrostatic_convergence.py test_mutation_gaps_hydrostatic.py` | 45 passed (1.10 s) |
| `tests/test_free_surface_kernel.py tests/test_ursell_benchmark.py tests/test_mutation_gaps.py` | 38 passed (60.8 s) |

The full 300+ case suite was **not** re-run in this window: the machine's load15 was 118
(≥ 100) and the standing capacity rule restricts such windows to light work. The three
invocations above are the affected modules (hydrostatics, 2-D free-surface kernel and the
Ursell benchmark that consumes it) and they are green.

## 4. Figure

`docs/m2970/outputs/mutation_gap_closure.png` — three panels:
**A** verdict of each named mutant before/after (survived → killed); **B** the eccentric
waterplane with its computed centroid and exact integrals; **C** the finite-difference
error curve of \(dG/ds_z\) with the guard tolerance overlaid. All numbers in the figure
and in this document are produced by `tools/make_figure.py` and
`tools/targeted_mutation.py` — none is transcribed by hand.

## 5. Notes, limits, and honesty

1. **One survivor could not be closed by a test, by construction.** `free_surface:243` in
   the legacy source mutates an intermediate variable that is never read. A test cannot
   observe it; the correct repair is the source change (composition), after which a test
   can. Both halves are delivered and the three-way experiment is recorded, so the claim
   "closed" is not overstated.
2. **Two of the guards use duck-typed meshes.** `hydrostatic:255` and `:314` sit on
   defensive branches that no *validated* `WettedHullMesh` can reach (the constructor
   rejects non-positive volume, and a bowtie waterline is not a physical hull). The tests
   therefore feed a minimal object with only the attributes the function reads, and say so
   in their docstrings. This is coverage of the defensive contract, not evidence about a
   physical hull.
3. **The verdicts are exact for the six named mutants, not a mutation score.** The audit's
   0.650 → 0.825 score was a *sample*; this mission deliberately spends the budget on the
   four known survivors instead of resampling. The population-level score is unchanged and
   not re-estimated here.
4. **Only the affected modules were regression-tested** (§3.4). The refactor touches the
   2-D free-surface kernel used by Ursell/Dean/MacCamy-Fuchs; the Ursell benchmark and the
   kernel suites are green, but 3-D suites were not re-run.
5. **No tolerance was loosened anywhere.** The single source change is an algebraic
   recomposition; the numerically transparent agreement (2.9×10⁻¹⁰) is reported in §3.3.

## 6. Error judgment (key quantity vs baseline, explicitly)

| key quantity | baseline | measured | error | < 3 %? |
|---|---|---|---|---|
| waterplane area (eccentric box) | 2.0 m² | 2.0 m² | 0 | **yes** |
| waterplane centroid x | 1.0 m | 1.0 m | 0 | **yes** |
| \(I_{xx}\) (eccentric box) | 1/6 m⁴ | 0.166667 m⁴ | 3.3×10⁻¹⁶ | **yes** |
| \(I_{yy}\) (eccentric box) | 2/3 m⁴ | 0.666667 m⁴ | 1.1×10⁻¹⁶ | **yes** |
| \(dG/ds_z\) vs FD of \(G\) | central difference (h=10⁻⁶) | — | 2.6–4.4×10⁻¹¹ | **yes** |
| mutant verdicts (6 cases) | audit / pre-fix behaviour | survived → killed ×5 | categorical | n/a (verdict, not a number) |

## 7. Reproduce

```bash
cd /root/tp-agent/work/m2970
PYTHONPATH=src python3 -m pytest tests/test_mutation_gaps_hydrostatic.py -q
PYTHONPATH=src python3 docs/m2970/tools/targeted_mutation.py --out docs/m2970/outputs  # ≈4 min
PYTHONPATH=src python3 docs/m2970/tools/make_figure.py
```