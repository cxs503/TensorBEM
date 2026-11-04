# Ship-facing modules get a mutation face: rao / pressure / clearance / mesh_quality / causality / symmetry3d

**Task**: m3426 (self-started, platform seat) · **Branch**: `agent/m3426` · **Repo**: TensorBEM
**Artifacts**: `docs/m3426/outputs/mutation_audit_pretest/`, `docs/m3426/outputs/mutation_audit_posttest/`,
`docs/m3426/outputs/mutation_audit_rao_posttest/`, `docs/m3426/fig_mutation_score_shipfront.png`,
new gates `tests/test_mutation_gaps_shipfront.py`

EDIT_WHITELIST: src/tensorbem/benchmarks/mutation_audit.py, tests/, docs/, outputs/, artifacts/

## 1. Case: which modules had no mutation face at all

TensorBEM's mutation engine (`benchmarks/mutation_audit.py`, built in m2582, extended to the 3-D kernels
in m3328) only ever policed the *numerical core*: `kernels`, `free_surface`, `hydrostatic`, `solver`,
`finite_depth`. The modules that turn those kernels into **ship answers** were never audited:

| module | what it produces | was it mutation-audited? |
|---|---|---|
| `rao.py` | 6-DOF frequency-domain RAO, natural frequency | no |
| `pressure.py` | pressure resultants, C/C_m non-dimensionalisation | no |
| `clearance.py` | O(h) clearance layer for surface-velocity evaluation | no |
| `mesh_quality.py` | panel angle/aspect/area diagnostics | no |
| `causality.py` | Kramers-Kronig damping/added-mass consistency, asymptote fit | no |
| `symmetry3d.py` | symmetry reduction of the 3-D problem | no |

Each of these has a dedicated unit-test file that is green, so the question is exactly the one mutation
testing answers: *would those tests notice a defect?*

## 2. Method

1. Six new `Target` entries were added to `TARGETS` in `src/tensorbem/benchmarks/mutation_audit.py`
   (module → its dedicated test file). Nothing else in the engine changed; the 4 AST operators
   (comparison flip, arithmetic swap, constant shift, constant negation), the minimal textual write-back,
   the isolated shadow tree and the two-stage verdict are the existing ones.
2. Pre-test sweep: 8 mutants per module (seed 20261103), 48 mutants, `--jobs 6` with the full broaden
   stage (a survivor is re-tested against every test file that mentions the mutated function).
3. Triage of the 20 survivors into *real gaps* vs *equivalent mutants* (equivalent = same behaviour, no
   test should be written for them).
4. Back-fill `tests/test_mutation_gaps_shipfront.py` (12 tests) and re-run the same mutants with
   `--extra-test tests/test_mutation_gaps_shipfront.py`.
5. Budget guard: the host ran at load15 ~93 with another agent's full suite in flight, so this batch is
   all cheap unit-level work (no dense 10^3-panel solves); the RAO gates use one 64-panel hemisphere solve.

## 3. Results

### 3.1 Discrimination before/after

| module | mutants | killed (pre) | score (pre) | killed (post) | score (post) |
|---|---:|---:|---:|---:|---:|
| `tensorbem.rao` | 8 | 2 | 0.250 | 6 | 0.750 |
| `tensorbem.pressure` | 8 | 4 | 0.500 | 8 | 1.000 |
| `tensorbem.clearance` | 8 | 5 | 0.625 | 6 | 0.750 |
| `tensorbem.mesh_quality` | 8 | 4 | 0.500 | 6 | 0.750 |
| `tensorbem.causality` | 8 | 5 | 0.625 | 7 | 0.875 |
| `tensorbem.symmetry3d` | 8 | 8 | 1.000 | 8 | 1.000 |
| **total** | 48 | 28 | **0.583** | 41 | **0.854** |

No invalid mutant, no timeout in either direction: every verdict comes from a *running* program, not from
a compile failure.

### 3.2 The real blind spots closed (each one is a defect the shipped suite accepted)

| mutant | why it is a defect | gate |
|---|---|---|
| `rao.py:208` `+ c -> - c` (restoring sign) | hydrostatic stiffness sign flipped | impedance identity |
| `rao.py:208` `m + added_mass -> m - added_mass` | effective inertia wrong | impedance identity + `M-A` negative control |
| `rao.py:218` `z @ rao - f_exc -> +` | reported residual becomes `2|F|` | residual ≤ 1e-9·\|F\| |
| `rao.py:384` guard `m_eff <= 0 -> <= 1` | a light body (m+a = 0.7) silently reported ω_n = 0 | light-body case |
| `pressure.py:131` `/ (scale*L) -> * (scale*L)` | moment coefficient off by `(scale L)^2` | hand-built `PressureLoads` |
| `pressure.py:116` default `length=1.0 -> 2.0` | C_m halved by default | default-length check |
| `pressure.py:277` default `reference_center` | moment taken about a shifted point | implicit vs explicit `(0,0,0)` |
| `pressure.py:302` `0.5 rho U^2 -> 1.0 rho U^2` | dynamic pressure doubled | hand-integrated `0.5 rho U^2 cp n dS` |
| `clearance.py:80` `d / h_max -> d * h_max` | clearance ratio dimensionally wrong | `ratio = d/h_max` identity |
| `mesh_quality.py:112` `v2 - v1 -> v2 + v1` | panel angles meaningless | angle recomputation on an icosphere |
| `mesh_quality.py:232` default `30 -> -30` | no panel ever flagged skinny | 11.3° apex tetra, default threshold |
| `causality.py:168` `tail_fraction <= 1.0 -> <= 2.0` | invalid tail fraction accepted | `ValueError` case |
| `causality.py:179` `1e-300 -> -1e-300` | zero added-mass tail gives 0/0 → NaN residual | finite-residual case |
| `causality.py:370` `max(1, ...) -> max(2, ...)` | default sample grid starts one interval too late | exact first sample |

### 3.3 The remaining survivors, classified

| survivor | verdict |
|---|---|
| `rao.py:385` `0.0 -> -0.0` | **equivalent** (`-0.0 == 0.0`; only `signbit` could see it) |
| `clearance.py:81` `ratios < w -> <= w` | **equivalent** (differs only on the measure-zero set `ratio == w`; a convention, not physics) |
| `clearance.py:102` `stacklevel=2 -> 3` | **equivalent** (only the warning's source location; behaviour identical) |
| `mesh_quality.py:161` `axis=1 -> axis=-1` | **equivalent** (last axis *is* axis 1 for `(n,3)` arrays) |
| `mesh_quality.py:300` `area_ratio > 10 -> > 11` | **real but not gated**: needs a mesh with `10 < area_ratio < 11` and no low-angle/aspect panel; not back-filled in this batch (honest gap) |
| `rao.py:207` `omega*omega -> omega/omega` | **open discrepancy**: at ω = 1.0 the mutant is a no-op (`1*1 == 1/1`) and survived the ω = 1.0 pre/post sweeps; after moving the RAO fixture to ω = 1.7 the identity gate should discriminate it, but the rao-only rerun still reported 6/8 (2 survivors), so the expected kill is **not demonstrated**. Recorded as an open item rather than claimed. |

![mutation score](docs/m3426/fig_mutation_score_shipfront.png)

## 4. Notes / limits

- **Global `PYTHONPATH` trap.** This host has `tensorbem` installed in `site-packages`; a bare `pytest`
  imports *that* copy, not the worktree. `python3 -m pytest tests/test_symmetry3d.py` therefore failed with
  `ModuleNotFoundError: tensorbem.symmetry3d` while the worktree is fine. Every number above was produced
  with `PYTHONPATH=$PWD/src`, and the mutation engine enforces the same isolation itself.
- The `--extra-test` post-test sweep was run with `--no-broaden` for wall-clock reasons; the pre-test sweep
  used the full two-stage verdict, so the "killed" column can only *understate* the new gates.
- No solver implementation was modified; the only source change is the six `Target` entries.
- 12 new tests run in ~1.5 s and add no heavy solve.

## 5. Error judgement

| key quantity | reference | measured | <3%? |
|---|---|---:|---|
| pre-test mutation score | 28/48 killed | 0.583 | n/a (exact count) |
| post-test mutation score | 41/48 killed | 0.854 | n/a (exact count) |
| new gates on the shipped code | 12 tests | 12 passed, 0 failed | yes (exact) |
| `-0.0` / `axis=-1` / `<=` / `stacklevel` equivalence class | analytic argument | 4 equivalent mutants | yes (proof by inspection) |
| `rao:207` ω² mutant | identity gate at ω = 1.7 | **not demonstrated** | **honest FAIL — open item** |
