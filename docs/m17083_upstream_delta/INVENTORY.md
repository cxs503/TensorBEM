# TensorBEM — upstream delta & recent-commit theme inventory (m17083)

## 1. Upstream delta verdict

- worktree branch: `agent/m17083`  HEAD: `135050dbaac8`
- upstream: `origin/main` = `135050dbaac8`
- `origin/main..HEAD` = **0** commits ahead, `HEAD..origin/main` = **0** commits behind
- verdict: **UP-TO-DATE (delta set empty)**
- consequence: there is no upstream delta to review, so this report falls back to the mandated inventory of the last 15 commits (theme evolution + risk).

- de-duplication: the window holds 15 commits but only **11 distinct changes** — 4 commit(s) are the merge that carries a mission commit already in the window (same `git patch-id`). All theme/churn accounting below uses distinct changes only; the raw table keeps both rows so the merge provenance stays visible.

## 2. Theme taxonomy and histogram

| theme | changed file-entries (distinct changes) | meaning |
|---|---:|---|
| `T5-docs-report` | 32 | docs/ reports, figures, JSON evidence |
| `T4-test-physics` | 13 | physics/verification test files |
| `T6-tooling-data` | 11 | scripts/, tools/, benchmarks/, data/ assets |
| `T7-root-scratch` | 9 | repo-root scratch/probe/report helper scripts |
| `T1-hydro-kernel` | 8 | source modules tagged as hydrodynamic/wave kernel or formulation |
| `T2-evidence-infra` | 3 | capability/registry/audit/coverage evidence plumbing (src) |
| `T4-test-evidence` | 1 | tests covering evidence infrastructure |

## 3. Commit-by-commit inventory (first-parent diff)

| # | commit | date | gate | mission | files | +/− | primary theme | dup-of | risk flags |
|---:|---|---|---|---|---:|---|---|---|---|
| 1 | `135050d` | 2026-11-13 08:27 | merge:push-merge | m17021 | 10 | +1484/−2 | T5-docs-report | 403be94 | RG1-provenance-push-merge, RG3-scratch-artifacts(1), RG4-evidence-registry-churn, RG5-binary-figure |
| 2 | `403be94` | 2026-11-13 08:26 | mission-commit | m17021 | 10 | +1484/−2 | T5-docs-report | - | RG3-scratch-artifacts(1), RG4-evidence-registry-churn, RG5-binary-figure |
| 3 | `8d36e4d` | 2026-11-13 08:14 | merge:merge-master | m17022 | 2 | +186/−0 | T7-root-scratch | - | RG7-docs-only, RG5-binary-figure |
| 4 | `38468b3` | 2026-11-13 07:45 | merge:merge-master | m16976 | 10 | +2582/−2 | T5-docs-report | - | RG4-evidence-registry-churn, RG5-binary-figure |
| 5 | `d922e72` | 2026-11-13 07:45 | merge:merge-master | m16955 | 9 | +1057/−0 | T5-docs-report | - | RG3-scratch-artifacts(1), RG5-binary-figure |
| 6 | `a20c695` | 2026-11-13 07:16 | merge:push-merge | m16953 | 8 | +1418/−0 | T5-docs-report | 7bc94b7 | RG1-provenance-push-merge, RG3-scratch-artifacts(2), RG5-binary-figure |
| 7 | `512b244` | 2026-11-13 07:16 | merge:merge-master | m16957 | 1 | +142/−0 | T4-test-physics | - | RG8-test-only-pin |
| 8 | `543b685` | 2026-11-13 07:16 | merge:merge-master | m16956 | 1 | +232/−0 | T4-test-physics | - | RG8-test-only-pin |
| 9 | `7bc94b7` | 2026-11-13 07:15 | mission-commit | m16953 | 8 | +1418/−0 | T5-docs-report | - | RG3-scratch-artifacts(2), RG5-binary-figure |
| 10 | `a0e536b` | 2026-11-13 06:51 | merge:merge-master | m16935 | 9 | +522/−26 | T5-docs-report | - | RG5-binary-figure |
| 11 | `f6c9330` | 2026-11-13 06:26 | merge:push-merge | m16913 | 7 | +1421/−84 | T7-root-scratch | eb116e5 | RG1-provenance-push-merge, RG3-scratch-artifacts(4) |
| 12 | `eb116e5` | 2026-11-13 06:26 | mission-commit | m16913 | 7 | +1421/−84 | T7-root-scratch | - | RG3-scratch-artifacts(4) |
| 13 | `f87b64b` | 2026-11-13 06:25 | merge:merge-master | m16890 | 12 | +1767/−5 | T5-docs-report | - | RG5-binary-figure |
| 14 | `8d22b31` | 2026-11-13 06:09 | merge:push-merge | m16892 | 8 | +2300/−0 | T6-tooling-data | a7cd712 | RG1-provenance-push-merge, RG3-scratch-artifacts(1), RG5-binary-figure |
| 15 | `a7cd712` | 2026-11-13 06:08 | mission-commit | m16892 | 8 | +2300/−0 | T6-tooling-data | - | RG3-scratch-artifacts(1), RG5-binary-figure |

## 4. Change-theme list (evolution of the window)

### T6-tooling-data — 1 commit(s), missions: m16892
- `a7cd712` m16892: matured mission deliverables — 8 file(s), +2300/−0; areas {'benchmarks': 2, 'docs': 2, 'scripts': 2, 'src': 1, 'tests': 1}

### T5-docs-report — 6 commit(s), missions: m16890, m16935, m16953, m16955, m16976, m17021
- `f87b64b` Merge #738 (m16890) via merge-master — 12 file(s), +1767/−5; areas {'docs': 6, 'src': 1, 'tests': 2, 'scripts': 3}
- `a0e536b` Merge #741 (m16935) via merge-master — 9 file(s), +522/−26; areas {'docs': 5, 'src': 1, 'tests': 3}
- `7bc94b7` m16953: matured mission deliverables — 8 file(s), +1418/−0; areas {'docs': 3, 'scripts': 3, 'src': 1, 'tests': 1}
- `d922e72` Merge #745 (m16955) via merge-master — 9 file(s), +1057/−0; areas {'docs': 5, 'root': 1, 'scripts': 1, 'src': 1, 'tests': 1}
- `38468b3` Merge #746 (m16976) via merge-master — 10 file(s), +2582/−2; areas {'root': 1, 'docs': 5, 'src': 3, 'tests': 1}
- `403be94` m17021: matured mission deliverables — 10 file(s), +1484/−2; areas {'root': 2, 'docs': 5, 'src': 2, 'tests': 1}

### T7-root-scratch — 2 commit(s), missions: m16913, m17022
- `eb116e5` m16913: matured mission deliverables — 7 file(s), +1421/−84; areas {'root': 4, 'src': 1, 'tests': 2}
- `8d36e4d` Merge #747 (m17022) via merge-master — 2 file(s), +186/−0; areas {'root': 1, 'docs': 1}

### T4-test-physics — 2 commit(s), missions: m16956, m16957
- `543b685` Merge #742 (m16956) via merge-master — 1 file(s), +232/−0; areas {'tests': 1}
- `512b244` Merge #743 (m16957) via merge-master — 1 file(s), +142/−0; areas {'tests': 1}

## 5. Mission-level aggregation (distinct changes only)

| mission | commits | files | +/− | theme mix | areas |
|---|---|---:|---|---|---|
| m16890 | `f87b64b` | 12 | +1767/−5 | T5-docs-report×6, T6-tooling-data×3, T4-test-physics×2 | {'docs': 6, 'src': 1, 'tests': 2, 'scripts': 3} |
| m16892 | `a7cd712` | 8 | +2300/−0 | T6-tooling-data×4, T5-docs-report×2, T1-hydro-kernel×1 | {'benchmarks': 2, 'docs': 2, 'scripts': 2, 'src': 1, 'tests': 1} |
| m16913 | `eb116e5` | 7 | +1421/−84 | T7-root-scratch×4, T4-test-physics×2, T1-hydro-kernel×1 | {'root': 4, 'src': 1, 'tests': 2} |
| m16935 | `a0e536b` | 9 | +522/−26 | T5-docs-report×5, T4-test-physics×3, T1-hydro-kernel×1 | {'docs': 5, 'src': 1, 'tests': 3} |
| m16953 | `7bc94b7` | 8 | +1418/−0 | T5-docs-report×3, T6-tooling-data×3, T1-hydro-kernel×1 | {'docs': 3, 'scripts': 3, 'src': 1, 'tests': 1} |
| m16955 | `d922e72` | 9 | +1057/−0 | T5-docs-report×5, T7-root-scratch×1, T6-tooling-data×1 | {'docs': 5, 'root': 1, 'scripts': 1, 'src': 1, 'tests': 1} |
| m16956 | `543b685` | 1 | +232/−0 | T4-test-physics×1 | {'tests': 1} |
| m16957 | `512b244` | 1 | +142/−0 | T4-test-physics×1 | {'tests': 1} |
| m16976 | `38468b3` | 10 | +2582/−2 | T5-docs-report×5, T1-hydro-kernel×2, T7-root-scratch×1 | {'root': 1, 'docs': 5, 'src': 3, 'tests': 1} |
| m17021 | `403be94` | 10 | +1484/−2 | T5-docs-report×5, T7-root-scratch×2, T2-evidence-infra×2 | {'root': 2, 'docs': 5, 'src': 2, 'tests': 1} |
| m17022 | `8d36e4d` | 2 | +186/−0 | T7-root-scratch×1, T5-docs-report×1 | {'root': 1, 'docs': 1} |

## 6. Risk annotations

| id | rule | why it is a risk | commits hit |
|---|---|---|---|
| `RG1-provenance-push-merge` | landed through the push-merge gate instead of merge-master | different CI/merge path → gate coverage must be assumed, not verified, per commit | 135050d, a20c695, f6c9330, 8d22b31 (4) |
| `RG1b-merge-unlabelled` | merge commit without (mNNNNN) via ... provenance tag | traceability gap: merge cannot be attributed to a mission from the subject alone | 0 (0) |
| `RG2-src-without-test` | src/ changed but tests/ untouched in the same commit | new/changed numerical behaviour ships without a same-commit regression pin | 0 (0) |
| `RG4-evidence-registry-churn` | capability/evidence registry files rewritten | registry drift: docs registry, capability index and code must stay consistent | 135050d, 403be94, 38468b3 (3) |
| `RG7-docs-only` | docs-only change | claims updated without runtime proof in this commit | 8d36e4d (1) |
| `RG8-test-only-pin` | test-only change | low risk; tightening pins against existing implementation | 512b244, 543b685 (2) |
| `RG5-binary-figure` | new PNG committed | repository size growth / figure-provenance discipline | 135050d, 403be94, 8d36e4d, 38468b3, d922e72, a20c695, 7bc94b7, a0e536b, f87b64b, 8d22b31, a7cd712 (11) |
| `RG3-scratch-artifacts` | probe/deliver scripts committed into scripts/ or repo root | repo hygiene: scratch helpers become unowned surface area | 9 commit(s) |

### 6.1 Churn hotspots (files touched more than once in the window)

| file | commits touching it |
|---|---:|
| `docs/audit_baseline.md` | 2 |
| `src/tensorbem/capability_evidence.py` | 2 |
| `tests/test_wigley3_recalled_ref.py` | 2 |

### 6.2 Cross-mission overlap (same file, ≥2 missions)

| file | missions |
|---|---|
| `docs/audit_baseline.md` | m16976, m17021 |
| `src/tensorbem/capability_evidence.py` | m16976, m17021 |
| `tests/test_wigley3_recalled_ref.py` | m16892, m16935 |

### 6.3 Capability-registry coverage of src modules touched in the window

Registry holds 78 capabilities. Matching is a documented heuristic (module stem substring, else all tokens) and is advisory, not a proof of absence.

| mission | module | match | matched registry key(s) |
|---|---|---|---|
| m16890 | `hemisphere_zero_frequency_envelope` | **none** | — |
| m16892 | `wigley3_recalled_ref` | **none** | — |
| m16913 | `neumann_kelvin` | exact | neumann-kelvin-steady |
| m16935 | `wigley3_hull` | exact | wigley3-hullform |
| m16953 | `shape_gradient` | **none** | — |
| m16955 | `steady_kelvin` | weak (token overlap only) | neumann-kelvin-steady |
| m16976 | `capability_evidence` | exact | capability-evidence |
| m16976 | `wave_flume_wavemaker` | exact | wave-flume-wavemaker |
| m16976 | `wavemaker` | exact | wave-flume-wavemaker |
| m17021 | `capability_evidence` | exact | capability-evidence |
| m17021 | `numeric_evidence` | exact | numeric-evidence |

- no registry entry at all: **3** — `hemisphere_zero_frequency_envelope` (m16890), `wigley3_recalled_ref` (m16892), `shape_gradient` (m16953)
- token-overlap only (entry may be a different formulation; needs a human/agent decision): **1** — `steady_kelvin` (m16955)


## 7. Repo size accounting
```
count: 2469
size: 17.18 MiB
in-pack: 15085
packs: 13
size-pack: 99.91 MiB
prune-packable: 2
garbage: 9
size-garbage: 5.00 MiB
```

## 8. Limitations (honest)

- Theme families are keyed on path/subject tokens per the explicit keyword lists in the extraction script; a commit touching two families is filed under its dominant family.
- Risk flags are structural (diff shape), not semantic: RG2 says *no same-commit test*, not *untested* — the suite may cover the module elsewhere.
- Upstream comparison is against the locally fetched `origin/main` ref; a fetch was not possible in the sandbox, so a newer server-side commit cannot be excluded.

_Generated by `scripts/m17083_upstream_delta_inventory.py`; every number in this file is emitted from git plumbing._
