# CASE-BEM-CUMMINS-WIGLEY Round 7 — Artifact Regeneration & Gate Expansion

**Mission**: m21859 | **Anchor**: `CASE-BEM-CUMMINS-WIGLEY` | **Report**: id=19722

## Summary

Regenerated `case_result.json` from committed sweep artifacts, adding 3 missing gates
(C3a, C7, C3a_coarse) and fixing stale `settings.dt`. All 13/13 gates pass within 10% tolerance.

## Changes

| File | Change |
|------|--------|
| `benchmarks/verified/casepool_bem_cummins_wigley/case_result.json` | Regenerated with 13 gates (was 10), fixed settings.dt 0.01→0.002 |
| `scripts/regenerate_cummins_wigley_verdict.py` | New: replay gates from .npz artifacts |
| `scripts/make_cummins_round7_figures.py` | New: generate report figures |
| `scripts/post_report_m21859.py` | New: publish report via API |

## Gate Results

| Gate | Status | Worst DOF | Value |
|------|--------|-----------|-------|
| C3 (fitted A_inf) | PASS | yaw | 3.60% |
| C3a (anchored A_inf) | PASS | yaw | 8.85% |
| C7 (A_inf consistency) | PASS | yaw | 5.25% (rel to max|A_ii|) |
| C3a_coarse | PASS | yaw | 8.40% |
| C4 (round trip) | PASS | heave | 0.087% |
| C5 (free decay) | PASS | heave | 3.96% |

## Tests

38/38 tests pass in `tests/test_case_bem_cummins_wigley.py`.

## Next Steps

1. Extended band (0.2–20 rad/s) sweep on 60×16 mesh to make surge/sway/yaw KK closure decidable
2. 6DOF free decay gates for roll/pitch modes
3. State-space fit bridge for causal extension of KK integrals
