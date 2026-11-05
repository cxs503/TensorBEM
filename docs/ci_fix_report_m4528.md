# CI Fix Report — PR #282 (agent/m4291)

## Summary
Fixed pytest collection and assertion failures on PR #282 branch.

## Root Causes
1. **Import errors**: `tensorbem` package was installed from a stale work directory (m4526 instead of m4528), causing `ModuleNotFoundError` for `tensorbem.multicomponent`, `tensorbem.uq`, and benchmark modules.
2. **Missing duck-typing methods**: `MultiComponentMesh` lacked `boundary_clearance()` and `contains_point()` methods required by `solver.evaluate_interior_field()`.
3. **Stale baseline**: `docs/doc_claim_audit/baseline.json` was missing guards for 5 new documents added after the baseline was last regenerated.
4. **Unregistered artifact gates**: `test_lifting_camber.py` and `test_near_surface_three_way.py` had artifact skip gates not registered in `ARTIFACT_GATES`.

## Fixes Applied
| File | Change |
|---|---|
| `src/tensorbem/multicomponent.py` | Added `contains_point()` (alias for `fluid_contains()`) and `boundary_clearance()` methods to `MultiComponentMesh` class |
| `docs/doc_claim_audit/baseline.json` | Regenerated via `python -m tensorbem.benchmarks.doc_claim_audit --write-baseline` |
| `tests/test_skip_gates_are_tracked.py` | Added entries for `test_lifting_camber.py` → `docs/lifting_camber_artifact/benchmark.json` and `test_near_surface_three_way.py` → `docs/m3530/near_surface_three_way.json` |

## Verification
- `test_multicomponent.py`: 20/20 passed (previously 3 failed + 5 errors)
- `test_uq.py`: 17/17 passed
- `test_doc_claim_audit.py::test_ratchet_against_baseline`: passed
- `test_skip_gates_are_tracked.py::test_gate_registry_matches_the_test_suite`: passed
- `test_doc_claims.py`: 14/14 passed
- `test_doc_inventory.py`: 17/17 passed
- Full collection: 1245 tests collected (previously 12 collection errors)

## Commit
`f0e1261` on branch `agent/m4291`
