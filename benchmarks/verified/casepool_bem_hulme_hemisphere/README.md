# `CASE-BEM-HULME-HEMISPHERE` — floating hemisphere radiation/diffraction

Case-pool anchor `BEM-HULME-HEMISPHERE` (`tensor_case_pool.id = 138`, repo
TensorBEM, tolerance `< 5 %`): a hemisphere of radius 1 m whose flat face is
flush with the mean free surface, in deep water, against the Hulme (1982)
wave-free multipole reference.

| path | what it is |
|---|---|
| `../../src/tensorbem/benchmarks/hulme_hemisphere_case.py` | the case itself: physics driver, reference, gates, independent judge, renderers |
| `../../src/tensorbem/benchmarks/case_pool.py` | anchor registry (`ANCHORS`, `run_anchor`) — the programmatic entry point |
| `run_case.py` | thin driver kept at the historical path; forwards to `case_pool` |
| `make_doc.py` | renders `docs/bem_hulme_hemisphere_case.md` from `evidence/results.json` |
| `evidence/` | the committed run: `results.json`, `benchmark.json`, `independent.json`, `reference.json`, `report.md`, `figure.png`, `repro.sh`, per-shard logs, `artifacts_sha256.json` |

Reproduce (single process, ~4 min: 6 × 1024-panel cells + the capytaine judge):

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.hulme_hemisphere_case \
    --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence --shard 0 --shards 1
PYTHONPATH=src python3 -m tensorbem.benchmarks.hulme_hemisphere_case \
    --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence --merge
```

Or through the registry:

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_pool \
    --anchor CASE-BEM-HULME-HEMISPHERE \
    --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence --merge
```

The old `results.json` / `report.md` that lived directly in this directory
(mission 17181, 256-panel case mesh) were removed in the mission-17337 rework:
they were superseded by `evidence/`, and they recorded a mesh that fails the
2 % displaced-volume precondition. `evidence/report.md` is the current table
report.
