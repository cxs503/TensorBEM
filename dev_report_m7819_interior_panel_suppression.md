# m7819 — TensorBEM interior (lid) panel irregular-frequency suppression

## Goal

Remove the spurious RAO peaks that the frequency-domain free-surface BEM produces at
the *irregular frequencies* of a surface-piercing hull, and prove it against pure
analytical anchors, without losing the panel convergence order.

## What was added

* `src/tensorbem/interior_panels.py` — discretised **interior waterplane lid**: a
  concentric-ring panel set inside the hull (inscribed waterline disc, shrunk to 75 %,
  lowered by >= 1.5 local panel heights), whose centroids are validated by a
  three-direction ray-cast point-in-solid test against the hull closed by its
  waterplane mirror. Points that escape the hull raise instead of silently polluting
  the null-field rows.
* `assemble_free_surface_operators_3d` / `solve_radiation_3d` / `solve_diffraction_3d`
  gained an `interior_panels=` keyword (None | "auto" | `InteriorPanelSet` | (K,3)
  points) that stacks the lid null-field rows on the body rows; `rao.py` passes the
  keyword through to both solves. Default behaviour is unchanged.
* `src/tensorbem/benchmarks/interior_panel_suppression.py` — the study: pre-declared
  frequency grid (`ka` = 0.1..3.0 step 0.1 plus a 0.01-step band over 2.40..2.80),
  three regularisations (`none` = negative control, `chief` = existing 5-point set,
  `lid` = new lid), curvature/conditioning/damping metrics, and four analytical
  anchors (Hulme hemisphere, Wang/Linton sphere, Ursell 2-D cylinder, barge hydrostatics).
* `tests/test_interior_panel_suppression.py` (26 items, incl. the negative control) and
  `tests/test_interior_panel_pins.py` (10 items pinning the committed artifacts).
* `tools/` drivers: sharded sweep, aggregation, anchor runs, report/figures generator.

## Measured results (all read from `runs/interior_panel_suppression/*.json`)

| quantity | none (control) | lid | criterion |
|---|---|---|---|
| max RAO curvature in the band, 64 panels | 339.18 | 0.321 | -99.91 % (>= 80 %) |
| max RAO curvature in the band, 256 panels | 983.77 | 0.283 | -99.97 % (>= 80 %) |
| min heave damping B33* in the band, 64 / 256 | -0.145 / -0.107 | +0.080 / +0.059 | sign restored |
| max cond(A) in the band, 256 panels | 145.7 | 4.2 | -97 % |

Anchors: Ursell `A` 0.851 %, `B` 1.336 % (limit 2 %); submerged sphere A33 2.09 % / B33
5.53 % at 320 panels and 0.524 % / 1.415 % at 1280; five interior rows change the
converged sphere result by < 4e-6 (relative), i.e. the rows are inert for a submerged
body; barge `C33` equals `rho g A_wp` to machine precision and RAO(ka = 0.1) = 1.0083.

Convergence at `ka = 1` against the measured mean edge length: A33 order 2.031 (lid) vs
2.061 (none), B33 2.095 vs 2.078 — the order is preserved. Sphere 320 -> 1280 gives
h-based 1.999 (panel-based 0.999).

## Limits / honest notes

* The lid must be deeper than ~1.5 panel heights: a fixed geometric lid depth degraded
  the least-squares fit at 1024 panels (A33 error grew to 9 %), which is why the depth is
  now resolution-coupled. Verified panel levels are 64 and 256; the 1024-panel full sweep
  and the 320/720/1280 panel ladder were not run (the lid needs a new tunable mesher for
  exact counts, and the cluster nodes could not run this code path: `leggauss`/`laggauss`
  raise `LinAlgError` on loongarch64 + numpy 2.0.1).
* The external cluster was tried twice and failed for that numerical-library reason, so
  all numbers are from the local 128-core host (x86_64), sharded per frequency.
* The central `benchmarks/declared_numbers` registry was **not** extended (adding a
  multi-minute group would slow that CI gate); the equivalent machine gate is
  `docs/interior_panel_suppression/pins.json` + `tests/test_interior_panel_pins.py`.

## Reproduce

```
PYTHONPATH=src python tools/shard_sweep.py --level 256 --config lid --shard 0 --shards 96 --grid full
PYTHONPATH=src python tools/aggregate_sweep.py
PYTHONPATH=src python tools/run_anchors.py --which ursell     # also box / sphere320 / sphere1280
PYTHONPATH=src python tools/make_interior_panel_report.py
PYTHONPATH=src python -m pytest tests/test_interior_panel_suppression.py tests/test_interior_panel_pins.py -q
```
