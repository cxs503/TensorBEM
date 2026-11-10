# Near-singular layer integrals of quadratic panels

Companion artifacts for `tensorbem.panel_quadratic`'s **near-singular** path:
the collocation point sits a distance `d` off an element whose size is `H`,
with `d/H` as small as `1e-4`.  Such pairs are produced by every real BEM
mesh in the near field (sharp corners, thin gaps, adjacent wetted surfaces of
multibody configurations) and they are *not* covered by the singular
self-integral rule, which assumes `d = 0` at a node.

## What is verified

| quantity | reference | tolerance | result |
|---|---|---|---|
| semi-analytic reference vs plain Gauss, `d/H=0.5` | two independent methods | `< 1e-10` | `1.0e-15` |
| graded rule, `d/H = 1e-3`, order 12 (library default) | semi-analytic | `1e-3` | `2.7e-4` (double layer, worst) |
| graded rule, `d/H <= 1e-3`, order 24 | semi-analytic | `1e-6` | `1.1e-7` |
| graded rule, `d/H <= 1e-3`, order 32 | semi-analytic | `1e-9` | `5.6e-10` |
| plain Gauss order 8, `d/H = 1e-3`, interior footprint | semi-analytic | *baseline* | `1.2e-1` (**12 % error**) |
| curved icosphere element, order 32 | graded order 40 | `1e-9` | `< 1e-12` |

The plain Gauss rule is the previous state: it needs `O(H/d)` points to
resolve the `1/r` peak and still stalls at ~1 % even at order 64; the graded
Duffy rule resolves it with `O(log(H/d))` radial intervals.

## Method

`graded_apex_quadrature(apex, layer_width, order, growth)` keeps the Duffy map
of `triangle_apex_quadrature` (which cancels a *true* `1/r` singularity) but
splits the radial direction `[0, 1]` into geometrically growing intervals
`[0, w], [w, Gw], [wG, wG^2], ...` with `w = d / extent`, so the boundary
layer of width `u ~ d/H` created by the near-singular peak is resolved by the
same order-`order` Gauss rule on every interval.  One width per
`apex_subtriangles` entry is allowed, so each sub-triangle's breakpoint lands
on its own peak edge.

`QuadraticTriangle.near_singular_layer_integrals(target)` centres the rule on
`project_to_reference(target)` (Gauss-Newton on the isoparametric projection,
clamped to the reference triangle) and returns the single- and double-layer
nodal integrals; `integrate_single_layer_near_singular` and
`integrate_double_layer_near_singular` are the convenience wrappers.

## Reference

For a **planar** element the layer integrals reduce to a one-dimensional
integral: polar coordinates about the foot of the perpendicular give
`r = sqrt(rho**2 + d**2)`, the shape functions are quadratic along a ray, and
the radial integral has a closed form (`asinh`/`sqrt` terms).  The remaining
`theta` integral runs over sectors bounded by the rays to the vertices, inside
which the exit distance is analytic, so a 64-point Gauss rule per sector makes
the reference accurate to `1e-15` — independent of the rule under test.

## Reproduce

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_near_singular
PYTHONPATH=src python -m pytest tests/test_panel_quadratic_near_singular.py
```

`study.json` carries every number in this directory (programmatically
injected) plus the `sha256` of `fig_near_singular.png`; the module exits
non-zero when an acceptance gate fails.

## Applicability / limits

* The reference covers **planar** elements; for curved elements the study
  falls back to order self-convergence (no closed form) — verified flat to
  `1e-12` between orders 32 and 40.
* Accuracy is set by the Gauss order: `1e-4` at order 12 (default, cheap),
  `1e-9` at order 32.  Cost grows only logarithmically with `1/H d`.
* `d = 0` (target *on* the element) is outside this path — use the singular
  self-integral rule (`integrate_single_layer_at_node`).
* Convergence in the Gauss order is algebraic (~ `order**-8`) rather than
  exponential, because the peak position in the radial variable varies with
  the angular variable across a sub-triangle.