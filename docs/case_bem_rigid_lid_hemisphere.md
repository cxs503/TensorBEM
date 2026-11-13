# CASE-BEM-RIGID-LID-HEMISPHERE — case card (round 1, mission m17464)

Rigid-lid (`ka -> 0`) heave added mass of a floating hemisphere, solved by the 3-D
Kelvin Green-function radiation solver of TensorBEM
(`tensorbem.free_surface_solver3d.solve_radiation_3d`) and judged against the
**even-parity double-body** (rigid-lid mirror) analytic endpoint.

## Case card

| field | value |
|---|---|
| `case_code` | `BEM-RIGID-LID-HEMISPHERE` |
| pool row `ref_source` | rigid-lid (double-body) analytic endpoint, `ka -> 0` |
| pool row `tol_policy` | `< 5 %` (declared by the pool, not by this case) |
| repo | TensorBEM |
| physics | floating hemisphere, waterplane on `z = 0`, heave; inviscid, linearised potential flow, zero forward speed |
| solver under test | `solve_radiation_3d` (deep-water Kelvin Green function; `ka -> 0` degenerates to the mirror Rankine kernel) |
| reference (referee) | `tensorbem.rigid_lid`: even-parity double-body Legendre series summed in **exact rational arithmetic** — no quadrature error at the `|mu|` kink, no truncation error |
| reference (corroboration) | driver routes `energy` (kinetic-energy identity) and `force` (pressure integral); Hulme (1982) multipole series extrapolated to `ka = 0` |
| negative control | the translating-image reading `A33(ka->0) = rho*nabla` (measured, refuted) |
| implementation | `src/tensorbem/rigid_lid.py`, `src/tensorbem/benchmarks/case_bem_rigid_lid_hemisphere.py` |
| entry | `python -m tensorbem.benchmarks.case_bem_rigid_lid_hemisphere --check` (case), `python -m tensorbem.rigid_lid` (referee) |
| integration caller | `tensorbem.free_surface_solver3d.verify_rigid_lid_heave_limit` / `rigid_lid_heave_added_mass_dimless` (main 3-D solver module calls the referee) |
| committed artifacts | `benchmarks/verified/case_bem_rigid_lid_hemisphere/` (`results.json`, `report.md`, `case.json`, `figures/`) |
| run artifacts (this mission) | `runs/m17464_rigid_lid_hemisphere/` (`run.log`, `results.json`, `benchmark_ref_vs_got.json`, `meta.json`, `report.md`, `case.json`, `figures/`) |
| tests | `tests/test_case_bem_rigid_lid_hemisphere.py`, `tests/test_rigid_lid_exact.py` |

## Reproduce (one command)

```
bash scripts/repro_case_bem_rigid_lid_hemisphere.sh
```

which is, in order:

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_bem_rigid_lid_hemisphere \
    --outdir benchmarks/verified/case_bem_rigid_lid_hemisphere --check
PYTHONPATH=src python -m pytest tests/test_case_bem_rigid_lid_hemisphere.py -q
PYTHONPATH=src python -m pytest tests/test_rigid_lid_exact.py -q
PYTHONPATH=src python -m tensorbem.rigid_lid     # referee value, JSON
```

Environment of the committed run: python 3.11.6, numpy 2.4.6, loongarch64, git `9ebee0f77bb8`.
Every number of the case lives in `results.json` and is rendered into `report.md`
and the figures by the driver itself.

## Benchmark: reference vs computed (snapshot)

Snapshot of `runs/m17464_rigid_lid_hemisphere/benchmark_ref_vs_got.json`
(sha256 of every artifact in `meta.json`); `ref = 0.8309513` dimensionless, 
`= 1783.85 kg` (`rho*nabla`); tolerance `< 5 %`.

| panels | ka | ref `A33/(rho*nabla)` | got | err | <3 %? | <5 %? | err vs Hulme | err vs naive image |
|---|---|---|---|---|---|---|---|---|
| 64 | 1e-03 | 0.8309513 | 0.8073879 | -2.836 % | yes | yes | -3.314 % | -19.26 % |
| 64 | 1e-02 | 0.8309513 | 0.8253131 | -0.679 % | yes | yes | -3.319 % | -17.47 % |
| 64 | 3e-02 | 0.8309513 | 0.8429342 | +1.442 % | yes | yes | -3.277 % | -15.71 % |
| 64 | 1e-01 | 0.8309513 | 0.8369115 | +0.717 % | yes | yes | -3.000 % | -16.31 % |
| 256 | 1e-03 | 0.8309513 | 0.8281517 | -0.337 % | yes | yes | -0.828 % | -17.18 % |
| 256 | 1e-02 | 0.8309513 | 0.8465655 | +1.879 % | yes | yes | -0.829 % | -15.34 % |
| 256 | 3e-02 | 0.8309513 | 0.8643584 | +4.020 % | NO | yes | -0.818 % | -13.56 % |
| 256 | 1e-01 | 0.8309513 | 0.8563606 | +3.058 % | NO | yes | -0.745 % | -14.36 % |

Headline row (256 panels, `ka = 1e-3`): **-0.337 %** vs the rigid-lid endpoint
— **within 3 % and within the pool's 5 %**. Worst band row: +4.020 % (`ka = 3e-2`,
256 panels), inside the pool's 5 % but **outside 3 %** — reported as such, not
smoothed away. The out-of-band sweep (`ka = 0.2 ... 1.0`) must and does violate the
band, by -5.1 %, -14.3 %, -35.5 % and -48.8 % at `ka = 0.2, 0.3, 0.6, 1.0`; that is
the case's own scope control (`G6`) — the endpoint is an endpoint, not a curve.

Verdict: **PASS**, 14/14 predeclared checks passed.

## Why the reference is not the "imaged sphere"

The pool row's one-line gloss "rigid lid = mirror = imaged sphere" invites the
translating-image reading `A33(ka->0) = 2/3 rho pi a^3 = rho*nabla`. Reflection in
the symmetry plane reverses the *normal* velocity of a body moving normal to it, so
the double body carries the sliding Neumann data `dphi/dr = -U|cos theta|`, whose
Legendre content is even and whose dimensionless added mass is
`0.8309513`, not `1.0 rho*nabla`. The table above refutes the translating image on every
mesh and `ka` tested (`err vs naive image` between -17 % and -19 %). Using it as the
reference files a spurious honest-FAIL of order ten percent; that history (report
#15809) and its correction are kept in
`docs/rigid_lid_zero_frequency_anchor/ERRATUM.md`.

## Scope declared by the case

* The endpoint reference applies to the band `ka <= 0.1` only; the driver runs an
  out-of-band sweep (`ka = 0.2 ... 1.0`) and *requires* it to violate the 5 % band,
  so the case cannot silently claim more than it verifies.
* The mesh evidence is 64 / 256 constant panels. The 1024-panel level exceeds this
  container's per-command budget and is not part of the predeclared ladder.
* Geometry is gated before any error is quoted (mesh area / volume / waterplane
  against the closed forms, plus a refinement-shrink gate). Raw geometry errors of
  both meshes are published in `report.md` section 2; an inscribed icosphere
  under-represents the hemisphere, which is why the coefficient is normalised by the
  *exact* `nabla` and never by the mesh volume.
* No experimental data enters this case: the absolute coefficient is verified
  against analysis (double-body series + Hulme multipole), not against EFD.
