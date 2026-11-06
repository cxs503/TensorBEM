# m4721 — new independent analytic benchmark: mutual added mass of two circular cylinders

## Goal / why this repository

TensorBEM's verification portfolio already pinned the *single-body* classical
references (sphere, ellipsoid, circular cylinder, Ursell, Dean, MacCamy-Fuchs,
Hulme hemisphere, submerged sphere, hydrostatics, lifting camber). None of them
carries a **mutual (body-to-body) coefficient**, and none assembles the exterior
Neumann operator over **two disjoint boundary loops**. Both gaps are covered
here by one new case: the `4x4` added-mass matrix of two equal circular
cylinders in an unbounded fluid, per unit span.

## What was added

| file | content |
|---|---|
| `src/tensorbem/analytic.py` | `two_cylinder_added_mass`, `two_cylinder_mode_potential`, `TwoCylinderAddedMassResult` — truncated multipole series, no BEM code path |
| `src/tensorbem/benchmarks/two_cylinder_added_mass.py` | multi-loop exterior Neumann solve (`0.5 I + H`), `run`, `refinement_study`, `interaction_decay`, `plot_refinement`, `--inject layer_sign` control |
| `tests/test_two_cylinder_added_mass.py` | 18 tests: reference self-consistency, exact geometric symmetries, far-field `rho*pi*a**2` limit, `(a/d)**2` interaction law, BEM-versus-reference on a 48/64/96 ladder, field-level boundary-potential comparison, injected-error controls |
| `docs/two_cylinder_added_mass.md` | formulation, numbers, reproduction commands, limitations |
| `docs/m4721/figures.py` + `docs/m4721/*.png` | report figures + JSON digest, regenerated programmatically |

No existing test was deleted, skipped or loosened.

## Reference (independent of the solver)

`Psi(z) = sum_k sum_n m_{k,n} (z - c_k)^{-n}`, `phi = Re[Psi]`, coefficients
from a real least-squares fit of the closed-form radial derivative on `2N+8`
collocation points per circle; `A_ij = -rho * oint(phi_j n_i ds)` by trapezoid.

Self-checks (asserted): body-condition residual `3.7e-11` at `N = 24`;
truncation drift `< 1e-8` between `N = 16/24/32`; exact reflection symmetries
`< 1e-9` of `rho*pi*a**2`; `self/rho*pi*a**2 -> 1 + 7.8e-7` at `d/a = 40`;
`A12_yy d^2 / (2 rho pi a^2) = 1.0000004` at `d/a = 40`.

## Numerics and accuracy

`d/a = 3`, `rho = 1025`:

| panels/body | max rel. error | self | mutual (line) | mutual (transverse) | cond |
|---|---|---|---|---|---|
| 48 | 2.20e-3 | 3322.54 | -728.90 | 728.90 | 2.55 |
| 64 | 1.24e-3 | 3322.72 | -729.60 | 729.60 | 2.54 |
| 96 | 5.51e-4 | 3322.84 | -730.10 | 730.10 | 2.54 |
| series | — | 3322.94 | -730.51 | 730.51 | — |

Boundary-potential field error at 64 panels/body: `8.3e-4` (body 0) / `8.4e-4`
(body 1), relative `L2`.

## Negative controls (the gate can fail)

* permanent, in-tree: `layer_sign = -1` (wrong exterior sign) →
  mutual relative errors `9.94e-1` / `9.99e-1`, self `3.6e-2`;
  `test_injected_double_layer_sign_error_turns_the_checks_red` asserts red.
* manual, for this report: dropping the minus sign in `A_ij` inside
  `two_cylinder_added_mass` turned `5 failed, 9 passed` (far-field, blockage,
  mode-potential-vs-matrix, BEM-vs-reference, convergence). Reverted at once.

## Commands run

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src python3 -m pytest tests/test_two_cylinder_added_mass.py -q   # 18 passed
ruff check src/tensorbem/analytic.py src/tensorbem/benchmarks/two_cylinder_added_mass.py tests/test_two_cylinder_added_mass.py  # All checks passed
ruff check .   # 182 pre-existing findings, unchanged from HEAD (all outside this diff)
```

Note: the environment has an editable install of `tensorbem` pointing at another
worktree (`/root/tp-agent/work/m4614/src`), so `PYTHONPATH=src` is required to
exercise *this* tree; without it pytest imports the other checkout.

## Limitations

2-D, unbounded fluid, no free surface, non-touching bodies only (`d/a >= 3`
registered; the series fit degrades below that). Verification of the exterior
operator and interaction algebra, not of wave radiation or ship hydrodynamics.
