# m22590 · GAP-ICE-GATE-POWER / S07′: 2-D heave added-mass anchor with information content

**Repo** TensorBEM · **Branch** agent/m22590 · **Anchor** `GAP-ICE-GATE-POWER`
(ledger `in_progress`; S02 closed, S07′ hydrodynamic leg under dispute)

## 1. What was done

* New module `src/tensorbem/benchmarks/case_ice_gate_power_section.py` computes both spectral
  limits of the 2-D heave added mass of a surface-piercing rectangular section by one
  infinite-fluid 2-D boundary-integral solve on the mirror-doubled body, halving the load
  integral over the physical panels only (the 2-D analogue of
  `tensorbem.added_mass_limit.added_mass_limit`, whose parity convention it reproduces).
* `mirror_diagnostics` proves numerically **which** free-surface condition each branch
  satisfies: `max|phi| = 3.8e-15` on `z = 0` for the `A_inf` branch (net flux 0), and
  `max|dphi/dz| = 4.8e-16` for the rigid-wall branch, whose net flux `2VB = 40 m` shows that
  the 2-D zero-frequency limit is not an admissible exterior problem.
* Informativeness machinery: `minimum_detectable_defect` (deterministic defect→deviation
  curve + Monte-Carlo detection probability) and `negative_controls` (five mutants).
* Tests `tests/test_ice_gate_power_section.py` (6 passed, 31 s): closed-form circle and
  half-immersed-circle baselines, mirror-line residuals, ladder monotonicity/order,
  rejected controls, and the factor-2 halving identity.
* Artifact `benchmarks/verified/ice_gate_power_section/results.json`, doc
  `docs/ice_gate_power_section.md`, figure `docs/fig_ice_gate_power_section.png`.

## 2. Numbers (programmatic injection, no hand-typed values)

| quantity | value | baseline | deviation | < 3 %? |
|---|---|---|---|---|
| full circle `A` | 3220.117 kg/m | `rho*pi*a^2` = 3220.132 | 5.0e-6 | yes |
| full ellipse `A` | 12885.36 kg/m | `rho*pi*a^2` = 12880.53 | 3.8e-4 | yes |
| half-immersed circle `A_inf` | 1610.058 kg/m | `0.5*rho*pi*a^2` = 1610.066 | 5.0e-6 | yes |
| section `A_inf` (Richarson, 4 levels) | **225.35 t/m** | audit #19234: 228.6 t/m | +1.44 % | yes |
| section `A_inf` | 225.35 t/m | chain declared 86.1 t/m | −61.8 % (165 % gap) | **no** |
| rigid-wall branch (not admissible) | −143.7 t/m | — | net flux 40 ≠ 0 | n/a |

Observed order on the fine ladder: `p = 1.25` (corner-singular section, constant panels);
finest-pair change 0.19 %. Detection: a 3 % gate sees a 1.3 % section defect, a 15 % gate a
7 % defect. Rejected controls: no-halving (+100 %), wrong spectral limit (−163.9 %),
waterplane-lid closure (+83.1 %), mode-normal mixing (−100 %); the coarse-mesh lock (+2.6 %)
is **not** rejected at 3 %, so the criterion pins the mesh explicitly.

## 3. Limitations / not_run

* Reduced local configuration (ladder to 64 panels/side, 6 MC draws); the 128-point ladder
  row and a larger detection sample are not_run (cluster job `cj4244` was still queued when
  the artifact was written).
* No frequency-domain sweep: `tensorbem.free_surface_solver` requires a strictly submerged
  body, so the surface-piercing section is outside its domain.
* 3-D caliber of the declared coefficient is out of scope here; only the 2-D per-length
  comparison is adjudicated.
