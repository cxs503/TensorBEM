# Multibody added mass and damping: frequency dependence and the two asymptotes

Campaign node `P106W2-N04`.  Code: `src/tensorbem/multibody3d.py` (module path
`tensorbem.multibody3d`: geometry, asymptote solver, radiation wrapper) and
`src/tensorbem/benchmarks/multibody_frequency_dependence_3d.py` (the study).
Generated record: `benchmarks/multibody_frequency_dependence_3d/`
(`report.md`, `results.json`, `summary.json`, figures).
Tests: `tests/test_multibody_frequency_dependence_3d.py`.

## 1. What this adds to the repository

`tensorbem.free_surface_solver3d.solve_radiation_3d` answers the radiation
problem of *one* wetted hull at *one* frequency.  A multihull design needs three
things it cannot give:

* the coefficients of several hulls abreast, i.e. the hydrodynamic interaction
  between them (`sigma = A_pair / (n * A_single)`);
* the zero-frequency limit `A(0)` (rigid-wall free surface `dphi/dz = 0`);
* the infinite-frequency limit `A(inf)` (pressure-release free surface
  `phi = 0`).

Neither limit is reachable from the Kelvin Green function in a converged way:
`nu = omega^2/g -> 0` degenerates the kernel and `nu -> infinity` makes it
oscillate on a scale far below any affordable panel.  Both limits are classical
*potential* problems, and this module solves them directly, so the study gets a
reference value for each end of the frequency axis that does not come from the
frequency-domain solver it is checking.

Existing anchors are untouched: `benchmarks/verified/bem_multihull_interference`
is steady wave *resistance* of a tandem pair (Michell/Kochin),
`benchmarks/verified/bem_ellipsoid_wave` and `bem_finite_depth` are single-body
frequency-domain anchors, and `benchmarks/bem_*` cover other formulations.  This
study is the first 3-D multihull *seakeeping* record in the repository.

## 2. Formulation

For the radiation potential of mode `j` with time factor `exp(-i omega t)`:

    laplace(phi_j) = 0            in the fluid,
    dphi_j/dn      = n_j          on the wetted hulls,
    dphi_j/dz      = nu phi_j     on the calm waterplane,  nu = omega^2 / g,
    outgoing waves                at infinity,
    A_ij + (i/omega) B_ij = -rho int_{hulls} phi_j n_i dS.

The free-surface condition gives the two limits directly:

| limit | condition on `z = 0` | parity `sigma` of `phi` | physics |
|---|---|---|---|
| `omega -> 0` | `dphi/dz = 0` | `+1` | no waves, the surface acts as a rigid wall |
| `omega -> infinity` | `phi = 0` | `-1` | pressure-release surface |

### Mirror-closure continuation

Close the wetted hull by its mirror image about `z = 0` into a closed double body
`D` (`mirror_closure`).  Solve the exterior Neumann problem on `D` with mode data
continued by `sigma`:

    q_j(M x) = sigma q_j(x),      M: (x, y, z) -> (x, y, -z).

Because `D` is invariant under `M`, the solution inherits
`phi_j(M x) = sigma phi_j(x)`; restricted to `z < 0` it is the required
half-space solution, so

    A_ij = -rho int_{Gamma_B} phi_j n_i dS

integrated over the original (upper) wetted panels *is* the limit value.  The
closure keeps the original panel order, so the continuation and the half-body
integral are index-exact and no factor of two is hidden anywhere.

Two internal diagnostics are reported with every solve:

* the **parity residual** `max |phi(M x) - sigma phi(x)|`, which measures how
  exactly the discretisation honours the continuation (values `1e-4 .. 1e-2` on
  the meshes used here -- the mirrored collocation rows differ by the quadrature
  error on near-singular panel pairs, so the residual is small but not zero --
  a diagnostic, not a proof);
* the **closure volume**, which must be twice the wetted displacement, so a
  broken weld cannot pass unnoticed.

Note the parity bookkeeping that makes this work for a *surface-piercing* body:
the rigid heave mode of the closed double body is `n_z^D(M x) = -n_z^D(x)`, i.e.
`sigma = -1`.  Rigid heave of `D` therefore *is* the pressure-release limit, and
the module's hemispheres unit test uses exactly that identity.

### Infinite-frequency value from the frequency sweep

The Kramers-Kronig relation for the radiation problem implies that once `B`
decays faster than `1/omega`,

    A(omega) = A(inf) + (2/pi) P int_0^inf B(w')/(w'^2 - omega^2) dw'
             = A(inf) + c/omega^2 + O(omega^-4).

The study fits `A = a + c/omega^2` to the resolved high-frequency ladder and
compares `a` with the pressure-release solve: two routes to the same number, one
numerical and one analytic-in-`omega`.

## 3. The study

* Hulls: two and three Wigley III hulls (Journee 1992, the dimensions of
  `tensorbem.wigley3_hull`) abreast, reference clear gap of half a beam.
  The exact principal particulars are injected into the generated report from
  the study record, not repeated here.
* Modes: sway, heave, pitch about a common origin, so the multihull moves as one
  rigid body and the ratio to `n_hulls x single hull` is pure interference.
* Frequencies: a decade and a half around the single-hull heave natural
  frequency `omega_n` (itself taken from the hull's own hydrostatics), up to the
  frequency where the affordable mesh stops resolving the radiated wave.
* Gap sweep: from touching hulls out to two and a half hull lengths, at two
  frequencies, with the power-law fit `|sigma - 1| = C (d/L)^(-n)`.
* Negative control: gap to zero and superposed hulls; failures are recorded.
* Discretisation: the longitudinal panel is tied to the radiated wavelength
  (an eight-panel target with a stated floor); rows below the floor are flagged
  and excluded from the criteria; Richardson triples at two frequencies.

Reproduce with

```bash
PYTHONPATH=src python runs/run_study.py runs/multibody_frequency_dependence_3d.json
PYTHONPATH=src python runs/make_report.py
PYTHONPATH=src python -m pytest tests/test_multibody_frequency_dependence_3d.py -q
```

The study is parallel (`workers` argument, default 8) and takes tens of minutes
on one machine; the unit tests take about six minutes because they include real
BEM solves.

## 4. Results

See `benchmarks/multibody_frequency_dependence_3d/report.md` for the generated
tables (mode-by-mode added mass and damping, interference factors, fitted decay
exponents, grid-convergence indices, criterion verdicts) and the figures
(`frequency_scan.png`, `interference_frequency.png`, `distance_decay.png`,
`asymptotes.png`, `negative_control.png`, `grid_convergence.png`).  All numbers
in that report are injected from `results.json`; none is typed by hand.

## 5. Limits

* Zero forward speed, deep water, linear potential flow; no viscous gap flow, no
  breaking, no hull-to-hull diffraction.
* The infinite-frequency sweep value is an extrapolation along the `1/omega^2`
  branch, not a sample at `omega = infinity`; the pressure-release solve is the
  independent anchor.
* Interference factors use mode diagonals; off-diagonal pair coupling is
  computed but not gated by the campaign criteria.
* The `omega/omega_n = 2 .. 3` rows are reported with their
  panels-per-wavelength count and excluded from the criteria when the affordable
  mesh stops resolving the radiated wave.