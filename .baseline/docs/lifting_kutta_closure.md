# Kutta-closed lifting boundary-element solve (2-D)

Every number in this note is written by `python -m tensorbem.benchmarks.lifting_solver`, from `docs/m2587/outputs/lifting_kutta/benchmark.json`; the figures next to it are produced by the same run. Reference: the closed-form symmetric Joukowski solution of `tensorbem.lifting`, which shares no code with the boundary-element solver.

## 1. Case

Steady, inviscid, incompressible potential flow past a symmetric (zero-camber) Joukowski section; the circulation is not boundary data but the unknown fixed by the Kutta condition at the cusp trailing edge.

| quantity | value |
|---|---|
| circle-plane offset / transform radius | 0.1 / 1.0 |
| chord c | 4.0333 m |
| thickness ratio t/c | 0.1179 |
| incidence | 4.0 deg |
| free-stream speed U, density rho | 1.0, 1.0 |
| panels | 32, 64, 128, 256 (equal arc length, vertex on the cusp) |
| singularity for the response | point vortex at leading edge + 0.25 c, strength 1 |

## 2. Method

The disturbance is split as `phi = phi_U + Gamma phi_v`. `phi_U` is the non-lifting disturbance (`d(phi)/dn_body = -U.n_body`), solved with the exterior Neumann second-kind operator `(1/2 I - H) u = S q` already used by `solve_laplace_exterior_neumann`. `phi_v` is the response to a unit counter-clockwise point vortex inside the section plus the Neumann correction that cancels its wall-normal velocity; its circulation around the section is therefore exactly 1, and the correction is single valued on the contour.

`Gamma` follows from the two panels adjacent to the cusp: they must carry the same velocity vector, and since their tangents are anti-parallel that is `v_t[0] + v_t[-1] = 0`, one linear equation. Surface velocities are reconstructed from the boundary potential with the package's three-point rule (one-sided stencils at the array ends, so the two sides of the cusp never mix).

### 2.1 Convention, and why the earlier attempt looked inverted

`JoukowskiAirfoil.exact_circulation()` returns `Gamma_doc = 4 pi a U sin(alpha)`, the value that pairs with `C_L = 2 Gamma/(U c)` in the complex potential. The point-vortex convention used by the solver is counter-clockwise positive, `Gamma_ccw = closed_integral(v.ds)`, and Kutta-Joukowski reads `C_L = -2 Gamma_ccw/(U c)`. The two differ by a sign: `Gamma_ccw = -Gamma_doc`. The previous closure compared its `Gamma` with `Gamma_doc` without that sign change and reported an inverted circulation together with a pressure lift that disagreed with the circulation lift. The composite field was in fact consistent; the bookkeeping was not.

## 3. Results

### 3.1 Mesh ladder at the reference incidence

| panels | Gamma_ccw | -Gamma_doc (exact) | circ. rel. error | C_L circulation | C_L pressure | C_L exact | lift rel. error | surface-speed rms error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | -0.87202 | -0.96424 | 9.56% | 0.4324 | 0.4484 | 0.4781 | 6.22% | 4.32% |
| 64 | -0.91856 | -0.96424 | 4.74% | 0.4555 | 0.4607 | 0.4781 | 3.65% | 2.76% |
| 128 | -0.94137 | -0.96424 | 2.37% | 0.4668 | 0.4673 | 0.4781 | 2.26% | 0.99% |
| 256 | -0.95287 | -0.96424 | 1.18% | 0.4725 | 0.4718 | 0.4781 | 1.32% | 0.27% |

Observed orders from successive mesh halvings: 32->64: 1.01, 64->128: 1.00, 128->256: 1.01 (first order, as expected from piecewise-constant panels).

### 3.2 Incidence sweep (solver runs on every row)

| alpha [deg] | panels | Gamma_ccw | circ. rel. error | C_L pressure | C_L exact | circulation vs pressure gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 32 | 0.00000 | 0.00% | -0.0000 | 0.0000 | 0.00% |
| 0.0 | 64 | 0.00000 | 0.00% | -0.0000 | 0.0000 | 0.00% |
| 0.0 | 128 | 0.00000 | 0.00% | -0.0000 | 0.0000 | 0.00% |
| 0.0 | 256 | 0.00000 | 0.00% | -0.0000 | 0.0000 | 0.00% |
| 2.0 | 32 | -0.43628 | 9.56% | 0.2248 | 0.2392 | 3.89% |
| 2.0 | 64 | -0.45956 | 4.74% | 0.2309 | 0.2392 | 1.33% |
| 2.0 | 128 | -0.47097 | 2.37% | 0.2342 | 0.2392 | 0.30% |
| 2.0 | 256 | -0.47672 | 1.18% | 0.2365 | 0.2392 | 0.04% |
| 4.0 | 32 | -0.87202 | 9.56% | 0.4484 | 0.4781 | 3.70% |
| 4.0 | 64 | -0.91856 | 4.74% | 0.4607 | 0.4781 | 1.14% |
| 4.0 | 128 | -0.94137 | 2.37% | 0.4673 | 0.4781 | 0.12% |
| 4.0 | 256 | -0.95287 | 1.18% | 0.4718 | 0.4781 | 0.14% |
| 6.0 | 32 | -1.30670 | 9.56% | 0.6699 | 0.7165 | 3.39% |
| 6.0 | 64 | -1.37645 | 4.74% | 0.6882 | 0.7165 | 0.83% |
| 6.0 | 128 | -1.41063 | 2.37% | 0.6982 | 0.7165 | 0.19% |
| 6.0 | 256 | -1.42785 | 1.18% | 0.7049 | 0.7165 | 0.45% |

The circulation error is the same for every incidence at a given mesh, which is what a first-order discretisation error in the closure geometry looks like: the Kutta condition is linear in `Gamma`, so a panel-size effect cannot depend on the incidence.

### 3.3 The singularity location does not set the circulation

| vortex position (fraction of chord from the leading edge) | Gamma_ccw | C_L pressure |
|---:|---:|---:|
| 0.15 | -0.94115 | 0.4674 |
| 0.25 | -0.94137 | 0.4673 |
| 0.50 | -0.94246 | 0.4679 |
| 0.75 | -0.94757 | 0.4708 |

Only the closure geometry depends on the singular point; the Kutta condition selects the same circulation up to the discretisation error, as uniqueness of the potential-flow solution requires.

### 3.4 Negative control: the layer-gradient reconstruction

| method | panels | Gamma_ccw | circ. rel. error |
|---|---:|---:|---:|
| layer_gradient_cauchy_kernel | 64 | -1.24507 | 29.1% |
| three-point rule (shipped) | 64 | -0.91856 | 4.7% |

`docs/lifting_foil.md` recommended replacing the three-point rule by the exterior limit of the layer gradients. At a cusp that fails: the piecewise-constant single-layer density is logarithmically singular at panel ends, and the two panels that define the closure sit exactly there, so the reconstruction is dominated by the singular term. It is retained only as this negative control.

## 4. Figures

![circulation convergence](m2587/outputs/lifting_kutta/circulation_convergence.png)

Left: circulation and lift error versus mesh against the exact solution, with the 15% acceptance limit. Right: circulation against the exact value on every mesh.

![lifting solution](m2587/outputs/lifting_kutta/lifting_solution.png)

Left: BEM surface speed against the exact `|W|/U`, including the stagnation point at the cusp. Right: the pressure distribution the lift integral is taken over.

## 5. Error budget and verdict

| quantity | 128 panels | 256 panels | acceptance |
|---|---|---|---|
| circulation magnitude vs `4 pi a U sin(alpha)` | 2.37% | 1.18% | <3% |
| lift coefficient from the pressure integral | 2.26% | 1.32% | <3% |
| circulation lift vs pressure lift (self-consistency) | 0.12% | 0.14% | <1% |
| surface speed rms vs exact field | 0.99% | 0.27% | <3% |

**Verdict.** The lifting solve is shipped (`solve_lifting_flow`). At 128 panels the circulation is 2.37% low and the lift 2.26% low, i.e. below 3%; at 256 panels the errors are 1.18% and 1.32%. The 15% acceptance limit is met with a factor of four at 128 panels.

## 6. Notes, limitations, what is still missing

* The closure uses the two panels adjacent to the cusp, so its accuracy is set by the panel size at the trailing edge; it is first order, not second, and the circulation is biased low, never oscillating.
* The exact Kutta solution stagnates *at* the cusp; one panel away the exact speed is already about 0.91 U. A closure that produced fast trailing-edge panels (the rejected variant) is therefore visibly non-physical in the surface-speed plot.
* Symmetric zero-camber sections only; no camber, no flap, no blunt trailing edge. Steady, inviscid, unbounded, no free surface, no forward speed, no boundary layer: this is a hydrofoil/fin reference case, not a hull-flow solver.
* The response needs one interior singularity; the circulation is independent of its location to 0.7% over the studied range, but the field near that point is not physical, so field queries inside the body are meaningless.
* Not verified against experiment or other published BEM code: the acceptance reference is the exact Joukowski solution.
