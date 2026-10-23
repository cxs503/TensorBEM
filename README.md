# TensorBEM

**TensorBEM** is the boundary-element solver in the Tensor scientific-computing ecosystem. Its goal is to provide verifiable boundary-integral solvers that can run independently and later be orchestrated by TensorPlatform agents. The current baseline includes a 2-D polygonal constant-panel Laplace solver and a 3-D triangular-surface constant-panel Laplace solver with interior and exterior Dirichlet formulations.

## Current milestones

The first milestone is a transparent 2-D Laplace BEM implementation with closed polygon geometry, constant panels, the 2-D fundamental solution, exact straight-panel integrals, collocation assembly, a manufactured unit-disk benchmark, and deterministic tests.

The second milestone adds a 3-D Laplace surface-element baseline: closed triangular meshes, constant panels, centroid collocation with the \(c=1/2\) free term, analytic double-layer solid angles, Duffy single-layer quadrature, an exact centroid self-panel integral, and interior/exterior Dirichlet formulations. The sphere added-mass benchmark, three cross-verification cases, and a refinement study are included.

The third milestone adds the classical benchmark layer that does not need a free surface: a **2-D exterior** Dirichlet formulation (`solve_laplace_exterior_dirichlet`, with an explicit logarithmic reference length and a reported capacity degeneracy), the **translating elliptic cylinder** added mass per unit span (\(\rho\pi b^2\) and \(\rho\pi a^2\)), the **translating ellipsoid** added mass along all three principal axes (\(\rho V\alpha_i/(2-\alpha_i)\) with Lamb's shape factors), and **interior solid harmonics** of degree 1-3 on the faceted sphere. Every reference is a classical closed form, checked against the sphere, thin-disc and slender-body limits. See [the classical benchmark report](docs/classical_benchmarks.md).

The fourth milestone adds the first free-surface layer: a deep-water frequency-domain Green function (principal-value integral evaluated by split Gauss-Legendre/Laguerre rules), the **indirect** source formulation that keeps the radiation condition intact, and the **Ursell submerged cylinder** radiation benchmark. The reference is Ursell's multipole solution, an independent analytical construction which reproduces the published zero-damping frequencies to machine precision; the panel solution is within \\(7.7\times10^{-3}\\) in added mass at 128 panels and a method-of-fundamental-solutions check within \\(2\times10^{-4}\\), with the near/far-field energy identity closing to \\(6\times10^{-3}\\) and improving under refinement.

The fifth milestone expands the naval-architecture & ocean-engineering verification layer across three previously uncovered regimes (see §§4.7-4.9 of [the marine BEM benchmark report](docs/marine_bem_benchmarks.md)):
1. **Exterior Neumann rigid-body tensors (`solve_laplace_exterior_neumann`, `solve_laplace_surface_neumann`, `compute_added_mass_matrix_2d`, `compute_added_mass_matrix_3d`)**: solves the physical rigid-body no-penetration condition \(\partial\phi_j/\partial n_{\text{body}}=n_j\) via the well-conditioned second-kind operator \((\tfrac12\mathbf I-\mathbf H)\boldsymbol\phi=\mathbf S(-\mathbf n)\) without prescribing analytical surface traces, verifying 2-D \((3\times 3)\) and 3-D \((6\times 6)\) translational added masses and rotational added moments of inertia against Lamb's (1932) closed forms at \(O(h^2)\) with \(O(1)\) condition numbers;
2. **2-D deep-water wave diffraction & Haskind reciprocity (`solve_diffraction_indirect`, `haskind_excitation_force`, `dean_cylinder_diffraction`)**: verifies Dean's (1948) zero-reflection theorem (\(|R|\approx 2\times 10^{-15}\)), Ursell/Ogilvie transmission phase shift, circular wave-load polarisation \(F_z=-\mathrm i F_x\), wave-energy conservation \(|R|^2+|T|^2=1\), and Haskind-Newman radiation-diffraction reciprocity on circular and elliptic submerged cylinders;
3. **MacCamy-Fuchs vertical-pile diffraction & CHIEF irregular-frequency suppression (`tensorbem.helmholtz`, `maccamy_fuchs_diffraction`)**: implements vectorised pure-NumPy Bessel/Hankel functions and a 2-D exterior Helmholtz BEM solver with Schenck (1968) interior CHIEF null-field augmentation, verifying \(O(h^2)\) convergence of 3-D horizontal wave force \(F_x\) and seabed overturning moment \(M_y\), the long-wave Morison limit \(C_M\to 2.0\), and suppression of the interior \(J_0/J_1\) irregular-frequency resonances (reducing condition numbers from \(665\text{-}1003\) to \(\sim 3\) and cutting the \(J_1\) dipole force error from \(10.35\%\) to \(0.28\%\) at 64 panels).

The sixth milestone (`v0.5.0`) delivers general **3-D deep-water free-surface radiation and diffraction** for both submerged (`TriangleMesh`) and waterline-piercing (`WettedHullMesh`) three-dimensional hulls (see §§4.10-4.11 of [the marine BEM benchmark report](docs/marine_bem_benchmarks.md)):
1. **3-D Kelvin/Havelock/Delhommeau Green function (`tensorbem.free_surface3d`)**: pure-NumPy vectorised Struve functions \(\mathbf H_0,\mathbf H_1\) and a 4-region dimensionless wave evaluator (near-axis factorial recurrence cone, Havelock finite-interval ODE representation, pole-subtracted Gauss-Legendre/Laguerre quadrature, and 17-term Newman asymptotic Legendre expansion) achieving \(10^{-11}\text{-}10^{-13}\) continuity across all seams and `< 1e-14` free-surface boundary-condition residual on \(z=0\);
2. **3-D 6-DOF wave radiation, diffraction, Kochin & CHIEF solver (`tensorbem.free_surface_solver3d`)**: splits both direct \(1/(4\pi r)\) and mirror-image \(1/(4\pi r_1)\) Rankine terms into analytic Van Oosterom-Strackee solid angles and polar/Duffy rules, evaluates 3-D azimuthal Kochin far-field damping and Haskind excitation reciprocity, and suppresses surface-piercing irregular frequencies via lid-free 3-D interior CHIEF null-field constraints;
3. **Hulme (1982) floating hemisphere & Wang/Linton submerged sphere benchmarks (`floating_hemisphere`, `submerged_sphere_wave`)**: verifies \(O(h^2)\) convergence (observed orders \(1.93\text{-}2.06\)) against Hulme's semi-analytical wave-free multipole expansion and an independent closed-form factorial/\(\operatorname{Ei}(2\nu f)\) submerged-sphere multipole solver, while demonstrating how 5 interior 3-D CHIEF equations eliminate the \(Ka\approx 2.58\) unphysical negative-damping resonance (\(B_{33}'=-0.104\to +0.0687\), condition number \(145.7\to 4.24\)).

Finite-depth 3-D John eigenseries kernels (e.g. Garrett's truncated dock), forward-speed Rankine-source wave resistance (Wigley hull), and Kutta lifting wakes remain outside current scope.

## Quick start

Requires Python 3.10+ and NumPy.

```bash
python -m pip install -e ".[dev]"
pytest
python -m tensorbem.benchmarks.unit_disk --panels 64
python -m tensorbem.benchmarks.cylinder_added_mass --panels 64
python -m tensorbem.benchmarks.refinement_study --panels 16 32 64 128 --output artifacts/refinement-study.json
python -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128 --output artifacts/analytical-study.json
python -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2
python -m tensorbem.benchmarks.sphere_velocity --subdivisions 2
python -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3 --output artifacts/three-dimensional-study.json
python -m tensorbem.benchmarks.ellipse_added_mass --panels 64 --axis x
python -m tensorbem.benchmarks.ellipse_added_mass --panels 64 --axis y
python -m tensorbem.benchmarks.ellipsoid_added_mass --subdivisions 2
python -m tensorbem.benchmarks.interior_harmonics --subdivisions 1 2 3 --degrees 1 2 3
python -m tensorbem.benchmarks.classical_study --panels 16 32 64 128 --subdivisions 0 1 2 3 --harmonic-levels 1 2 3 --output artifacts/classical-study.json
python -m tensorbem.benchmarks.ursell_cylinder --panels 128 --ka 0.6 --output artifacts/ursell.json
python -m tensorbem.benchmarks.rigid_body_added_mass --panels 16 32 64 128 --subdivisions 0 1 2 --output artifacts/rigid-body-added-mass.json
python -m tensorbem.benchmarks.dean_diffraction --panels 128 --ka 0.6 --output artifacts/dean-diffraction.json
python -m tensorbem.benchmarks.maccamy_fuchs --panels 16 32 64 128 --output artifacts/maccamy-fuchs.json
python -m tensorbem.benchmarks.floating_hemisphere --subdivisions 1 2 3 --output artifacts/floating-hemisphere.json
python -m tensorbem.benchmarks.submerged_sphere_wave --subdivisions 0 1 2 --output artifacts/submerged-sphere-wave.json
```

The benchmark writes JSON to stdout; the refinement study can also persist a machine-readable JSON report. The manufactured field is \\(u(r,\\theta)=r\\cos\\theta\\), with exact outward normal derivative \\(q=\\cos\\theta\\) on the unit circle. The polygon approximates the circle, so results should be assessed by refinement trends, not by treating the polygon as an exact circle.

The circular-cylinder benchmark reports the added mass per unit span for a translating cylinder, with the analytical reference \\(\\rho\\pi a^2\\). With exact straight-panel integration, its relative error falls from \\(2.62\\times10^{-2}\\) at 16 panels to \\(4.02\\times10^{-4}\\) at 128 panels. It is a restricted boundary-trace verification case, not evidence that TensorBEM supports general exterior-flow or wave-body problems. See [the marine BEM benchmark report](docs/marine_bem_benchmarks.md) for the benchmark landscape, derivation, and limitations.

The 3-D sphere benchmark reports the added mass of a translating sphere, with analytical reference \\(\\tfrac12\\rho V\\). The surface is a flat-faceted icosphere, the boundary trace is the sphere solution at panel-centroid angular coordinates, and the solve uses the exterior Dirichlet equation \\(\\mathbf S\\mathbf q_e=(\\tfrac12\\mathbf I-\\mathbf H)\\mathbf u\\). At 20/80/320/1280 panels the relative error is \\(2.13\\times10^{-1}\\), \\(6.48\\times10^{-2}\\), \\(1.74\\times10^{-2}\\),
\\(4.47\\times10^{-3}\\), with observed order near 2. The pointwise exterior flux converges only near first order because of polyhedral-edge singularities; the integral added mass is the primary benchmark. This is a Laplace verification case, not free-surface, wave-body, or ship-hull validation. See [the three-dimensional verification report](docs/three_dimensional_bem.md).
The 3-D **velocity** field is available through `evaluate_surface_velocity`. The single-layer gradient kernel is integrable and uses the same fixed-order Duffy rule as the potential; the double-layer gradient is hypersingular and is evaluated from the closed-form gradient of each panel's signed solid angle, which is exact for constant panels. Against the exact exterior velocity of a translating sphere the relative L2 error is \\(1.02\\times10^{-1}\\), \\(2.76\\times10^{-2}\\), and \\(7.07\\times10^{-3}\\) at 80/320/1280 panels, with observed order 2.03 and 2.01. The reconstructed velocity **on** the surface is a different matter. Differentiating the Green formula with the piecewise-constant traces of the discretization amplifies the \(O(h)\) trace error into an \(O(1)\) gradient error, so that path stops at 53-65% and does not improve under refinement; the cause is confirmed by four controlled experiments in `tensorbem.benchmarks.surface_velocity_study`. The **recovered** surface velocity `evaluate_surface_velocity_at_panels` avoids that amplification: it uses the exact identity \(\nabla\phi|_\Gamma=\nabla_{\mathrm{tan}}u+n\,q\), with the tangential derivative of the boundary values recovered by a superconvergent least-squares patch fit. Its relative L2 error is \(1.50\times10^{-1}\), \(4.37\times10^{-2}\), and \(1.38\times10^{-2}\) at 80/320/1280 panels, with observed order 1.78 and 1.67, so the surface velocity is usable. The same diagnosis shows that the representation path is also unreliable in a boundary layer of thickness \(O(h)\) around the body, so off-body velocity for force integration must be evaluated clear of that layer.

The 3-D velocity benchmark also records a per-probe surface-clearance report (`probe_clearance`), including distance, distance divided by `h_max`, threshold, and policy. Call `diagnose_surface_clearance(mesh, points, warning_factor=1.0, policy="warn")` before interpreting near-body velocity samples. This is a risk diagnostic, not an accuracy certificate; the field evaluator retains its separate hard guard for fixed-order near-singular quadrature.

CI runs the 2-D and 3-D benchmark families, executes both refinement studies, and uploads the resulting JSON evidence as a workflow artifact. This is a reproducibility gate, not a claim of general exterior-flow validation.

The classical benchmark layer reports \(h^2\) convergence for both added-mass families. The elliptic cylinder reaches a relative error of \(7.54\times10^{-4}\) (\(x\)) and \(9.00\times10^{-4}\) (\(y\)) at 128 panels, with observed order 2.00; the ellipsoid reaches \(4.17\times10^{-3}\), \(4.46\times10^{-3}\) and \(4.94\times10^{-3}\) along \((1.5,1.0,0.75)\) at 1280 panels, with observed order 1.99-2.01. Its per-panel flux still converges only near first order because of polyhedral-edge singularities, exactly as for the sphere, so both numbers are reported. The interior solid-harmonic suite quantifies the degree dependence that the earlier linear-field verification could not show: the flux error at 1280 panels grows from \(3.76\times10^{-3}\) (degree 1) to \(1.19\times10^{-2}\) (degree 3) while the interior probe error stays near \(2\times10^{-4}\).

The additional analytical suite covers a cubic harmonic field on a disk, a quadratic field on an ellipse, a separated solution on a square, and a singular field at an L-shaped re-entrant corner. It reports boundary-flux errors, fixed-probe field errors, flux balance, conditioning, and observed convergence orders. See [the analytical verification report](docs/verification.md) for case definitions, measured results, and limitations. CI also archives this suite's JSON evidence.

Install the optional plotting dependency and generate unit-disk field cloudplots (BEM potential, analytical \\(u=x\\), and absolute error) over the interior region \\(r\\le0.9\\), avoiding the near-boundary region where midpoint field evaluation is less accurate:

```bash
python -m pip install -e ".[visualization]"
python -m tensorbem.benchmarks.unit_disk --panels 64 --plot unit_disk_comparison.png
python -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128 --plot analytical_convergence.svg --output artifacts/analytical-study.json
python -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3 --plot artifacts/three-dimensional-convergence.svg --output artifacts/three-dimensional-study.json
```

## Project layout

- `src/tensorbem/geometry.py`: 2-D panel geometry and validation, regular polygons and ellipses
- `src/tensorbem/kernels.py`: 2-D Laplace kernels, exact straight-panel integrals, reference length
- `src/tensorbem/solver.py`: 2-D collocation assembly, interior/exterior Dirichlet & exterior Neumann solves, `(3, 3)` rigid-body added-mass tensor
- `src/tensorbem/analytic.py`: closed-form references (solid harmonics, ellipse/ellipsoid translational & rotational added mass, Bessel/Hankel/Struve functions, Ursell radiation, Dean diffraction, MacCamy-Fuchs pile loads, Hulme floating hemisphere, and Wang/Linton submerged-sphere multipoles)
- `src/tensorbem/geometry3d.py`: closed triangular surface meshes (`TriangleMesh`), waterline-truncated floating hull meshes (`WettedHullMesh`), icosphere/ellipsoid/hemisphere generation, and quality diagnostics
- `src/tensorbem/kernels3d.py`: 3-D Laplace kernels, solid angles, Duffy and self-panel quadrature
- `src/tensorbem/solver3d.py`: 3-D collocation assembly, interior/exterior Dirichlet & exterior Neumann solves, `(6, 6)` rigid-body added-mass tensor, and field/velocity evaluation
- `src/tensorbem/recovery.py` & `clearance.py`: superconvergent surface tangential gradient recovery and near-surface clearance diagnostics
- `src/tensorbem/free_surface.py` & `free_surface_solver.py`: 2-D deep-water Green function, indirect radiation/diffraction panel & MFS solvers, and Haskind excitation forces
- `src/tensorbem/free_surface3d.py` & `free_surface_solver3d.py`: 3-D deep-water Kelvin/Delhommeau Green function, 6-DOF radiation/diffraction panel solver, azimuthal Kochin far-field damping, Haskind reciprocity, and lid-free 3-D CHIEF irregular-frequency suppression
- `src/tensorbem/helmholtz.py`: 2-D exterior Helmholtz solver, Schenck CHIEF irregular-frequency suppression, and 3-D vertical-cylinder diffraction loads
- `src/tensorbem/benchmarks/unit_disk.py`: manufactured-solution benchmark
- `src/tensorbem/benchmarks/cylinder_added_mass.py`: translating-cylinder added-mass benchmark
- `src/tensorbem/benchmarks/refinement_study.py`: reproducible 2-D multi-resolution JSON report
- `src/tensorbem/benchmarks/analytical_suite.py`: four additional 2-D analytical cases and convergence plots
- `src/tensorbem/benchmarks/sphere_added_mass.py`: translating-sphere added-mass benchmark
- `src/tensorbem/benchmarks/sphere_velocity.py`: translating-sphere velocity-field benchmark
- `src/tensorbem/benchmarks/surface_velocity_study.py`: controlled study of the surface-velocity accuracy and its cause
- `src/tensorbem/benchmarks/three_dimensional_study.py`: 3-D verification cases, refinement study, and plots
- `src/tensorbem/benchmarks/ellipse_added_mass.py`: elliptic-cylinder added mass through the 2-D exterior Dirichlet solve
- `src/tensorbem/benchmarks/ellipsoid_added_mass.py`: ellipsoid and spheroid added mass along three axes
- `src/tensorbem/benchmarks/rigid_body_added_mass.py`: 2-D `(3, 3)` and 3-D `(6, 6)` exterior Neumann added-mass and rotational-inertia benchmark
- `src/tensorbem/benchmarks/interior_harmonics.py`: interior solid harmonics of degree 1-3
- `src/tensorbem/benchmarks/classical_study.py`: reproducible study of the classical benchmark layer
- `src/tensorbem/benchmarks/ursell_cylinder.py`: 2-D deep-water submerged-cylinder wave radiation benchmark
- `src/tensorbem/benchmarks/dean_diffraction.py`: 2-D deep-water wave diffraction, Dean's zero-reflection theorem, and Haskind reciprocity benchmark
- `src/tensorbem/benchmarks/maccamy_fuchs.py`: MacCamy-Fuchs vertical-pile wave diffraction and CHIEF irregular-frequency benchmark
- `src/tensorbem/benchmarks/floating_hemisphere.py`: Hulme (1982) 3-D floating-hemisphere radiation, diffraction, and 3-D CHIEF irregular-frequency benchmark
- `src/tensorbem/benchmarks/submerged_sphere_wave.py`: Wang/Linton 3-D submerged-sphere wave radiation and diffraction benchmark
- `tests/`: unit and numerical regression tests
- `docs/verification.md`: 2-D analytical verification cases, equations, numerical results, and limitations
- `docs/marine_bem_benchmarks.md`: marine BEM benchmark survey and numerical report
- `docs/three_dimensional_bem.md`: 3-D formulation, quadrature, verification cases, results, and limitations
- `docs/classical_benchmarks.md`: exterior 2-D formulation, ellipse/ellipsoid added mass, solid harmonics, and the free-surface roadmap

## Numerical conventions

The 2-D fundamental solution is \\(G(x,y)=-\\frac{1}{2\\pi}\\log(|x-y|/2)\\), satisfying \\(-\\nabla^2G=\\delta\\). In 2-D it is defined up to an additive constant; the reference length of 2 avoids the unit-capacity degeneracy of the single-layer matrix on a unit circle. Panels are ordered counter-clockwise with outward normals. The baseline uses constant boundary elements and midpoint collocation; both single- and double-layer straight-panel integrals are analytical, including off-diagonal and interior-field evaluation terms. This is a correctness-oriented baseline, not a high-order production method.

The 3-D fundamental solution is \\(G(x,y)=1/(4\\pi|x-y|)\\), satisfying \\(-\\nabla^2G=\\delta\\). Closed triangular meshes are ordered consistently with outward normals, and collocation is at triangle centroids where \\(c=1/2\\). Double-layer panel integrals are evaluated analytically from signed solid angles; single-layer self-panels use a polar integral about the centroid and other panels use an 8x8 Duffy-transformed Gauss rule. The interior Dirichlet equation is \\(\\mathbf S\\mathbf q=(\\tfrac12\\mathbf I+\\mathbf H)\\mathbf u\\); the exterior equation is \\(\\mathbf S\\mathbf q_e=(\\tfrac12\\mathbf I-\\mathbf H)\\mathbf u\\), where \\(\\mathbf q_e\\) uses the fluid-domain normal. The dense direct solve is intended for verification meshes, not large models. Field evaluation requires points to be at least \\(0.05h_{\\max}\\) from the surface; near-boundary near-singular quadrature is not implemented.

## Roadmap

1. **Done:** reproducible Laplace benchmarks and convergence/error studies for 2-D polygons and 3-D triangles.
2. **Done:** classical no-free-surface benchmarks with closed-form references (elliptic cylinder, ellipsoid, interior solid harmonics) plus exterior Neumann `3x3`/`6x6` rigid-body translational and rotational added-mass tensors.
3. **Done:** 2-D deep-water free-surface radiation (Ursell 1949), incident-wave diffraction, Dean's zero-reflection theorem, and Haskind reciprocity, plus vertical-cylinder wave diffraction (MacCamy-Fuchs 1954) with Schenck CHIEF irregular-frequency suppression.
4. **Done:** 3-D deep-water Kelvin/Delhommeau free-surface Green function, 6-DOF radiation/diffraction BEM for submerged and waterline-piercing hulls (`WettedHullMesh`), lid-free 3-D CHIEF irregular-frequency suppression, Hulme's (1982) floating hemisphere, and Wang/Linton's submerged sphere.
5. **In progress:** improve geometry, quadrature, and conditioning diagnostics. Exact 2-D panel integrals, analytic 3-D double-layer solid angles, self-panel polar integration, superconvergent tangential gradient recovery, and clearance diagnostics are in place; graded meshes, symmetry-plane reduction, higher-order/curved panels, and near-singular field quadrature remain.
6. **Next, needs new physics:** finite-depth 3-D John eigenseries Green function (Garrett's truncated vertical cylinder & multi-body gap resonance), hydrostatic restoring + 6-DOF frequency-domain RAO coupling, forward-speed Rankine-source wave resistance (Wigley hull), and Kutta-condition lifting hydrofoils.
7. **Future:** expose a stable task/result contract for TensorPlatform.

## License

Apache-2.0.
