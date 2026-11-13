# Elastic normal contact by the boundary-integral route

<!-- anchor: bem-elastic-contact-boussinesq -->

`tensorbem.elastic_contact` solves the frictionless normal contact problem of an
isotropic elastic half-space by the Boussinesq boundary integral

    w(x) = 1/(pi E*) * int_A p(y)/|x - y| dA(y),      E* = E/(1 - nu^2)

discretised on annular elements with the angular average of the kernel done
analytically, `int_0^{2 pi} dphi/|x-y| = 4 K(k)/(r+rho)`, `k^2 = 4 r rho/(r+rho)^2`.
The contact problem is closed as a complementarity problem (Signorini) solved by
a primal active-set QP on the nodal pressures; contact radius, peak pressure and
the load at a prescribed approach are all outputs.

## Why it is in this repository

The library holds layer-potential kernels only (Laplace, Helmholtz,
free-surface Green functions). Contact is the one classical boundary-integral
problem with a closed-form benchmark that the repository could not previously
answer, and it is the independent second numerical route for the anchor
`CASE-FEM-HERTZ-CONTACT` (the finite-element route lives in `TensorFEM`). No
code, mesh or discretisation is shared with that route.

## What is verified

| check | reference | how |
|---|---|---|
| ring kernel | `int_0^{2 pi} dphi/sqrt(r^2+rho^2-2 r rho cos phi)` | direct adaptive quadrature, rel 1e-9 |
| influence matrix | `w(0) = 2 p a / E*` for a uniformly loaded circle (Love) | assembled matrix times a constant pressure |
| contact solver | rigid flat punch `P = 2 E* a delta` (Johnson 3.3) | same solver, different solution class |
| anchor metrics | Hertz `a`, `delta`, `p0`, `p(r)` (Johnson 4.2) | `benchmarks/verify_elastic_contact_hertz.py` |

Run `python benchmarks/verify_elastic_contact_hertz.py` to regenerate
`artifacts/elastic_contact_hertz/results.json` and the two figures.

## Limits and numerical pitfalls

* The Boussinesq inverse is a smoothing operator's inverse and is therefore
  ill-conditioned: near the axis the surface displacement is almost constant, so
  neighbouring small elements have nearly identical influence rows and an exact
  solve of the contact system returns spiked pressures (measured 25x the true
  peak, growing with the element count) that still reproduce the right
  displacements. The working-set subproblems are therefore solved by truncated
  SVD; the reported quantities are invariant over the truncation range
  1e-8 .. 1e-14 (tested).
* `scipy.linalg.cho_factor` segfaults for `n >= 60` in this environment (plain
  random SPD matrix, independent of this module). Use `numpy.linalg.cholesky`.
* Only axisymmetric, frictionless, linear-elastic, half-space (or equivalent
  sphere-sphere) problems are covered: no layered media, no plasticity, no
  adhesion (JKR/DMT), no tangential loading, no finite-thickness substrates.
