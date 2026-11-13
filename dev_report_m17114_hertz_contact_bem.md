# m17114 — CASE-FEM-HERTZ-CONTACT by the BEM (Boussinesq) route

**Anchor** `CASE-FEM-HERTZ-CONTACT` (feature ledger, `open`): contact radius `a`
and penetration `delta` of a sphere pressed into an elastic half-space versus the
Hertz closed form, tolerance < 3 %, load prescribed.

**Route chosen and why.** The dispatched worktree is `TensorBEM` and the anchor's
acceptance is a *physical* requirement on `(a, delta)`, not on a discretisation.
`TensorFEM` already carries the finite-element route
(`tensorfem.hertz_axisymmetric` + `benchmarks/verified/fem_hertz_contact`), so
re-solving the same case with a second, structurally different method is the
increment with the highest information content: it turns a single-route claim into
a cross-method verification, and it fills the one classical boundary-integral
capability the repository was missing entirely (it had no elasticity at all).
The same parameters as the FEM route are used (two 50 mm steel spheres, 10 kN →
equivalent `R = 25 mm`, `E* = 115.38 GPa`), so the two routes are directly
comparable.

## Method

Axisymmetric Boussinesq boundary integral on annular elements,

    w(x) = 1/(pi E*) int_A p(y)/|x-y| dA,   E* = E/(1-nu^2),
    int_0^{2pi} dphi/|x-y| = 4 K(k)/(r+rho),  k^2 = 4 r rho/(r+rho)^2,

with the self element (log singularity) integrated by adaptive quadrature split
at the collocation radius. The contact problem is the Signorini complementarity
problem, solved as `min_{p>=0} 1/2 p^T C p - p^T b`, `b = delta - g`, by a primal
active-set QP with line search. Load control by root finding on `delta` (bracket +
Brent); the domain radius is grown automatically until the pressure decays at the
truncation boundary. Contact radius is an output: a power-law rim fit
`p = K (a-r)^m` with `m` free, plus a shape-free outer-edge estimate.

## Evidence

`benchmarks/verify_elastic_contact_hertz.py` → `artifacts/elastic_contact_hertz/results.json`
(+ `pressure_profile.png`, `convergence.png`). Numbers, not prose, are in that
JSON; the platform report is generated from it.

Independent checks of the two ingredients (so the Hertz comparison is not
self-consistent-only): ring kernel vs direct angular quadrature (rel 1e-9);
influence matrix vs Love's `w(0) = 2 p a / E*`; and the solver itself vs the
**flat-punch** closed form `P = 2 E* a delta`, a different solution class.

Verification: `tests/test_elastic_contact.py` — 19 tests, all passing.

## Deviations / notes

* The anchor is tagged FEM but was dispatched to this (BEM) worktree; the
  physical acceptance is met by an independent route. Suggested ledger follow-up:
  add `CASE-BEM-HERTZ-CONTACT` (same acceptance, BEM route) and cite both rows in
  the case pool so the case carries two independent methods.
* Out of scope here: layered media, plasticity, adhesion, tangential loading,
  rough surfaces, finite-thickness substrates.
