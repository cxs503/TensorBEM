# Reference anchor for CASE-BEM-SPHERE-ADDED-MASS (audit copy)

Cited closed form (translating sphere in an unbounded inviscid fluid):

    m_A = (1/2) * rho * V = (1/2) * rho * (4/3) * pi * R^3

Provenance as recorded in the subject implementation docstring
(`references/sphere_added_mass_m17262.py`, `run_refinement` docstring):

* Lamb, H. *Hydrodynamics*, 6th ed., Cambridge University Press, 1932 — sec. 6.10
  (motion of a sphere in an infinite liquid: kinetic energy T = (1/4) m' U^2 with
  m' the displaced mass, i.e. added mass = half the displaced mass).
* Newman, J.N. *Marine Hydrodynamics*, MIT Press, 1977 — eq. 4.132 sphere limit
  of the ellipsoid added-mass tensor (k1 = 1/2 for a/b -> 1).

Machine cross-checks actually executed (no hand-typed number is used as a
reference value):

1. In-repo analytic module:
   `tensorbem.analytic.ellipsoid_added_mass([R, R, R], rho)[0]` compared against
   `0.5*rho*4/3*pi*R^3`; the subject test asserts the relative difference < 1e-9.
2. Independent second code path in the repo:
   `tensorbem.benchmarks.ice_ship_verification.sphere_added_mass_reference(R, rho)`.
3. Auditor numeric anchor: for R = 1 m, rho = 1025 kg/m^3 the closed form
   evaluates to 2146.754979953025 kg, reproduced from the artifacts, not typed.

Caveat (audit finding L5-not-claimed): the citations above are docstring-level;
no page scan / table image is attached, and the closed form is elementary, so the
mechanical cross-check (1)(2) was preferred over page provenance.