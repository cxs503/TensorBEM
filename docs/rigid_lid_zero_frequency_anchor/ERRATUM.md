# ERRATUM: the ka -> 0 anchor of this artifact is superseded

This directory is the m16802 record of an honest FAIL and is **not** rewritten
(honest FAILs are never overwritten).  The sweep rows (A33, B33, condition
number, relative residual, mesh ladder) remain valid measurements.

What is superseded is the *reference* of two criteria:

* **R1-rigid-lid-limit.**  The module claimed ``A33(ka -> 0) = 2/3 rho pi a^3 =
  rho*grad`` "by the method of images".  Reflecting the body is right, but the
  mirrored half carries the mirrored velocity: under ``z -> -z`` a vertical
  velocity is preserved while the outward normal of the mirrored surface flips,
  so the equivalent full sphere has the *sliding* Neumann data
  ``dphi/dn = -V|cos theta|`` (two halves moving in opposite directions), not the
  translating data ``V cos theta``.  The corrected limit is
  ``A33(ka -> 0) = 0.8309570 rho*grad = 1783.8612 kg``, obtained independently by
  a Legendre spectral solve and by the exact Hulme (1982) multipole series
  (0.85 % agreement at ``ka = 2e-3``).  The "14.4 % honest FAIL" was therefore a
  false FAIL: the solver is within 0.8 % of the exact solution at ``ka = 0.1``.
* **R3-infinite-frequency-floor.**  The exact heave added mass is *not* bounded
  below by the pressure-release value ``0.5 rho*grad``: the exact series has an
  interior minimum ``0.38629 rho*grad`` at ``ka = 1.72``.  R3 would fail a correct
  solver.

Replacement gate (pointwise, no endpoint constant):
``docs/hemisphere_zero_frequency_envelope.md`` and
``docs/hemisphere_zero_frequency_envelope/{results.json,summary.md}``, pinned by
``tests/test_hemisphere_zero_frequency_envelope.py``.

## m17547 addendum: the corrected limit is now pinned (and the 0.8309570 is 6.8 ppm high)

The correction above quoted ``0.8309570 rho*grad``.  That number itself carries
a ~6.8 ppm error.  m17547 re-derived the even-parity series with **exact
rational Legendre coefficients** (closed form for ``int_0^1 mu P_n dmu`` using
``P_2j(0) = (-1)^j C(2j,j)/4^j``, instead of a Gauss-Legendre quadrature of the
kinked ``|mu|``) and removed the measured ``O(n**-4)`` tail by Richardson
extrapolation:

    A33(ka -> 0) / (rho*grad) = 0.830951320470   (tail 1.9e-11)

For reference, the values that were in circulation and their deviation from this
pin: ``0.8309541`` (the Gauss-quadrature series committed in
``benchmarks/verified/bem_rigid_lid_hemisphere{,_ka0}/results.json``) +3.4 ppm;
``0.8309570`` (quoted above) +6.8 ppm; ``0.826075`` (stale docstring of
``rigid_lid_hemisphere_attribution``) -5868 ppm; ``1.0`` (report #15809)
+203440 ppm.  None of these change any verdict: the solver headline stays
+0.283 % and the case gate (< 5 %) passes either way.

An **independent** axisymmetric MFS ring-source solve (no panels, no Legendre
series, no singular integrals) reproduces the pin to 3.6e-5 relative, and the
same machinery reproduces the exact translating-sphere answer 0.5 to 1.5e-7.
Artifacts: ``benchmarks/verified/bem_rigid_lid_hemisphere_axisym/``; code:
``tensorbem.benchmarks.rigid_lid_reference_pin``; tests:
``tests/test_rigid_lid_reference_pin.py``.
