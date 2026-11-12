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
