# External anchors for CASE-BEM-IRREG-FREQ-CYL (audit m17396)

## A1 — Bessel zero data (validates the constants, not the prediction)
Abramowitz & Stegun, *Handbook of Mathematical Functions*, Table 9.5 (pp. 370–373):
j_{0,1}=2.404825557695773, j_{1,1}=3.831705970207512, j_{2,1}=5.135622301840683.
Auditor re-derived the same zeros with mpmath at 30 significant digits
(`bessel_zeros_mpmath`, dps=30); max |library literal − mpmath| = **4.441e-16**.
The library literals are stored at `src/tensorbem/irregular_frequency.py:82-86`; the mission's
"reference" uses the same numbers, which is why the mission benchmark cannot validate them.

## A2 — Free-surface-consistent eigenvalue (route B, non-adjudicating here)
John (1950) *Comm. Pure Appl. Math.* 3; Hulme (1982) *JFM* 121; Zhu (1993) *Proc. 8th IWWWFB*.
For a wall-sided cylinder: tan(k_z T) = k_z / nu, nu = hypot(j_{m,n}/R, k_z).
Implementation: `src/tensorbem/benchmarks/irregular_frequency_cylinder_exact.py:237-261`.
Result: 18.63 % (T/R=1, m=2) to 70.57 % (T/R=0.5, m=0) above the shipped predictor.

## A3 — Solver-side empirical anchor (route C, the adjudicating one)
Signature of an irregular frequency in the *discrete* problem: condition-number blow-up and loss of
radiation damping (passivity) in the unaugmented radiation solve; band located by parabola refinement
of the conditioning peak (`locate_peak`). Auditor rerun 2026-11-13:
nu_obs = 2.4863 rad/m (112 panels) and 2.4389 rad/m (264 panels) vs predictor 2.4048 rad/m
(+3.39 % / +1.42 %). m16777 documents 2.45 rad/m, so the empirical anchor carries ~2 % resolution
uncertainty — it supports trend/magnitude, not four-digit agreement.
