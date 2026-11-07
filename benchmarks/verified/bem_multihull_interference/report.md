# BEM verification anchor: tandem multihull interference

**Case.** Two identical Wigley III hulls (`L = 10.0 m`,
`B = 1.0 m`, `T = 0.625 m`) on the same
track, the aft hull's FP a clear distance `gap = L/2, L, 2L` behind the fore
hull's AP, i.e. centre to centre `d = L (1 + gap)`.

**Definition.** `sigma = R_pair / (2 R_single)`.  `sigma = 2` means the pair
behaves like one hull of doubled strength (perfectly coherent wave systems),
`sigma = 1` means the two wave systems are incoherent and each hull pays its own
resistance.  Because `sigma` is a *ratio*, it is independent of the constant in
front of Michell's integral: it tests the physics of the interference, not the
absolute level.

**Method.** The far-field amplitude of the pair is the superposition
`A_pair = A_single (1 + exp(i alpha d))` with `alpha = k_0 lambda` the streamwise
phase wavenumber, so the reference is the same one-dimensional integral as the
single hull with that factor folded in.  The panel path sums the panels of both
centreplane sheets, the aft sheet shifted by `d` -- no superposition identity is
used, the phase comes out of the panel coordinates.  Panel grid
`40 x 40` per hull, with
`sigma` itself evaluated on three grids (20x20, 40x40, 80x80) so the
interference factor carries its own grid-convergence index
(`GCI = 0.023 %`,
observed order
`p = 2.63 ...
8.14`).

**Results.**

| Fr | gap/L | d/L | sigma reference | sigma panels | |delta sigma| | single-hull panel error % |
|---|---|---|---|---|---|---|
| 0.20 | 0.50 | 1.50 | 1.3773 | 1.3766 | 0.0007 | +0.65 |
| 0.20 | 1.00 | 2.00 | 1.3150 | 1.3139 | 0.0011 | +0.65 |
| 0.20 | 2.00 | 3.00 | 1.2670 | 1.2657 | 0.0013 | +0.65 |
| 0.30 | 0.50 | 1.50 | 1.1270 | 1.1269 | 0.0000 | +0.03 |
| 0.30 | 1.00 | 2.00 | 0.8829 | 0.8829 | 0.0000 | +0.03 |
| 0.30 | 2.00 | 3.00 | 0.7877 | 0.7877 | 0.0000 | +0.03 |
| 0.40 | 0.50 | 1.50 | 0.5035 | 0.5035 | 0.0000 | +0.00 |
| 0.40 | 1.00 | 2.00 | 1.4015 | 1.4015 | 0.0000 | +0.00 |
| 0.40 | 2.00 | 3.00 | 1.3182 | 1.3182 | 0.0000 | +0.00 |

Analytic limits and trend (at Fr = 0.3):
coincident hulls give `sigma = 2.000000` (exact limit 2);
separating the hulls to `d = 60 L` gives `sigma = 1.0107`
(limit 1; the residual is the slowly decaying stationary-phase tail of the
interference integral, not a discretisation error).

Spacing sweep:

| gap/L | sigma |
|---|---|
| 0.05 | 1.6013 |
| 0.10 | 1.4373 |
| 0.25 | 0.6953 |
| 0.50 | 1.1268 |
| 0.75 | 1.0112 |
| 1.00 | 0.8831 |
| 1.50 | 0.7702 |
| 2.00 | 0.7931 |
| 3.00 | 1.0599 |
| 4.00 | 1.1649 |
| 6.00 | 0.8233 |

**Acceptance.**

| criterion | measured | limit | verdict |
|---|---|---|---|
| sigma_agreement | 0.00128058 | 0.02 | PASS |
| sigma_coincident_limit | 0 | 1e-06 | PASS |
| sigma_far_field_trend | 0.0107283 | 0.1 | PASS |
| single_hull_panel_error | 0.00653283 | 0.05 | PASS |
| gci_sigma | 0.000226323 | 0.03 | PASS |

**Notes and limitations.** The tandem interference factor oscillates with
spacing and Froude number and decays slowly to 1; the values here are linear
superposition results (no hull-to-hull diffraction beyond the far-field phase,
no viscous interaction, no wave breaking between the hulls).  They are therefore
a *discretisation and consistency* anchor for the interference machinery, not a
prediction for a real multihull.
