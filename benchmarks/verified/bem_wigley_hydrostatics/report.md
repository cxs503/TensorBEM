# BEM verification anchor: Wigley III hydrostatics

**Case.** Wigley III parabolic hullform, `L = 10.0 m`,
`B = 1.0 m`, `T = 0.625 m`
(`L/B = 10`, `B/T = 1.6`),
rho = 1025.0 kg/m^3, g = 9.80665 m/s^2.
The wetted surface is the port/starboard triangle mesh of
`tensorbem.wigley_hydrostatics.wigley_wetted_hull_mesh` (the post meridians and
the keel line are shared grid lines, so the surface is closed there and the only
boundary is the calm-waterline loop).

**Method.** Displaced volume and centre of buoyancy come from the divergence
theorem over the wetted mesh, the waterplane area and its second moments from
the waterline polygon integrals (`tensorbem.hydrostatic`); the wetted area is
the sum of the panel areas.  None of these routines knows the Wigley closed
forms, which are quoted independently:

- `V = 4 L B T / 9 = 2.777778 m^3`, `Cb = 4/9 = 0.444444`
- `KB = 5T/8 = 0.390625 m`, `A_wp = 2LB/3 = 6.666667 m^2`
- `I_xx = 4B^3L/105 = 0.380952 m^4`, `I_yy = B L^3/30 = 33.333333 m^4`
- wetted area by 2-D Gauss quadrature of `sqrt(1+y_x^2+y_z^2)`:
  `14.879063 m^2`

**Results.**

| mesh | panels | V [m^3] | Cb | KB [m] | S_w [m^2] | err V % | err Cb % | err S_w % |
|---|---|---|---|---|---|---|---|---|
| 16x8 | 512 | 2.764257 | 0.442281 | 0.390399 | 14.87226 | 0.4868 | 0.4868 | 0.0457 |
| 32x16 | 2048 | 2.774390 | 0.443902 | 0.390568 | 14.87737 | 0.1220 | 0.1220 | 0.0114 |
| 64x32 | 8192 | 2.776930 | 0.444309 | 0.390611 | 14.87864 | 0.0305 | 0.0305 | 0.0028 |
| 128x64 | 32768 | 2.777566 | 0.444411 | 0.390621 | 14.87896 | 0.0076 | 0.0076 | 0.0007 |

Longitudinal centre of buoyancy is zero by fore-aft symmetry and is checked as
such (`|LCB| = 2e-17 m`); the
non-trivial vertical position is `KB`, matched to
`0.0009 %`.

Grid convergence (Roache, refinement ratio 2, safety factor 1.25):
volume `p = 2.00`, `GCI = 0.01 %`;
wetted area `p = 2.00`, `GCI = 0.00 %`.

**Acceptance.**

| criterion | measured | limit | verdict |
|---|---|---|---|
| Cb_error | 0.000305116 | 0.005 | PASS |
| volume_error | 0.000305116 | 0.01 | PASS |
| wetted_area_error | 2.84053e-05 | 0.02 | PASS |
| gci_volume | 9.54447e-05 | 0.03 | PASS |
| gci_wetted_area | 8.85896e-06 | 0.03 | PASS |
| LCB_zero_by_symmetry | 1.99901e-17 | 1e-09 | PASS |

**Limitations.** The reference is closed form, so this anchor quantifies mesh
discretisation only; it says nothing about the accuracy of a physical
displacement measurement.  The hull is the mathematical Wigley form used for
numerical benchmarking, not a real ship: no appendages, no trim or sinkage, and
the waterline is exactly planar.
