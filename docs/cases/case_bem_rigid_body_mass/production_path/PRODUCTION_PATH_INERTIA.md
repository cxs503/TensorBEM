# Production RAO default inertia route — geometry-exact by default

Anchor ``CASE-BEM-RIGID-BODY-MASS``. The two frequency-domain chains
(`tensorbem.rao.compute_rao_at_frequency`, `tensorbem.ice_rao.compute_ice_rao_spectrum`)
accept ``inertia=None``. Up to this change that branch built the rigid-body mass
matrix with `hydrostatic.mass_matrix_from_hull_geometry`, i.e. the *equivalent
half-ellipsoid* fitted to the displaced volume and the waterplane second moments.
That reconstruction is exact for an elliptic waterplane and **8–20 % low** on the
rotational block of a box-like hull, which is outside the anchor's 3 % tolerance:
the approximation was the *default* of the production path.
`hydrostatic.mass_matrix_from_hull` now selects the wetted-mesh route
(`mass_matrix_from_wetted_hull`, exact to round-off for a polyhedral hull) and
falls back to the reconstruction only when the panels cannot be closed or
integrated, with a `RuntimeWarning` naming the reason.

## 1. Rigid-body mass matrix, box barge 20 × 8 × 3 m at neutral buoyancy

m = ρLB T = 4.920000e+05 kg, CG = (0, 0, -1.5) m, ρ = 1025 kg/m³.
Reference = the closed-form block form of Fossen (2011) §3.3.1 with the solid-box
second moments `m(L²+B²)/12` etc., evaluated in machine arithmetic.

| DOF | closed form | legacy route | legacy error | new default | new error |
|---|---|---|---|---|---|
| roll | 4.100000e+06 | 3.751742e+06 | -8.49% | 4.100000e+06 | +0.00e+00% |
| pitch | 1.787600e+07 | 1.477254e+07 | -17.36% | 1.787600e+07 | +0.00e+00% |
| yaw | 1.902400e+07 | 1.521920e+07 | -20.00% | 1.902400e+07 | +0.00e+00% |

Worst entry of the whole 6×6 matrix: legacy 20.00%,
new 0.00e+00% (round-off). Every rotational
entry of the legacy route violates the 3 % criterion.

## 2. Consequence in the response (RAO, ω = 1.0 rad/s, head seas)

| DOF | RAO with closed-form inertia | RAO with legacy inertia | legacy rel. diff | RAO with new default | new rel. diff |
|---|---|---|---|---|---|
| surge | 5.143977e-01 | 5.256096e-01 | 2.18e+00% | 5.143977e-01 | 0.00e+00% |
| sway | 4.930271e-03 | 4.195291e-03 | 1.49e+01% | 4.930271e-03 | 0.00e+00% |
| heave | 1.102132e+00 | 1.102133e+00 | 1.36e-04% | 1.102132e+00 | 0.00e+00% |
| roll | 4.999972e-03 | 4.451266e-03 | 1.10e+01% | 4.999972e-03 | 0.00e+00% |
| pitch | 1.138563e-01 | 9.987645e-02 | 1.23e+01% | 1.138563e-01 | 0.00e+00% |
| yaw | 1.038990e-03 | 1.189360e-03 | 1.45e+01% | 1.038990e-03 | 0.00e+00% |

Worst response error: legacy 14.91%,
new 0.00e+00%.

## 3. Curved hull: the mesh route converges to the analytic half-sphere

Sphere radius 1 m, m = 1000 kg, CG = (0, 0, −0.375) m; reference =
`I_xx = I_yy = (2/5 − 9/64) m a²`, `I_zz = (2/5) m a²`.

| subdivisions | panels | mesh route rel. error | legacy route rel. error |
|---|---|---|---|
| 2 | 64 | 2.3759e-02 | 1.1935e-02 |
| 3 | 256 | 6.1961e-03 | 3.2314e-03 |
| 4 | 1024 | 1.5658e-03 | 8.2460e-04 |

Fitted log-log slope of the mesh route over 2 → 4: **1.96**
(second order, as the surface integral of a quadratic moment should be). On an
elliptic waterplane the reconstruction is well matched to the geometry, so this
hull does not discriminate between the routes — the barge does.

## 4. Body mass vs displaced mass

The volume integral yields the second moments of a *fluid-filled* solid. A body of
the same shape and another mass has the same specific inertia, so the tensor is
scaled by `mass / (ρ ∇)`. Half-sphere at subdivisions = 3, m = 1000 kg against
neutral buoyancy (ρ ∇ = 2096.9 kg):
rotational diagonal ratio **0.4769**
(= m / (ρ ∇)); before the scaling the rotational block described the displaced
water and was 2.097× too
large for this body.

## 5. Error verdict

| quantity | reference | worst error | < 3 % ? |
|---|---|---|---|
| barge 6×6 matrix, new default | closed-form block form | 0.00e+00% | yes (round-off) |
| barge 6×6 matrix, legacy route | closed-form block form | 20.00% | **no** |
| barge RAO, new default | RAO with closed-form inertia | 0.00e+00% | yes (round-off) |
| barge RAO, legacy route | RAO with closed-form inertia | 14.91% | **no** |
| half-sphere matrix, new default, subdiv 4 | analytic half-sphere | 0.16% | yes |

Honest limit: a *non-uniform* mass distribution is not recoverable from the
surface — the mesh route assumes homogeneous density (CG is then the volume
centroid). A caller whose CG differs from that centroid still gets the
second-moment tensor of the homogeneous body; pass an explicit tensor when the
mass distribution is known to be inhomogeneous.

## 6. Reproduce

```
PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/production_path_inertia.py
PYTHONPATH=src python3 -m pytest tests/test_rao_default_inertia_mesh.py -q
```
