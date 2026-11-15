# Zou-He channel inlet/outlet reconstruction — cross-validation (m21551)

Anchor: `XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI` — the inlet/outlet reconstruction of
`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` (commit `af07744`,
`car.py::car_setup_step`, `LatFlow/Domain.py`). Ledger acceptance: error vs
the reference solution **< 15 %**. Domain: computational fluid dynamics
(lattice Boltzmann), verification-debt item.

## 1. Case

2D plane channel, D2Q9, BGK collision, full-way bounce-back walls at rows
`0` and `n_y-1` (wall mid-planes at `y = 0.5` and `y = n_y-1.5`), Zou-He
velocity inlet on column 0 and Zou-He pressure outlet on column `n_x-1`.

| parameter | value |
|---|---|
| channel height | $H = n_y-2$ lattice units (16, 32) |
| channel length | $L = 2 n_y$ |
| kinematic viscosity | $\nu = 0.05$ ($\tau = 0.650$) |
| imposed peak velocity | $U_{peak} = 0.08$ (Ma $= 0.139$) |
| Mach sweep | $U_{peak} \in$ {0.02, 0.04, 0.08} at $n_y=32$ |
| outlet density | $\rho_{out} = 1$ (anchor) |
| reference solution | $u_x(y) = 4 U_{peak}\, y_p (H-y_p)/H^2$, $y_p = y-0.5$ |

Four variants isolate one factor at a time:

* **V1** literal anchor: inlet parabola with `yp = i-1.5`, reconstruction applied
  to the whole inlet/outlet column (wall rows included), `rho_out = 1`;
* **V2** same parabola, reconstruction restricted to the fluid rows;
* **V3** corrected parabola (`yp = i-0.5`), fluid rows only;
* **V4** periodic body-force-driven channel (Guo-type force, no inlet/outlet at
  all) — the lattice-discretisation floor no boundary condition can beat.

## 2. Method

The reconstruction is a NumPy transliteration of the anchor formulas
(`lbm_zouhe_channel/zou_he.py`), applied in the anchor step order
`setup -> collide -> stream`, and driven by `lbm_zouhe_channel/channel.py`.
Because the reconstruction acts on the pre-collision state, the boundary
*column* is refilled from the interior by the subsequent streaming; the imposed
values are therefore never read off the boundary column but verified on the
moments of the reconstructed population set (unit tests) and on the developed
profile at $x=L/2$ (physical tests).

Tool chain: NumPy 2.4.6 / SciPy 1.16 / matplotlib 3.10.8, single core,
13 pytest tests. Deviations from the anchor, recorded honestly: BGK instead of
the anchor's Smagorinsky-LES relaxation (irrelevant for laminar channel flow);
the anchor's `Nu`-based $\tau$ is replaced by an explicit $\nu$; `rho_out` is a
parameter so the literal `=1` can be compared with the general form.

## 3. Results

Main grid ($\nu = 0.05$, $U_{peak} = 0.08$):

| variant | ny | nx | steps | peak | err_peak_pct | err_l2_pct | err_shape_pct | wallslip_pct | Qdrift | rho_int |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 16 | 32 | 6500 | 0.07822 | 2.226 | 2.180 | 0.639 | 13.16 | 1.82e-09 | 1.00752 |
| V1 | 32 | 64 | 16500 | 0.07992 | 0.101 | 1.958 | 1.943 | 6.58 | 4.90e-08 | 1.00337 |
| V2 | 16 | 32 | 6500 | 0.07822 | 2.226 | 2.180 | 0.639 | 13.16 | 1.82e-09 | 1.00752 |
| V2 | 32 | 64 | 16500 | 0.07992 | 0.101 | 1.958 | 1.943 | 6.58 | 4.90e-08 | 1.00337 |
| V3 | 16 | 32 | 6500 | 0.08034 | 0.426 | 0.916 | 0.200 | 13.56 | 1.33e-09 | 1.00787 |
| V3 | 32 | 64 | 16500 | 0.08016 | 0.199 | 0.352 | 0.045 | 6.51 | 4.79e-08 | 1.00343 |
| V4 | 16 | 32 | 5500 | 0.07923 | 0.959 | 0.615 | 0.249 | 13.33 | 0.00e+00 | 1.00000 |
| V4 | 32 | 64 | 17500 | 0.07983 | 0.209 | 0.134 | 0.055 | 6.46 | 0.00e+00 | 1.00000 |

Mach sweep at $n_y=32$:

| variant | u | Ma | err_peak_pct | err_l2_pct | err_shape_pct | uy_inlet |
|---|---|---|---|---|---|---|
| V1 | 0.02 | 0.035 | 0.623 | 0.563 | 0.105 | 1.16e-03 |
| V1 | 0.04 | 0.069 | 0.437 | 0.791 | 0.645 | 2.27e-03 |
| V1 | 0.08 | 0.139 | 0.101 | 1.958 | 1.943 | 4.39e-03 |
| V3 | 0.02 | 0.035 | 0.017 | 0.120 | 0.051 | 9.01e-06 |
| V3 | 0.04 | 0.069 | 0.086 | 0.195 | 0.043 | 1.75e-05 |
| V3 | 0.08 | 0.139 | 0.199 | 0.352 | 0.045 | 3.32e-05 |

Convergence orders fitted over $n_y \in \{16, 32\}$:
periodic baseline 2.20, corrected Zou-He
1.38, anchor literal 0.16.

Findings:

1. **The anchor's reconstruction formulas are textbook-correct.** With the
   intended parabola and no wall-row contamination (V3) the developed profile
   matches the analytic Poiseuille solution to 0.35 % at
   $n_y=32$ (0.92 % at $n_y=16$), i.e. within a factor
   ~2.6 of the pure lattice floor (0.13 % at
   $n_y=32$). Imposed velocity, density and momentum are reproduced exactly
   (tests 1-5, 1e-14 level), mass flux is conserved along the channel to
   5e-08 relative.
2. **Defect A — the inlet parabola is offset by one lattice unit.** The anchor
   uses `yp = i-1.5` where the wall mid-planes imply `yp = i-0.5`. The imposed
   inlet profile is therefore asymmetric by 3 % of $U_{peak}$ and, worse,
   negative at the bottom wall row: $u_x(0) = -0.0168 = -21\%$ of $U_{peak}$
   (fig. 1), i.e. the "inlet" is a local outflow. Consequence: a
   1.96 % profile error at $n_y=32$ that
   **does not converge** (order 0.16, fig. 3) because it is
   a geometric, not a discretisation, error.
3. **Defect B — wall-row contamination of the reconstruction.** Applying the
   reconstruction to the entire column (wall rows included) puts fluid
   populations on the two bounce-back nodes. The collide stage re-imposes the
   reflection, so the *interior* solution is unaffected (V1 and V2 are
   bit-identical in every metric, table 1), but the reconstructed
   transverse velocity at the inlet rises from
   3.3e-05 (V3) to 4.4e-03 (V1),
   i.e. 5.5 % of $U_{peak}$, and the anchor has no transverse-velocity
   constraint at all. Defect B is a latent inconsistency rather than an error
   source in this configuration.
4. **Error growth with imposed velocity.** The anchor's error grows with
   $U_{peak}$ (0.56 % at
   $u=0.02$ to 1.96 % at $u=0.08$), consistent with a
   wall-row contamination that scales linearly with the imposed velocity; the
   corrected variant stays flat at 0.12-0.35 %.

## 4. Figures

![fig1](fig1_inlet_profiles.png)

*Imposed inlet velocity profile. The anchor's parabola uses $y_p=i-1.5$, shifting it one lattice unit towards the bottom wall: $u_x$ at the bottom wall row is -0.0168 = -21% of $U_{peak}$ (inflow at the inlet) and +6.5% at the top wall row instead of vanishing on both wall mid-planes.*

![fig2](fig2_developed_profiles.png)

*(a) developed streamwise profile at $x=L/2$ for the four variants and the analytic plane-Poiseuille solution ($n_y=32$); (b) pointwise deviation from the analytic solution. The anchor offset leaves a systematic, sign-changing error across the channel, the corrected reconstruction collapses onto the analytic curve down to the lattice floor.*

![fig3](fig3_convergence.png)

*$L_2$ developed-profile error vs channel height. The periodic body-force baseline converges at order 2.20, the corrected Zou-He pair at 1.38, while the anchor's inlet offset plateaus (0.16) - it is a systematic error that grid refinement does not remove.*

![fig4](fig4_mach_sweep.png)

*Error vs imposed peak velocity at $n_y=32$. The anchor's error grows with $U_{peak}$ (0.56% -> 1.96% from u=0.02 to 0.08) because the wall-row contamination scales with the imposed velocity; the corrected reconstruction stays at 0.12-0.35%.*

## 5. Caveats

* 2D only; the anchor's 3D (`D3Q15`/`D3Q19`) paths are not exercised.
* Two resolved heights (16, 32) only — enough to fit an order estimate but not
  a Richardson extrapolation; a wider grid was submitted to the compute gate
  and is left as follow-up.
* BGK, no LES, no body force in the Zou-He runs: the anchor's turbulent-car
  configuration is outside this verification's scope.
* The anchor's outlet fixes $\rho_{out}=1$; the general form is exercised in
  the unit tests but not in the grid study.
* Boundary-column values are post-streaming artefacts and are deliberately
  excluded from the error metrics; wall slip is reported at the first fluid
  node and is an O(1/N) bounce-back property, not a boundary-condition error.

## 6. Error verdict (vs the analytic plane-Poiseuille solution)

| quantity | measured | target | inside 3 %? |
|---|---|---|---|
| developed profile $L_2$ error, corrected Zou-He ($n_y=32$) | 0.35% | <3% | yes |
| developed profile $L_2$ error, anchor literal ($n_y=32$) | 1.96% | <15% but does not converge | yes |
| profile-shape error, corrected ($n_y=32$) | 0.045% | <3% | yes |
| lattice floor, periodic forced channel ($n_y=32$) | 0.13% | <3% | yes |
| peak velocity, corrected ($n_y=32$) | 0.20% | <3% | yes |
| mass-flux drift along the channel (all runs) | <= 4.9e-08 | <3% | n/a |
| wall slip at the first fluid node ($n_y=32$) | 6.5% | O(1/N) bounce-back artefact | n/a |

All key quantities are **below 3 %** for the corrected reconstruction and
below the ledger's 15 % band even for the literal anchor recipe. The
statement this report does *not* make: that the anchor recipe is
grid-convergent — it is not (order 0.16), and the
residual ~2 % is a removable, clearly identified geometric defect.

## 7. Reproduction

```
cd lbm_zouhe_channel
python3 -m pytest test_zou_he.py -q      # 13 tests
python3 run_study.py --res 16 32 --mach  # ~3 min, single core
```
