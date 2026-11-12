# Wigley III calm-water resistance: geometry gate, four wave-resistance routes and the Holtrop/ITTC empirical baseline

**Case** `CASE-BEM-WIGLEY-CALM-WATER-RESISTANCE` (TensorBEM) — driver `python -m tensorbem.benchmarks.wigley_calm_water_resistance --output artifacts/bem_wigley_calm_water_resistance`, artefacts `artifacts/bem_wigley_calm_water_resistance/{results.json,figures/*.png}`, tests `tests/test_wigley_calm_water_resistance.py` (10 cases).

**Verdict in one line**: geometry gate, the four wave-resistance routes and the ITTC-57 identity all pass; the pre-declared *total resistance vs Holtrop* criterion (**<= 10 % at every Fn in 0.15-0.40**) is an **honest FAIL** — 3 of 6 points pass under the primary convention, and no published coefficient convention reaches 6/6 (pass counts across the 12 conventions: 3-4).

## 1. Case introduction

### 1.1 Physical background

Wave resistance is the component of a ship's calm-water resistance that potential-flow theory can compute; friction is not, so a total-resistance prediction is always an assembly of a solved potential-flow part and an empirical friction line. This case verifies exactly that assembly on a hullform whose geometry is closed-form, and then asks the honest question the acceptance criteria ask: does the assembled total land within 10 % of the standard empirical prediction (Holtrop 1984 on the ITTC-1957 line)?

### 1.2 Geometry and parameter table

Parabolic Wigley III: $y(x,z) = \frac{B}{2}\left[1-(2x/L)^2\right]\left[1-(z/T)^2\right]$, $x\in[-L/2,L/2]$, $z\in[-T,0]$, symmetric about $y=0$.

| Parameter | Symbol | Value | Unit | Source |
|---|---|---:|---|---|
| Length | $L$ | 10.0 | m | case definition |
| Beam | $B$ | 1.0 | m | case definition |
| Draft | $T$ | 0.625 | m | case definition |
| Length/beam | $L/B$ | 10.0 | – | exact check (PASS) |
| Beam/draft | $B/T$ | 1.60 | – | exact check (PASS) |
| Block coefficient | $C_b$ | 0.444444 | – | closed form $4/9$ |
| Midship coefficient | $C_m$ | 0.666667 | – | closed form $2/3$ |
| Prismatic coefficient | $C_p$ | 0.666667 | – | closed form $2/3$ |
| Displacement | $\nabla$ | 2.777778 | m^3 | closed form $4LBT/9$ |
| Waterplane area | $A_{wp}$ | 6.666667 | m^2 | closed form $2LB/3$ |
| Wetted area | $S_w$ | 14.879063 | m^2 | 2-D Gauss reference |
| Centre of buoyancy | $KB$ | 0.390625 | m | closed form $5T/8$ |
| Water density | $\rho$ | 1025 | kg/m^3 | case definition |
| Gravity | $g$ | 9.80665 | m/s^2 | case definition |
| Kinematic viscosity | $\nu$ | 1.19e-6 | m^2/s | ITTC, sea water 15 °C |

Froude numbers: figure scan [0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]; criterion band [0.15, 0.2, 0.25, 0.3, 0.35, 0.4] (pre-declared: inside it both theories claim validity — thin-ship Michell for a slender form, Holtrop for $Fn\le0.45$).

## 2. Method

1. **Geometry gate.** The hull is panelised (`wigley_wetted_hull_mesh`) at 64x32 per side, and $C_b$, $\nabla$, $A_{wp}$ and $S_w$ from the mesh are compared with the closed forms. A failure here invalidates everything downstream, so it is a gate.

2. **Wave resistance, four routes.** (a) closed-form Michell amplitude with the $\lambda=1+s^2$ substitution and $\lambda$-truncation extrapolation (production); (b) independent $\theta$-form angular quadrature; (c) the **BEM** route — a Rankine centreplane source sheet `wigley_panels` on 16x8/32x16/64x32 panel grids with a Roache GCI; (d) a brute-force 2-D midpoint quadrature of $\iint y_x e^{kz+i\alpha x}dx\,dz$ that uses none of the closed-form factorisation.

3. **Friction and total.** $R_T = R_F(1+k) + R_A + R_W$ with $C_F$ from ITTC-1957, $1+k$ from Holtrop (1984) and $R_A = C_A\tfrac12\rho U^2 S$. The *same* $R_F(1+k)$ and the *same* $R_A$ appear in the Holtrop prediction, so the two totals differ through **one** variable only: the wave term. That is the controlled comparison of this case.

4. **Empirical baseline family.** Holtrop (1984) is implemented in `tensorbem.resistance` with the published convention switches exposed (`include_c17`, `regime ∈ {m4, cp2}`, `ca_convention ∈ {none, holtrop84, ittc78}`) and with `holtrop_applicability` reporting where the hull sits relative to the regression's data set.

5. **Pre-declared thresholds** (fixed before the sweep): `cb` 0.005, `volume` 0.01, `waterplane` 0.01, `wetted_area` 0.02, `route_theta` 0.01, `route_brute` 0.01, `route_panel` 0.02, `panel_gci` 0.03, `panel_order` 0.8, `empirical` 0.1.

Toolchain: pure NumPy/SciPy-free kernels, matplotlib (Agg) for the figures, pytest for the pins. All numbers below are injected from `results.json` by the report generator, not transcribed.

## 3. Results

### 3.1 Geometry gate

| Quantity | Mesh (64x32) | Closed form | Rel. error | Limit | Verdict |
|---|---:|---:|---:|---:|---|
| $C_b$ | 0.444309 | 0.444444 | +0.0305 % | 0.5 % | PASS |
| $\nabla$ [m^3] | 2.776930 | 2.777778 | +0.0305 % | 1 % | PASS |
| $A_{wp}$ [m^2] | 6.665039 | 6.666667 | +0.0244 % | 1 % | PASS |
| $S_w$ [m^2] | 14.878640 | 14.879063 | +0.0028 % | 2 % | PASS |

$KB$ from the mesh: 0.390611 m (closed form 0.390625 m); waterplane centroid $x = 0.0e+00$ m and $I_{xx}$, $I_{yy}$ (0.380735 vs 0.380952; 33.31977 vs 33.33333 m^4) agree to the mesh resolution.

### 3.2 Wave resistance: four routes

| $Fn$ | closed form $R_W$ [N] | $\lambda$-tail corr. [N] | $\theta$-form err. | brute 160x80 err. | panel 16x8 / 32x16 / 64x32 err. | GCI $p$ / GCI | legacy path ratio |

|---:|---:|---:|---:|---:|---|---|---:|
| 0.12 | 1.9341 | +1.49e-02 | +0.63 % | +1.49 % | +801.76 / +74.88 / +5.46 % | 3.39 / +8.69 % | 0.103 |
| 0.15 | 6.0928 | +7.74e-03 | +0.12 % | +0.84 % | +135.35 / +10.81 / +0.62 % | 3.61 / +1.13 % | 0.100 |
| 0.20 | 26.5478 | -1.11e-03 | +0.01 % | +0.31 % | +12.44 / +0.84 / +0.05 % | 3.87 / +0.07 % | 0.097 |
| 0.25 | 49.7256 | +4.91e-04 | +0.00 % | +0.15 % | +2.96 / +0.18 / +0.00 % | 3.97 / +0.02 % | 0.072 |
| 0.30 | 144.1403 | +1.44e-03 | +0.00 % | +0.06 % | +0.51 / +0.02 / +0.00 % | 4.25 / +0.00 % | 0.085 |
| 0.35 | 114.3185 | +3.69e-03 | +0.00 % | +0.05 % | +0.33 / +0.00 / +0.00 % | 5.57 / +0.00 % | 0.041 |
| 0.40 | 327.1142 | +8.36e-03 | +0.00 % | +0.02 % | +0.06 / +0.00 / +0.00 % | 5.00 / +0.00 % | 0.080 |
| 0.45 | 629.0874 | +1.73e-02 | +0.00 % | +0.01 % | +0.01 / +0.00 / +0.00 % | 3.83 / +0.00 % | 0.077 |

The BEM centreplane sheet converges monotonically on every Froude number; the residual at 64x32 is below 0.01 % above $Fn=0.20$. The brute 2-D quadrature is the weakest route at low Froude number (1.5 % at $Fn=0.15$, 12.8 % at $Fn=0.12$, outside the criterion band) because the $e^{kz}$ integrand is resolved by the same cell count while $k_0$ grows as $Fn^{-2}$; inside the band it stays below 1 %.

### 3.3 Total resistance and the empirical comparison (primary convention: c17 in, m4 term, $C_A = 0$)

| $Fn$ | $U$ [m/s] | $Re$ | $C_F$ (ITTC-57) | $1+k$ | $R_F(1+k)$ [N] | $R_W$ Michell [N] | $R_W$ Holtrop [N] | $R_T$ case [N] | $R_T$ Holtrop [N] | error | wave share |

|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.12 | 1.188 | 9.986e+06 | 0.00300 | 1.1505 | 37.18 | 1.934 | 0.16 | 39.11 | 37.34 | +4.75 % | +5 % |
| 0.15 | 1.485 | 1.248e+07 | 0.00289 | 1.1505 | 55.90 | 6.093 | 1.44 | 61.99 | 57.34 | +8.12 % | +10 % |
| 0.20 | 1.981 | 1.664e+07 | 0.00275 | 1.1505 | 94.68 | 26.548 | 13.75 | 121.23 | 108.43 | +11.80 % | +22 % |
| 0.25 | 2.476 | 2.080e+07 | 0.00265 | 1.1505 | 142.59 | 49.726 | 55.67 | 192.32 | 198.26 | -3.00 % | +26 % |
| 0.30 | 2.971 | 2.497e+07 | 0.00257 | 1.1505 | 199.35 | 144.140 | 134.70 | 343.49 | 334.05 | +2.83 % | +42 % |
| 0.35 | 3.466 | 2.913e+07 | 0.00251 | 1.1505 | 264.73 | 114.318 | 242.62 | 379.05 | 507.35 | -25.29 % | +30 % |
| 0.40 | 3.961 | 3.329e+07 | 0.00246 | 1.1505 | 338.55 | 327.114 | 569.65 | 665.66 | 908.20 | -26.71 % | +49 % |
| 0.45 | 4.456 | 3.745e+07 | 0.00241 | 1.1505 | 420.65 | 629.087 | 1091.43 | 1049.73 | 1512.07 | -30.58 % | +60 % |

Rows with $Fn>0.40$ are shown for the figure only and are outside the pre-declared criterion band.

### 3.4 Convention grid: does the verdict depend on the published variant?

| include_c17 | regime | $C_A$ convention | $C_A$ | pass / points | max abs error |
|---|---|---|---:|---:|---:|
| True | m4 | none | 0.000000 | 3 / 6 | +26.71 % |
| True | m4 | holtrop84 | 0.000778 | 4 / 6 | +24.22 % |
| True | m4 | ittc78 | 0.001950 | 4 / 6 | +21.25 % |
| True | cp2 | none | 0.000000 | 3 / 6 | +28.27 % |
| True | cp2 | holtrop84 | 0.000778 | 4 / 6 | +25.69 % |
| True | cp2 | ittc78 | 0.001950 | 4 / 6 | +22.59 % |
| False | m4 | none | 0.000000 | 4 / 6 | +21.47 % |
| False | m4 | holtrop84 | 0.000778 | 4 / 6 | +18.12 % |
| False | m4 | ittc78 | 0.001950 | 4 / 6 | +14.67 % |
| False | cp2 | none | 0.000000 | 4 / 6 | +23.65 % |
| False | cp2 | holtrop84 | 0.000778 | 4 / 6 | +19.90 % |
| False | cp2 | ittc78 | 0.001950 | 4 / 6 | +16.06 % |

No convention reaches a clean pass: the pass count spans 3-4 of 6 and the worst deviation spans 14.7-28.3 %. The sign of the disagreement even flips with `include_c17`, which is the quantitative statement that this baseline is **not** a 10 % yardstick for this hull.

## 4. Figures

![resistance_vs_froude.png](../artifacts/bem_wigley_calm_water_resistance/figures/resistance_vs_froude.png)

*Left: Wigley III calm-water resistance. The case kernel (Michell thin-ship wave resistance on top of the ITTC-1957 friction line with Holtrop's form factor) against the Holtrop (1984) regression, which shares the identical friction protocol, so the two curves differ only through the wave term. Right: the wave term of both routes and the resulting total-resistance error with the pre-declared +-10 % band (dotted). Shaded: the criterion band 0.15 <= Fn <= 0.40.*

![wave_resistance_routes.png](../artifacts/bem_wigley_calm_water_resistance/figures/wave_resistance_routes.png)

*Left: convergence of the BEM centreplane source sheet (16x8, 32x16, 64x32 panels) to the closed-form Michell value. Right: wave-resistance coefficient from the converged Michell route, the Holtrop-Mennen regression and the legacy tensorbem.wigley path, whose lambda^-2 weight and missing g^2 prefactor make it ~12x low.*

## 5. Pitfalls, limits and scope

1. **The registered case `bem_wigley_wave_resistance` carries numbers from a defective path.** `tensorbem.wigley.michell_wave_resistance` uses a $\lambda^{-2}$ weight and no $g^2$ prefactor; at $Fn=0.30$ it returns 12.29 N where the standard form gives 144.14 N (ratio 0.085, i.e. ~12x low), which reproduces the $C_W$ of that case. The defect is pinned, not patched: changing it moves pinned numbers in `tests/test_wigley_verification.py`, `docs/wigley_wave_resistance.md` and the declared-number ledger, which is a separate, deliberate change.

2. **Holtrop is outside its fitted shape range here.** $iE = 6.33$ deg (inside-fit threshold 5 deg: yes), $\nabla/L^3 = 2.78e-03$, no bulb, no immersed transom. The regression was fitted to real, fuller hulls, so its wave term is an extrapolation; the disagreement at $Fn\ge0.35$ is therefore a property of the baseline, not of the kernel.

3. **Coefficient conventions are not cosmetic.** $c_{17} = 1.6146$ for this hull, so including or omitting it changes $R_W$ by that factor and flips the sign of the total-resistance error; the `m4` vs `cp2` exponential term and the three $C_A$ conventions add another few percent each. The primary convention is declared in the JSON and in §3.3.

4. **Form-factor transcription open item.** The 1984 expression implemented here gives $1+k = 1.1656$ for the KCS hullform where the ship-performance-platform registration quotes 1.1362 (2.6 %). It does not affect this case's comparison (the factor is common to both routes) but it does move the absolute $R_T$ by up to 2 %.

5. **Thin-ship theory.** Michell has no viscosity, no wave breaking, no sinkage/trim and no finite thickness; it is used here for $B/L = 0.1$, $Fn \le 0.45$, deep water. A thick-hull Rankine solver with forward speed is *not* implemented in TensorBEM (`NOT_SUPPORTED`), so this case is not a CFD validation.

6. **No experiment.** The absolute total resistance against tank data is `WITHHELD`: no digitised Wigley resistance data ships with the repository (see `docs/bem_verified_anchors.md`, item G10).

## 6. Error assessment: does each key quantity meet < 3 %?

| Quantity | Reference | Worst measured error | Threshold | < 3 %? | Verdict |
|---|---|---:|---:|---|---|
| $C_b$, $\nabla$, $A_{wp}$, $S_w$ (mesh) | closed forms | +0.0305 % | 0.5-2 % | **yes** | PASS |
| $R_W$: $\theta$-form / brute 2-D / BEM sheet | closed-form Michell | +0.84 % | 1-2 % | **yes** | PASS |
| panel-route GCI (64x32 vs 32x16) | Roache, $p\ge0.8$ | +1.129 % | 3 % | **yes** | PASS |
| $C_F$ inside the assembly | ITTC-1957 closed form | 0.0e+00 % | 1e-12 identity | **yes** | PASS |
| $R_T$ vs Holtrop (primary convention) | Holtrop 1984 | -26.71 % at Fn=0.40 | 10 % | **no** | **FAIL** (3/6) |

Explicit answers: the geometry, the wave-resistance route agreement, the panel convergence and the friction identity are all **below 3 %** (the worst is +1.13 %). The total resistance against the empirical baseline is **not** below 3 %, and not below 10 % either: at Fn=0.40 it is -26.71 %, and under the alternative conventions the worst point stays between 14.7 % and 28.3 %. That failure is reported as such; it says that Holtrop's regression is not a reference for a Wigley form, not that the wave-resistance kernel is wrong — the kernel is verified against three independent routes.

## 7. Reproduction

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.wigley_calm_water_resistance \
    --output artifacts/bem_wigley_calm_water_resistance
PYTHONPATH=src python -m pytest tests/test_wigley_calm_water_resistance.py -q
```

Runtime of the driver: a few seconds on one core (no cluster job needed). Capability row: `tensorbem.resistance` is listed in `AUXILIARY_MODULES` of `tensorbem.capability_evidence` with its evidence level (REAL formula implementation; the case is WITHHELD against experiment).

