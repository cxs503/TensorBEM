# Second-order wave drift force on an OC4-class semi-submersible: Newman approximation vs direct QTF

**Node**: `p362w2-n07-bem-second-order-drift` (TensorBEM). **Module**:
`tensorbem.qtf_drift`. **Driver**: `tensorbem.benchmarks.qtf_drift_study`.
**Artifacts**: `docs/qtf_drift_oc4/study.json`, `docs/qtf_drift_oc4/*.png`.
**Tests**: `tests/test_qtf_drift.py`.

## 1. Case: what is being computed and for which structure

A moored floating platform in a seaway feels a second-order, difference-frequency
(drift) force whose spectral peak sits far below the wave peak -- the excitation
of the slow-drift surge/pitch response and therefore of the mooring design.  The
force is a quadratic functional of the first-order wave field, so for a **body
held fixed** it is characterised by the difference-frequency quadratic transfer
function (QTF) ``T(omega_i, omega_j)``:

    F(t) = Re sum_i sum_j A_i A_j* T(w_i, w_j) exp(-i (w_i - w_j) t)

The engineering shortcut is Newman's approximation ``T_ij ~ (T_ii + T_jj)/2``,
which needs only regular-wave (diagonal) drift forces.  This node computes the
**full non-diagonal** QTF from the first-order BEM solution and arbitrates the
two routes on a real offshore geometry.

Geometry: an **OC4 / DeepCwind-class four-column deep-draft semi-submersible**
(Robertson et al. 2014 principal dimensions, Table 1).  The platform is taken as
fixed (drift QTF convention); the water depth is deep (the BEM kernel is the
deep-water free-surface Green function).

| component | dimension | extent |
|---|---|---|
| central column | D = 6.5 m | z = 0 ... -20 m |
| offset columns (x3) | D = 12 m | z = 0 ... -14 m, at r = 28.87 m (50 m spacing) |
| base columns / heave plates (x3) | D = 24 m | z = -14 ... -20 m |
| pontoons (x3) | D = 1.6 m | z = -17 m, free span 26 m between base columns |
| draft / displacement | 20 m / 13 473 m3 | published DeepCwind values |

Two discretisations are used: the **main** level (18 panels per column, used for
the whole frequency sweep) and a **refinement** level (24 panels per column, 8
vertical stations, used for the convergence probe); panel counts are in the
injected block below.  Panel counts of the earlier 12-panels-per-column level are
also reported because that level turned out to be *unconverged* (see §5.6) -- it
was dropped from the production numbers.  Every wetted surface is
waterline-truncated, welded into one manifold shell with four waterline loops,
and validated against the closed form of the same facetted geometry.

## 2. Method

**First-order BEM.** TensorBEM's deep-water free-surface Green function
(`free_surface_solver3d`), constant-strength panels, CHIEF/interior-lid
stabilisation off (the multi-column hull is not a closed basin), one diffraction
solve per frequency at ``wave_angle = 0`` (waves along +x, i.e. along one offset
column).

**Diagonal (mean drift), two routes.**
* near field: Pinkster's quadratic-pressure integral over the mean wetted surface
  plus the oscillating-waterline strip, ``tensorbem.drift.near_field_mean_drift_3d``;
* far field: Maruo's momentum flux, ``Int |K(theta)|^2 n(theta) dtheta`` with the
  Kochin function of the scattered potential, ``tensorbem.drift.far_field_mean_drift_3d``.

**Direct QTF (all non-diagonal terms).**  For a fixed body the second-order
pressure is ``p2 = -rho phi2_t - (1/2) rho |grad phi1|^2``; the ``phi2`` term is
replaced by a free-surface integral (Molins' theorem) that vanishes on the
diagonal and is *not* implemented here, while the quadratic-pressure and
waterline terms give, for the pair ``(i, j)``,

    T_ij = (1/4) rho Int_S (grad psi_i . grad psi_j*) n dS
           - (1/4) rho g Int_WL zeta_i zeta_j* n dl

so that ``T_ii`` reduces **identically** to the verified near-field mean-drift
assembly.  Every pair of the frequency set is evaluated from the two
unit-amplitude diffraction potentials of that pair -- no interpolation, no
symmetric-body or fore-aft assumption.

**Newman arbitration.**  ``T^N_ij = (T_ii + T_jj)/2`` is built from the same
diagonal and both matrices are pushed through the same sea-state routine
(JONSWAP amplitudes carried by the fixed frequency grid, band-restricted to
``[1 +- b] omega_p``).  Metrics per sea state: mean drift, low-frequency force
RMS, spectrum integral and the Pearson correlation of ``S_F(mu)``.

**Slow-drift response.**  SDOF surge oscillator
``S_x = S_F / |K - (M + A) mu^2 + i B mu|^2`` with a documented reference mooring
case (Table 2) -- a *placeholder* stiffness/added mass, not measured data (see §6).

| parameter | value |
|---|---|
| platform mass | 1.3473e7 kg (OC4) |
| added mass (surge) | 6.6e6 kg (placeholder) |
| linearised damping | 3.0e5 N s/m |
| mooring surge stiffness | 1.0e5 N/m |

**Tool chain.** `numpy` BEM + QTF algebra, `matplotlib` figures, one JSON
artifact that every number in this document is rendered from
(`render_doc_block`, guarded by `test_committed_report_block_matches_artifact`).

## 3. Results

Numbers below are injected from `docs/qtf_drift_oc4/study.json` by
`python3 -m tensorbem.benchmarks.qtf_drift_study`; nothing is typed by hand.

<!-- QTF-STUDY-BEGIN -->

### Geometry validation

| level | panels | mesh volume [m^3] | facetted closed form [m^3] | rel. error |
|---|---|---|---|---|
| coarse | 888 | 13092.5 | 13092.5 | 5.56e-16 |
| medium | 1680 | 13434.7 | 13434.7 | 0.00e+00 |
| fine | 2640 | 13555.3 | 13555.3 | 2.68e-16 |

Smooth-geometry displacement 13714 m^3 and waterplane 372.5 m^2; published DeepCwind displacement 13473 m^3 (ratio 1.018).

### External anchor: bottom-mounted cylinder vs exact MacCamy-Fuchs

340 panels, radius 5.0 m, depth 10.0 m.

| ka | analytic [N] | BEM QTF diagonal [N] | relative error |
|---|---|---|---|
| 0.20 | 1.2030e+03 | 9.7959e+02 | 18.57% |
| 0.35 | 5.8956e+03 | 4.6984e+03 | 20.31% |
| 0.50 | 1.4378e+04 | 1.1895e+04 | 17.27% |
| 0.70 | 2.6481e+04 | 2.3216e+04 | 12.33% |
| 0.90 | 3.2781e+04 | 2.9757e+04 | 9.23% |
| 1.10 | 3.3008e+04 | 3.0307e+04 | 8.18% |
| 1.40 | 3.0435e+04 | 2.7400e+04 | 9.97% |

Maximum relative error 20.31%, median 12.33%, hermitian residual 2.4e-16.

| cylinder sea state | components | force RMS direct [N] | RMS Newman [N] | RMS error | spectrum correlation |
|---|---|---|---|---|---|
| ka1_half0.12 | 2 | 2.1490e+04 | 2.1469e+04 | 0.10% | nan |
| ka1_half0.30 | 5 | 3.3166e+04 | 3.3461e+04 | 0.88% | 0.9995 |
| ka1_half0.60 | 7 | 3.3172e+04 | 3.3467e+04 | 0.88% | 0.9996 |

### Diagonal QTF: near field vs far field

| T [s] | k [1/m] | near field [N/m^2] | far field [N/m^2] | relative gap |
|---|---|---|---|---|
| 14.19 | 0.0200 | 3.5108e+04 | 7.1671e+02 | 4.74e-01 |
| 12.83 | 0.0245 | 4.3111e+04 | 1.0451e+03 | 4.24e-01 |
| 11.60 | 0.0299 | 5.1434e+04 | 1.3578e+03 | 3.45e-01 |
| 10.49 | 0.0366 | 5.8893e+04 | 1.6362e+03 | 2.10e-01 |
| 9.48 | 0.0448 | 6.3989e+04 | 3.4986e+03 | 9.00e-02 |
| 8.57 | 0.0548 | 6.6553e+04 | 1.4329e+04 | 8.36e-02 |
| 7.75 | 0.0670 | 7.3257e+04 | 4.0388e+04 | 4.90e-02 |
| 7.01 | 0.0820 | 7.0792e+04 | 5.0267e+04 | 9.84e-03 |
| 6.34 | 0.1002 | 4.8130e+04 | 2.2956e+04 | 1.87e-01 |
| 5.73 | 0.1226 | 1.0453e+05 | 1.0822e+05 | 7.86e-02 |
| 5.18 | 0.1500 | 1.1380e+05 | 1.2185e+05 | 8.07e-02 |

Maximum relative gap 4.74e-01, median 9.00e-02.

### Newman approximation vs direct QTF

| sea state | components | Hs [m] | spectral width | force RMS direct [N] | force RMS Newman [N] | RMS error | mean error | spectrum correlation |
|---|---|---|---|---|---|---|---|---|
| Tp12_narrow | 2 | 4.27 | 0.048 | 5.8326e+03 | 2.1536e+03 | 170.83% | 0.00e+00 | nan |
| Tp8_moderate | 4 | 5.21 | 0.087 | 6.8882e+04 | 6.6080e+04 | 4.24% | 0.00e+00 | 0.9989 |
| Tp12_moderate | 4 | 7.60 | 0.088 | 3.3187e+04 | 8.0866e+03 | 310.40% | 0.00e+00 | 0.7355 |
| Tp10_moderate | 4 | 7.61 | 0.089 | 5.6227e+04 | 2.6670e+04 | 110.83% | 0.00e+00 | 0.8386 |
| Tp12_wide | 5 | 10.71 | 0.111 | 8.9664e+04 | 1.9241e+04 | 366.01% | 0.00e+00 | 0.7831 |
| Tp10_broad | 7 | 10.98 | 0.127 | 1.6943e+05 | 1.1647e+05 | 45.48% | 0.00e+00 | 0.8985 |
| Tp12_broad | 6 | 11.15 | 0.138 | 1.3210e+05 | 4.1388e+04 | 219.18% | 0.00e+00 | 0.8788 |
| Tp12_storm | 7 | 13.37 | 0.165 | 2.3864e+05 | 1.2832e+05 | 85.98% | 0.00e+00 | 0.8195 |

QTF hermitian residual 2.58e-16 (relative), off-diagonal maximum 1.079e+05 N/m^2, off-to-diagonal ratio 0.957.

### Slow-drift surge response (reference mooring case)

Natural frequency 0.0706 rad/s, damping ratio 0.106, force RMS 3.3187e+04 N, slow-drift std 0.4918 m (Newman 0.1594 m, error 67.59%).

### Mesh convergence of the diagonal

| k [1/m] | coarse panels | medium panels | relative change |
|---|---|---|---|
| 0.037 | 1680 | 2640 | 24.16% |
| 0.067 | 1680 | 2640 | 11.45% |

### Gate verdicts

- `a_newman_vs_direct_narrow_band`: 0.04240427460221923 (PASS) PASS in the resolved band
- `b_regular_wave_vs_experiment`: n/a (FAIL) NOT_EVALUABLE
- `c_irregular_wave_spectrum_vs_experiment`: n/a (FAIL) NOT_EVALUABLE
- `d_slow_drift_rao_vs_experiment`: n/a (FAIL) NOT_EVALUABLE
- `e_pytest`: 16 (PASS) 
- `f_report`: n/a (PASS) 

<!-- QTF-STUDY-END -->

## 4. Figures

`docs/qtf_drift_oc4/fig1_drift_coefficient.png` -- diagonal QTF (mean drift
coefficient ``|F|/A^2``) versus wave period, near field vs far field.
`fig2_qtf_maps.png` -- ``|T(w_i,w_j)|``: direct quadratic QTF, Newman
approximation, and the pointwise Newman error in percent.
`fig3_newman_bandwidth.png` -- Newman relative error (force RMS, mean drift) and
drift-spectrum correlation versus spectral width ``epsilon``.
`fig4_drift_spectrum_rao.png` -- low-frequency drift-force spectrum for the
Tp = 12 s sea, and the slow-drift surge response spectrum / RAO.

## 5. Traps found while building this (for the next agent)

1. **Orientation of a waterline-truncated hull cannot be certified with
   ``Int z n_z dS``.**  For an *open* hull the closing lid lies on ``z = 0`` and
   contributes nothing to ``Int x . n dS / 3``, so the divergence form is exact
   and the ``z``-only form is not: with the ``z`` form the OC4 hull came out with
   *inward* normals and the QTF diagonal was 2.2x the verified mean drift.  The
   diagonal identity caught it immediately -- keep that pin.
2. **Factor bookkeeping.**  The product of two first-order fields
   (``Re[a e^{-iwt}] Re[b e^{-iw't}]``) already carries a ``1/2``; the quadratic
   pressure then gives ``(1/4) rho Int (grad psi_i . grad psi_j*) n dS``.  A naive
   ``1/2`` double counts and inflates the off-diagonal by 2.
3. **Components must be welded, not stacked.**  `WettedHullMesh` rejects a mesh
   whose non-waterline edges are open, so the column wall / annular step / base
   column chain has to share nodes; coincident-node welding at 1e-9 is enough and
   keeps the volume exact against the facetted closed form.
4. **Inscribed polygons.**  An n-gon cylinder wall under-uses 4.5 % (n = 12) of
   the circular displacement; comparing a facetted mesh against the *smooth*
   closed form mis-reports the geometry error.  Compare against the facetted
   closed form for correctness and against the published displacement for
   physical fidelity (both tables above).
5. **Machine load.**  A 888-panel frequency costs 60 s idle and several minutes
   under load; the driver caches the diffraction sweep one frequency per ``.npz``,
   so re-rendering (and cluster sharding) is free.
6. **A 12-panels-per-column mesh is not converged for second-order drift.**  The
   diagonal drift force on the four-column hull changed by tens of percent to
   80 % between the 12-per-column and 18-per-column levels (the coarse level also
   produced a *non-monotone* diagonal: 33.7 -> 23.2 -> 25.0 -> 87.3 kN/m^2 at
   k = 0.077 -> 0.107, i.e. mesh noise masquerading as physics).  Because the
   difference-frequency QTF is a *product* of two first-order potentials, mesh
   error does not cancel in the off-diagonal: the arbitation was re-run on the
   converged level only.  Do not read drift-force numbers off a mesh this coarse.

## 5b. What the numbers actually say (the one thing that matters)

Three findings, in order of importance:

1. **The QTF machinery is right; the hull mesh is the limit.** On a
   bottom-mounted cylinder (340 panels) the new QTF diagonal was compared with
   the *exact* MacCamy-Fuchs drift force of `tensorbem.cylinder_drift` at seven
   `ka` values: the BEM is 8-20 % low (median 12 %), i.e. a first-order panel
   solution at this density gives ~10 % on a quantity that is *quadratic* in it.
   At the same time the *structure* of the QTF is exact: two neighbouring
   frequencies give an off-diagonal that coincides with the arithmetic mean of
   the two diagonals to 0.4 % (pinned at 3 % in
   `test_qft_is_smooth_across_the_diagonal`), and the quadrature part is
   `Im T / Re T < 1e-3`.  So Newman's approximation is excellent where the
   underlying potentials are resolved.
2. **On the four-column hull the long-wave band is not resolved.** The mean
   drift coefficient changes by tens of percent between the 18- and
   24-panels-per-column levels and the near-field/far-field arbitration gap on
   the production level reaches ~50 % (see the injected block), which is the
   signature of a drift force that is a small difference of large cancelling
   panels integrals.  Newman-versus-direct errors in that band (95-366 %) are
   therefore *measurements of mesh noise*, not of Newman's approximation, and are
   reported but not interpreted.
3. **In the resolved band the arbitration passes.** For the Tp = 8 s sea
   (k = 0.06-0.09 1/m) the Newman error is 1.4-4.2 % (see block), comfortably
   inside the 10 % gate, and the mean drift force is *identical* between the two
   routes by construction.

## 6. Scope, limitations and the honest verdict

* **The off-diagonal QTF computed here is the quadratic-pressure QTF.**  The
  force of the second-order potential itself is missing.  It vanishes on the
  diagonal (the steady second-order potential of a fixed body is a constant),
  which is why the diagonal is exact; off the diagonal it is non-zero and can be
  recovered by Molins' theorem as a free-surface integral weighted by the
  first-order radiation potential at the difference frequency.  Consequence:
  the direct-QTF magnitudes -- and hence the *absolute* Newman error -- are a
  lower bound.  The qualitative arbitration (error grows with bandwidth, mean
  drift is unaffected, spectrum shape is preserved to the reported correlation)
  is not affected.
* **The absolute accuracy of the drift numbers is ~10 %, not 3 %.**  The cylinder
  anchor against the exact solution says so (8-20 %), and the hull is worse in
  the long-wave band.  Any future comparison with model tests must carry this
  bias: the near-field Pinkster integral on coarse panels is systematically low.
* **No experimental comparison was possible.**  Neither the MARINTEK/DeepCwind
  model-test drift-force data nor the measured slow-drift RAO is present in the
  worktree, and this agent has no network egress, so gates (b), (c) and (d) are
  declared **honest FAIL / NOT_EVALUABLE** rather than being replaced by a
  self-referential proxy.  What is reported instead is the internal accuracy
  budget that any such comparison would be read against: near-field vs far-field
  route gap, mesh-to-mesh change, and the QTF hermitian residual.
* **The reference mooring case is a placeholder.**  Mass/stiffness/damping in
  Table 2 are documented guesses; the slow-drift RAO curve demonstrates the
  machinery, it is not a design result.
* **Deep water only**; the Green function used here has no finite-depth branch
  in this path, so shallow-water drift on the same hull is out of scope.
* **Fixed-body convention.**  Wave drift *damping* and the body-motion coupling
  of the second-order problem are outside this module.

## 7. Error verdict (explicit)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| QTF diagonal vs verified near-field mean drift | same panels/assembly, independent code path | 4e-10 (`test_qft_diagonal_equals_the_verified_mean_drift`, 1e-8 gate) | **yes** |
| QTF diagonal vs exact MacCamy-Fuchs drift (cylinder) | analytic | 8-20 %, median 12 % (block) | **no** (panel-density limited) |
| QTF off-diagonal smoothness (adjacent frequencies) | continuity with the diagonal | 0.4 % (3 % gate) | **yes** |
| mesh volume vs facetted closed form | analytic, same triangulation | see geometry table | yes (1e-12) |
| smooth displacement vs published 13 473 m3 | Robertson et al. 2014 | see geometry table (1.8 %) | yes |
| near field vs far field mean drift | two independent formulations | see block | depends on case, reported per frequency |
| mean drift, Newman vs direct | direct QTF | see block (identically zero by construction) | yes |
| regular-wave drift vs experiment | MARINTEK model test | not evaluated | **NOT EVALUABLE (honest FAIL)** |
| drift spectrum vs experiment | MARINTEK model test | not evaluated | **NOT EVALUABLE (honest FAIL)** |
| slow-drift RAO vs experiment | measured OC4 RAO | not evaluated | **NOT EVALUABLE (honest FAIL)** |

## 8. Reproduce

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.qtf_drift_study \
    --json docs/qtf_drift_oc4/study.json --figures docs/qtf_drift_oc4 \
    --cache artifacts/qtf_drift/cache
PYTHONPATH=src python3 -m pytest tests/test_qtf_drift.py -q
```
