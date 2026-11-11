# 3-D finite-depth kernel: off-axis deep-water limit — eigenseries vs quadrature split

**Mission** m6304 (self-start, generalist seat) · **branch** `agent/m6304` · **repo** TensorBEM
**EDIT_WHITELIST**: `src/tensorbem/`, `tests/`, `docs/`, `outputs/`

**Verdict**: the open item in `docs/finite_depth3d_offaxis_diagnosis.md` (m2930,
strict-xfail `test_deep_water_limit_matches_free_surface3d`) is **implementation-split**,
not a single kernel defect:

| implementation | off-axis (R>0) | on-axis (R=0) |
|---|---|---|
| eigenseries `finite_depth3d.green_function_3d` | **converges** to deep water as h→∞ (err → 0) | singular (`K_0(μ_n R)` diverges) — representation limit |
| quadrature `finite_depth3d_quadrature.green_function_finite_depth_3d` | **fails** (34 % / 320 %, systematic, h-independent) | converges (err → 0) |

The m2930 diagnosis only exercised the quadrature form and concluded "formula correct,
quadrature converged, residual systematic".  The missing half — that the *independent*
eigenseries reproduces the deep-water kernel off-axis to roundoff — is what localises the
defect to the quadrature's off-axis PV handling and gives the next fixer a clean reference.

## 1. Case

Source `(0,0,-1.0)`, `ν = ω²/g = 0.6`, deep-water reference
`free_surface3d.green_function_3d` (Kelvin kernel).  Off-axis field points
`(0.5,0,-0.5)` (R=0.5) and `(2.0,1.0,-1.5)` (R=2.24).  Depth ladder
`h = 4…128 m` (k0·h from ≈2.4 to ≈48).

## 2. Method

Both shipped implementations evaluated at identical targets; relative error
`|G−G_dw|/|G_dw|` per point, per depth.  All numbers rendered from
`outputs/m6304/offaxis_split.json` (no transcription); figure
`docs/figures/fig_finite_depth3d_offaxis_split.png`.

## 3. Results (from the ledger)

| h [m] | eigenseries err (R=0.5 / R=2.24) | quadrature err (R=0.5 / R=2.24) |
|---:|---|---|
| 4 | 2.2e-3 / 1.6e-3 | 0.347 / 3.20 |
| 8 | 2.2e-3 / 1.6e-3 | 0.347 / 3.20 |
| 16 | 1.0e-4 / 1.0e-4 | 0.345 / 3.20 |
| 32 | 1.0e-4 / 0.0 | 0.345 / 3.20 |
| 64 | 0.0 / 0.0 | 0.345 / 3.20 |
| 128 | 0.0 / 0.0 | 0.345 / 3.20 |

The quadrature error is **constant in h** (the deep-water limit is already
reached by h=8) and **grows with R** — the signature of a wavenumber-integral
error in the oscillatory `J_0(kR)` tail, not of a depth-asymptotics error.
The eigenseries error decays with h and is at roundoff by h=64.

## 4. Error judgement (<3%?)

| quantity | baseline | measured | <3%? |
|---|---|---|---|
| eigenseries off-axis, h=128 | deep-water kernel | 0.0 / 0.0 | **yes** |
| eigenseries off-axis, h=8 | deep-water kernel | 2.2e-3 / 1.6e-3 | **yes** |
| quadrature off-axis, h=128 | deep-water kernel | 0.345 / 3.20 | **no** (open item, unchanged) |

## 5. Notes / scope

* New gate `tests/test_finite_depth3d_offaxis_split.py` (2 tests, both pass):
  pins the eigenseries off-axis convergence (monotone, <1e-3 at h=128) and
  pins the *presence* of the quadrature defect (order of magnitude) so a
  future fix is detected as the error collapsing.
* The strict xfail in `tests/test_finite_depth3d_quadrature.py` is left in
  place: the quadrature defect is real and unfixed; this document does not
  claim closure of it.
* On-axis (R=0) eigenseries divergence is a known eigenseries representation
  limit (log singularity of `K_0`), out of scope here.
* Compute: 12 kernel evaluations × 2 implementations, seconds of single-core
  work; run under load15 ≈ 124 (lightweight-only discipline respected).


---

## Corrigendum (m13811, 2026-11-11): the quadrature "defect" was an input-projection bug

The verdict above localises the error to "the quadrature's off-axis PV handling". That
attribution is **wrong**. Root cause: `_green` clamped the whole point array
(`np.minimum(tgt, 0.0)` / `np.minimum(src, 0.0)`), which sets `x = y = 0`, so every pair was
evaluated at `R = 0` and the off-axis value collapsed onto the on-axis value (which is why the
error was flat in `h` and grew with `R`). Restricting the clamp to the vertical component fixes
it: quadrature now agrees with the eigenseries to `<= 1.94e-10` at every off-axis point and
reaches the deep-water kernel with error `1.8e-7` at `h = 128`. The strict xfail in
`tests/test_finite_depth3d_quadrature.py` is therefore replaced by a real gate; the "defect
present" pin in `tests/test_finite_depth3d_offaxis_split.py` became a "defect closed" pin.
Details and numbers: `docs/finite_depth3d_quadrature_offaxis_fix.md`.
