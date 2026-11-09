# KP505 official open-water data: transcription anchor (research_plan item 1)

*Mission*: `paper-exp-propeller-openwater-bem` / experiment 1 (official open-water data transcription).
*Artefact*: `benchmarks/anchors/kp505_openwater.json` — **machine-generated**, never hand-edited.
*Builder*: `tools/build_kp505_openwater_anchor.py`. *Contract tests*: `tests/test_kp505_openwater_anchor.py` (10 tests).

## 1. What this is (and what it is not)

The parent study asks how large the BEM error is on the **KP505** propeller in open water, and how
panel density and the wake model move that error. Before any BEM number can be quoted, the
reference curve has to be pinned down with auditable provenance. This artefact is that pin.

**Honest provenance statement (do not skip).** The primary handout (KRISO/KCS-workshop EFD table,
Hino 2005 proceedings) was **not reachable** from this working environment: the sandbox egress
allowlist rejects `nmri.go.jp` (DNS), `iopscience.iop.org` (bot wall), `mdpi.com` (403),
`onepetro`/`tandfonline`/`wiley` (403), `web.archive.org` (DNS), and the 50 MB SMP'15
*Second Workshop on Cavitation and Propeller Performance* proceedings download stalled at 14/50 MB
(transfer ~40 kB/s). The curve transcribed here is therefore a **SECONDARY_REPRODUCTION**: the
"Experimental" columns of **Table 2, page 10** of

> H. Iftikhar, A. C. Takinaci, *Performance Prediction of KP505 Propeller Using Lifting Line Theory
> and RANS*, MARINE 2023 (X Int. Conf. on Computational Methods in Marine Engineering), CIMNE,
> DOI 10.23967/marine.2023.120 — PDF committed at
> `benchmarks/anchors/sources/kp505_marine2023_iftikhar.pdf`
> (sha256 `161ef626…a5bc7c`).

That paper does **not** cite where its experimental column comes from, so the anchor is labelled
secondary and **must be re-verified against the KRISO table before it is used for an accuracy
claim** in the parent study. This is a documented gap, not a silent substitution.

## 2. How it was done

1. The PDF is committed; the builder extracts the page containing
   `Table 2. Comparison of CFD and Experimental Results` with `pypdf`, writes the page text
   verbatim to `benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt`, and parses the ten
   data rows from that transcript. Every row in the JSON keeps its own verbatim source line.
2. Efficiency is **computed** in the builder from the tabulated coefficients,
   `eta0 = J·KT/(2π·KQ)` with `KQ = 10KQ/10`, i.e. `eta0 = 5·J·KT/(π·10KQ)`; the JSON stores both
   `eta0` and `eta0_recomputed` plus the residual (0.0 to machine precision, well inside 1e-6).
3. The design advance ratio is `J = 0.70`; the peak of the efficiency curve is located by search
   over the transcribed points, not asserted by hand.
4. Geometry: the official KCS workshop geometry archive
   (`http://www.t2015.nmri.go.jp/file/Geometry_IGES_files/kcs/KP505.zip`, downloaded through the
   reachable workshop page `kcs_gc.html`, quote: *"Following archive contains IGES&STP geometories of
   the KCS propeller."*) is committed (model-scale IGES member). Its member name
   `KP505_modelScale_250mm.igs` states **D = 250 mm**, and Table 1 of the source paper
   (`No. of Blades 5`, `Diameter (m) 0.25`) is transcribed to
   `benchmarks/anchors/sources/kp505_marine2023_table1_propeller_characteristics.txt`.

## 3. Results (all numbers injected by the builder)

| J | KT | 10KQ | eta0 = 5·J·KT/(π·10KQ) |
|---|----|------|------------------------|
| 0.1 | 0.494 | 0.683 | 0.1151 |
| 0.2 | 0.447 | 0.636 | 0.2237 |
| 0.3 | 0.392 | 0.585 | 0.3199 |
| 0.4 | 0.343 | 0.530 | 0.4120 |
| 0.5 | 0.290 | 0.471 | 0.4899 |
| 0.6 | 0.241 | 0.409 | 0.5627 |
| **0.7 (design)** | **0.193** | **0.342** | **0.6287** |
| 0.8 | 0.141 | 0.270 | **0.6649 (peak)** |
| 0.9 | 0.087 | 0.192 | 0.6493 |
| 1.0 | 0.029 | 0.108 | 0.4274 |

Self-consistency: max |eta0 − recomputed| = 0 (float-exact) < 1e-6. Peak efficiency at
**J = 0.80** with eta0 = 0.6649; at the design point J = 0.70, eta0 = 0.6287.

Cross-check of the source itself: its published error columns reproduce from its own CFD and
"experimental" columns within rounding (see `source_consistency_checks` in the JSON), i.e. the
transcription is internally faithful; this says nothing about whether the "experimental" column is
the KRISO table.

Geometry: D = 0.25 m (VERIFIED: archive member name + quote), Z = 5 (VERIFIED_SECONDARY:
Table 1 quote); **P/D and AE/A0 are WITHHELD** — the only reachable statement is Table 1
(`P/D (mean) 0.950`, `Ae/Ao 0.800`), which is not corroborated by any official document reachable
here and conflicts with the geometry package, so it is recorded but not adopted.

## 4. Figures

See the mission report (`figures` field): eta0(J) with the peak marked, and KT(J)/10KQ(J) with the
design point marked. The curve is drawn from the same JSON, so the figure cannot drift from the
artefact.

## 5. Pitfalls / limitations

* Secondary provenance (see §1) — the single largest limitation of this deliverable.
* The published table gives KT and 10KQ to 3 decimals only; eta0 inherits ~1e-3 relative resolution.
  This is why the acceptance rule is applied to the *stored* eta0 versus its recomputation from the
  *stored* KT/10KQ (float-exact) and not to a hand-quoted eta0 column.
* The source's own geometry table disagrees with the official KP505 particulars (Ae/Ao 0.800 vs the
  commonly quoted KP505 value); that mismatch is exactly why geometry claims are held to the
  official archive for D and to a quoted source for Z, and withheld elsewhere.
* A direct measurement of D and Z from the committed IGES was attempted and **failed honestly**: the
  NURBS control cage gives a tip diameter of 440 mm against the documented 250 mm and resolves only
  2 near-tip clusters. The attempt is recorded in the artefact under
  `geometry.measurement_attempt.status = UNRELIABLE` rather than being dressed up as a result.
* Scope: this deliverable writes no paper text and touches no platform code; it only pins the
  reference data for the BEM study that follows.

## 6. Acceptance verdict (criteria fixed before the work)

| id | criterion | verdict |
|----|-----------|---------|
| C1 | every point carries a verbatim quote + table number | **PASS** (Table 2, page 10; 10/10 rows with source line) |
| C2 | >= 8 J points, design J included | **PASS** (10 points, J = 0.1…1.0, J = 0.70 present) |
| C3 | eta0 peak J recorded and self-consistent, residual < 1e-6 | **PASS** (peak J = 0.80, residual 0) |
| C4 | geometry: D, P/D, AE/A0, blade count | **PARTIAL / honest FAIL** — D and Z pinned by quotation (D also by the official archive), P/D and AE/A0 WITHHELD for lack of a verified source |

So: the transcription criteria are met; the geometry criterion is only half met and the *official*
(primary) provenance of the open-water curve is **not** established. Error versus a KRISO anchor
therefore cannot yet be stated, and this deliverable does not state one.