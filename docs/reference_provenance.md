# Reference-constant provenance: which numbers a driver is allowed to trust

TensorBEM's verification claims are error percentages: *this* added mass against
WAMIT, *this* RAO peak against Journee (1992), *this* displacement against the
published OC4 value.  A percentage is only worth what its denominator is worth,
and the repository's rule is that no number is transcribed by hand into a claim.

The capability registry (`docs/audit_baseline.md`) grades code, tests and
documents; it does not look at the constants those drivers compare against.
This document is the missing half.  `tensorbem.reference_provenance` scans the
package for module-level constants whose **name** announces a literature origin
(`WAMIT`, `JOURNEE`, `EFD`, `EXPERIMENT`, `LITERATURE`, `PUBLISHED`, `MEASURED`,
`OBSERVED`, `REFERENCE_DATA`) and requires every one of them to be triaged:

* **CITED** -- the value is published by a document this repository ships, and
  the registry names that document plus the literal digits, which are re-read on
  every run (a citation that rots is a red build, not a dangling footnote);
* **UNSOURCED** -- a recalled value.  It is recorded, published here, and
  explicitly *not* evidence.

Being "not evidence" is a property of the *graph*, not of the constant, so two
further columns are computed rather than promised.  **read by** is the
cross-module reader set over `src/`, `tests/` and `scripts/`: a recalled
constant may be read by its registered adjudicator and by nothing else, because
any other reader would let an unsourced number reach a claim.  **adjudicated**
is that adjudicator's verdict, read back from
`benchmarks/verified/wigley3_recalled_ref/results.json`, where each recalled
number was put against two independent codes on three panel counts.  A
`REFUTED` verdict is the registry working as designed -- the recalled value was
wrong, which is precisely why no claim may be built on it.

The gate is fail-closed in the direction that matters: an un-triaged constant is
`undeclared` and turns `tests/test_reference_provenance.py` red, so a recalled
reference table cannot be introduced into a driver silently -- which is exactly
how the four legacy Wigley III drivers below accumulated theirs.

Regenerate this block with::

    python -m tensorbem.reference_provenance --write


<!-- reference-provenance:begin -->

Scanned `src/tensorbem` for module-level constants that announce a literature origin: **15** found, **1** CITED, **14** UNSOURCED.

### CITED -- the value is published in this repository

| constant | location | evidence (file :: digits) | read by |
|---|---|---|---|
| `SEMISUB_PUBLISHED_DISPLACEMENT` | `tensorbem.qtf_drift` | `docs/qtf_drift_oc4.md` :: `13473` | `tensorbem.benchmarks.qtf_drift_study`, `tests/test_qtf_drift.py` |

### UNSOURCED -- open provenance debt, not evidence

These constants are recalled rather than read. They are kept because the drivers that carry them are legacy, but **no published claim may rest on them**: the CITED surface above and the computed anchors in `docs/audit_baseline.md` are what the verification claims use.

Two columns below are machine-checked rather than asserted. **read by** is the cross-module reader graph over `src/`, `tests/` and `scripts/`: a recalled constant is allowed exactly one reader, the adjudicator registered in `ADJUDICATED`, and any other reader turns the suite red. **adjudicated** says what that adjudicator measured when it put the recalled number against two independent codes (`benchmarks/verified/wigley3_recalled_ref/results.json`): a `REFUTED` verdict is the registry's expectation confirmed -- the number was wrong, which is why it is not evidence.

| constant | location | line | read by | adjudicated | why it is still here |
|---|---|---|---|---|---|
| `_JOURNEE_HEAVE_RAO_PEAK` | `tensorbem.benchmarks.wigley3_complete_verification` | 62 | none | not adjudicated | recalled Journee (1992) heave RAO peak (comment in the driver says 'approximate values from experimental data'); the report it cites is not shipped. Legacy driver, ungraded. |
| `_JOURNEE_HEAVE_RAO_PEAK_OMEGA` | `tensorbem.benchmarks.wigley3_complete_verification` | 63 | none | not adjudicated | recalled frequency of the Journee (1992) heave RAO peak; source report not shipped. Legacy driver, ungraded. |
| `_WAMIT_A33_MASS_RATIO` | `tensorbem.benchmarks.wigley3_complete_verification` | 58 | `tensorbem.benchmarks.wigley3_recalled_ref` | FREQUENCY_MISMATCH 1/1 (computed/recalled 4.3-4.3x) | recalled WAMIT heave added-mass ratio at omega = 1 rad/s, marked 'approximate' in the driver itself; no digitised WAMIT table is shipped in the repository. Legacy driver with no capability row: the number reaches no published claim. Read only by the adjudicating driver ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two independent codes and publishes the verdict in ``benchmarks/verified/wigley3_recalled_ref/results.json``. |
| `JOURNEE1992_RAO` | `tensorbem.benchmarks.wigley3_freq_domain_verification` | 68 | none | not adjudicated | recalled Journee (1992) RAO table; the report is not shipped, so the driver's error column cannot be reproduced from repository contents. Ungraded driver. |
| `WAMIT_REF` | `tensorbem.benchmarks.wigley3_freq_domain_verification` | 58 | none | not adjudicated | recalled WAMIT reference block (added mass and damping) typed into a legacy Wigley III driver; no shipped artefact carries the table. Ungraded driver. |
| `_JOURNEE_EXCITATION` | `tensorbem.benchmarks.wigley3_freq_verification` | 88 | none | not adjudicated | recalled Journee (1992) excitation amplitudes at beta = 0 deg; source report not shipped. Ungraded driver. |
| `_JOURNEE_RAO_PEAKS` | `tensorbem.benchmarks.wigley3_freq_verification` | 98 | none | not adjudicated | recalled Journee (1992) RAO peaks (driver comment: 'approximate'). Ungraded driver. |
| `_WAMIT_ADDED_MASS_DIAGONAL` | `tensorbem.benchmarks.wigley3_freq_verification` | 68 | `tensorbem.benchmarks.wigley3_recalled_ref` | FREQUENCY_MISMATCH 1/6, REFUTED 5/6 (computed/recalled 0.27-12.8x) | recalled diagonal WAMIT added mass at omega = 1 rad/s (legacy driver); no shipped table. No capability row reads this constant; its only reader is the adjudicating driver ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two independent codes and publishes the verdict in ``benchmarks/verified/wigley3_recalled_ref/results.json``. |
| `_WAMIT_DAMPING_DIAGONAL` | `tensorbem.benchmarks.wigley3_freq_verification` | 78 | `tensorbem.benchmarks.wigley3_recalled_ref` | REFUTED 6/6 (computed/recalled 3.17e-06-1.91x) | recalled diagonal WAMIT damping at omega = 1 rad/s; no shipped table. Its only reader is the adjudicating driver ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two independent codes and publishes the verdict in ``benchmarks/verified/wigley3_recalled_ref/results.json``. |
| `JOURNEE_HEAVE_RAO_REF` | `tensorbem.benchmarks.wigley_diffraction_radiation` | 116 | none | not adjudicated | recalled Journee (1992) heave RAO; source report not shipped. Ungraded driver. |
| `JOURNEE_PITCH_RAO_REF` | `tensorbem.benchmarks.wigley_diffraction_radiation` | 126 | none | not adjudicated | recalled Journee (1992) pitch RAO; source report not shipped. Ungraded driver. |
| `WAMIT_A33_REF` | `tensorbem.benchmarks.wigley_diffraction_radiation` | 86 | `tensorbem.benchmarks.wigley3_recalled_ref` | REFUTED 1/1 (computed/recalled 446-446x) | recalled WAMIT heave added mass for Wigley III; the WAMIT verification case is not shipped with the repository. Its only reader is the adjudicating driver ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two independent codes and publishes the verdict in ``benchmarks/verified/wigley3_recalled_ref/results.json``. |
| `WAMIT_B33_REF` | `tensorbem.benchmarks.wigley_diffraction_radiation` | 96 | `tensorbem.benchmarks.wigley3_recalled_ref` | REFUTED 1/1 (computed/recalled 120-120x) | recalled WAMIT heave damping for Wigley III; no shipped table. Its only reader is the adjudicating driver ``tensorbem.benchmarks.wigley3_recalled_ref``, which measures it against two independent codes and publishes the verdict in ``benchmarks/verified/wigley3_recalled_ref/results.json``. |
| `WAMIT_F3_BETA0_REF` | `tensorbem.benchmarks.wigley_diffraction_radiation` | 106 | none | not adjudicated | recalled WAMIT heave excitation at beta = 0 deg; no shipped table. Ungraded driver. |

Fail-closed checks: undeclared **0**, stale registry entries **0**, unresolvable citations **0**, unregistered readers of a recalled constant **0**, adjudicator/artefact mismatches **0**.  A new literature-marked constant that is not triaged turns `tests/test_reference_provenance.py` red, so an un-sourced reference number cannot enter a driver silently.

<!-- reference-provenance:end -->
