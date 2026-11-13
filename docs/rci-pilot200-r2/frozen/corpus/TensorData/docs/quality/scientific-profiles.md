# Scientific data quality profiles — interface drafts (stage 2+ design only)

Per the stage-1 directive: these are **interface and design drafts, not
implementations**. Nothing here is registered in `build_default_registry()`
or wired into the API `Literal` until real data contracts from the
TensorPhysics pilot (stage 3) pin the formats. The stage-1 audit's lesson —
"先审计再迁移，不得重复实现" — applies doubly to profiles without real
payloads to validate against.

## Common extension path (no engine changes)

Adding a profile after stage 1 requires exactly three additions:

```
tensordata/quality/rules/<domain>.py      # rules, stable ids + versions
tensordata/quality/profiles/<domain>.py   # Profile(name, version, rules)
                                            + registration in build_default_registry()
tensordata/quality/base.py                # optional: a new memoized scan type
                                            # on RuleContext (e.g. ctx.series_scan())
```

Plus: API `Literal["..."]` extension in `schemas.py` (a breaking schema
change — coordinate with SDK consumers), gate profile-derivation rules in
`main.py`, and tests using **real pilot payloads** (fixture files committed
under `tests/fixtures/<domain>/`, never mocked scans).

### Proposed scan contracts

Stage 1 established the pattern: one streaming pass per format, memoized on
`RuleContext`, shared by all rules of that format.

- `SeriesScan` (timeseries): per-column min/max/NaN/Inf counts, first-order
  diffs for monotonicity and gap statistics — computable in a single chunked
  pass; no full-column buffering.
- `FieldScan` (scientific_field): shape/dtype/axis metadata + NaN map summary
  (count, bounding boxes of NaN islands) via chunked iteration over the
  array file (NPY/NPZ first; NetCDF later — blocked on real pilot format).
- `MeshScan` (mesh): vertex count, element count, per-type element tallies,
  degenerate-element count (zero-area/volume), orientation histogram.
  Requires a streaming mesh reader — format decision (OBJ? VTK? custom?)
  is an open question, not a design gap.
- `MultimodalCheckContext`: no new scan; rules assert **cross-member
  consistency** (frame count vs row count vs manifest entries) and therefore
  need read access to the other members of a candidate dataset version —
  this crosses the current single-artifact engine boundary and is the main
  API-level open question below.

## Draft rule inventories

Ids are `domain.*`, severities follow the stage-1 semantics (error ⇒ report
failed; warning/info recorded). All rules carry independent versions.

### timeseries (stage-3 pilot: ship-motion / LBM probe streams)

| rule id | severity | fires when |
|---|---|---|
| `series.parseable` | error | payload not decodable as the agreed series format |
| `series.monotonic_time` | error | time column non-monotonic or non-increasing |
| `series.time_gaps` | warning | Δt deviations beyond tolerance (report max/median gap) |
| `series.finite_values` | error | NaN/Inf outside an allowed "sentinel" contract |
| `series.physical_range` | warning | values outside per-column envelope from profile config |
| `series.summary` | — | row_count, t_min/t_max, columns, per-column min/max/NaN (like `tabular.csv_summary`) |

Open decisions: envelope config source (per-asset metadata vs profile
constant vs caller-supplied `params` on the check request); sampling-rate
contract (fixed Δt vs event-based).

### scientific_field (CFD/LBM/FEM field dumps)

| rule id | severity | fires when |
|---|---|---|
| `field.shape_consistent` | error | axes/dtype disagree with declared metadata |
| `field.coordinate_monotonic` | error | any coordinate axis non-monotonic |
| `field.nan_islands` | warning | NaN clusters (count + max bbox reported) |
| `field.unit_metadata` | warning | units missing/unknown for physical quantities |
| `field.summary` | — | shape, dtype, coordinate ranges, NaN totals |

Open decisions: file format for stage-3 pilots (NPY per field? NPZ bundle?
HDF5 later); whether unit metadata lives in artifact `metadata_json` or a
sidecar member.

### mesh (DTMB/SPH geometry assets)

| rule id | severity | fires when |
|---|---|---|
| `mesh.manifold` | error | non-manifold edges/vertices (welding tolerance configurable) |
| `mesh.degenerate_elements` | error | zero-area/volume elements |
| `mesh.orientation` | warning | inconsistent winding (report mixed fraction) |
| `mesh.waterproof` | warning | open boundary edges (info for open surfaces — severity context-dependent) |
| `mesh.summary` | — | vertex/element counts, bbox, per-type tallies |

Prior art in-house: DTMB 5415 geometry work (water-tight manifold
validation) and TensorCAD output conventions — reuse their definitions, do
not re-derive.

### multimodal (paired sensor + image + config datasets)

| rule id | severity | fires when |
|---|---|---|
| `multimodal.member_completeness` | error | required modalities missing from the manifest |
| `multimodal.count_alignment` | error | cross-modal counts disagree (rows vs frames vs entries) |
| `multimodal.timestamp_alignment` | warning | timestamps not joinable within tolerance |
| `multimodal.summary` | — | per-member counts and alignment stats |

## API-level open questions (owner decisions before stage 2)

1. **Boundary crossing**: multimodal/cross-artifact checks do not fit the
   single-artifact `POST /artifacts/{id}/quality-checks` shape. Options:
   (a) check-on-version endpoint `POST /datasets/{id}/versions/quality-checks`
   that runs profiles over a candidate member set, (b) keep engine
   single-artifact and express cross-modal rules at the version gate.
   Recommendation: (a), because the gate must stay a cheap lookup.
2. **Gate-time re-measure**: stage 1 kept the gate as report-lookup +
   metadata equality. Post-check on-disk tampering is detectable only by
   re-running a check. Candidate stage-2 hardening: gate re-hashes payload
   (streaming SHA-256 is cheap) before accepting a report.
3. **Profile parameterization**: physical envelopes/tolerances need a
   config channel. Candidate: optional `params` dict on QualityCheckCreate,
   snapshotted into `summary_json["profile_params"]` for reproducibility.
4. **Severity policy for scientific findings**: NaN may be legitimate
   (masked land cells) or fatal (corrupt solver output). The profile config
   should own per-rule severity overrides, versioned with the profile.

## Pilot sequencing (stage 3, TensorPhysics)

1. Collect real payloads from the first ship-physics pipeline (LBM probe
   time series → `timeseries`; field dumps → `scientific_field`) and commit
   fixtures.
2. Implement `timeseries` first (smallest format surface, direct LBM
   mapping), run as **non-gating advisory profile** alongside tabular until
   two full pilot cycles validate its findings.
3. Only then extend the dataset-version gate derivation — advisory →
   enforced is a one-line policy change once trust is earned.
