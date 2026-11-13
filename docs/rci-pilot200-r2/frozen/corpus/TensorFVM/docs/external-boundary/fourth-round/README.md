# Independent fourth-round raw audit

This report is separate from the earlier 23-case development report. It checks
21 actual artifacts: four local moving-disk cases, ten fixed-boundary DEM
cases, four smooth moving-plane cases and three live rigid fluid-feedback cases.
The implementation imports NumPy only, without the other repositories' solvers
or audit functions. Every consumed raw artifact and physics source is SHA256
bound in `report.json`.

Checks reconstruct fluid mass and momentum from D2Q9 populations; local
wall/conversion impulse balance and absence of global rescaling; DEM tributary
mass, fixed support, kinetic/spring/contact energy and momentum; smooth-plane
force-displacement work and impulse quadrature. Fixed-plane histories are
sampled, so their complete tool impulses are independently recomputed through
NumPy velocity-Verlet replay, also checking final displacement and velocity.
Live feedback checks each exchange's current clock, pose and held velocity,
body impulse update, midpoint kinetic-energy work identity and final combined
fluid/body momentum, explicitly accounting for the legacy global reservoir.

Numerical bookkeeping passes, **physical qualification remains false**:

- Local disk refinement impulse changes **7.6302%**, exceeding 3%; the
  co-moving artificial impulse remains exposed in the report.
- Abrupt stationary-plane contact impulse changes **10.1258%**, exceeding 3%.
- Smooth-plane impulse and end response satisfy the declared 3% gate; this
  fixed reference network does not establish fractured disk ice convergence.
- Live rigid feedback has combined time/compressibility impulse sensitivity
  **46.3372%**, exceeding 3%. It is a moving rigid disk, without ice, flexible
  shell, rotation or SUBOFF geometry. Body work identity is not total fluid
  energy accuracy.

Reproduce from TensorFVM with sibling development repositories:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_fourth_round.py \
  --lbm-root ../TensorLBM --dem-root ../TensorDEM --fem-root /data/TensorFEM
PYTHONPATH=src python -m pytest -q tests/test_fourth_round_audit.py
```

The report records source paths from the executing workspace; these paths are
provenance, not a portable installation requirement. CLI roots can be replaced
with other checkouts containing the exact bound revisions. Qualification badge,
momentum ledger, global rescale, invalid population and stale pose tampering
are rejected by six tests.
