# TensorPlatform Data Capability Migration Plan

## Objective
Extract reusable data-asset capabilities into TensorData without duplicating mature logic, breaking TensorPlatform, or importing its private runtime/database dependencies. This is a migration policy, not proof that all components have been audited.

## Ownership
- TensorData: asset catalog, ingestion metadata, cleaning/transforms, profiling, quality, lineage, masking, dataset governance, manifests and training-data preparation.
- TensorPlatform: agent runtime, identity, scheduling, workflow execution, tools, compute orchestration and training-job execution.
- TensorKnowledge: knowledge reasoning and semantic evolution; TensorData stores its outputs as versioned assets.
- Domain projects: domain-specific scientific validation through pluggable validators.

## Required inventory
Audit API routers/schemas, ORM models/migrations, services/repositories, background tasks/queues, frontend pages/API clients/state, tests/fixtures/mocks, configuration/deployment/observability, downstream imports and direct database access.

For each component record path, responsibility, dependencies, persistence, API consumers, test evidence, mock/hardcoded behavior, rights/security implications, maturity, target owner, migration action and acceptance tests.

Maturity classes: A verified reusable; B useful logic requiring refactor; C incomplete/reimplement; D demo/mock/do not migrate as production; E belongs to TensorPlatform.

## Migration sequence
1. Freeze/inventory behavior and capture API/schema contracts.
2. Define canonical entities and cross-system APIs.
3. Implement independent persistence and artifact storage abstraction.
4. Migrate catalog and dataset registry.
5. Migrate validation/profiling/cleaning with regression tests.
6. Migrate provenance and lineage.
7. Migrate ingestion connectors and dataset packaging.
8. Integrate TensorPlatform through API/SDK and events.
9. Move UI ownership only after service/API stability.
10. Deprecate old routes only after usage telemetry and a compatibility window.

## Safety constraints
No bulk copy before dependency and rights review; no shared business tables; no TensorData imports from TensorPlatform internals; no breaking changes without compatibility plan; no promotion of mocks as production; preserve source origin commit/path; use adapters/feature flags; every migration unit needs tests, runtime evidence, rollback instructions and independent review.

## Exit criteria
TensorData runs independently, owns persistence, exposes documented APIs, passes integration/regression tests, and TensorPlatform consumes it without direct access to TensorData internal tables/modules.
