# TensorPlatform Data Capability Inventory — Initial Evidence

**Scope:** code-search discovery plus direct inspection of selected backend modules on the TensorPlatform default branch.  
**Status:** preliminary; not a complete repository-wide dependency audit. Classifications below are provisional until tests, callers, migrations and runtime behavior are traced.

## Findings

| Component | Evidence observed | Provisional class | TensorData action |
|---|---|---:|---|
| `backend/models/data_catalog.py` + `backend/api/data_catalog.py` + `backend/services/data_catalog_service.py` | SQLAlchemy asset/tag/classification models and catalog API; service includes a method explicitly registering virtual data and assigns a fixed quality score of 85 | B/D mixed | Reuse model concepts only; redesign stable IDs, provenance and rights. Do not copy virtual-data assumptions |
| `backend/services/data_cleaning_service.py` | Declarative cleaning operations; AST literal parsing; explicit limits; documentation says uploaded Python rule files are not executed | A/B | High-priority candidate; isolate dataframe transforms behind a pure, tested interface and preserve safe parsing |
| `backend/api/data_cleaning.py` | API layer tied to TensorPlatform database, dataset paths and authentication | B/E mixed | Reuse service logic; reimplement API against TensorData contracts |
| `backend/services/data_quality_service.py` | Completeness, accuracy, consistency, timeliness, uniqueness and validity checks; rules and reports held in process dictionaries | B | Reuse/check individual deterministic validators; replace in-memory state with immutable persisted QualityReport |
| `backend/api/data_quality.py` | Large API module defines SQLAlchemy rule/result models and integrates Great Expectations adapter | B | Audit DB migration and tenancy dependencies; extract rule/result contracts, do not copy router |
| `backend/api/data_validation.py` | Uses TensorPlatform Dataset table, hard-coded upload directory and local DataFrame loading | B/E | Reuse validation algorithms after decoupling file lookup and ORM |
| `backend/services/data_governance_acceptance_service.py` | Explicit basic/strict profiles; deterministic completeness and duplicate-rate checks | A/B | Strong candidate for reusable rule logic; make profiles versioned and reports immutable |
| `backend/api/dataset_governance.py` | Read-only governance admission API; requires platform permissions, organization scoping and Dataset stage == cleaned | B/E | Preserve admission semantics where useful; replace platform auth/tenant and Dataset coupling |
| `backend/api/data_governance.py` | Stats query fixed domain tables; catches broad exceptions and reports virtual-data summaries | D | Do not migrate as core governance service |
| `backend/services/data_collection_service.py` | Thread/queue task implementation; imports a sensor generator; collection state appears process-local | B/D | Inspect real connector support; separate connector protocol from demo sensor generator and process-local scheduler |
| `backend/services/data_pipeline.py` | Orchestrates upload, governance, standardization and training using TensorPlatform Dataset/Model/TrainingJob models | E/B | Do not migrate as-is; split data transformation/manifest logic from orchestration and training execution |
| `backend/api/data_lineage.py` + `backend/models/lineage.py` | Substantial lineage API/model surface; depends on platform auth, tenant scoping and SQL lineage extraction | B | Audit schema and tests; migrate domain model/lineage engine behind new API, not platform router |
| `backend/api/data_profiling.py` | Dataset CSV loading, local upload paths, editing and alignment; platform ORM and LLM service dependencies | B/E mixed | Extract profiling/alignment algorithms selectively; keep agent/LLM orchestration outside TensorData |
| `backend/api/data_sync.py`, `backend/services/data_sync_service.py` | Data synchronization surfaces found by code search | C pending | Inspect source, persistence, retries, idempotency and connector credentials before classification |
| `backend/api/data_standard.py`, `backend/models/data_standard.py` | Data standard/schema surfaces found by code search | C pending | Inspect and map to versioned DataSchema/transform contracts |
| `backend/api/data_pipeline_api.py` | Pipeline API found by code search | C pending | Determine whether it is data transformation or general workflow orchestration |
| `backend/api/datasets.py`, `backend/models/core.py` Dataset | Existing platform dataset lifecycle | E/B | Avoid importing core Dataset model; create TensorData Dataset and immutable DatasetVersion independently |
| `backend/tasks/training_tasks.py`, `backend/api/real_training.py`, `backend/services/real_training_service.py` | Training execution surfaces found | E | Keep execution, GPU/process scheduling and training job lifecycle out of TensorData; consume immutable dataset refs |

## Key architectural conclusions

1. TensorPlatform has reusable data-processing logic, but many APIs are coupled to its ORM, tenant/auth system, upload paths and service container.
2. Data cleaning and governance acceptance logic are the strongest early extraction candidates.
3. Data quality has a split implementation: deterministic checks exist, but one service stores rules/reports in memory. Extract checks; redesign persistence.
4. The generic governance statistics API includes domain-specific virtual tables and broad exception swallowing; do not treat it as a reusable governance core.
5. Collection must not be migrated until synthetic/demo sources are separated from real authorized connectors.
6. The existing end-to-end pipeline mixes data engineering with platform orchestration and model training. Split by bounded context instead of moving it wholesale.
7. Dataset lineage is a large and potentially valuable capability, but platform tenancy and auth dependencies require a deliberate adapter/reimplementation.

## Next audit actions

- Enumerate every related model, migration, router registration, frontend route and API client.
- Trace all callers/imports and table ownership for candidate services.
- Inspect tests and fixtures; identify mocked, virtual and hard-coded paths.
- Inspect collection connectors, sync idempotency, data-standard versioning and lineage SQL extraction.
- Build a component dependency graph and migration-by-migration acceptance matrix.
- Do not begin bulk code migration until the above evidence is attached to each A/B candidate.
