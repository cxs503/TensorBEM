# TensorData Roadmap

## M0 — Architecture and contracts
- [x] Establish project boundary and principles
- [x] Define canonical data asset concepts
- [x] Define governance and release gates
- [x] Define TensorPlatform migration policy
- [x] Complete an initial, explicitly provisional code scan of TensorPlatform data capabilities
- [ ] Complete repository-wide dependency, caller, migration and runtime audit

## M1 — Minimal real service
- [x] Python package and application entry point
- [x] Environment configuration and health endpoint
- [x] Relational metadata persistence baseline (SQLite/PostgreSQL URL support; schema migrations still missing)
- [x] Filesystem artifact backend with SHA-256 content addressing
- [x] Basic DataAsset registration and catalog read APIs
- [x] DataArtifact upload, listing and content retrieval
- [x] API-key gate for all asset APIs
- [x] Alembic baseline migration with upgrade/downgrade support
- [ ] Full DataAsset/DataArtifact lifecycle and update/delete policy
- [x] Dataset registry API
- [x] Immutable DatasetVersion manifest API with content checksum
- [x] Initial provenance activity record API
- [ ] JSON Schema validation
- [ ] CI fully green on current default branch
- [x] Docker-based local development entry point

## M2 — Trust and governance
- [ ] Source and rights registry with evidence-backed approval
- [x] Validate artifact references and supplied checksums in provenance activities
- [x] Traverse connected artifact lineage graph from a target artifact
- [ ] Add paginated lineage queries, dataset-version nodes and external source identity resolution
- [x] Initial general/tabular quality rule execution and immutable reports
- [x] Dataset-version gate requiring a passing quality report for the current artifact checksum
- [ ] Domain validators (physical constraints, time continuity, multimodal alignment)
- [ ] Quarantine/approval/publication transitions
- [ ] Audit events and purpose-aware access checks
- [ ] Prevent training/export use of assets without explicit rights and quality clearance

## M3 — Processing and connectors
- [ ] Profiling, cleaning and transformation contracts
- [ ] Connector SDK and first authorized source connectors
- [ ] Dataset manifest resolve/export API
- [ ] Idempotency, retries and operational observability

## M4 — Tensor ecosystem integration
- [x] Initial synchronous TensorData Python client for Agent workflows
- [ ] TensorPlatform production integration and workflow contract
- [ ] TensorKnowledge asset registration contract
- [ ] Domain validator plugin contract
- [ ] Ship/physics AI pilot datasets and evaluation feedback

## M5 — Scale when justified
- [ ] S3-compatible storage
- [ ] Columnar/large-scale query engine
- [ ] Streaming ingestion
- [ ] Search/vector index as derived indexes
- [ ] Distributed processing and high-volume scheduling

Scale components are gated by measured workload, not adopted pre-emptively.

## Current milestone note

Artifact quality checks currently verify stored size and SHA-256; the tabular profile additionally checks CSV headers/rows and reports missing cells and duplicate rows. Dataset manifests snapshot the exact passing quality report ID, profile and ruleset version used for admission. Reports are append-only and tied to the exact artifact checksum. Dataset version creation now rejects artifacts without a passing report for their current checksum. This is a baseline validator, not yet a domain-scientific quality system.

Dataset versions snapshot artifact IDs, content hashes, roles, selectors, rights/lifecycle state at snapshot time, split definitions, schema refs and transform refs. A manifest SHA-256 is calculated from canonical JSON. Version creation is append-only through the API: an existing dataset/version pair cannot be overwritten.

The provenance endpoint validates artifact references and optional checksums when artifact IDs are supplied. The lineage endpoint traverses connected provenance activities from a target artifact and returns artifact nodes, activity edges and external references. This is an initial internal graph contract, not yet a complete W3C PROV implementation; dataset-version nodes, pagination, external source identity resolution and authorization-aware graph filtering remain open.

The service remains an early development baseline. Alembic now owns the initial schema baseline. Local development retains opt-in-by-default `create_all` for compatibility; the Docker image disables automatic schema creation and applies migrations before startup. Production readiness still requires migration testing against the target database, authorization hardening and operational validation.
