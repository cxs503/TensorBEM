# Local Quickstart and API Contract

## Status

The current code is an early MVP: asset metadata registry, content-addressed artifact upload/retrieval, dataset registry, immutable dataset-version manifests and append-only provenance activity records. It is not yet a production data lake or complete governance engine.

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export TENSORDATA_API_KEY='replace-with-a-long-random-secret'
alembic upgrade head
uvicorn tensordata.main:app --reload
```

Health: `GET http://localhost:8000/health`

Interactive API documentation: `http://localhost:8000/docs`

By default metadata uses `./data/tensordata.db` and artifact payloads use `./data/artifacts`. Local quickstart keeps `TENSORDATA_AUTO_CREATE_SCHEMA=true` for backwards-compatible convenience; set it to `false` when migrations are the sole schema authority. Configure `TENSORDATA_DATABASE_URL`, `TENSORDATA_STORAGE_ROOT`, and `TENSORDATA_MAX_UPLOAD_BYTES` for other environments. Keep the API key out of source control.

## Authentication

All `/api/v1` routes require `X-API-Key`. If `TENSORDATA_API_KEY` is not configured, data API access is disabled. The health endpoint is intentionally unauthenticated and returns no asset information.

This initial API-key mechanism is suitable only for controlled development and trusted internal deployment behind TLS/network controls. Production deployment still requires identity integration, scoped authorization, secret rotation, audit logging and rate limiting.

## Current endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health |
| POST | `/api/v1/assets` | Register an asset in quarantine |
| GET | `/api/v1/assets` | List/filter assets |
| GET | `/api/v1/assets/{asset_id}` | Read asset metadata |
| POST | `/api/v1/assets/{asset_id}/artifacts` | Upload immutable content-addressed artifact |
| GET | `/api/v1/assets/{asset_id}/artifacts` | List artifact metadata |
| GET | `/api/v1/artifacts/{artifact_id}/content` | Retrieve artifact payload |
| POST | `/api/v1/datasets` | Create a logical dataset |
| GET | `/api/v1/datasets` | List datasets |
| POST | `/api/v1/datasets/{dataset_id}/versions` | Create an immutable dataset version |
| GET | `/api/v1/datasets/{dataset_id}/versions` | List dataset versions |
| GET | `/api/v1/datasets/{dataset_id}/versions/{version}` | Resolve a version manifest |
| POST | `/api/v1/provenance` | Record a processing/production activity |
| GET | `/api/v1/provenance` | List/filter activity records |

## Dataset version example

Create a logical dataset:

```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "ship-motion-training",
    "owner": "tensor-physics",
    "intended_uses": ["research", "model_training"]
  }'
```

Create an immutable version using one or more artifact IDs returned by artifact upload:

```bash
curl -X POST http://localhost:8000/api/v1/datasets/DATASET_ID/versions \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "1.0.0",
    "members": [{"artifact_id": "ARTIFACT_ID", "role": "observations"}],
    "split_definition": {"strategy": "not_yet_split"},
    "schema_refs": ["ship-motion-timeseries/v1"],
    "transform_refs": [],
    "created_by": "tensor-data-agent"
  }'
```

The response includes a canonical manifest and its SHA-256 checksum. The manifest snapshots each member's artifact checksum, size, media type, role, selector, and asset rights/lifecycle state at creation time. Reusing the same dataset/version pair returns HTTP 409; create a new version instead of mutating an existing one.

## Provenance example

```bash
curl -X POST http://localhost:8000/api/v1/provenance \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "activity_type": "simulation_export",
    "inputs": [{"asset_id": "INPUT_ASSET_ID"}],
    "outputs": [{"artifact_id": "OUTPUT_ARTIFACT_ID"}],
    "parameters": {"solver": "TensorLBM"},
    "responsible_actor": "simulation-agent",
    "code_digest": "git:COMMIT",
    "environment_digest": "container:DIGEST"
  }'
```

The provenance API stores declared activity evidence. It does not yet validate every referenced entity or provide lineage graph traversal.

## Asset upload and retrieval

Example asset registration:

```bash
curl -X POST http://localhost:8000/api/v1/assets \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "ship motion observations",
    "asset_type": "timeseries",
    "owner": "tensor-physics",
    "rights_status": "review_required",
    "allowed_purposes": ["research"],
    "metadata": {"domain": "ship"}
  }'
```

Upload the source payload using the returned asset ID:

```bash
curl -X POST 'http://localhost:8000/api/v1/assets/ASSET_ID/artifacts' \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -F 'file=@observations.csv'
```

Download using the artifact ID returned by the upload:

```bash
curl -L 'http://localhost:8000/api/v1/artifacts/ARTIFACT_ID/content' \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -o observations.csv
```

The server computes SHA-256 and stores payloads under a digest-derived path. Duplicate payload bytes within the same asset reuse the artifact record; the same payload may be referenced by multiple assets. The original filename is metadata only and is never used to construct a storage path.

## Known limitations / next gates

- An initial Alembic migration and downgrade path are implemented. Future model changes must be added as reviewed migrations; do not rely on `create_all` to evolve an existing database.
- The API key is a single shared secret, not user/service identity or resource-scoped authorization.
- Asset lifecycle transitions and approval gates are not yet exposed.
- Source registry, evidence-backed rights approval, executable quality reports and verified lineage graph are not yet implemented.
- Dataset manifest references (schema/transform IDs) are currently recorded as declared identifiers, not yet resolved against registries.
- Dataset versions currently permit snapshots of quarantined or rights-review-required assets; this records state but does not authorize training or publication.
- Local filesystem storage is for development; S3-compatible backend and retention/garbage collection are future work.
- The upload handler enforces a configurable application-level size limit; production deployments also need proxy/server request limits and malware/content inspection.
- CI must complete successfully before treating the service as deployable.


## Artifact quality checks

Run the general integrity profile or the tabular CSV profile:

```bash
curl -X POST "$TENSORDATA_URL/api/v1/artifacts/$ARTIFACT_ID/quality-checks" \
  -H "X-API-Key: $TENSORDATA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"profile":"tabular","executed_by":"tensor-platform-agent"}'
```

Retrieve immutable reports:

```bash
curl "$TENSORDATA_URL/api/v1/artifacts/$ARTIFACT_ID/quality-reports" \
  -H "X-API-Key: $TENSORDATA_API_KEY"
```

Dataset version creation requires a passing report whose recorded SHA-256 matches the artifact's current checksum. The tabular profile currently reports missing cells and duplicate rows as warnings; empty/malformed files, inconsistent row widths and integrity mismatches fail the report. Every report's summary additionally records `measured_sha256` (the bytes actually examined) and `executed_rules` (rule id + version pairs) for traceability — engine design and extension recipe: [docs/quality/engine.md](../quality/engine.md). Domain-specific physical and multimodal validators are not yet included.


## Provenance and lineage

Provenance activity creation validates any supplied `artifact_id` references against registered artifacts and rejects mismatched supplied SHA-256 values. External source references may be represented without an artifact ID. Query the connected activity graph for an artifact:

```bash
curl "$TENSORDATA_URL/api/v1/lineage/artifacts/$ARTIFACT_ID" \
  -H "X-API-Key: $TENSORDATA_API_KEY"
```

The response includes connected artifact nodes, provenance activities, and external input references. This initial graph is not yet a full W3C PROV implementation.


## Python client for agents

Install TensorData with its runtime dependencies, then use the client to call the governed API workflow:

```python
from tensordata.client import TensorDataClient

with TensorDataClient("https://your-tensordata-service", api_key="YOUR_API_KEY") as client:
    asset = client.create_asset({
        "name": "ship motion observations",
        "asset_type": "timeseries",
        "owner": "ship-data-agent",
    })
    artifact = client.upload_artifact(
        asset["id"], "motion.csv", b"time,speed\\n0,2.1\\n1,2.4\\n",
        media_type="text/csv",
    )
    report = client.run_quality_checks(
        artifact["id"], profile="tabular", executed_by="ship-data-agent"
    )
    if report["status"] != "passed":
        raise RuntimeError(f"Quality gate failed: {report['findings_json']}")
```

The client raises `TensorDataAPIError` for non-success responses and exposes the HTTP status and structured detail. Policy enforcement remains on the service.
