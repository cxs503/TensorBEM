# TensorData

**AI-native data production, governance, and delivery infrastructure.**

TensorData is the data-asset infrastructure for the Tensor ecosystem. It turns heterogeneous evidence and observations into governed, versioned, discoverable, and reusable data products for AI training, agents, scientific computing, and future world models.

## Position in the Tensor ecosystem

- **TensorPlatform** runs agents and workflows: identity, orchestration, execution, tools, and compute coordination.
- **TensorData** manages data assets: acquisition, storage references, schemas, quality, provenance, lineage, versions, dataset assembly, and data delivery.
- **TensorKnowledge** produces and evolves explicit knowledge: concepts, claims, evidence relationships, theories, uncertainty, and knowledge validation.
- **Domain projects** produce and consume domain-specific observations, simulations, algorithms, and evaluations.

These systems integrate through explicit APIs and stable identifiers. They must not share private implementation modules or rely on shared business tables.

## Core principles

1. **Use-case driven:** define downstream tasks and acceptance tests before scaling collection.
2. **Evidence first:** preserve source artifacts and acquisition context; never silently overwrite raw evidence.
3. **Multimodal by design:** support text, tables, images, audio, video, time series, spatial data, scientific fields, and agent traces.
4. **Provenance by default:** every derived artifact must trace to inputs, transformations, versions, and responsible actors.
5. **Quality is executable:** quality rules produce machine-readable results and block promotion when required.
6. **Rights-aware:** record license, consent, restrictions, retention, and permitted purposes before use.
7. **Reproducible datasets:** a dataset version resolves to immutable membership, schema, transformation versions, and checksums.
8. **No fake readiness:** demos, synthetic examples, and production-verified capabilities are clearly distinguished.
9. **Open interfaces:** prefer interoperable formats and APIs; avoid unnecessary infrastructure duplication.
10. **Agent-operated, human-governed:** agents may produce and review assets, while high-impact releases require independent checks and explicit policy.

## Architecture and migration

- [Architecture overview](docs/architecture/overview.md)
- [Data asset model](docs/architecture/data-asset-model.md)
- [Governance and quality](docs/governance/data-governance.md)
- [TensorPlatform migration plan](docs/migration/tensorplatform-migration.md)
- [Initial TensorPlatform capability audit](docs/migration/tensorplatform-inventory-initial.md)
- [Roadmap](docs/roadmap.md)
- [Local quickstart and API contract](docs/api/quickstart.md)

## Current implementation

An early development service now includes:
- FastAPI service and health endpoint
- Synchronous Python SDK for Agent-side asset, upload, quality, dataset, provenance and lineage workflows
- Asset metadata registration and catalog listing
- API-key gate for all data APIs
- Local content-addressed artifact upload/retrieval with SHA-256 metadata
- Dataset registry and immutable DatasetVersion manifest API
- Canonical JSON manifest checksum and artifact/rights/lifecycle snapshot fields
- Append-only provenance activity records with validated artifact references and checksum matching
- Artifact lineage graph traversal across connected provenance activities
- Dataset manifests snapshot the exact quality report and ruleset used for admission
- Alembic baseline migration with upgrade/downgrade support
- Executable general/tabular artifact quality checks with immutable reports
- Dataset-version admission gate requiring a passing report for the current artifact checksum
- API integration tests and GitHub Actions test/lint workflow

**Not production-ready:** evidence-backed rights approval, domain-specific quality validators, scoped identity/authorization, S3-compatible storage, retention/garbage collection, and deployment hardening remain future gates. Dataset manifests may currently snapshot assets in quarantine or rights-review states; a snapshot is not an authorization to train, publish, or redistribute data.

## Development

Schema migrations are managed with Alembic. Local quickstart mode retains automatic table creation for convenience; production containers disable it and run `alembic upgrade head` before starting the API.

See [Local quickstart](docs/api/quickstart.md). Do not commit secrets, real restricted data, or generated datasets to this repository.

## License

See [LICENSE](LICENSE) for the software license and [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the boundary between project software and third-party or content assets. Dataset, paper, model, customer, and other non-software assets retain their own rights and restrictions.
