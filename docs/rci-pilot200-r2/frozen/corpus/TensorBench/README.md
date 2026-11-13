# TensorBench

**Reproducible benchmarking and evidence infrastructure for the Tensor ecosystem.**

TensorBench defines, executes, records, compares, and publishes reproducible benchmarks for scientific-computing software, AI/data systems, and agent-built applications in the Tensor ecosystem.

It is intended to answer a simple but demanding question:

> Given a stated workload, dataset, software version, and hardware/software environment, what was measured, how was it measured, and can another person reproduce the result?

TensorBench is not merely a collection of performance numbers or demonstration screenshots. A benchmark result must be tied to an explicit specification, a versioned workload, an execution environment, raw or traceable measurements, and a validation status.

## Mission

Build a common, auditable benchmark layer that helps Tensor projects and external users:

- compare implementations under controlled conditions;
- measure correctness as well as speed, memory use, scalability, and robustness;
- detect regressions across commits and releases;
- assess numerical accuracy against trusted references;
- evaluate AI/agent-produced software using repeatable tasks and objective evidence;
- publish results that can be independently inspected and reproduced.

## Scope

TensorBench is a cross-domain infrastructure project. It should support multiple benchmark families through shared contracts rather than hard-code one solver or scientific domain.

Initial benchmark families may include:

1. **Scientific computing:** solver correctness, convergence, numerical error, throughput, memory, and scaling.
2. **AI and data systems:** ingestion, transformation, retrieval, training/inference throughput, resource use, and data-quality metrics.
3. **Agent-produced systems:** task completion, artifact validity, test coverage, reproducibility, cost, latency, and regression rate.
4. **Application workflows:** end-to-end workflows with declared inputs, outputs, acceptance criteria, and resource budgets.

Domain projects such as TensorLBM, TensorFEM, TensorDEM, TensorBEM, TensorSPH, TensorDG, TensorFVM, TensorPhysics, TensorData, and TensorKnowledge own their domain implementations and domain-specific scientific meaning. TensorBench owns the shared benchmark contracts, execution/evidence conventions, result records, and cross-project comparison mechanisms.

## Core principles

- **Correctness before speed:** a faster incorrect result is a failed benchmark.
- **Reproducibility by construction:** record exact code, workload, configuration, environment, and execution metadata.
- **Evidence over claims:** distinguish measured facts, derived metrics, estimates, and narrative interpretation.
- **Comparable conditions:** comparisons must disclose differences in hardware, software, precision, and workload.
- **No fabricated results:** missing measurements remain missing; synthetic examples must be explicitly labelled.
- **Layered validation:** schema validity, execution validity, scientific/numerical validity, and independent reproduction are distinct states.
- **Domain ownership:** TensorBench does not silently redefine a solver's scientific acceptance criteria.
- **Open by default:** publish software and benchmark specifications openly where rights and security permit; data and third-party assets retain their own terms.

## Planned architecture

- **Benchmark specification:** machine-readable workload, inputs, metrics, acceptance criteria, and environment constraints.
- **Benchmark adapters:** project-owned integrations that expose a consistent execution interface.
- **Runner:** controlled execution, resource capture, timeout handling, and artifact collection.
- **Result store:** immutable run records linked to source revisions and benchmark versions.
- **Validation:** schema checks, correctness checks, reference comparisons, and reproducibility checks.
- **Comparison and reporting:** regression analysis, scaling curves, dashboards, and exportable reports.

These are architectural targets, not claims that all components already exist.

## Repository layout

```text
TensorBench/
├── README.md
├── LICENSE
├── LICENSE_SCOPE.md
├── docs/
│   ├── ARCHITECTURE.md
│   └── BENCHMARK_SPEC.md
└── schemas/
    └── benchmark-result.schema.json
```

The layout will evolve as executable components are implemented.

## Quick start

Python 3.10+ is required. From the repository root:

```bash
python -m pip install -e ".[test]"
tensorbench run --spec examples/hello-benchmark.json --output tensorbench-runs
python -m pytest -q
```

The runner writes a `*.result.json` record and separate stdout/stderr logs for each run. The example is only a smoke test of the runner; it is not a scientific benchmark and its output must not be cited as performance evidence.

The first runner deliberately executes a declared argv command locally with `shell=False`. It is not a sandbox for untrusted benchmark code. Run only workloads you trust, and use an appropriately isolated environment for external contributions.

## Development status

<mark>Foundation / specification stage</mark>

The repository currently establishes the project charter and initial contracts. It does not yet claim to provide a production benchmark runner, hosted result service, dashboard, or validated benchmark corpus.

Implementation should proceed in small, testable increments. Every capability must be backed by executable tests and real benchmark evidence; mock results must never be presented as measurements.

## Relationship to other Tensor projects

TensorBench is a shared evaluation and evidence layer, not a replacement for TensorPlatform or domain projects.

- **TensorPlatform** may orchestrate benchmark work and agents.
- **TensorBench** defines benchmark contracts, executes or coordinates repeatable evaluations, and stores evidence.
- **Tensor domain projects** provide adapters, workloads, reference solutions, and domain-specific correctness criteria.
- **TensorData** governs datasets and data assets; TensorBench references dataset identifiers and versions rather than assuming ownership of their rights.
- **TensorKnowledge** may consume validated benchmark evidence as structured knowledge, while TensorBench remains the source of benchmark run records.

## Contributing

Contributions should include:

- a clearly stated problem and intended behavior;
- tests for changed code or schemas;
- benchmark evidence for performance-related claims;
- provenance and licensing information for imported workloads and datasets;
- documentation of limitations and known comparability constraints.

Do not add third-party benchmark suites, datasets, papers, model weights, or reference implementations without recording their provenance and applicable terms.

## License

The TensorBench software and project-authored documentation are licensed under Apache-2.0. See [LICENSE](LICENSE) and [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the scope and exclusions.
