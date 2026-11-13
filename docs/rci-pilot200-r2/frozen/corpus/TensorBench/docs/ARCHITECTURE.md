# TensorBench Architecture

## 1. Architectural goal

TensorBench provides a common evidence pipeline for heterogeneous benchmarks without forcing every project into one implementation language, runtime, or scientific methodology.

The core flow is:

```text
Benchmark specification
        |
        v
Project adapter ----> Workload + inputs + configuration
        |                         |
        +------------+------------+
                     v
              Controlled runner
                     |
          +----------+----------+
          |                     |
          v                     v
    Raw measurements       Execution artifacts
          |                     |
          +----------+----------+
                     v
              Result record
                     |
                     v
       Validation and reproducibility
                     |
                     v
        Comparison / report / export
```

## 2. Logical components

### 2.1 Benchmark registry

Stores versioned benchmark specifications and their lifecycle state. A benchmark must have a stable identifier, owner, domain, version, workload definition, metrics, and acceptance criteria.

### 2.2 Workload and dataset references

A workload defines the task and its inputs. Large datasets and external assets should be referenced by stable identifiers, versions, checksums, and provenance records. TensorBench should not duplicate or assume ownership of TensorData assets.

### 2.3 Adapters

An adapter connects a project or external implementation to the common execution contract. Adapters are responsible for translating benchmark inputs and configuration into the target application's invocation, then collecting outputs and measurements.

Adapters must not conceal failed runs, silently alter inputs, or substitute mock outputs.

### 2.4 Runner

The runner is responsible for controlled invocation and collection of:

- start/end timestamps and exit status;
- command or invocation metadata, with secrets removed;
- source revision and dirty-tree state;
- hardware and operating-system metadata;
- runtime, memory, and resource measurements where available;
- logs, output files, and checksums;
- timeout, crash, and infrastructure failure information.

The runner should be designed for local and cluster execution. Distributed scheduling and remote execution are later capabilities, not prerequisites for the first vertical slice.

### 2.5 Result store

A result record is immutable evidence for one execution. Corrections should create a new record or an explicit superseding relation; they should not silently rewrite historical measurements.

The result record links to the benchmark specification version, implementation revision, workload/data version, configuration, environment, metrics, artifacts, and validation state.

### 2.6 Validation

Validation is layered:

1. **Structural:** result conforms to the schema.
2. **Execution:** run completed and artifacts are internally consistent.
3. **Correctness:** output satisfies benchmark-specific acceptance criteria.
4. **Numerical/scientific:** error and physical or domain constraints are within declared limits.
5. **Reproducibility:** independent rerun meets declared tolerances.
6. **Comparability:** conditions are sufficiently aligned for the requested comparison.

A result can pass one layer and fail another. Do not collapse all validation into a single unqualified "passed" label.

### 2.7 Analysis and reporting

Reports may calculate speedup, efficiency, scaling, error, memory ratios, and regression deltas. Every derived metric must identify its source runs and calculation method.

Comparisons must disclose hardware, software, precision, parallelism, and workload differences. Results from materially different conditions must not be presented as directly comparable without qualification.

## 3. Core entities

- **Benchmark:** stable identity and purpose.
- **BenchmarkVersion:** immutable specification revision.
- **Implementation:** software/project and source revision being evaluated.
- **Workload:** task and input definition.
- **DatasetReference:** versioned external or internal data asset.
- **Environment:** hardware, operating system, runtime, dependencies, and relevant configuration.
- **Run:** one execution attempt.
- **Metric:** measured or derived quantity with unit and aggregation semantics.
- **Artifact:** log, output, trace, profile, or other evidence.
- **Validation:** result of a named validation layer.
- **Comparison:** a declared analysis across compatible runs.

## 4. Trust boundaries

- Treat benchmark inputs, downloaded assets, adapter output, and logs as untrusted.
- Do not execute arbitrary benchmark code without isolation and resource limits.
- Never store API keys, credentials, private tokens, or unredacted secrets in result records.
- Separate benchmark metadata from potentially large artifacts.
- Preserve provenance and license metadata for all imported material.
- Mark generated or synthetic workloads explicitly.

## 5. Initial implementation sequence

1. Validate the result schema and add schema tests.
2. Implement a local, single-process runner for a minimal benchmark contract.
3. Capture environment, source revision, exit status, elapsed time, and artifacts.
4. Add correctness validation hooks and explicit validation states.
5. Add one real adapter from a Tensor project and verify it end-to-end.
6. Add baseline comparison and regression reporting.
7. Only then generalize to cluster execution, distributed scheduling, and dashboards.

## 6. Non-goals for the first increment

- Replacing TensorPlatform's agent orchestration or workbench.
- Replacing TensorData's dataset governance.
- Claiming a universal scientific correctness metric.
- Treating raw runtime as sufficient evidence of quality.
- Building a dashboard before the underlying run records are trustworthy.
