# m3556 — conclusion-class prose numbers become an executable contract

The four result tables of `docs/three_dimensional_bem.md` are pinned by
`three_dimensional_doc_pins` (§11). Everything a *reader quotes*, however, sits
outside those tables: the abstract, §5.5, §6.4, §7 and §9 restate the same three
headline facts in 10 independently worded sentences. This mission binds
them to measurements with `tensorbem.benchmarks.prose_claims`.

Generated output — do not edit by hand, re-run `build_artifacts.py`:

* `prose_claims.json` — full `--check` verdict plus the coverage delta;
* `figures/prose_claims.png` — left: both surface-velocity paths with the quoted
  band shaded and the quoted levels ringed; right: the two near-body probes from
  a live 80-panel solve;
* the table below, rendered from the measurement side.

## Verdict

| quantity | value |
|---|---:|
| registered restatements | 10 |
| distinct printed numbers bound | 23 |
| qualitative claims checked | 2 |
| drift | 0 |
| overall | **PASS** |

Coverage of the document's numeric tokens: tables protect
86 of 249; this ledger adds
7 distinct quoted numbers
(53, 65, 0.518, 1.000, 48, 0.984, 1.6), i.e. unguarded drops from
163 to 156
(guarded fraction 0.373). It is **not** a claim that the
document is fully verified: derived tokens and formula coefficients remain
unguarded, and the honest statement stays "these quoted numbers are now counted".

## Claim ledger

| # | 节 | 锚句（文档原文片段） | 印出值 | 实测值 | 相对偏差 | 容差 | 判定 |
|---:|---|---|---|---:|---:|---:|---|
| 1 | 摘要 | `表示公式路径在紧贴表面处重构误差达 53%–65%` | 53 | 53.1618 | 3.05e-03 | 6.0e-02 | PASS |
| 1 | 摘要 | `表示公式路径在紧贴表面处重构误差达 53%–65%` | 65 | 65.3139 | 4.83e-03 | 6.0e-02 | PASS |
| 2 | 5.5 | `在面元质心（配点）上重构的速度有 53%–65% 的误差` | 53 | 53.1618 | 3.05e-03 | 6.0e-02 | PASS |
| 2 | 5.5 | `在面元质心（配点）上重构的速度有 53%–65% 的误差` | 65 | 65.3139 | 4.83e-03 | 6.0e-02 | PASS |
| 3 | 5.5 | `速度为 0.518 而正确值为 1.000（相对误差 48%）` | 0.518 | 0.518017 | 3.31e-05 | 6.0e-03 | PASS |
| 3 | 5.5 | `速度为 0.518 而正确值为 1.000（相对误差 48%）` | 1.000 | 1 | 0.00e+00 | 6.0e-04 | PASS |
| 3 | 5.5 | `速度为 0.518 而正确值为 1.000（相对误差 48%）` | 48 | 48.1983 | 4.13e-03 | 6.0e-02 | PASS |
| 4 | 5.5 | `在体心处则为 0.984（1.6%）` | 0.984 | 0.984447 | 4.54e-04 | 6.0e-03 | PASS |
| 4 | 5.5 | `在体心处则为 0.984（1.6%）` | 1.6 | 1.55529 | 2.79e-02 | 6.0e-02 | PASS |
| 5 | 6.4 | `停在 53%–65% 且几乎不降` | 53 | 53.1618 | 3.05e-03 | 6.0e-02 | PASS |
| 5 | 6.4 | `停在 53%–65% 且几乎不降` | 65 | 65.3139 | 4.83e-03 | 6.0e-02 | PASS |
| 6 | 7 | `重构误差为 53%–65% 且不随加密下降` | 53 | 53.1618 | 3.05e-03 | 6.0e-02 | PASS |
| 6 | 7 | `重构误差为 53%–65% 且不随加密下降` | 65 | 65.3139 | 4.83e-03 | 6.0e-02 | PASS |
| 7 | 7 | `紧贴表面误差 48%，体心处 1.6%` | 48 | 48.1983 | 4.13e-03 | 6.0e-02 | PASS |
| 7 | 7 | `紧贴表面误差 48%，体心处 1.6%` | 1.6 | 1.55529 | 2.79e-02 | 6.0e-02 | PASS |
| 8 | 9 | `停在 53%–65%` | 53 | 53.1618 | 3.05e-03 | 6.0e-02 | PASS |
| 8 | 9 | `停在 53%–65%` | 65 | 65.3139 | 4.83e-03 | 6.0e-02 | PASS |
| 9 | 9 | `紧贴表面误差 48%，体心处 1.6%` | 48 | 48.1983 | 4.13e-03 | 6.0e-02 | PASS |
| 9 | 9 | `紧贴表面误差 48%，体心处 1.6%` | 1.6 | 1.55529 | 2.79e-02 | 6.0e-02 | PASS |
| 10 | 11 | `如 53%–65%、48%、1.6%` | 53 | 53.1618 | 3.05e-03 | 6.0e-02 | PASS |
| 10 | 11 | `如 53%–65%、48%、1.6%` | 65 | 65.3139 | 4.83e-03 | 6.0e-02 | PASS |
| 10 | 11 | `如 53%–65%、48%、1.6%` | 48 | 48.1983 | 4.13e-03 | 6.0e-02 | PASS |
| 10 | 11 | `如 53%–65%、48%、1.6%` | 1.6 | 1.55529 | 2.79e-02 | 6.0e-02 | PASS |

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m tensorbem.benchmarks.prose_claims --check       # DRIFT -> exit 1
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m tensorbem.benchmarks.prose_claims --coverage
PYTHONPATH=src python -m pytest tests/test_prose_claims.py -q
```
