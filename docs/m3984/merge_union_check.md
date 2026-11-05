# PR #109 三方合并静默丢内容审计（机械可复算）

base=`d22b6ad` · main=`ecba798` · branch=`ca4ba73` · merge=`HEAD`

| 判据 | 结果 |
|---|---|
| `inventory/no-main-file-lost` | **PASS** |
| `inventory/no-branch-file-lost` | **PASS** |
| `inventory/merge-adds-only-declared-evidence` | **PASS** |
| `content/one-side-owned-files-intact` | **PASS** |
| `content/both-side-additions-survive` | **PASS** |
| `content/guard-inventory-both-sides` | **PASS** |
| `content/composition-is-declared` | **PASS** |
| `content/no-side-deletion-undone` | **PASS** |
| `integration/doc_claims-selects-claimed-row` | **PASS** |

总体：**PASS**

## 两处语义冲突的逐行裁决

- `docs/marine_bem_benchmarks.md` [branch] `| 基本解方法（MFS，配点网格 64 面元） | 2543.24 | \(7.57\times10^{-4}\) | 2708.57 | \(3.53\times10^{-3}\` → m3229 (main) renamed every MFS row to the generated 'N panels / S sources' label; the branch's rows keep their numbers under main's label
- `docs/marine_bem_benchmarks.md` [branch] `| 基本解方法（MFS，配点网格 128 面元） | 2544.69 | \(1.89\times10^{-4}\) | 2715.76 | \(8.82\times10^{-4}` → same label renaming, 128-panel row
- `docs/marine_bem_benchmarks.md` [branch] `| 基本解方法（MFS，配点网格 256 面元） | 2545.05 | \(4.72\times10^{-5}\) | 2717.55 | \(2.21\times10^{-4}` → same label renaming, 256-panel row (the row main does not have)
- `tests/test_documented_benchmarks.py` [main] `fine = elementary_log_integral(xi, length)` → the guard loop that main added is kept, with its loop variables renamed from the branch's outer-scope names; the closed form does not touch the panel primitive, which is the property main's assertion pins
- `tests/test_documented_benchmarks.py` [main] `coarse = composite_gauss(xi, h, length, pieces=1, order=24)` → same loop rewrite: positional vs keyword call, same rule and order
- `tests/test_documented_benchmarks.py` [main] `fine = composite_gauss(xi, h, length, pieces=400, order=20)` → same loop rewrite: 400 pieces, order 20, unchanged
- `tests/test_documented_benchmarks.py` [branch] `nodes, weights = np.polynomial.legendre.leggauss(20)` → main's referee takes (pieces, order) as parameters, so the branch's fixed 20-point rule becomes the default argument of the shared helper
- `tests/test_documented_benchmarks.py` [branch] `def composite_gauss(pieces: int) -> float:` → same helper, generalised so main's three smooth-panel configurations can reuse it instead of duplicating it
- `tests/test_documented_benchmarks.py` [branch] `errors = [abs(composite_gauss(n) - exact) for n in pieces]` → same convergence sweep (100/200/400/800 pieces, order 20)
- `tests/test_documented_benchmarks.py` [branch] `assert abs(composite_gauss(1) - exact) > 1.0e-4` → same single-straddling-panel defect assertion

## 两处跨侧联动接缝（一方独有文件被合并改动）

- `src/tensorbem/doc_claims.py`（main 独有）：主人版本无一行丢失（lost=[]）。理由：publishing the branch's third MFS row (256 panels) turned main's row-by-position parser into a silent mis-read: the last published MFS row would win the 'MFS_PANELS' claim.  The seam selects the row by the panel count in its own label. 由 `tests/test_ursell_doc_reproducibility.py` 覆盖。
- `tests/test_ursell_doc_reproducibility.py`（branch 独有）：主人版本无一行丢失（lost=[]）。理由：main renamed every MFS row to the generated '<N> 面元、<S> 源' label (m3229); the branch's guard parsed '<N> 面元' by column counting.  The seam re-points the regex at the generated label and adds the 256-panel row to MFS_DOCUMENTED, keeping the 'count columns, do not trust the label' intent. 由 `tests/test_ursell_doc_reproducibility.py` 覆盖。

## 复现

```
python docs/m3984/merge_union_check.py    # 退出码 0 = 全过
```

输出同时落在 `docs/m3984/outputs/merge_union_check.json`。
