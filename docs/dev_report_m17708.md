# m17708 · CASE-BEM-ANNULUS-LAPLACE 第 2 轮：偏心环形域（双极坐标闭式解）

**仓** TensorBEM · **分支** `agent/m17708` · **锚点** `CASE-BEM-ANNULUS-LAPLACE`
（算例池·环形域 Laplace（多连通））· **证据等级** REAL

## 1. 选题（为什么不重复第 1 轮）

总账查得该锚 `open`；仓内已有第 1 轮实现（m17178，`case_annulus_laplace.py`，**同心**双环
+ `log r` / `(r+c²/r)cosθ` 两个闭式解，已过 3% 判据）。所以本轮不重做同心情形，而是补它显式留下的
验证盲区：**同心**是两条平行边界、间隙恒定，最温和的多连通配置。本轮换成**偏心**双环
（双极坐标族的两条圆），几何非平行、间隙沿周向变化，同时保持「奇点全在流体外」使闭式解仍精确。

## 2. 增量

| 文件 | 内容 | 性质 |
|---|---|---|
| `src/tensorbem/multiloop.py` | `eccentric_annulus_geometry` / `eccentric_annulus_mesh`（+ 内接多边形互不重叠守卫） | 库级新能力 |
| `src/tensorbem/benchmarks/eccentric_annulus.py` | 三算例闭式解、合法性门、梯队、两项敏感性、两负控、verdict、报告/四图渲染、CLI | 算例池第 2 轮 |
| `tests/test_case_annulus_laplace_eccentric.py` | 22 条验收测试 | 验收 |
| `benchmarks/verified/bem_annulus_laplace_eccentric/` | `results.json` / `report.md` / `figures/{convergence,clearance_sweep,field}.png` | 机器产物 |
| `docs/case_annulus_laplace.md` | 追加第 2 轮章节 | 文档 |
| `src/tensorbem/benchmarks/case_pool_registry.py` | 该算例的 artifacts/figures 追加偏心产物 | 登记 |

求解器本体一行未改（复用 `solve_laplace_multiloop` / `evaluate_multiloop_field` / `boundary_flux`）。

## 3. 验证（跑了什么）

```
PYTHONPATH=src python -m tensorbem.benchmarks.eccentric_annulus \
    --outdir benchmarks/verified/bem_annulus_laplace_eccentric --check
# -> verdict.passed = true, checks 37, failed 0

PYTHONPATH=src python -m pytest tests/test_case_annulus_laplace_eccentric.py \
    tests/test_case_annulus_laplace.py tests/test_multiloop_annulus.py \
    tests/test_multicomponent.py -q
# -> 67 passed
```

预算 64/32 面元：q 相对误差（L2/Linf）2.3e-3 / 2.1e-3、内部场 6.2e-4、通量 1.3e-3，均远低于 3%；
拟合阶 1.96 / 2.18（二阶）。逐环通量对解析 `±2π`，两环和 / ∫|q|ds = 3.2e-4。

## 4. 三条诚实记录

1. **自己的 gate 抓到自己的错**：第一版把 `bipolar_ratio` 的理论常数写成 `−2τ`，被常数性门判 FAIL（偏差 0.5）；
   `τ = ½log(ρ+/ρ−)`，正确常数是 `−τ`。门有效。
2. **「薄间隙一定炸」是错的（本轮实测）**：固定预算下误差对间隙是 U 形，最坏不是最薄处；
   最薄处先失败的是**探针协议/几何可表示性**（内接多边形重叠 → 守卫拒绝），不是积分精度。已加守卫，
   把会静默返回乱数的装配变成显式 ValueError。
3. **`page_or_article_number` 仍 `WITHHELD`**：容器内拿不到 Lamb / Milne-Thomson 的条目号，不编页码；
   改用「常数性 + 调和性」两把机器门替代引文，残差 `≤ 6e-15`。

## 5. 复现要点

`_bipolar_circle` 曾用 `centre + r*(cos,sin)` 广播错误（圆心 x 被加到了两列），
由探针域守卫（`domain_membership`）当场暴露为 39/48 探针"在外环外"——域守卫不只是防御，也是几何装配的单元测试。
