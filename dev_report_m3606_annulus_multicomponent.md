# TensorBEM m3606: 多连通（圆环）Dirichlet 验证 —— 清 `verification.md` 最后一格欠账

**分支**：`agent/m3606`　**任务**：tp-ver-7 自主工时（verifier 席）
**证据等级**：REAL（全部数字由 `python -m tensorbem.benchmarks.multicomponent_annulus` 生成，
文档由 `docs/m3606/build_doc.py` 从 JSON 注入，零手抄）

## 1. 欠账来源（为什么是这一件）

`docs/verification.md` §1 的二维 Dirichlet 对照表里只剩一格没闭合：

```
| 圆环径向导热，u = A log r + B | 多连通边界、内外法向 | 后续：需多边界组件 |
```

其余每一例（圆域高阶调和、椭圆域、方形域分离变量、L 形凹角、混合边界方形）都是**单连通**域。
也就是说：**"两个边界环、其中一环的法向指向洞内"这种拓扑，在库里既无实现也无解析验证**。
根因在几何层而非求解器层：`geometry.PanelMesh` 只存一个循环顶点数组，隐式闭合边是 `N-1 -> 0`，
硬塞第二个环会凭空造出一条跨环面元。求解器本身是 duck-typed，不需要改。

选它的另外两个理由：①纯轻量（二维面元 16–128/环，单核 45 s），与 load15≈120 的容量纪律相容；
②产出的是**可判定**的数字（闭式解误差 + 观测阶 + 逐环环量恒等式 + 负控），不是又一份文档。

## 2. 做了什么

| 文件 | 内容 |
|---|---|
| `src/tensorbem/multicomponent.py` | `MultiComponentMesh`（多环网格，**每环自成闭合**，面板法向为"流体域外法向"）、`build_multicomponent_mesh`（洞环自动反向 + 洞必须在外环内且互不重叠/嵌套）、`solve_multicomponent_dirichlet`、`evaluate_fluid_field`（**洞内/域外探针直接报错**）、闭式场 `log_field` / `power_field`（复势 $f(z)$，梯度取 $f'(z)$）、`run_annulus_case`、正负控 `run_wrong_orientation_control` |
| `src/tensorbem/benchmarks/multicomponent_annulus.py` | 细化研究 + 场云图 + 图件 + 程序化 markdown 表 |
| `tests/test_multicomponent.py` | 20 条测试（几何/法向/守卫/解析梯度/二阶收敛/逐环环量/负控/文档账） |
| `docs/annulus_multicomponent.md` | 报告正文（数字全部注入） |
| `docs/m3606/` | `outputs/annulus_study.json`、`outputs/annulus_field.json`、`outputs/annulus_tables.md`、`figures/annulus_verification.png`、`build_doc.py` |

求解器一行未改：解析面元积分、第一类算子 $Sq=(\tfrac12 I+H)u$、条件数护栏全部复用
`tests/test_solver.py` 已覆盖的实现。

## 3. 三个算例（同一圆环 $0.5\le r\le 1$）

* `log`：$u=A\log r+B$（登记的那个案例）。**逐环不守恒**：$\oint_{\rm outer}q\,ds=+2\pi A$、
  $\oint_{\rm hole}q\,ds=-2\pi A$，总通量为零。单环网格连这个记账都无法表达。
* `power1`/`power2`：$u=\mathrm{Re}(z^n+z^{-n})$，$n=1,2$。$z^{-n}$ 在环内调和、但奇点在原点
  （在流体之外），对洞边界的加载与外环同量级。

## 4. 结果（`annulus_study.json`）

128 面元/环时：

| 算例 | 通量 $L_2$ 相对误差 | 观测阶 |
|---|---:|---:|
| log | 1.761e-04 | 1.97 |
| power1 | 6.049e-04 | 2.00 |
| power2 | 5.240e-04 | 2.02 |

逐环环量（log，128/环）：外环 6.283673、洞环 −6.283207，对 $\pm2\pi A$ 相对误差 1.23e-04 / 1.97e-04；
$\oint q\,ds / \oint|q|\,ds = 3.71e-05$。条件数 271.06（128/环），相对残差 4.8e-16。

## 5. 两条诚实的负面观察

1. **残差抓不到洞方向反了**：把洞环不反向（法向指进流体）时，同一系统仍解到残差 3.7e-16，
   但通量误差 **1.413（141%）**，是正确网格（6.90e-04，$N=64$）的 **2049 倍**；
   条件数也完全一样（$S$ 与法向无关，只有双层 $H$ 在洞环上变号）。所以方向检查必须写进
   `MultiComponentMesh` 的类型约束，不能留给调用者。
2. **第二个环带来固定的条件数代价**：在**总面元数相同**下与单圆比较，
   128 面元：76.1（单环）vs 135.7（双环）；256 面元：152.2 vs 271.1 —— 比值恒为 1.78。

## 6. 未做 / 边界

* 只验证单洞圆环；`MultiComponentMesh` 接受多洞，但多洞参考解（如偏心双圆柱的双极坐标解）未做。
* 仍是代码验证（闭式解对表），不涉及真实船体剖面；包内仍无多连通**自由液面**求解器。

## 7. 复现

```
PYTHONPATH=src python -m tensorbem.benchmarks.multicomponent_annulus \
    --json docs/m3606/outputs/annulus_study.json --field-json docs/m3606/outputs/annulus_field.json \
    --figure docs/m3606/figures/annulus_verification.png --markdown docs/m3606/outputs/annulus_tables.md
python3 docs/m3606/build_doc.py
PYTHONPATH=src python -m pytest tests/test_multicomponent.py -q   # 20 passed
```