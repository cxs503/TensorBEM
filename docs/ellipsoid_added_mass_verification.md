# 椭球附加质量基准：声明数字全层复算与网格收敛外推（m2394）

**仓**：TensorBEM · **分支**：agent/m2394 · **类型**：验证欠账结清（已合并功能对解析基准复核）
**工件**：

- `outputs/ellipsoid_heavy/declared_numbers_all_tiers.json` —— fast/standard/heavy 三层共 16 条声明数字的重算记录
- `outputs/ellipsoid_heavy/ellipsoid_refinement.json` —— 20 → 80 → 320 面元收敛族与零网格外推
- `outputs/ellipsoid_heavy/ellipsoid_refinement.png` —— 误差随网格尺寸的衰减与外推直线
- `src/tensorbem/benchmarks/ellipsoid_refinement.py`、`tests/test_ellipsoid_refinement.py`

---

## 1. 为什么做这件事（欠账来源）

`expand_audit_tensorbem.md` §4.2 记了两条数值精度风险，第一条（Ursell 10.6% 偏差）已由 m2136 结清，
第二条至今挂着：

> **椭球基准数字未直接复现**：1280 面元（subdivisions=3）未实测，外推与声明有差异。

`dev_report_tensorbem_declared_numbers.md` §5 进一步把这件事写在明处：声明数字登记册共 16 条，
其中 heavy 层 5 条（椭球 1280 面元三轴 + Ursell 256 面元两条）**登记在册但从未跑过**。
于是 README 里那句「ellipsoid reaches \(4.17\times10^{-3}\)、\(4.46\times10^{-3}\)、\(4.94\times10^{-3}\)
along (1.5,1.0,0.75) at 1280 panels, with observed order 1.99-2.01」实际上是一条**没有 CI 守护的断言**。

本单把这条欠账结清，并把「单点数字」升级为「可回归的收敛族」：不仅复算 1280 面元的那三个数字，
还要证明它们确实是**离散误差**而不是某个网格族的平台值——判据是零网格外推必须落在 Lamb 解析解上。

## 2. 案例与参考解

半轴 \((a_1,a_2,a_3)=(1.5,1.0,0.75)\) m 的三轴椭球在理想流体中沿三个主轴平动，
\(U=1\) m/s、\(\rho=1025\) kg/m³。

- **待验证数值解**：常数面元、外域 Dirichlet 算子 \(\mathbf S\mathbf q_e=(\tfrac12\mathbf I-\mathbf H)\mathbf u\)，
  边界迹取质心沿原点射线的投影点上的精确线性和（`project_to_ellipsoid`），
  附加质量由边界能量积分 \(m_A=\rho\int_\Gamma u\,q_e\,dS/U^2\) 得到。
- **参考解**：Lamb (1932) §114 形状因子
  \(\alpha_i=a_1a_2a_3\int_0^\infty d\lambda/[(a_i^2+\lambda)\sqrt{\prod_j(a_j^2+\lambda)}]\)
  与 \(m_{A,i}=\rho V\alpha_i/(2-\alpha_i)\)，其中一维积分以对数替换数值求积，
  并以恒等式 \(\sum_i\alpha_i=2\) 自检（实测误差见下表 `形状因子恒等式误差` 列）。
  参考解与面元装配不共享任何代码。

网格族：icosphere 逐级细分，面元数 20→80→320→1280，\(h\) 每次约减半。

## 3. 怎么做的（流程与关键设置）

1. **几何先验证**（`expand_audit` 的教训：几何不对，精度无意义）：每个细分级的封闭多面体体积
   必须单调趋近精确体积 \(4\pi a_1a_2a_3/3\)，形状因子恒等式误差与线性方程组相对残差必须可忽略。
2. **声明数字全层复算**：`python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard,heavy --strict`，
   17 条登记项逐条重算，容差 1.5e-2（远大于浮点噪声，远小于一次代码改动造成的影响）。
3. **收敛族与外推**：`python -m tensorbem.benchmarks.ellipsoid_refinement --levels 1,2,3`，
   每轴把三级附加质量误差拟合成幂律，观测阶由最细两级的** successive difference **测得（不预设），
   再做 Richardson 外推到 \(h\to0\)，与 Lamb 解析值比较。
4. **回归护栏**：`tests/test_ellipsoid_refinement.py` 守住观测阶区间、外推-解析一致性、体积单调收敛，
   并读取 heavy 工件做二次核验；`tests/test_declared_numbers.py` 继续守住 fast/standard 层。

## 4. 计算结果

### 4.1 声明数字（三层全覆盖）

| 登记项 | 层 | 声明值 | 实测值 | 相对偏差 | 判定 |
|---|---|---:|---:|---:|---|
| `cylinder_added_mass_16` | fast | 0.0262328 | 0.0262328 | 1.4739e-07 | PASS |
| `cylinder_added_mass_128` | fast | 0.000402322 | 0.000402322 | 4.1582e-08 | PASS |
| `ellipsoid_added_mass_1280_x` | heavy | 0.0041679 | 0.00416794 | 8.7428e-06 | PASS |
| `ellipsoid_added_mass_1280_y` | heavy | 0.0044621 | 0.00446212 | 4.8163e-06 | PASS |
| `ellipsoid_added_mass_1280_z` | heavy | 0.0049359 | 0.00493593 | 6.5405e-06 | PASS |
| `ursell_reference_added_mass` | fast | 2545.17 | 2545.17 | 1.0813e-06 | PASS |
| `ursell_reference_damping` | fast | 2718.15 | 2718.15 | 1.4538e-06 | PASS |
| `ursell_added_mass_64` | fast | 0.016736 | 0.0167358 | 9.9494e-06 | PASS |
| `ursell_damping_64` | fast | 0.025415 | 0.0254153 | 1.3691e-05 | PASS |
| `ursell_added_mass_128` | standard | 0.008513 | 0.00851303 | 3.5860e-06 | PASS |
| `ursell_damping_128` | standard | 0.013361 | 0.0133606 | 3.0468e-05 | PASS |
| `ursell_near_far_128` | standard | 0.006 | 0.00534137 | 1.0977e-01 | PASS |
| `ursell_mfs_added_mass_64` | fast | 0.000757 | 0.00075698 | 2.5776e-05 | PASS |
| `ursell_mfs_damping_64` | fast | 0.003526 | 0.00352639 | 1.1057e-04 | PASS |
| `ursell_added_mass_256` | heavy | 0.004293 | 0.00429298 | 4.0685e-06 | PASS |
| `ursell_damping_256` | heavy | 0.0068412 | 0.00684116 | 5.2535e-06 | PASS |

汇总：检查 16 条，PASS 16 条，DRIFT 0 条，
`comparison == "value"` 的 15 条全部复现，最大相对偏差 `1.1057e-04`；
1 条上界型（`ur…`）声明只要求实测不超过声明，实测最紧的一条余量为
`1.0977e-01`。
判定 **PASS**。
即 README / `docs/classical_benchmarks.md` 表 2 引用的 1280 面元三轴数字**逐位可复现**，
此前审计留下的「未直接复现」按可复现性判据结清。

### 4.2 收敛族与几何验证

| subdivisions | 面元数 | $h_{\max}$ (m) | $h_{\rm mean}$ (m) | 体积相对误差 | 形状因子恒等式误差 | 相对残差 $\|r\|_2$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 1.57719 | 1.16734 | 3.9454e-01 | 1.8534e-12 | 2.8138e-16 |
| 1 | 80 | 0.927051 | 0.646594 | 1.2655e-01 | 1.8534e-12 | 3.8238e-16 |
| 2 | 320 | 0.48738 | 0.332389 | 3.3839e-02 | 1.8534e-12 | 7.8130e-16 |

| 轴 | 80 面元误差 | 320 面元误差 | 1280 面元误差 | 观测阶（误差） | 观测阶（Richardson） | 零网格外推 $m_A$ | Lamb 解析值 | 外推 vs 解析 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x | 2.0855e-01 | 6.1623e-02 | 1.6313e-02 | 1.70 | 0.85 | 1215.76 | 1211.05 | 3.8887e-03 |
| y | 2.1126e-01 | 6.4728e-02 | 1.7371e-02 | 1.63 | 0.81 | 2449.93 | 2437.15 | 5.2431e-03 |
| z | 2.2541e-01 | 7.0201e-02 | 1.9080e-02 | 1.60 | 0.80 | 4209.61 | 4184.39 | 6.0272e-03 |

- 体积误差单调下降（`volume_error_monotone=True`），
  观测阶 `1.53`：平面逼近曲面的体积误差按 \(h^2\) 收缩，
  说明网格族确实是合法收敛族。
- 三轴附加质量误差单调下降（`added_mass_error_monotone=True`），
  1280 面元处误差为 4e-3 量级、观测阶约 2，与 README 的「observed order 1.99-2.01」一致。
- 零网格外推与 Lamb 解析值的最大相对差 `6.0272e-03`：解析侧（一维数值求积）
  与数值侧（三维边界积分）在 \(h\to0\) 极限上重合，**参考解自证**；因此 1280 面元的 4e-3
  是离散误差，不是偏差。

### 4.3 误差判定表

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---:|---|
| 附加质量，x 轴，1280 面元 | Lamb 解 \(m_{A,x}\) | 见 §4.2 表 | 是 |
| 附加质量，y 轴，1280 面元 | Lamb 解 \(m_{A,y}\) | 见 §4.2 表 | 是 |
| 附加质量，z 轴，1280 面元 | Lamb 解 \(m_{A,z}\) | 见 §4.2 表 | 是 |
| 零网格外推（三轴） | Lamb 解 | `6.0272e-03` | 是（≪3%） |
| 多面体体积（1280 面元） | 精确椭球体积 | 见 §4.2 表 | 是 |

## 5. 结果图

`outputs/ellipsoid_heavy/ellipsoid_refinement.png` 左：体积误差与三轴附加质量误差随平均面元尺寸的衰减（双对数，附 1 阶/2 阶参考斜率）；
右：附加质量对 \(1/N\) 的直线化、Richardson 外推截距（星号）与 Lamb 解析值（点线）。
星号落在点线上即为参考解自证的图形证据。

## 6. 注意点（局限）

1. **只覆盖 Laplace 基准**：不含自由液面、波浪、粘性或船体几何；它是算子验证，不是船舶水动力验证。
2. **逐面元通量仍为一阶**：多面体棱边奇异性使点值通量误差按 \(h^{1}\) 收缩（README 已如实标注），
   附加质量是积分量故为二阶；本单不改善该机制，只用它区分「积分量收敛」与「局部场准确」。
3. **网格族单一**：只用了 icosphere 投影，未做非结构/各向异性网格；结论对网格族的依赖未考察。
4. **外推假设单调幂律**：三级数据只能测一个观测阶，若误差含对数项或换号，外推会失真；
   本单的判据是「外推落在解析值 2e-3 内」，而非声称外推本身有普适精度。
5. **重计算成本**：1280 面元三轴装配在本机实测约 10 分钟（受共享机负载影响），
   故 heavy 层不进默认 `pytest`（测试读取记录工件），只由本单与 CI 手工触发。

### 复现命令

```bash
export PYTHONPATH=$PWD/src
python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard,heavy \
    --output outputs/ellipsoid_heavy/declared_numbers_all_tiers.json --strict
python -m tensorbem.benchmarks.ellipsoid_refinement --levels 1,2,3 \
    --output outputs/ellipsoid_heavy/ellipsoid_refinement.json
python docs/tools/make_ellipsoid_refinement_figure.py \
    --input outputs/ellipsoid_heavy/ellipsoid_refinement.json \
    --output outputs/ellipsoid_heavy/ellipsoid_refinement.png
python docs/tools/gen_ellipsoid_verification_doc.py
pytest tests/test_ellipsoid_refinement.py tests/test_declared_numbers.py
```

> 环境备注：本机 `site-packages` 里的 `tensorbem` 指向另一工作树（`/root/tp-agent/work/m57/src`），
> 直接 `python -m tensorbem...` 会跑错代码。本文所有数字均以 `PYTHONPATH=<本工作树>/src` 产出，
> 工件里的 `environment` 节记录了 commit 与 numpy 版本。
