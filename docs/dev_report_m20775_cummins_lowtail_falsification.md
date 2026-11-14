# CASE-BEM-CUMMINS-WIGLEY — 低频尾带不是 KK 残余的来源（否定结论）+ 扩展带管线改为可分片续跑

锚点 `CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核 6DOF；ref = Cummins 方程 + Kramers-Kronig 因果；容差 <10 %），任务 m20775，分支 `agent/m20775`。

## 1 案例介绍

Wigley III：L=3.0 m，B=0.3 m，T=0.1875 m，Cb=0.543，ρ=1025 kg/m³，g=9.80665 m/s²，无限水深，旋转中心 (0,0,−T/2)。判据为 Cummins 方程的 Hilbert 对

```
B(ω) = −(2/π) PV∫₀^∞ s²(A(s)−A∞)/(s²−ω²) ds        A(ω) = A∞ + (2/π) PV∫₀^∞ B(s)/(s²−ω²) ds
```

与五道预声明闸门 G1–G5（交叉求解器 10 %、K(0) 2 %、KK 10 %、自由衰减 10 %、往返 5 %）。

## 2 怎么做的（两件事，均不新起面板重算）

**(a) 把「残余来自低频尾带」这一归因做成可判定实验。** round-2/3/4 反复把 20–80 % 的 G3 残余归给频带截断（尤其未解的低频段）。该归因可在**有真值**的合成对上定量：指数记忆核 `K(t)=Σ c_j e^(−a_j t)` 有闭式对

```
B(ω) = Σ c_j a_j/(a_j²+ω²),    A(ω) − A∞ = −Σ c_j/(a_j²+ω²)
```

系数用最小二乘约束到 `Σ c_j/a_j = 0`，使 `B ~ ω^q`（三维辐射体的低频起点），在**预声明的** `EXTENDED_GRID [0.10, 20]` 上采样，比较出货闭包（`s_min` 以下 `B` 取常值）与「用拟合幂律 `B = C s^q` 把子带精确积分」两种闭合的残差，均对解析 `A(ω)` 逐点取真值。脚本 `scripts/cummins_wigley_kk_lowtail_bound.py`，产物 `docs/case_bem_cummins_wigley_lowtail_bound.json`（报告数字由该 JSON 程序化注入）。

**(b) 扩展带管线改为可分片续跑。** rounds 2–5 的扩展带扫频都丢在「一个长顺序作业跑到预算耗尽」。`scripts/run_cummins_wigley_closure_full.py` 新增 `--stage solve|gates`、`--grid extended|closure`、`--shard/--shards`：求解阶段可切成 n 个短作业填同一份点缓存（可续跑），`gates` 阶段**绝不插值、绝不重解**，缺节点即拒判，不允许悄悄缩带。

## 3 计算结果（程序化注入，零手抄）

合成对：低频拟合指数 q = 2.173（幂律系数 0.7774），`Σc/a = 5.3e-15`，频带 0.1–20.0 rad/s（28 节点），归一化尺度 max|A| = 2.734，B 在最低节点的值 0.005244。

| 闭合方式（同一份合成数据、同一组采样点） | 最大相对残差 | RMS 相对残差 |
|---|---|---|
| shipped closure (B held constant below s_min) | 0.3444% | 0.2273% |
| exact sub-band integral from fitted B = C s^q | 0.4348% | 0.2272% |

## 4 结果图

本轮是归因实验 + 管线改造，没有新的场量云图：合成对的全部结论就是上表两行（两种闭合相差 <0.1 个百分点），画图不增信息，故不附。面板数据的 A/B(ω) 与 KK 逐点残差曲线须由扩展带工件生成，工件未回传，本轮不预造图。

## 5 注意点

* **主结论是否定**：在有真值的合成对上，出货闭包最大残差仅 0.344%；把低频子带**精确积分**后为 0.435%（略差——拟合幂律的偏置大于常值闭合的偏差）。因此「低频尾带」最多解释 **0.4 个百分点**量级，与面板数据 20–80 % 的残余差 2–3 个数量级：G3 残余**不可能**由低频尾带解释。这与 round-5（m20594）把失败模式定位到 `A∞` 估计器（剖面/Chebyshev 中心校准后六 DOF 全部 <10 %）方向一致；本轮独立贡献是给出该归因的**定量上界**。
* 合成对 `B` 的高频尾部为 `p = 2`，`tail_fit=False`（幂尾拟合对 `p ≤ 2` 本就判无效），故本轮不检验高频尾——高频尾仍是未排除的候选（round-4 的扩展带正是为它准备的）。
* 分片管线只解决「跑不完」，不解决「残差是否真实」；G1–G5 在扩展带上**仍未重算**。

## 6 误差判定（逐项，显式回答是否 <3 %）

| 量 | 基准 | 误差 | <3 %? |
|---|---|---|---|
| 低频指数 q（拟合 vs 构造的 ω² 起点） | 解析 q = 2 | 8.653% | 否 |
| 出货 KK 闭包（合成对，有真值） | 解析 A(ω) | 0.344% | 是 |
| 精确低频尾闭包（合成对，有真值） | 解析 A(ω) | 0.435% | 是 |
| 面板数据 G1–G5（扩展带） | — | 未计算 | 不适用（无数字，不声明） |

**honest 状态：锚点仍 open。** 本轮既没有把 G1–G5 推到全绿，也没有把任何 FAIL 改成 PASS。

## 7 未完成与续跑指令

扩展带 medium（28 节点）与 coarse（28 节点）分片扫频已递交集群：**cj2918**（medium，8 分片）/ **cj2919**（coarse，8 分片）；递交时队列 25 作业在飞、位次 4–5，本单预算内未回传，故本轮无面板数字。回传后三步闭合（先解共振簇，再判，再跑测试）：

```
PYTHONPATH=src python3 scripts/run_cummins_wigley_closure_full.py --stage solve --grid closure --level medium --shard i --shards 8
PYTHONPATH=src python3 scripts/run_cummins_wigley_closure_full.py --stage gates --level medium
python3 -m pytest tests/test_cummins_wigley_closure_pipeline.py -q      # 5 passed
```

## 8 测试（本机实跑）

`tests/test_cummins_wigley_closure_pipeline.py`（新，5 passed）锁三条不变式：分片**恰好划分**网格（无丢节点/无重复，非法分片参数报错）、求解阶段只解自己切片且能从缓存续跑（二次运行 `solved_now = 0`）、`gates` 阶段遇缺节点**拒判**且不写 `results.json`。
`tests/test_cummins_wigley_lowtail_bound.py`（新，2 passed）锁本轮否定结论：合成对低频指数落在 (1.5, 2.6)、两种闭合残差都 <3 % 且相差 <1 个百分点、`lower_tail_is_the_culprit = False`。
回归：`tests/test_cummins_wigley_kk_causality.py` + `tests/test_case_bem_cummins_wigley.py` 同轮实跑，合计 **51 passed, 1 skipped**（该 skip 仍在等生产扫频工件）。
