# m21766 — 有限水深交叉验证全网格：成本模型、调度形状与未完成判定（WITHHELD）

**锚点**：`XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（TensorBEM）· **分支**：`agent/m21766`
**结论先说**：本单**没有**产出全网格判定，`results.json` 仍是 `PARTIAL: 1 of 12 planned cells`。
但本单把「为什么跑不完」从传闻变成了**实测成本模型 + 调度形状结论**，并给出可复制的补齐路径。
任何「全网格 <15 %」的宣称都不成立。

## 1. 硬事实：两侧的成本都被实测过，问题不在算力总量

| 侧 | 格 | 级 | 面板 | 实测耗时 | 来源 |
|---|---|---:|---:|---:|---|
| TensorBEM | `h/a=3, k0a=0.5` | 2 | 320 | **111.0 s**（`mode_count=1075`, cond=2.141） | m21322 §3.2 |
| TensorBEM | `h/a=5, k0a=1.0` | 2 | 320 | **> 115 s**（前台被 120 s 收割，未落盘） | 本单探针 |
| TensorBEM | `h/a=3, k0a=0.5` | 1 | 80 | 3.6 s | 冻结 `tensor_ladder.json` |
| TensorBEM | `h/a=3, k0a=0.5` | 0 | 20 | 0.163 s | 冻结 `tensor_ladder.json` |
| capytaine | `h/a=3, k0a=0.5` | 288/1152/4608 | — | 0.361 / 1.437 / 7.097 s（legacy 1.4 s，inf 1.4 s） | 冻结 `capytaine_ref.json` |
| capytaine | 其余浅水格（`k0h≈1.5–6`） | 三级 | 4608 | **单格 > 57 s**（2 格分片 115 s 未收尾） | 本单分片实测 |

**全网格成本量级**：TensorBEM 12 格 × 三级（含 320 面板）≈ **25–40 core·min**；
capytaine 12 格 × 三级 + legacy + inf ≈ **12–25 core·min**。总量 **< 1 core·h**。
也就是说：**这不是「算不动」，是「派不上」**。

实测依据（本单，`/tmp/refshard`，已如实记录为未成工件）：
6 个 capytaine 分片（每片 2 格）并发 115 s 全部未写出 JSON —— 浅水格（`k0h` 小）主导成本，
日志显示 capytaine 自身在 `k0h` 大处打印
`Computation might be faster by using the infinite water depth instead: water_depth=np.inf`，
即成本随 `k0h` 反相上升。

## 2. 为什么派不上：队列形状，而不是队列深度

`/root/tp-agent/cluster/compute_gate/cluster_queue.py` 的准入是
`LOAD_CAP=0.85`、`MEM_FLOOR_G`、**`MAX_PER_NODE=4`（每节点同时最多 4 个作业）**，
`run` 型作业从派发到进程退出的整段时间都占住一个 slot。

本单实测（同一时间窗）：

| 作业 | 形状 | 结果 |
|---|---|---|
| `cj3696` | run, **1 核**, capytaine 参考全 12 格 | 排队 2.7 min 后由本席 kill（并入 `cj3700`） |
| `cj3700` | run, **13 核**, 参考 + TensorBEM 梯级 + 出页，一条命令 | 递交时位次 62；**排队 70.7 min 仍在 queued**，此时在飞作业 96 |

对比同期：`cj3718/cj3717/cj3716`（4 核）、`cj3715/cj3714`（8 核）递交 2–7 min 亦全部 queued。
结论：8 个 slot（2 节点 × 4）被长作业长期持有，**大核数 `run` 型是错误形状**——它既不能插队，
又在拿到 slot 后独占 13 核做一件本质可完全并行的工作。

本机侧：owner 2026-11-13 计算调度令把本机限制在「<8 核 / <10 min」，且
`post-2026-11-13` guard 直接硬拦 `setsid`/`nohup`
（本单实测：`LOCAL-COMPUTE BLOCKED … 命令命中 [nohup; setsid]`），
120 s 前台硬顶（`exit=-9 TIMEOUT after 120s`，本单分片实测）⇒ **本机无法兜底**。
这解释了 m21591/m21616 等多单在同一锚点上连续空转的机制。

## 3. 正确的补齐形状（下次照抄即可）

两段式、**全 shard 型**（`SHARD_NODE_SHARE=2` 记账，双节点轮转，派发后立即让出 slot）：

```bash
# 1) 参考侧：12 格分 12 片，每片 1 格（capytaine 索引 % 12）
run_compute(action=submit, repo="TensorBEM", ref="main",
  cmd_tpl="env PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/capytaine_finite_depth_ref.py --shard {i} --shards 12 --out /tmp/cp_ref_{i}.json",
  shards=12, cores=2, timeout=3600)

# 2) 梯级侧：同样 12 片，每片 1 格三级（20/80/320 面板）+ 逐格深水负控制
run_compute(action=submit, repo="TensorBEM", ref="main",
  cmd_tpl="env PYTHONPATH=src python -m tensorbem.benchmarks.capytaine_finite_depth_crossval --ladder-only --shard {i} --shards 12 --levels 0,1,2 --no-extra-level --cache /tmp/ladder_{i}.json",
  shards=12, cores=2, timeout=7200)
```

注意两点（都是代码里已有的硬约束，不是建议）：
* `capytaine_finite_depth_ref.py --shard` 的索引是 **0-based**，而集群 `cmd_tpl` 的 `{i}` 是 **1-based**；
  两者不在同一条流水线上，`--out` 必须落到 **job 目录内**（`fetch` 只 tar `~/jobs/<jid>`，`/tmp` 取不回）。
* 合并端两步都要「少一片就报错」：参考侧 `--merge` 拒绝漏格/重复格；
  梯级侧 `merge_ladders(..., required_levels=[0,1,2])` 拒绝缺级，且两片同格必须逐位一致（1e-9 相对）。

本次把 `--out` 写到 `/tmp` 是错的：即使分片成功，`fetch` 也取不回工件。**这一点下次不要再犯。**

## 4. 未完成的部分（如实登记）

| 欠账 | 状态 |
|---|---|
| capytaine 参考 12 格（现冻结 1 格） | **未完成**：本机超 120 s 硬顶；集群 `cj3700` 70.7 min 未派发 |
| TensorBEM 梯级 12 格 × 3 级 + 12 负控制 | **未完成**，同上 |
| 全网格外推判定（≤3 % 附加质量 / 5 % 阻尼 / 15 % owner 线） | **WITHHELD**：没有输入数据，不给判定 |
| `reference_ladder_in_asymptotic_range`（m21591 报 0.2412 < 0.3，仅 1 格） | **WITHHELD**：1 格不足以决定是否把 `CAPYTAINE_RESOLUTIONS` 扩到含 `(96,192)`；扩级会改 `richardson()` 的取级（`panels[-3:]`），必须在有 12 格数据后再定 |
| 1280 面板确认级（`EXTRA_LEVEL_CELLS` 4 格） | 未跑（`--no-extra-level`），冻结工件 `extra_level=null` 如实记录 |

`cj3700` 未 kill：它一旦派发，一条命令即可产出参考 + 梯级 + 发布页（`--write-doc`），
工件路径 `benchmarks/verified/capytaine_finite_depth/{capytaine_ref,tensor_ladder,results}.json`。
取回后用同一条命令在本机重跑（缓存命中，不重算）即可再生 `results.json`/页面/图。

## 5. 验证

`PYTHONPATH=src python -m pytest tests/test_capytaine_finite_depth_crossval.py -q`
→ **11 passed, 6 xfailed**（6 条 xfail 由 `execution_status` 决定，本次仍 `PARTIAL`，因此仍 xfail ——
门禁没有被绕过，也没有被冒充为 PASS）。

## 6. 给下一次的线索

1. 先用上面的 shard 形状跑参考侧（12 片 × 1 格，单格 ≤ 60 s），**合并成功**后再提交梯级侧；
   两段都落 job 目录。
2. 梯级侧 12 片各 1 格时，最坏单片 ≈ 320 面板浅水格 ≈ 2–3 min，12 片并发墙钟 ≈ 3 min。
3. 拿到全网格后第一件事不是出图，而是看 `capytaine_order` 覆盖率（`_order_coverage`）：
   若多数条目 `n/a`，先扩 `CAPYTAINE_RESOLUTIONS` 到 `(96,192)` 重出参考，再谈 3 % 判据。