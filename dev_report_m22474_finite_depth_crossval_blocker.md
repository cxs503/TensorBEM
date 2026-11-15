# XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN 收口尝试与阻塞诊断（m22474）

**锚点** `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` · 仓 TensorBEM · 分支 `agent/m22474`

## 1. 开工核对

- 总账：`slug=XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`，`status=open`，`acceptance=<15%` → 可做，未被他单收口。
- 仓内现状（HEAD `4666634c`）：`benchmarks/verified/capytaine_finite_depth/`
  - `capytaine_ref.json`：**12/12 格齐**（288/1152/4608 面板 + `legacy` 1152 面板带），provenance 完整；
  - `tensor_ladder.json`：**仅 1 格**（`h/a=3.0, k0a=0.5`，档 0/1 = 20/80 面板，缺 320 面板档）；
  - `results.json`：`execution_status = "PARTIAL: 1 of 12 planned cells"`。
- 因此本单的实质增量只能是**把 TensorBEM 侧面板梯补齐并外推**，把 6 个 `XFAIL` 门转真、给出 <15%（目标 3%/5%）的判定。计划文档 `docs/xval_bem_finite_depth_crossval_plan.md` 已由 m22103 冻结，不再重写。

## 2. 本单已完成的动作

1. **诊断了 m22103 遗留的提交为何没产出**（关键新信息，可复用）：
   `cluster_jobs` 记录 `cj4149`（`repo=TensorBEM, ref=main`）**rc=127**，
   日志 `/srv/tensor/agent/cluster/jobs/cj4149.log`：
   `bash: scripts/run_capytaine_finite_depth_xval.sh: No such file or directory`。
   即 **驱动脚本只在本地区/分支上，GitHub main 上不存在**——用 `repo=TensorBEM ref=main`
   递交这类"新建脚本"型重算，永远 rc=127 且**不会**报脚本缺失以外的错误。
   复现纪律：凡本单新增的驱动脚本，重算必须 `stage_dir=<worktree>` 递交。
2. **本单再递交一次** `cj4149`（同命令、`cores=12`、`timeout=7200`）：排队 28 min 后
   在 `n315` 上执行，**仍是 rc=127**（同一原因），取证 tar 落
   `/srv/tensor/agent/cluster/fetch/cj4149_n315.tar.gz`。
3. **识别出第二个阻塞（调度）**：双节点 `n315`/`n330` 健康
   （`cores=128`，load 76.0 / 41.0，mem_free 923/927 GB），但队列
   `jobs_in_flight` 在 1 h 内从 51 涨到 63，排队位次 31 的作业等了 **27.7 min** 仍未启动；
   期间新作业（cj4150…cj4191）亦全部 `queued`。即有容量却不派发，属调度侧现象，
   不在本仓可修范围（且本席不得动 TensorPlatform）。
4. **交付物（本单实际产出）**：`make_report_m22474.py`——
   论文级报告的**程序化生成器**：读 `results.json`（逐格 TB/capy 外推值、偏差、
   观测阶、门判定）+ 冻结参考 + 面板梯，零手抄生成 5 张图
   （12 格 A33/B33 深度梯对照、$(h/a,k_0a)$ 偏差图、**以深水多极子解析解为独立锚的
   有限水深放大率图**、最差格双侧面板梯、体表面辐射势场云图）与六要素 markdown；
   `--publish` 直发平台。**该脚本已在当前 PARTIAL 工件上跑通**（5 图产出、markdown 生成），
   研究一补齐即可零改动复用。

## 3. 结果（诚实边界）

**本研究仍未收口**：TensorBEM 侧只有 1/12 格、且缺 320 面板档 →
`added_mass_agreement`（预注册 3%）与 `damping_agreement`（5%）在未外推状态下
**不可判**，不能给出 <3% 或 <15% 的结论。当前唯一可判的是负控：
`k0h=1.5` 格上有限水深求解器与深度盲深水求解器的 added-mass 分离
7.01 %（门限 2 %）→ **PASS**，说明该格确实在测深度。

**显式回答「是否 <3%」**：不适用/未定——证据不足以判定，本单不做粉饰。

## 4. 下次接单的唯一动作（已定死，不再探索）

```bash
# 1) 递交（必须 stage_dir，不能用 ref=main，见 §2.1）
run_compute submit repo=TensorBEM stage_dir=<worktree> cores=12 timeout=7200 \
    cmd="EXTRA_LEVEL=1 SHARDS=12 bash scripts/run_capytaine_finite_depth_xval.sh"
# 2) 取回工件后
PYTHONPATH=src python3 make_report_m22474.py            # 图 + 报告 markdown
python -m pytest tests/test_capytaine_finite_depth_crossval.py -q
PYTHONPATH=src python3 make_report_m22474.py --publish  # 发布
```
驱动脚本自身即完整流水线：capytaine 参考已冻结（12/12）会**跳过**参考阶段
（capytaine 不在计算节点上），12 分片跑梯 → 完整性校验合并（缺片/缺档即失败）→
消费者写 `results.json` + `docs/capytaine_finite_depth_crossval.md` + 图。
