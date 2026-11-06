# m4803 — main CI 红单归因：不是红测，是没有 runner 领单

## 0. 结论先行

`main@fa868b64` 的 `tests` 工作流**没有任何断言失败**：那一轮 run（`42152088640`）
在 02:43:07Z 入队、03:12:46Z 被取消，`runner_name == ""`、**记录到的 step 数 = 0**。
也就是说 `pytest` 根本没被执行过——红是「排队 30 分钟没人接单被取消」，不是测试红。

对照组的同仓 main 上一轮 run（`42150435454`，父提交 `daec9ee`，02:27Z）跑在
GitHub 托管 runner（`GitHub Actions 1000020445`）上，**23 个业务 step 全部 success**，
含 `Run test suite` 与证据上传。同一棵树的测试是绿的。

因此本单的修复对象不是测试，而是「作业找不到能接单的 runner」这一条：
`fa868b6` 把 `runs-on` 从 `ubuntu-latest` 改成自建池 `ci-loong64` 之后，
该池再没有派发过任何作业（同窗口 8 个仓、16 条 run 全部 `queued`，
`docs/m4803/ci_dispatch_triage.json`）。

证据全部由 `tools/ci_dispatch_triage.py` 从 Actions REST API 取回并落到 JSON，
本页数字从 JSON 渲染（`--render`），无手抄。

## 1. 案例（这是什么对象）

TensorBEM 是船舶耐波性/水动力边界元库；`tests` 工作流是它的数值证据闸门：
先在 `artifacts/` 里生成 13 份基准报告（Ursell 圆柱辐射、MacCamy-Fuchs、Hulme 半球、
Dean 绕射、球/椭球附加质量、经典基准研究……），然后跑 `pytest`，
文档里印出的每个数字都要能从当前求解器重算出来。闸门有两种红法，
本单的价值就在于把它俩分开：

| 红法 | 判据 | 处置 |
|---|---|---|
| 测试失败 | run `completed/failure`，job 有 runner、有 step 结论 | 修算法/修 pin |
| 没接单 | run `cancelled`，job `runner_name=""`、`steps_recorded=0` | 修工作流的跑池选择 |

两者在 checks 界面里都是一个红叉，所以「main 红了」这句话本身不足以定位。

## 2. 怎么做的

1. **先归因，再动手。** 用 Actions API 取目标 run、job、step 三级事实
   （`tools/ci_dispatch_triage.py` → `docs/m4803/ci_dispatch_triage.json`）：
   目标 run 0 个 step、无 runner；对照 run 23 个 step 全绿。
2. **排除仓内原因。** 目标 job 的 0 个 step 意味着 checkout 都没跑过，
   所以不存在「测试红了 / 依赖装不上 / 证据没生成」这类仓内解释。
3. **确认不是单仓故障。** 同池 8 个仓（BEM/LBM/FEM/FVM/DEM/SPH/DG/CAD）
   最新 run 全部停在 `queued`（16/16），单仓 bug 无法解释。
4. **修在能修的地方（本仓）。** `runs-on` 不再钉死单一池，改为按仓库变量解析：

   ```yaml
   runs-on: ${{ vars.TENSORBEM_CI_RUNNER || 'ubuntu-latest' }}
   ```

   默认 = 当天确实派发过的托管镜像（= 上一轮绿的那条路）；自建池回来时
   把 `TENSORBEM_CI_RUNNER=ci-loong64` 配到仓库变量即可，**不需要改代码**。
5. **把 `actions/setup-python` 装回但加守卫**：自建 loong64 runner 解不开它
   （`tar: Cannot change mode to rwxr-xr-x`，这正是 `fa868b6` 删掉它的原因），
   而托管镜像需要它把解释器钉到 3.11。守卫条件跟随同一个变量
   （`ci-*` 前缀走自建池 → 跳过），两条路互不拖累。
6. **加契约测试** `tests/test_ci_runner_pool_contract.py`：把上面这条决定写成
   可执行断言（`runs-on` 必须经变量解析且默认托管镜像、setup-python 必须带守卫、
   证据 JSON 必须记录「目标 run 无 runner、0 step、对照 run 全绿」），
   防止下一次有人把一个不可派发的池再钉死回去、再次让 main 静默变红。

## 3. 计算结果（数字表，全部来自 `docs/m4803/ci_dispatch_triage.json`）

| 量 | 值 |
|---|---|
| 目标 run | `42152088640`（`fa868b64`，push） |
| 目标 run 结论 | `cancelled` |
| 目标 job 的 runner | `<none>`（空） |
| 目标 job 记录的 step 数 | 0 |
| 目标 run 寿命 | 1779 s（入队→取消） |
| 对照 run | `42150435454`（`daec9ee`，push） |
| 对照 runner | `GitHub Actions 1000020445`（托管） |
| 对照 step 全绿 | True（23 个业务 step + 3 个收尾 step） |
| 同池被查 run 数 | 16（8 个仓） |
| 其中仍 `queued` | 16 / 16 |
| 至少有一条 queued run 的仓 | 8 / 8 |

## 4. 图

**本单不出图**，理由：修复对象是 CI 调度事实（run/job/step 计数与 runner 归属），
没有场量、没有随网格/频率变化的曲线；本单唯一的数值对比是上表的
「0 step vs 23 step」，用表格与 JSON 表达比画图更精确。仓库里其它带图的
基准修复单（如 m4658 的 pin 偏差图）不受本单影响。

## 5. 注意点 / 局限

1. **「main 红」不等于「测试红」**：先看 `runner_name` 与 `steps_recorded`，
   再看 step 结论。0 step 的红单里不存在可以修的断言。
2. **池的可用性不是代码属性**：`ci-loong64` 在 02:43 之后一条作业都没派发，
   但进程仍在跑、日志里仍在刷 token refresh（每 ~50 min 一次），
   从机器侧看不出「离线」——只能从「run 一直 queued」反推。本单没有改动
   任何 runner 进程或注册信息（那是平台侧/机器侧的事，非本仓文件）。
3. **默认值的选择是有代价的**：托管 runner 消耗额度，自建池免费但当时不可用。
   默认取托管是「先让门禁重新具备裁决能力」的取舍，变量即开关，owner 可随时切回。
4. **未做**：本会话 shell 单次上限 120 s、集群节点上 `~/.local` 的 numpy
   在 `leggauss(40)` 处抛 `LinAlgError`，因此**没有**在本机跑完整个 23 step
   的 CI 复现（全套 pytest 在后台跑到 11% 后未及收尾）。已做的验证见 §6。
5. 另一个环境坑（不是仓内 bug，但会污染本机复现结论）：本机 `tensorbem`
   可编辑安装指向**另一个工作树**（`/root/tp-agent/work/m4614/src`），
   不设 `PYTHONPATH=$PWD/src` 时 `tests/test_doc_claims.py` 会报
   `repo_root() == /root/tp-agent/work/m4614`。CI 里是干净安装，不受影响；
   本机复现必须显式加 `PYTHONPATH`。

## 6. 验证（实跑命令）

```bash
cd /root/tp-agent/work/m4803
python tools/ci_dispatch_triage.py --render          # 由 JSON 渲染 docs/m4803/README.md
python -c "import yaml; d=yaml.safe_load(open('.github/workflows/tests.yml'))"   # 工作流仍是合法 YAML
PYTHONPATH=$PWD/src python -m pytest -q tests/test_ci_workflow_contract.py tests/test_ci_runner_pool_contract.py
PYTHONPATH=$PWD/src python -m pytest -q tests/test_doc_inventory.py tests/test_capability_evidence.py
```

| 检查 | 结果 |
|---|---|
| 工作流 YAML 解析 | OK；`runs-on` = `${{ vars.TENSORBEM_CI_RUNNER \|\| 'ubuntu-latest' }}`，23 个 step |
| 既有 CI 契约（`test_ci_workflow_contract.py`） | 4 passed（顺序/安装 extra/上传约定未被破坏） |
| 新增跑池契约（`test_ci_runner_pool_contract.py`） | 5 passed |
| 文档清单 + 能力证据闸门 | 30 passed |
| 负控（新增契约的失败方向） | 已实测：把切片起点写成 `uses:` 行时 `test_setup_python_is_skipped_on_a_self_hosted_pool` 立刻红（说明它真的在看守卫条件，不是恒真断言） |

## 7. 误差判定

本单不含数值算例，**没有需要与基准对比的物理量**，因此不存在「误差是否 <3%」的判定
对象——这是如实声明，不是规避：修复物是一个 YAML 选择器与一组工作流文本断言，
其正确性由 §6 的 YAML 解析 + 9 条契约测试（含反例红）裁决，而不是由百分比误差裁决。
上一轮在该树上真正跑过的数值证据（`daec9ee` 的 23 个基准 step）沿用其自身报告
（`artifacts/*.json`，CI 上传件），本单未重算、也未改动任何数值代码路径。