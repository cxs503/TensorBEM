# 2-D 自由面 Green 函数主值积分尾部缺陷修复（m16829）

**锚点**：`NEW: GAP-BEM-GREEN-FAMILY-CROSSCHECK 登记的未修缺陷（2-D 深水 PV 尾段 6.2e-2 @ nu|X|=13.3）→ 本单修复`
（该锚点在 `docs/oss_inventory_bem/registry.json` 有量化验收但未进 feature_ledger，建议条目见文末。）

**产出**：`src/tensorbem/free_surface.py`（尾部闭式 + 头段节点阶梯）、
`tests/test_free_surface_pv_tail.py`（9 项新测）、`tests/test_green_family_crosscheck.py`（钉值反转）、
`scripts/green_family_crosscheck.py`（裁判升级 + 状态改写）、
`docs/green_family_crosscheck/{README.md,evidence.json,fig_depth_limit.png}`（重生成）、
`docs/free_surface_pv_tail_fix/fig1_pv_tail.png`。

---

## ① 案例介绍

2-D 频域自由面 Green 函数（线源）是 Ursell 柱、2-D 浮体剖面的求解内核：

    G = (1/2pi)[ log r - log r1 - 2 I ] - i exp(-nu A) cos(nu X),
    I = PV∫_0^∞ exp(-kA) cos(kX)/(k - nu) dk,   A = -(z+z0) > 0, X = x - x0.

每个面元对都要算 `I`，而 A 与 X 在整个装配面上连续变化。仓里原实现为
「`[0,2nu]` Gauss-Legendre 头段 + `[2nu,∞)` 160 点 Gauss-Laguerre 尾段」。
尾段被积函数在积分变量里以相位 `nu|X|/2pi` 振荡，固定节点数不能随 `nu|X|` 增大而分辨，
误差随水平偏移增长。`docs/green_family_crosscheck`（m16435）已登记该缺陷但**未修**
（理由：涉及每个 2-D BEM 求解、需全量回归）。

## ② 怎么做的

1. **尾段闭式化**。令 `b = nu*A - i*nu*X`（`Re b = nu*A > 0`），代换 `k = nu*s`、`s = 1+u`：

       ∫_{2nu}^∞ exp(-kA) exp(ikX)/(k-nu) dk = exp(-b)∫_1^∞ exp(-bu)/u du = exp(-b) E1(b)

   实部 = 余弦主值，虚部 = 正弦主值（源点梯度 `green_gradient` 用到正弦主值）。
   恒等式 `PV∫_{-1}^∞ exp(-bu)/u du = -Ei(b)`（`Re b>0` 上解析，
   已验证 `Re b>0` 与 30 位 mpmath 一致）把奇性吸收进单一特殊函数；`E1` 取 principal
   branch（`Re b>0` 保证即物理支），因此**不需要**额外 iπ 项——iπ 残差是辐射条件项，
   在 `green_function` 里已单独相加。
2. **头段节点阶梯**：头段是 `[-1,1]` 上光滑积分，保留 Gauss-Legendre，节点数按块内最大
   振荡相位从 `(80,160,320,640,1280,2560)` 选取（实测 80 点覆盖 `nu|X|<=100`；
   固定 80 点在 `nu|X|=200` 绝对误差 3.5，阶梯后 `nu|X|=400` 仍 1e-13）。
3. **独立裁判升级**：原 `_pv_reference`（细梯形 + 抠窗 `|k-nu|<eps`）自带
   `O(h f'(nu)) ~ 1e-4` 偏差（以 30 位 mpmath 对照确认）；改为「面板边界强制落在
   `k=nu`、`k=2nu` ＋ 每振荡周期 8 点的分块 Gauss-Legendre」，与 mpmath 30 位一致到
   `1e-14`，自身 refinement spread `4.95e-15`（原档 5e-3）。

## ③ 计算结果（相对独立裁判的偏差；`nu=1` 故 depth=nu*A、offset=X）

| nu\|X\| | nu*A=0.05 | nu*A=0.48（审计几何） | nu*A=5.0 |
|---|---|---|---|
| 0.0 | 3.0e-05 → **2.2e-15** | 1.8e-12 → **4.6e-16** | 3.9e-15 → **3.9e-15** |
| 3.0 | 1.6e+00 → **1.4e-15** | 8.8e-08 → **1.1e-15** | 8.0e-15 → **8.0e-15** |
| 5.3 | 5.2e-01 → **5.3e-15** | 4.8e-04 → **6.8e-16** | 1.8e-14 → **1.9e-14** |
| 9.4 | **1.9e+01** → **4.2e-14** | 3.9e-01 → **1.5e-14** | 3.5e-14 → **3.5e-14** |
| 13.3 | 5.0e-02 → **2.2e-15** | **5.2e-02** → **1.4e-15** | 4.2e-14 → **4.2e-14** |
| 30 | 1.6e-02 → **1.1e-14** | 1.2e-02 → **4.4e-15** | 9.0e-14 → **9.0e-14** |
| 120 | 2.8e-02 → **3.7e-14** | 1.3e-01 → **5.3e-15** | 4.7e-11 → **4.7e-11** |

粗体 = 修复后。**缺陷比审计登记的更严重**：审计只测 `nu*A=0.48`；浅缩放深度
（`nu*A=0.05`，掠射面元对的常态）在 `nu|X|=9.4` 处误差 **1920 %**，该内核在那里完全失效。

`docs/green_family_crosscheck/evidence.json` 重生成（同 5 点网格，`nu=0.4`、A=0.48）：

| nu\|X\| | PV 相对偏差 | 深水核 G 偏差 | 有限水深核 G 偏差 |
|---|---|---|---|
| 1.0 | 0.0 → **0.0** | 5.9e-05 → **0.0** | 8.7e-07（不变） |
| 5.3 | 1.0e-03 → **3.2e-15** | 8.6e-04 → **2.9e-15** | 8.4e-07（不变） |
| 13.3 | 6.2e-02 → **5.2e-15** | 5.1e-02 → **4.3e-15** | 8.1e-07（不变） |

**连带纠正一条已发表结论**：审计曾把「2-D 有限水深核在 `nu*h=20` 与深水核差 5e-2」
写成「已验证的 2-D 核适用域限制」并建议加 fail-closed 守卫。修复后同一差值
**3.9e-4**（`nu*h=100` 1.4e-5 达标 1e-4；`nu*h=400` 1.2e-6），即那 5e-2 几乎全部
来自深水核求积误差而非物理限制；有限水深核测量值全程未变（8e-07 级），佐证「错的是深水核」。

## ④ 结果图

`docs/free_surface_pv_tail_fix/fig1_pv_tail.png`：左 = 三种缩放深度的相对偏差 vs `nu|X|`
（虚线=旧 Laguerre 尾，实线=闭式），右 = 旧算法带符号绝对误差（缺陷签名：随相位增长）。
基准线 1e-10 与 1.0。裁判曲线由 `scripts/green_family_crosscheck._pv_reference` 程序化生成。

## ⑤ 注意点

1. **先修裁判再判对错**：旧细梯形裁判 1e-4 偏差大于它要判的近场偏差（3e-4 档），
   不升级裁判，修完也会被误判「仍有 1e-4 误差」。
2. **面板边界必须落在扣除点上**：头段把 `f(k)-f(nu)` 的扣除限于 `[0,2nu]`，
   若面板跨越 `k=2nu`（扣除突然消失），单面板跳变即造成 2e-3 假误差（第一版裁判踩到）。
3. **头段并非无罪**：闭式尾段之后头段是唯一误差源，固定 80 点只到 `nu|X|~100`；
   已按相位分档并写进测试（`test_head_rule_ladder_resolves_large_phase`）。
4. **适用域**：`nu*A >= 1e-6`（模块既有守卫）；`nu*A` 很大时 `E1(b)` 下溢到 0，
   而尾段本身在 `e^{-2 nu A}` 量级，绝对意义可忽略，不改判定。
5. **未做（如实）**：3-D 深水核 `free_surface3d` 区域 3 同样用固定节点 Gauss-Laguerre
   尾段（`[2,∞)`），大 `nu|X|` 时可能同病，本单未审计未修；2-D 有限水深核求积路径未动。

## ⑥ 误差判定

| 量 | 目标 | 实测 | 达标 |
|---|---|---|---|
| 2-D PV 核 vs 独立裁判（审计 5 点网格） | < 3 % | ≤ 5.2e-15 | **是** |
| 2-D PV 核 vs 独立裁判（5 深度 × 9 相位网格） | < 3 % | ≤ 1e-11 | **是** |
| 2-D 深水 Green 函数 vs 裁判（审计几何） | < 3 % | ≤ 4.3e-15 | **是** |
| 裁判自身 refinement spread | < 1e-12 | 4.95e-15 | **是** |
| 2-D 深度极限（`nu*h=100`） | < 1e-4 | 1.44e-05 | **是** |
| 2-D 深度极限（`nu*h=20`） | < 1e-4 | 3.91e-04 | **否（如实：`nu*h>=100` 才达标）** |

**是否 <3 %**：是——关键量达 1e-14 级，远低于 3 %；唯一未达标项（`nu*h=20` 深度极限）
是真实残差（物理差异+离散），如实列出不粉饰。

## 验证（跑了什么）

- `pytest tests/test_free_surface_pv_tail.py` → **9 passed**（独立裁判 5 深度 × 9 相位，
  含旧算法负对照：证明被弃的 Laguerre 尾确实 >50 % 错）。
- `pytest tests/test_green_family_crosscheck.py` → **7 passed**（钉值从「缺陷指纹」反转为
  「已修复」，含 `nu*h=100` 达标断言）。
- `pytest tests/test_free_surface_kernel.py tests/test_ursell_benchmark.py
  tests/test_ursell_band_audit.py tests/test_causality.py` → **66 passed**（内核守恒律/
  自由面条件/辐射条件/Ursell 基准无劣化）。
- `PYTHONPATH=src python scripts/green_family_crosscheck.py` 重生成证据链（README/
  evidence.json/fig），referee spread 5e-3 → 4.95e-15。

## 建议 feature_ledger 新条目

- `GAP-BEM-2D-PV-TAIL`（域 TensorBEM / 频域势流内核）：本单完成，建议 `verified`。
- `GAP-BEM-3D-FS-TAIL`（域 TensorBEM）：3-D 深水核区域 3/4 尾段相位自适应求积；
  验收 `nu|X| ∈ [10,200]` 网格 vs 独立裁判 `< 1e-10`，既有锚点（BC/互易/深度极限）
  劣化 `< 1%`。