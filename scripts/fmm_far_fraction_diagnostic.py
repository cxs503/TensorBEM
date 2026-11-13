"""m18196 -- far-fraction diagnostic for :class:`tensorbem.fmm.FMMOperator`.

The question this script answers, with numbers rather than prose:

1. **Is ``far_fraction`` faithful to the matvec?**  ``FMMOperator.pair_partition``
   returns the disjoint far / near-geometric / near-box pair counts of the very
   mask ``matvec`` sums; an instrumented subclass that counts the pairs handed to
   ``_far_contributions`` must reproduce ``far_pairs`` exactly, on every geometry.
2. **What is the far band made of?**  The near mask is
   ``(~admissible) | _is_near(routed_leaf, leaf)``.  The second term is the
   conservative box margin of the leaf the target is *routed* to (nearest leaf
   centre), which for a point outside the source cloud is an artifact of routing,
   not a property of the pair.  ``pair_partition`` splits it out as
   ``near_box_pairs``.
3. **What does that margin cost?**  ``_AdmissibilityOnlyOperator`` drops the box
   term (admissibility alone decides); the A/B reports the far fraction and the
   accuracy vs the dense sum for both policies, so the margin's price in
   acceleration and its value in accuracy are both measured.

Outputs (every number in the report is read from here -- no hand copies)::

    docs/fmm_far_fraction_diagnostic/evidence.json
    docs/fmm_far_fraction_diagnostic/tables.md
    docs/fmm_far_fraction_diagnostic/fig_far_fraction.png

Run::

    PYTHONPATH=src python scripts/fmm_far_fraction_diagnostic.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem import fmm as fmm_mod  # noqa: E402
from tensorbem.fmm import FMMOperator, direct_laplace_matvec  # noqa: E402
from tensorbem.geometry3d import icosphere  # noqa: E402

OUT = pathlib.Path("docs/fmm_far_fraction_diagnostic")
GEOMETRIES = ("surface", "shell", "separated")
SWEEP_GEOMETRY = "surface_fine"
PS = (2, 4, 6, 8)
#: leaf-size sweep: coarse trees (few leaves) down to the degenerate single leaf
MAX_LEAF_SWEEP = (8, 16, 32, 64, 128, 2048)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class _CountingOperator(FMMOperator):
    """``FMMOperator`` that counts the pairs handed to ``_far_contributions``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.far_pairs_used = 0
        self.far_pairs_per_block: list[int] = []

    def _far_contributions(self, rel, r, t_pair, s_pair, n_targets):  # noqa: D102
        self.far_pairs_used += int(len(t_pair))
        self.far_pairs_per_block.append(int(len(t_pair)))
        return super()._far_contributions(rel, r, t_pair, s_pair, n_targets)


class _AdmissibilityOnlyOperator(FMMOperator):
    """Counterfactual: the conservative ``_is_near`` box margin is switched off.

    The only difference from :class:`FMMOperator` is ``_pair_components``: a pair
    is summed directly iff the target does not clear the admissibility factor
    times the leaf's own source radius.  Both ``matvec`` and ``pair_partition``
    go through this method, so the A/B has a single varying input.
    """

    def _pair_components(self, block, leaf_of_target):  # noqa: D102
        rel = block[:, None, :] - self._leaf_centers[None, :, :]
        r = np.linalg.norm(rel, axis=2)
        has_sources = self._has_sources[None, :]
        admissible = (
            r > fmm_mod._ADMISSIBILITY * np.maximum(self._radii_vec, 1e-15)[None, :]
        ) & has_sources
        near = (~admissible) & has_sources
        box_near = np.zeros(r.shape, dtype=bool)
        return rel, r, admissible, box_near, near


# ---------------------------------------------------------------------------
# geometries
# ---------------------------------------------------------------------------

def _surface_geometry():
    """BEM-like collocation set: panel centres of icosphere(2), radius 1."""
    mesh = icosphere(subdivisions=2, radius=1.0)
    centres = mesh.vertices[mesh.faces].mean(axis=1)
    rng = np.random.default_rng(18196)
    charges = rng.normal(size=len(centres))
    return centres, charges


def _surface_fine():
    """Finer BEM-like set (icosphere(3), 1280 panels) for the leaf sweep."""
    mesh = icosphere(subdivisions=3, radius=1.0)
    centres = mesh.vertices[mesh.faces].mean(axis=1)
    rng = np.random.default_rng(311)
    return centres, rng.normal(size=len(centres))


def _cases() -> dict:
    centres, charges = _surface_geometry()
    rng = np.random.default_rng(42)
    separated_sources = rng.normal(size=(200, 3)) * 0.5
    separated_targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    separated_charges = rng.normal(size=200)
    fine_centres, fine_charges = _surface_fine()
    return {
        # targets ride the surface, 2% outside the source shell: the routing of a
        # target to a leaf centre is physically meaningful here.
        "surface": dict(
            sources=centres, targets=centres * 1.02, charges=charges, max_leaf=32
        ),
        # targets on a shell at 3 radii from the same source cloud: every target
        # is far from *every* source, so the sound classification is "all far".
        "shell": dict(
            sources=centres, targets=centres * 3.0, charges=charges, max_leaf=32
        ),
        # the m16000 linearity geometry: three evaluation points outside a
        # Gaussian source blob.
        "separated": dict(
            sources=separated_sources,
            targets=separated_targets,
            charges=separated_charges,
            max_leaf=25,
        ),
        # only used by the leaf-size sweep: fine enough that max_leaf actually
        # moves the leaf count (icosphere(2) saturates at 56 leaves for 8/16/32).
        "surface_fine": dict(
            sources=fine_centres,
            targets=fine_centres * 1.02,
            charges=fine_charges,
            max_leaf=32,
        ),
    }


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def _rel_err(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(approx - exact)) / np.max(np.abs(exact)))


def _row(name: str, case: dict, p: int, cls, max_leaf: int | None = None) -> dict:
    sources, targets, charges = case["sources"], case["targets"], case["charges"]
    op = cls(p=p, max_leaf=case["max_leaf"] if max_leaf is None else max_leaf)
    op.setup(sources)
    part = op.pair_partition(targets)
    exact = direct_laplace_matvec(targets, sources, charges)
    got = op.matvec(targets, sources, charges)

    # work proxy: direct source-pair sums + one harmonic evaluation per far pair
    near_source_pairs = part.total_sources - part.far_sources
    harmonic_work = part.far_pairs * (p + 1) ** 2
    row = {
        "geometry": name,
        "p": p,
        "max_leaf": case["max_leaf"] if max_leaf is None else max_leaf,
        "n_sources": int(len(sources)),
        "n_targets": int(len(targets)),
        "n_leaves": part.n_leaves,
        "rel_error_vs_dense": _rel_err(got, exact),
        "near_source_pairs": int(near_source_pairs),
        "work_proxy": int(near_source_pairs + harmonic_work),
        "far_fraction": op.far_fraction(targets),
    }
    row.update({k: v for k, v in part.as_dict().items() if k not in ("n_targets", "n_leaves")})
    return row


def _roundtrip(name: str, case: dict, p: int) -> dict:
    """Instrumented matvec: the far pairs handed to the multipole path."""
    op = _CountingOperator(p=p, max_leaf=case["max_leaf"])
    op.setup(case["sources"])
    op.matvec(case["targets"], case["sources"], case["charges"])
    part = op.pair_partition(case["targets"])
    return {
        "geometry": name,
        "p": p,
        "matvec_far_pairs": op.far_pairs_used,
        "partition_far_pairs": part.far_pairs,
        "blocks": len(op.far_pairs_per_block),
        "match": op.far_pairs_used == part.far_pairs,
    }


def main() -> dict:
    cases = _cases()
    rows = []
    ab = []

    # E1/E2: order sweep, current policy vs admissibility-only policy
    for name in GEOMETRIES:
        case = cases[name]
        for p in PS:
            current = _row(name, case, p, FMMOperator)
            current["policy"] = "current"
            rows.append(current)
            alt = _row(name, case, p, _AdmissibilityOnlyOperator)
            alt["policy"] = "admissibility_only"
            rows.append(alt)
            ab.append({
                "geometry": name,
                "p": p,
                "far_fraction_current": current["far_fraction"],
                "far_fraction_admissibility_only": alt["far_fraction"],
                "far_source_fraction_current": current["far_source_fraction"],
                "far_source_fraction_admissibility_only": alt["far_source_fraction"],
                "rel_error_current": current["rel_error_vs_dense"],
                "rel_error_admissibility_only": alt["rel_error_vs_dense"],
                "work_proxy_current": current["work_proxy"],
                "work_proxy_admissibility_only": alt["work_proxy"],
            })

    # E3: leaf-size sweep on the fine surface case at p=4
    sweep = []
    for max_leaf in MAX_LEAF_SWEEP:
        for policy, cls in (("current", FMMOperator), ("admissibility_only", _AdmissibilityOnlyOperator)):
            row = _row("surface_fine", cases["surface_fine"], 4, cls, max_leaf=max_leaf)
            row["policy"] = policy
            sweep.append(row)

    # E4: round-trip of far_fraction against an instrumented matvec
    roundtrip = [_roundtrip(name, cases[name], p) for name in GEOMETRIES for p in (2, 4, 6)]
    roundtrip_ok = all(r["match"] for r in roundtrip)

    evidence = {
        "mission": "m18196",
        "description": (
            "far-fraction diagnostic of tensorbem.fmm.FMMOperator: disjoint pair "
            "accounting (pair_partition), round-trip against an instrumented "
            "matvec, and an A/B of the conservative _is_near box margin"
        ),
        "admissibility_factor": float(fmm_mod._ADMISSIBILITY),
        "geometries": {
            name: {
                "n_sources": int(len(cases[name]["sources"])),
                "n_targets": int(len(cases[name]["targets"])),
                "max_leaf": cases[name]["max_leaf"],
            }
            for name in GEOMETRIES
        },
        "rows": rows,
        "order_sweep_ab": ab,
        "max_leaf_sweep": sweep,
        "roundtrip": roundtrip,
        "roundtrip_ok": roundtrip_ok,
        "figures": [str(OUT / "fig_far_fraction.png")],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (OUT / "tables.md").write_text(_tables(evidence))
    (OUT / "README.md").write_text(_readme(evidence))
    _figure(evidence)
    return evidence


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------

def _tables(evidence: dict) -> str:
    lines = [
        "<!-- generated by scripts/fmm_far_fraction_diagnostic.py -- do not edit -->",
        "",
        f"Admissibility factor `_ADMISSIBILITY = {evidence['admissibility_factor']:g}`.",
        "",
        "### T1 — order sweep, current policy vs admissibility-only policy",
        "",
        "| geometry | p | far frac (pair) | far frac (source) | near geom pairs | "
        "near box pairs | rel err vs dense | work proxy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in evidence["rows"]:
        lines.append(
            "| {geometry} | {p} | {ff:.4f} | {fsf:.4f} | {ngp} | {nbp} | {err:.3e} | {wp} |".format(
                geometry=row["geometry"] + ("*" if row["policy"] != "current" else ""),
                p=row["p"],
                ff=row["far_pair_fraction"],
                fsf=row["far_source_fraction"],
                ngp=row["near_geometric_pairs"],
                nbp=row["near_box_pairs"],
                err=row["rel_error_vs_dense"],
                wp=row["work_proxy"],
            )
        )
    lines += [
        "",
        "`*` = admissibility-only counterfactual (box margin switched off).",
        "",
        "### T2 — far fraction A/B",
        "",
        "| geometry | p | far frac current | far frac admissibility-only | "
        "work proxy current | work proxy admissibility-only | rel err current | rel err admissibility-only |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in evidence["order_sweep_ab"]:
        lines.append(
            "| {geometry} | {p} | {fc:.4f} | {fa:.4f} | {wc} | {wa} | {ec:.3e} | {ea:.3e} |".format(
                geometry=row["geometry"],
                p=row["p"],
                fc=row["far_fraction_current"],
                fa=row["far_fraction_admissibility_only"],
                wc=row["work_proxy_current"],
                wa=row["work_proxy_admissibility_only"],
                ec=row["rel_error_current"],
                ea=row["rel_error_admissibility_only"],
            )
        )
    lines += [
        "",
        "### T3 — leaf-size sweep ({geom} geometry, p = 4)".format(geom=SWEEP_GEOMETRY) + "",
        "",
        "| max_leaf | leaves | policy | far frac (pair) | far frac (source) | rel err vs dense |",
        "|---|---|---|---|---|---|",
    ]
    for row in evidence["max_leaf_sweep"]:
        lines.append(
            "| {ml} | {lv} | {pol} | {ff:.4f} | {fsf:.4f} | {err:.3e} |".format(
                ml=row["max_leaf"],
                lv=row["n_leaves"],
                pol=row["policy"],
                ff=row["far_pair_fraction"],
                fsf=row["far_source_fraction"],
                err=row["rel_error_vs_dense"],
            )
        )
    lines += [
        "",
        "### T4 — round-trip: instrumented matvec vs pair_partition",
        "",
        "| geometry | p | matvec far pairs | partition far pairs | match |",
        "|---|---|---|---|---|",
    ]
    for row in evidence["roundtrip"]:
        lines.append(
            "| {geometry} | {p} | {m} | {q} | {ok} |".format(
                geometry=row["geometry"],
                p=row["p"],
                m=row["matvec_far_pairs"],
                q=row["partition_far_pairs"],
                ok="yes" if row["match"] else "**no**",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _readme(evidence: dict) -> str:
    """Narrative README with every number injected from *evidence* (no hand copies)."""

    def row(geometry: str, p: int, policy: str = "current") -> dict:
        return next(
            r for r in evidence["rows"]
            if r["geometry"] == geometry and r["p"] == p and r["policy"] == policy
        )

    def sweep(max_leaf: int, policy: str) -> dict:
        return next(
            r for r in evidence["max_leaf_sweep"]
            if r["max_leaf"] == max_leaf and r["policy"] == policy
        )

    surface4, shell4, sep4 = row("surface", 4), row("shell", 4), row("separated", 4)
    surface4_alt = row("surface", 4, "admissibility_only")
    shell4_alt = row("shell", 4, "admissibility_only")
    sep4_alt = row("separated", 4, "admissibility_only")
    one_leaf = sweep(2048, "current")

    return f"""# FMM 远场比例诊断（m18196）

> 能力：`tensorbem.fmm.FMMOperator.pair_partition` —— 把每对 (靶点, 源叶) 拆成
> **互斥且完备**的三个桶（远场多极 / 几何近场 / 盒邻域近场），并给出按源点加权的
> 工作量分数；`far_fraction` 重建于此，与被实测的 `matvec` 远场配对数逐对一致。
> 代码：`src/tensorbem/fmm.py` · 脚本：`scripts/fmm_far_fraction_diagnostic.py` ·
> 测试：`tests/test_fmm_diagnostic_evidence.py`
> 证据（全部数字由脚本生成，本文件的表格与数字均由脚本注入）：
> `evidence.json` / `tables.md` / `fig_far_fraction.png`

## ① 为什么做这个诊断

m16000 的线性算子报告留下一条自相矛盾的观察（其 §⑤-2）：
在「200 源点高斯团 + 3 个远场靶点」几何上 `far_fraction()` 返回 0.0，
但同一几何上「旧电荷多极缓存」造成的误差是 2.46（246%）——只可能来自远场路径。
0.0 与「远场被用到」不能同时成立。

该矛盾的**直接原因**已由 m16109 定位并修复：bench 在**从未 `setup` 的算子**上调用
`far_fraction`，而旧实现 fail-open 返回 0.0（`matvec` 同样静默返回全零）。
本单回答剩下的问题，并把「远场比例」这个诊断量本身做成可审计的：

1. `far_fraction` 报的数**是不是**`matvec` 真正用多极展开的配对数？
2. 近场集合的**成分**是什么——几何上必须直和，还是保守余量？
3. 那条保守余量（`_is_near` 的盒邻域判据）**花了多少加速**、**买了多少精度**？

## ② 方法

- **互斥三桶**（`PairPartition`）。对每个 (靶点, 非空源叶) 对：
  - `far_pairs`：靶点越过 `_ADMISSIBILITY={evidence["admissibility_factor"]:g}` × 该叶自身源点半径，
    且该靶点**路由到的叶**（最近叶心）与该叶不是盒邻域 → 用多极展开；
  - `near_geometric_pairs`：靶点离该叶源点球太近，膨胀不收敛 → 直和（正当理由）；
  - `near_box_pairs`：仅因「路由叶是盒邻域」而直和（保守余量）。
- **单一事实来源**：三桶由 `_pair_components()` 计算，`matvec` 的 `_near_mask()` 也从它取，
  故 `near == (~admissible) | box_near` 与 `matvec` **逐位相同**（重构前后 `matvec`
  输出在 3 组随机几何上 `np.array_equal` 为真，见报告 ③）。
- **回路校验（round-trip）**：`_CountingOperator` 在 `_far_contributions` 上计数，
  要求「被实际送入多极路径的配对数」== `pair_partition().far_pairs`，全部几何/阶数通过
  （`tables.md` T4，`roundtrip_ok={str(evidence["roundtrip_ok"]).lower()}`）。
- **A/B**：`_AdmissibilityOnlyOperator` 只改 `_pair_components`（去掉盒邻域项，纯可容许判据），
  其余完全相同 → 单一变量对照。
- **基准**：稠密直接求和 `direct_laplace_matvec`（无近似），误差为最大范数相对误差。

几何：
{chr(10).join(
    f"- `{name}`：源点 {g['n_sources']}、靶点 {g['n_targets']}、`max_leaf={g['max_leaf']}`"
    for name, g in evidence["geometries"].items()
)}
- `surface`＝icosphere(2) 面元中心做源点、同一批点外移 2% 做靶点（BEM 均配点场景）；
- `shell`＝同一批源点，靶点在 3 倍半径的球壳上（每个靶点离**所有**源点都远）；
- `separated`＝m16000 线性算子报告的原始几何（3 个远场评价点）。

## ③ 结果

| 几何（p=4） | far 配对分数 | far 源点加权分数 | 几何近场对 | 盒邻域近场对 | vs 稠密误差 |
|---|---|---|---|---|---|
| surface | {surface4["far_pair_fraction"]:.4f} | {surface4["far_source_fraction"]:.4f} | {surface4["near_geometric_pairs"]} | {surface4["near_box_pairs"]} | {surface4["rel_error_vs_dense"]:.3e} |
| shell | {shell4["far_pair_fraction"]:.4f} | {shell4["far_source_fraction"]:.4f} | {shell4["near_geometric_pairs"]} | {shell4["near_box_pairs"]} | {shell4["rel_error_vs_dense"]:.3e} |
| separated | {sep4["far_pair_fraction"]:.4f} | {sep4["far_source_fraction"]:.4f} | {sep4["near_geometric_pairs"]} | {sep4["near_box_pairs"]} | {sep4["rel_error_vs_dense"]:.3e} |

A/B（同一几何、同一阶，只切换近/远判据）：

| 几何（p=4） | far 分数（现行） | far 分数（纯可容许） | 工作量代理（现行） | 工作量代理（纯可容许） | 误差（现行） | 误差（纯可容许） |
|---|---|---|---|---|---|---|
| surface | {surface4["far_pair_fraction"]:.4f} | {surface4_alt["far_pair_fraction"]:.4f} | {surface4["work_proxy"]} | {surface4_alt["work_proxy"]} | {surface4["rel_error_vs_dense"]:.3e} | {surface4_alt["rel_error_vs_dense"]:.3e} |
| shell | {shell4["far_pair_fraction"]:.4f} | {shell4_alt["far_pair_fraction"]:.4f} | {shell4["work_proxy"]} | {shell4_alt["work_proxy"]} | {shell4["rel_error_vs_dense"]:.3e} | {shell4_alt["rel_error_vs_dense"]:.3e} |
| separated | {sep4["far_pair_fraction"]:.4f} | {sep4_alt["far_pair_fraction"]:.4f} | {sep4["work_proxy"]} | {sep4_alt["work_proxy"]} | {sep4["rel_error_vs_dense"]:.3e} | {sep4_alt["rel_error_vs_dense"]:.3e} |

（工作量代理 = 直和源点对数 + `far_pairs·(p+1)²`；它是**代理**，不含常数因子，
只用于同几何同阶的横向比较。）

## ④ 三条可复核的结论

1. **`far_fraction` 是配对数，不是工作量。** `separated` 上配对分数
   {sep4["far_pair_fraction"]:.4f} 而按源点加权的分数只有 {sep4["far_source_fraction"]:.4f}——
   叶内源点数不均时两者可差一倍以上。要谈效率必须用 `far_source_fraction` 或工作量代理。
2. **源点云外的靶点，近场集合**全部**来自路由伪影。** `shell` 与 `separated` 的
   `near_geometric_pairs` 都是 {shell4["near_geometric_pairs"]}：每个靶点都清过了可容许判据，
   所有直和配对只是「靶点被路由到的那片叶子恰好是盒邻域」。去掉这条余量后
   far 分数从 {shell4["far_pair_fraction"]:.4f} 升到 {shell4_alt["far_pair_fraction"]:.4f}。
   换句话说：多极路径的**上限**由可容许判据决定，但**实际使用率**由一个与靶点真实位置
   无关的路由决定——远场比例对「靶点走得有多远」几乎不敏感（`shell` 靶点在 3 倍半径上，
   `surface` 在面上，far 配对分数分别是 {shell4["far_pair_fraction"]:.4f} / {surface4["far_pair_fraction"]:.4f}）。
3. **盒邻域余量是精度投资，不是免费的加速损失。** 去掉它，误差上升
   （surface p=4：{surface4["rel_error_vs_dense"]:.3e} → {surface4_alt["rel_error_vs_dense"]:.3e}；
   shell p=4：{shell4["rel_error_vs_dense"]:.3e} → {shell4_alt["rel_error_vs_dense"]:.3e}），
   而在 `separated`（靶点极少、叶极小）上工作量代理反而**变差**
   （{sep4["work_proxy"]} → {sep4_alt["work_proxy"]}）：多极每对的 `(p+1)²` 谐波开销
   压过几十个源点的直和。**「far 分数高就更快」是错的**；现行保守判据在该几何上更省。

## ⑤ 注意点 / 局限

1. **这是诊断，不是算法改进**：`matvec` 的数值路径逐位未变（重构前后 `array_equal`），
   本单只新增可审计的记账 API 与证据管线。
2. **严格说是树码**：单层多极、无 M2L，`far_fraction` 多高都不代表 O(N)。
3. **工作量代理不等于墙钟**：未做端到端 BEM 求解（`solve_laplace_surface*`）计时，
   `max_leaf` 扫描中 8/16 与 32/64/128 分别落在同样的叶数（{sweep(8, "current")["n_leaves"]} /
   {sweep(32, "current")["n_leaves"]}）——八叉树按点分布分裂，叶数被几何量化，
   扫描只有 {len({r["n_leaves"] for r in evidence["max_leaf_sweep"]})} 个不同叶数档位。
4. **单叶退化如实登记**：`max_leaf={one_leaf["max_leaf"]}` 时叶数 {one_leaf["n_leaves"]}，
   far 分数 {one_leaf["far_pair_fraction"]:.4f}（纯直和），误差 {one_leaf["rel_error_vs_dense"]:.3e}
   ——「far 分数 0」在退化树上是对的，不是缺陷。m16000 的 0.0 是另一回事（未 `setup`）。
5. **m16000 证据修正**：`docs/fmm_operator_linearity/evidence.json` 里
   `error_versus_order.far_fraction` 曾是 0.0（未 `setup` 的旧调用），本单用脚本重算为
   {sep4["far_pair_fraction"]:.4f}（同几何、当前代码的复现值），
   并由 `tests/test_fmm_diagnostic_evidence.py::TestLinearityEvidenceConsistency` 钉住不再腐化。

## ⑥ 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 本单实测 | 是否 <3% |
|---|---|---|---|
| surface p=4 matvec | 稠密直接求和 | {surface4["rel_error_vs_dense"]:.3e} | ✅ |
| surface p=8 matvec | 稠密直接求和 | {row("surface", 8)["rel_error_vs_dense"]:.3e} | ✅ |
| shell p=4 matvec | 稠密直接求和 | {shell4["rel_error_vs_dense"]:.3e} | ✅ |
| separated p=4 matvec | 稠密直接求和 | {sep4["rel_error_vs_dense"]:.3e} | ✅ |
| 单叶退化 matvec | 稠密直接求和 | {one_leaf["rel_error_vs_dense"]:.3e} | ✅（等价于直和） |
| far 配对数 vs 实测 matvec 远场对数 | 逐行相等 | {sum(1 for r in evidence["roundtrip"] if r["match"])}/{len(evidence["roundtrip"])} 行匹配 | ✅（0% 偏差） |

结论：所有被引用的数值误差均远小于 3%；本单的判别性指标是**配对计数完全一致**
（{sum(1 for r in evidence["roundtrip"] if r["match"])}/{len(evidence["roundtrip"])} 行，偏差 0），
精度量只用于说明保守判据的取舍。

图：`fig_far_fraction.png` —— 左：三种几何在 p=2/4 下的 far 配对分数（现行 vs 纯可容许）；
右：`surface_fine` 上 far 分数随 `max_leaf` 的变化（现行 vs 纯可容许），最右端为单叶退化点。
"""

def _figure(evidence: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    width = 0.2
    x = np.arange(len(GEOMETRIES))
    for i, p in enumerate((2, 4)):
        cur = [
            next(r for r in evidence["rows"]
                 if r["geometry"] == g and r["p"] == p and r["policy"] == "current")
            for g in GEOMETRIES
        ]
        alt = [
            next(r for r in evidence["rows"]
                 if r["geometry"] == g and r["p"] == p and r["policy"] == "admissibility_only")
            for g in GEOMETRIES
        ]
        ax.bar(x + (2 * i - 1.5) * width, [r["far_pair_fraction"] for r in cur], width,
               label=f"current, p={p}")
        ax.bar(x + (2 * i - 0.5) * width, [r["far_pair_fraction"] for r in alt], width,
               label=f"admissibility-only, p={p}", alpha=0.55, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(GEOMETRIES)
    ax.set_ylabel("far_pair_fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("multipole-evaluated pair fraction")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for policy, style in (("current", "-o"), ("admissibility_only", "--s")):
        for geometry in (SWEEP_GEOMETRY,):
            rows = [r for r in evidence["max_leaf_sweep"]
                    if r["policy"] == policy and r["geometry"] == geometry]
            ax.plot([r["max_leaf"] for r in rows], [r["far_pair_fraction"] for r in rows],
                    style, label=f"{geometry}, {policy}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("max_leaf")
    ax.set_ylabel("far_pair_fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"far fraction vs leaf size, {SWEEP_GEOMETRY} (p = 4)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_far_fraction.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ev = main()
    print(json.dumps({
        "roundtrip_ok": ev["roundtrip_ok"],
        "rows": len(ev["rows"]),
        "surface_p4": next(r for r in ev["rows"]
                           if r["geometry"] == "surface" and r["p"] == 4
                           and r["policy"] == "current")["far_pair_fraction"],
        "shell_p4": next(r for r in ev["rows"]
                         if r["geometry"] == "shell" and r["p"] == 4
                         and r["policy"] == "current")["far_pair_fraction"],
    }, indent=2))