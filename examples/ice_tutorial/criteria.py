"""Machine-readable criteria registry for the ice-navigation tutorial.

Every row of every chapter's 判据表 (criteria table) is an entry of
:data:`CRITERIA`.  Two kinds of rows exist:

``computable=True``
    The statement is a *numeric* claim about something this repository can
    compute.  ``metric`` names a key of the chapter payload's ``metrics``
    dict (see :mod:`examples.ice_tutorial.model`), and the evaluation in
    :func:`evaluate` decides pass/fail.  ``tests/test_ice_tutorial.py``
    re-evaluates these rows on every run, so a chapter can never silently
    drift away from the verified code.

``computable=False``
    The statement is a literature / regulatory claim (ice class, Polar Code
    category, resistance decomposition, ...).  The repository does not compute
    it, so the row carries the evidence grade and the reference instead of a
    number; the tutorial renders ``实测 = —`` for it.  Marking these rows
    ``NOT_SUPPORTED``/``WITHHELD`` is deliberate: a teaching chapter must not
    pretend that an external formula is a verified capability of this code.

Reference provenance
--------------------
``REFS`` holds the bibliography entries verbatim as cited by the rows.  This
seat works offline, therefore every entry carries
``ref_status = "题录已核（离线，DOI/页码待知识策展单复核）"``: the title, authors,
year and venue are the ones the repository already cites (or, for the
regulatory items, the instrument identifiers); the DOI/page-level check
belongs to the ``ICE-SHOW-KNOWLEDGE-CURATION`` line and is *not* claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

VALID_EVIDENCE = ("REAL", "MOCK", "STUB", "WITHHELD", "NOT_SUPPORTED")

OFFLINE = "题录已核（离线，DOI/页码待知识策展单复核）"

REFS: dict[str, str] = {
    "WEITZ-KELLER-1950": "Weitz, M. & Keller, J. B. (1950). Reflection of water waves from "
    "floating ice in water of finite depth. Comm. Pure Appl. Math. 3, 305-318.",
    "LAMB-1932": "Lamb, H. (1932). Hydrodynamics, 6th ed. Cambridge University Press, art. 92.",
    "TIMCO-WEEKS-2010": "Timco, G. W. & Weeks, W. F. (2010). A review of the engineering "
    "properties of sea ice. Cold Regions Science and Technology 60(2), 107-129.",
    "SQUIRE-2007": "Squire, V. A. (2007). Of ocean waves and sea-ice revisited. "
    "Cold Regions Science and Technology 49(2), 110-133.",
    "URSELL-1949": "Ursell, F. (1949). On the heaving motion of a circular cylinder on the "
    "surface of a fluid. Quart. J. Mech. Appl. Math. 2(2), 218-231.",
    "LINDQVIST-1989": "Lindqvist, G. (1989). A straightforward method for calculation of ice "
    "resistance of ships. POAC'89, vol. 2, 722-735.",
    "RISKA-1997": "Riska, K., Wilhelmson, M., Englund, K. & Leiviska, T. (1997). Performance "
    "of a chemical tanker in level ice - full scale and model scale tests. POAC'97.",
    "KUJALA-2012": "Kujala, P. & Arughadhoss, S. (2012). Statistical analysis of ice crushing "
    "pressures on a ship's hull during hull-ice interaction. "
    "Cold Regions Science and Technology 70, 1-11.",
    "ISO-19906": "ISO 19906:2019. Petroleum and natural gas industries - Arctic offshore "
    "structures. ISO, Geneva.",
    "POLAR-CODE": "IMO Polar Code (International Code for Ships Operating in Polar Waters), "
    "MSC.385(94), in force 2017-01-01.",
    "IACS-I1I2": "IACS Unified Requirements I1 (Polar Class descriptions), I2 (Polar Class "
    "hull structural design), I3 (machinery).",
    "FSICR": "Finnish-Swedish Ice Class Rules (TRAFI ice class regulations, 2017).",
    "POLARIS": "Stoddard, M. A., Etienne, L., Fournier, M., Pelot, R. & Beveridge, L. (2016). "
    "Making sense of Arctic maritime traffic using POLARIS. IOP Conf. Ser. Earth Environ. "
    "Sci. 34, 012034.",
    "MULTIPOLE": "Thorne/Wang/Linton multipole solution as implemented in "
    "tensorbem.analytic.submerged_sphere_wave_multipole (no discretisation).",
    "REPO-TESTS": "TensorBEM tests/test_ice_rao.py and tests/test_ice_mass_loading_dispersion.py "
    "(merged verification tests of this repository).",
    "REPO-EVIDENCE": "TensorBEM docs/ice_water_ship_numbers.md and "
    "docs/ice_water_ship_evidence (merged evidence bundle, programmatically rendered).",
    "PLATFORM-STD": "TensorPlatform 精度标准：对标实验/参考值误差 >15% 不可接受；"
    "算例池容差 5%；平台判据 3%。",
}


@dataclass(frozen=True)
class Criterion:
    """One row of a chapter's criteria table."""

    id: str
    chapter: str
    statement: str
    evidence: str
    ref_key: str
    metric: str | None = None
    comparator: str | None = None
    threshold: Any = None
    unit: str = ""
    computable: bool = False

    def __post_init__(self) -> None:
        if self.evidence not in VALID_EVIDENCE:
            raise ValueError(f"{self.id}: illegal evidence grade {self.evidence!r}")
        if self.ref_key not in REFS:
            raise ValueError(f"{self.id}: unknown reference key {self.ref_key!r}")
        if self.computable and (self.metric is None or self.comparator is None):
            raise ValueError(f"{self.id}: a computable criterion needs metric+comparator")
        if self.comparator not in (None, "<", "<=", ">", ">=", "true", "in_range"):
            raise ValueError(f"{self.id}: unsupported comparator {self.comparator!r}")

    @property
    def ref(self) -> str:
        return REFS[self.ref_key]

    def hold_text(self) -> str:
        """Human-readable bound, e.g. ``< 1 %``."""
        if self.comparator == "true":
            return "必须成立"
        if self.comparator == "in_range":
            lo, hi = self.threshold
            return f"{lo:g} ~ {hi:g} {self.unit}".strip()
        if self.comparator in ("<", "<=", ">", ">="):
            return f"{self.comparator} {self.threshold:g} {self.unit}".strip()
        return "文献/法规条款"


def _c(**kwargs: Any) -> Criterion:
    """Construct a criterion; a ``metric`` implies ``computable=True``."""
    if kwargs.get("metric") is not None:
        kwargs.setdefault("computable", True)
    return Criterion(**kwargs)


CRITERIA: tuple[Criterion, ...] = (
    # ------------------------------------------------------------------ ch01
    _c(
        id="CH01-C1",
        chapter="ch01",
        statement="质量加载参数 mu = rho_i h_i / rho_w 随冰厚 h_i 严格单调递增",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="mu_monotone_up",
        comparator="true",
    ),
    _c(
        id="CH01-C2",
        chapter="ch01",
        statement="mass-loading 截止周期 T_c = 2*pi*sqrt(mu/g) 在 h_i in [0.3, 2.5] m 内不超过 3.1 s，"
        "即更长周期的涌浪在无耗散质量加载模型下总能穿入冰区",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="cutoff_period_max",
        comparator="<=",
        threshold=3.1,
        unit="s",
    ),
    _c(
        id="CH01-C3",
        chapter="ch01",
        statement="默认冰密度 917 kg/m^3 落在一年冰工程取值范围 900-930 kg/m^3 内",
        evidence="REAL",
        ref_key="TIMCO-WEEKS-2010",
        metric="ice_density_inside_range",
        comparator="true",
    ),
    _c(
        id="CH01-C4",
        chapter="ch01",
        statement="同周期下冰下波长比 lambda_ice/lambda_open 随 h_i 单调递减（冰越厚，波越短）",
        evidence="REAL",
        ref_key="SQUIRE-2007",
        metric="wavelength_ratio_monotone_down",
        comparator="true",
    ),
    # ------------------------------------------------------------------ ch02
    _c(
        id="CH02-C1",
        chapter="ch02",
        statement="垂直二维 BEM（strip）色散与 Weitz-Keller 闭式解在 k in [0.02, 0.5] 1/m 上"
        "的最大相对误差 < 1%（平台 3% 判据的更严内控）",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="max_omega_rel_error",
        comparator="<",
        threshold=0.01,
        unit="-",
    ),
    _c(
        id="CH02-C2",
        chapter="ch02",
        statement="深水极限（kH >= 10）下闭式解与有限水深解相对偏差 < 0.1%",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="deep_limit_max_rel_error",
        comparator="<",
        threshold=1e-3,
        unit="-",
    ),
    _c(
        id="CH02-C3",
        chapter="ch02",
        statement="长波极限（kH = 0.05）相对偏差 < 1%",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="long_wave_rel_error",
        comparator="<",
        threshold=0.01,
        unit="-",
    ),
    _c(
        id="CH02-C4",
        chapter="ch02",
        statement="BEM 面板加密（8 -> 64 面板/波长）使色散误差单调下降",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="convergence_monotone_down",
        comparator="true",
    ),
    # ------------------------------------------------------------------ ch03
    _c(
        id="CH03-C1",
        chapter="ch03",
        statement="传播分支内冰下波数恒大于敞水波数（k_ice > k_open，波长恒缩短）",
        evidence="REAL",
        ref_key="SQUIRE-2007",
        metric="k_ice_gt_k_open",
        comparator="true",
    ),
    _c(
        id="CH03-C2",
        chapter="ch03",
        statement="h_i = 0.9 m 的截止周期落在 1.7-1.9 s（解析值 1.800 s）",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="cutoff_period_ref_ice",
        comparator="in_range",
        threshold=(1.7, 1.9),
        unit="s",
    ),
    _c(
        id="CH03-C3",
        chapter="ch03",
        statement="周期短于截止周期（T = 1.5 s < T_c）的波为消失波：propagating = False 且衰减率 > 0",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="evanescent_branch_ok",
        comparator="true",
    ),
    _c(
        id="CH03-C4",
        chapter="ch03",
        statement="传播分支上 mass-loading 无耗散：衰减因子恒为 1，能量通量幅值比 A_ice/A_open > 1"
        "（冰盖缩短并陡化波场，而不是衰减它）",
        evidence="REAL",
        ref_key="SQUIRE-2007",
        metric="conservative_amplitude_ratio_ok",
        comparator="true",
    ),
    _c(
        id="CH03-C5",
        chapter="ch03",
        statement="冰区波场衰减系数（碎冰散射/耗散）本仓不建模：衰减只可能来自消失波分支",
        evidence="WITHHELD",
        ref_key="SQUIRE-2007",
    ),
    # ------------------------------------------------------------------ ch04
    _c(
        id="CH04-C1",
        chapter="ch04",
        statement="潜球垂荡附加质量 BEM（352 面元，d/R = 3）相对 Lamb 无限流体极限 (2/3)pi rho R^3 "
        "误差 < 5%（案例池 BEM-ICE-SPHERE-HEAVE 容差）",
        evidence="REAL",
        ref_key="LAMB-1932",
        metric="bem_error_vs_lamb_fine",
        comparator="<",
        threshold=0.05,
        unit="-",
    ),
    _c(
        id="CH04-C2",
        chapter="ch04",
        statement="面板 168 -> 352 使 BEM 相对多极精确解的误差单调下降",
        evidence="REAL",
        ref_key="MULTIPOLE",
        metric="bem_ladder_monotone_down",
        comparator="true",
    ),
    _c(
        id="CH04-C3",
        chapter="ch04",
        statement="无离散化的多极精确解在 d/R >= 12 处与 Lamb 极限相差 < 0.1%",
        evidence="REAL",
        ref_key="MULTIPOLE",
        metric="exact_deep_rel_error",
        comparator="<",
        threshold=1e-3,
        unit="-",
    ),
    _c(
        id="CH04-C4",
        chapter="ch04",
        statement="【负结果】mass-loading 冰盖与敞水相比不改变深潜球附加质量："
        "d/R = 3、omega in [0.3, 3.0] rad/s 内相对差 < 1%（冰的作用在波场，不在附加惯性）",
        evidence="REAL",
        ref_key="WEITZ-KELLER-1950",
        metric="ice_open_contrast_max",
        comparator="<",
        threshold=0.01,
        unit="-",
    ),
    _c(
        id="CH04-C5",
        chapter="ch04",
        statement="圆柱垂荡附加质量的解析锚（Ursell 多极解）已被本仓独立复核，"
        "本教程不重复推导，只引用",
        evidence="REAL",
        ref_key="URSELL-1949",
    ),
    # ------------------------------------------------------------------ ch05
    _c(
        id="CH05-C1",
        chapter="ch05",
        statement="冰区船体六自由度 RAO 链在 mu -> 0 极限必须精确复现敞水附加质量、阻尼与激励力"
        "（阻尼须按 omega/sqrt(g k_i) 重标定）",
        evidence="REAL",
        ref_key="REPO-TESTS",
    ),
    _c(
        id="CH05-C2",
        chapter="ch05",
        statement="任何船体算例必须几何先验证：Cb / L / B / T 与型线参数一致才允许进入水动力计算",
        evidence="REAL",
        ref_key="PLATFORM-STD",
    ),
    _c(
        id="CH05-C3",
        chapter="ch05",
        statement="真实 PC6 冰区船型值表未取得（私有数据）：本仓 RAO 结果基于参数化船体，"
        "对真实船型的适用性证据不足",
        evidence="WITHHELD",
        ref_key="IACS-I1I2",
    ),
    _c(
        id="CH05-C4",
        chapter="ch05",
        statement="算例池已合并证据：理想化 PC6 船（L=120 m, B=24 m, T=9 m, Cb=0.719）"
        "在 h_i=0.9 m 冰盖下 RAO 与敞水对照齐备，数字全部程序化渲染",
        evidence="REAL",
        ref_key="REPO-EVIDENCE",
    ),
    # ------------------------------------------------------------------ ch06
    _c(
        id="CH06-C1",
        chapter="ch06",
        statement="冰阻力按 R_ice = R_break + R_subm + R_open 分解（破冰/浸没/敞水三分量），"
        "分项定义与 Lindqvist 方法一致；本仓不含该实现，实现与标定在 ship-performance-platform",
        evidence="WITHHELD",
        ref_key="LINDQVIST-1989",
    ),
    _c(
        id="CH06-C2",
        chapter="ch06",
        statement="冰区阻力复算精度判据：对标模型试验/实船实测误差 ≤ 10%（平台 KPI 静水 ≤5% / 冰区 ≤10%）",
        evidence="REAL",
        ref_key="PLATFORM-STD",
    ),
    _c(
        id="CH06-C3",
        chapter="ch06",
        statement="碎冰航道（brash ice channel）阻力随航道厚度与航道宽度单调增；"
        "冰厚 0.9 m 量级的一年冰航道对应 FSICR 1A Super 设计工况",
        evidence="WITHHELD",
        ref_key="RISKA-1997",
    ),
    _c(
        id="CH06-C4",
        chapter="ch06",
        statement="冰阻力全谱经验链（含 c_break / c_submersion 标定）已由平台 C1-ICE-RESIST 条目落地，"
        "本教程只做判据教学，不重复实现",
        evidence="REAL",
        ref_key="LINDQVIST-1989",
    ),
    # ------------------------------------------------------------------ ch07
    _c(
        id="CH07-C1",
        chapter="ch07",
        statement="局部冰压力随接触面积近似幂律衰减，压力-面积指数约 -0.5，"
        "设计局部压力不得超过该包络",
        evidence="WITHHELD",
        ref_key="ISO-19906",
    ),
    _c(
        id="CH07-C2",
        chapter="ch07",
        statement="冰级与设计冰厚须成对给出：Polar Class PC1-PC7 与 IACS UR I1 冰级描述一一对应，"
        "PC6 对应 medium first-year ice 量级",
        evidence="WITHHELD",
        ref_key="IACS-I1I2",
    ),
    _c(
        id="CH07-C3",
        chapter="ch07",
        statement="低温材料判据：冰区结构钢须按 -40 °C 量级韧性等级（E/EH 级）选用，"
        "并单独校核脆性转变",
        evidence="WITHHELD",
        ref_key="IACS-I1I2",
    ),
    _c(
        id="CH07-C4",
        chapter="ch07",
        statement="冰载荷统计分位（p95/p99）必须给 n_eff 修正置信区间，不得只报点值"
        "（平台 GAP-ICE-TAIL-NEFF 在飞条目）",
        evidence="WITHHELD",
        ref_key="KUJALA-2012",
    ),
    # ------------------------------------------------------------------ ch08
    _c(
        id="CH08-C1",
        chapter="ch08",
        statement="Polar Code 船舶分为 A/B/C 三类：A 类对应至少 medium first-year ice（0.70-1.20 m），"
        "B 类对应 thin first-year ice（0.30-0.70 m），C 类对应敞水或更轻冰况",
        evidence="WITHHELD",
        ref_key="POLAR-CODE",
    ),
    _c(
        id="CH08-C2",
        chapter="ch08",
        statement="冰区营运须随船携带 PWOM（Polar Water Operational Manual），"
        "其结论必须引用可核查的冰情与风险指数",
        evidence="WITHHELD",
        ref_key="POLAR-CODE",
    ),
    _c(
        id="CH08-C3",
        chapter="ch08",
        statement="POLARIS 风险指数法把船舶冰级与冰型按 RIO/RISK 矩阵打分，"
        "可作为冰区航速/航线决策的可追溯依据（不是本仓计算能力）",
        evidence="WITHHELD",
        ref_key="POLARIS",
    ),
    _c(
        id="CH08-C4",
        chapter="ch08",
        statement="波罗的海 1A Super 冰级与 Polar Class 不可直接互换：两套体系的等效力须逐项论证",
        evidence="WITHHELD",
        ref_key="FSICR",
    ),
    # ------------------------------------------------------------------ ch09
    _c(
        id="CH09-C1",
        chapter="ch09",
        statement="本教程每一个可计算判据在 pytest 门禁中重算并通过（章节不得脱离已验证代码）",
        evidence="REAL",
        ref_key="REPO-TESTS",
        computable=False,   # enforced by tests/test_ice_tutorial.py
        metric="computable_criteria_all_pass",
        comparator="true",
    ),
    _c(
        id="CH09-C2",
        chapter="ch09",
        statement="章节内嵌数值与新鲜计算结果逐字一致（渲染指纹一致，禁止手抄）",
        evidence="REAL",
        ref_key="PLATFORM-STD",
        computable=False,   # enforced by tests/test_ice_tutorial.py
        metric="rendered_fingerprint_matches",
        comparator="true",
    ),
    _c(
        id="CH09-C3",
        chapter="ch09",
        statement="每个章节至少 3 行判据且每行都带出处；无出处的判据不得进入教程",
        evidence="REAL",
        ref_key="PLATFORM-STD",
        computable=False,   # enforced by tests/test_ice_tutorial.py
        metric="chapter_criteria_rows_ok",
        comparator="true",
    ),
    _c(
        id="CH09-C4",
        chapter="ch09",
        statement="证据分级五档 REAL / MOCK / STUB / WITHHELD / NOT_SUPPORTED 逐行标注，"
        "未验证能力必须以 WITHHELD 出现",
        evidence="REAL",
        ref_key="PLATFORM-STD",
    ),
)


def criteria_for(chapter: str) -> tuple[Criterion, ...]:
    """All criteria registered for one chapter id (``ch01`` ... ``ch09``)."""
    return tuple(c for c in CRITERIA if c.chapter == chapter)


def chapters() -> tuple[str, ...]:
    seen: list[str] = []
    for c in CRITERIA:
        if c.chapter not in seen:
            seen.append(c.chapter)
    return tuple(seen)


def _check(criterion: Criterion, metrics: Mapping[str, Any]) -> tuple[bool | None, str]:
    """Evaluate one computable criterion against freshly computed metrics."""
    if not criterion.computable:
        return None, "文献/法规条款"
    if criterion.metric not in metrics:
        return False, "缺指标"
    value = metrics[criterion.metric]
    if criterion.comparator == "true":
        ok = bool(value)
        shown = "成立" if ok else "不成立"
        return ok, shown
    if criterion.comparator == "in_range":
        lo, hi = criterion.threshold
        ok = bool(lo <= float(value) <= hi)
        return ok, f"{float(value):.4g} {criterion.unit}".strip()
    ref = float(criterion.threshold)
    got = float(value)
    ops = {
        "<": got < ref,
        "<=": got <= ref,
        ">": got > ref,
        ">=": got >= ref,
    }
    return bool(ops[criterion.comparator]), f"{got:.6g} {criterion.unit}".strip()


def evaluate(chapter: str, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evaluate every criterion of ``chapter``; returns render-ready rows."""
    rows: list[dict[str, Any]] = []
    for criterion in criteria_for(chapter):
        ok, shown = _check(criterion, metrics)
        rows.append(
            {
                "id": criterion.id,
                "statement": criterion.statement,
                "hold": criterion.hold_text(),
                "observed": shown,
                "verdict": "—" if ok is None else ("通过" if ok else "未通过"),
                "evidence": criterion.evidence,
                "ref": criterion.ref,
                "computable": criterion.computable,
            }
        )
    return rows


def summary(chapter_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of pass / fail / not-computable rows."""
    passed = sum(1 for r in chapter_rows if r["verdict"] == "通过")
    failed = sum(1 for r in chapter_rows if r["verdict"] == "未通过")
    textual = sum(1 for r in chapter_rows if r["verdict"] == "—")
    return {"passed": passed, "failed": failed, "textual": textual}