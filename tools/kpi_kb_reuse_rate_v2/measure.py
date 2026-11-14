#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KPI-KB-REUSE-RATE 度量管线 v2（RCI 核心 KPI①：知识复用率）。

为什么有 v2（v1 = ship-performance-platform/tools/kpi_kb_reuse_rate/measure.py, m17130）
------------------------------------------------------------------------------------
v1 口径 = 在 ``tensor_dev_missions.goal`` / ``tk_work_packages.contract`` 正文里用
kb_key 边界正则扫引用，证据门 = ``tensor_dev_missions.report_id``。实测恒为 0，
而台账 ``knowledge_injections`` 上有 42 个 KO 有引用记录 —— 不是"知识没被复用"，
是 v1 的三处口径缺陷（本单证据见 docs/audit/KPI-KB-REUSE-RATE-V2-MEASUREMENT.md）：

  D1 通道错位：活体引用只在注入台账（tools/context_package/loop.py 环④回写
     ``citation_count/cited_at/citation_ref``）；goal 正文扫描看不到它。
  D2 键空间错位：台账用 ``kb:<knowledge_base.id>`` / ``negk:loop:<mission_id>``，
     而 KO 注册表键是 ``cap:...`` / ``lesson:m<id>`` / ``KB-...``。不做解析，join 恒空。
  D3 计数重复：同一 (kb_key, mission) 会随上下文包多轮注入多行，loop.py 每行各 +1
     → ``citation_count`` 是"轮次累计"不是"引用主体数"；直接求和会虚增（实测 15×）。
  D4 别名重复：632 条 ``negk:loop:<m>`` 与 ``knowledge_base.lesson:m<m>`` 是同一知识
     的镜像（loop.py 命中 negk 时还会 bump 镜像卡的 cited_count），两边都计 = 分母/分子双计。
  D5 证据门取错表：``tensor_dev_missions.report_id`` 全表仅 84/17931 非空，真正的
     产物台账是 ``tensor_reports``（12075 行 / 10151 mission）。

v2 定义（本模块唯一实现，纯函数可单测）
--------------------------------------
    KO 宇宙 U = de-alias(knowledge_base[active] ∪ negative_knowledge_ledger[active])
    引用事件 = knowledge_injections 中 citation_count>0 的行，
               去重到 (canonical KO, consumer=mission_id)（计数取 max，不求和）
    后续性   = 该事件 cited_at >= KO 建档时间（kb.created_at / negk.registered_at）
    证据     = consumer mission 在 tensor_reports 有非空报告（artifact ledger）
    reuse_rate(gate) = |{ko ∈ U : ∃ 事件 满足 gate}| / |U|

gate 阶梯（预注册，见报告②；顺序固定，不看结果再调）：
    L0 text_only  : 仅 goal/contract 正文引用（= v1 口径，复现用）
    L1 any        : 任意台账引用
    L2 subsequent : L1 + 后续性
    L3 evidenced  : L2 + 产物佐证（= KPI 验收口径）

文献锚点（external_anchor，方法谱系见报告①）：
  - 引用计数的"唯引证主体"原则：bibliometrics 计被引次数按 citing document 去重，
    不按出现次数（Garfield, Citation Indexing, 1979；h-index 亦按论文计）。
  - "词面重合 ≠ 引用"：Kessler (1963) bibliographic coupling / Small (1973) co-citation
    区分 coupling 与 direct citation。loop.py 的 ``shared_terms>=min_terms`` 命中属
    coupling（词面耦合），故 v2 不把它升格为 citation（只作 ``source`` 标注）。
  - 复用须有"另一个主体、另一个用途"：Markus (2001) MISQ 知识复用四情境；
    故 self/注入器自评不单独计为复用（需 consumer 产物佐证）。
  - 溯源证据本体：W3C PROV-O（2013）——引用须能解析到产物实体。
  - 复用率定义式：Poulin (1997) Measuring Software Reuse, reuse% = reused/total。

零手抄：所有数字由 SQL 取数后程序化注入；本脚本无任何业务数字常量。

用法：
    python3 tools/kpi_kb_reuse_rate_v2/measure.py --out runs/kpi_kb_reuse_rate_v2
    python3 tools/kpi_kb_reuse_rate_v2/measure.py --snapshot runs/.../evidence.json --out ...
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

KPI = "KPI-KB-REUSE-RATE"
SCHEMA = "kpi-kb-reuse-rate/2"
GATES = ("L0_text_only", "L1_any", "L2_subsequent", "L3_evidenced")
_PSQL_DEFAULT = "runuser -u postgres -- psql -d tensorplatform"


# ---------------------------------------------------------------- 键空间解析
def canonical_ko_id(kb_key: str,
                    kb_by_id: Dict[int, str],
                    kb_keys: set,
                    negk_entry_ids: set) -> Optional[str]:
    """把台账/正文里的"键"解析成规范 KO id；解析不了返回 None。

    支持四种写法（现网实测全出现过）：
      kb:<knowledge_base.id>        -> "kb:<kb_key>"      (v1 完全没处理)
      negk:loop:<mission_id>        -> "negk:<entry_id>"  (负知识台账)
      lesson:m<mission_id> 等真键   -> "kb:<kb_key>"
    """
    key = (kb_key or "").strip()
    if not key:
        return None
    m = re.fullmatch(r"kb:(\d+)", key)
    if m:
        kid = int(m.group(1))
        return f"kb:{kb_by_id[kid]}" if kid in kb_by_id else None
    if key.startswith("negk:"):
        # 台账写法 negk:<entry_id>；负知识表 entry_id 自身有的带 negk: 前缀、有的不带
        # 规范 id 统一为 "negk:<bare entry_id>"（一次解析，两种写法收敛到同一 id）
        bare = key[len("negk:"):]
        if bare in negk_entry_ids or key in negk_entry_ids:
            return f"negk:{bare}"
        return None
    if key in kb_keys:
        return f"kb:{key}"
    return None


def alias_map(kb_keys: Iterable[str],
              negk_rows: Iterable[Tuple[str, Optional[int]]]) -> Dict[str, str]:
    """负知识 -> knowledge_base 镜像卡的别名表（canonical id -> 镜像 canonical id）。

    镜像规则（现网实测 632/632 命中，非本模块发明）：loop.py 对
    ``negk:loop:<m>`` 命中时 bump ``knowledge_base.kb_key='lesson:m<m>'``。
    """
    keys = set(kb_keys)
    out: Dict[str, str] = {}
    for entry_id, mission_id in negk_rows:
        if mission_id is None:
            continue
        mirror = f"lesson:m{mission_id}"
        if mirror in keys:
            bare = entry_id[len("negk:"):] if entry_id.startswith("negk:") else entry_id
            out[f"negk:{bare}"] = f"kb:{mirror}"
    return out


def dealiased_universe(kb_active: set, negk_active: set,
                       aliases: Dict[str, str]) -> Dict[str, str]:
    """返回 {member_id: canonical_id}：镜像对折叠为同一知识（ktype 不同但同源）。"""
    members: Dict[str, str] = {f"kb:{k}": f"kb:{k}" for k in kb_active}
    for e in negk_active:
        bare = e[len("negk:"):] if e.startswith("negk:") else e
        nid = f"negk:{bare}"
        members[nid] = aliases.get(nid, nid)
    # canonical 自身也必须落在宇宙里（镜像卡的 kb side 一般已在 kb_active）
    for cid in set(members.values()):
        members.setdefault(cid, cid)
    return members


# ---------------------------------------------------------------- 引用事件去重
@dataclass
class Event:
    ko: str                 # 原始键（未解析）
    consumer: Optional[int]  # mission_id
    count: int              # 台账 citation_count（轮次累计，不可直接求和）
    cited_at: str           # ISO，'' = 无
    ref: str                # citation_ref
    source: str             # 注入来源（context_package/loop 等）
    injected_at: str = ""


def dedup_events(events: Sequence[Event]) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """(ko, consumer) -> 聚合。计数取 max（轮次累计的最大值≈轮次数），不求和。

    返回的 value:
      n_rows   : 该对在台账里的行数（多轮注入痕迹）
      count    : max(citation_count)（保留供审计，KPI 不使用）
      counted  : 该对是否算"1 个引用主体"（citation_count>0 即算）
      cited_at : 最早一次引用的时间（后续性用最早，避免"越晚越容易过门"）
      refs     : 去重后的 citation_ref 列表
    """
    agg: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for e in events:
        k = (e.ko, e.consumer)
        slot = agg.setdefault(k, {"n_rows": 0, "count": 0, "cited_at": "",
                                  "refs": [], "source": e.source, "injected_at": ""})
        slot["n_rows"] += 1
        slot["count"] = max(slot["count"], e.count)
        if e.cited_at and (not slot["cited_at"] or e.cited_at < slot["cited_at"]):
            slot["cited_at"] = e.cited_at
        if e.ref and e.ref not in slot["refs"]:
            slot["refs"].append(e.ref)
        if e.injected_at and (not slot["injected_at"] or e.injected_at < slot["injected_at"]):
            slot["injected_at"] = e.injected_at
    for v in agg.values():
        v["counted"] = v["count"] > 0
    return agg


def merges_citation(citation_count: int, shared_terms: int, min_terms: int) -> str:
    """把 loop.py 的命中方式分类（口径分级用）。

    direct (键/注入号出现在交付文本) -> "direct"：算 citation（作者显式提及）
    仅词面重合 shared_terms>=min_terms -> "coupling"：按 Kessler(1963) 只算
    bibliographic coupling，不升格为 citation。
    """
    return "coupling" if (shared_terms and shared_terms >= max(1, min_terms)) else "direct"


# ---------------------------------------------------------------- 度量
def parse_ts(s: str) -> str:
    """归一化到可比较的 'YYYY-MM-DD HH:MM:SS'（UTC，字典序=时间序）。"""
    if not s:
        return ""
    t = str(s).strip().replace("T", " ")
    return t[:19]


def measure(universe: Dict[str, str],
            ko_created: Dict[str, str],
            events: Sequence[Event],
            consumers_with_report: set,
            merges_off: bool = False,
            dedup: bool = True) -> Dict[str, Any]:
    """纯函数：给出 L0..L3 阶梯与分域明细。所有输入都是原始证据，无业务常量。"""
    canon = (lambda x: x) if merges_off else (lambda x: universe.get(x, x))
    universe_canon = {canon(m) for m in universe}
    agg = dedup_events(events)

    cited_any: set = set()
    cited_sub: set = set()
    cited_ev: set = set()
    detail: Dict[str, List] = {}
    raw_sum = 0
    rows_cited = 0
    for (ko, consumer), v in agg.items():
        if not v["counted"]:
            continue
        c = canon(ko)
        if c not in universe_canon:
            continue
        rows_cited += 1
        raw_sum += v["count"] if not dedup else 1
        cited_any.add(c)
        if v["cited_at"] and v["cited_at"] >= ko_created.get(c, ""):
            cited_sub.add(c)
            if consumer in consumers_with_report:
                cited_ev.add(c)
                detail.setdefault(c, []).append(
                    {"mission_id": consumer, "cited_at": v["cited_at"],
                     "refs": v["refs"], "n_rows": v["n_rows"], "count": v["count"]})

    total = len(universe_canon)
    by_domain: Dict[str, Dict[str, int]] = {}
    for c in universe_canon:
        d = c.split(":", 1)[0]
        s = by_domain.setdefault(d, {"total": 0, "cited_evidenced": 0})
        s["total"] += 1
        if c in cited_ev:
            s["cited_evidenced"] += 1

    def rate(n: int) -> Optional[float]:
        return (n / total) if total else None

    return {
        "kpi": KPI,
        "schema": SCHEMA,
        "gates": {
            "L0_text_only": None,  # 由调用方用 text_scan 注入
            "L1_any": {"count": len(cited_any), "rate": rate(len(cited_any))},
            "L2_subsequent": {"count": len(cited_sub), "rate": rate(len(cited_sub))},
            "L3_evidenced": {"count": len(cited_ev), "rate": rate(len(cited_ev))},
        },
        "universe": {"canonical_total": total,
                     "raw_members": len(universe),
                     "alias_pairs_collapsed": len(universe) - total},
        "citation_events": {"deduped_consumer_ko_pairs": rows_cited,
                            "raw_unit_count_sum_or_pair": raw_sum,
                            "raw_rows": len(events)},
        "_sets": {"cited_any": sorted(cited_any), "cited_subsequent": sorted(cited_sub),
                  "cited_evidenced": sorted(cited_ev)},
        "by_namespace": by_domain,
        "cited_evidenced_detail": {k: sorted(v, key=lambda r: -int(r["mission_id"]))
                                   for k, v in sorted(detail.items())},
    }


# ---------------------------------------------------------------- v1 口径复现
_LEAD = r"(?<![A-Za-z0-9_:])"
_TAIL = r"(?![A-Za-z0-9_-])"


def combined_keys_regex(keys: Iterable[str]):
    ordered = sorted(set(keys), key=len, reverse=True)
    if not ordered:
        return re.compile(r"(?!)")
    return re.compile(_LEAD + "(" + "|".join(re.escape(k) for k in ordered) + ")" + _TAIL)


def text_scan(texts: Iterable[str], keys: Iterable[str]) -> set:
    """v1 口径：正文精确键引用（带前后边界，防前缀吞并）。"""
    pat = combined_keys_regex(keys)
    hit: set = set()
    for t in texts:
        if not t:
            continue
        hit.update(m.group(1) for m in pat.finditer(t))
    return hit


# ---------------------------------------------------------------- 取数（只读）
def _psql(psql_cmd: str, sql: str) -> str:
    r = subprocess.run(psql_cmd.split() + ["-A", "-F", "\t", "-t", "-c", sql],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"psql failed: {r.stderr.strip()[:400]}")
    return r.stdout


# 只按 \n 切行：str.splitlines() 还会在 \x0b/\x0c/\x85/\u2028 上切，
# 而报告正文里确有换页符（chr(12)），会把一行拆成两行、把后续字段拼错（本单实测踩到）。
def _rows(out: str) -> List[List[str]]:
    return [ln.split("\t") for ln in out.split("\n") if ln.strip()]


def fetch_evidence(psql_cmd: str) -> Dict[str, Any]:
    """只读快照（全部用 SELECT，不改任何数据）。"""
    kb = _rows(_psql(psql_cmd,
        "SELECT id, kb_key, ktype, domain, COALESCE(status,'active'), "
        "to_char(created_at AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS'), cited_count "
        "FROM knowledge_base ORDER BY id"))
    negk = _rows(_psql(psql_cmd,
        "SELECT entry_id, mission_id, COALESCE(status,'active'), "
        "to_char(registered_at AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') "
        "FROM negative_knowledge_ledger ORDER BY entry_id"))
    inj = _rows(_psql(psql_cmd,
        "SELECT injection_id, COALESCE(mission_id::text,''), kb_key, "
        "COALESCE(source,''), COALESCE(citation_count,0)::text, COALESCE(cited_at,''), "
        "COALESCE(citation_ref,''), COALESCE(injected_at,'') "
        "FROM knowledge_injections ORDER BY injection_id"))
    missions = _rows(_psql(psql_cmd,
        "SELECT id, COALESCE(report_id::text,'') FROM tensor_dev_missions"))
    reports = _rows(_psql(psql_cmd,
        "SELECT DISTINCT mission_id FROM tensor_reports "
        "WHERE mission_id IS NOT NULL AND COALESCE(length(markdown),0)>0"))
    cited_reports = _rows(_psql(psql_cmd,
        "SELECT mission_id, regexp_replace(markdown, '[' || chr(9) || chr(10) || chr(11) || chr(12) || chr(13) || ']', ' ', 'g') "
        "FROM tensor_reports WHERE mission_id IN (SELECT DISTINCT mission_id "
        "FROM knowledge_injections WHERE citation_count>0) "
        "AND COALESCE(length(markdown),0)>0 ORDER BY mission_id"))
    goals = _rows(_psql(psql_cmd,
        "SELECT id, regexp_replace(goal, '[' || chr(9) || chr(10) || chr(11) || chr(12) || chr(13) || ']', ' ', 'g'), "
        "to_char(created_at AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') "
        "FROM tensor_dev_missions"))
    return {"knowledge_base": kb, "negative_knowledge_ledger": negk,
            "knowledge_injections": inj, "missions": missions,
            "reports": reports, "cited_reports": cited_reports,
            "mission_goals": goals,
            "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}


def analyse(snap: Dict[str, Any], merges_off: bool = False) -> Dict[str, Any]:
    kb = snap["knowledge_base"]
    kb_by_id = {int(r[0]): r[1] for r in kb}
    kb_active = {r[1] for r in kb if r[4] == "active"}
    kb_created = {f"kb:{r[1]}": parse_ts(r[5]) for r in kb if r[4] == "active"}
    negk_active = {r[0] for r in snap["negative_knowledge_ledger"] if r[2] == "active"}
    negk_created = {f"negk:{r[0]}": parse_ts(r[3])
                    for r in snap["negative_knowledge_ledger"] if r[2] == "active"}
    negk_rows = [(r[0], int(r[1]) if r[1] else None) for r in snap["negative_knowledge_ledger"]]

    aliases = alias_map(kb_active, negk_rows)
    universe = dealiased_universe(kb_active, negk_active, aliases)
    ko_created = {**negk_created, **kb_created}
    # 别名折叠后，canonical 的建档时间取两者中较早者（保守：更容易满足"后续性"）
    for a, c in aliases.items():
        if a in ko_created and c in ko_created:
            ko_created[c] = min(ko_created[a], ko_created[c])

    events: List[Event] = []
    for r in snap["knowledge_injections"]:
        ko = canonical_ko_id(r[2], kb_by_id, kb_active, negk_active)
        if ko is None:
            # 解析不了的键：登记为 unresolved，绝不静默丢弃
            events.append(Event(ko=f"?{r[2]}", consumer=int(r[1]) if r[1] else None,
                                count=int(r[4] or 0), cited_at=parse_ts(r[5]),
                                ref=r[6], source=r[3], injected_at=r[7]))
            continue
        events.append(Event(ko=ko, consumer=int(r[1]) if r[1] else None,
                            count=int(r[4] or 0), cited_at=parse_ts(r[5]),
                            ref=r[6], source=r[3], injected_at=r[7]))

    with_report = {int(r[0]) for r in snap["reports"]}
    v1_gate_missions = {int(r[0]) for r in snap["missions"] if r[1]}
    res = measure(universe, ko_created, events, with_report, merges_off=merges_off)
    canon = (lambda x: x) if merges_off else (lambda x: universe.get(x, x))
    cited_any = set(res["_sets"]["cited_any"])
    total = res["universe"]["canonical_total"]

    # L0：v1 口径复现（goal 正文 + report_id 证据门 + 后续性）
    seen = text_scan([r[1] for r in snap["mission_goals"]], kb_active)
    l0_ev = set()
    for r in snap["mission_goals"]:
        mid, g = int(r[0]), r[1]
        m_created = parse_ts(r[2]) if len(r) > 2 else ""
        if mid not in v1_gate_missions:      # v1 证据门：report_id 非空
            continue
        for k in text_scan([g], kb_active):
            if m_created >= kb_created.get(f"kb:{k}", ""):   # v1 后续性门
                l0_ev.add(k)
    res["gates"]["L0_text_only"] = {
        "count": len(l0_ev),
        "rate": (len(l0_ev) / len(kb_active)) if kb_active else None,
        "denominator": "knowledge_base[active]（v1 原口径分母）",
        "cited_keys": sorted(seen),
        "n_missions_with_report_id": len(v1_gate_missions),
    }
    # L4（bibliometric "direct citation" 严格层）：引用方报告正文里真的出现该 KO 键
    reports_by_mission: Dict[int, List[str]] = {}
    for r in snap.get("cited_reports", []):
        reports_by_mission.setdefault(int(r[0]), []).append(r[1])
    raw_key_by_canon: Dict[str, set] = {}
    for r in snap["knowledge_injections"]:
        c = canonical_ko_id(r[2], kb_by_id, kb_active, negk_active)
        if c:
            raw_key_by_canon.setdefault(c, set()).add(r[2])
    agg_all = dedup_events(events)
    l4 = set()
    for (ko, consumer), v in agg_all.items():
        if not v["counted"] or ko.startswith("?") or consumer is None:
            continue
        text = " ".join(reports_by_mission.get(int(consumer), []))
        if not text:
            continue
        needles = set(raw_key_by_canon.get(ko, set())) | {ko.split(":", 1)[1]}
        if any(n and n in text for n in needles):
            l4.add(canon(ko))
    res["gates"]["L4_direct_text"] = {
        "count": len(l4), "rate": (len(l4) / total) if total else None,
        "rule": "引用方 tensor_reports.markdown 正文出现该 KO 键（bibliometric direct citation）",
        "detail": sorted(l4),
    }
    # 朴素口径对照：逐行求和 citation_count（v1 若直接求和会得到的"引用次数"）
    naive = sum(e.count for e in events if e.count > 0)
    worst = max(((v["count"], v["n_rows"], k) for k, v in agg_all.items() if v["counted"]),
                default=(0, 0, ("", None)))
    res["counter_audit"] = {
        "naive_row_sum": naive,
        "dedup_pairs": sum(1 for v in agg_all.values() if v["counted"]),
        "inflation_factor": (naive / max(1, sum(1 for v in agg_all.values() if v["counted"]))),
        "worst_pair": {"max_count": worst[0], "rows": worst[1],
                       "pair": [worst[2][0], worst[2][1]]},
    }
    # 自报 cited_count>0 vs 证据法
    self_rep = {f"kb:{r[1]}" for r in kb if int(r[6] or 0) > 0}
    cited_any_canon = {canon(c) for c in cited_any}
    res["self_report_consistency"] = {
        "self_reported_positive": len(self_rep),
        "evidence_positive": len(cited_any_canon),
        "self_only": sorted(self_rep - cited_any_canon)[:20],
        "evidence_only": sorted(cited_any_canon - self_rep)[:20],
        "both": len(self_rep & cited_any_canon),
    }
    res["unresolved_keys"] = sorted({e.ko for e in events if e.ko.startswith("?")})
    res["alias_pairs"] = len(aliases)
    res["evidence"] = {
        "n_reports": len(snap["reports"]),
        "n_missions_with_report": len(with_report),
        "n_missions_total": len(snap["missions"]),
        "n_missions_with_report_id_v1": len(v1_gate_missions),
    }
    res["run"] = {"fetched_at_utc": snap.get("fetched_at_utc"),
                  "n_injections": len(snap["knowledge_injections"]),
                  "n_kb": len(kb_active), "n_negk": len(negk_active),
                  "merges_off": merges_off}
    # 未解析键的引用事件也算"引用存在"，但无 KO 归属 → 单列风险值
    res["unresolved_injections"] = sum(1 for e in events if e.ko.startswith("?"))
    return res


# ---------------------------------------------------------------- 报告渲染
def render_md(res: Dict[str, Any]) -> str:
    g = res["gates"]
    lines = [
        f"# {KPI} v2 实测（程序化注入，零手抄）",
        "",
        f"- 生成时间: {res['run']['fetched_at_utc']} UTC",
        f"- KO 宇宙: {res['universe']['canonical_total']} 个规范 KO"
        f"（原始成员 {res['universe']['raw_members']}，折叠别名 {res['universe']['alias_pairs_collapsed']} 对）",
        f"- 台账行: {res['citation_events']['raw_rows']} 行注入；去重后 (KO,consumer) 对 "
        f"{res['citation_events']['deduped_consumer_ko_pairs']} 个",
        f"- 产物台账: {res['evidence']['n_reports']} 份报告 / "
        f"{res['evidence']['n_missions_with_report']} mission（v1 用的 report_id 仅 "
        f"{res['evidence']['n_missions_with_report_id_v1']} 个）",
        "",
        "## 口径阶梯（预注册顺序 L0→L3）",
        "",
        "| gate | 定义 | 分子 | 分母 | reuse_rate |",
        "|---|---|---|---|---|",
    ]
    den = res["universe"]["canonical_total"]
    d0 = res["evidence"].get("n_kb") or 0
    l0 = g["L0_text_only"]
    def fmt(rate):
        return "-" if rate is None else f"{rate:.4f}"

    lines.append(f"| L0_text_only | goal 正文精确键引用 + report_id 证据门（v1） | {l0['count']} | "
                 f"{res['run']['n_kb']} | {fmt(l0['rate'])} |")
    for name in ("L1_any", "L2_subsequent", "L3_evidenced", "L4_direct_text"):
        v = g[name]
        lines.append(f"| {name} | 见模块 docstring | {v['count']} | {den} | {fmt(v['rate'])} |")
    ca = res["counter_audit"]
    lines += ["", "## 计数口径审计（D3）", "",
              f"- 逐行求和 citation_count（虚拟增口径）: {ca['naive_row_sum']}",
              f"- 去重后 (KO,consumer) 引用主体数: {ca['dedup_pairs']}",
              f"- 全局虚增倍数: {ca['inflation_factor']:.4f}",
              f"- 最坏一对: count={ca['worst_pair']['max_count']} / rows={ca['worst_pair']['rows']} "
              f"@ {ca['worst_pair']['pair']}",
              "", "## 分命名空间", "", "| namespace | KO | 有证据引用 | 域内复用率 |", "|---|---|---|---|"]
    for ns, v in sorted(res["by_namespace"].items(), key=lambda kv: -kv[1]["total"]):
        r = v["cited_evidenced"] / v["total"] if v["total"] else 0.0
        lines.append(f"| {ns} | {v['total']} | {v['cited_evidenced']} | {r:.4f} |")
    lines += ["", "## 明细（L3 命中）", ""]
    for ko, rows in res["cited_evidenced_detail"].items():
        mids = ",".join(str(r["mission_id"]) for r in rows[:6])
        more = f" …(+{len(rows)-6})" if len(rows) > 6 else ""
        lines.append(f"- `{ko}` ← mission {mids}{more}")
    if res["unresolved_keys"]:
        lines += ["", "## 未解析键（风险）", ""] + [f"- `{k}`" for k in res["unresolved_keys"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/kpi_kb_reuse_rate_v2")
    ap.add_argument("--psql-cmd", default=_PSQL_DEFAULT)
    ap.add_argument("--snapshot", default=None,
                    help="已有证据快照 JSON（离线复算，免连库）")
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.snapshot:
        snap = json.loads(Path(args.snapshot).read_text())
    else:
        snap = fetch_evidence(args.psql_cmd)
        (outdir / "evidence_snapshot.json").write_text(
            json.dumps(snap, ensure_ascii=False))

    variants = {}
    for merges_off in (False, True):
        res = analyse(snap, merges_off=merges_off)
        tag = "alias_merged" if not merges_off else "alias_split"
        variants[tag] = res
        (outdir / f"measure_{tag}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2))
        (outdir / f"measure_{tag}.md").write_text(render_md(res))

    main_res = variants["alias_merged"]
    print(json.dumps({k: (v["rate"] if isinstance(v, dict) else v)
                      for k, v in main_res["gates"].items()}, ensure_ascii=False))
    print(f"universe={main_res['universe']['canonical_total']} "
          f"L3={main_res['gates']['L3_evidenced']['count']} "
          f"-> wrote {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())