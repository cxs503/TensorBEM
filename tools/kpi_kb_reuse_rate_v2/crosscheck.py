#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立复算路径：纯 SQL 一条查询给出 L1(任意引用) 与 L3(有产物佐证) 分子。

注意：``knowledge_injections.cited_at`` 是 TEXT 且写成 ``YYYY-MM-DDTHH:MM:SS+00:00`` 形式，
直接与 ``YYYY-MM-DD HH:MM:SS`` 做文本比较会因 'T'(0x54) > ' '(0x20) 把同一天全部误排除，
故所有时间过滤都先 ``replace(cited_at,'T',' ')``。

与 measure.py（Python 解析）**不共享任何解析代码**：键空间解析、别名折叠、
去重全部用 SQL 集合语义表达。两条路径必须给出一致的数字（本单误差 0），
否则视为实现缺陷。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SQL_L1 = r"""
WITH cited AS (
    SELECT DISTINCT mission_id, kb_key FROM knowledge_injections
     WHERE citation_count > 0 AND replace(cited_at,'T',' ') <= '{asof}'
), resolved AS (
    SELECT c.mission_id,
           CASE
             WHEN c.kb_key ~ '^kb:[0-9]+$'
               THEN (SELECT 'kb:' || kb_key FROM knowledge_base
                      WHERE id = substring(c.kb_key from 4)::int)
             WHEN c.kb_key LIKE 'negk:%'
               THEN COALESCE(
                      (SELECT 'negk:' || substring(c.kb_key from 6) FROM negative_knowledge_ledger
                        WHERE entry_id IN (c.kb_key, substring(c.kb_key from 6)) LIMIT 1), NULL)
             ELSE (SELECT 'kb:' || kb_key FROM knowledge_base WHERE kb_key = c.kb_key)
           END AS ko
      FROM cited c
), canon AS (
    SELECT DISTINCT r.mission_id,
           COALESCE((SELECT 'kb:lesson:m' || n.mission_id
                       FROM negative_knowledge_ledger n JOIN knowledge_base k
                         ON k.kb_key = 'lesson:m' || n.mission_id::text
                      WHERE 'negk:' || n.entry_id = r.ko
                         OR n.entry_id = substring(r.ko from 6)
                         OR n.entry_id = r.ko LIMIT 1), r.ko) AS ko
      FROM resolved r WHERE r.ko IS NOT NULL
), uni AS (
    -- 宇宙口径与 measure.py 对齐：只认 active 的 kb KO（负知识现网全 active）
    SELECT cz.ko, cz.mission_id FROM canon cz
     WHERE COALESCE((SELECT COALESCE(status,'active') FROM knowledge_base k
                      WHERE 'kb:' || k.kb_key = cz.ko), 'active') = 'active'
)
SELECT (SELECT count(DISTINCT ko) FROM uni) AS l1,
       (SELECT count(DISTINCT c.ko) FROM uni c
         WHERE EXISTS (SELECT 1 FROM tensor_reports r
                        WHERE r.mission_id = c.mission_id
                          AND COALESCE(length(r.markdown),0) > 0)) AS l3
"""

SQL_UNIVERSE = r"""
WITH kb AS (SELECT 'kb:' || kb_key AS ko FROM knowledge_base
             WHERE COALESCE(status,'active') = 'active'),
mirror AS (SELECT 'negk:' || n.entry_id AS raw FROM negative_knowledge_ledger n
            WHERE COALESCE(n.status,'active') = 'active'
              AND NOT EXISTS (SELECT 1 FROM knowledge_base k
                               WHERE k.kb_key = 'lesson:m' || n.mission_id::text))
SELECT (SELECT count(*) FROM kb) AS n_kb,
       (SELECT count(*) FROM kb) + (SELECT count(*) FROM mirror) AS n_union
"""


def psql(cmd: str, sql: str):
    r = subprocess.run(cmd.split() + ["-A", "-F", "\t", "-t", "-c", sql],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    return [ln.split("\t") for ln in r.stdout.split("\n") if ln.strip()]


def main(argv=None) -> int:
    psql_cmd = "runuser -u postgres -- psql -d tensorplatform"
    out = Path(arguments[0]) if (arguments := (argv or sys.argv[1:])) else Path(
        "runs/kpi_kb_reuse_rate_v2")
    # 与 Python 路线对齐到同一时间截面：以 measure_*.json 的抓取时刻为上界，
    # 否则台账在两次运行之间仍在增长，差异无法归因（本单实测 14 行/3 键/分钟量级）。
    ref = out / "measure_alias_merged.json"
    asof = json.loads(ref.read_text())["run"]["fetched_at_utc"] if ref.exists() else None
    sql = SQL_L1.format(asof=asof) if asof else SQL_L1.replace(
        "AND replace(cited_at,'T',' ') <= '{asof}'", "")
    l1, l3 = map(int, psql(psql_cmd, sql)[0])
    n_kb, n_union = map(int, psql(psql_cmd, SQL_UNIVERSE)[0])
    res = {"route": "sql-independent", "asof_utc": asof, "L1_any": l1, "L3_evidenced": l3,
           "universe_kb": n_kb, "universe_union": n_union}
    out.mkdir(parents=True, exist_ok=True)
    (out / "crosscheck_sql.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
