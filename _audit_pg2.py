import json
import psycopg
conn = psycopg.connect("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz", connect_timeout=10)
cur = conn.cursor()
cur.execute("SELECT anchor_slug, case_name, metric, tolerance, ref_source, status, repo FROM tensor_benchmark_cases WHERE anchor_slug LIKE '%SUBMERGED%' OR anchor_slug LIKE '%HAVELOCK%'")
rows = cur.fetchall()
print("ROWS:", len(rows))
for r in rows:
    print(json.dumps(dict(zip(["anchor_slug","case_name","metric","tolerance","ref_source","status","repo"], r)), ensure_ascii=False, indent=1))
conn.close()
