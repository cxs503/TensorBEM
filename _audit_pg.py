import json
try:
    import psycopg
    print("psycopg v3 available")
    conn = psycopg.connect("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz", connect_timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT case_name, metric, tolerance, ref_source, status FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-SUBMERGED-SPHERE-HAVELOCK'")
    rows = cur.fetchall()
    print("ROWS:", len(rows))
    for r in rows:
        print(r)
    cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
    print("total benchmark cases:", cur.fetchone())
    cur.execute("SELECT anchor_slug, case_name, metric, tolerance FROM tensor_benchmark_cases ORDER BY anchor_slug")
    for r in cur.fetchall():
        print(r)
    conn.close()
except Exception as e:
    print("ERR:", type(e).__name__, str(e)[:300])
