import json
import psycopg
conn = psycopg.connect("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz", connect_timeout=10)
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='tensor_benchmark_cases'")
print("cols:", [r[0] for r in cur.fetchall()])
cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
print("total:", cur.fetchone())
cur.execute("SELECT anchor_slug, case_name FROM tensor_benchmark_cases WHERE case_name ILIKE '%sphere%' OR anchor_slug ILIKE '%sphere%' OR case_name ILIKE '%havelock%' OR anchor_slug ILIKE '%havelock%' OR case_name ILIKE '%submerged%'")
for r in cur.fetchall():
    print(r)
cur.execute("SELECT anchor_slug FROM tensor_benchmark_cases WHERE anchor_slug LIKE 'CASE-BEM%' ORDER BY 1")
print("CASE-BEM anchors:")
for r in cur.fetchall():
    print(" ", r[0])
conn.close()
