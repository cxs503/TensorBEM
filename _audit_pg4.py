import psycopg
conn = psycopg.connect("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz", connect_timeout=10)
cur = conn.cursor()
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='tensor_maturity_verdicts' ORDER BY ordinal_position")
for r in cur.fetchall():
    print(r)
cur.execute("SELECT count(*) FROM tensor_maturity_verdicts")
print("rows:", cur.fetchone())
cur.execute("SELECT repo, scope, mission_id, level, verdict_by FROM tensor_maturity_verdicts ORDER BY id DESC LIMIT 5")
for r in cur.fetchall():
    print(r)
conn.close()
