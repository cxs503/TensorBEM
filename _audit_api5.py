import os, json, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
tok = open("/tmp/tok.txt").read().strip()

def get(path):
    r = urllib.request.Request(base + path)
    r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except Exception as e:
        return -1, {"err": str(e)}

s, d = get("/tensor/missions/17334")
print("MISSION", s)
json.dump(d, open("/tmp/m17334_mission.json", "w"), ensure_ascii=False, indent=1)
for k in ["id", "repo", "status", "goal", "acceptance", "result", "verdict", "maturity"]:
    if k in d:
        print("==", k, ":", str(d[k])[:1500])

s, d = get("/tensor/reports?mission_id=17334&limit=10")
print("REPORTS", s, len(d.get("items", [])))
for it in d.get("items", []):
    print(" -", it.get("id"), it.get("title")[:100])
json.dump(d, open("/tmp/m17334_reports.json", "w"), ensure_ascii=False, indent=1)
