import os, json, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
tok = open("/tmp/tok.txt").read().strip()

def get(path):
    r = urllib.request.Request(base + path)
    r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except Exception as e:
        return -1, {"err": str(e)}

cands = [
    "/tensor/missions/17334",
    "/tensor/mission/17334",
    "/tensor/reports?mission_id=17334",
    "/tensor/reports?mission_id=17334&limit=5",
    "/tensor/benchmark-cases?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark-cases",
    "/tensor/benchmark/cases?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmarks?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/anchors/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/case-anchors/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/maturity-verdicts?scope=m17334",
]
for p in cands:
    s, d = get(p)
    print(s, p, "->", str(d)[:300])
