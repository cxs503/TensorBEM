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
    "/tensor/case-pool?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/case-pool",
    "/tensor/case_pool?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark-cases/anchor/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark/case/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark-case/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark-cases/anchor/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/anchors?slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/anchor/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/case-anchors?slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark/anchor/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark/anchors/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/cases/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/case/CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark-cases?case_name=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
    "/tensor/benchmark-cases?anchor=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
]
for p in cands:
    s, d = get(p)
    print(s, p, "->", str(d)[:200])
