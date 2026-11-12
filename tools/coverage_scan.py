import os, re, glob, collections

root = "/root/tp-agent/work/m16890"
mods = sorted(os.path.basename(p)[:-3] for p in glob.glob(root + "/src/tensorbem/*.py"))
docs = []
for d in glob.glob(root + "/docs/**/*.md", recursive=True) + glob.glob(root + "/*.md") + glob.glob(root + "/docs/*.json"):
    try:
        docs.append((d, open(d, errors="ignore").read()))
    except Exception:
        pass
tests = []
for d in glob.glob(root + "/tests/**/*.py", recursive=True):
    try:
        tests.append((d, open(d, errors="ignore").read()))
    except Exception:
        pass
rows = []
for m in mods:
    pat = re.compile(r"\btensorbem\." + re.escape(m) + r"\b|\bimport " + re.escape(m) + r"\b")
    nd = sum(1 for _, t in docs if pat.search(t))
    nt = sum(1 for _, t in tests if pat.search(t))
    rows.append((nt, nd, m))
for nt, nd, m in sorted(rows):
    print(f"tests={nt:3d} docs={nd:3d}  {m}")