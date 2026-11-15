import tarfile, pathlib
root = pathlib.Path(".").resolve()
want = ("benchmarks/verified/case_bem_laplace_cylinder/",)
tar = tarfile.open("/root/tp-agent/cluster/fetch/cj3195_n315.tar.gz")
done = []
for member in tar.getmembers():
    name = member.name.lstrip("./")
    index = name.find("benchmarks/verified/case_bem_laplace_cylinder/")
    if index < 0 or not member.isfile():
        continue
    rel = name[index:]
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(tar.extractfile(member).read())
    done.append(rel)
tar.close()
print(len(done), "files restored")
for rel in sorted(done):
    print(" ", rel)
