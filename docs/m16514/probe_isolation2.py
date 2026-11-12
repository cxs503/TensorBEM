"""Probe 2: which ``tensorbem`` does a *repository test file* import, and does
pinning ``tensorbem.__path__`` restore isolation?

Run with the shadow tree on ``PYTHONPATH``.  ``tests/conftest.py`` prepends the
repository ``src`` to ``sys.path``, so the plain run may silently exercise the
unmutated package.  The second run loads a ``-p`` plugin that pins the package
``__path__`` to the shadow tree before ``conftest`` is imported.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # docs/m16514/ -> repo root

PLUGIN = '''
import os, pathlib, sys

def pytest_configure(config):
    shadow = pathlib.Path(os.environ["PROBE_SHADOW"]).resolve()
    import tensorbem
    repo_src = pathlib.Path(os.environ["PROBE_REPO_SRC"]).resolve()
    sys.path[:] = [p for p in sys.path if pathlib.Path(p or ".").resolve() != repo_src]
    sys.modules["tensorbem"].__path__ = [str(shadow / "tensorbem")]
    import tensorbem.rao
    print("PINNED_IMPORT", tensorbem.rao.__file__)
'''

WHICH = '''
def test_which():
    import tensorbem
    import tensorbem.rao
    print("SESSION_IMPORT", tensorbem.rao.__file__)
'''


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="probe2-"))
    shadow = work / "src"
    shutil.copytree(REPO / "src" / "tensorbem", shadow / "tensorbem")
    target = shadow / "tensorbem" / "rao.py"
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace("m + added_mass", "m - added_mass", 1), encoding="utf-8")

    plugin_dir = work / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "pt_probe.py").write_text(PLUGIN, encoding="utf-8")
    (REPO / "tests" / "_probe_which_import.py").write_text(WHICH, encoding="utf-8")

    env = {
        "PYTHONPATH": str(shadow),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PROBE_SHADOW": str(shadow),
        "PROBE_REPO_SRC": str(REPO / "src"),
    }
    base = [sys.executable, "-m", "pytest", "tests/test_rao_verification.py", "-q",
            "--no-header", "--tb=no", "-p", "no:cacheprovider"]

    plain = subprocess.run(base, cwd=str(REPO), env=env, capture_output=True, text=True)
    print("plain run     rc=", plain.returncode, "|", (plain.stdout).strip().splitlines()[-1])

    env2 = dict(env, PYTHONPATH=os.pathsep.join([str(plugin_dir), str(shadow)]))
    pinned = subprocess.run(
        base + ["-p", "pt_probe"], cwd=str(REPO), env=env2,
        capture_output=True, text=True,
    )
    out = pinned.stdout + pinned.stderr
    for line in out.splitlines():
        if "PINNED_IMPORT" in line:
            print("pinned  import->", line.split("PINNED_IMPORT", 1)[1].strip())
        if "SESSION_IMPORT" in line:
            print("session import->", line.split("SESSION_IMPORT", 1)[1].strip())
    print("pinned run    rc=", pinned.returncode, "|", out.strip().splitlines()[-1])

    (REPO / "tests" / "_probe_which_import.py").unlink()
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())