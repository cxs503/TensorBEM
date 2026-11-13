#!/usr/bin/env python3
"""Wave-3 gap-reference pack for anchor FEAT-OSS-INVENTORY-SPH.

Wave 1 of the campaign (`docs/oss_inventory_sph/registry.json`, mission m15815)
surveys 23 public SPH codes but only at *documentation / file-tree* level: its own
`evidence_grade_note` says "Neither channel reads third-party source code".
Wave 2 produced prose deep dives for a handful of codes.

This module fills the third, code-level channel *for the five open GAP-SPH-*
anchors*: for each gap it pins the upstream repositories that actually ship the
missing capability and extracts, programmatically, a `path:line` + verbatim
source quote at a frozen commit. Nothing is transcribed by hand: the quote is
whatever the file holds at that line, and `verify` re-derives it.

Usage
-----
    python3 scripts/oss_sph_inventory/gap_refs.py build  --root /tmp/sph
    python3 scripts/oss_sph_inventory/gap_refs.py verify --root /tmp/sph
    python3 scripts/oss_sph_inventory/gap_refs.py report            # md + png

`--root` is a directory holding one shallow clone per repository (directory name
given by `repo.dir` below, cloned at the pinned commit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path("docs/oss_sph_inventory/gap_refs") if Path("docs").is_dir() else Path(
    "docs/oss_inventory_sph/gap_refs"
)
EVIDENCE_NAME = "code_evidence.json"

# ---------------------------------------------------------------------------
# Pinned sources. Each entry: local clone directory -> (full_name, commit, licence)
# ---------------------------------------------------------------------------
REPOS = {
    "gpusph": {
        "full_name": "GPUSPH/gpusph",
        "commit": "08ffd94ddc4a076361f0e5f6e2bfde7874ba087f",
        "commit_date": "2021-05-29",
        "language": "C/C++/CUDA",
        "licence": "GPL-3.0",
        "why": "only surveyed code that combines MPI domain decomposition with a "
        "volume-based semi-analytical (SA) boundary model and an SPS/k-epsilon "
        "turbulence closure -- one source per open gap.",
    },
    "jax-sph": {
        "full_name": "tumaer/jax-sph",
        "commit": "808f87489e2d4d1f39efa69709fda45118d8f7f3",
        "commit_date": "2025-10-29",
        "language": "Python/JAX",
        "licence": "MIT",
        "why": "density-diffusion (delta-SPH) and Riemann-solver formulations plus a "
        "ghost-wall boundary condition, all in <=700 lines of readable Python and "
        "shipped with a validation harness + reference data.",
    },
    "salva": {
        "full_name": "dimforge/salva",
        "commit": "34eda88e559248f21e4dbbb5894a7df997d16fb3",
        "commit_date": "2026-11-01",
        "language": "Rust",
        "licence": "Apache-2.0",
        "why": "the only surveyed DFSPH *and* IISPH projection pair with explicit "
        "divergence/density-error stopping criteria -- the acceptance variable of "
        "GAP-SPH-ISPH-PROJECTION.",
    },
}

# ---------------------------------------------------------------------------
# Claim specs. `anchor` is a regular expression; the claim's line is the first
# match, and the quote is that line verbatim (stripped). `gap` binds the claim to
# a feature_ledger anchor proposed in docs/oss_inventory_sph/gaps.json.
# ---------------------------------------------------------------------------
SPECS = [
    # ---- GAP-SPH-MPI-DOMAIN ----
    dict(id="MPI-01", repo="gpusph", gap="GAP-SPH-MPI-DOMAIN",
         path="src/NetworkManager.cc", anchor=r"MPI_Comm_size\(MPI_COMM_WORLD",
         statement="MPI world size/rank are taken at start-up and every device is "
                   "mapped to a global (rank, dev) pair, so a domain can span nodes.",
         port="TensorSPH would need the same global-device indexing before any halo "
              "exchange can be addressed."),
    dict(id="MPI-02", repo="gpusph", gap="GAP-SPH-MPI-DOMAIN",
         path="src/NetworkManager.cc", anchor=r"void NetworkManager::sendBuffer",
         statement="Halo exchange is an explicit point-to-point byte-buffer send "
                   "(count, void* src_data), not a collective.",
         port="Halo payload for TensorSPH = particle SoA slice; message = raw bytes, "
              "so the same routine serves float32 and float64."),
    dict(id="MPI-03", repo="gpusph", gap="GAP-SPH-MPI-DOMAIN",
         path="src/NetworkManager.cc", anchor=r"MPI_Recv\(dst_data, count, MPI_BYTE",
         statement="The matching receive lands the halo directly into the destination "
                   "device buffer (zero extra copy inside the manager).",
         port="Mirror the send/recv pair over a dict of (src,dst) -> buffer handles."),
    dict(id="MPI-04", repo="gpusph", gap="GAP-SPH-MPI-DOMAIN",
         path="src/NetworkManager.h", anchor=r"class NetworkManager",
         statement="All inter-rank traffic is funnelled through one object that owns "
                   "the tag space and the device-pair routing.",
         port="Single choke point is what makes the 1-rank/2-rank equivalence test "
              "inspectable; keep TensorSPH's exchange in one module too."),
    # ---- GAP-SPH-AKINCI-BOUNDARY ----
    dict(id="BND-01", repo="gpusph", gap="GAP-SPH-AKINCI-BOUNDARY",
         path="src/cuda/boundary_conditions_kernel.cu",
         anchor=r"Auxiliary structures and functions for SA_BOUNDARY",
         statement="Boundary treatment is isolated in its own kernel translation unit "
                   "specialised to SA_BOUNDARY (semi-analytical walls), i.e. one "
                   "selectable boundary model, not a hard-wired plane reflection.",
         port="TensorSPH can add a volume-weighted boundary module behind the same "
              "interface its Adami extrapolation uses today."),
    dict(id="BND-02", repo="gpusph", gap="GAP-SPH-AKINCI-BOUNDARY",
         path="src/cuda/buildneibs_kernel.cu",
         anchor=r"Specific variables used in neibsInCell with SA boundary",
         statement="Boundary particles are counted in the neighbour list with a "
                   "dedicated per-particle fluid+boundary counter, which is what "
                   "volume-weighted boundary forces need.",
         port="Requires the neighbour structure to distinguish fluid vs boundary "
              "neighbours; TensorSPH's neighbours3 currently does not."),
    dict(id="BND-03", repo="gpusph", gap="GAP-SPH-AKINCI-BOUNDARY",
         path="src/cuda/sa_bc_params.h", anchor=r"Turbulent Kinetic Energy",
         statement="The SA boundary path optionally carries k-epsilon turbulence "
                   "state alongside the wall data.",
         port="Boundary data and turbulence state must be co-located for wall "
              "functions; a separate array per rank would break locality."),
    dict(id="BND-04", repo="jax-sph", gap="GAP-SPH-AKINCI-BOUNDARY",
         path="jax_sph/solver.py", anchor=r"def gwbc_fn_riemann_wrapper",
         statement="Alternative reference: ghost-wall boundary conditions implemented "
                   "as a velocity/pressure mirror rather than as boundary particles.",
         port="Cheapest boundary upgrade for TensorSPH (no new particle class); "
              "Akinci is only needed when wall shear accuracy dominates."),
    # ---- GAP-SPH-ISPH-PROJECTION ----
    dict(id="PRJ-01", repo="salva", gap="GAP-SPH-ISPH-PROJECTION",
         path="src/solver/pressure/dfsph_solver.rs", anchor=r"pub min_pressure_iter",
         statement="DFSPH exposes min/max pressure-iteration bounds as public solver "
                   "state (defaults 1/50), i.e. the projection is an iterative "
                   "fixed-point with explicit termination.",
         port="TensorSPH's acceptance wants density fluctuation <=1e-3; the number of "
              "iterations that buys it must be a reported quantity, not a constant."),
    dict(id="PRJ-02", repo="salva", gap="GAP-SPH-ISPH-PROJECTION",
         path="src/solver/pressure/dfsph_solver.rs",
         anchor=r"max_divergence_error: na::convert",
         statement="DFSPH stops on a *divergence* error threshold (default 0.1) -- a "
                   "different stopping variable from WCSPH's acoustic density error.",
         port="The GAP acceptance should report residual divergence, not only rho/rho0."),
    dict(id="PRJ-03", repo="salva", gap="GAP-SPH-ISPH-PROJECTION",
         path="src/solver/pressure/iisph_solver.rs",
         anchor=r"max_density_error: na::convert",
         statement="The sibling IISPH solver instead stops on average density error "
                   "(default 0.05), giving the two standard incompressibility "
                   "criteria side by side.",
         port="Pick one criterion per benchmark; mixing them hides the accuracy gain."),
    dict(id="PRJ-04", repo="salva", gap="GAP-SPH-ISPH-PROJECTION",
         path="src/solver/pressure/mod.rs", anchor=r"pub use self::iisph_solver",
         statement="IISPH is re-exported from the same pressure-solver module that "
                   "exports DFSPHSolver, so switching projection is a type change.",
         port="Mirror as a strategy object in TensorSPH rather than a branch in the "
              "time-integration loop."),
    # ---- GAP-SPH-SPS-LES ----
    dict(id="SPS-01", repo="gpusph", gap="GAP-SPH-SPS-LES",
         path="src/cuda/visc_params.h", anchor=r"struct tau_sps_params",
         statement="Sub-particle-scale stresses are carried as an explicit tau array, "
                   "filled only when the SPS_STORE_TAU flag is set.",
         port="SPS-LES in TensorSPH needs one extra per-particle tensor field, "
              "allocated on demand to keep the WCSPH path unchanged."),
    dict(id="SPS-02", repo="gpusph", gap="GAP-SPH-SPS-LES",
         path="src/cuda/forces_params.h", anchor=r"static const TurbulenceModel turbmodel",
         statement="The turbulence closure is a compile-time template parameter "
                   "resolved per viscosity spec, i.e. the model is a policy of the "
                   "force kernel.",
         port="Same policy pattern keeps TensorSPH's kernel-type dispatch (a hot "
              "path) branch-free."),
    # ---- accuracy lever shared by several gaps ----
    dict(id="ACC-01", repo="jax-sph", gap="GAP-SPH-ISPH-PROJECTION",
         path="jax_sph/solver.py", anchor=r"c_ref \* delta \* support \* diff_term",
         statement="delta-SPH density evolution: the continuity sum gets an additive "
                   "c_ref*delta*support*diff_term -- a diffusion term that suppresses "
                   "the acoustic floor without any projection step.",
         port="Cheapest partial answer to the WCSPH noise documented in "
              "verification_startup_quiescence.md: ~2 lines in the density update, "
              "with delta as a tunable (default 0.1 in jax-sph defaults.py)."),
    dict(id="ACC-02", repo="jax-sph", gap="GAP-SPH-ISPH-PROJECTION",
         path="jax_sph/solver.py", anchor=r"U_star = U_avg \+ 0\.5 \* \(p_L - p_R\)",
         statement="Riemann-solver formulation (Zhang 2017 eq. 7): an interface star "
                   "velocity built from the pressure jump, i.e. dissipative upwinding "
                   "inside the density evolution instead of an added diffusion term.",
         port="Two mutually exclusive accuracy levers; TensorSPH should benchmark "
              "delta-SPH first (no EOS change) and Riemann second."),
    dict(id="ACC-03", repo="jax-sph", gap="GAP-SPH-OBV",
         path="validation/validate.py", anchor=r"^def val_TGV\(",
         statement="Each validation case is a function comparing a run against a "
                   "closed-form/reference series (here Taylor-Green at two "
                   "resolutions), i.e. the repository ships its own "
                   "reference-value harness.",
         port="Directly reusable pattern for the benchmark judge that TensorSPH "
              "benchmarks still lack."),
    dict(id="ACC-04", repo="jax-sph", gap="GAP-SPH-OBV",
         path="jax_sph/kernel.py", anchor=r"class QuinticKernel",
         statement="Kernel family is a class with analytic w and grad_w per "
                   "dimensionality, so a new kernel is a class, not an if-branch.",
         port="TensorSPH kernel3.py already has the same shape -- confirms the "
              "design, no action."),
    dict(id="BND-05", repo="gpusph", gap="GAP-SPH-AKINCI-BOUNDARY",
         path="src/cuda/boundary_conditions_kernel.cu",
         anchor=r"if \(pout\.shepard_div > 0\.1f\*pdata\.gam\)",
         statement="Boundary kernels are renormalised by their Shepard divisor with a "
                   "0.1*gamma floor guard -- the numerical core of a volume-weighted "
                   "boundary, where the wall must not divide by a near-empty "
                   "neighbourhood.",
         port="Any volume boundary in TensorSPH needs this guard, otherwise the first "
              "wall layer spikes on re-entry."),
    dict(id="PRJ-05", repo="salva", gap="GAP-SPH-ISPH-PROJECTION",
         path="src/solver/pressure/dfsph_solver.rs",
         anchor=r"if avg_err <= max_err && i >= self\.min_divergence_iter",
         statement="The divergence-free projection exits only when the mean residual "
                   "divergence is under threshold *and* the minimum iteration count is "
                   "met -- the concrete form of the criterion GAP-SPH-ISPH-PROJECTION "
                   "must report.",
         port="TensorSPH's acceptance test can reuse this exit criterion and log "
              "(avg_err, iterations) per step."),
]

OBV = "GAP-SPH-OBV"  # observability/verification support claims, not capability gaps


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def extract(root: Path, spec: dict) -> dict:
    """Locate the claim's line by regex at the pinned commit.

    Returns the claim record; raises LookupError/FileNotFoundError with the spec id
    so a drifted anchor is a hard failure rather than a silently missing row.
    """
    repo = REPOS[spec["repo"]]
    fpath = root / spec["repo"] / spec["path"]
    if not fpath.is_file():
        raise FileNotFoundError(f"{spec['id']}: {spec['path']} missing under {root}")
    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
    pat = re.compile(spec["anchor"])
    for idx, line in enumerate(lines, start=1):
        if pat.search(line):
            return {
                "id": spec["id"],
                "gap": spec["gap"],
                "repo": repo["full_name"],
                "commit": repo["commit"],
                "path": spec["path"],
                "line": idx,
                "quote": line.strip(),
                "file_sha256": sha256_file(fpath),
                "statement": spec["statement"],
                "port_recipe": spec["port"],
            }
    raise LookupError(f"{spec['id']}: anchor {spec['anchor']!r} not found in {spec['path']}")


def build(root: Path, out_dir: Path) -> dict:
    claims = [extract(root, s) for s in SPECS]
    by_gap: dict[str, int] = {}
    for c in claims:
        by_gap[c["gap"]] = by_gap.get(c["gap"], 0) + 1
    doc = {
        "anchor": "FEAT-OSS-INVENTORY-SPH",
        "wave": 3,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "channel": "CODE-LEVEL",
        "evidence_grade_note": (
            "Every row is a verbatim source line at a frozen commit, located by "
            "regex and re-derivable with `gap_refs.py verify`. This is a code-read "
            "of the cited line, not a claim about numerical accuracy of the "
            "upstream code."
        ),
        "repos": {
            k: {kk: vv for kk, vv in v.items()} for k, v in REPOS.items()
        },
        "claims": claims,
        "claims_per_gap": dict(sorted(by_gap.items())),
        "gaps_not_covered": {
            "GAP-SPH-MULTIPHASE-SHARP": (
                "none of the three wave-3 repositories implements a sharpening "
                "multiphase scheme; the wave-2 DualSPHysics and SPlisHSPlasH deep "
                "dives are the relevant references, so no new code evidence is "
                "claimed here."
            )
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EVIDENCE_NAME).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def verify(root: Path, evidence_path: Path) -> tuple[int, list[str]]:
    doc = json.loads(evidence_path.read_text())
    problems: list[str] = []
    for rec in doc["claims"]:
        spec = next((s for s in SPECS if s["id"] == rec["id"]), None)
        if spec is None:
            problems.append(f"{rec['id']}: no spec (record not reproducible)")
            continue
        try:
            fresh = extract(root, spec)
        except (FileNotFoundError, LookupError) as exc:
            problems.append(str(exc))
            continue
        for key in ("repo", "commit", "path", "line", "quote", "file_sha256", "gap"):
            if fresh[key] != rec[key]:
                problems.append(
                    f"{rec['id']}: {key} drifted (recorded {rec[key]!r} vs fresh {fresh[key]!r})"
                )
    return len(doc["claims"]), problems


def report(evidence_path: Path, out_dir: Path) -> Path:
    doc = json.loads(evidence_path.read_text())
    claims = doc["claims"]
    gaps = json.loads(
        (Path("docs/oss_inventory_sph/gaps.json").read_text()
         if Path("docs/oss_inventory_sph/gaps.json").is_file() else "{}")
    )
    gap_meta = {p["slug"]: p for p in gaps.get("proposals", [])}

    rows = []
    for c in claims:
        rows.append(
            f"| {c['id']} | {c['gap']} | `{c['repo'].split('/')[1]}@{c['commit'][:8]}` "
            f"| `{c['path']}:{c['line']}` | {c['statement']} |"
        )
    quote_rows = []
    for c in claims:
        q = c["quote"].replace("|", "\\|")
        if len(q) > 118:
            q = q[:115] + "..."
        quote_rows.append(
            f"| {c['id']} | `{c['path']}:{c['line']}` | `{q}` | {c['file_sha256'][:12]} |"
        )
    per_gap = "\n".join(
        f"| {g} | {n} | {gap_meta.get(g, {}).get('goal', 'observability support')} |"
        for g, n in doc["claims_per_gap"].items()
    )
    repo_rows = "\n".join(
        f"| `{v['full_name']}` | {v['language']} | {v['licence']} | `{v['commit'][:12]}` "
        f"| {v['commit_date']} | {sum(1 for c in claims if c['repo'] == v['full_name'])} |"
        for v in doc["repos"].values()
    )

    # figure: gap x repo evidence matrix (injected numbers only)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repo_keys = [v["full_name"].split("/")[1] for v in doc["repos"].values()]
    repo_full = [v["full_name"] for v in doc["repos"].values()]
    gap_keys = sorted(doc["claims_per_gap"])
    mat = [[sum(1 for c in claims if c["repo"] == rf and c["gap"] == g) for rf in repo_full]
           for g in gap_keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(repo_keys)), repo_keys, fontsize=9)
    ax.set_yticks(range(len(gap_keys)), gap_keys, fontsize=8)
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > max(max(r) for r in mat) / 2 else "black",
                    fontsize=9)
    ax.set_title(f"Code-level evidence rows per gap and repository "
                 f"(n={len(claims)} claims, {len(doc['repos'])} repos)", fontsize=9)
    fig.colorbar(im, ax=ax, label="claims")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "gap_evidence_matrix.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    md = f"""# GAP-SPH-* code-level reference pack (OSS-SPH wave 3)

*Anchor `FEAT-OSS-INVENTORY-SPH`. Generated by `scripts/oss_sph_inventory/gap_refs.py report` from `{EVIDENCE_NAME}`; every file:line, quote and count below is injected, none typed by hand.*

## 1. What wave 1 and wave 2 left open

Wave 1 (`docs/oss_inventory_sph/registry.json`, mission m15815) graded 23 public SPH codes on two channels -- README prose (`CLAIMED-BY-DOCS`) and file-tree names (`CODE-TREE`) -- and stated the limitation itself: *neither channel reads third-party source code*. Five `GAP-SPH-*` anchors were proposed from that survey. Wave 2 wrote prose deep dives for a subset (SPlisHSPlasH, DualSPHysics, SPHinXsys, sphexa, Phantom, WeldForm, ...).

Wave 3 answers a narrower question: **for each open gap, which upstream file actually implements the missing capability, at which line, and what does that line say?** No new capability is claimed for TensorSPH; what is new is that a developer picking up a `GAP-SPH-*` anchor starts from a pinned, verified source pointer instead of prose.

## 2. Method

1. Three repositories were cloned and pinned (`git clone --depth 1`, commit recorded below). They were chosen because each holds a capability that *no previously deep-dived code* covers in the same file: MPI halo exchange + volume boundary model (`gpusph`), delta-SPH/Riemann evolution + ghost-wall BC in readable Python (`jax-sph`), the DFSPH and IISPH projection pair with explicit stopping criteria (`salva`).
2. `SPECS` in `scripts/oss_sph_inventory/gap_refs.py` holds one record per claim: a regular expression anchor, the gap anchor it serves, a statement and a port recipe for TensorSPH.
3. `build` resolves each anchor to its **first matching line**, stores `path`, `line`, the **verbatim** stripped line and the file's `sha256`. An anchor that does not match is a hard error -- a drifted upstream cannot silently produce an empty row.
4. `verify` re-derives every record from the clones and compares all seven fields; it is wired into `tests/test_oss_sph_gap_refs.py`.
5. Scope honesty: this is a *code-read of the cited lines*. It is not a numerical validation of the upstream codes and not a proof that porting the recipe reproduces the upstream result.

### 2.1 Pinned sources

| repository | language | licence | commit | date | claims |
|---|---|---|---|---|---|
{repo_rows}

## 3. Results

{len(claims)} claims, {len(doc['repos'])} repositories, {len(doc['claims_per_gap'])} gap anchors touched.

| gap anchor | claims | upstream capability it would give TensorSPH |
|---|---|---|
{per_gap}

### 3.1 Claim index

| id | gap | source | location | what it evidences |
|---|---|---|---|---|
{chr(10).join(rows)}

### 3.2 Verbatim quotes (the evidence itself)

| id | location | quote | file sha256 (12) |
|---|---|---|---|
{chr(10).join(quote_rows)}

### 3.3 Figure

`gap_evidence_matrix.png` -- rows = gap anchors, columns = repositories, cell = number of verified claims. The sparsity is the point: no single upstream code answers every gap, so porting must be assembled from several.

## 4. Notes, limits, and what would falsify this

- **Not covered:** {', '.join(doc['gaps_not_covered'])}. `GAP-SPH-MULTIPHASE-SHARP` is unrepresented because none of the three wave-3 codes implements a sharpening multiphase scheme; the wave-2 DualSPHysics / SPlisHSPlasH deep dives remain its references. Recording this instead of padding the table is deliberate.
- **Licence boundary:** `gpusph` is GPL-3.0 and `TensorSPH` is not; the recipes above are *algorithmic* (message layout, boundary bookkeeping, iteration criteria), not copies of GPL code. Anyone porting beyond the algorithm must re-check this row.
- **Commit freshness:** `gpusph` is pinned at its last push (2021); it is used as a *design reference*, not as an accuracy baseline.
- **Falsifier:** if `verify` fails on a pinned clone, or if a claim's line no longer carries the described construct, the claim is withdrawn rather than reworded.
- Claims tagged `{OBV}` describe the upstream *validation harness* rather than a missing TensorSPH capability; they are recorded because TensorSPH's benchmarks still lack a judge, but they must not be counted as gap coverage.

## 5. Error adjudication

There is no numerical benchmark in this wave, so no error percentage is claimed -- and none may be inferred. The measurable quality gates used instead are: (a) `build` resolves every anchor (no silent misses), (b) `verify` reproduces all seven fields per claim against the clones, (c) `tests/test_oss_sph_gap_refs.py` proves the verifier *rejects* a tampered quote. Gate (c) exists because a verifier that only ever passes proves nothing.
"""
    md_path = out_dir / "GAP_REFS_REPORT.md"
    md_path.write_text(md)
    return md_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "report"])
    ap.add_argument("--root", default="/tmp/sph", help="dir with one clone per repo")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    ev = out_dir / EVIDENCE_NAME
    if args.cmd == "build":
        doc = build(Path(args.root), out_dir)
        print(f"wrote {ev}: {len(doc['claims'])} claims, gaps={doc['claims_per_gap']}")
        return 0
    if args.cmd == "verify":
        n, problems = verify(Path(args.root), ev)
        for p in problems:
            print("FAIL", p)
        print(f"verified {n - len(problems)}/{n} claims")
        return 0 if not problems else 1
    md = report(ev, out_dir)
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())