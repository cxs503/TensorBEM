# OSS-LBM claim/evidence landing -- independent audit (FEAT-TK-OSS-LANDING)

* knowledge root: `/srv/tensor/workspace/TensorKnowledge`
* oss-cache: `/root/oss-cache` -> 4 repo(s): hschwane, hschwane-grasph, maarten-vd-sande-lbm, pylbm
* claims discovered: **162**, evidence records: **197**

## Verdict

* acceptance_met = **False** (green=False)
* C4 verbatim: verified 18 / unverifiable 135

### Blocking findings

* C1 schema: 113 record(s) fail schema
* C2 closure: 0 dangling, 12 claim(s) with no evidence
* C5 gap anchor: 38 claim(s) unmapped
* C7 A2 is WITHHELD, not met: differentiable-only claims = 9 (< gate 15) while framework-inclusive claims = 47 (>= gate); the ledger wording '可微/框架' is ambiguous, so the gate cannot be declared met

## Check matrix

| check | result |
|---|---|
| C1 schema | 347 validated, 113 failed |
| C2 closure | 0 dangling ref(s), 12 claim(s) w/o evidence, 30 orphan evidence |
| C3 locator | pinned 147, unpinned 9, missing/unparsable 35 |
| C4 verbatim | ok 16, ok-with-drift 2, failed 0, unverifiable 135 |
| C5 gap anchor | 124/162 mapped |
| C6 self-approval | 0 self-approved, 0/197 evidence assessed |
| C7 tally | A1=85 (gate 20); A2 broad-keyword=63, framework-system=47, differentiable-only=9 (gate 15) |

## Per-repository verbatim re-read

| repo key | records | repo absent | quote ok | drift | re-rendered (numbers ok) | falsified | unparsable |
|---|---|---|---|---|---|---|---|
| `?` | 32 | 0 | 0 | 0 | 0 | 0 | 30 |
| `MF-LBM` | 10 | 5 | 0 | 0 | 0 | 0 | 5 |
| `Maarten-vd-Sande/lbm` | 16 | 0 | 16 | 0 | 0 | 0 | 0 |
| `ThermalMultiphaseOpenLB` | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| `fingering_dynamics` | 23 | 23 | 0 | 0 | 0 | 0 | 0 |
| `lbm-principles-practice` | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| `lbm-principles-practice/code` | 36 | 36 | 0 | 0 | 0 | 0 | 0 |
| `lbmpy` | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `lettuce` | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| `ludwig-cf/ludwig` | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| `openlb` | 13 | 13 | 0 | 0 | 0 | 0 | 0 |
| `pylbm` | 5 | 0 | 0 | 2 | 2 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#ib_lbm_poiseuille` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#port_chapter5` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#taylor_green` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#young_laplace` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `xlb` | 4 | 4 | 0 | 0 | 0 | 0 | 0 |

## Findings by severity

* medium: 44
* high: 35
* low: 1

* **medium** `C3-locator` tke:oss-lbm:pylbm-equiv-eq-src: cited path 'analysis/equivalent_equation.py' is not repository-root-relative; uniquely resolved to 'pylbm/analysis/equivalent_equation.py' -- locator should be corrected
* **medium** `C4-verbatim` tke:oss-lbm:pylbm-equiv-eq-src: quote is a re-rendering, not verbatim text of pylbm@f9f5d16:pylbm/analysis/equivalent_equation.py; every asserted number is nevertheless present (numeric cross-check passed)
* **medium** `C3-locator` tke:oss-lbm:pylbm-stability-src: cited path 'analysis/stability.py' is not repository-root-relative; uniquely resolved to 'pylbm/analysis/stability.py' -- locator should be corrected
* **low** `C4-verbatim` tke:oss-lbm:pylbm-stability-src: quote differs from pylbm@f9f5d16:pylbm/analysis/stability.py only in quote-mark style (semantically identical, not literal -- acceptance asks for 逐字)
* **medium** `C3-locator` tke:oss-lbm:pylbm-bfl-src: cited path 'boundary.py' is not repository-root-relative; uniquely resolved to 'pylbm/boundary.py' -- locator should be corrected
* **medium** `C4-verbatim` tke:oss-lbm:pylbm-bfl-src: quote present in pylbm@f9f5d16:pylbm/boundary.py but outside cited lines 500-560 (locator drift)
* **medium** `C3-locator` tke:oss-lbm:pylbm-linear-bb-src: cited path 'boundary.py' is not repository-root-relative; uniquely resolved to 'pylbm/boundary.py' -- locator should be corrected
* **medium** `C4-verbatim` tke:oss-lbm:pylbm-linear-bb-src: quote is a re-rendering, not verbatim text of pylbm@f9f5d16:pylbm/boundary.py; every asserted number is nevertheless present (numeric cross-check passed)
* **medium** `C3-locator` tke:oss-lbm:pylbm-sympy-src: cited path 'analysis/equivalent_equation.py' is not repository-root-relative; uniquely resolved to 'pylbm/analysis/equivalent_equation.py' -- locator should be corrected
* **medium** `C4-verbatim` tke:oss-lbm:pylbm-sympy-src: quote present in pylbm@f9f5d16:pylbm/analysis/equivalent_equation.py but outside cited lines 10-10 (locator drift)
* **high** `C3-locator` tke:oss-lbm:mflbm-csf-src: locator_type='code-location' value='MF-LBM@11be324 README.md:About The Code + multiphase_3D/0.src/Kernel_multiphase.F90' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mflbm-mpi-openacc-src: locator_type='code-location' value='MF-LBM@11be324 README.md:Technical details + multiphase_3D/makefile' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mflbm-soa-src: locator_type='code-location' value='MF-LBM@11be324 README.md:Technical details' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mflbm-decomp-src: locator_type='code-location' value='MF-LBM@11be324 README.md:Important Notes > Domain decomposition' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mflbm-contact-angle-src: locator_type='code-location' value='MF-LBM@11be324 README.md:Important Notes > Contact angle' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-set-equi-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-collide-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-get-epsilon-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-get-normal-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-interface-mass-exchange-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-update-rho-u-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-set-inlet-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-set-outlet-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **medium** `C3-locator` tke:oss-lbm:lbm-principles-practice-evolve-src: ref 'master' is not a commit sha (branch/tag => provenance mobile, verbatim re-read not reproducible later)
* **high** `C3-locator` tke:oss-lbm:kruger-force-matrix: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:kruger-guo-force: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:kruger-poiseuille-l2: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:kruger-sc-potential: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:kruger-tg-decay: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:kruger-young-laplace: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:lbmpy-creation-functions: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:lbmpy-force-models-list: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:lbmpy-mcmp-tutorial: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:lbmpy-scmp-test: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:ludwig-free-energy: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:luma-kernels: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mcmp-lbm-eos: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mcmp-lbm-mrt: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mcmp-lbm-taichi: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* **high** `C3-locator` tke:oss-lbm:mplbm-ut-omega-order: locator_type=None value='' is not a machine-resolvable '<repo>@<commit> <path>:<lines|symbol>' locator
* ... 40 more in audit.json
