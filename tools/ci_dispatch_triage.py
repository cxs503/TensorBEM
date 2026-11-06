#!/usr/bin/env python3
"""Collect the evidence behind the 2026-11-06 ``TensorBEM`` main-CI triage (m4803).

The triage question is narrow and must not be answered from memory: was
``main@fa868b64`` red because a *test* failed, or because *no runner ever picked
the job up*?  The two look identical in the checks UI (both are "red on main"),
but only the second one needs a workflow fix, so the distinction is collected
from the Actions API rather than asserted.

What is collected
-----------------
1. ``target`` -- the run that made main red (``TARGET_RUN``): its status,
   conclusion, timestamps, and, per job, the runner name plus how many steps
   were recorded.  A cancelled job with an empty runner name and zero steps
   proves the suite never ran.
2. ``control`` -- the last main run that *did* execute (``CONTROL_RUN``): its
   runner and every step conclusion.  If the control is green, the tests were
   green on that tree and the red is not a numerical regression.
3. ``fleet`` -- for every repository served by the self-hosted pool, the newest
   runs and the runner each job got.  A pool-wide outage shows up as many
   repositories stuck in ``queued`` with no runner, which a single-repository
   bug cannot explain.

Outputs
-------
``docs/m4803/ci_dispatch_triage.json``  raw evidence (every number in the
                                       companion ``README.md`` is rendered from
                                       it -- nothing is copied by hand)
``docs/m4803/README.md``               findings table, regenerated on each run

Usage
-----
    python tools/ci_dispatch_triage.py            # query and write
    python tools/ci_dispatch_triage.py --render   # re-render README from JSON

The public API is used unauthenticated (60 requests/hour is ample: one run per
repository); set ``GITHUB_TOKEN`` to raise the limit.  A network failure is
reported as such and never silently replaced by a guess.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OWNER = "cxs503"
#: The run that made ``main`` red (workflow ``tests``, event ``push``, sha fa868b64).
TARGET_RUN = 42152088640
#: The last ``main`` run that actually executed, on the parent commit daec9ee.
CONTROL_RUN = 42150435454
#: Repositories served by the self-hosted ``ci-loong64`` pool.
FLEET = (
    "TensorBEM",
    "TensorLBM",
    "TensorFEM",
    "TensorFVM",
    "TensorDEM",
    "TensorSPH",
    "TensorDG",
    "TensorCAD",
)
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "m4803"
JSON_OUT = OUT_DIR / "ci_dispatch_triage.json"
MD_OUT = OUT_DIR / "README.md"


#: How many times a rate-limited call waits for the window to reset before giving up.
RATE_LIMIT_RETRIES = 3


def api(path: str) -> dict:
    """GET a GitHub API path, returning decoded JSON.

    The unauthenticated quota (60 requests/hour) is shared with every other
    agent on this host, so a ``403`` here usually means "wait, do not give up":
    the window reset time is in the response and is honoured, up to
    :data:`RATE_LIMIT_RETRIES` times.
    """
    url = path if path.startswith("https://") else f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "tensorbem-ci-dispatch-triage",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            reset = error.headers.get("X-RateLimit-Reset") if error.headers else None
            if error.code != 403 or reset is None or attempt == RATE_LIMIT_RETRIES:
                raise
            wait = max(1.0, float(reset) - datetime.now(timezone.utc).timestamp() + 2.0)
            print(f"rate limited; sleeping {wait:.0f}s (attempt {attempt + 1})")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _lifetime_seconds(start: str | None, end: str | None) -> float | None:
    begin, finish = _timestamp(start), _timestamp(end)
    if begin is None or finish is None:
        return None
    return (finish - begin).total_seconds()


def describe_run(repo: str, run_id: int) -> dict:
    """Run-level facts plus a per-job summary (runner name, recorded steps)."""
    run = api(f"/repos/{OWNER}/{repo}/actions/runs/{run_id}")
    jobs = api(f"/repos/{OWNER}/{repo}/actions/runs/{run_id}/jobs?per_page=20")["jobs"]
    return {
        "repo": f"{OWNER}/{repo}",
        "run_id": run["id"],
        "head_sha": run["head_sha"],
        "event": run["event"],
        "status": run["status"],
        "conclusion": run["conclusion"],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "lifetime_seconds": _lifetime_seconds(run.get("created_at"), run.get("updated_at")),
        "jobs": [
            {
                "name": job["name"],
                "status": job["status"],
                "conclusion": job["conclusion"],
                "runner_name": job.get("runner_name") or "",
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "steps_recorded": len(job["steps"]),
                "steps": [
                    {"number": step["number"], "name": step["name"], "conclusion": step["conclusion"]}
                    for step in job["steps"]
                ],
            }
            for job in jobs
        ],
    }


def describe_fleet(repo: str, per_repo: int = 3) -> dict:
    """Newest runs of one repository (run level only: one API call per repo).

    A run that is still ``queued``/``pending`` has, by definition, no runner:
    GitHub creates the job record first and fills ``runner_name`` when a runner
    picks it up.  That is enough to separate a repository-local test failure
    from a pool-wide dispatch outage, so the per-job calls are spent on the
    target and control runs only (the shared hourly quota is small).
    """
    runs = api(f"/repos/{OWNER}/{repo}/actions/runs?per_page={per_repo}")["workflow_runs"]
    return {
        "repo": f"{OWNER}/{repo}",
        "runs": [
            {
                "run_id": run["id"],
                "head_sha": run["head_sha"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "created_at": run["created_at"],
            }
            for run in runs
        ],
    }


def summarise(payload: dict) -> dict:
    """Derive the verdict booleans from the collected evidence."""
    target_jobs = payload["target"]["jobs"]
    control_jobs = payload["control"]["jobs"]
    fleet_runs = [run for repo in payload["fleet"] for run in repo["runs"]]
    return {
        "target_runner_assigned": any(job["runner_name"] for job in target_jobs),
        "target_steps_recorded": sum(job["steps_recorded"] for job in target_jobs),
        "target_was_cancelled": payload["target"]["conclusion"] == "cancelled",
        "control_runner_assigned": all(job["runner_name"] for job in control_jobs),
        "control_all_steps_green": all(
            step["conclusion"] == "success" for job in control_jobs for step in job["steps"]
        ),
        "fleet_runs_seen": len(fleet_runs),
        "fleet_runs_queued": sum(
            1 for run in fleet_runs if run["status"] in {"queued", "pending"}
        ),
        "fleet_repositories_with_a_queued_run": len(
            {
                repo["repo"]
                for repo in payload["fleet"]
                if any(run["status"] in {"queued", "pending"} for run in repo["runs"])
            }
        ),
    }


def _format_lifetime(seconds: float | None) -> str:
    """Seconds as a one-decimal string, or ``n/a`` when the API omits a stamp."""
    return "n/a" if seconds is None else f"{seconds:.0f}"


def render_markdown(payload: dict) -> str:
    """Render the findings page; every number comes from ``payload``."""
    summary = payload["summary"]
    target = payload["target"]
    control = payload["control"]
    lines = [
        "# main CI triage, 2026-11-06 (m4803)",
        "",
        "Generated by `tools/ci_dispatch_triage.py` from the GitHub Actions API; do not",
        "edit by hand.  The question: was `main@fa868b64` red because of a failing test,",
        "or because no runner was ever assigned?",
        "",
        "## Verdict",
        "",
        "| check | value | reading |",
        "| --- | --- | --- |",
        f"| target run | `{target['run_id']}` on `{target['head_sha'][:8]}` | {target['event']}, {target['conclusion']} |",
        f"| runner assigned to the target job | {summary['target_runner_assigned']} | False means no runner took the job |",
        f"| steps recorded on the target job | {summary['target_steps_recorded']} | 0 means the suite never started |",
        f"| target lifetime (s) | {_format_lifetime(target['lifetime_seconds'])} | queued to cancellation |",
        f"| control run | `{control['run_id']}` on `{control['head_sha'][:8]}` | {control['conclusion']} |",
        f"| control runner | `{control['jobs'][0]['runner_name']}` | the pool that did dispatch |",
        f"| control steps all green | {summary['control_all_steps_green']} | the suite was green on the parent commit |",
        f"| fleet runs inspected | {summary['fleet_runs_seen']} | across {len(payload['fleet'])} repositories |",
        f"| fleet runs still queued (no runner by definition) | {summary['fleet_runs_queued']} of {summary['fleet_runs_seen']} | pool-wide, not repository-local |",
        f"| repositories with at least one queued run | {summary['fleet_repositories_with_a_queued_run']} of {len(payload['fleet'])} | a single-repository bug cannot do this |",
        "",
        f"Conclusion: **{'no runner dispatched the job; the tests never ran' if not summary['target_runner_assigned'] and summary['target_steps_recorded'] == 0 else 'a runner did run steps; inspect the step conclusions below'}**.",
        "",
        "## Target run steps",
        "",
        "| job | status | conclusion | runner | steps recorded |",
        "| --- | --- | --- | --- | --- |",
    ]
    for job in target["jobs"]:
        lines.append(
            f"| {job['name']} | {job['status']} | {job['conclusion']} | "
            f"`{job['runner_name'] or '<none>'}` | {job['steps_recorded']} |"
        )
    lines += [
        "",
        "## Control run steps",
        "",
        "| # | step | conclusion |",
        "| --- | --- | --- |",
    ]
    for job in control["jobs"]:
        for step in job["steps"]:
            lines.append(f"| {step['number']} | {step['name']} | {step['conclusion']} |")
    lines += [
        "",
        "## Fleet census",
        "",
        "| repository | run | status | conclusion | created (UTC) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for repo in payload["fleet"]:
        for run in repo["runs"]:
            lines.append(
                f"| {repo['repo'].split('/')[-1]} | `{run['run_id']}` | {run['status']} | "
                f"{run['conclusion']} | {run['created_at']} |"
            )
    lines += [
        "",
        "## What this means for the workflow",
        "",
        "`runs-on:` in `.github/workflows/tests.yml` resolves through the repository",
        "variable `TENSORBEM_CI_RUNNER`, defaulting to `ubuntu-latest` (the pool that",
        "dispatched in the control run).  Setting `TENSORBEM_CI_RUNNER=ci-loong64` moves",
        "the job back to the self-hosted pool without touching code, once that pool",
        "starts handing out jobs again.",
        "",
        f"_Collected {payload['generated_at']} (UTC)._",
        "",
    ]
    return "\n".join(lines)


def collect() -> dict:
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "owner": OWNER,
        "target": describe_run("TensorBEM", TARGET_RUN),
        "control": describe_run("TensorBEM", CONTROL_RUN),
        "fleet": [describe_fleet(repo) for repo in FLEET],
    }
    payload["summary"] = summarise(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--render",
        action="store_true",
        help="re-render docs/m4803/README.md from the stored JSON instead of querying",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.render:
        payload = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    else:
        try:
            payload = collect()
        except (urllib.error.URLError, TimeoutError) as error:  # pragma: no cover - network
            print(f"network failure, evidence not refreshed: {error}")
            return 2
        JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    MD_OUT.write_text(render_markdown(payload), encoding="utf-8")
    summary = payload["summary"]
    print(
        f"target runner assigned: {summary['target_runner_assigned']}, "
        f"steps recorded: {summary['target_steps_recorded']}, "
        f"fleet queued: {summary['fleet_runs_queued']}"
        f"/{summary['fleet_runs_seen']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())