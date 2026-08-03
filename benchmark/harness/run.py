#!/usr/bin/env python3
"""Execute one trial: (harness, arm, task, repeat) -> a graded, archived result.

Layout of a task on disk:

    benchmark/tasks/<task_id>/
        meta.json          area, tier, timeout, provenance (mined | authored)
        prompt.md          the instruction handed to the agent, verbatim
        workspace/         files copied into the run directory as the starting state
        test_solution.py   the grader -- NEVER copied into the workspace

The grader lives outside the workspace and is only introduced after the agent has
finished, so the agent cannot read, edit, or satisfy the tests by inspection. It runs
against the MLX venv, not the harness interpreter.

Every run archives its full evidence: the event stream, the rollout, stderr, the
final message, the post-run diff of the workspace, and the grader's output. A result
that cannot be re-examined later is not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from arms import Arm, build_arms, claude_command, codex_command, prepare_codex_home
from metrics import RunMetrics, parse_claude, parse_codex

REPO = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO / "benchmark" / "tasks"
CODEX_AUTH = Path.home() / ".codex" / "auth.json"
DEFAULT_VENV_PY = Path("/Users/alpha/asic/.venv-mlx/bin/python")

KIT_PREFIX = "mlx-agentic-development"

# Agent workspaces MUST live outside the kit repository. Codex discovers repo-scoped
# skills by walking up from the working directory to the repo root, so a workspace
# nested inside the repo hands the kit to EVERY arm, including the bare control, and
# silently destroys the experiment. This was observed in the first pilot: all three
# arms read skills/mlx-*/SKILL.md by absolute path.
WORKSPACE_ROOT = Path("/private/tmp/mlx-bench-workspaces")


@dataclass
class TrialResult:
    task_id: str
    harness: str
    arm: str
    repeat: int
    model: str
    web_search: bool
    ok: bool  # agent process exited cleanly
    timed_out: bool
    graded: bool  # grader ran at all
    passed: bool  # grader reported success
    tests_passed: int
    tests_total: int
    wall_clock_s: float
    speedup: float | None
    metrics: dict
    rate_limited: bool
    error: str | None = None

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def load_task(task_id: str) -> tuple[dict, str, Path, Path]:
    tdir = TASKS_DIR / task_id
    meta = json.loads((tdir / "meta.json").read_text())
    prompt = (tdir / "prompt.md").read_text()
    return meta, prompt, tdir / "workspace", tdir / "test_solution.py"


def assert_outside_repo(task_dir: Path) -> None:
    """Refuse to run if the agent's workspace sits inside the kit repository.

    Codex walks up from the working directory to find repo-scoped `.agents/skills`.
    A workspace inside the repo therefore leaks the kit into the control arm. Failing
    loudly is the only safe behaviour: the resulting data looks completely normal.
    """
    resolved = task_dir.resolve()
    if REPO.resolve() in resolved.parents or resolved == REPO.resolve():
        raise RuntimeError(
            f"workspace {resolved} is inside the kit repo {REPO}; every arm would "
            "discover the kit through repo-scoped skill lookup"
        )


def prepare_workspace(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        dest.mkdir(parents=True)


def run_codex(
    arm: Arm,
    prompt: str,
    task_dir: Path,
    out_dir: Path,
    model: str,
    effort: str,
    timeout: int,
    scratch: Path,
    web_search: bool,
) -> tuple[RunMetrics, bool, bool, str | None]:
    home = prepare_codex_home(arm, scratch, CODEX_AUTH)
    cmd = codex_command(arm, prompt, task_dir, model, effort, out_dir, web_search)
    env = {**os.environ, "CODEX_HOME": str(home)}

    events = out_dir / "events.jsonl"
    timed_out = False
    err: str | None = None
    with open(events, "w") as fh, open(out_dir / "stderr.log", "w") as eh:
        try:
            # macOS has no `timeout`; the subprocess timeout is the enforcement point.
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=eh,
                timeout=timeout,
                env=env,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, -1
    if rc not in (0, -1):
        err = (out_dir / "stderr.log").read_text()[-500:]

    metrics = parse_codex(events, home, f"{KIT_PREFIX}:")

    # Preserve the rollout alongside the run; the per-arm CODEX_HOME is disposable.
    from metrics import find_rollout

    rollout = find_rollout(home, None)
    if rollout and rollout.is_file():
        shutil.copy2(rollout, out_dir / "rollout.jsonl")

    return metrics, timed_out, False, err


def run_claude(
    arm: Arm,
    prompt: str,
    task_dir: Path,
    out_dir: Path,
    model: str,
    timeout: int,
    max_usd: float,
) -> tuple[RunMetrics, bool, bool, str | None]:
    cmd = claude_command(arm, prompt, model, max_usd)
    stream = out_dir / "stream.jsonl"
    timed_out = False
    err: str | None = None
    with open(stream, "w") as fh, open(out_dir / "stderr.log", "w") as eh:
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=eh,
                timeout=timeout,
                cwd=task_dir,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, -1
    if rc not in (0, -1):
        err = (out_dir / "stderr.log").read_text()[-500:]

    metrics, rate_limited = parse_claude(stream, f"{KIT_PREFIX}:")
    return metrics, timed_out, rate_limited, err


def grade(
    test_file: Path, task_dir: Path, out_dir: Path, venv_py: Path
) -> tuple[bool, int, int]:
    """Run the hidden grader against the finished workspace.

    The test file is placed in a sibling directory, never inside the workspace, so
    it is introduced only after the agent has stopped.
    """
    if not test_file.is_file():
        return False, 0, 0

    grader_dir = out_dir / "grader"
    grader_dir.mkdir(exist_ok=True)
    shutil.copy2(test_file, grader_dir / "test_solution.py")

    proc = subprocess.run(
        [
            str(venv_py),
            "-m",
            "pytest",
            str(grader_dir / "test_solution.py"),
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            # -s so tasks can print machine-readable metrics (SPEEDUP_METRIC);
            # pytest captures stdout by default and the speedup would be lost.
            "-s",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=task_dir,
        env={**os.environ, "PYTHONPATH": str(task_dir), "SOLUTION_DIR": str(task_dir)},
    )
    (out_dir / "grader.log").write_text(proc.stdout + "\n" + proc.stderr)

    passed = failed = 0
    for line in proc.stdout.splitlines():
        if " passed" in line or " failed" in line:
            import re

            if m := re.search(r"(\d+) passed", line):
                passed = int(m.group(1))
            if m := re.search(r"(\d+) failed", line):
                failed = int(m.group(1))
    total = passed + failed
    return proc.returncode == 0 and total > 0, passed, total


def parse_speedup(out_dir: Path) -> float | None:
    """Read the SPEEDUP_METRIC line a performance task prints, if any."""
    log = out_dir / "grader.log"
    if not log.is_file():
        return None
    import re as _re

    m = _re.search(r"SPEEDUP_METRIC .*?speedup=([0-9.]+)", log.read_text())
    return float(m.group(1)) if m else None


def run_trial(
    task_id: str,
    harness: str,
    arm_id: str,
    repeat: int,
    *,
    kit: Path,
    placebo: Path,
    results_root: Path,
    model: str,
    effort: str = "medium",
    venv_py: Path = DEFAULT_VENV_PY,
    max_usd: float = 5.0,
    web_search: bool = True,
) -> TrialResult:
    meta, prompt, ws_src, test_file = load_task(task_id)
    timeout = int(meta.get("timeout_s", 900))
    arm = build_arms(kit, placebo)[arm_id]

    run_id = f"{harness}_{task_id}_{arm_id}_s{int(web_search)}_r{repeat}"
    out_dir = results_root / "runs" / run_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    scratch = WORKSPACE_ROOT / run_id
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    task_dir = scratch / "workspace"
    prepare_workspace(ws_src, task_dir)
    assert_outside_repo(task_dir)

    t0 = time.perf_counter()
    if harness == "codex":
        metrics, timed_out, rate_limited, err = run_codex(
            arm, prompt, task_dir, out_dir, model, effort, timeout, scratch, web_search
        )
    elif harness == "claude":
        metrics, timed_out, rate_limited, err = run_claude(
            arm, prompt, task_dir, out_dir, model, timeout, max_usd
        )
    else:
        raise ValueError(f"unknown harness {harness}")
    wall = time.perf_counter() - t0

    graded, tp, tt = grade(test_file, task_dir, out_dir, venv_py)

    # Archive what the agent actually produced.
    produced = out_dir / "produced"
    produced.mkdir(exist_ok=True)
    for p in task_dir.rglob("*.py"):
        rel = p.relative_to(task_dir)
        target = produced / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)

    result = TrialResult(
        task_id=task_id,
        harness=harness,
        arm=arm_id,
        repeat=repeat,
        model=model,
        web_search=web_search,
        ok=err is None and not timed_out,
        timed_out=timed_out,
        graded=graded,
        passed=graded and tp == tt and tt > 0,
        tests_passed=tp,
        tests_total=tt,
        wall_clock_s=round(wall, 2),
        speedup=parse_speedup(out_dir),
        metrics=metrics.as_dict(),
        rate_limited=rate_limited,
        error=err,
    )
    (out_dir / "result.json").write_text(json.dumps(result.as_dict(), indent=2))

    # The per-arm CODEX_HOME holds a copy of auth.json; do not leave it lying around.
    # The whole external scratch tree goes with it, now that sources are archived.
    shutil.rmtree(scratch, ignore_errors=True)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--harness", default="codex", choices=("codex", "claude"))
    ap.add_argument("--arm", default="A", choices=("A", "B", "C"))
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--no-web-search", action="store_true")
    ap.add_argument("--kit", type=Path, default=REPO)
    ap.add_argument("--placebo", type=Path, default=REPO)
    ap.add_argument("--results", type=Path, default=REPO / "benchmark" / "results")
    ap.add_argument("--venv-python", type=Path, default=DEFAULT_VENV_PY)
    args = ap.parse_args()

    r = run_trial(
        args.task,
        args.harness,
        args.arm,
        args.repeat,
        kit=args.kit,
        placebo=args.placebo,
        results_root=args.results,
        model=args.model,
        effort=args.effort,
        venv_py=args.venv_python,
        web_search=not args.no_web_search,
    )
    print(json.dumps(r.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
