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

from arms import (
    Arm,
    build_arms,
    claude_command,
    codex_command,
    pi_command,
    pi_sandbox_profile,
    prepare_codex_home,
    prepare_pi_skills,
)
from metrics import RunMetrics, parse_claude, parse_codex, parse_pi

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
# Nested one level deeper than /private/tmp on purpose. Agents explore their
# surroundings: in the first 250-run sweep, ten runs found leftover TEST FIXTURES
# sitting beside the workspaces in /private/tmp -- including directories holding the
# gold solutions -- and seven read them. Two arm-A runs of p01 passed a task the bare
# model otherwise never passes. Isolation is now structural, and preflight_clean()
# refuses to start if fixture-shaped directories are anywhere near.
WORKSPACE_ROOT = Path("/private/tmp/mlx-bench/workspaces")


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
    # Tool invocations that reached outside the workspace. Must be empty; a
    # non-empty list means the run is evidence about the harness, not the kit.
    contamination: list = None
    # Tool calls that reached the network. Reported, never grounds for exclusion.
    network: list = None

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def load_task(task_id: str) -> tuple[dict, str, Path, Path]:
    tdir = TASKS_DIR / task_id
    meta = json.loads((tdir / "meta.json").read_text())
    prompt = (tdir / "prompt.md").read_text()
    return meta, prompt, tdir / "workspace", tdir / "test_solution.py"


# Names that mean "someone left an oracle, a cheat, or a spare kit lying around".
FIXTURE_MARKERS = ("oracle", "cheat", "kit", "placebo", "solution", "gold")


def preflight_clean(root: Path = WORKSPACE_ROOT) -> list[Path]:
    """Refuse to run if fixture- or kit-shaped directories sit in any temp tree.

    Agents explore. This guard exists because they keep finding things:

      incident 3  ten runs of the Codex sweep found leftover fixtures beside the
                  workspaces in /private/tmp -- including gold solutions -- and
                  seven read them.
      incident 4  a pi pilot's BARE arm read a full copy of the kit out of
                  `$TMPDIR/dbg2-*/pi_C/pi_skills_C/skills/mlx-metal-kernels/SKILL.md`,
                  left behind by the harness author's own debugging.

    Both were self-inflicted debris, and in both cases the data looked normal. So
    the scan covers the macOS per-user temp directory as well as /private/tmp.

    The test is on **content, not names**. A name-based rule is both too loose and
    too tight here: it flags Apple's own `com.apple.pluginkit` and `StatusKitAgent`
    temp directories on the word "kit", and the empty `solution-pyc*` pycache
    prefixes agents leave behind, while still missing `dbg2-mvo44mk2`. So a
    directory is an offender only if it actually holds answer-bearing files.
    """
    import tempfile

    hazards = ("SKILL.md", "solution.py", "test_solution.py")
    roots = {root.parent, Path(tempfile.gettempdir()), Path("/private/tmp")}
    offenders: list[Path] = []
    for parent in roots:
        if not parent.is_dir():
            continue
        try:
            entries = list(parent.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            # Skip the workspace tree itself. Live runs legitimately contain the
            # agent's own solution.py, and /private/tmp/mlx-bench is an entry of
            # /private/tmp, so without this the guard reports itself every time.
            if entry == root or entry == root.parent or root == entry or root.is_relative_to(entry):
                continue
            try:
                # Bounded walk: a kit copy sits within a few levels of the top, and
                # an unbounded scan of $TMPDIR on every trial is not affordable.
                found = False
                for depth in ("*", "*/*", "*/*/*", "*/*/*/*", "*/*/*/*/*"):
                    for h in hazards:
                        if next(entry.glob(f"{depth}/{h}"), None) is not None:
                            found = True
                            break
                    if found:
                        break
                if found:
                    offenders.append(entry)
            except (PermissionError, OSError):
                continue
    return sorted(set(offenders))


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


def audit_contamination(
    out_dir: Path,
    workspace: Path | None = None,
    skills_dir: Path | None = None,
) -> list[str]:
    """Scan a finished run's tool calls for anything it should not have reached.

    The sandbox is the enforcement; this is the evidence. A guard that is never
    verified is indistinguishable from one that silently stopped working, and this
    experiment has already lost three datasets to leaks that looked normal in the
    logs. Every run therefore carries its own proof, and the sweep can report a
    contamination count instead of relying on a grep someone remembered to run.

    Matches are on the *invocation*, not on tool output: the repo path appearing in
    a file the agent legitimately read is not the agent reaching for the repo.

    Scoping matters in both directions, and the first version got both wrong. It
    flagged the agent writing its own `solution.py` and `_test_solution.py` -- which
    the task asks for -- while completely missing a bare arm reading the kit out of
    `$TMPDIR/dbg2-*/.../skills/mlx-metal-kernels/SKILL.md`, because that path
    contained none of the marker words. Name-shaped rules cannot catch a directory
    called `dbg2-mvo44mk2`.

    So the rule is positional: a path is an offence when it points at answer-bearing
    material *outside* the run's own workspace, plus any `skills/` path that is not
    the arm's own injected kit. `workspace` and `skills_dir` must be passed for that
    distinction to exist at all.
    """
    answers = ("test_solution", "/oracle/", "solution.py", "gold")
    ws = str(workspace.resolve()) if workspace else None
    own_skills = str(skills_dir.resolve()) if skills_dir else None
    offenders: list[str] = []
    for stream_name in ("stream.jsonl", "events.jsonl"):
        stream = out_dir / stream_name
        if not stream.is_file():
            continue
        for line in stream.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "tool_execution_start":
                continue
            tool = ev.get("toolName")
            for path in _paths_in(ev.get("args") or {}):
                if str(REPO.resolve()) in path:
                    offenders.append(f"{tool}: repo {path}")
                    continue
                if "/skills/" in path or path.endswith("SKILL.md"):
                    if not (own_skills and path.startswith(own_skills)):
                        offenders.append(f"{tool}: foreign kit {path}")
                    continue
                if any(a in path.lower() for a in answers):
                    if not (ws and path.startswith(ws)):
                        offenders.append(f"{tool}: answer material {path}")
    return offenders


def _paths_in(args: dict) -> list[str]:
    """Every filesystem-path-looking string in a tool invocation.

    Bash commands are the awkward case: the path is embedded in a command line, not
    in a `path` argument, so the whole command is scanned for absolute-path tokens.
    """
    import re as _re

    out: list[str] = []
    for value in args.values():
        if not isinstance(value, str):
            out.extend(
                v for v in (json.dumps(value),) if isinstance(v, str)
            )
            continue
        out.append(value)
    text = " ".join(out)
    return _re.findall(r"/[A-Za-z0-9_./\-]{4,}", text)


def audit_network(out_dir: Path) -> list[str]:
    """Tool calls that reached the network. Reported, never grounds for exclusion.

    pi ships no web-search tool, which is what makes it the search-off condition,
    but it does ship bash, so the agent can still fetch documentation by hand. An
    arm that curls the MLX docs is not running without documentation access and the
    search-off claim would be false for that run.

    This is kept separate from `contamination` on purpose: contaminated runs are
    dropped, and dropping every run that touched the network would delete exactly
    the runs where the model worked hardest, biasing the sample toward easy cells.
    It is a caveat to report, not invalid data.
    """
    net = ("curl ", "wget ", "urllib", "requests.get", "httpx", "pip install")
    hits: list[str] = []
    for stream_name in ("stream.jsonl", "events.jsonl"):
        stream = out_dir / stream_name
        if not stream.is_file():
            continue
        for line in stream.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "tool_execution_start":
                continue
            blob = json.dumps(ev.get("args") or {})
            for m in net:
                if m in blob:
                    hits.append(m.strip())
                    break
    return hits


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


PI_SKILLS_DIR: Path | None = None


def run_pi(
    arm: Arm,
    prompt: str,
    task_dir: Path,
    out_dir: Path,
    model: str,
    provider: str,
    thinking: str,
    timeout: int,
    scratch: Path,
    venv_py: Path,
) -> tuple[RunMetrics, bool, bool, str | None]:
    global PI_SKILLS_DIR
    skills_dir = prepare_pi_skills(arm, scratch)
    PI_SKILLS_DIR = skills_dir

    # pi has no sandbox of its own, so one is imposed. See pi_sandbox_profile: the
    # first pilot's bare arm read the hidden grader and ran it. Written per run so
    # the profile is archived with the evidence rather than assumed.
    profile = scratch / "sandbox.sb"
    profile.write_text(pi_sandbox_profile(REPO, venv_py.parent.parent))
    cmd = pi_command(arm, prompt, model, provider, thinking, skills_dir, profile)

    # The output files must live OUTSIDE the repo, then be copied in.
    #
    # `results/runs/` sits inside the repo, which the sandbox denies reads on, and
    # node fstats its own stdio during startup: handing it a stdout fd inside a
    # denied subpath aborts the process with SIGABRT before it runs a single line.
    # The failure gives no message, only a native stack trace, and looks exactly
    # like a crashed CLI. Writing to scratch and archiving afterwards keeps the full
    # evidence in results/ without granting the agent read access to prior runs --
    # which hold every earlier arm's produced solutions.
    stream = scratch / "stream.jsonl"
    stderr_path = scratch / "stderr.log"
    timed_out = False
    err: str | None = None
    with open(stream, "w") as fh, open(stderr_path, "w") as eh:
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

    # Archive a filtered stream. `--mode json` emits a message_update event per
    # token delta: one bare-arm run produced 166 MB, of which 99.8% was
    # message_update and 16,761 of 16,972 events. At 250 runs that is tens of
    # gigabytes to store and re-parse for no gain, since none of the retained
    # metrics read those events. The dropped types carry only partial, superseded
    # copies of content that arrives complete in message_end.
    archived = out_dir / "stream.jsonl"
    kept = 0
    with open(archived, "w") as out:
        for line in stream.read_text(errors="replace").splitlines():
            if '"message_update"' in line or '"tool_execution_update"' in line:
                continue
            out.write(line + "\n")
            kept += 1
    if stderr_path.is_file():
        shutil.copy2(stderr_path, out_dir / "stderr.log")
    shutil.copy2(profile, out_dir / "sandbox.sb")

    if rc not in (0, -1):
        err = stderr_path.read_text()[-500:]

    metrics, rate_limited = parse_pi(archived, f"{KIT_PREFIX}:")
    if metrics.provider_error and not err:
        err = (
            f"provider error: {metrics.model_calls} model call(s) returned no "
            "tokens (upstream stream failure); this run is not evidence about the arm"
        )
    return metrics, timed_out, rate_limited, err


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
    # `graded` means the grader RAN and produced a verdict -- not that everything
    # passed. Keying it on returncode conflated "the agent produced nothing and the
    # grader crashed" with "the code was wrong", and the pre-registration requires
    # counting execution errors separately from wrong answers. `passed` already
    # carries the outcome.
    return total > 0, passed, total


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
    docs: Path | None = None,
    facts: Path | None = None,
    model: str,
    effort: str = "medium",
    provider: str = "opencode-go",
    thinking: str = "medium",
    venv_py: Path = DEFAULT_VENV_PY,
    max_usd: float = 5.0,
    web_search: bool = True,
) -> TrialResult:
    meta, prompt, ws_src, test_file = load_task(task_id)
    timeout = int(meta.get("timeout_s", 900))
    arm = build_arms(kit, placebo, docs, facts)[arm_id]

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

    # Checked per trial, not once per sweep. Incident 3's fixtures and incident 4's
    # kit copy both appeared *during* a run of work, so a single check at startup
    # would have passed and the sweep would have carried on reading them. 0.3s.
    offenders = preflight_clean()
    if offenders:
        raise RuntimeError(
            "answer-bearing directories are readable near the workspace: "
            + ", ".join(str(o) for o in offenders)
        )

    t0 = time.perf_counter()
    if harness == "codex":
        metrics, timed_out, rate_limited, err = run_codex(
            arm, prompt, task_dir, out_dir, model, effort, timeout, scratch, web_search
        )
    elif harness == "pi":
        metrics, timed_out, rate_limited, err = run_pi(
            arm,
            prompt,
            task_dir,
            out_dir,
            model,
            provider,
            thinking,
            timeout,
            scratch,
            venv_py,
        )
    elif harness == "claude":
        metrics, timed_out, rate_limited, err = run_claude(
            arm, prompt, task_dir, out_dir, model, timeout, max_usd
        )
    else:
        raise ValueError(f"unknown harness {harness}")
    wall = time.perf_counter() - t0

    pi_skills_dir = PI_SKILLS_DIR if harness == "pi" else None
    graded, tp, tt = grade(test_file, task_dir, out_dir, venv_py)
    offenders = audit_contamination(out_dir, task_dir, pi_skills_dir)
    net_hits = audit_network(out_dir)

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
        contamination=offenders,
        network=net_hits,
    )
    (out_dir / "result.json").write_text(json.dumps(result.as_dict(), indent=2))

    # The per-arm CODEX_HOME holds a copy of auth.json; do not leave it lying around.
    # The whole external scratch tree goes with it, now that sources are archived.
    shutil.rmtree(scratch, ignore_errors=True)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--harness", default="codex", choices=("codex", "claude", "pi"))
    ap.add_argument("--arm", default="A", choices=("A", "B", "C", "D", "E"))
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--provider", default="opencode-go")
    ap.add_argument("--thinking", default="medium")
    ap.add_argument("--no-web-search", action="store_true")
    ap.add_argument("--kit", type=Path, default=REPO)
    ap.add_argument("--placebo", type=Path, default=REPO / "benchmark" / "placebo-kit")
    ap.add_argument("--docs", type=Path, default=REPO / "benchmark" / "docs-kit")
    ap.add_argument("--facts", type=Path, default=REPO / "benchmark" / "facts-kit")
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
        docs=args.docs,
        facts=args.facts,
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
