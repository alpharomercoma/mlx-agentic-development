#!/usr/bin/env python3
"""Prove the control arm is clean before spending tokens on it.

This is the experiment's validity check. If it fails, every number downstream is
meaningless, so it runs before any scored trial and is re-run and archived with the
results.

What it asserts
---------------
1. Codex: `codex debug prompt-input` renders the exact model-visible prompt without
   invoking the agent (free). Arm A must contain zero kit-identifying tokens; arm C
   must contain the kit. The A->C delta must be non-trivial, or the kit is not
   actually reaching the model and any measured difference is noise.
2. The task workspace itself contains no kit files, so a control-arm agent cannot
   discover the kit by listing directories. This is the failure mode that killed the
   repo-local delivery design.

The Claude arm has no prompt-input equivalent, so it is verified behaviourally by
`--check-claude`, which costs a small number of tokens and asks the model to
enumerate its own skills.

Usage:
    python3 validity_gate.py --kit ../.. --placebo /path/to/placebo
    python3 validity_gate.py --kit ../.. --placebo /path/to/placebo --check-claude
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from arms import CODEX_COMMON_FLAGS, KIT_TOKENS, build_arms, prepare_codex_home

CODEX_AUTH = Path.home() / ".codex" / "auth.json"

failures: list[str] = []
notes: list[str] = []


def run_prompt_input(home: Path, task_dir: Path) -> str:
    """Render Codex's model-visible prompt without invoking the agent.

    CODEX_COMMON_FLAGS is entirely -c and --disable overrides, all of which
    `codex debug prompt-input` accepts. The exec-only flags (--sandbox, --json,
    --output-last-message, ...) are added by codex_command and are deliberately not
    used here: they affect execution, not what the model is shown.
    """
    proc = subprocess.run(
        ["codex", "debug", "prompt-input", *CODEX_COMMON_FLAGS, "hi"],
        cwd=task_dir,
        capture_output=True,
        text=True,
        env=_env(home),
        timeout=120,
    )
    if proc.returncode != 0:
        failures.append(f"codex debug prompt-input failed: {proc.stderr.strip()[:300]}")
        return ""
    return proc.stdout


def _env(home: Path) -> dict:
    import os

    env = dict(os.environ)
    env["CODEX_HOME"] = str(home)
    return env


def check_codex(kit: Path, placebo: Path, workdir: Path) -> None:
    arms = build_arms(kit, placebo)
    task_dir = workdir / "taskdir"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "main.py").write_text("def solve():\n    pass\n")

    rendered: dict[str, str] = {}
    for arm_id in ("A", "C"):
        home = prepare_codex_home(arms[arm_id], workdir, CODEX_AUTH)
        rendered[arm_id] = run_prompt_input(home, task_dir)

    if not rendered.get("A") or not rendered.get("C"):
        return

    # 1. control must be clean
    leaked = [t for t in KIT_TOKENS if t.lower() in rendered["A"].lower()]
    if leaked:
        failures.append(f"codex arm A leaked kit tokens: {leaked}")
    else:
        notes.append(f"codex arm A clean ({len(rendered['A'])} chars, 0 kit tokens)")

    # 2. treatment must actually carry the kit
    present = [t for t in KIT_TOKENS if t.lower() in rendered["C"].lower()]
    if not present:
        failures.append(
            "codex arm C contains no kit tokens -- the kit is not reaching the model, "
            "so any measured difference would be noise"
        )
    else:
        notes.append(
            f"codex arm C carries the kit ({len(rendered['C'])} chars, "
            f"{len(present)} kit tokens)"
        )

    delta = len(rendered["C"]) - len(rendered["A"])
    notes.append(f"codex arm C - arm A prompt delta: {delta} chars")
    if delta <= 0:
        failures.append("codex arm C prompt is not larger than arm A; kit not injected")


def check_workspace_clean(task_dir: Path) -> None:
    """The control must not be able to find the kit by listing the workspace."""
    forbidden = (
        ".agents",
        ".codex",
        ".claude-plugin",
        ".codex-plugin",
        "AGENTS.md",
        "CLAUDE.md",
        "skills",
    )
    found = [f for f in forbidden if (task_dir / f).exists()]
    if found:
        failures.append(
            f"task workspace contains {found}; a control-arm agent could read the kit "
            "directly from disk"
        )
    else:
        notes.append("task workspace is kit-free")


def check_claude(kit: Path, workdir: Path) -> None:
    q = (
        "List the exact names of every skill available to you right now as a bare "
        "comma-separated list. Do not use any tools. If none, reply exactly: NONE"
    )
    task_dir = workdir / "taskdir"
    base = [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--output-format",
        "json",
    ]

    results: dict[str, str] = {}
    for arm_id, extra in (("A", []), ("C", ["--plugin-dir", str(kit)])):
        proc = subprocess.run(
            [*base, *extra, q], cwd=task_dir, capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0:
            failures.append(f"claude arm {arm_id} failed: {proc.stderr.strip()[:200]}")
            return
        try:
            results[arm_id] = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            failures.append(f"claude arm {arm_id} produced non-JSON output")
            return

    leaked = [t for t in KIT_TOKENS if t.lower() in results["A"].lower()]
    if leaked:
        failures.append(f"claude arm A leaked kit tokens: {leaked}")
    else:
        notes.append("claude arm A clean (no kit skills enumerated)")

    if not any(t.lower() in results["C"].lower() for t in KIT_TOKENS):
        failures.append("claude arm C does not enumerate any kit skill")
    else:
        notes.append("claude arm C enumerates kit skills")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--placebo", required=True, type=Path)
    ap.add_argument(
        "--check-claude",
        action="store_true",
        help="also verify the Claude arms (spends a few tokens)",
    )
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="validity-gate-"))
    try:
        check_codex(args.kit.resolve(), args.placebo.resolve(), workdir)
        check_workspace_clean(workdir / "taskdir")
        if args.check_claude:
            check_claude(args.kit.resolve(), workdir)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"scratch retained at {workdir}")

    for n in notes:
        print(f"  ok   {n}")
    if failures:
        print(f"\nVALIDITY GATE FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print("\nValidity gate passed. The control arm is clean and the kit reaches arm C.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
