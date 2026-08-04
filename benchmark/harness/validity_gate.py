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

from arms import (
    CODEX_COMMON_FLAGS,
    KIT_TOKENS,
    build_arms,
    pi_command,
    prepare_codex_home,
    prepare_pi_skills,
)

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


def check_pi(
    kit: Path,
    placebo: Path,
    docs: Path | None,
    facts: Path | None,
    workdir: Path,
    provider: str,
    model: str,
) -> None:
    """Verify the pi arms behaviourally, and measure what each one injects.

    pi has no `codex debug prompt-input` equivalent, so the prompt is probed rather
    than rendered: each arm is asked to enumerate its own skills with tools denied,
    and the first model call's `input` token count is recorded as the prompt size.
    That count is the quantity that actually matters -- it is what the arm pays
    before doing any work -- and it is measured, not estimated from file sizes.

    Every arm is checked, not just A and C. The Codex sweep reported the mechanism
    for arm C alone at first, which is precisely the asymmetry that would let a
    control arm quietly load something and go unnoticed.
    """
    q = (
        "List the exact names of every skill available to you right now as a bare "
        "comma-separated list. Do not use any tools. If none, reply exactly: NONE"
    )
    task_dir = workdir / "taskdir"
    arms = build_arms(kit, placebo, docs, facts)

    def probe(arm, skills_dir) -> tuple[str, int]:
        """One probe: returns (assistant text, prompt tokens for the first call)."""
        cmd = pi_command(arm, q, model, provider, "off", skills_dir)
        proc = subprocess.run(
            cmd, cwd=task_dir, capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0:
            failures.append(f"pi arm {arm.id} failed: {proc.stderr.strip()[:200]}")
            return "", 0
        text, prompt_tokens = "", 0
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "message_end":
                continue
            msg = ev.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage") or {}
            if usage and not prompt_tokens:
                prompt_tokens = (usage.get("input") or 0) + (usage.get("cacheRead") or 0)
            for block in msg.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text") or ""
        return text, prompt_tokens

    sizes: dict[str, int] = {}
    for arm_id in sorted(arms):
        arm = arms[arm_id]
        skills_dir = prepare_pi_skills(arm, workdir / f"pi_{arm_id}")
        expected = (
            sorted(p.name for p in skills_dir.iterdir() if p.is_dir())
            if skills_dir
            else []
        )

        # The model is asked to introspect, and deepseek-v4-flash answers this
        # meta-question unreliably: an arm that demonstrably carries its catalog
        # still sometimes replies NONE. Injection is therefore proven structurally
        # by the token delta below; this probe corroborates that the catalog is
        # visible, and gets three attempts before it is called a failure. One
        # correct enumeration is sufficient evidence of visibility.
        text, prompt_tokens = "", 0
        for _ in range(3):
            text, prompt_tokens = probe(arm, skills_dir)
            if not expected or any(n.lower() in text.lower() for n in expected):
                break

        sizes[arm_id] = prompt_tokens
        low = text.lower()
        # Assert against the arm's OWN skill names, not KIT_TOKENS. Arms D and E
        # ship `mlx-docs` and `mlx-facts`, neither of which is a kit token, so a
        # KIT_TOKENS check marks a correctly-injected control as broken.
        own = expected
        named = [n for n in own if n.lower() in low]
        leaked = [t for t in KIT_TOKENS if t.lower() in low]

        if arm.kit_path is None:
            # The bare arm must name nothing from ANY arm, not merely nothing MLX.
            if leaked or "none" not in low:
                failures.append(
                    f"pi arm {arm_id} (bare) did not report NONE; leaked={leaked}"
                )
            else:
                notes.append(f"pi arm {arm_id} bare: reports NONE, 0 kit tokens")
        elif arm_id == "B":
            # The placebo must inject its own skills while naming nothing MLX-shaped.
            if leaked:
                failures.append(f"pi arm B (placebo) leaked kit tokens: {leaked}")
            elif not named:
                failures.append("pi arm B enumerates none of its own skills")
            else:
                notes.append(f"pi arm B placebo: {len(named)}/{len(own)} injected, no MLX")
        elif not named:
            failures.append(
                f"pi arm {arm_id} enumerates none of its own skills {own}"
            )
        else:
            notes.append(f"pi arm {arm_id}: {len(named)}/{len(own)} skills enumerated")

    # The deterministic check. Every content arm must pay strictly more prompt
    # tokens than bare, because its catalog is injected before any work happens.
    # This does not depend on the model choosing to answer a meta-question, so it
    # is the assertion the gate actually rests on.
    bare = sizes.get("A", 0)
    if bare:
        for arm_id in sorted(sizes):
            if arm_id == "A":
                continue
            delta = sizes[arm_id] - bare
            if delta < 50:
                failures.append(
                    f"pi arm {arm_id} prompt is only {delta} tokens over bare "
                    f"({sizes[arm_id]} vs {bare}); its content is not reaching the model"
                )
            else:
                notes.append(f"pi arm {arm_id} injects +{delta:,} prompt tokens over bare")
    notes.append(
        "pi prompt tokens by arm: "
        + ", ".join(f"{a}={sizes[a]:,}" for a in sorted(sizes))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--placebo", required=True, type=Path)
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--facts", type=Path)
    ap.add_argument(
        "--check-pi",
        action="store_true",
        help="also verify the pi arms (spends a few tokens)",
    )
    ap.add_argument("--pi-provider", default="opencode-go")
    ap.add_argument("--pi-model", default="deepseek-v4-flash")
    ap.add_argument(
        "--skip-codex",
        action="store_true",
        help="skip the Codex checks (for a pi-only run)",
    )
    ap.add_argument(
        "--check-claude",
        action="store_true",
        help="also verify the Claude arms (spends a few tokens)",
    )
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="validity-gate-"))
    try:
        if not args.skip_codex:
            check_codex(args.kit.resolve(), args.placebo.resolve(), workdir)
        else:
            (workdir / "taskdir").mkdir(parents=True, exist_ok=True)
        check_workspace_clean(workdir / "taskdir")
        if args.check_pi:
            check_pi(
                args.kit.resolve(),
                args.placebo.resolve(),
                args.docs.resolve() if args.docs else None,
                args.facts.resolve() if args.facts else None,
                workdir,
                args.pi_provider,
                args.pi_model,
            )
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
