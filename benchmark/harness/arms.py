#!/usr/bin/env python3
"""Arm construction for the kit A/B experiment.

Three arms, differing in exactly one thing: what kit content is injected.

    A  bare      no kit
    B  placebo   token-matched kit, unrelated content
    C  kit       the real mlx-agentic-development kit

Arm B exists because without it a difference between A and C could be "a longer
prompt helped" rather than "this content helped". C-B isolates content from length.

Design note -- why the kit is NOT delivered repo-locally
--------------------------------------------------------
The obvious delivery route is to put the kit in the task repo as `.agents/skills`.
That was rejected after measurement: Codex's `<environment_context>` enumerates the
workspace's top-level directories, so a control-arm agent can see that `.agents/`
exists, `ls` it, and read the kit itself. The control would silently stop being a
control.

So the task workspace is always kit-free, and the kit arrives out-of-band:

    Codex        a per-arm $CODEX_HOME whose skills/ directory holds the kit
    Claude Code  --plugin-dir pointing at the kit

The consequence is that **command-line flags are byte-identical across arms**; only
the filesystem content of the injected directory differs. That is the stronger of
the two control designs, because no flag can have a second-order effect on the model.

Verified on 2026-08-04 (codex-cli 0.146.0, claude 2.1.220):

    Codex   arm A 9,091 chars / arm C 10,370 chars of prompt input.
            Arm A contains zero occurrences of any kit-identifying token.
    Claude  arm A sees 12 built-in skills and no kit skills; arm C sees the same
            12 built-ins plus the kit's skills, namespaced <plugin>:<skill>.

Built-in skills survive isolation in both clients. They are left enabled because
they are *identical across arms* and therefore a constant, not a confound. Codex's
bundled skills are disabled only because a flag exists to do so uniformly.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

# Kit-identifying substrings the control arm must never contain. Checked by
# validity_gate.py. Deliberately includes both naming and domain vocabulary: a
# leak could arrive as a skill name or as prose.
KIT_TOKENS = (
    "mlx-agentic-development",
    "mlx-metal-kernels",
    "mlx-core-semantics",
    "mlx-performance",
    "metal_kernel",
    "NAX",
    "Neural Accelerator",
)

# Flags shared by every Codex arm. Identical across arms by construction.
#
#   --disable memories     memories written in arm C and read in a later arm A is a
#                          live cross-arm contamination path
#   --disable plugins      also removes the <recommended_plugins> marketplace block,
#                          which is pure noise
#   --disable multi_agent  sub-agent spawning would confound token accounting
#   project_doc_max_bytes=0  suppresses AGENTS.md discovery
#   --strict-config        a typo'd -c override is an error, not a silent no-op,
#                          which would otherwise produce a null experiment
CODEX_COMMON_FLAGS: tuple[str, ...] = (
    "--disable",
    "plugins",
    "--disable",
    "memories",
    "--disable",
    "hooks",
    "--disable",
    "multi_agent",
    "--disable",
    "multi_agent_v2",
    "-c",
    "skills.bundled.enabled=false",
    "-c",
    "project_doc_max_bytes=0",
    "-c",
    "check_for_update_on_startup=false",
)


def codex_flags(web_search: bool) -> tuple[str, ...]:
    """Shared flags plus the web-search factor.

    Web search is a deliberate experimental factor, not a convenience. Mining showed
    the base model rarely fails when it can read documentation -- it pays instead --
    so `on` asks whether the kit beats a model that can already search, and `off`
    asks how much of the kit's value is substituting for documentation access.
    """
    return (*CODEX_COMMON_FLAGS, "-c", f"tools.web_search={str(web_search).lower()}")

# stream-json rather than json: the single-object `json` format reports no tool
# calls at all, so tool use is invisible. --verbose is required for stream-json
# to emit assistant messages.
CLAUDE_COMMON_FLAGS: tuple[str, ...] = (
    "--setting-sources",
    "",
    "--strict-mcp-config",
    "--output-format",
    "stream-json",
    "--verbose",
)


@dataclass(frozen=True)
class Arm:
    """One experimental condition."""

    id: str
    label: str
    kit_path: Path | None  # None for the bare arm


def build_arms(kit: Path, placebo: Path) -> dict[str, Arm]:
    return {
        "A": Arm("A", "bare", None),
        "B": Arm("B", "placebo", placebo),
        "C": Arm("C", "kit", kit),
    }


def prepare_codex_home(arm: Arm, root: Path, auth_source: Path) -> Path:
    """Materialise a per-arm CODEX_HOME and return it.

    Auth must be copied in: --ignore-user-config skips config.toml but auth still
    resolves through CODEX_HOME, so without auth.json the arm cannot run at all.
    """
    home = root / f"codex_home_{arm.id}"
    if home.exists():
        shutil.rmtree(home)
    (home / "skills").mkdir(parents=True)

    if not auth_source.is_file():
        raise FileNotFoundError(
            f"{auth_source} not found; Codex must be logged in before running arms"
        )
    shutil.copy2(auth_source, home / "auth.json")

    if arm.kit_path is not None:
        src_skills = arm.kit_path / "skills"
        if not src_skills.is_dir():
            raise FileNotFoundError(f"{src_skills} does not exist")
        for skill_dir in sorted(p for p in src_skills.iterdir() if p.is_dir()):
            # copy, not symlink: the arm's CODEX_HOME must be self-contained so a
            # later edit to the kit cannot retroactively change a completed run
            shutil.copytree(skill_dir, home / "skills" / skill_dir.name)

    return home


def codex_command(
    arm: Arm,
    prompt: str,
    task_dir: Path,
    model: str,
    effort: str,
    out_dir: Path,
    web_search: bool = True,
) -> list[str]:
    return [
        "codex",
        "exec",
        "--cd",
        str(task_dir),
        "--skip-git-repo-check",
        "--model",
        model,
        "-c",
        f"model_reasoning_effort={effort}",
        "-c",
        "approval_policy=never",
        "--sandbox",
        "workspace-write",
        "--json",
        "--strict-config",
        "--output-last-message",
        str(out_dir / "last_message.txt"),
        *codex_flags(web_search),
        prompt,
    ]


def claude_command(arm: Arm, prompt: str, model: str, max_usd: float) -> list[str]:
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--permission-mode",
        "bypassPermissions",
        "--max-budget-usd",
        str(max_usd),
        *CLAUDE_COMMON_FLAGS,
    ]
    if arm.kit_path is not None:
        cmd += ["--plugin-dir", str(arm.kit_path)]
    cmd.append(prompt)
    return cmd
