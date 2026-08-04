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


# Flags shared by every pi arm. Identical across arms by construction.
#
#   --no-skills          suppresses skill *discovery* only. Verified in
#                        core/resource-loader.js: with noSkills set, skillPaths
#                        collapses to the explicitly-passed --skill paths, and
#                        loadSkills runs with includeDefaults:false. So arm A gets
#                        zero skills and arm C gets exactly the kit -- no other
#                        skill on this machine (superpowers et al.) can leak in.
#   --no-context-files   suppresses AGENTS.md / CLAUDE.md discovery
#   --no-extensions      an extension could register tools, changing the action space
#   --no-prompt-templates
#   --no-session         a session reused across arms is a contamination path, and
#                        ephemeral runs cannot resume into each other
#
# pi ships bash/read/write/edit/grep/ls/find and NO web-search tool, so every pi
# arm is inherently a search-off condition. That is not a defect here: the Codex
# search-off cell was quota-blocked, and this supplies the same factor on a second
# model. Curl-through-bash remains possible, so the harness measures network use
# rather than assuming its absence.
PI_COMMON_FLAGS: tuple[str, ...] = (
    "--print",
    "--mode",
    "json",
    "--no-session",
    "--no-skills",
    "--no-context-files",
    "--no-extensions",
    "--no-prompt-templates",
)


@dataclass(frozen=True)
class Arm:
    """One experimental condition."""

    id: str
    label: str
    kit_path: Path | None  # None for the bare arm


def build_arms(
    kit: Path,
    placebo: Path,
    docs: Path | None = None,
    facts: Path | None = None,
) -> dict[str, Arm]:
    """Five arms. Every one is delivered the same way, so only content varies.

    D and E exist because the corrected leakage audit found all 10 tasks in-kit.
    Once the kit states every answer, C-A and C-B measure how much cheaper it is to
    be told than to search -- real, but not "the kit makes the agent better".

    C-E holds the FACTS constant and removes the structure, so leakage cancels and
    what remains is whatever the skill format contributes. C-D holds the material
    constant and removes the curation. Those two are the contrasts that survive.
    """
    arms = {
        "A": Arm("A", "bare", None),
        "B": Arm("B", "placebo", placebo),
        "C": Arm("C", "kit", kit),
    }
    if docs is not None:
        arms["D"] = Arm("D", "raw-docs", docs)
    if facts is not None:
        arms["E"] = Arm("E", "bare-facts", facts)
    return arms


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


def prepare_pi_skills(arm: Arm, root: Path) -> Path | None:
    """Materialise a per-arm skills directory outside the repo, or None for bare.

    pi accepts `--skill <dir>` pointing anywhere, so the obvious move is to point it
    straight at the kit in the repo. That was rejected: the catalog hands the agent
    an absolute *filesystem path*, and the kit lives two directories above
    `benchmark/tasks/*/oracle/solution.py`. Arm C would be one `ls ..` from the gold
    solutions -- the same class of leak that invalidated the first Codex pilot.

    Copying also freezes the arm's content, so editing the kit mid-sweep cannot
    retroactively change a completed run.
    """
    if arm.kit_path is None:
        return None

    src_skills = arm.kit_path / "skills"
    if not src_skills.is_dir():
        raise FileNotFoundError(f"{src_skills} does not exist")

    dest = root / f"pi_skills_{arm.id}" / "skills"
    if dest.parent.exists():
        shutil.rmtree(dest.parent)
    dest.mkdir(parents=True)
    for skill_dir in sorted(p for p in src_skills.iterdir() if p.is_dir()):
        shutil.copytree(skill_dir, dest / skill_dir.name)
    return dest


def pi_sandbox_profile(repo: Path, venv: Path) -> str:
    """A macOS seatbelt profile that hides the benchmark repo from the agent.

    This is not optional hardening; without it the pi arms are uninterpretable.
    Codex runs under `--sandbox workspace-write`, so its agents physically cannot
    read outside the workspace. **pi ships no sandbox**, and its bash tool inherits
    the full permissions of the user. The first pi pilot demonstrated the
    consequence: the bare arm listed the repo, read `run.py`, read the docs-kit,
    read the hidden grader `test_solution.py`, and finally ran pytest against the
    real grader to check its own work. It "passed" 9/9.

    Worse, the contamination was *differential* -- arm C, which had the kit, never
    went looking, so the bare arm alone was inflated. A control that cheats and a
    treatment that does not is the one asymmetry that can move a result in either
    direction while looking entirely normal in the logs.

    `allow default` with a targeted deny is used rather than an allow-list because
    the agent legitimately needs the Metal stack, the GPU device, system frameworks
    and its own venv; enumerating that surface is fragile, whereas the thing that
    must never be readable is exactly one subtree. The venv is re-allowed because it
    sits under the same parent as the repo.
    """
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            f'(deny file-read* (subpath "{repo.resolve()}"))',
            f'(allow file-read* (subpath "{venv.resolve()}"))',
        ]
    )


def pi_command(
    arm: Arm,
    prompt: str,
    model: str,
    provider: str,
    thinking: str,
    skills_dir: Path | None,
    sandbox_profile: Path | None = None,
) -> list[str]:
    cmd: list[str] = []
    if sandbox_profile is not None:
        cmd += ["sandbox-exec", "-f", str(sandbox_profile)]
    cmd += [
        "pi",
        "--provider",
        provider,
        "--model",
        model,
        "--thinking",
        thinking,
        *PI_COMMON_FLAGS,
    ]
    if skills_dir is not None:
        cmd += ["--skill", str(skills_dir)]
    cmd.append(prompt)
    return cmd


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
