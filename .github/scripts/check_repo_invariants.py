#!/usr/bin/env python3
"""CI checks for the invariants this repository depends on.

Each check exists because violating it fails silently rather than loudly:

  skill-names   Codex requires a skill's frontmatter `name` to equal its directory
                name. A mismatch means the skill is simply never found.
  symlinks      .agents/skills/* must resolve into skills/*. A broken symlink means
                Codex sees nothing, while Claude Code still works -- so the repo
                looks fine from one client and is empty from the other.
  hook-exec     A hook script that is not executable silently no-ops.
  complexity    House rule: every SKILL.md carries a Complexity Assessment, so the
                agent tiers its reference loading instead of reading everything.
  sources       House rule: any skill with references/ carries references/SOURCES.md,
                because upstream docs carry per-page attribution obligations.
  dangling-ref  Every references/... path named in a SKILL.md must resolve. This is
                the check that would have caught nine live defects: every skill's
                Complexity Assessment pointed at a reference page that was never
                written, so on exactly the Complex tasks where the kit should earn
                its keep, the agent was sent to a file that does not exist, burned
                tool calls discovering that, and fell back to web search. The kit
                was a token tax rather than a token saving, and CI passed green.
  sources-ghost References/SOURCES.md must not attribute a page that does not exist.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "skills"
AGENT_SKILLS = ROOT / ".agents" / "skills"
HOOK_SCRIPTS = ROOT / "hooks" / "scripts"

failures: list[str] = []


def fail(check: str, msg: str) -> None:
    failures.append(f"[{check}] {msg}")


def frontmatter_name(skill_md: Path) -> str | None:
    """Read `name:` from the YAML frontmatter without a YAML dependency."""
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    for line in text[3:end].splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            return stripped[len("name:") :].strip().strip("\"'")
    return None


def check_skills() -> None:
    if not SKILLS.is_dir():
        return
    for skill_dir in sorted(p for p in SKILLS.iterdir() if p.is_dir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            fail("skill-names", f"{skill_dir.name}/ has no SKILL.md")
            continue

        name = frontmatter_name(skill_md)
        if name is None:
            fail("skill-names", f"{skill_dir.name}/SKILL.md has no frontmatter `name`")
        elif name != skill_dir.name:
            fail(
                "skill-names",
                f"{skill_dir.name}/SKILL.md declares name '{name}'; Codex requires it "
                f"to equal the directory name '{skill_dir.name}'",
            )

        body = skill_md.read_text(encoding="utf-8")
        if "Complexity Assessment" not in body:
            fail("complexity", f"{skill_dir.name}/SKILL.md has no Complexity Assessment")

        refs = skill_dir / "references"
        if refs.is_dir() and not (refs / "SOURCES.md").is_file():
            fail("sources", f"{skill_dir.name}/references/ has no SOURCES.md")

        check_references_resolve(skill_dir, body)


REF_PATH = re.compile(r"references/[A-Za-z0-9_.-]+\.md")
# Same defect class: a SKILL.md that tells the agent to run a script which is not
# there wastes a tool call and sends it back to web search.
# The optional leading segment matters: `hooks/scripts/detect-apple-silicon.py` is a
# repo-root path, not a skill-local one, and resolving it against the skill directory
# would report a false failure.
SCRIPT_PATH = re.compile(r"((?:[A-Za-z0-9_.-]+/)*scripts/[A-Za-z0-9_.-]+\.py)")


def check_references_resolve(skill_dir: Path, body: str) -> None:
    """Every references/... and scripts/... path a SKILL.md names must exist."""
    for rel in sorted(set(REF_PATH.findall(body))):
        if not (skill_dir / rel).is_file():
            fail(
                "dangling-ref",
                f"{skill_dir.name}/SKILL.md points at {rel}, which does not exist; "
                "the agent will be sent to a missing file",
            )

    for rel in sorted(set(SCRIPT_PATH.findall(body))):
        # Skill-local paths resolve against the skill; anything else against the repo.
        target = skill_dir / rel if rel.startswith("scripts/") else ROOT / rel
        if not target.is_file():
            fail(
                "dangling-ref",
                f"{skill_dir.name}/SKILL.md tells the agent to run {rel}, "
                "which does not exist",
            )
        elif not os.access(target, os.X_OK):
            fail(
                "dangling-ref",
                f"{target.relative_to(ROOT)} is not executable",
            )

    # A SOURCES.md row citing provenance for a page that was never written claims
    # attribution for content that does not exist.
    sources = skill_dir / "references" / "SOURCES.md"
    if sources.is_file():
        text = sources.read_text(encoding="utf-8")
        for rel in sorted(set(REF_PATH.findall(text))):
            if not (skill_dir / rel).is_file():
                fail(
                    "sources-ghost",
                    f"{skill_dir.name}/references/SOURCES.md attributes {rel}, "
                    "which does not exist",
                )
        for name in sorted(set(re.findall(r"`([a-z0-9_-]+\.md)`", text))):
            if name == "SOURCES.md":
                continue
            if not (skill_dir / "references" / name).is_file():
                fail(
                    "sources-ghost",
                    f"{skill_dir.name}/references/SOURCES.md attributes `{name}`, "
                    "which does not exist",
                )


def check_symlinks() -> None:
    if not AGENT_SKILLS.is_dir():
        return
    linked = set()
    for entry in sorted(AGENT_SKILLS.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_symlink():
            fail("symlinks", f".agents/skills/{entry.name} is not a symlink")
            continue
        target = entry.resolve()
        if not target.is_dir():
            fail(
                "symlinks", f".agents/skills/{entry.name} does not resolve to a directory"
            )
            continue
        if target.parent != SKILLS:
            fail(
                "symlinks",
                f".agents/skills/{entry.name} resolves to {target}, outside skills/",
            )
            continue
        linked.add(target.name)

    if SKILLS.is_dir():
        for skill_dir in sorted(p for p in SKILLS.iterdir() if p.is_dir()):
            if skill_dir.name not in linked:
                fail(
                    "symlinks",
                    f"skills/{skill_dir.name} has no .agents/skills symlink, so Codex "
                    "will not discover it",
                )


def check_hook_exec() -> None:
    if not HOOK_SCRIPTS.is_dir():
        return
    for script in sorted(HOOK_SCRIPTS.iterdir()):
        if script.suffix in (".sh", ".py") and not os.access(script, os.X_OK):
            fail("hook-exec", f"{script.relative_to(ROOT)} is not executable")


def main() -> int:
    check_skills()
    check_symlinks()
    check_hook_exec()

    if failures:
        print(f"{len(failures)} invariant failure(s):\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("All repository invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
