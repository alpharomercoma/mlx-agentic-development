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
"""

from __future__ import annotations

import os
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
