# Contributing

## Two rules that matter most

1. **Never copy from `neuron-agentic-development` or `xla-agentic-development.`**
   Not scripts, not reference pages, not prose. Write everything from scratch.
2. **Never write an API page from memory.** MLX ships roughly monthly and the API
   moves. Pin a version, verify against it, and record the version on the page.

## House rules, enforced by CI

- Every `SKILL.md` carries a **Complexity Assessment** that tiers reference loading.
- Every skill with `references/` carries `references/SOURCES.md` mapping each page to
  its upstream URL.
- A skill's frontmatter `name` must equal its directory name. Codex requires this;
  a mismatch means the skill is silently never found.
- Every skill in `skills/` has a matching `.agents/skills/<name>` symlink, or Codex
  sees nothing while Claude Code works fine.
- Hook scripts must be executable. A non-executable hook silently no-ops.
- All chip constants live in `hooks/scripts/detect-apple-silicon.py`. Never hardcode
  a chip figure anywhere else.

## Honesty rails

Agents must distinguish what they measured from what they inferred. "Passes against a
CPU-stream reference" is not "verified on the GPU". "Predicted speedup" is not
"measured speedup". `VERIFICATION.md` records, with dates, which claims were executed
on real hardware and which are doc-derived.

## Never import mlx in a hook

Importing `mlx` initializes Metal and allocates GPU resources. Hooks run on every
session and must not disturb a workload the user already has running. Read package
metadata with `importlib.metadata` instead.
