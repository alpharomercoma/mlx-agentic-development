# Placebo kit

Arm B. Structurally identical to the real kit, topically unrelated.

Its only purpose is to separate **"a longer prompt helped"** from **"this content
helped"**. Without it, a C−A difference is uninterpretable: the treatment arm's prompt
is larger, and larger prompts change behaviour on their own. **C−B is the claim that
matters.**

## Matching

Same shape as the real kit: 8 skills, YAML frontmatter with a trigger-phrase
description, a Complexity Assessment that tiers reference loading, a routing table,
a traps section, honesty rails, and `references/SOURCES.md`.

| | Real kit | Placebo | Match |
|---|---|---|---|
| Skills | 8 | 8 | — |
| Description characters (always injected as the catalog) | 2,584 | 2,413 | **93%** |
| Body characters (injected only when a skill is loaded) | 27,978 | ~19,300 | 69% |

The description total is the number that matters most, because the catalog is
injected into every prompt whether or not a skill is used. Bodies enter the context
only when the model chooses to load one.

**The body shortfall is a real limitation and is reported, not hidden.** If arm C
loads skill bodies more often than arm B, part of any C−B token difference reflects
that asymmetry rather than content quality. The mechanism check — counting skill
invocations per arm — is what makes this visible, and it is reported alongside.

## Why PostgreSQL

It has to be plausible, well-written, and genuinely useful in its own domain, or the
control is degenerate: an obviously worthless kit would be ignored rather than read,
and would not control for anything. It also has to be far enough from MLX that it
cannot accidentally help. Server-side database work satisfies both — no GPU, no
kernels, no array semantics, no Apple silicon.

The content is deliberately written to the same standard as the real kit.

## Measured match in the actual injected prompt

The figures above are file sizes. What matters is what reaches the model, measured
with `codex debug prompt-input` on 2026-08-04:

| Arm | Prompt input | Delta vs bare |
|---|---|---|
| A bare | 9,197 chars | — |
| B placebo | 13,478 chars | +4,281 |
| C kit | 13,651 chars | +4,454 |

**Arm B and arm C differ by 173 characters, or 1.3%.** The catalog carries fixed
per-skill formatting overhead, so the injected sizes converge more tightly than the
raw description totals suggested. As a control for prompt length this is about as
close as it is practical to get, and it means a C−B difference cannot plausibly be
attributed to prompt size.
