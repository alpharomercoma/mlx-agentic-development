# Results

**250 runs. Codex CLI 0.146.0, `gpt-5.6-terra`, effort medium, web search on.
10 tasks × 5 arms × 5 repeats. 44,006,907 tokens. Apple M5, MLX 0.32.0, 2026-08-04.**

Analysed exactly as specified in [`benchmark/PREREGISTRATION.md`](benchmark/PREREGISTRATION.md)
and its two amendments, all committed before these runs executed.

---

## Headline: no pre-registered contrast reached significance

| Contrast | Effect | Clustered SE | Bootstrap 95% CI | Permutation p |
|---|---|---|---|---|
| C−A correctness | +0.10 | 0.049 | [0.00, +0.22] | 0.25 |
| C−B correctness | +0.06 | 0.034 | [−0.08, +0.22] | 0.75 |
| **C−E tokens** | **−3,582** | 8,187 | [−18,780, +13,760] | **0.69** |
| C−D tokens | −31,250 | 21,310 | [−74,990, +8,407] | 0.22 |
| D−E tokens | +27,670 | 17,740 | [−7,731, +68,980] | 0.24 |
| C−B tokens | −66,520 | 26,390 | [−136,200, −7,113] | 0.10 |
| C−A tokens | −44,970 | 23,080 | [−110,500, +12,250] | 0.25 |

α = 0.025 per family head (Holm across the two co-primary families). McNemar C vs A:
b=1, c=0, exact p = 1.00.

**Prediction 1 (correctness null) confirmed. Prediction 6 (C−E null) confirmed.**
Predictions 2 and 7 were directionally right and statistically null.

## The result that matters: structure bought nothing

**C−E is a 3,582-token difference on a ~150,000-token task, p = 0.69.**

Arm C is eight skills with YAML frontmatter, description-based routing, Complexity
Assessment tiering, progressive disclosure, routing tables, worked examples and
honesty rails — 33,625 characters. Arm E is the same facts as a flat 40-line cheat
sheet — 3,913 characters, one file, no structure of any kind.

They performed the same. On the primary efficiency metric the structured kit was
2.4% cheaper than the bare list, well inside noise.

The obvious escape — "the kit never loaded" — is closed. **Arms C, D and E each
loaded their content on 50 of 50 runs.** The kit fired every time and drew on all
eight skills. Cache share was 88.1–88.7% across all five arms, so cache warmth is not
driving anything either.

If this replicates, the apparatus around a skill is decoration and the text is the
product.

## Descriptive: cost per success

Not a hypothesis test. Total tokens across all attempts ÷ successes — what a user
actually pays, charging each arm for its failures.

| Arm | Content | Passed | Median tokens/run | Cost per success |
|---|---|---|---|---|
| **C** | full kit | **45/50** | 138,178 | **163,070** |
| E | bare facts | 43/50 | 123,584 | 174,820 |
| D | raw docs | 45/50 | 152,424 | 197,795 |
| A | bare | 40/50 | 152,144 | 239,667 |
| B | placebo | 42/50 | 160,392 | 253,906 |

The ordering is sensible and the spread is large — C is 32% cheaper per success than
bare — but **none of it survives the paired test at n = 10 tasks**, exactly as the
pre-registered 20–25 point minimum detectable effect warned. Reporting the ordering
without that caveat would be the single easiest way to oversell this.

## Exploratory per-task detail

Pre-registered as exploratory. Three cells are worth stating because they contradict
each other, and a single average would hide all three.

| Task | A bare | B placebo | C kit | D docs | E facts |
|---|---|---|---|---|---|
| `p01_metal_kernel_fused` | 0/5 | 2/5 | **0/5** | 2/5 | **3/5** |
| `p05_kernel_rowsum` | 2/5 | 2/5 | **5/5** | **5/5** | **5/5** |
| `p07_memory_api` | 4/5 | 3/5 | 5/5 | 5/5 | **1/5** |
| all seven others | 5/5 or 4/5 | — | 5/5 | — | — |

**`p01` — the kit fails the one task it was written for.** The scalar-binding rule
(0-d arrays bind by value, 1-d bind as pointers) is documented in
`mlx-metal-kernels` as trap #3, with both compiler errors quoted. Arm C scored
**0/5**. The flat cheat sheet, stating the same rule in three lines, scored **3/5**.
The prose version is longer, better organised, and worse.

**`p05` — content helps, and it does not matter how it is packaged.** Bare and
placebo scored 2/5; all three MLX-content arms scored 5/5. This is the clearest
evidence in the dataset that supplying the material works, and it is equally clear
that kit, docs and cheat sheet were interchangeable.

**`p07` — the cheat sheet actively hurt.** Arm E scored 1/5 where bare scored 4/5.
Compressing facts into a terse list appears to have produced confident wrong answers
where searching would have produced right ones. Published work finds ~19% of tasks get
*worse* with skills; this is one.

## Validity

| Check | Result |
|---|---|
| Contamination (workspace or fixtures) | 0 of 250 runs |
| Mechanism: content loaded | C 50/50, D 50/50, E 50/50, B 2/50, A 0/50 |
| Cache share of input tokens | 88.1–88.7%, uniform across arms |
| Process errors / timeouts / zero-token runs | 0 / 0 / 0 |
| Runs producing nothing gradable | 0 |
| Cells per arm | 50 / 50 / 50 / 50 / 50 |

**Three contamination incidents were found and fixed before this dataset existed**,
each caught by a check rather than by luck:

1. Workspaces nested inside the kit repo, so Codex's repo-scoped skill lookup handed
   the kit to every arm including bare. Fixed by moving workspaces out and adding
   `assert_outside_repo()`.
2. Nine dangling `references/` pointers, so arm C was sent to missing files on Complex
   tasks. Fixed, with a CI check that resolves every referenced path.
3. **Leftover test fixtures in `/private/tmp`** — including directories holding the
   gold solutions — found by 10 runs, 7 of which read them. Both arm-A passes of `p01`
   came from reading an oracle. Those 10 cells were quarantined and re-run, and
   `preflight_clean()` now refuses to start if fixture-shaped directories sit near the
   workspace root.

The third was self-inflicted debris from building the other two checks. Re-running the
affected cells moved arm A from 42/50 to 40/50 and changed no conclusion.

## Limitations

- **All 10 tasks are in-kit.** The corrected leakage audit finds symbol coverage
  0.62–1.00, three at 1.00. C−A therefore bounds how much cheaper it is to be *told*
  than to search; it is not evidence of assistance. Only C−E and D−E hold the facts
  constant, which is why they carry the report.
- **C−E is confounded with catalog size** — C ships 8 catalog entries against E's 1,
  worth ~3.4 KB of always-injected prompt. D−E is the clean comparison (15 characters
  apart) and is also null.
- **n = 10 tasks.** The design detects a 20–25 point swing, not a 5-point one. Every
  null here means "no large effect", never "no effect".
- **One model, one machine, one day, web search on throughout.** No claim generalises
  beyond `gpt-5.6-terra` on an M5. The search-off condition has not been run.
- **The kit was written by an LLM.** Published work puts self-authored skills at
  −1.3pp on average; this result is consistent with that and does not refute it.
- Token counts are within-harness only; the Claude Code replication has not run.

## What I take from this

The honest reading is that **the facts did the work and the format did not**, on this
task set, with this model, with documentation available.

`p05` shows supplying material can convert 2/5 into 5/5. `p01` shows a carefully
structured page losing to three lines of plain text. `p07` shows compression producing
confident errors. None of the aggregate contrasts clear significance.

The durable artifacts here are not the prose. They are the **probe scripts**, which
re-derive their numbers on the user's machine instead of decaying with each 2.5-week
MLX release, and the **harness** — arm isolation via `CODEX_HOME`, byte-identical
flags, the three contamination guards, and a pre-registered analysis — which would
measure any skill kit and will outlive MLX 0.32.0.
