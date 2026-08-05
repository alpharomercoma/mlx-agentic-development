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

---

# Replication: pi 0.83.0 + `gpt-5.6-luna`, web search OFF

**250 runs. 10 tasks × 5 arms × 5 repeats. 16,761,473 tokens, $0.76. 2026-08-05.**
Analysed exactly as pre-registered, per Amendment 3, committed before these runs.

This is a *conceptual* replication: harness, model and search availability all change
at once, so **no pi-vs-Codex difference is attributable to any single one of them.**
Token counts are never compared across harnesses.

## The C−E null replicates

| Contrast | Effect | Clustered SE | Bootstrap 95% CI | Permutation p |
|---|---|---|---|---|
| C−A correctness | +0.06 | 0.052 | [0.00, +0.18] | 1.00 |
| C−B correctness | +0.06 | 0.052 | [0.00, +0.18] | 1.00 |
| **C−E tokens** | **+13,490** | 5,344 | [−4,881, +31,010] | **0.19** |
| C−D tokens | −33,250 | 12,430 | [−61,780, −4,338] | 0.062 |
| **D−E tokens** | **+46,740** | 12,630 | [+23,230, +71,670] | **0.0058** ✅ |
| C−B tokens | −21 | 7,865 | [−16,890, +17,360] | 1.00 |
| C−A tokens | +11,190 | 9,610 | [−5,356, +27,630] | 0.24 |

**Prediction 6 confirmed.** The structured 8-skill kit again failed to beat the same
facts as a flat cheat sheet — and this time it was 13,490 tokens *more* expensive,
still inside noise. Two harnesses, two models, with and without documentation
access: **the apparatus around a skill does not pay for itself.**

## The one thing that is significant: curation, not structure

**D−E = +46,740 tokens, p = 0.0058**, clearing the α = 0.025 head. Arm D is upstream
MLX documentation concatenated; arm E is the same facts hand-reduced to 40 lines.
Same information, ~9× the prose, and it costs nearly 50k more tokens per run. Arm D's
median passing run is 93,963 tokens against arm E's 45,288, and its cost per success
is the worst of all five arms at 104,870.

So the two content contrasts point in opposite directions, and together they say
something sharper than either alone:

- **Cutting text down to the facts pays.** (D−E, significant)
- **Wrapping those facts in skill machinery does not.** (C−E, null in both conditions)

Also significant: **C−A model calls −1.8, p = 0.0080.** The kit reaches an answer in
fewer round-trips than the bare model, even where it does not reach it in fewer tokens.

## Correctness: null, but arm C is the only clean sweep

| Arm | Passed | Median tokens (passing) | Cost per success |
|---|---|---|---|
| A bare | 47/50 | 44,061 | 57,595 |
| B placebo | 47/50 | 49,454 | 69,520 |
| **C kit** | **50/50** | 50,739 | 65,327 |
| D raw docs | 47/50 | 93,963 | 104,870 |
| **E bare facts** | 47/50 | 45,288 | **55,145** |

Arm C is the only arm that passed every run, and `pass@3 = pass^3 = 1.000` on all ten
tasks. It is **not significant** — McNemar b=1, c=0, p=1.00 — and it rests on a single
task. Reporting it as "the kit makes the agent perfect" would be exactly the overclaim
this experiment exists to avoid.

Everything hinges on `p07_memory_api`, the only task anyone failed:

| Task | A bare | B placebo | C kit | D docs | E facts |
|---|---|---|---|---|---|
| `p07_memory_api` | 2/5 | 2/5 | **5/5** | 2/5 | 3/5 |
| `p10_bool_mask_assign` | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| all eight others | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

`p07` is a naming task — which memory API actually exists in 0.32.0 — where guessing
plausibly is worse than knowing. Arm C got it right 5/5; the flat facts sheet managed
3/5; everything else 2/5. One task cannot carry a claim, but it is the same shape as
the Codex condition's `p05`: **content decides the cases where the answer cannot be
derived, only recalled.**

## Two pre-registered predictions were wrong

**Prediction 7 — that removing search would favour the content arms — is contradicted.**
Arm C spent *more* tokens than bare (+11,190), and bare beat the kit on cost per
success (57,595 vs 65,327). Turning search off did not make the kit pay.

**Prediction 8 — that pass rates would fall on a smaller model — is contradicted.**
They rose: bare went 40/50 → 47/50 and the kit 45/50 → 50/50 versus `gpt-5.6-terra`.
`gpt-5.6-luna` without documentation outperformed `gpt-5.6-terra` with it on this task
set, which mostly says these ten tasks are easier than the design assumed.

## Validity

| Check | Result |
|---|---|
| Contamination | **0 of 250** |
| Mechanism: content loaded | C 50/50, D 50/50, E 50/50, **A 0/50, B 0/50** |
| Cells per arm | 50 / 50 / 50 / 50 / 50, every cell exactly 5 repeats |
| Provider errors / timeouts | 0 / 0 |
| Cache share of prompt | 99.9–100.0%, uniform across arms |
| Network use | **2 of 50 arm-A runs** used `curl` |

Two caveats the table understates:

- **"Search off" is not absolute.** Two bare runs of `p07` fetched MLX source from
  raw.githubusercontent.com. pi has no web-search tool but does have bash, which is
  why Amendment 3 said to measure this rather than assume it. Both were arm A — the
  arm with the least information — and `p07` is the task it was failing.
- **Arm B never loaded its placebo (0/50).** It pays the +1,613-token catalog cost and
  then ignores the content, having correctly judged eight PostgreSQL skills irrelevant.
  So C−B here measures prompt length, not content, which is what makes C−E the load-
  bearing contrast.

**Six contamination incidents were found and fixed before this dataset existed**, four
of them during this condition alone: an unsandboxed CLI whose bare arm read the hidden
grader and ran pytest against it; the author's own debug kit copies in `$TMPDIR` and
in `/private/tmp`, which two *placebo* runs read the real kit from; agent-written
`solution.py` files in `/private/tmp` read by later runs; and a sibling workspace left
by an interrupted sweep, which arm E read arm C's solution out of.

Each was caught by a check rather than by luck, and the fixes are structural rather
than janitorial — the sandbox now denies the repository and all of `/private/tmp`
except the run's own scratch, and at most one workspace exists at any moment. On this
machine, anything readable and answer-bearing is eventually found by an agent.

## What the two conditions say together

`RESULTS.md` opened with a null and closes with a sharper version of it.

Across **500 scored runs**, two harnesses, two models, and search both on and off, the
skill *format* never paid for itself: C−E was −3,582 (p=0.69) with search and +13,490
(p=0.19) without. The only significant content effect runs the other way — **raw
documentation costs 46,740 more tokens than the same facts written down plainly**.

The kit does buy something real and small: fewer model calls (−1.8, p=0.008), and the
only clean sweep on correctness. What it does not buy is the thing its structure is
for. The durable artifacts remain the probe scripts and the harness.
