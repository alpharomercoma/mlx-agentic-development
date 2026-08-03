# Pre-registration

Committed and tagged **before any scored run**. Mining runs (`benchmark/MINING.md`)
preceded this document and were used to select tasks; they are not scored data and no
result below was chosen after seeing scored outcomes.

Amendments after the tag must be recorded in "Deviations" with a date and reason,
never by silent edit.

## Question

Does an agent skills kit for Apple MLX change what a coding agent produces, and at
what cost, relative to (a) no kit and (b) a content-free kit of the same size?

## Design

Factorial, paired by task.

| Factor | Levels |
|---|---|
| Arm | **A** bare · **B** placebo (token-matched, unrelated content) · **C** real kit |
| Web search | **on** · **off** |
| Harness | Codex CLI (primary) · Claude Code (replication) |

Arm B is what separates "this content helped" from "a longer prompt helped".
C−A is the effect of having a kit; **C−B is the effect of the kit's content**, and
C−B is the claim that matters.

The web-search factor exists because mining showed the base model rarely fails when
it can read documentation — it pays instead. `off` measures how much of the kit's
value is substituting for documentation access; `on` is the harder and more honest
question.

**Tasks:** 10, in `benchmark/tasks/`, each with a prompt, a pinned workspace, a
hidden grader, and an oracle solution verified to pass that grader. Ten rather than
the twelve originally planned; see Deviations.

**Repeats:** 5 per cell (Codex), 3 (Claude).

**Run counts:** Codex 10 × 5 × 3 × 2 = 300. Claude 10 × 3 × 2 × 2 = 120.
Staged: web-search-`on` first, since it carries the primary claim.

## Metrics

Two **co-primary** families. Alpha is split between them (Holm across the two family
heads, so each head is tested at the level that preserves a 0.05 family-wise rate).
Neither is promoted over the other after seeing results.

### Family 1 — Correctness

- **`passed`**: all hidden tests green. Primary correctness statistic is the paired
  per-task pass rate.
- Secondary: fraction of tests passed; `pass@k` via the unbiased estimator
  `1 − C(n−c,k)/C(n,k)`; `pass^k`; execution errors counted separately from wrong
  answers.
- **Guardrail:** the kit must not reduce pass rate on the ceiling tasks
  (`p02`, `p03`, `p05`, `p07`), which the bare model already passes. A significant
  drop there is a regression and is reported as such regardless of other results.

### Family 2 — Efficiency

Reported at **matched correctness** (restricted to runs that passed), because a
cheap failure is not an efficiency win.

- **Primary:** total tokens per run.
- Secondary: web-search count, tool calls, model calls, wall clock.
- `input_tokens` is reported raw, with cached tokens separate. Cache warmth swings
  between cold and warm runs, so any single blended cost figure would largely
  measure run order.

**Token counts are never compared across harnesses.** Claude reports nearly the whole
prompt as `cache_read` (observed `input_tokens=4` against `cache_read=55,750`) where
Codex reports large raw input. Comparisons stay within a harness.

### Speed (task-specific)

For `p03`, KernelBench `fast_p`: correct **and** at least p× the naive baseline,
swept over p rather than reported at one point. 100 timed trials, 3 warmups, `mx.eval`
inside the timed region, CV reported. Arms interleaved and a thermal canary logged,
because the chassis is fanless and sustained work throttles.

### Mechanism check (not a hypothesis test)

Count kit-skill invocations in arm C. **If arm C never loads a kit skill, the result
is uninterpretable** — any difference came from prompt length, not the kit — and that
is reported prominently rather than buried.

## Analysis

- Unit of analysis is the **task**; repeats are nested within it.
- Paired per-task differences (C−B primary, C−A secondary).
- **Clustered standard errors by task area.** Tasks cluster (2 metal-kernels,
  2 quantization, 2 performance, …) and treating them as independent understates
  the error bars.
- **Hierarchical bootstrap** CIs: resample tasks with replacement, then repeats
  within each resampled task, B = 10,000.
- Exact **McNemar** on paired task-level pass/fail, with the discordant counts b and
  c reported raw so readers can see when power is thin.
- **Paired permutation test** (arm label flipped within task, 10,000 draws) as an
  assumption-free cross-check. If it disagrees with McNemar, the effect is being
  carried by a couple of tasks and that is stated.
- Per-task and per-area breakdowns are **exploratory**, labelled as such, and
  Benjamini–Hochberg corrected. The two family heads are the only confirmatory tests.

### Strata, reported separately

- **Leakage stratum.** `leakage_audit.py` splits tasks into *in-kit* (shares a
  13-gram with some kit file) and *novel*. The novel stratum is the one that
  generalises; a gain on the in-kit stratum bounds memorisation-substitution.
  Pre-kit baseline: **all 10 tasks novel, zero shared 13-grams.**
- **Ceiling stratum.** Tasks the bare model already passes, versus the band where it
  fails.

## Minimum detectable effect, stated in advance

At 10 tasks × 5 repeats, paired, the design detects roughly a **20–25 percentage
point** swing in per-task pass rate at α = 0.05 and 80% power. It cannot resolve a
5-point difference, and a null correctness result must be read as "no large effect",
not "no effect".

Repeats shrink within-task variance but never between-task variance, so beyond ~5
repeats only more *tasks* buy power.

The efficiency family is continuous and therefore substantially better powered at the
same run count. This is why correctness alone was judged insufficient after mining.

## Predictions, recorded before running

1. Correctness: **no significant C−A or C−B difference overall.** Baseline pass rate
   is 5/6 in mining; there is little room.
2. Efficiency: **C < A and C < B in tokens**, concentrated on the high-search tasks
   (`p06`, `p07`, `p09`), with little effect on `p02`.
3. `p01` is the one task where correctness may move, being the case where searching
   did not surface the needed fact.
4. With web search **off**, the correctness gap should widen in the kit's favour.
5. Most individual skills will show no measurable effect. A published blind A/B of 40
   prompt artifacts found only 7 changed behaviour; that is the prior.

Recording these means a result matching them is confirmation and a result
contradicting them is a finding, rather than both being narrated as success.

## Deviations from the original plan

| Deviation | Reason |
|---|---|
| 10 tasks, not 12 | Ten verified tasks with oracles and negative tests were judged worth more than twelve rushed. Reduces power modestly; the MDE above already reflects 10. |
| Co-primary correctness **and** efficiency, rather than correctness alone | Mining found correctness at ceiling (5/6) and token cost varying 5.5× with r = 0.998 against web-search count. Correctness alone would have reported a null while missing the real effect. Decided before any scored run. |
| Web search added as a full second factor | Follows from the same finding. |
| Gold solutions **not** authored independently | The plan called for a separate author so the kit author had never seen the solutions. The same author wrote both. Author independence cannot be claimed; the mechanical leakage audit is the actual defence and its table is published. This is a real limitation, not a resolved one. |

---

# Amendment 1 — 2026-08-04, before any scored run

Recorded rather than silently applied, as the document requires. No scored data
existed when this was written: the only sweep had been stopped at 12 runs and
discarded, and the pilot is explicitly non-confirmatory.

## The leakage audit was broken, and the corrected result removes the novel stratum

The original audit compared **only fenced Python blocks**. The kit's payload is
Markdown tables and prose, so almost all of it was discarded before comparison.
`p07_memory_api` was classified "novel" while its exact answer sat in a six-row table
in `skills/mlx-performance/SKILL.md`.

The audit now measures all delivered kit text and adds **symbol coverage**: the
fraction of API symbols and literal keys the oracle uses that appear anywhere in the
kit. Scoped to `skills/`, which is exactly what an arm receives.

Corrected result:

| Task | Shared 13-grams | Symbol coverage | Stratum |
|---|---|---|---|
| `p01_metal_kernel_fused` | 5 | 0.81 | in-kit |
| `p02_train_step_grads` | 26 | 0.67 | in-kit |
| `p03_attention_speed` | 2 | 0.71 | in-kit |
| `p04_quantized_matmul` | 4 | **1.00** | in-kit |
| `p05_kernel_rowsum` | 9 | 0.83 | in-kit |
| `p06_mxfp4_quant` | 0 | **1.00** | in-kit |
| `p07_memory_api` | 0 | 0.62 | in-kit |
| `p08_compile_state` | 16 | **1.00** | in-kit |
| `p09_rope_offsets` | 0 | 0.71 | in-kit |
| `p10_bool_mask_assign` | 0 | 0.83 | in-kit |

**10 of 10 tasks are in-kit.** The novel stratum does not exist.

The earlier line "Pre-kit baseline: all 10 tasks novel, zero shared 13-grams" is
**withdrawn**. It was vacuous — the kit did not exist when it was computed — and it
read as a reassurance it could not support.

## What this does to the claims

**C−A and C−B can no longer be read as evidence of assistance.** With the answer
present in the kit for every task, those contrasts bound *memorisation-substitution*:
they measure how much cheaper it is to be told than to search. That is a real and
reportable quantity, but it is not "the kit makes the agent better at MLX", and it
will not be described as such.

**The contrasts that survive are C−E and C−D**, and they become the primary questions:

- **C−E (kit vs. the same facts as an unstructured cheat sheet).** Both arms contain
  the facts, so leakage is held constant and cancels. This isolates whether the skill
  *format* — frontmatter, description-based routing, Complexity Assessment tiering,
  progressive disclosure, honesty rails — contributes anything beyond the text.
- **C−D (kit vs. size-matched raw upstream docs).** Both contain the material; this
  isolates whether *curation* beats *concatenation*.

These were added for independent reasons before this audit was corrected. They now
carry the experiment. Without them, the corrected leakage result would have left
nothing interpretable.

## Revised predictions

6. **C−E is null or small.** If the facts are what matter, structure is decoration.
   This is the prediction most likely to be uncomfortable and most worth recording.
7. **C−D favours C**, because raw docs are long and unrouted — but by less than C−A.
8. C−A remains large on the high-search tasks and is reported as an upper bound on
   memorisation-substitution, never as assistance.
