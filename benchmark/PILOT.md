# Pilot (superseded — retained as a record)

> **These numbers do not support any claim and are not a result.**
>
> Two reasons. First, this pilot ran against the kit as first written, which carried
> nine dangling `references/` pointers, so arm C was sent to missing files on Complex
> tasks — a handicap, not a fair measurement. Second, the reported 40% token
> difference is **smaller than the observed within-cell spread** (bare `p07` spanned
> 269k–484k on two runs), so at 2 repeats over 3 tasks it is indistinguishable from
> noise.
>
> The runs are quarantined under `benchmark/results_prerepair/` and excluded from
> analysis. What this pilot did establish is operational and still valid: the harness
> works, the isolation holds, and the per-run cost and wall-clock figures used for
> sizing are real.

# Pilot (2026-08-04)

3 tasks × 3 arms × 2 repeats = 18 runs, web search on. Codex 0.146.0,
`gpt-5.6-terra`, effort medium.

The first pilot was discarded for control-arm contamination; see `MINING.md`. This is
the replacement, run after the workspace fix.

## Validity checks — both pass

- **Contamination: 0 of 18 runs referenced the kit's skills directory.** In the
  discarded pilot every arm did.
- **Mechanism: 6 of 6 arm-C runs loaded a kit skill**, drawing on
  `mlx-metal-kernels`, `mlx-performance`, `mlx-compile-and-transforms`, and
  `mlx-env-setup`. The kit fires, and description-based routing picks plausible
  skills.

## Results

| Task | Arm A (bare) | Arm B (placebo) | Arm C (kit) |
|---|---|---|---|
| `p01_metal_kernel_fused` | 0/2 — 268k, 284k | 0/2 — 183k, 436k | **1/2** — 143k, 279k |
| `p02_train_step_grads` | 2/2 — 82k, 98k | 2/2 — 74k, 75k | 2/2 — 133k, 78k |
| `p07_memory_api` | 1/2 — 484k, 269k | 1/2 — 414k, 260k | **2/2** — 97k, 161k |

| Arm | Pass | Mean tokens |
|---|---|---|
| A bare | 3/6 | 247,663 |
| B placebo | 3/6 | 240,484 |
| **C kit** | **5/6** | **148,466** |

## Reading it

**The placebo does nothing.** That is the one directionally useful signal here, and
it should be read cautiously: Arm B
lands on the same pass rate as arm A and within 3% of its token cost, despite adding
a prompt of the same size as the kit. So the arm-C effect is not a prompt-length
artefact — which is exactly what arm B exists to establish, and it could easily have
come out the other way.

**Arm C is better on both axes**: 5/6 versus 3/6, and 40% fewer tokens. On `p07` the
kit reached 2/2 at 97k and 161k tokens where the bare arm spent 484k and 269k for
1/2. On `p01`, which nothing else has ever passed, the kit passed once.

**This also retrospectively confirms the contamination fix mattered.** In the
discarded pilot, bare `p07` cost 132k; clean, it costs 269–484k, back in line with
mining's 536k. The contaminated control looked cheap precisely because it was reading
the kit.

## Sizing

18 runs took 18.9 minutes, mean 63 s and 212k tokens per run. Quota moved to 31%.

Projected for the full web-search-on sweep at 10 tasks × 5 repeats × 3 arms = 150
runs: **~2.6 hours and ~32M tokens, about 27% of the weekly window.** Affordable.
The search-off condition follows and should be cheaper, since search is what costs.

**Caveats.** Two repeats per cell over three tasks is far too small to test anything;
these are directional numbers only, and the pre-registered analysis is what counts.
Within-cell variance remains large — bare `p07` spanned 269k–484k, and placebo `p01`
spanned 183k–436k, roughly 2×. That variance is the reason the efficiency family
needs its five repeats.
