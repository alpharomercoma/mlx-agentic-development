# Failure mining log

Bare-agent runs against candidate tasks, used to decide which tasks enter the scored
suite. A task only earns a place if the base model actually fails it. Tasks the model
already passes cannot show a kit effect -- they are at ceiling, and including them
would dilute the experiment while costing full price to run.

Recorded before the kit existed, so nothing here is contaminated by it.

## Round 1 -- codex-cli 0.146.0, gpt-5.6-terra, effort medium, arm A (bare)

| Task | Area | Result | Tokens | Wall | Discriminates? |
|---|---|---|---|---|---|
| `p01_metal_kernel_fused` | metal-kernels | **FAIL** 1/9 | 291,716 | 100 s | **yes** |
| `p02_train_step_grads` | transforms | pass 4/4 | 96,902 | 43 s | no |
| `p03_attention_speed` | performance | pass 6/6, **3.69x** | 178,974 | 61 s | no |

Mean cost 189k tokens per run. Codex's weekly `used_percent` did not move off 24.0%
across all three, so at whole-percent granularity these 568k tokens are under 1% of
the window.

### p01 -- the one real failure, and it is specific

The model wrote a well-structured kernel: built at module scope with a comment
explaining that construction JIT-compiles a Metal library, correct bounds guard,
correct use of `x_ndim`/`x_shape`. It failed on one thing, and failed every test:

```
error: subscripted value is not an array, pointer, or vector
    out[index] = x[index] * scale[0] + bias[0];
```

It passed the scalars as **0-dimensional** `mx.array`s and then **subscripted** them.
Those two choices are individually reasonable and jointly a compile error.

Established by direct experiment (MLX 0.32.0, M5):

| What is passed as an input | How it is bound | Correct usage in `source` |
|---|---|---|
| Python scalar (`3.0`) | by value | `scale` |
| 0-d array (`mx.array(3.0)`) | by value | `scale` |
| 1-d array (`mx.array([3.0])`), even 1 element | pointer | `scale[0]` |

Both mismatched combinations fail to compile, with different errors: subscripting a
by-value binding gives "subscripted value is not an array"; using a pointer binding
as a value gives "invalid operands to binary expression".

This is the kind of fact the kit exists to carry: unguessable, unmemorable, and fatal.

### p02 and p03 -- at ceiling, and why that matters

`p02` targeted the `mx.value_and_grad` vs `nn.value_and_grad` footgun, where
differentiating the wrong argument silently leaves parameters untouched. The model
used `nn.value_and_grad` correctly and evaluated `optimizer.state`.

`p03` targeted performance routing. The model went straight to
`mx.fast.scaled_dot_product_attention` with `mask="causal"` -- essentially identical
to the oracle -- and measured 3.69x.

**Read-across:** the base model already handles mainstream MLX well. A kit cannot
improve on a task the model already passes, so a suite built from plausible-sounding
MLX tasks would mostly measure nothing. The scored suite must concentrate where the
base model demonstrably fails: hardware-specific and API-binding details rather than
idiomatic usage.

Both tasks are retained as **ceiling controls**. A kit that makes a passing task fail
is a regression, and the experiment should be able to see that.

## Round 2 -- harder candidates, same configuration

| Task | Area | Result | Tokens | Web searches | Wall |
|---|---|---|---|---|---|
| `p05_kernel_rowsum` | metal-kernels | pass 9/9 | 240,738 | 2 | 86 s |
| `p06_mxfp4_quant` | quantization | pass 4/4 | 426,944 | 5 | 79 s |
| `p07_memory_api` | api-currency | pass 3/3 | 535,522 | 7 | 90 s |

`p05` was meant to force atomics and `init_value`. The model avoided the trap by
picking a better algorithm: a threadgroup shared-memory tree reduction writing
`out[row]` once from lane 0, which never accumulates and so never needs an
initialised buffer. That is a legitimately good solution, not a lucky escape.

`p07` produced the exactly-correct six-line answer using the current top-level
`mx.get_active_memory` / `mx.device_info` spellings -- after seven web searches.

## The result that reframes the experiment

Combining both rounds, ordered by cost:

| Task | Result | Tokens | Web searches |
|---|---|---|---|
| `p02_train_step_grads` | pass | 96,902 | 0 |
| `p03_attention_speed` | pass | 178,974 | 1 |
| `p05_kernel_rowsum` | pass | 240,738 | 2 |
| `p01_metal_kernel_fused` | **fail** | 291,716 | 3 |
| `p06_mxfp4_quant` | pass | 426,944 | 5 |
| `p07_memory_api` | pass | 535,522 | 7 |

**Pass rate 5/6. Token spread 5.5x. Correlation between token cost and web-search
count: 0.998.**

With web search available, a strong model faced with an unfamiliar accelerator API
does not usually get it wrong. **It pays.** It searches the web, reads, experiments,
and arrives at the right answer having spent five times the tokens of a familiar
task. The single failure, `p01`, is the narrow case where even that was not enough,
because the fact in question -- how MLX binds 0-d versus 1-d array inputs to a Metal
kernel -- is not clearly stated in any page the model found.

Consequences for the scored experiment:

1. **Correctness is at ceiling and cannot carry the headline.** A pass-rate
   comparison would be measuring an 83% baseline with little room to move, and would
   report "no significant effect" while missing the real one.
2. **Efficiency is the discriminating axis**, and it is continuous rather than
   binary, so it has far more statistical power at the same run count. The minimum
   detectable effect problem noted in the pre-registration is much less severe for
   tokens than for pass/fail.
3. **Leaving web search enabled in both arms was the right call and must stay.**
   Disabling it would likely have turned several of these passes into failures and
   made the kit look dramatically better than it is. The honest question is not
   "does the kit beat a model with no documentation" but "does it beat a model that
   can already search the documentation".

The hypothesis under test therefore becomes: *at matched correctness, the kit
reduces tokens, web searches, and tool calls* -- and additionally fixes the narrow
band of facts, like `p01`, that searching does not reliably surface.
