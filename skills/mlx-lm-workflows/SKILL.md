---
name: mlx-lm-workflows
description: |
  Run, serve, convert, and fine-tune language models with mlx-lm. Use when the user
  says "mlx_lm.generate", "run a model locally on my Mac", "mlx_lm.server", "convert
  to MLX", "LoRA", "QLoRA", "fine-tune on Apple silicon", "mlx-community", "KV
  cache", or "speculative decoding".
---

# mlx-lm workflows

> **Unverified.** No model was downloaded, served, converted, or fine-tuned while
> writing this page. The commands are documentation-derived, and the 24 GB sizing
> figures are extrapolated from upstream measurements taken on a 64 GB M4 Max, not
> measured here. Run `mlx_lm.<command> --help`, which is authoritative for your
> installed version, before trusting any flag below.
>
> This page is also **narrower than mlx-lm actually is**: continuous batching, the LRU
> prompt cache, the per-model tool-call parsers, `batch_generate`, `evaluate`,
> `perplexity`, and the learned-quantisation commands all exist and are not covered.

## Complexity Assessment

**Simple** — generate text or serve a model. Two commands below. Stop.

**Medium** — convert or quantise a model, or fit one into limited memory. Add the
memory section.

**Complex** — LoRA/QLoRA fine-tuning or learned quantisation. Read
the LoRA section below and `mlx_lm.lora --help`, which is
authoritative for your installed version.

## Generate and serve

```bash
mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt "..."
mlx_lm.server  --model mlx-community/Llama-3.2-3B-Instruct-4bit    # port 8080
```

The server is OpenAI-compatible at `POST /v1/chat/completions`. Its own docs warn it
implements only basic security checks and is not for production. Note `temperature`
defaults to **0.0** there, unlike most OpenAI-compatible servers.

Python:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, verbose=True))
```

## Convert and quantise

```bash
mlx_lm.convert --model <hf-repo> -q            # 4-bit by default
```

Learned quantisation, in increasing cost: `mlx_lm.dynamic_quant` (fastest,
per-layer sensitivity), `mlx_lm.awq`, `mlx_lm.gptq`, `mlx_lm.dwq` (distilled, best at
2–4 bits). Methods cascade. Distilling to 8 or 6 bits "often doesn't work well" per
upstream; DWQ is for the low-bit end.

## Fitting in unified memory

On a 24 GB machine, a 4-bit ~8B model is roughly 4.5–5 GB of weights; a 4-bit
30B-A3B MoE was measured upstream at 18.2 GB, and its 8-bit form at 33.5 GB does not
fit. Levers, in order of effect:

- `--kv-bits 4` or `8` — quantised KV cache
- `--max-kv-size` — rotating fixed-size cache; 512 is frugal and worse, 4096+ better
- lower `--batch-size` and `--max-seq-length` when training
- `mx.clear_cache()` between phases

MLX's default memory limit is 1.5× the recommended working-set size, so it will let
you into swap before raising. Swapping looks like a hang, not an error.

## LoRA

```bash
uv pip install "mlx-lm[train]"
mlx_lm.lora --model <m> --train --data <dir> --iters 600
mlx_lm.fuse --model <m>          # merge adapters into a standalone model
```

**QLoRA is implicit**: if `--model` points at a quantised model you get QLoRA,
otherwise plain LoRA. There is no `--qlora` flag. `--fine-tune-type` is
`lora` (default), `dora`, or `full`. `--data` needs `train.jsonl`, optionally
`valid.jsonl` and `test.jsonl`.

The model list in upstream's LORA.md is stale relative to the code; do not treat it
as the set of supported architectures.

## Honesty rails

- **Quote the model, quantisation, and prompt length behind any throughput number.**
  Tokens/sec without those is meaningless.
- Do not claim a model "fits" unless you ran it; swap makes the difference between
  fits and unusable invisible to a size calculation.
