# Attribution

This repository's *structure* is informed by two predecessors:

- AWS's [`neuron-agentic-development`](https://github.com/aws-neuron/neuron-agentic-development)
  (Apache-2.0), for Trainium/NKI.
- This author's `xla-agentic-development`, for TPU/Pallas.

**No file here is a copy or derivative work of any file in either repository.** No
script, no reference page, no prose was copied. What was borrowed is methodology:

- Router-style `SKILL.md` with tiered `references/` loading.
- A Complexity Assessment that decides how much reference material to load.
- Per-stage role assignment across specialised agents.
- Iteration caps and explicit blocked-state ladders.
- Honesty rails separating what was measured from what was inferred.
- A single canonical hardware-constants file with CI drift checking.
- A dated verification ledger distinguishing executed claims from doc-derived ones.

Amazon's copyright notice deliberately does **not** appear in `NOTICE`. Including it
would assert that this project redistributes their work, which it does not.

MLX documentation and source are cited per page in each skill's
`references/SOURCES.md`. MLX is MIT-licensed; Apple's published specifications are
cited, not reproduced.
