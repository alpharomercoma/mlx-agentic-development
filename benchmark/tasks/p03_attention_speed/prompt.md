`attention.py` in this directory implements multi-head scaled dot-product attention
with a causal mask. It is correct but slow.

Create `solution.py` exposing a function with the identical signature:

    attention(q, k, v) -> mx.array

where `q` has shape [B, H, T, D] and `k`, `v` have shape [B, H, S, D]. It must apply
the same scaling (1/sqrt(D)) and the same causal mask, and return the same result as
the provided implementation to within float32 tolerance.

Make it as fast as you can on this machine. Correctness is not negotiable: a faster
function that changes the numerics is a failure.
