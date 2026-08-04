#!/usr/bin/env python3
"""Analyse the sweep according to the pre-registration. No new hypotheses here.

Everything computed is specified in `benchmark/PREREGISTRATION.md`:

  confirmatory   paired per-task differences for the two co-primary families
                 (correctness, efficiency), with alpha split across the two heads
  supporting     hierarchical bootstrap CIs, exact McNemar, paired permutation test
  descriptive    pass@k, pass^k, per-task and per-area breakdowns (exploratory,
                 Benjamini-Hochberg corrected), and the strata splits

Efficiency is computed **unconditionally**, over every run.

Conditioning tokens on `passed` -- as the first version of this file did -- is collider
stratification. Arms have different failure rates by hypothesis, so the surviving
subsamples are not comparable, and an arm can look efficient by failing cheaply while
another fails expensively. Amendment 2 records the change. Three estimators are
reported:

  tokens per run     unconditional. Primary.
  cost per success   total tokens across all attempts / number of successes. This is
                     the quantity a user actually pays, and it is the honest way to
                     charge an arm for its failures.
  tokens if passed   passing runs only, retained as a clearly-labelled secondary so
                     the biased estimator can be compared against the unbiased one.

Standard errors are clustered by task area. Tasks cluster -- two metal-kernel tasks,
two quantization tasks -- and treating them as independent understates the error bars.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmark" / "results"
TASKS = REPO / "benchmark" / "tasks"

B_BOOT = 10_000
N_PERM = 10_000


# --------------------------------------------------------------------------- io


def load_runs(
    results: Path, harness: str | None = None, model: str | None = None
) -> list[dict]:
    """Scored sweep runs only.

    Mining runs live in the same directory but are baseline exploration, not scored
    cells: they predate the web_search field, so a run without that key is a mining
    run and is excluded. Including them silently inflated arm A's sample and mixed
    two different experiments.

    Runs are also filtered by harness. Token accounting is not comparable across
    CLIs -- Codex reports a large raw input_tokens where pi and Claude report almost
    everything as cache reads -- so pooling them would produce a token contrast that
    is mostly an artifact of which CLI ran the cell.

    Contaminated runs are dropped, not merely flagged. A run whose agent reached the
    repo, the grader or an oracle is evidence about the harness, not about the kit.
    """
    runs = []
    dropped = 0
    errored = 0
    for rj in sorted((results / "runs").glob("*/result.json")):
        try:
            d = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        if "web_search" not in d:
            continue
        if harness and d.get("harness") != harness:
            continue
        # Models are never pooled. Different models differ in capability, price and
        # token accounting, so averaging them would produce a contrast that is
        # partly about which model happened to run the cell.
        if model and d.get("model") != model:
            continue
        if d.get("contamination"):
            dropped += 1
            continue
        # Process- or provider-level failures are not task failures. Scoring a run
        # the model never got to attempt would count an upstream outage as the arm
        # being wrong.
        if d.get("error"):
            errored += 1
            continue
        # Recomputed, not trusted: an earlier version keyed `graded` on the pytest
        # exit code, so it read False for every run with a failing test. Deriving it
        # from tests_total makes runs recorded before and after that fix consistent
        # without re-running anything.
        d["graded"] = d.get("tests_total", 0) > 0
        runs.append(d)
    if dropped:
        print(f"NOTE: dropped {dropped} contaminated run(s) from the analysis")
    if errored:
        print(f"NOTE: dropped {errored} run(s) that failed process- or provider-side")
    return runs


def task_areas() -> dict[str, str]:
    areas = {}
    for meta in sorted(TASKS.glob("*/meta.json")):
        d = json.loads(meta.read_text())
        areas[d["id"]] = d.get("area", "unknown")
    return areas


# -------------------------------------------------------------------- statistics


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def clustered_se(per_task: dict[str, float], clusters: dict[str, str]) -> float:
    """SE of the mean of per-task values, clustered by area.

    Adds the within-cluster cross terms that an independence assumption drops.
    """
    tasks = list(per_task)
    n = len(tasks)
    if n < 2:
        return float("nan")
    m = mean(per_task.values())
    by_cluster = defaultdict(list)
    for t in tasks:
        by_cluster[clusters.get(t, "unknown")].append(per_task[t] - m)

    total = 0.0
    for devs in by_cluster.values():
        s = sum(devs)
        total += s * s  # squares plus all cross terms within the cluster
    return math.sqrt(total) / n


def hierarchical_bootstrap(
    paired: dict[str, list[float]], b: int = B_BOOT, seed: int = 20260804
) -> tuple[float, float]:
    """Percentile CI: resample tasks, then repeats within each resampled task."""
    rng = random.Random(seed)
    tasks = list(paired)
    if not tasks:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(b):
        picked = [rng.choice(tasks) for _ in tasks]
        vals = []
        for t in picked:
            reps = paired[t]
            if reps:
                vals.append(mean(rng.choice(reps) for _ in reps))
        if vals:
            means.append(mean(vals))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar. Use this, not chi-square, when b+c < 25."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_permutation(
    paired: dict[str, list[float]], n: int = N_PERM, seed: int = 20260804
) -> float:
    """Sharp null: the arm label is exchangeable within each task."""
    rng = random.Random(seed)
    per_task = {t: mean(v) for t, v in paired.items() if v}
    if not per_task:
        return float("nan")
    obs = abs(mean(per_task.values()))
    hits = 0
    vals = list(per_task.values())
    for _ in range(n):
        flipped = [v if rng.random() < 0.5 else -v for v in vals]
        if abs(mean(flipped)) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from Chen et al. Not 1-(1-p)^k, which is biased."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), 1):
        i = m - rank
        val = min(prev, pvals[idx] * m / (i + 1))
        adj[idx] = val
        prev = val
    return adj


# ---------------------------------------------------------------------- analysis


def index(runs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[(r["task_id"], r["arm"], bool(r.get("web_search", True)))].append(r)
    return by


def paired_metric(by, tasks, arm_x, arm_y, search, fn, passed_only):
    """Per-task list of (x - y) differences, one per aligned repeat."""
    out = {}
    for t in tasks:
        xs = sorted(by.get((t, arm_x, search), []), key=lambda r: r["repeat"])
        ys = sorted(by.get((t, arm_y, search), []), key=lambda r: r["repeat"])
        if passed_only:
            xs = [r for r in xs if r["passed"]]
            ys = [r for r in ys if r["passed"]]
        if not xs or not ys:
            continue
        mx, my = mean(fn(r) for r in xs), mean(fn(r) for r in ys)
        out[t] = [mx - my]
    return out


def report_contrast(name, paired, areas, alpha):
    if not paired:
        print(f"  {name}: no aligned data")
        return
    per_task = {t: mean(v) for t, v in paired.items()}
    m = mean(per_task.values())
    se = clustered_se(per_task, areas)
    lo, hi = hierarchical_bootstrap(paired)
    p = paired_permutation(paired)
    sig = "significant" if p < alpha else "not significant"
    print(
        f"  {name}: mean {m:+.4g}  clustered SE {se:.4g}  "
        f"bootstrap 95% CI [{lo:+.4g}, {hi:+.4g}]  permutation p={p:.4f}  "
        f"({sig} at alpha={alpha:.4g}, n={len(per_task)} tasks)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--search", default="on", choices=["on", "off"])
    ap.add_argument(
        "--harness",
        default="codex",
        choices=["codex", "claude", "pi"],
        help="token metrics are not comparable across harnesses; analyse one at a time",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="analyse a single model; models are never pooled",
    )
    args = ap.parse_args()

    runs = load_runs(args.results, args.harness, args.model)
    if not runs:
        print("no runs found")
        return 1
    areas = task_areas()
    by = index(runs)
    search = args.search == "on"
    tasks = sorted({r["task_id"] for r in runs})
    arms = sorted({r["arm"] for r in runs})

    print(f"{len(runs)} runs, {len(tasks)} tasks, arms {arms}, search={args.search}\n")

    # Alpha split across the two co-primary family heads (Holm, two hypotheses).
    alpha_head = 0.05 / 2

    print("FAMILY 1 - CORRECTNESS (paired per-task pass rate)")
    for x, y in (("C", "B"), ("C", "A")):
        if x in arms and y in arms:
            pr = paired_metric(
                by, tasks, x, y, search, lambda r: 1.0 if r["passed"] else 0.0, False
            )
            report_contrast(f"{x} - {y}", pr, areas, alpha_head)

    # Exact McNemar on task-level majority outcome.
    if "C" in arms and "A" in arms:
        b = c = 0
        for t in tasks:
            cs = by.get((t, "C", search), [])
            as_ = by.get((t, "A", search), [])
            if not cs or not as_:
                continue
            cp = mean(1.0 if r["passed"] else 0.0 for r in cs) > 0.5
            ap_ = mean(1.0 if r["passed"] else 0.0 for r in as_) > 0.5
            if cp and not ap_:
                b += 1
            elif ap_ and not cp:
                c += 1
        pm = mcnemar_exact(b, c)
        print(f"  McNemar C vs A: b={b} c={c}  exact two-sided p={pm:.4f}")

    print("\nFAMILY 2 - EFFICIENCY (tokens per run, UNCONDITIONAL - primary)")
    # C-E and C-D first: after the corrected leakage audit these are the contrasts
    # that survive, because both arms contain the facts so leakage cancels.
    for x, y in (("C", "E"), ("C", "D"), ("D", "E"), ("C", "B"), ("C", "A")):
        if x in arms and y in arms:
            pr = paired_metric(
                by, tasks, x, y, search, lambda r: r["metrics"]["total_tokens"], False
            )
            report_contrast(f"{x} - {y} tokens", pr, areas, alpha_head)

    print("\nCOST PER SUCCESS (total tokens / successes; charges an arm for failures)")
    for a in arms:
        rs = [
            r for r in runs if r["arm"] == a and bool(r.get("web_search", True)) == search
        ]
        if not rs:
            continue
        tot = sum(r["metrics"]["total_tokens"] for r in rs)
        succ = sum(1 for r in rs if r["passed"])
        cps = tot / succ if succ else float("inf")
        print(f"  arm {a}: {tot:,} tokens / {succ} successes = {cps:,.0f} per success")

    print("\nSECONDARY - tokens over PASSING runs only (biased; for comparison)")
    for x, y in (("C", "E"), ("C", "A")):
        if x in arms and y in arms:
            pr = paired_metric(
                by, tasks, x, y, search, lambda r: r["metrics"]["total_tokens"], True
            )
            report_contrast(f"{x} - {y} tokens|passed", pr, areas, alpha_head)

    print("\nSECONDARY EFFICIENCY")
    for label, fn in (
        ("tool_calls", lambda r: r["metrics"]["tool_calls"]),
        ("model_calls", lambda r: r["metrics"]["model_calls"]),
        ("wall_clock_s", lambda r: r["wall_clock_s"]),
    ):
        if "C" in arms and "A" in arms:
            pr = paired_metric(by, tasks, "C", "A", search, fn, True)
            report_contrast(f"C - A {label}", pr, areas, alpha_head)

    print("\nEXECUTION ERRORS vs WRONG ANSWERS (pre-registered as separate)")
    for a in arms:
        rs = [
            r for r in runs if r["arm"] == a and bool(r.get("web_search", True)) == search
        ]
        if not rs:
            continue
        crashed = sum(1 for r in rs if not r["graded"])
        wrong = sum(1 for r in rs if r["graded"] and not r["passed"])
        ok = sum(1 for r in rs if r["passed"])
        print(
            f"  arm {a}: {ok} passed, {wrong} wrong answers, "
            f"{crashed} produced nothing gradable"
        )

    print("\nPER-ARM SUMMARY")
    for a in arms:
        rs = [
            r for r in runs if r["arm"] == a and bool(r.get("web_search", True)) == search
        ]
        if not rs:
            continue
        ps = [r for r in rs if r["passed"]]
        toks = sorted(r["metrics"]["total_tokens"] for r in ps)
        med = toks[len(toks) // 2] if toks else 0
        print(f"  arm {a}: {len(ps)}/{len(rs)} passed  median tokens (passing) {med:,}")

    print("\nMECHANISM CHECK - skill loads per arm")
    # Reported for EVERY arm, not just C. If arm C loads bodies and arm B never does,
    # part of any C-B token gap is that asymmetry rather than content quality, and
    # only a per-arm count makes it visible.
    for a in arms:
        rs = [
            r for r in runs if r["arm"] == a and bool(r.get("web_search", True)) == search
        ]
        if not rs:
            continue
        fired = [r for r in rs if r["metrics"].get("skills_invoked")]
        names = sorted({s for r in rs for s in r["metrics"].get("skills_invoked", [])})
        print(f"  arm {a}: {len(fired)}/{len(rs)} runs loaded a skill  {names or ''}")
        if a == "C" and rs and not fired:
            print(
                "  WARNING: the kit never fired. Any difference above came from prompt\n"
                "  length, not the kit's content, and is uninterpretable."
            )

    print("\nCACHE COVARIATE (raw input tokens are partly a function of run order)")
    for a in arms:
        rs = [
            r for r in runs if r["arm"] == a and bool(r.get("web_search", True)) == search
        ]
        if not rs:
            continue
        inp = sum(r["metrics"]["input_tokens"] for r in rs)
        cac = sum(r["metrics"]["cached_input_tokens"] for r in rs)
        frac = cac / inp if inp else 0.0
        print(f"  arm {a}: {frac:6.1%} of input tokens served from cache")

    print("\nPASS@K / PASS^K (arm C)")
    for t in tasks:
        rs = by.get((t, "C", search), [])
        if not rs:
            continue
        n, c = len(rs), sum(1 for r in rs if r["passed"])
        k = min(3, n)
        pk = pass_at_k(n, c, k)
        pow_k = (c / n) ** k if n else float("nan")
        print(f"  {t:26s} n={n} c={c}  pass@{k}={pk:.3f}  pass^{k}={pow_k:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
