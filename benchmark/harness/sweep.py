#!/usr/bin/env python3
"""Run the scored sweep: every (task, arm, web-search, repeat) cell, interleaved.

Ordering matters as much as the cells. Runs are **shuffled with a fixed seed** rather
than run arm-by-arm, so that arm never correlates with cache warmth, rate-limit state,
chassis temperature, or time of day. On a fanless machine executing hundreds of runs
over many hours, running all of A then all of C would silently confound the arm with
the thermal and quota state of the machine.

Checkpointing: a completed run writes `result.json`, and the sweep skips any cell that
already has one. Interrupt and re-invoke freely.

Quota: Codex's weekly `used_percent` is read from each run's rollout and recorded. The
sweep stops when it crosses `--quota-stop`, so a long sweep degrades into a partial
result with intact data rather than a stream of failures.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from run import REPO, make_run_id, preflight_clean, run_trial

RESULTS = REPO / "benchmark" / "results"


def read_used_percent(run_dir: Path) -> float | None:
    rollout = run_dir / "rollout.jsonl"
    if not rollout.is_file():
        return None
    last = None
    for line in rollout.read_text(errors="replace").splitlines():
        if '"rate_limits"' not in line:
            continue
        try:
            payload = json.loads(line).get("payload") or {}
        except json.JSONDecodeError:
            continue
        rl = payload.get("rate_limits") or {}
        primary = rl.get("primary") or {}
        if "used_percent" in primary:
            last = primary["used_percent"]
    return last


def build_cells(tasks, arms, searches, repeats, harness, seed):
    cells = [
        {"task": t, "arm": a, "web_search": w, "repeat": r, "harness": harness}
        for t in tasks
        for a in arms
        for w in searches
        for r in range(repeats)
    ]
    random.Random(seed).shuffle(cells)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--search", nargs="+", default=["on"], choices=["on", "off"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--harness", default="codex", choices=["codex", "claude", "pi"])
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--provider", default="opencode-go")
    ap.add_argument("--thinking", default="medium")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--kit", type=Path, default=REPO)
    ap.add_argument("--placebo", type=Path, default=REPO / "benchmark" / "placebo-kit")
    ap.add_argument("--docs", type=Path, default=REPO / "benchmark" / "docs-kit")
    ap.add_argument("--facts", type=Path, default=REPO / "benchmark" / "facts-kit")
    ap.add_argument(
        "--quota-stop",
        type=float,
        default=85.0,
        help="stop when Codex weekly used_percent reaches this",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    offenders = preflight_clean()
    if offenders:
        print(
            "REFUSING TO RUN: fixture-shaped directories sit beside the workspace "
            "root, and agents will find them:",
            file=sys.stderr,
        )
        for o in offenders:
            print(f"  {o}", file=sys.stderr)
        return 3

    searches = [s == "on" for s in args.search]
    cells = build_cells(
        args.tasks, args.arms, searches, args.repeats, args.harness, args.seed
    )

    todo = []
    for c in cells:
        rid = make_run_id(
            c["harness"], c["task"], c["arm"], c["web_search"], c["repeat"], args.model
        )
        # A cell counts as done only if it produced a USABLE result. A run that
        # failed provider-side still writes result.json, so checkpointing on the
        # file's existence alone turns every outage into a permanent hole in the
        # dataset -- the cell is skipped forever on resume, and the arm silently
        # ends up with fewer repeats than its neighbours.
        done_file = RESULTS / "runs" / rid / "result.json"
        if done_file.is_file():
            try:
                prev = json.loads(done_file.read_text())
            except json.JSONDecodeError:
                prev = {}
            # Contaminated cells are re-run too. They are dropped from the analysis,
            # so treating them as done would leave the same permanent hole an
            # errored cell used to leave.
            if not prev.get("error") and not prev.get("contamination"):
                continue
        todo.append(c)

    done = len(cells) - len(todo)
    print(f"{len(cells)} cells, {done} already done, {len(todo)} to run")
    if args.dry_run:
        for c in todo[:15]:
            print("  ", c)
        if len(todo) > 15:
            print(f"   ... and {len(todo) - 15} more")
        return 0

    log = RESULTS / "sweep_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    # A bug in the per-trial code path shows up only after a cell finishes, and the
    # loop deliberately swallows exceptions so one bad cell cannot kill a long
    # sweep. Those two facts together mean a NameError can burn all 250 cells while
    # looking like progress -- which is exactly what happened on the first pi launch
    # (`name 'net' is not defined`, every cell). If nothing succeeds early, stop.
    consecutive_errors = 0
    ABORT_AFTER = 3

    # Transient provider outages are expected over a multi-day sweep -- one lasted
    # about 15 minutes and failed every call with "Stream ended without
    # finish_reason". A cell is retried through the backoff below before it counts
    # as a failure, so a blip costs minutes rather than aborting the run; a genuine
    # outage still exhausts the retries and trips the abort guard.
    RETRY_BACKOFF_S = (60, 300)

    for i, c in enumerate(todo, 1):
        t0 = time.time()
        r = None
        exc = None
        for attempt, backoff in enumerate((0, *RETRY_BACKOFF_S)):
            if backoff:
                print(
                    f"    retrying in {backoff}s (attempt {attempt + 1})", flush=True
                )
                time.sleep(backoff)
            try:
                r = run_trial(
                    c["task"],
                    c["harness"],
                    c["arm"],
                    c["repeat"],
                    kit=args.kit.resolve(),
                    placebo=args.placebo.resolve(),
                    docs=args.docs.resolve(),
                    facts=args.facts.resolve(),
                    results_root=RESULTS,
                    model=args.model,
                    effort=args.effort,
                    provider=args.provider,
                    thinking=args.thinking,
                    web_search=c["web_search"],
                )
                exc = None
            except Exception as e:  # noqa: BLE001 - a failed cell must not kill the sweep
                r, exc = None, e
            if r is not None and r.ok:
                break

        if exc is not None:
            print(f"[{i}/{len(todo)}] ERROR {c}: {exc}", flush=True)
            with log.open("a") as fh:
                fh.write(json.dumps({**c, "error": str(exc)}) + "\n")
            consecutive_errors += 1
            if consecutive_errors >= ABORT_AFTER:
                print(
                    f"\nABORTING: {consecutive_errors} consecutive cells failed. "
                    "This is a harness bug, not a flaky run; fix it and resume.",
                    file=sys.stderr,
                )
                return 4
            continue

        # A provider outage returns a *normal-looking* result: the trial completes,
        # the grader finds nothing, and the cell records FAIL at 0 tokens. Counting
        # that as a task failure would score an arm zero on a task it never
        # attempted, so it feeds the abort counter rather than the dataset.
        if not r.ok:
            consecutive_errors += 1
            print(
                f"[{i}/{len(todo)}] RUN FAILED {c['task']} arm={c['arm']}: "
                f"{(r.error or '')[:120]}",
                flush=True,
            )
            if consecutive_errors >= ABORT_AFTER:
                print(
                    f"\nABORTING: {consecutive_errors} consecutive runs failed "
                    "process- or provider-side. Fix the cause and resume.",
                    file=sys.stderr,
                )
                return 4
            continue

        consecutive_errors = 0

        rid = make_run_id(
            c["harness"], c["task"], c["arm"], c["web_search"], c["repeat"], args.model
        )
        used = read_used_percent(RESULTS / "runs" / rid)
        elapsed = time.time() - t0
        rate = (time.time() - started) / i
        remaining = (len(todo) - i) * rate

        print(
            f"[{i}/{len(todo)}] {c['task']:24s} arm={c['arm']} "
            f"search={'on' if c['web_search'] else 'off'} r{c['repeat']} -> "
            f"{'PASS' if r.passed else 'FAIL'} "
            f"{r.metrics['total_tokens']:>7,}tok {elapsed:5.0f}s "
            f"quota={used if used is not None else '?'}% "
            f"eta={remaining / 3600:.1f}h",
            flush=True,
        )
        with log.open("a") as fh:
            fh.write(json.dumps({**c, "used_percent": used, **r.as_dict()}) + "\n")

        if used is not None and used >= args.quota_stop:
            print(
                f"\nSTOPPING: weekly quota at {used}% >= {args.quota_stop}%. "
                f"{len(todo) - i} cells remain; re-run to resume.",
                file=sys.stderr,
            )
            return 2

    print(f"\nSweep complete in {(time.time() - started) / 3600:.1f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
