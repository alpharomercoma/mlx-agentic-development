#!/usr/bin/env python3
"""Measure how much of each task's solution already appears inside the kit.

If a kit reference page contains a task's answer, the experiment measures retrieval,
not assistance. This audit quantifies that mechanically, and its table is published
with the results.

Why it is load-bearing here
---------------------------
The intended design had a separate author write the gold solutions so that whoever
wrote the kit had never seen them. That did not happen: the same author wrote both.
So author independence cannot be claimed, and this mechanical audit is the only
real defence. It is reported honestly rather than assumed away.

Two measures, per (task, kit file) pair:

  n-gram overlap    longest shared run of normalised tokens, and the count of shared
                    13-grams. 13 tokens is the conventional contamination threshold.
  similarity        difflib ratio over normalised source, as a blunt whole-file check
                    that catches paraphrase the n-gram measure misses.

Tasks are then reported in strata:

  in-kit      shares a 13-gram with some kit file. The kit may be supplying the
              answer; treat any gain here as an upper bound.
  novel       no 13-gram shared with any kit file. This is the stratum whose result
              actually generalises.

Comments and docstrings are stripped before comparison: a kit that explains a concept
in prose is doing its job, whereas a kit that ships the code is not.
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import tokenize
from pathlib import Path

NGRAM = 13


def strip_python(source: str) -> str:
    """Remove comments and docstrings, leaving executable structure only."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source

    out: list[str] = []
    prev_type = tokenize.INDENT
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        # A string that stands alone as a statement is a docstring.
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.DEDENT,
        ):
            prev_type = tok.type
            continue
        skip = (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
        if tok.type not in skip:
            out.append(tok.string)
        prev_type = tok.type
    return " ".join(out)


def normalise(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", text)


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def longest_common_run(a: list[str], b: list[str]) -> int:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return max((m.size for m in matcher.get_matching_blocks()), default=0)


def code_blocks_from_markdown(text: str) -> str:
    return "\n".join(re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL))


def kit_corpus(kit: Path) -> dict[Path, str]:
    """Every piece of code the kit ships, keyed by path."""
    corpus: dict[Path, str] = {}
    for p in sorted(kit.rglob("*")):
        if not p.is_file() or ".git" in p.parts or "benchmark" in p.parts:
            continue
        if p.suffix == ".py":
            corpus[p] = strip_python(p.read_text(encoding="utf-8", errors="replace"))
        elif p.suffix == ".md":
            blocks = code_blocks_from_markdown(
                p.read_text(encoding="utf-8", errors="replace")
            )
            if blocks.strip():
                corpus[p] = strip_python(blocks)
    return corpus


def audit(kit: Path, tasks_dir: Path) -> dict:
    corpus = kit_corpus(kit)
    corpus_tokens = {p: normalise(t) for p, t in corpus.items()}
    corpus_ngrams = {p: ngrams(t, NGRAM) for p, t in corpus_tokens.items()}

    rows = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        oracle = task_dir / "oracle" / "solution.py"
        if not oracle.is_file():
            continue
        sol = strip_python(oracle.read_text(encoding="utf-8"))
        sol_tokens = normalise(sol)
        sol_ngrams = ngrams(sol_tokens, NGRAM)

        # Score every kit file, then report the single closest one. Rank by shared
        # 13-grams first (the contamination signal that matters), then by whole-file
        # similarity as a tiebreak.
        shared_total = 0
        best: tuple[int, float, int, Path] | None = None
        for path, tokens in corpus_tokens.items():
            shared = len(sol_ngrams & corpus_ngrams[path])
            shared_total += shared
            ratio = difflib.SequenceMatcher(None, sol, corpus[path]).ratio()
            run = longest_common_run(sol_tokens, tokens)
            cand = (shared, ratio, run, path)
            if best is None or cand[:3] > best[:3]:
                best = cand

        worst_run = best[2] if best else 0
        worst_ratio = best[1] if best else 0.0
        worst_file = best[3] if best else None

        rows.append(
            {
                "task": task_dir.name,
                "shared_13grams": shared_total,
                "longest_common_run": worst_run,
                "max_similarity": round(worst_ratio, 3),
                "closest_kit_file": str(worst_file.relative_to(kit))
                if worst_file
                else None,
                "stratum": "in-kit" if shared_total > 0 else "novel",
            }
        )
    return {"ngram": NGRAM, "kit_files_scanned": len(corpus), "tasks": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="also write the report here")
    args = ap.parse_args()

    report = audit(args.kit.resolve(), args.tasks.resolve())
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))

    print(
        f"Kit files scanned: {report['kit_files_scanned']}  "
        f"(n-gram = {report['ngram']})\n"
    )
    hdr = (
        f"{'task':28s} {'13-grams':>9s} {'run':>5s} {'sim':>6s}  "
        f"{'stratum':8s} closest kit file"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in report["tasks"]:
        print(
            f"{r['task']:28s} {r['shared_13grams']:9d} {r['longest_common_run']:5d} "
            f"{r['max_similarity']:6.3f}  {r['stratum']:8s} "
            f"{r['closest_kit_file'] or '-'}"
        )

    in_kit = [r["task"] for r in report["tasks"] if r["stratum"] == "in-kit"]
    if in_kit:
        print(
            f"\n{len(in_kit)} task(s) share a {NGRAM}-gram with the kit: "
            f"{', '.join(in_kit)}.\nReport these separately; a gain on them bounds "
            "memorisation-substitution rather than demonstrating assistance."
        )
    else:
        print("\nNo task shares a 13-gram with any kit file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
