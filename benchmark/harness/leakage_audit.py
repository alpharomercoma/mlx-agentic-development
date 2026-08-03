#!/usr/bin/env python3
"""Measure how much of each task's answer the kit already states.

If a kit page contains a task's answer, the experiment measures retrieval, not
assistance. This quantifies that mechanically, and the table is published with the
results.

Why this file was rewritten
---------------------------
The first version compared **only fenced Python blocks**. The kit's payload is
Markdown tables and prose, so essentially all of it was discarded before comparison.
The consequence was not subtle: `p07_memory_api` was classified **"novel"** while its
exact answer -- the four current memory/device calls and the `device_name` key -- sat
in a six-row table in `skills/mlx-performance/SKILL.md`. The "novel stratum" the
pre-registration called "the one that generalises" was an artifact of a regex.

Two measures are now computed per task:

  n-gram overlap    shared 13-grams over ALL kit text (prose, tables, code), the
                    conventional contamination threshold. Catches copied wording.
  symbol coverage   the fraction of the API symbols and literal keys the ORACLE uses
                    that appear anywhere in the kit. Catches the case above, where the
                    kit states the answer without sharing any phrasing.

Symbol coverage is the one that matters here. A kit that names every symbol the
solution needs has supplied the answer, however differently it is worded.

Strata:
  in-kit   shares a 13-gram, or symbol coverage >= SYMBOL_THRESHOLD
  novel    neither

Comments and docstrings are stripped from the ORACLE before symbol extraction, so a
comment in the gold solution cannot inflate coverage.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import tokenize
from pathlib import Path

NGRAM = 13
# A task counts as in-kit when the kit names at least this fraction of the symbols the
# oracle uses. Not a tuned value: at 0.6 a kit must supply a clear majority of the
# specific API surface, which is the point at which "helped" and "told" stop being
# distinguishable.
SYMBOL_THRESHOLD = 0.6

# Symbols too generic to be evidence of anything.
STOP_SYMBOLS = {
    "mx.array",
    "mx.eval",
    "mlx.core",
    "mlx.nn",
    "int",
    "str",
    "float",
    "bool",
    "range",
    "len",
    "print",
    "list",
    "dict",
    "tuple",
    "shape",
    "dtype",
}


def strip_python(source: str) -> str:
    """Remove comments and docstrings, leaving executable structure only."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    out: list[str] = []
    prev = tokenize.INDENT
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.DEDENT,
        ):
            prev = tok.type
            continue
        if tok.type not in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            out.append(tok.string)
        prev = tok.type
    return " ".join(out)


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :]
    return text


def normalise(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", flat)


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def oracle_symbols(source: str) -> set[str]:
    """Dotted API calls and literal string keys the oracle depends on.

    `mx.get_active_memory()` and `["device_name"]` are both extracted, because both
    are things the solver has to know and either can be handed over by a kit.
    """
    symbols: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return symbols

    def dotted(node: ast.AST) -> str | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = dotted(node)
            if name and "." in name:
                symbols.add(name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{2,}", node.value):
                symbols.add(node.value)
        elif isinstance(node, ast.keyword) and node.arg:
            symbols.add(node.arg)
    return {s for s in symbols if s not in STOP_SYMBOLS}


def kit_corpus(kit: Path) -> dict[Path, str]:
    """Every piece of text the agent actually RECEIVES -- prose, tables and code.

    Scoped to `skills/`, because that is exactly what `prepare_codex_home` copies into
    an arm. Counting README.md, AGENTS.md or VERIFICATION.md would overstate leakage
    by crediting the kit with files the agent is never given. Being wrong in the
    kit's favour and being wrong against it are both wrong.
    """
    root = kit / "skills" if (kit / "skills").is_dir() else kit
    corpus: dict[Path, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git" in p.parts or "benchmark" in p.parts:
            continue
        if p.suffix == ".py":
            corpus[p] = strip_python(p.read_text(encoding="utf-8", errors="replace"))
        elif p.suffix == ".md":
            corpus[p] = strip_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
    return corpus


def audit(kit: Path, tasks_dir: Path) -> dict:
    corpus = kit_corpus(kit)
    joined = "\n".join(corpus.values())
    corpus_tokens = {p: normalise(t) for p, t in corpus.items()}
    corpus_ngrams = {p: ngrams(t, NGRAM) for p, t in corpus_tokens.items()}

    rows = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        oracle = task_dir / "oracle" / "solution.py"
        if not oracle.is_file():
            continue
        raw = oracle.read_text(encoding="utf-8")
        sol = strip_python(raw)
        sol_ngrams = ngrams(normalise(sol), NGRAM)

        symbols = oracle_symbols(raw)
        present = sorted(s for s in symbols if s in joined)
        coverage = len(present) / len(symbols) if symbols else 0.0

        shared_total = 0
        best: tuple[int, float, Path] | None = None
        for path in corpus_tokens:
            shared = len(sol_ngrams & corpus_ngrams[path])
            shared_total += shared
            hits = sum(1 for s in symbols if s in corpus[path])
            score = hits / len(symbols) if symbols else 0.0
            cand = (shared, score, path)
            if best is None or cand[:2] > best[:2]:
                best = cand

        stratum = (
            "in-kit" if shared_total > 0 or coverage >= SYMBOL_THRESHOLD else "novel"
        )
        rows.append(
            {
                "task": task_dir.name,
                "shared_13grams": shared_total,
                "symbols_total": len(symbols),
                "symbols_in_kit": len(present),
                "symbol_coverage": round(coverage, 3),
                "closest_kit_file": str(best[2].relative_to(kit)) if best else None,
                "stratum": stratum,
                "leaked_symbols": present,
            }
        )
    return {
        "ngram": NGRAM,
        "symbol_threshold": SYMBOL_THRESHOLD,
        "kit_files_scanned": len(corpus),
        "tasks": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--verbose", action="store_true", help="list the leaked symbols")
    args = ap.parse_args()

    report = audit(args.kit.resolve(), args.tasks.resolve())
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))

    print(
        f"Kit files scanned: {report['kit_files_scanned']}  "
        f"(13-gram + symbol coverage, in-kit at >= {report['symbol_threshold']})\n"
    )
    hdr = (
        f"{'task':26s} {'13-gr':>6s} {'symbols':>9s} {'cover':>6s}  "
        f"{'stratum':8s} closest"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in report["tasks"]:
        print(
            f"{r['task']:26s} {r['shared_13grams']:6d} "
            f"{r['symbols_in_kit']:4d}/{r['symbols_total']:<4d} "
            f"{r['symbol_coverage']:6.2f}  {r['stratum']:8s} "
            f"{r['closest_kit_file'] or '-'}"
        )
        if args.verbose and r["leaked_symbols"]:
            print(f"{'':26s} leaked: {', '.join(r['leaked_symbols'])}")

    in_kit = [r["task"] for r in report["tasks"] if r["stratum"] == "in-kit"]
    n = len(report["tasks"])
    print(
        f"\n{len(in_kit)}/{n} tasks are in-kit: {', '.join(in_kit) or 'none'}."
        "\nA gain on those bounds memorisation-substitution rather than demonstrating"
        "\nassistance. The novel stratum is the one whose result generalises."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
