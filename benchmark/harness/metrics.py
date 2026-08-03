#!/usr/bin/env python3
"""Extract per-run efficiency metrics from each CLI's machine-readable output.

Codex writes two different machine-readable streams, with *different schemas*, and
both are needed:

  --json stdout stream   thread.started / turn.started / item.* / turn.completed.
                         `turn.completed.usage` carries the authoritative token
                         totals for the turn. It does NOT expose per-model-call
                         granularity.
  rollout file           $CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl,
                         with event_msg/token_count once per model call and
                         response_item/{function_call,custom_tool_call} per tool
                         invocation.

So token totals come from the stream and call counts come from the rollout. Note
that --ephemeral suppresses the rollout entirely, so the harness must not use it.

Claude Code's --output-format json emits a single object with `usage`, `num_turns`,
and `total_cost_usd`.

Raw input_tokens is the primary efficiency metric; cached tokens are reported
separately. Cache warmth swings enormously between a cold and a warm run, so folding
cached tokens into one "cost" number would make the metric depend mostly on run
order rather than on the arm.

Schemas verified by execution against codex-cli 0.146.0 on 2026-08-04.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Codex item types that represent the agent acting on the world rather than talking.
CODEX_TOOL_ITEMS = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
    "patch_apply",
}


@dataclass
class RunMetrics:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    turns: int = 0
    tool_calls: int = 0
    cost_usd: float | None = None
    # Kit skills the agent actually invoked. If this stays empty in the treatment
    # arm, the kit never fired, and any measured difference came from prompt length
    # rather than from the kit's content. This is the check that makes the headline
    # number interpretable.
    skills_invoked: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _iter_json_lines(path: Path):
    if not path or not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


# Codex does not expose a dedicated "invoke skill" tool. It loads a skill body by
# READING THE FILE from a shell command, e.g. `sed -n '1,240p' .../mlx-performance/
# SKILL.md`. Detecting invocation therefore means looking for a SKILL.md path in
# executed commands, not for a namespaced skill name in a tool call. The first pilot
# reported zero invocations across every arm-C run purely because of this.
_SKILL_PATH = re.compile(r"skills/([A-Za-z0-9_-]+)/SKILL\.md")


def _collect_kit_skills(blob: str, kit_prefix: str, out: list[str]) -> None:
    for name in _SKILL_PATH.findall(blob):
        if name not in out:
            out.append(name)
    # Claude Code namespaces skills as <plugin>:<skill> in tool calls, so that form
    # is still checked.
    if kit_prefix and kit_prefix in blob:
        for token in blob.replace("'", '"').split('"'):
            if token.startswith(kit_prefix) and token not in out:
                out.append(token)


def find_rollout(codex_home: Path, thread_id: str | None) -> Path | None:
    """Locate the rollout for a run. Thread id appears in the filename."""
    sessions = codex_home / "sessions"
    if not sessions.is_dir():
        return None
    candidates = sorted(sessions.rglob("rollout-*.jsonl"))
    if thread_id:
        for c in candidates:
            if thread_id in c.name:
                return c
    return candidates[-1] if candidates else None


def parse_codex(events_path: Path, codex_home: Path, kit_prefix: str) -> RunMetrics:
    m = RunMetrics()
    thread_id: str | None = None

    for rec in _iter_json_lines(events_path):
        etype = rec.get("type")

        if etype == "thread.started":
            thread_id = rec.get("thread_id")

        elif etype == "turn.started":
            m.turns += 1

        elif etype == "turn.completed":
            u = rec.get("usage") or {}
            # Usage is reported per turn, so accumulate rather than overwrite.
            m.input_tokens += u.get("input_tokens", 0)
            m.cached_input_tokens += u.get("cached_input_tokens", 0)
            m.cache_write_tokens += u.get("cache_write_input_tokens", 0)
            m.output_tokens += u.get("output_tokens", 0)
            m.reasoning_tokens += u.get("reasoning_output_tokens", 0)

        elif etype in ("item.completed", "item.started"):
            item = rec.get("item") or {}
            if etype == "item.completed" and item.get("type") in CODEX_TOOL_ITEMS:
                m.tool_calls += 1
            _collect_kit_skills(json.dumps(item), kit_prefix, m.skills_invoked)

    m.total_tokens = m.input_tokens + m.output_tokens

    # Per-model-call granularity only exists in the rollout.
    rollout = find_rollout(codex_home, thread_id)
    if rollout:
        rollout_tool_calls = 0
        for rec in _iter_json_lines(rollout):
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type")
            if rec.get("type") == "event_msg" and ptype == "token_count":
                m.model_calls += 1
            elif rec.get("type") == "response_item" and ptype in (
                "function_call",
                "custom_tool_call",
            ):
                rollout_tool_calls += 1
                _collect_kit_skills(json.dumps(payload), kit_prefix, m.skills_invoked)
        # The rollout sees every tool call, including ones the stream summarises
        # into a single file_change item, so prefer it when it saw more.
        m.tool_calls = max(m.tool_calls, rollout_tool_calls)

    return m


def parse_claude(stream_path: Path, kit_prefix: str) -> tuple[RunMetrics, bool]:
    """Parse a Claude Code --output-format stream-json JSONL file.

    Returns (metrics, rate_limited). stream-json is used rather than the single
    `json` object because the latter reports no tool calls at all; tool use is only
    visible as tool_use content blocks on assistant messages.

    Caution: Claude's token accounting is NOT comparable to Codex's. Claude reports
    almost the entire prompt as cache_read (observed: input_tokens=4 alongside
    cache_read=55,750), whereas Codex reports a large raw input_tokens. Compare
    token metrics within a harness only, never across the two.
    """
    m = RunMetrics()
    rate_limited = False

    for d in _iter_json_lines(stream_path):
        dtype = d.get("type")

        if dtype == "rate_limit_event":
            # A rate_limit_event is emitted routinely to report quota utilisation,
            # so its mere presence means nothing. Only a status other than
            # allowed/allowed_warning indicates the run was actually throttled.
            # Observed statuses include "allowed_warning" at 0.92 utilisation on a
            # completely unthrottled run.
            status = (d.get("rate_limit_info") or {}).get("status", "")
            if status not in ("allowed", "allowed_warning", ""):
                rate_limited = True

        elif dtype == "assistant":
            for block in (d.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    m.tool_calls += 1
                    _collect_kit_skills(json.dumps(block), kit_prefix, m.skills_invoked)

        elif dtype == "result":
            usage = d.get("usage") or {}
            m.input_tokens = usage.get("input_tokens", 0)
            m.cached_input_tokens = usage.get("cache_read_input_tokens", 0)
            m.cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
            m.output_tokens = usage.get("output_tokens", 0)
            m.total_tokens = m.input_tokens + m.output_tokens
            m.turns = d.get("num_turns", 0)
            m.model_calls = m.turns
            m.cost_usd = d.get("total_cost_usd")

    return m, rate_limited
