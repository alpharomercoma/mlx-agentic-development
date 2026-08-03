---
name: pg-profiling
description: |
  Capture and read PostgreSQL execution evidence: auto_explain, log analysis, and
  wait events. Use when the user says "profile postgres", "auto_explain", "slow query
  log", "pg_stat_activity", "wait events", "where is the time going", or needs
  evidence rather than a guess.
---

# Profiling PostgreSQL

## Complexity Assessment

**Start here, and usually stop here.** Before capturing anything, check
`pg_stat_statements` and `pg_stat_activity`. If one query dominates total time, or
sessions are piling up on a single wait event, you already have the answer and a
deeper capture will only confirm it more slowly.

**Medium** — a query that is slow only in production. Use `auto_explain`.

**Complex** — intermittent stalls with no obvious query. Sample wait events over time.

## auto_explain

```sql
LOAD 'auto_explain';
SET auto_explain.log_min_duration = '250ms';
SET auto_explain.log_analyze = on;
SET auto_explain.log_nested_statements = on;
```

Two things that silently defeat this:

1. **`log_analyze` adds real instrumentation overhead**, which on very short queries
   can exceed the query itself. Set a `log_min_duration` that excludes them.
2. **Without `log_nested_statements`, statements inside functions are invisible**, so
   a slow function shows as a single opaque call and the actual cost is hidden.

Loading it per-session works for investigation; for continuous capture it belongs in
`shared_preload_libraries`, which needs a restart.

## Wait events

`pg_stat_activity.wait_event_type` and `wait_event` tell you what sessions are
blocked on right now. One sample is nearly useless; the value is in sampling
repeatedly and looking at the distribution. Lock waits point at concurrency design,
IO waits at memory or storage, and a large `Client` share usually means the
application is the bottleneck rather than the database.

## Honesty rails

- **A log line shows what happened, not why.** State the inference and what would
  falsify it.
- **One sample of `pg_stat_activity` is an anecdote.** Say how long you sampled and
  how often.
- Say whether timing came from the server or the client; they measure different
  things and are not interchangeable.
