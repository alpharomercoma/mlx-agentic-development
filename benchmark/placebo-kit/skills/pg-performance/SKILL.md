---
name: pg-performance
description: |
  Make PostgreSQL fast: configuration that actually matters, connection handling,
  vacuum and bloat, and where time really goes. Use when the user says "postgres is
  slow", "high CPU", "connection limit", "too many connections", "autovacuum",
  "bloat", "shared_buffers", "work_mem", "checkpoint", or asks how to tune a server.
---

# PostgreSQL performance

## Complexity Assessment

**Quick win** — a slow application with no specific query in hand. Check connection
handling and `pg_stat_statements` first. Most problems end here and cost nothing to
find.

**Measured** — a specific regression. Follow "Measure before tuning" and read the
vacuum section.

**Deep** — configuration for a specific workload shape. Read
`references/configuration.md`.

## Look here first, in this order

1. **Connection count.** Each connection is a process. Hundreds of idle connections
   cost memory and context switches, and the fix is a pooler, not a bigger
   `max_connections`. Raising `max_connections` to solve connection exhaustion
   reliably makes things worse.
2. **`pg_stat_statements`.** Order by `total_exec_time`, not by mean. The query that
   takes 3 ms and runs a million times is the problem far more often than the one
   that takes 8 seconds and runs twice.
3. **Vacuum health.** Check `pg_stat_user_tables` for dead tuples and last autovacuum.
   A table autovacuum cannot keep up with will degrade steadily and silently.
4. **Only then**, individual query plans.

## Configuration that actually matters

| Setting | Rough shape |
|---|---|
| `shared_buffers` | around a quarter of RAM; more is not better, the OS cache also helps |
| `work_mem` | **per sort or hash node, per query**, not per server. A high value times many concurrent nodes is how servers run out of memory |
| `maintenance_work_mem` | generous; it bounds index build and vacuum speed |
| `effective_cache_size` | a planner hint about total cache, not an allocation |
| `random_page_cost` | lower it on SSDs; the default assumes spinning disks |

`work_mem` is the one that surprises people. A single query with several sorts can use
a multiple of it, and every concurrent connection can do the same.

## Vacuum and bloat

`UPDATE` and `DELETE` leave dead tuples; vacuum reclaims them. When autovacuum falls
behind, tables and indexes bloat, scans read more pages for the same rows, and
performance decays in a way no query rewrite will fix.

Long-running transactions are the usual root cause: vacuum cannot remove tuples still
visible to any open transaction, so one forgotten session holding a transaction open
can block reclamation database-wide. Check `pg_stat_activity` for old
`xact_start` values before tuning anything else.

`VACUUM FULL` rewrites the table and takes an `ACCESS EXCLUSIVE` lock. It is not a
routine operation.

## Measure before tuning

Change one setting at a time and measure against a fixed, repeatable workload. Take
several runs and report the spread, not a single number. A warm cache and a cold cache
differ by more than most configuration changes, so state which you measured, and
discard the first run.

Server-side timing from `pg_stat_statements` excludes network and client time.
Application-side latency includes them. They answer different questions and mixing
them produces confident nonsense.

## Honesty rails

- **A measurement is not a conclusion.** State the number, the inference, and what
  would falsify it.
- **Report the workload, concurrency, and data volume** behind any figure. A benchmark
  at one connection says nothing about behaviour at two hundred.
- **Never report a predicted improvement as measured.** If you changed a setting and
  did not re-run the workload, say so.
- Defaults are conservative but not wrong. "The defaults are bad" is not a diagnosis.
