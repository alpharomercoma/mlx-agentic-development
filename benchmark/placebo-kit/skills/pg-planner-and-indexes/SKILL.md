---
name: pg-planner-and-indexes
description: |
  Read PostgreSQL query plans and choose indexes. Use when the user says "EXPLAIN",
  "sequential scan", "index not used", "slow query", "ANALYZE", "wrong row estimate",
  "should I add an index", "composite index order", or pastes a plan and asks what it
  means. "BRIN", "partial index", "covering index", "stale statistics".
---

# The planner and indexes

## Complexity Assessment

**Simple** — one slow query. Run `EXPLAIN (ANALYZE, BUFFERS)` and read the estimate
versus actual rows. Usually finishes here.

**Medium** — index design for a workload. Read the index-type table.

**Complex** — planner misestimation, extended statistics, partial and expression
indexes. Read `references/statistics.md`.

## Read the plan in one specific way

`EXPLAIN (ANALYZE, BUFFERS)` and then compare **estimated rows against actual rows**,
node by node, starting at the deepest node where they diverge. Almost every planner
problem is a bad estimate propagating upward. A node estimating 1 row and returning
100,000 explains everything above it.

Costs are in arbitrary units and are not milliseconds. Do not compare them across
machines or configurations.

## Index types

| Type | For |
|---|---|
| B-tree | equality and range on scalars; the default and usually right |
| Hash | equality only; rarely worth it over B-tree |
| GIN | containment: arrays, `jsonb`, full-text |
| GiST | geometric, ranges, nearest-neighbour |
| BRIN | very large, naturally ordered tables; tiny and cheap |

Composite index column order follows the query, not the schema: equality columns
first, then the range column, then anything used only for ordering. An index on
`(a, b)` serves `WHERE a = ? AND b = ?` and `WHERE a = ?`, but not `WHERE b = ?`.

## Why an index is ignored

In rough order of frequency: the statistics are stale, so run `ANALYZE`; the predicate
is not sargable because a function wraps the column; a type mismatch forces a cast;
the table is small enough that a sequential scan is genuinely cheaper; or the query
returns a large enough fraction of the table that the index would cost more.

An unused index is not free — it slows every write and consumes cache.

## Honesty rails

- **`EXPLAIN` without `ANALYZE` shows estimates, not reality.** Say which you ran.
- **A faster plan on an empty table is not a faster plan.** State the row counts.
- Never claim an index helped without measuring before and after on comparable data.
