---
name: pg-partitioning
description: |
  Partition large PostgreSQL tables and manage the partitions over time. Use when the
  user says "partition this table", "declarative partitioning", "RANGE partition",
  "LIST partition", "partition pruning", "detach partition", "my table is too big",
  or is planning a retention policy.
---

# Partitioning

## Complexity Assessment

**Simple** — a time-series table with a retention window. Range-partition by the
timestamp, one partition per month. Read the pruning section and stop.

**Medium** — choosing a strategy, or partitioning an existing populated table. Read
the strategy table and the migration note.

**Complex** — sub-partitioning, hash partitioning for parallelism, foreign-key and
unique-constraint interactions. Read `references/constraints.md`.

## Strategy

| Strategy | Use when |
|---|---|
| `RANGE` | continuous keys, above all time. The common case by a wide margin |
| `LIST` | a small, stable set of discrete values, such as region or tenant class |
| `HASH` | no natural grouping and the goal is even spread rather than pruning |

Partitioning is not a performance feature in general. It helps for three specific
reasons: **pruning** lets the planner skip partitions entirely; **maintenance**
becomes per-partition, so `VACUUM` and index builds are bounded; and **retention**
becomes `DROP TABLE` rather than a `DELETE` that bloats. If none of those apply,
partitioning will usually make things slower by adding planning overhead.

## Pruning is the whole point, and it is easy to lose

The planner can only prune when the partition key appears in the predicate in a form
it can evaluate. Wrapping the key in a function, comparing against a volatile
expression, or filtering on a column derived from the key all defeat it. Verify with
`EXPLAIN` that the plan touches the partitions you expect and no others — the number
of scanned partitions in the plan is the measurement, not an assumption.

Pruning happens at planning time and, for some parameterised cases, at execution time.
A prepared statement with a generic plan may prune less than the same query run
directly.

## Managing partitions over time

Create the next partition **before** it is needed. A row that matches no partition is
an error, not a silent insert into a default, unless a `DEFAULT` partition exists —
and a `DEFAULT` partition prevents adding new partitions that would overlap rows it
already holds, so it is a trap as often as a safety net.

Detaching is the cheap way to expire data:

```sql
ALTER TABLE events DETACH PARTITION events_2025_01 CONCURRENTLY;
DROP TABLE events_2025_01;
```

`CONCURRENTLY` avoids holding an `ACCESS EXCLUSIVE` lock on the parent, which
otherwise blocks every reader for the duration.

## Migrating an existing table

There is no in-place conversion. The realistic options are a new partitioned table
plus a backfill and a cutover, or attaching the existing table as a single partition
of a new parent and growing sideways from there. Both need a plan for writes arriving
during the migration. Budget for this; it is the expensive part, not the DDL.

## Honesty rails

- **State how many partitions the plan actually scanned.** "Pruning works" without
  that number is not a finding.
- **A partitioned table with a bad key is slower than no partitioning.** Say what key
  you chose and why.
- Do not claim a retention policy works until you have run a full cycle, including
  the partition that did not exist yet.
