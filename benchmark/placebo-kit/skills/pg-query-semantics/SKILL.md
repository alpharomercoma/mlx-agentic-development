---
name: pg-query-semantics
description: |
  PostgreSQL's evaluation semantics: NULL handling, type coercion, transaction
  isolation, and locking. Use when the user says "why is my query wrong", "NULL
  comparison", "unexpected cast", "deadlock", "serialization failure", "row not
  visible", "SELECT FOR UPDATE", or is porting SQL from MySQL or SQLite. "NOT IN with nulls", "coalesce", "advisory lock", "read committed".
---

# PostgreSQL query semantics

## Complexity Assessment

**Simple** — one surprising result. Read "The four rules" and stop.

**Medium** — a concurrency bug or a port from another engine. Add
`references/porting.md`.

**Complex** — custom isolation, advisory locks, long transactions. Read everything.

## The four rules

**1. NULL is not a value and does not compare.** `NULL = NULL` is NULL, not true.
`x IN (1, NULL)` is never false — it is true or NULL. Use `IS NULL`, `IS DISTINCT
FROM`, and be deliberate about `NOT IN` against a nullable subquery, which is the
classic silent wrong-answer bug.

**2. Aggregates skip NULLs, but `count(*)` does not.** `avg(x)` over a column that is
half NULL averages the non-NULL half. That is usually what you want and almost never
what the reader assumes.

**3. Type resolution is eager and sometimes surprising.** An untyped literal takes its
type from context; a mismatch can silently choose a cast that prevents index use.
`WHERE id = '42'` may work while `WHERE text_col = 42` errors.

**4. Read Committed is the default, and it is not repeatable.** Two identical queries
in one transaction can return different rows. Statements see rows committed before the
statement began, not before the transaction began. If you need stability across
statements, ask for `REPEATABLE READ` explicitly and be ready to handle
serialization failures with a retry.

## Locking reflexes

| Intent | Use |
|---|---|
| read a row you are about to update | `SELECT ... FOR UPDATE` |
| avoid blocking on a locked row | `FOR UPDATE SKIP LOCKED` |
| avoid holding a lock during slow work | fetch, release, re-check on write |
| serialise a whole workflow by key | advisory locks, not table locks |

Deadlocks are normal under concurrency; the fix is consistent lock ordering plus a
retry loop, not a bigger lock.

## Honesty rails

- **A query that returns rows is not a correct query.** Say what you compared it
  against.
- Report the isolation level any concurrency claim depends on.
- If you did not run it against real data volumes, say so — semantics that hold at ten
  rows can hide at ten million.
