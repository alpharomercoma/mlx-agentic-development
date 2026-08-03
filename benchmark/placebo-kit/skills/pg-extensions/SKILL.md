---
name: pg-extensions
description: |
  Write and use PostgreSQL extensions and server-side functions. Use when the user
  says "CREATE EXTENSION", "write a PL/pgSQL function", "custom aggregate", "C
  extension", "pg_stat_statements", "trigger function", "my function is slow", or
  needs behaviour the built-in SQL surface does not provide.
---

# Extensions and server-side code

## Complexity Assessment

**Simple** — install and use an existing extension. One statement, below. Stop.

**Medium** — a PL/pgSQL function or trigger. Read the volatility section; it is the
thing that silently costs performance.

**Complex** — a C extension, custom aggregate, or custom type. Read
`references/c-extensions.md`.

**Before writing anything, ask whether it is needed.** `pg_stat_statements`,
`pg_trgm`, `postgis`, and `hstore` cover an enormous amount of what people write by
hand, and they are maintained by people who do this full time.

## Installing

```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
SELECT * FROM pg_available_extensions ORDER BY name;
```

Extensions are per-database, not per-cluster. Installing one in `postgres` does not
make it available in your application database, which is a recurring source of
"but I installed it" confusion.

## The five things that go wrong in server-side functions

**1. Volatility is a promise the planner believes.** Marking a function `IMMUTABLE`
when it reads tables lets the planner cache results and produce wrong answers. Marking
a genuinely immutable function `VOLATILE` — the default — prevents indexing on it and
forces re-evaluation per row.

**2. `SECURITY DEFINER` without a fixed `search_path` is a privilege-escalation
hole.** Always set `search_path` explicitly on such functions.

**3. Exception blocks are not free.** A `BEGIN ... EXCEPTION` block establishes a
subtransaction on every call. In a hot loop this dominates.

**4. Row-by-row processing in PL/pgSQL is usually the wrong shape.** A single
set-based statement typically beats a loop by orders of magnitude.

**5. Triggers run per row and are easy to make quadratic.** A trigger that queries the
table it is attached to is the standard way to turn a bulk load into an outage.

## Debugging

1. `RAISE NOTICE` for values; it goes to the client and the log.
2. `EXPLAIN ANALYZE` the statements *inside* the function, not just the call.
3. `auto_explain` with nested statements enabled shows what the function actually ran.
4. Check `pg_stat_user_functions` for call counts and total time.

## Honesty rails

- **A function that returns the right answer once is not verified.** Say which inputs
  and volumes you tested.
- **State the volatility you declared and why.** It is a correctness claim, not a hint.
- Do not claim a performance win without measuring the surrounding workload, not just
  the function in isolation.
