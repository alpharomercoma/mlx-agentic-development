---
name: pg-env-setup
description: |
  Install and verify a PostgreSQL development environment. Use when the user says
  "install postgres", "psql won't connect", "which postgres version", "initdb",
  "connection refused", "pg_ctl", "docker postgres", or a script fails at connect
  time.
---

# PostgreSQL environment setup

## Complexity Assessment

**Simple** — fresh local install. Run the block below and the verification gate. Stop.

**Medium** — an existing cluster misbehaving. Run the gate first; it usually names
the problem.

**Complex** — multiple clusters, custom builds, extensions from source. Read
`references/clusters.md`.

## Install

```bash
brew install postgresql@17 && brew services start postgresql@17
createdb "$(whoami)"
```

Requirements worth checking before anything else:

- **Server and client major versions should match.** A newer `psql` against an older
  server mostly works; the reverse frequently does not, and the failures are obscure.
- **`initdb` locale and encoding are fixed at cluster creation** and cannot be changed
  later without a dump and reload. UTF-8 unless you have a specific reason.
- The data directory is version-specific. A major upgrade needs `pg_upgrade` or a
  dump/restore, never an in-place binary swap.

## Verification gate — run this before anything else

```sql
SELECT version();
SHOW data_directory;
SHOW shared_buffers;
```

If `version()` fails you have a connection problem, not a query problem, and nothing
else here applies. Check in this order: is the server running, is it listening on the
expected socket or port, does `pg_hba.conf` permit your user and method, and is the
database name right.

## Current versions and dead ends

- PostgreSQL 17 is current; 12 and earlier are out of support and receive no security
  fixes.
- `pg_dump` from a newer major version can read older servers, not the reverse.
- Trust authentication on a network-reachable port is a standing vulnerability, not a
  convenience.

## Honesty rails

- Report the exact server version you verified against. "Works with Postgres" is not
  a claim.
- If you could not run a query, say so rather than asserting the behaviour.
