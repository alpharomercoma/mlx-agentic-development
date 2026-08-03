---
name: pg-replication-workflows
description: |
  Set up and operate PostgreSQL replication, backups, and failover. Use when the user
  says "streaming replication", "read replica", "logical replication", "replication
  lag", "pg_basebackup", "WAL", "point in time recovery", "failover", or is planning
  for durability.
---

# Replication and recovery

## Complexity Assessment

**Simple** — one read replica for reporting. `pg_basebackup` plus a standby signal.
Read the lag section and stop.

**Medium** — logical replication for a subset of tables, or a major-version upgrade
path. Read the physical-versus-logical table.

**Complex** — synchronous commit, automated failover, point-in-time recovery targets.
Read `references/failover.md`.

## Physical or logical

| | Physical (streaming) | Logical |
|---|---|---|
| Granularity | whole cluster | selected tables |
| Cross-version | no | yes |
| Standby writable | no | yes |
| DDL replicated | yes | **no** |
| Cost | low | higher, per-row decoding |

Logical replication not replicating DDL is the detail that bites. A column added on
the publisher and not on the subscriber breaks replication at the next write to that
table, and the failure surfaces later than the change that caused it.

## Lag is not one number

Replay lag, write lag, and flush lag are different, and `pg_stat_replication` reports
them separately. A replica can have received everything and replayed none of it. For
read-after-write correctness, replay lag is the one that matters.

Long-running queries on a standby conflict with replay. Either the query is cancelled
or replay stalls, depending on `max_standby_streaming_delay` and
`hot_standby_feedback`. Both settings trade replica freshness against query success,
and there is no configuration that gives both.

## Backups are not replication

A replica follows your mistakes. `DROP TABLE` replicates in under a second. Backups
exist for the failure modes replication shares rather than protects against.

Point-in-time recovery needs a base backup **and** a continuous archive of WAL. Test
the restore. An untested backup is a hypothesis, and the moment you need it is the
worst time to discover the archive has a gap.

## Honesty rails

- **State the lag metric you mean.** "Replication is healthy" without one is not a
  claim.
- **An untested restore is not a backup.** Say when it was last exercised end to end.
- Do not describe a failover procedure you have not run. The gap between the
  documented steps and the real ones is where outages live.
