# Monitoring Database and Operations

## Storage boundary

Phase 8 uses SQLAlchemy 2.x repositories and an explicit unit of work. Alembic
owns schema migrations. The local default is SQLite:

```text
data/monitoring/portfolio_monitor.db
```

Override it with a SQLAlchemy URL in the environment:

```text
QCPRF_MONITORING_DATABASE_URL
```

The application describes configured storage with a sanitized dialect/database
label. It must never print a password, host, complete secret-bearing URL, or
absolute local database path. PostgreSQL compatibility informs the repository
boundary, but a production PostgreSQL deployment is outside Phase 8.

## Initialize or upgrade the schema

Install the exact locked environment first, then run the committed migrations:

```bash
uv sync --locked --extra app --extra dev
uv run --locked --no-sync alembic upgrade head
```

The Streamlit monitor checks for the expected migration revision and shows the
same migration command if storage is not ready. Do not replace Alembic with
`Base.metadata.create_all()` for a durable application database; that helper is
reserved for isolated tests.

Migration files are append-only release history. Review an upgrade against a
backup, especially when using a non-default database. Phase 8 does not provide
automated backup, restore, retention, encryption, or disaster recovery.

## Run the app and one-shot updater

Start the interactive application:

```bash
uv run --locked --no-sync streamlit run app.py
```

The app can perform one bounded **Update Now** operation for an active Live or
Hybrid experiment. It does not run a scheduler or background loop.

For scheduled operation, invoke the installed one-shot command from an external
service such as cron or launchd:

```bash
uv run --locked --no-sync qcprf-monitor \
  --experiment-id 00000000-0000-0000-0000-000000000000

uv run --locked --no-sync qcprf-monitor --all-active
```

Optional arguments include `--requested-cutoff YYYY-MM-DD`, timezone-aware
`--as-of`, `--code-version`, `--calculation-version`, and `--database-url`.
Prefer the environment variable over command-line database credentials because
process arguments can be visible to other local users or job logs.

The command prints JSON and exits nonzero when an all-active run has any failed
experiment. Provider failures are recorded with sanitized summaries; related
financial writes roll back. Repeating an identical successful update is
idempotent and should not duplicate finalized records.

## Persistence and lifecycle

The schema stores:

- experiment identities, date boundaries, modes, statuses, and events;
- one immutable activated optimization snapshot and target allocation;
- normalized price observations and provider provenance;
- daily portfolio and per-asset states;
- origin-safe risk forecasts and matured evaluations; and
- monitoring-run cutoffs, counts, source metadata, and sanitized failures.

Archive changes lifecycle status and retains history. The current public API and
UI do not expose hard deletion or unarchive. Database rows should not be edited
manually to bypass lifecycle, immutability, or audit rules.

## Export and privacy

Dashboard CSV/JSON downloads and full experiment bundles are private by default.
They may disclose target holdings, fixed quantities, launch prices, daily
prices, NAV, risk forecasts, breaches, and realized performance. Exporting does
not make the files publication-safe.

The full bundle manifest omits database URLs and credentials, labels privacy,
and records file hashes. Before any deliberate publication:

1. inspect every exported file;
2. confirm the holdings and results are synthetic or explicitly approved;
3. verify source-data redistribution rights;
4. remove local identifiers not needed for the research claim; and
5. run the repository publication-boundary checks on the intended Git content.

The database, WAL/SHM/journal sidecars, downloaded data, and generated outputs
are ignored by Git. Never force-add them. A local ignore rule is not encryption
or access control: protect the workspace, backups, and any external database
with appropriate operating-system and infrastructure controls.

## Operational limitations

- Single-user local SQLite is the supported MVP, not a high-availability service.
- Authentication, authorization, multi-tenancy, and role-based access are absent.
- No internal retry daemon, job queue, alerting service, or health monitor exists.
- Concurrent and production workloads require separate design and validation.
- No broker, exchange, transaction, order, or real-money integration is present.

See [User guide](user-guide.md), [Portfolio monitoring](portfolio-monitoring.md),
and [Public release checklist](public-release-checklist.md).
