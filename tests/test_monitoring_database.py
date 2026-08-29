"""Database configuration and deterministic Alembic migration tests."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from var_cvar_crypto_risk.monitoring.database import (
    DATABASE_URL_ENV,
    create_monitoring_engine,
    resolve_database_url,
    sanitized_database_label,
)
from var_cvar_crypto_risk.monitoring.models import Base


EXPECTED_TABLES = {
    "alembic_version",
    "daily_asset_states",
    "daily_portfolio_states",
    "daily_risk_forecasts",
    "experiment_events",
    "experiments",
    "monitoring_runs",
    "optimization_snapshots",
    "price_observations",
    "snapshot_allocations",
}


def _upgrade_database(path: Path) -> None:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{path.as_posix()}")
    command.upgrade(config, "head")


def test_default_database_url_and_safe_label(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    url = resolve_database_url(project_root=tmp_path)
    assert url.endswith("/data/monitoring/portfolio_monitor.db")
    assert sanitized_database_label(url) == "sqlite/portfolio_monitor.db"


def test_explicit_url_precedes_environment_without_leaking_credentials(
    monkeypatch,
) -> None:
    configured_url = "postgresql://private-user:" + "private-pass@example/db"
    monkeypatch.setenv(DATABASE_URL_ENV, configured_url)
    assert resolve_database_url("sqlite+pysqlite:///:memory:") == (
        "sqlite+pysqlite:///:memory:"
    )
    label = sanitized_database_label(
        configured_url
    )
    assert label == "postgresql/configured-database"
    assert "private" not in label and "example" not in label


def test_migration_head_creates_expected_schema_without_model_drift(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "migration.db"
    _upgrade_database(database_path)
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{database_path.as_posix()}"
    )
    try:
        assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            assert compare_metadata(context, Base.metadata) == []
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            assert revision == "0004_batch5_run_metadata"
    finally:
        engine.dispose()


def test_sqlite_foreign_keys_are_enabled_on_every_engine_connection(
    tmp_path: Path,
) -> None:
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{(tmp_path / 'foreign-keys.db').as_posix()}"
    )
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("PRAGMA foreign_keys")) == 1
    finally:
        engine.dispose()


def test_migration_can_downgrade_to_empty_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "downgrade.db"
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url", f"sqlite+pysqlite:///{database_path.as_posix()}"
    )
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{database_path.as_posix()}"
    )
    try:
        assert inspect(engine).get_table_names() == ["alembic_version"]
    finally:
        engine.dispose()
