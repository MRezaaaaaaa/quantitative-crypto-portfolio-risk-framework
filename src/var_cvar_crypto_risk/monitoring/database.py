"""SQLAlchemy engine and transaction configuration for monitoring storage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker


DATABASE_URL_ENV = "QCPRF_MONITORING_DATABASE_URL"
DEFAULT_DATABASE_RELATIVE_PATH = Path("data/monitoring/portfolio_monitor.db")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_database_url(
    database_url: str | None = None, *, project_root: Path | None = None
) -> str:
    """Resolve an explicit URL, environment URL, or safe local SQLite default."""
    explicit = database_url.strip() if database_url is not None else ""
    configured = explicit or os.getenv(DATABASE_URL_ENV, "").strip()
    if configured:
        make_url(configured)
        return configured
    root = (project_root or PROJECT_ROOT).resolve()
    database_path = (root / DEFAULT_DATABASE_RELATIVE_PATH).resolve()
    return f"sqlite+pysqlite:///{database_path.as_posix()}"


def sanitized_database_label(database_url: str | URL) -> str:
    """Describe a database without returning credentials, host, or full URL."""
    url = make_url(database_url) if isinstance(database_url, str) else database_url
    dialect = url.get_backend_name()
    if dialect == "sqlite":
        database = url.database or ":memory:"
        label = ":memory:" if database == ":memory:" else Path(database).name
        return f"sqlite/{label}"
    return f"{dialect}/configured-database"


def create_monitoring_engine(
    database_url: str | None = None,
    *,
    project_root: Path | None = None,
    echo: bool = False,
) -> Engine:
    """Create a SQLAlchemy 2.x engine with SQLite safety settings."""
    resolved = resolve_database_url(database_url, project_root=project_root)
    url = make_url(resolved)
    if url.get_backend_name() == "sqlite" and url.database not in {None, ":memory:"}:
        Path(url.database).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
    connect_args = (
        {"check_same_thread": False}
        if url.get_backend_name() == "sqlite"
        else {}
    )
    engine = create_engine(
        resolved,
        echo=echo,
        future=True,
        connect_args=connect_args,
        hide_parameters=True,
    )
    if url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create explicit-transaction sessions that retain values after commit."""
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit one explicit unit of work or roll it back atomically."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
