"""Tests for the repository public-boundary scanner."""

from __future__ import annotations

from scripts.check_public_boundary import scan_path, scan_text


def test_public_boundary_rejects_private_artifact_paths() -> None:
    assert scan_path(".env")
    assert scan_path(".streamlit/secrets.toml")
    assert scan_path("data/raw/client_prices.csv")
    assert scan_path("monitoring.sqlite")
    assert scan_path("data/monitoring/portfolio_monitor.db")
    assert scan_path("data/monitoring/portfolio_monitor.db-wal")
    assert scan_path("data/monitoring/portfolio_monitor.db-shm")
    assert scan_path("data/monitoring/portfolio_monitor.db-journal")


def test_public_boundary_allows_public_placeholders() -> None:
    assert scan_path(".env.example") == []
    assert scan_path("data/raw/.gitkeep") == []
    assert scan_path("data/monitoring/.gitkeep") == []
    assert scan_path("tests/fixtures/synthetic_daily_prices.csv") == []


def test_public_boundary_detects_credentials_without_echoing_value() -> None:
    fake_github_token = "gh" + "p_" + "A" * 24
    findings = scan_text("example.txt", f"TOKEN={fake_github_token}\n")
    assert [finding.rule for finding in findings] == ["github_token"]
    assert fake_github_token not in findings[0].message


def test_public_boundary_detects_private_key_and_local_path() -> None:
    private_key_header = "-----BEGIN " + "PRIVATE KEY-----"
    local_path = "/" + "Users/example/private/research.csv"
    text = private_key_header + f"\n{local_path}\n"
    rules = {finding.rule for finding in scan_text("notes.txt", text)}
    assert rules == {"private_key", "local_user_path"}


def test_public_boundary_accepts_documented_environment_lookup() -> None:
    text = 'api_key = os.getenv("COINGECKO_API_KEY")\nCOINGECKO_API_KEY=\n'
    assert scan_text(".env.example", text) == []


def test_public_boundary_detects_database_url_credentials_without_echoing() -> None:
    secret_url = "postgresql://private-user:" + "private-pass@example/db"
    findings = scan_text("config.txt", secret_url)
    assert [finding.rule for finding in findings] == ["database_url_credentials"]
    assert secret_url not in findings[0].message
