"""Add Batch 4 origin-safe forecast provenance fields.

Revision ID: 0003_batch4_forecast
Revises: 0002_batch3_valuation
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0003_batch4_forecast"
down_revision: str | None = "0002_batch3_valuation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "daily_risk_forecasts",
        sa.Column("portfolio_definition", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "daily_risk_forecasts",
        sa.Column("input_max_date", sa.Date(), nullable=True),
    )
    op.add_column(
        "daily_risk_forecasts",
        sa.Column("input_data_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "daily_risk_forecasts",
        sa.Column(
            "forecast_metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("daily_risk_forecasts", "forecast_metadata")
    op.drop_column("daily_risk_forecasts", "input_data_hash")
    op.drop_column("daily_risk_forecasts", "input_max_date")
    op.drop_column("daily_risk_forecasts", "portfolio_definition")
