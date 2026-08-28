"""Add Batch 3 valuation provenance and drift fields.

Revision ID: 0002_batch3_valuation
Revises: 0001_phase8_monitoring
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_batch3_valuation"
down_revision: str | None = "0001_phase8_monitoring"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "daily_portfolio_states",
        sa.Column("base_100_nav", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_portfolio_states",
        sa.Column("total_drift", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_portfolio_states",
        sa.Column("return_interval_days", sa.Integer(), nullable=True),
    )
    op.add_column(
        "daily_portfolio_states",
        sa.Column(
            "quality_metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("daily_portfolio_states", "quality_metadata")
    op.drop_column("daily_portfolio_states", "return_interval_days")
    op.drop_column("daily_portfolio_states", "total_drift")
    op.drop_column("daily_portfolio_states", "base_100_nav")
