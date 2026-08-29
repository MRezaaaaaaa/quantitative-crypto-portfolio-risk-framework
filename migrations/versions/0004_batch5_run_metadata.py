"""Add Batch 5 monitoring-run provenance metadata.

Revision ID: 0004_batch5_run_metadata
Revises: 0003_batch4_forecast
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0004_batch5_run_metadata"
down_revision: str | None = "0003_batch4_forecast"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "monitoring_runs",
        sa.Column(
            "run_metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("monitoring_runs", "run_metadata")
