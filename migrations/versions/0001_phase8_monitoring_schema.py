"""Create the Phase 8 portfolio-monitoring persistence schema.

Revision ID: 0001_phase8_monitoring
Revises:
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0001_phase8_monitoring"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("experiment_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("base_currency", sa.String(16), nullable=False),
        sa.Column("initial_capital", sa.Float(), nullable=False),
        sa.Column("benchmark_symbol", sa.String(64), nullable=True),
        sa.Column("training_start", sa.Date(), nullable=True),
        sa.Column("training_end", sa.Date(), nullable=True),
        sa.Column("optimization_as_of", sa.Date(), nullable=True),
        sa.Column("launch_date", sa.Date(), nullable=True),
        sa.Column("historical_evaluation_end", sa.Date(), nullable=True),
        sa.Column("live_tracking_end", sa.Date(), nullable=True),
        sa.Column("source_metadata", sa.JSON(), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "mode IN ('historical_oos', 'live_forward', 'hybrid')",
            name=op.f("ck_experiments_mode_values"),
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'backfilling', 'active', 'completed', "
            "'failed', 'archived')",
            name=op.f("ck_experiments_status_values"),
        ),
        sa.CheckConstraint(
            "initial_capital > 0",
            name=op.f("ck_experiments_positive_initial_capital"),
        ),
        sa.PrimaryKeyConstraint("experiment_id", name="pk_experiments"),
    )
    op.create_index("ix_experiments_name", "experiments", ["name"])
    op.create_index("ix_experiments_status", "experiments", ["status"])
    op.create_index("ix_experiments_mode", "experiments", ["mode"])
    op.create_index("ix_experiments_launch_date", "experiments", ["launch_date"])
    op.create_index("ix_experiments_updated_at", "experiments", ["updated_at"])

    op.create_table(
        "optimization_snapshots",
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("experiment_id", sa.String(36), nullable=False),
        sa.Column("package_version", sa.String(64), nullable=False),
        sa.Column("code_version", sa.String(128), nullable=False),
        sa.Column("objective", sa.String(128), nullable=False),
        sa.Column("solver", sa.String(128), nullable=False),
        sa.Column("solver_status", sa.String(64), nullable=False),
        sa.Column("assumptions", sa.JSON(), nullable=False),
        sa.Column("constraints", sa.JSON(), nullable=False),
        sa.Column("launch_forecast", sa.JSON(), nullable=False),
        sa.Column("scenario_metadata", sa.JSON(), nullable=False),
        sa.Column("return_policy", sa.JSON(), nullable=False),
        sa.Column("loss_convention", sa.JSON(), nullable=False),
        sa.Column("residual_validation", sa.JSON(), nullable=False),
        sa.Column("source_data_hash", sa.String(64), nullable=False),
        sa.Column("assumption_recipe_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "length(source_data_hash) = 64",
            name=op.f("ck_optimization_snapshots_source_hash_length"),
        ),
        sa.CheckConstraint(
            "length(assumption_recipe_hash) = 64",
            name=op.f("ck_optimization_snapshots_recipe_hash_length"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_optimization_snapshots_experiment_id_experiments",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("snapshot_id", name="pk_optimization_snapshots"),
        sa.UniqueConstraint(
            "experiment_id", name="uq_optimization_snapshots_one_per_experiment"
        ),
    )

    op.create_table(
        "snapshot_allocations",
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("asset_type", sa.String(32), nullable=False),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.Column("launch_price", sa.Float(), nullable=True),
        sa.Column("initial_value", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("is_cash", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["optimization_snapshots.snapshot_id"],
            name="fk_snapshot_allocations_snapshot_id_optimization_snapshots",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "snapshot_id", "asset", name="pk_snapshot_allocations"
        ),
    )

    op.create_table(
        "price_observations",
        sa.Column("observation_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("observation_date", sa.Date(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("quote_currency", sa.String(16), nullable=False),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data_status", sa.String(32), nullable=False),
        sa.CheckConstraint(
            "data_status IN ('complete', 'incomplete', 'corrected', 'rejected')",
            name=op.f("ck_price_observations_data_status_values"),
        ),
        sa.CheckConstraint(
            "price > 0",
            name=op.f("ck_price_observations_positive_price"),
        ),
        sa.PrimaryKeyConstraint("observation_id", name="pk_price_observations"),
        sa.UniqueConstraint(
            "symbol",
            "observation_date",
            "quote_currency",
            "source",
            name="uq_price_observations_natural_key",
        ),
    )
    op.create_index(
        "ix_price_observations_date", "price_observations", ["observation_date"]
    )
    op.create_index(
        "ix_price_observations_symbol_date",
        "price_observations",
        ["symbol", "observation_date"],
    )

    op.create_table(
        "daily_portfolio_states",
        sa.Column("experiment_id", sa.String(36), nullable=False),
        sa.Column("state_date", sa.Date(), nullable=False),
        sa.Column("nav", sa.Float(), nullable=True),
        sa.Column("cash_value", sa.Float(), nullable=True),
        sa.Column("daily_return", sa.Float(), nullable=True),
        sa.Column("cumulative_return", sa.Float(), nullable=True),
        sa.Column("realized_volatility", sa.Float(), nullable=True),
        sa.Column("running_peak", sa.Float(), nullable=True),
        sa.Column("drawdown", sa.Float(), nullable=True),
        sa.Column("maximum_drawdown", sa.Float(), nullable=True),
        sa.Column("benchmark_nav", sa.Float(), nullable=True),
        sa.Column("benchmark_return", sa.Float(), nullable=True),
        sa.Column("data_quality_status", sa.String(32), nullable=False),
        sa.Column("calculation_version", sa.String(64), nullable=False),
        sa.Column("finalized", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "data_quality_status IN ('complete', 'incomplete', 'missing', "
            "'partial', 'corrected')",
            name=op.f("ck_daily_portfolio_states_quality_status_values"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_daily_portfolio_states_experiment_id_experiments",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "experiment_id", "state_date", name="pk_daily_portfolio_states"
        ),
    )
    op.create_index(
        "ix_daily_portfolio_states_experiment_date",
        "daily_portfolio_states",
        ["experiment_id", "state_date"],
    )
    op.create_index(
        "ix_daily_portfolio_states_quality",
        "daily_portfolio_states",
        ["data_quality_status"],
    )

    op.create_table(
        "daily_asset_states",
        sa.Column("experiment_id", sa.String(36), nullable=False),
        sa.Column("state_date", sa.Date(), nullable=False),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("market_value", sa.Float(), nullable=True),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.Column("current_weight", sa.Float(), nullable=True),
        sa.Column("drift_percentage_points", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_daily_asset_states_experiment_id_experiments",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "experiment_id", "state_date", "asset", name="pk_daily_asset_states"
        ),
    )

    op.create_table(
        "daily_risk_forecasts",
        sa.Column("forecast_id", sa.String(36), nullable=False),
        sa.Column("experiment_id", sa.String(36), nullable=False),
        sa.Column("origin_date", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("evaluation_mode", sa.String(32), nullable=False),
        sa.Column("estimation_window", sa.Integer(), nullable=False),
        sa.Column("var_method", sa.String(64), nullable=False),
        sa.Column("cvar_method", sa.String(64), nullable=False),
        sa.Column("confidence_level", sa.Float(), nullable=False),
        sa.Column("horizon_construction", sa.String(64), nullable=False),
        sa.Column("convention_version", sa.String(64), nullable=False),
        sa.Column("forecast_var", sa.Float(), nullable=True),
        sa.Column("forecast_cvar", sa.Float(), nullable=True),
        sa.Column("forecast_volatility", sa.Float(), nullable=True),
        sa.Column("realized_horizon_loss", sa.Float(), nullable=True),
        sa.Column("var_breach", sa.Boolean(), nullable=True),
        sa.Column("evaluation_status", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "confidence_level > 0 AND confidence_level < 1",
            name=op.f("ck_daily_risk_forecasts_confidence_range"),
        ),
        sa.CheckConstraint(
            "evaluation_status IN ('pending', 'evaluated', 'insufficient_window')",
            name=op.f("ck_daily_risk_forecasts_evaluation_status_values"),
        ),
        sa.CheckConstraint(
            "horizon_days > 0",
            name=op.f("ck_daily_risk_forecasts_positive_horizon"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_daily_risk_forecasts_experiment_id_experiments",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("forecast_id", name="pk_daily_risk_forecasts"),
        sa.UniqueConstraint(
            "experiment_id",
            "origin_date",
            "target_date",
            "horizon_days",
            "evaluation_mode",
            "var_method",
            "cvar_method",
            "confidence_level",
            "model_version",
            name="uq_daily_risk_forecasts_natural_key",
        ),
    )
    op.create_index(
        "ix_daily_risk_forecasts_experiment_origin",
        "daily_risk_forecasts",
        ["experiment_id", "origin_date"],
    )
    op.create_index(
        "ix_daily_risk_forecasts_target_status",
        "daily_risk_forecasts",
        ["target_date", "evaluation_status"],
    )

    op.create_table(
        "monitoring_runs",
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("experiment_id", sa.String(36), nullable=False),
        sa.Column("run_type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_cutoff", sa.Date(), nullable=True),
        sa.Column("actual_cutoff", sa.Date(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("inserted_count", sa.Integer(), nullable=False),
        sa.Column("updated_count", sa.Integer(), nullable=False),
        sa.Column("skipped_count", sa.Integer(), nullable=False),
        sa.Column("warning_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name=op.f("ck_monitoring_runs_status_values"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_monitoring_runs_experiment_id_experiments",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("run_id", name="pk_monitoring_runs"),
    )
    op.create_index(
        "ix_monitoring_runs_experiment_started",
        "monitoring_runs",
        ["experiment_id", "started_at"],
    )

    op.create_table(
        "experiment_events",
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column("experiment_id", sa.String(36), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("event_type", sa.String(128), nullable=False),
        sa.Column("event_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_experiment_events_experiment_id_experiments",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("event_id", name="pk_experiment_events"),
    )
    op.create_index(
        "ix_experiment_events_experiment_created",
        "experiment_events",
        ["experiment_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_experiment_events_experiment_created", table_name="experiment_events"
    )
    op.drop_table("experiment_events")
    op.drop_index(
        "ix_monitoring_runs_experiment_started", table_name="monitoring_runs"
    )
    op.drop_table("monitoring_runs")
    op.drop_index(
        "ix_daily_risk_forecasts_target_status", table_name="daily_risk_forecasts"
    )
    op.drop_index(
        "ix_daily_risk_forecasts_experiment_origin",
        table_name="daily_risk_forecasts",
    )
    op.drop_table("daily_risk_forecasts")
    op.drop_table("daily_asset_states")
    op.drop_index(
        "ix_daily_portfolio_states_quality", table_name="daily_portfolio_states"
    )
    op.drop_index(
        "ix_daily_portfolio_states_experiment_date",
        table_name="daily_portfolio_states",
    )
    op.drop_table("daily_portfolio_states")
    op.drop_index(
        "ix_price_observations_symbol_date", table_name="price_observations"
    )
    op.drop_index("ix_price_observations_date", table_name="price_observations")
    op.drop_table("price_observations")
    op.drop_table("snapshot_allocations")
    op.drop_table("optimization_snapshots")
    op.drop_index("ix_experiments_updated_at", table_name="experiments")
    op.drop_index("ix_experiments_launch_date", table_name="experiments")
    op.drop_index("ix_experiments_mode", table_name="experiments")
    op.drop_index("ix_experiments_status", table_name="experiments")
    op.drop_index("ix_experiments_name", table_name="experiments")
    op.drop_table("experiments")
