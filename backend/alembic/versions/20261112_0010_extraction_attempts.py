"""Recall Jobs: count failed extractions, so a listing the model cannot parse is not retried forever

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-12 08:45:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "job_analyses",
        sa.Column("extraction_attempts", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("job_analyses", "extraction_attempts")
