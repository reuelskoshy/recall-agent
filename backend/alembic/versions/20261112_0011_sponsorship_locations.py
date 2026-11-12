"""Recall Jobs: places the candidate accepts only with visa sponsorship

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-12 09:05:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "candidate_profile",
        sa.Column(
            "sponsorship_locations",
            postgresql.ARRAY(sa.String(length=200)),
            server_default="{}",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("candidate_profile", "sponsorship_locations")
