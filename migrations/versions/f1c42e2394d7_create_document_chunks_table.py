"""create document chunks table

Revision ID: f1c42e2394d7
Revises: 
Create Date: 2025-04-05 11:05:37.048358

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f1c42e2394d7'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('document_chunks', 'embedding',
               existing_type=postgresql.ARRAY(sa.REAL()),
               type_=postgresql.ARRAY(sa.FLOAT(precision=6)),
               existing_nullable=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('document_chunks', 'embedding',
               existing_type=postgresql.ARRAY(sa.FLOAT(precision=6)),
               type_=postgresql.ARRAY(sa.REAL()),
               existing_nullable=True)
    # ### end Alembic commands ###
