"""Create saved_visualizations table for custom report snapshots

Revision ID: 002_add_saved_visualizations
Revises: 001_add_admin_auth
Create Date: 2026-06-04 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_saved_visualizations'
down_revision = '001_add_admin_auth'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'saved_visualizations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_id', sa.Integer(), nullable=False),
        sa.Column('viz_id', sa.String(length=128), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('chart_data', sa.JSON(), nullable=True),
        sa.Column('chart_image', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['report_id'], ['reports.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('viz_id')
    )
    op.create_index(op.f('ix_saved_visualizations_report_id'), 'saved_visualizations', ['report_id'], unique=False)
    op.create_index(op.f('ix_saved_visualizations_viz_id'), 'saved_visualizations', ['viz_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_saved_visualizations_viz_id'), table_name='saved_visualizations')
    op.drop_index(op.f('ix_saved_visualizations_report_id'), table_name='saved_visualizations')
    op.drop_table('saved_visualizations')
