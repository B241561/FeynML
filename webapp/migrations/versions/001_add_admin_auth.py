"""Create AdminProfile and OTPToken tables for admin authentication

Revision ID: 001_add_admin_auth
Revises: 
Create Date: 2024-06-04 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001_add_admin_auth'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create admin_profiles table
    op.create_table(
        'admin_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=128), nullable=False),
        sa.Column('email', sa.String(length=180), nullable=False),
        sa.Column('password_hash', sa.String(length=256), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_admin_profiles_username'), 'admin_profiles', ['username'], unique=False)
    op.create_index(op.f('ix_admin_profiles_email'), 'admin_profiles', ['email'], unique=False)

    # Create otp_tokens table
    op.create_table(
        'otp_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('admin_id', sa.Integer(), nullable=False),
        sa.Column('otp_code', sa.String(length=6), nullable=False),
        sa.Column('is_used', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['admin_id'], ['admin_profiles.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_otp_tokens_admin_id'), 'otp_tokens', ['admin_id'], unique=False)
    op.create_index(op.f('ix_otp_tokens_otp_code'), 'otp_tokens', ['otp_code'], unique=False)


def downgrade():
    # Drop otp_tokens table
    op.drop_index(op.f('ix_otp_tokens_otp_code'), table_name='otp_tokens')
    op.drop_index(op.f('ix_otp_tokens_admin_id'), table_name='otp_tokens')
    op.drop_table('otp_tokens')

    # Drop admin_profiles table
    op.drop_index(op.f('ix_admin_profiles_email'), table_name='admin_profiles')
    op.drop_index(op.f('ix_admin_profiles_username'), table_name='admin_profiles')
    op.drop_table('admin_profiles')
