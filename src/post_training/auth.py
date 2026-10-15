"""High-entropy API keys are stored as hashes, never as plaintext."""
import hashlib
import secrets

from sqlalchemy import select
from post_training.database import APIKey


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def issue_api_key(session, tenant_id):
    raw_key = secrets.token_urlsafe(32)
    session.add(APIKey(tenant_id=tenant_id, key_hash=hash_api_key(raw_key)))
    return raw_key


def authenticate_key(session, raw_key):
    if not raw_key:
        return None
    return session.scalar(select(APIKey.tenant_id).where(
        APIKey.key_hash == hash_api_key(raw_key), APIKey.revoked_at.is_(None)))
