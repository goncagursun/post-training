import os

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, event, select

from post_training.api import create_app
from post_training.auth import issue_api_key
from post_training.database import APIKey, Base, DatasetRecord, Job, Tenant, make_session_factory, utcnow


@pytest.fixture
def platform(tmp_path):
    # Set TEST_DATABASE_URL to exercise the same suite against a dedicated EMPTY
    # PostgreSQL test database. Never point this at the application database.
    url = os.environ.get('TEST_DATABASE_URL', f'sqlite:///{tmp_path}/platform.db')
    engine = create_engine(url)
    if engine.dialect.name == 'sqlite':
        @event.listens_for(engine, 'connect')
        def foreign_keys(connection, _):
            connection.execute('PRAGMA foreign_keys=ON')
    if __import__('sqlalchemy').inspect(engine).get_table_names():
        raise RuntimeError('Test database must be empty')
    Base.metadata.create_all(engine)
    sessions = make_session_factory(engine)
    keys = {}
    with sessions.begin() as session:
        for letter in ('a', 'b'):
            session.add(Tenant(id=letter))
        session.flush()
        for letter in ('a', 'b'):
            session.add(DatasetRecord(id=f'dataset_{letter}', tenant_id=letter, uri=f'demo://{letter}'))
            keys[letter] = issue_api_key(session, letter)
    try:
        with TestClient(create_app(sessions)) as client:
            yield client, sessions, keys
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def headers(keys, tenant='a'):
    return {'X-API-Key': keys[tenant]}


def body(**overrides):
    return {'dataset_id': 'dataset_a', 'base_model_id': 'smollm2-135m', **overrides}


def test_submission_ownership_and_persistence(platform):
    client, sessions, keys = platform
    response = client.post('/training/jobs', json=body(), headers=headers(keys))
    assert response.status_code == 202
    job = response.json()
    assert job['status'] == 'QUEUED'
    assert job['config']['lora_rank'] == 8
    assert client.get(f"/training/jobs/{job['id']}", headers=headers(keys)).status_code == 200
    assert client.get(f"/training/jobs/{job['id']}", headers=headers(keys, 'b')).status_code == 404
    assert client.post('/training/jobs', json=body(), headers=headers(keys, 'b')).status_code == 404
    b = client.post('/training/jobs', json=body(dataset_id='dataset_b'), headers=headers(keys, 'b'))
    assert b.status_code == 202
    # New application instance reads the same durable database.
    with TestClient(create_app(sessions)) as restarted:
        assert restarted.get(f"/training/jobs/{job['id']}", headers=headers(keys)).json() == job
    with sessions() as session:
        assert len(session.scalars(select(Job)).all()) == 2
        hashes = session.scalars(select(APIKey.key_hash)).all()
        assert all(key not in hashes for key in keys.values())
        assert all(len(value) == 64 for value in hashes)


def test_authentication_and_revocation(platform):
    client, sessions, keys = platform
    for h in ({}, {'X-API-Key': 'invalid'}):
        assert client.post('/training/jobs', json=body(), headers=h).status_code == 401
        assert client.get('/training/jobs/unknown', headers=h).status_code == 401
    with sessions.begin() as session:
        key = session.scalar(select(APIKey).where(APIKey.tenant_id == 'a'))
        key.revoked_at = utcnow()
    assert client.post('/training/jobs', json=body(), headers=headers(keys)).status_code == 401


@pytest.mark.parametrize('overrides', [
    {'base_model_id': 'unsupported'}, {'epochs': 0}, {'epochs': 11},
    {'epochs': '1'}, {'learning_rate': 0}, {'learning_rate': 0.1},
    {'lora_rank': 7}, {'tenant_id': 'b'}, {'run_dir': '/tmp/escape'},
])
def test_invalid_requests_do_not_create_jobs(platform, overrides):
    client, sessions, keys = platform
    assert client.post('/training/jobs', json=body(**overrides), headers=headers(keys)).status_code == 422
    with sessions() as session:
        assert session.scalar(select(Job)) is None
