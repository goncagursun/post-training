# First platform increment: submit and inspect tenant-owned jobs

The API persists jobs as QUEUED. The separate [dispatcher](dispatcher_quickstart.md) executes them.
FastAPI handles HTTP, Pydantic validates requests, SQLAlchemy maps records to
PostgreSQL tables, psycopg connects to PostgreSQL, and Uvicorn serves the app.
The API does not import Ray, PyTorch, or the training loop.

## 1. Install dependencies in tiny-gpt

From the repository root:

```bash
conda activate tiny-gpt
python -m pip install -e '.[platform,dev]'
```

## 2. Start PostgreSQL

This local workflow requires Docker Desktop with Docker Compose running.
Docker/PostgreSQL were not available in the development shell during implementation.
If you already have PostgreSQL, use an empty application database and set DATABASE_URL
accordingly; Compose is just a convenient way to provide it.

```bash
export POSTGRES_PASSWORD=$(python -c 'import secrets; print(secrets.token_hex(24))')
export DATABASE_URL="postgresql+psycopg://post_training:${POSTGRES_PASSWORD}@localhost:5432/post_training"
docker compose up -d --wait postgres
```

Keep this shell open. The generated password is not committed or printed. In another
terminal you must set the same DATABASE_URL. Store it in your local secret manager if
needed later. Changing POSTGRES_PASSWORD does not change the password inside an
already initialized PostgreSQL volume; reuse the original credentials on later starts.
The database uses a named Docker volume and survives `docker compose down`.
`docker compose down -v` deletes that data. This local Docker volume is independent
of RunPod's network volume; moving the control plane there requires a storage plan.

## 3. Initialize schema and onboard the demo tenants

```bash
python scripts/setup_platform.py init-db
python scripts/setup_platform.py onboard-demo
source .env.demo
```

`init-db` explicitly creates missing tables; this is not a migration system.
`onboard-demo` creates tenant-a/dataset_a and tenant-b/dataset_b and generates
independent, high-entropy keys. PostgreSQL stores only SHA-256 key hashes.
The original keys are written once to a new `.env.demo` file with owner-only permissions
for this local simulation; the script never prints them. That file is ignored by Git.
Sourcing it sets TENANT_A_API_KEY and TENANT_B_API_KEY in the current shell.
Do not share it. Normal customers would receive their own key separately.
Re-running onboarding refuses existing tenants/keys rather than silently rotating them.

Dataset records use `demo://` references resolving to separate two-example subsets
for the dispatcher. This registers ownership metadata; it does not upload real data.

## 4. Start the API

In the same shell, with DATABASE_URL set:

```bash
python -m uvicorn post_training.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs. Click Authorize and enter the appropriate tenant's key.
Keep the server bound to localhost for this development workflow.

Try POST /training/jobs with:

```json
{
  "dataset_id": "dataset_a",
  "base_model_id": "smollm2-135m",
  "learning_rate": 0.001,
  "epochs": 1,
  "lora_rank": 8
}
```

A successful response is HTTP 202 with `id`, `status: "QUEUED"`, timestamp, and the
resolved customer configuration. Copy `id` into GET /training/jobs/{job_id}.
The record is committed before the submission response is returned.

## 5. Demonstrate ownership and persistence

1. Authorize as tenant A, submit dataset_a, and read its job: 202 then 200.
2. Authorize as tenant B and request A's job: 404.
3. As B, submit dataset_a: 404. Submit dataset_b: 202.
4. Remove authorization or use an invalid/revoked key: 401.
5. Supply an unsupported model, rank, or tenant_id/output path: 422.
6. Restart Uvicorn and retrieve the same job: its recorded state persists in PostgreSQL.

Returning 404 for other tenants' resources avoids revealing their existence.
Keys are looked up on every request; setting revoked_at in the admin database revokes
access immediately. There is no public key-management endpoint in this increment.

## Verification

```bash
python -m pytest tests/test_platform_api.py -q
```

By default tests use temporary SQLite files with foreign keys enabled. They exercise
API authentication, revocation, isolation, validation, and persistence across app
instances. This does not validate PostgreSQL connectivity or locking.
To run the same suite on PostgreSQL, set TEST_DATABASE_URL to a separate, empty test
database. The suite refuses a nonempty database and creates/drops its own tables.
Never use the application database for tests.

## Boundaries of this increment

The dispatcher is now available separately. There are no quotas, idempotency keys,
cancellation, retries, artifact downloads, or dedicated result endpoint yet. Repeating a valid POST creates
a new job. The existing direct/Ray training CLIs remain independent. Schema changes
will need migrations when we evolve beyond initial tables. Tenant isolation is
application/database ownership enforcement for managed code, not process isolation
for arbitrary customer Python.
