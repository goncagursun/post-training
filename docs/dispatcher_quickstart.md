# Dispatch queued jobs through Ray

This increment connects the existing API jobs to the existing Ray SFT worker.
The API and dispatcher run as separate processes on this machine. PostgreSQL
stores the queue and lifecycle. Ray executes one task at a time.

## 1. Add the execution table, preserving your existing jobs

Stop Uvicorn with Ctrl+C. In the shell where DATABASE_URL is set:

```bash
python scripts/setup_platform.py init-db
```

This creates the new job_executions table without resetting tenants, keys, or jobs.
Do not rerun onboard-demo. Existing demo dataset registrations now resolve to
different two-example subsets: tenant A uses the first two QA examples, B the last two.
They remain mechanics datasets, not a quality evaluation workload.

Restart the API:

```bash
python -m uvicorn post_training.api:create_app --factory --host 127.0.0.1 --port 8000
```

## 2. Start the dispatcher in a second terminal

Activate tiny-gpt and set the SAME DATABASE_URL used by the API. This variable is
not carried automatically into a new terminal. Do not generate a new database
password; use the existing connection credentials. The dispatcher needs DATABASE_URL,
not customer API keys. From the repository root:

```bash
python scripts/run_dispatcher.py --device cpu --data-root runs/platform --drain
```

`--drain` exits once the queue is empty. Omit it to poll every two seconds for more work.
This command will process all existing QUEUED jobs, not only the last two submissions.
Run in the foreground and let it complete. The API remains responsive in its terminal.

For RunPod, after the code and database connectivity are available there:

```bash
python scripts/run_dispatcher.py --device cuda --data-root /workspace/post-training-data --drain
```

This starts local Ray on the dispatcher's machine. It does not connect a Mac driver
to RunPod or transfer files across machines. Moving execution to RunPod is a separate
setup step. CUDA mode explicitly requires CUDA/BF16, not an Apple GPU.

## 3. Check status through the API

GET /training/jobs/{job_id} now includes an execution object:

- QUEUED: execution is null.
- CLAIMED: configuration, dataset reference, and output path have been reserved.
- RUNNING: Ray accepted the task; dispatched_at and ray_task_id are recorded.
- SUCCEEDED: execution.result contains adapter_dir, global_step, training_loss.
- FAILED: execution.error contains a safe failure category; finished_at is recorded.

RUNNING/dispatched_at means submitted to Ray, not a precise first GPU-kernel timestamp.
The result is metadata; the API does not yet serve adapter downloads. Tenant checks
still apply to the whole job response. Full task errors appear in the Ray worker output;
only exception types are persisted for customers to avoid leaking internal details.

Artifacts are saved under:

```text
<data-root>/customers/<tenant>/jobs/<job>/attempts/001/
  config.json
  checkpoints/
  adapter/
  result.json
  metrics.json
```

The database execution record snapshots the selected dataset URI and full runtime config.
The direct training CLIs retain the original four-example default when no dataset is supplied.

## How the claim works

A PostgreSQL transaction-level advisory lock serializes the single-slot capacity check.
While holding it, the dispatcher checks for CLAIMED/RUNNING work, then selects the oldest
QUEUED job using FOR UPDATE SKIP LOCKED and records its claim. Training runs outside
that transaction. Another dispatcher cannot reserve a second job while the slot is active.
The slot is scoped to this application database; unrelated CLI training is not coordinated.

If dispatch or the Ray task fails normally, the job becomes FAILED and the next job can run.
If the dispatcher crashes or loses database access, the reserved job blocks further dispatch.
The next dispatcher refuses to start duplicate work. Inspect execution and confirm the old
worker has stopped before manually repairing state. Automatic retries, leases, reconciliation,
cancellation, and checkpoint resume are deferred; this is not crash-safe exactly-once execution.
Do not manually reset an active job to QUEUED while its task might still be running.

## Tests

```bash
python -m pytest tests -q
```

PostgreSQL claim/concurrency tests require TEST_DATABASE_URL pointing at a dedicated EMPTY
test database. The suite creates/drops its tables. Never use your application database.
Set RUN_RAY_SMOKE=1 as well to opt into two actual CPU training runs and adapter-save checks.
Without PostgreSQL, locking tests are skipped; API tests still run using temporary SQLite.
