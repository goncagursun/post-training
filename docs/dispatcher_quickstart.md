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
worker has stopped before manually repairing state. Automatic retries, leases, reconciliation, and
cancellation are deferred; explicit checkpoint recovery is available below; this is not crash-safe exactly-once execution.
Do not manually reset an active job to QUEUED while its task might still be running.

## Tests

```bash
python -m pytest tests -q
```

PostgreSQL claim/concurrency tests require TEST_DATABASE_URL pointing at a dedicated EMPTY
test database. The suite creates/drops its tables. Never use your application database.
Set RUN_RAY_SMOKE=1 as well to opt into two actual CPU training runs and adapter-save checks.
Without PostgreSQL, locking tests are skipped; API tests still run using temporary SQLite.

## Basic instrumentation

New runs write `telemetry.json` (summary) and `events.jsonl` (timestamped phases,
Trainer metrics, and optional GPU samples). Existing runs are not backfilled.
GET job status returns the same summary under `execution.result.telemetry`, plus
`execution.result.platform_metrics` calculated by the dispatcher. No schema change
or database initialization is required. Restart the API and dispatcher to load code.
The original metrics.json and validation.json remain available.

Recorded measurements:

- Run start/finish and PID; tenant/job/attempt and Ray task IDs for dispatched jobs.
- Wall time for tokenizer loading, dataset preparation, model/adapters, Trainer setup,
  training, validation, checkpoint writes, and adapter export. Checkpoint writes are
  NESTED inside training time; do not sum these as independent phases.
- Observed non-padding input tokens and supervised next-token targets, optimizer steps,
  training tokens/sec and mean seconds/update. The denominator is trainer.train wall
  time including checkpointing. Token counts cover actual training microbatches only,
  including repeats across epochs; validation tokens are excluded. This is single-worker
  instrumentation, not aggregated distributed training telemetry.
- Process-lifetime peak RSS, in bytes (not total host RAM or just training memory).
- CUDA peak allocated/reserved memory across the run, or null on CPU. PyTorch's own
  Trainer memory tracker is disabled to prevent its resets corrupting these counters.
- Optional nvidia-smi GPU utilization and device memory samples roughly every second,
  if the CUDA GPU UUID and nvidia-smi are available. These are whole-device metrics,
  not process/tenant utilization, and utilization.gpu is not SM occupancy. Raw samples
  appear in events.jsonl; the summary reports sample count, mean utilization, and maximum
  sampled memory. Missing samples are explicitly unavailable, never zero utilization.

Run timing starts when run_sft is entered in the worker. It excludes worker process
startup/imports before that entry. `ray_submission_to_run_start_seconds` includes that
startup and scheduling delay, using UTC wall clocks (same-node assumption). CUDA phase
boundaries synchronize to avoid timing only asynchronous kernel launch. Sampling and
logging add overhead, so these are operational measurements, not kernel benchmarks.

Platform metrics distinguish submission-to-claim queue time, claim-to-completion slot
reservation, and submission-to-completion latency. They are finalized when the dispatcher
records completion. The worker's summary is persisted to disk; platform timing is stored
in PostgreSQL, not retroactively added to the worker's result.json.

Optionally supply your configured GPU allocation price (USD/hour):

```bash
python scripts/run_dispatcher.py --device cuda --data-root /workspace/post-training-data --gpu-hourly-rate-usd 0.50 --drain
```

0.50 is an illustrative input, not a verified RunPod price. Cost = reserved slot seconds
/ 3600 x configured rate, assuming the slot owns one allocated GPU. It includes dispatch
and Ray waiting time, not just training; it excludes pod idle time outside jobs and
storage/network charges. The rate is platform-controlled, not customer-submitted.
CPU GPU-seconds/cost are zero. GPU cost is null if no rate is supplied.

On ordinary training failures, partial telemetry and a safe exception type remain in
telemetry.json and are exposed by the dispatcher when available. Worker hard kills or
OOM process termination can leave events without a finalized summary. Metrics are not
streamed into the API during a run, and this does not add automatic recovery or billing.

## Explicit checkpoint recovery

Recovery is an administrator action for this single-machine MVP. It is not an
unbounded automatic retry policy. New checkpoints contain a complete.json marker
with file checksums and a fingerprint of the training configuration, exact training/
validation examples, and resolved base-model commit. Training restores adapters,
optimizer, scheduler, random state, progress, and Trainer data skipping.

### Enable the new metadata table

Stop the API/dispatcher, then run in the shell with DATABASE_URL set:

```bash
python scripts/setup_platform.py init-db
```

This adds job_recovery without resetting existing tables or jobs. Restart the API.
Old runs remain readable; checkpoints saved BEFORE this update have no completion
marker and are intentionally not accepted by this recovery path. Use a new job for
this demonstration. Never manufacture a completion marker for old/partial state.

### Reproducible small CPU demonstration

Submit a fresh tenant-A job using dataset_a (the original TWO-example demo dataset),
smollm2-135m, four epochs, learning_rate 0.001, and lora_rank 8. This is eight optimizer
steps total, with a checkpoint at step four. Then run:

```bash
python scripts/run_dispatcher.py --device cpu --data-root runs/platform --drain --interrupt-after-checkpoint 4
```

This admin-only test option raises a controlled interruption AFTER a completed
checkpoint. The job becomes INTERRUPTED, its attempt-001 files stay intact, and the
dispatcher exits. The option applies to the next queued job (FIFO), so ensure no older
jobs precede the intended demonstration. It does not create a job itself.

Resume using the interrupted job's ID, omitting the interruption option:

```bash
python scripts/run_dispatcher.py --device cpu --data-root runs/platform --drain --resume-job JOB_ID
```

Recovery requires the old worker's OS lock to be free. It selects the latest valid
checkpoint (falls back past incomplete/corrupt newer ones), archives the old execution
in job_recovery.history, and requeues with the original config. The dispatcher runs
attempt 002 in a new directory, restoring step four and finishing at step eight.
The API exposes current execution plus recovery history. Subsequent recoveries use
003, 004, etc.; no attempt directory is overwritten. Normal FIFO still applies and
--drain will also execute other queued jobs.

### Real interruptions

Ctrl+C in the dispatcher requests a force stop through Ray, records INTERRUPTED,
and exits. A recovery command refuses to proceed while a worker still holds the job
lock; retry after it has exited. A hard driver/pod crash can leave CLAIMED/RUNNING in
the database. The same explicit --resume-job command handles it after the lock is
free and a completed checkpoint is available. Use the same persistent filesystem.

A per-job filesystem lock prevents two guarded workers from training simultaneously.
An attempt marker fences delayed old tasks: even if they start later, they reject a
superseded attempt before loading/training. Stale dispatcher updates are rejected by
checking the execution directory. This is a single-host/shared-POSIX-filesystem design,
not a distributed lease or an exactly-once guarantee across independent machines.
A Ray launch/DB failure still needs this explicit inspection/recovery path.

No valid checkpoint means recovery refuses rather than silently starting from scratch.
Dataset/config/device/precision/base-commit changes fail fingerprint validation. Use a
new job for a changed recipe. The same supported runtime/library versions should be
used; cross-version and CPU-to-GPU recovery have not been validated.

If a checkpoint already reached the training target, recovery reloads it and runs only
validation/export without another optimizer update. This attempt has zero observed
steps and null training_loss. If the old run already has result.json, recovery refuses
and asks for reconciliation rather than rerunning completed work; automatic completed-
result reconciliation is not yet implemented.

Telemetry records the source checkpoint and initial global step. Global step is total
training progress, while observed_optimizer_steps/tokens/throughput and training_loss
refer only to work in this attempt. Trainer's native aggregate progress display can
include pre-resume steps; use the explicit observed counters for throughput comparisons.
Attempt history preserves earlier costs; the latest attempt cost is not the whole job's
cumulative cost. Hard-crash attempts may have incomplete timing/cost information.

Recovery checks include byte-for-byte matching final CPU adapter weights against an
uninterrupted run, plus PostgreSQL/Ray interrupted-job execution, fencing, checksum
fallback, and attempt-history checks. CUDA recovery remains to be tested on RunPod.

## Customer cancellation

Restart the API and dispatcher to load this update. No database schema change or
new dependencies are needed. The API has POST /training/jobs/{job_id}/cancel,
authenticated with the same X-API-Key as submit/status. Use /docs to select the job
and execute the request; it has no JSON body.

- QUEUED -> CANCELLED immediately: no worker exists and the dispatcher will skip it.
- CLAIMED/RUNNING -> CANCEL_REQUESTED: the API acknowledges the request (HTTP 202).
- The dispatcher polls every 0.25 seconds while waiting on Ray. It calls ray.cancel
  with force=True, waits for the Ray task result to become terminal, and verifies the
  job-wide worker lock is free. It fences old/pending attempts before recording
  CANCELLED with execution.finished_at and final platform timing/cost metrics.
- Repeat requests preserve CANCEL_REQUESTED/CANCELLED. Already terminal outcomes
  (SUCCEEDED, FAILED, INTERRUPTED) remain unchanged. Another tenant gets 404 and
  invalid/missing credentials get 401, just as with job retrieval.

Try queued cancellation first: submit a job while the dispatcher is stopped, call
cancel, then GET its status. It should be CANCELLED with execution null. Starting
the dispatcher will not train it.

For running cancellation, submit a longer Bitext job and start the dispatcher in a
second terminal. While it is RUNNING, call the cancel endpoint in /docs. The first
response may say CANCEL_REQUESTED; poll GET until CANCELLED. The API must remain
running. The dispatcher continues to the next queued job after confirming the stop.

Completion race policy: a result already observed ready by the dispatcher is
processed normally; a concurrent cancel request does not overwrite that completed
outcome. If cancellation wins the polling decision, partial files may remain,
including a checkpoint or export written just before the stop. They are not returned
as a successful published adapter. Existing files are retained for inspection.
Force-stop cancellation does not promise a new checkpoint or finalized worker
telemetry; a killed process may leave only earlier complete checkpoints/events.
Cancelled jobs are not eligible for --resume-job; use a new job to train again.

This requires the dispatcher to remain alive. If it crashes or cannot confirm the
worker stopped, CANCEL_REQUESTED stays reserved and further admission is blocked
rather than claiming cancellation succeeded. Automatic cancellation reconciliation
after a dispatcher crash is a later increment. Do not manually reset such a job while
a worker might still be active. The same single-host filesystem-lock assumptions as
recovery apply; this is not cancellation of arbitrary external training processes.
