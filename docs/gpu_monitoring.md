# GPU performance and cost monitoring

Run the monitor independently of the API and dispatcher so that it observes time
between training jobs. No database schema change is needed. This implementation is
for the existing single-host, single-training-slot platform.

## Start a monitoring session

Install the project and optional platform dependencies in your active CUDA environment:

```bash
python -m pip install -e '.[platform,dev]'
```

Keep the existing CUDA PyTorch installation. Set DATABASE_URL to the same database
used by the API and dispatcher, then start the monitor in another terminal:

```bash
python scripts/monitor_gpus.py \
  --output-dir /workspace/post-training-data/monitoring/session-001 \
  --with-database
```

The output directory must be new. Stop with Ctrl+C or SIGTERM. Add `--duration 60`
for a bounded session; `--interval 1` controls the polling interval in seconds.
The monitor should stay running while the dispatcher waits for work.

Supply `--gpu-hourly-rate-usd` with your actual configured allocation price to enable
cost estimates. The rate is USD per GPU-hour and applies equally to every visible GPU.
It is never fetched from a provider and has no default dollar value.

Without `--with-database`, GPU measurements still work; allocation is unattributed.
On a multi-GPU host, use `--gpu-uuid GPU-...` to identify the GPU assigned to the
platform slot. This flag labels attribution; it does not constrain Ray placement.
You must ensure the worker actually runs on that GPU. Without a selected GPU,
multi-GPU measurements remain unattributed. The one-GPU A40 setup selects its UUID
automatically. All visible GPUs are monitored regardless of this selection.

## Read the results

`summary.json` is replaced atomically after each poll and on graceful shutdown:

- `gpus`: per-device utilization, used/total VRAM, temperature, and power, including
  valid observation counts, sample means, and maxima.
- `tenants`: reserved GPU seconds, estimated allocation cost, and whole-device
  observations taken while each tenant held the platform slot.
- `attempts`: the same measurements separated by attempt directory, preserving
  recovery attempts as separate records.
- `observed_allocation_gpu_seconds`: observed capacity time summed across GPUs.
- `reserved_gpu_seconds`: time associated with the platform's CUDA slot.
- `unreserved_gpu_seconds`: selected-GPU time with a successful database observation
  showing no CUDA reservation. This is scheduler idle capacity; unrelated processes
  may still be using the device.
- `unattributed_gpu_seconds`: time without a known slot association, including
  database failures, monitoring without a database, and unselected GPUs.
- `estimated_cost_usd` and `estimated_unreserved_cost_usd`: observed allocation time
  and unreserved time multiplied by the configured rate. Null means no rate supplied.

`gpu_samples.jsonl` records timestamped measurements, slot owner (tenant/job/attempt),
and safe query failure categories. These are administrator files containing multiple
tenants; do not expose them through a tenant API or store them in Git. Raw samples
grow with session duration; rotate sessions according to your retention policy.

Reservation state is sampled and held until the next poll. Timing near transitions
is approximate, normally within one polling interval plus query latency. Query
failures are explicit; gaps after a failed GPU query are excluded from observed time
and cost. Check query errors and coverage before treating any estimate as complete.
Monitoring starts with the first successful observation and excludes startup,
shutdown after a hard kill, and any time outside this monitor session. A hard kill
leaves the last summary and raw events available but cannot finalize the session.
The monitor does not connect to cloud billing or determine whether a GPU is billed.
It assumes each visible device represents allocated capacity during observed time.
MIG/vGPU/shared-host billing and unequal per-GPU rates require a different inventory model.

## Job throughput and useful work

Workers still write `telemetry.json` and `events.jsonl` in their attempt directory.
They now include GPU UUID, total VRAM, temperature, power, observed training samples,
and samples/second alongside input/supervised tokens/second, loss, phase timings,
and CUDA peak allocated/reserved memory. Existing job-status responses include the
final telemetry under `execution.result.telemetry` when the dispatcher persists it.
GPU samples are taken throughout the run, while throughput measures training only.
No live metrics API or dashboard is introduced.

Compare tenant/attempt allocation cost with completed training work and held-out
accuracy (`scripts/evaluate_intents.py`). Include failed and recovered attempts when
calculating cost per successful job. Platform allocation cost already includes job
reservations: do not add job costs to it. Dispatcher cost and sampled monitor cost
have different boundaries and need not match exactly.

The zero-utilization sample fraction is a sampled inactivity indicator, not an exact
idle duration or a 50% alert policy. Whole-device utilization is not SM occupancy,
per-process utilization, or useful-model-work percentage. Temperature alone does not
prove throttling, and free VRAM alone does not establish that a bigger batch is faster.
No alerts or automatic batch-size changes are applied.

Tenant device activity is contextual attribution under the single-slot design,
not hardware isolation. Unrelated processes contaminate it. Temperature, power, and
VRAM are whole-device readings; PyTorch peak memory is specific to the worker allocator.
Database outages do not stop GPU sampling; they make tenant association unknown.
Force-killed workers may lack a finalized telemetry summary; the independent monitor
can retain observations if it survives the worker.

NVIDIA documents the query fields and utilization semantics in its
[nvidia-smi reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Verification

```bash
python -m pytest tests -q
python scripts/monitor_gpus.py --output-dir /tmp/gpu-check-001 --duration 5
```

Tests cover tenant transitions, idle allocation cost, missing readings, query failures,
and multi-GPU attribution. PostgreSQL and real Ray/training tests remain opt-in as
described in the dispatcher quickstart. A device sampling check alone does not validate
an end-to-end GPU training job or tenant accounting against provider invoices.
