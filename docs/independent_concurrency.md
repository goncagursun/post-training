# Scenario 2: independent concurrent LoRA training

The FIFO API/dispatcher is scenario 1. This separate benchmark submits two Ray
workers together on one physical GPU. Each process loads its own frozen base model,
adapter, optimizer, and dataset. No base weights or optimizer state are shared in
GPU memory. Fractional Ray GPU reservations admit both workers; they do not impose
memory limits, guarantee equal compute, or enable CUDA MPS.

The existing FIFO queue, job records, and dispatcher are not used or changed by this
runner. Run it when the FIFO dispatcher and other GPU workloads are stopped. Both
scenarios are single-host experiments. Running both launchers together is not a
supported admission-control scheme.

## Prepare a specification

Use the same activated project environment and HF_HOME as scenario 1. Create a JSON
file outside Git with exactly two tenants. Paths can be absolute or relative to the
specification file. For each tenant, point config_path to its original FIFO attempt's
config.json and baseline_adapter_dir to that attempt's adapter directory:

```json
{
  "tenants": [
    {
      "tenant_id": "tenant-a",
      "config_path": "/path/to/fifo/tenant-a/attempts/001/config.json",
      "manifest_path": "/path/to/bitext/tenant-a/manifest.json",
      "baseline_adapter_dir": "/path/to/fifo/tenant-a/attempts/001/adapter"
    },
    {
      "tenant_id": "tenant-b",
      "config_path": "/path/to/fifo/tenant-b/attempts/001/config.json",
      "manifest_path": "/path/to/bitext/tenant-b/manifest.json",
      "baseline_adapter_dir": "/path/to/fifo/tenant-b/attempts/001/adapter"
    }
  ]
}
```

baseline_adapter_dir is optional, but supplying it verifies the complete training
config and saved dataset manifest match the baseline before launching either worker.
Dataset ownership and checksums are validated. CUDA configs are required. The runner
does not tune batch size, learning rate, seeds, epochs, or checkpoint cadence.

```bash
python scripts/run_independent_concurrency.py \
  --spec /workspace/post-training-data/experiments/independent-concurrency-001.spec.json \
  --output-dir /workspace/post-training-data/experiments/independent-concurrency-001 \
  --gpu-hourly-rate-usd 0.59
```

Use a new output directory for every invocation. The configured $0.59/GPU-hour rate
is an input for our experiment, not a provider lookup. On a multi-GPU host select
`--gpu-uuid GPU-...`; the runner exposes only that device to its local Ray runtime.
At least two Ray CPU resources are required. Each worker requests one CPU and 0.5
logical GPU. Existing thread settings are inherited; CPU contention can affect the
result. The same model revision should remain available for all scenarios; mutable
`main` revisions are still a reproducibility limitation.

## Outputs and measurement boundaries

- report.json: workload configuration, baseline links, per-tenant worker results,
  task submission/completion times, aggregate throughput, cost, and concurrency checks.
- customers/<tenant>/jobs/<id>/attempts/001/: normal worker artifacts, including
  adapter/, config.json, dataset.json, telemetry.json, events.jsonl, and checkpoints/.
- monitoring/: platform GPU samples and summary across the full benchmark session.
- monitor.log: diagnostics from the independent monitor; redirect runner output to
  a separate file to retain Ray/Trainer diagnostics.

A successful report requires distinct worker PIDs, the same GPU UUID, and a positive
overlap between the workers' recorded training phases. This proves overlapping
training-loop lifetimes, not simultaneous SM execution or improved performance.
Training-phase overlap includes checkpoint time. Per-tenant training throughput and
PyTorch peak memory remain worker measurements. The worker's nvidia-smi utilization,
memory, power, and temperature include both processes and cannot be added together
or interpreted as per-tenant GPU activity.

Aggregate input tokens/s and samples/s use the total completed work divided by the
wall time from first submission until both results are observed. That denominator
includes worker startup, loading, checkpointing, validation, and export. It is not
the sum or average of the workers' training-only throughput. The single-GPU execution
cost uses the same wall interval once. No per-tenant dollar allocation is invented.
The full monitor session also includes local Ray initialization/shutdown, so its
cost is a different scope and should not be added to execution cost.

The existing platform monitor runs without database attribution in this scenario.
Its time is labelled unattributed; that does not mean the GPU was idle. Worker
context and report.json identify both tenants independently. Missing samples remain
explicit errors, rather than zeros. Hard crashes may leave partial reports/events;
there are no automatic retries. Worker cancellation on runner interruption can leave
partial checkpoints or artifacts, just as in the direct training path.

For held-out quality evaluation, run scripts/evaluate_intents.py on each new adapter
and its original manifest after both workers finish. Keep evaluation outputs and
timing separate from the training monitor session. For later FIFO comparisons,
match workload settings and timing boundaries; retain startup/cache effects and
CPU/thread contention as possible confounders.

## Checks

```bash
python -m pytest tests/test_concurrent_scenario.py -q
```

The tests reject changed baseline data/config, verify interval overlap, reject
sequential training as concurrent, and check single-GPU cost accounting. Actual
GPU validation is performed by the benchmark's saved PID, UUID, and interval checks.

Ray documents fractional accelerators and their limitations in its
[accelerator guide](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html).
