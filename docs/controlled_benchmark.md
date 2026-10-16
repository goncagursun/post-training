# Controlled three-scenario benchmark

The original FIFO, independent-concurrency, and shared-base runs remain exploratory
artifacts. This runner creates a new experiment using one worker engine for every
mode: solo A, shared A/B, FIFO A then B, and independent A/B. It does not replace the
platform dispatcher or alter earlier results.

## Common training recipe

Every mode calls run_controlled_worker and the same isolated_update function. The
model revision is pinned. Adapter initialization and per-tenant epoch permutations
are seeded identically and hashed. All modes use eager attention, deterministic
PyTorch algorithms, TF32 disabled, the same CUDA workspace setting, completion-only
loss, AdamW settings, clipping, checkpoint cadence/retention, per-update logging,
validation, integrity hashing, and export code. Batch size/accumulation must be one
and LoRA/model dropout zero. Shared mode alternates complete tenant updates; it does
not perform simultaneous kernel execution.

FIFO launches the single-tenant worker twice in sequence. Independent mode submits
two single-tenant workers before awaiting either, using logical 0.5-GPU reservations.
Shared mode launches one worker containing both adapters. Solo A uses that same
engine with A only. There is no special reduced-overhead solo implementation.

## Isolation gate

The driver runs solo A and shared A/B first. It checks initial adapter and batch-order
hashes, final adapter tensors, and full-vocabulary next-token logits for three fixed
validation prompts. Prespecified tolerances are weight atol=1e-6/rtol=1e-5 and FP32
inference-logit atol=1e-5/rtol=1e-5; exact equality is also reported. Raw logits are
saved. A failed comparison stops the benchmark before FIFO/independent measurements
are published. All workers also check trainable optimizer membership, disjoint
optimizer parameters when shared, and unchanged base-weight hashes.

This is evidence for this model, recipe, data, and update schedule. It is not a proof
of arbitrary tenant-code isolation. The implementation does not accept customer
training code and is not a security sandbox.

## Timing and monitoring

For every mode, execution starts immediately before first task submission and ends
after the last result is observed. The interval includes process startup, loading,
preparation, training, checkpointing, validation, integrity checks, and export. Ray
initialization, monitor readiness, isolation inference, and held-out generation
checks are outside the interval. Aggregate throughput divides the total completed
input tokens by this wall interval. GPU-seconds equal elapsed seconds on the one
allocated GPU. Dollars are the same quantity multiplied by the configured hourly
price, not an independent efficiency measurement.

The independent monitor is ready before the clock starts. GPU sample aggregates in
each mode report are filtered to that exact execution interval, avoiding different
startup/shutdown gaps. Sampling still has approximately one-second granularity;
missing data must be checked before interpretation. These are whole-device readings,
not per-tenant SM occupancy. Full monitor-session summaries are retained separately.

## Run

Activate the project environment, set HF_HOME, and ensure no other GPU workloads or
FIFO dispatchers are active. The specification is the same two-tenant format used
by independent concurrency; original baseline configs/manifests can be referenced.
They are validated before the model revision is pinned consistently in all effective
configs. Use a real model commit, not a mutable branch:

```bash
python scripts/run_controlled_benchmark.py \
  --spec /workspace/post-training-data/experiments/shared-base-001.spec.json \
  --output-dir /workspace/post-training-data/experiments/controlled-001 \
  --model-revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --gpu-hourly-rate-usd 0.59
```

Use a new directory. Exactly one visible GPU and at least two Ray CPUs are required.
Each worker declares one CPU; existing OMP/MKL settings are inherited consistently.
The driver sets CUDA_VISIBLE_DEVICES and CUBLAS_WORKSPACE_CONFIG before Ray starts.
The engine snapshot and complete effective inputs are saved with the experiment.

Per-mode report.json files retain complete worker reports, job artifacts, primary
execution-window metrics, and filtered GPU measurements. isolation.json records the
solo/shared comparison, and isolation_fixed_logits.pt contains the measured logits.
The top-level report records progress and isolation status. A completed training
report is not a substitute for a separate held-out quality report.

## Interpretation limits

The initial controlled experiment has one repetition per mode with fixed mode order
and a warm shared disk cache. It cannot establish statistical significance, steady
state behavior, or remove all host-load/thermal variance. Repeat in randomized order
before making broad performance claims. Process count and model residency still
differ by design; a memory difference cannot be decomposed into weights, contexts,
activations, and caches from total VRAM alone. Report observed execution-mode results
rather than attributing all differences solely to base-weight sharing.

Checkpoints are inspection snapshots, with no coordinated resume support. Existing
experiment directories are never overwritten. The tests include a small local model
where solo A and interleaved A must have exactly equal final weights.

After training completes, publish the controlled comparison with:

```bash
python scripts/finish_controlled_benchmark.py \
  --output-dir /workspace/post-training-data/experiments/controlled-001
```

This checks each tenant's final weights, initialization, and batch order across all
three modes, then evaluates all six adapters on their own held-out test splits.
It writes trajectory_checks.json, quality_report.json, and comparison.json. It refuses
to publish if the trajectory checks fail and refuses to overwrite an existing
comparison. Quality generation runs are outside the primary measurement windows.
