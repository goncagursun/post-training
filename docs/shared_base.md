# Scenario 3: shared frozen base with interleaved tenant adapters

This separate benchmark uses one Ray worker, one loaded base model, two named LoRA
adapters, and two AdamW optimizers. It alternates complete updates A, B, A, B until
each tenant finishes its configured epochs. Both jobs are active together, but GPU
updates are sequential. This is not simultaneous kernel execution, mixed-tenant
batching, CUDA MPS, or shared-base training across independent processes.

FIFO (scenario 1) and independent concurrency (scenario 2) remain separate. Neither
runner nor its results are modified. Do not run another scenario or FIFO dispatcher
against the same GPU during this benchmark. The new runner operates outside the API
and database queue and requires a trusted administrator; sharing a process does not
provide a security boundary for arbitrary tenant code.

## Run

Use the same environment, HF_HOME, and specification format as
[independent concurrency](independent_concurrency.md). Each specification entry has
tenant_id, config_path, manifest_path, and optional baseline_adapter_dir. Supply the
original FIFO config and adapter paths to validate that data and configs match the
baseline before training. Both tenants must use the same base, model revision,
precision, and CUDA device. Initial support requires batch size 1, gradient
accumulation 1, and LoRA dropout 0, matching our recorded baseline.

```bash
python scripts/run_shared_base.py \
  --spec /workspace/post-training-data/experiments/shared-base-001.spec.json \
  --output-dir /workspace/post-training-data/experiments/shared-base-001 \
  --gpu-hourly-rate-usd 0.59
```

Use a new output directory. One visible GPU is selected automatically; specify
--gpu-uuid on multi-GPU hosts. The worker requests one CPU and one GPU. It loads the
base once and creates adapters on that object, with disjoint optimizer parameter
sets. Base weights are frozen and stay resident during adapter switches. The
selected adapter alone has requires_grad enabled during its update. Gradients are
cleared before switching. Updates do not merge adapters into the base.

## Validation and artifacts

The implementation checks:

- Each active trainable parameter belongs to that tenant's optimizer.
- Optimizer parameter sets are disjoint.
- Each tenant's first update leaves the inactive adapter byte-for-byte unchanged.
- A hash of every non-LoRA parameter is unchanged after training.
- Each adapter exports independently and carries its own config and dataset manifest.

The worker uses PEFT's named-adapter APIs. PEFT stores named exports in nested
directories; the runner normalizes each to a standalone adapter/ directory for the
existing generation and evaluation scripts. The frozen base remains separate.

Outputs:

- report.json: driver completion status, costs, shared-worker results, and full
  platform monitor summary.
- worker/report.json: intermediate progress every 100 tenant updates and final
  worker measurements; worker/events.jsonl records every tenant update.
- worker/telemetry.json: shared process phase timings. These are not per-tenant
  process memory metrics.
- worker/customers/<tenant>/attempts/001/: config.json, dataset.json, result.json,
  the exported adapter/, and the latest two inspection checkpoints.
- monitoring/: raw whole-GPU samples and a finalized monitor summary.

Checkpoints contain the tenant adapter and optimizer state, with the current step.
They are inspection snapshots, not coordinated recovery checkpoints for both jobs.
Resume is unsupported; do not use the FIFO recovery command on these files. A new
attempt is required after failure. Hard termination can leave partial outputs.

## Throughput, memory, and cost

The driver reports submission-to-result wall time, aggregate input tokens/s and
samples/s over that interval, and one-GPU allocation cost at the configured rate.
This includes worker startup, shared loading, preparation, training, checkpointing,
validation, integrity checks, and export. The worker's narrower execution timing is
also retained. The monitor session includes Ray startup/shutdown and has a separate
cost scope; do not add the two cost estimates.

Per-tenant active_update_seconds sums synchronized forward/backward/optimizer time,
including switching and gradient checks. It excludes waiting for the other tenant,
input transfer, checkpoints, and shared setup. Per-tenant throughput fields use the
full shared training interval, explicitly named *_per_shared_training_second; they
are not comparable to a dedicated tenant's active-compute throughput without
aligning denominators. AdamW state, adapters, activations, and allocator caches still
consume memory alongside the shared base. base_parameter_bytes measures base tensor
storage only, while CUDA peaks and device VRAM measure broader memory scopes.

Whole-GPU activity, power, and temperature belong to the shared process/workload;
they cannot be split into tenant GPU utilization. The independent monitor deliberately
runs without the FIFO database attribution. Unattributed time does not imply idle.
No Tensor Core or SM counters are collected.

## Comparison limitations

The first two scenarios use TRL's Trainer. This scenario uses an explicit PyTorch
loop to schedule updates safely on one mutable model object. It reuses TRL's
completion-only collator and reproduces its prompt/completion tokenization, failing
on truncation or an unstable boundary. It uses constant learning rate, no weight
decay, gradient clipping at 1, BF16 autocast for BF16 configs, and separate optimizer
states. LoRA initialization starts from the same CPU RNG state for each adapter.
Each tenant has its own seeded epoch permutations.

Sampler implementation, optimizer implementation, logging, checkpoint format,
validation batching, and Trainer bookkeeping differ from scenarios 1/2. Therefore a
speed difference cannot be attributed solely to base-memory sharing. Keep held-out
quality evaluation separate, and treat this as a shared-base scheduling proof of
concept rather than a controlled optimizer benchmark. Mutable model revision main
also remains a reproducibility limitation; the resolved revision is recorded.

```bash
python -m pytest tests/test_shared_base.py -q
```

This test uses a tiny locally initialized model, requiring no downloads or GPU. It
checks real adapter updates, inactive-adapter isolation, frozen weights, and exports.
The full experiment additionally verifies base integrity on the configured model.

PEFT documents named adapter activation and export in its
[model reference](https://huggingface.co/docs/peft/package_reference/peft_model).
