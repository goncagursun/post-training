# Physical batching and independent concurrency

This experiment combines larger per-job batches with multiple simultaneous tenants
on one A40. It preserves the FIFO, shared-base, original concurrency sweep, and
single-job batch counterfactual artifacts.
It submits simulated tenant jobs directly to Ray; API authentication, PostgreSQL
request handling, and production dispatcher overhead are outside these windows.

## Prespecified matrix

Three seeded randomized blocks each contain these ten cells:

| Physical batch per job | Simultaneous jobs |
| --- | --- |
| 4 | 1, 2, 4, 8 |
| 8 | 1, 2, 4, 8 |
| 1 (fresh endpoint controls) | 1, 8 |

Every job uses the same 500-example Bitext tenant-A clone, one epoch, fixed learning
rate 0.001, rank-8 BF16 LoRA, eager attention, pinned SmolLM2-135M revision, and one
CPU thread. Each simulated tenant has a distinct fixed seed (42–49), an independent
frozen base allocation, adapter, optimizer, process, and CUDA context. Each cell
uses the first N tenant profiles. There is no shared-base allocation and no CUDA MPS.
Useful input tokens per tenant are 32,749, including prompt and completion tokens
and excluding padding. Optimizer update counts are 500, 125, and 63 at batches one,
four, and eight. Checkpoints occur at 100-example boundaries rounded to the next
complete batch, retaining the latest two. The final partial batch is retained.

Create 16 fresh solo references (all eight tenants at batches four and eight) in
seeded randomized order. Batch-one references are reused from the original sweep;
fresh batch-one endpoint cells check consistency, but their slowdown denominators
still use historical one-shot solo timings. For each batch, concurrent adapters must
exactly match their own solo adapter weights, initialization hash, sample-order hash,
and useful token budgets. The first tenant's fresh batch-four/eight reference must
also reproduce the previous single-job counterfactual's adapter exactly. Integrity
failures stop the matrix; ordinary resource failures remain recorded and exclude
that cell from a capacity claim. There are no silent retries.

## Common execution boundaries

Wait for the GPU to have no compute processes before each trial (up to 20 seconds).
Initialize a zero-CPU/GPU Ray synchronization actor before the execution timer. The
same actor protocol is used for every solo reference and matrix cell, including N=1.
Each training worker joins a barrier after model loading, tokenization, and optimizer
setup and before its training timer. All workers must arrive before training starts.

This synchronized start is deliberate: batch-eight jobs have just 63 updates, so
otherwise staggered loading could prevent sufficient all-worker training overlap.
Loading and barrier wait are included in submission-to-result latency; they are
excluded from the training-only window. The barrier protocol differs from the older
unbarriered concurrency sweep, so fresh cells are the primary comparison here.

Ray reserves 0.5 logical CPUs and 1/N GPUs per training worker, with one task per
process and no task retry. These are admission reservations, not physical GPU/CPU
partitions. Eight workers require only four logical Ray CPUs and fit the host's
seven whole logical CPUs; the container quota is 7.65 cores. Actual GPU UUIDs,
distinct worker PIDs, useful token budgets, and frozen-base integrity are checked.

## Measurements and decision rule

Retain first task submission to last observed result for all-job completion time
and aggregate end-to-end input tokens/s. Record each tenant's own submission-to-result
latency and training speed. Measure aggregate training throughput over the intersection
of all workers' training phases, counting actual useful input tokens in completed
physical batches. The final partial batch is counted using its actual token count.
Boundary uncertainty is at most one completed batch per worker, so this estimate is
less precise for short high-batch training windows than for long runs.

A valid trial requires at least ten seconds of common training overlap, at least
ten completed optimizer updates per worker in that overlap, and at least five
GPU samples in it. This is a distinct, prespecified overlap threshold from the
old batch-one sweep's 50-update threshold; batch-eight jobs have only 63 updates
in total. Missing monitoring data, insufficient overlap, and failures cannot be
treated as successful capacity cells.

Sample GPU utilization, memory, power, temperature, container CPU quota usage,
throttling, and container memory every second. Retain summaries over both the full
execution interval and the common training interval. Whole-device samples do not
identify SM activity or separate weights, contexts, and activations. Container
measurements include other container processes. Keep worker-local CUDA allocator
peaks separately; summing independent per-process peaks is not a device peak.

At each physical batch size, the exploratory capacity estimate is the largest
successful tested N where every tenant's median completion slowdown is at most
1.20 times its matching solo reference. All three repeats must succeed, monitoring
and common overlap must be valid, and adapter checks must pass. Report throughput
maxima and candidate throughput plateaus separately from this latency criterion.
The sparse N levels cannot establish an exact untested boundary. Single solo
timings and three repeats leave uncertainty near the threshold.
Batch-one endpoint controls are diagnostic and are not a new capacity sweep.

The relative slowdown denominator changes with the physical batch size. A faster
solo job therefore has a smaller absolute time allowance under the 20% rule. A
smaller passing tenant count at a larger batch does not by itself mean lower
platform efficiency: compare absolute completion time and aggregate throughput at
the same N and total example budget as well as the within-batch latency ratio.

## Quality and reproducibility

After every timing run finishes, generate predictions on all 100 held-out test
examples for each of the 24 solo reference adapters (eight seeds at each batch).
Use the same FP32 model, tokenizer, greedy decoding, 16-token generation budget,
and exact-label scoring as the previous quality evaluation. Reuse one loaded frozen
base during this untimed stage. The first tenant's generated predictions at each
batch must exactly match the earlier evaluation, checking this optimization.

Concurrent adapters inherit a solo reference's measured quality only when their
full adapter weights exactly match. Report individual tenant quality as well as
batch-level ranges; dataset clones are not independent customer populations.
Timing repeats use fixed seeds and are not additional independent quality trials.
Different batch sizes perform different numbers of updates and are not required to
produce identical adapters. No learning-rate/epoch tuning or quality noninferiority
guarantee is implied.
The test set was already inspected in earlier experiments, and batches four/eight
were chosen after that exploration. These repeated quality measurements are
descriptive regression evidence on a reused test set, not an untouched confirmatory
test. A stronger publication claim would require new held-out data and additional
independent runs.

The protocol, source snapshots, and SHA256 hashes of runner/engine/evaluator sources
are recorded before timing. The historical controlled engine remains unchanged;
the batch engine adds an optional pre-training barrier callback whose default
preserves existing callers. Earlier run snapshots preserve their original versions.

## Run and artifacts

```bash
python scripts/run_batch_concurrency.py \
  --reference-sweep /workspace/post-training-data/experiments/concurrency-sweep-001 \
  --batch-counterfactual /workspace/post-training-data/experiments/batch-counterfactual-001 \
  --output-dir /workspace/post-training-data/experiments/batch-concurrency-001 \
  --repeats 3
```

Use a new output directory. `protocol.json` precedes timing; `report.json` is updated
atomically after each solo reference and matrix trial. `quality_report.json` records
evaluation progress after training. `summary.json` is written on complete success.
Raw samples, worker reports, per-step events, checkpoints, adapters, held-out
predictions, reference provenance, and source snapshots remain outside Git.

## Results from run 001 (2026-10-16)

All 16 new solo references and 30 matrix trials completed. The 117 trial adapters
exactly matched their batch-specific solo reference weights, initial adapter hashes,
sample-order hashes, and useful token budgets. Every trial's frozen base remained
unchanged, and all common-overlap and monitoring checks passed. Three repeats per
cell are exploratory; the ranges below are observed repeat ranges, not confidence
intervals. Batch-one rows are endpoint controls, not a new batch-one capacity sweep.

| Batch | N | All finish, seconds (range) | Training-overlap input tokens/s | End-to-end input tokens/s | Worst tenant median completion ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 129.4 (121.5–132.7) | 344 | 253 | 0.987× |
| 1 | 8 | 152.4 (149.2–153.2) | 2,400 | 1,719 | 1.236× |
| 4 | 1 | 58.8 (58.0–62.9) | 1,226 | 557 | 1.017× |
| 4 | 2 | 58.6 (57.9–59.6) | 2,501 | 1,118 | 1.013× |
| 4 | 4 | 60.1 (59.7–61.9) | 4,973 | 2,178 | 1.026× |
| 4 | 8 | 69.6 (67.0–71.6) | 8,171 | 3,767 | 1.186× |
| 8 | 1 | 45.6 (45.2–46.5) | 2,330 | 717 | 1.026× |
| 8 | 2 | 47.2 (43.7–49.4) | 4,389 | 1,388 | 1.073× |
| 8 | 4 | 47.8 (45.2–50.3) | 8,634 | 2,743 | 1.070× |
| 8 | 8 | 60.4 (57.9–67.0) | 13,129 | 4,336 | 1.381× |

The largest tested N meeting the 20% worst-tenant median completion criterion is
eight at physical batch four, and four at physical batch eight. Batch four/N=8
passes narrowly: its per-repeat worst ratios were approximately 1.19, 1.24, and
1.15, so the rule is not a promise that every run stays within 20%. The decision
uses each tenant's median across repeats, then takes the worst tenant; it is not
the median of the per-repeat worst values. One-shot reference timing variability
also matters near this boundary. No exact untested tenant limit is established.

At N=8 and the same total 4,000-example budget, batch four finishes about 2.2 times
faster than batch one (69.6 versus 152.4 seconds); batch eight finishes about 2.5
times faster (60.4 seconds) but exceeds its own relative slowdown criterion. These
are equal-example, fixed-learning-rate comparisons with different update counts,
not identical optimization trajectories. Batch eight/N=4 provides shorter individual
job latency, while batch four/N=8 supplies more aggregate throughput within the
prespecified relative rule. The relative rule has a different solo denominator at
each batch size and must be interpreted alongside absolute latency.

Combining batching with concurrency improves training throughput beyond either
alone: batch eight rises from 2,330 training input tokens/s at N=1 to 13,129 at N=8.
Training throughput still rises 64% from N=4 to N=8 at batch four and 52% at batch
eight. Neither batch sweep finds a <=10% throughput plateau through eight.

During common training overlap at N=8, median container CPU quota use is 99.6%,
98.6%, and 93.3% at batches one, four, and eight, respectively. Median GPU activity
is 95.5%, 93.5%, and 96.2%; sampled training-window peak VRAM is 6.81, 12.82, and
18.85 GiB. CPU quota pressure remains substantial. GPU activity is not SM occupancy
or proof of compute saturation: it is already about 93% at batch eight/N=4, yet
throughput rises another 52% at N=8. The measurements do not isolate a single
hardware bottleneck or establish the A40's maximum capacity.

All 24 reference adapters were evaluated on the same 100-example test set:

| Physical batch | Mean accuracy across eight tenant seeds | Per-tenant accuracy range |
| --- | --- | --- |
| 1 | 96.50% | 93–99% |
| 4 | 98.38% | 97–100% |
| 8 | 97.75% | 97–99% |

These are descriptive scores on reused test data, not independent examples across
tenants or proof of quality superiority/noninferiority. Within each batch, exact
weight equality transfers the solo quality result to its concurrent adapter. Across
batch sizes, adapters differ because the optimizer trajectory changes. The optimized
single-base evaluator exactly reproduced the historical tenant-1 generated
predictions at all three batches.

Artifacts are under
`/workspace/post-training-data/experiments/batch-concurrency-001/`:
`report.json`, `analysis.json`, `batch_concurrency.png`, `quality_report.json`,
individual held-out predictions, source snapshots, adapters, and raw resource/event
logs. Generate the final audit and plot with:

```bash
python scripts/analyze_batch_concurrency.py \
  --experiment-dir /workspace/post-training-data/experiments/batch-concurrency-001
```

Regression verification after the timing phase: 52 passed, four skipped. Historical
FIFO and other experiment artifacts were preserved.
