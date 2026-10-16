# Repeated independent-concurrency sweep

This benchmark varies the number of simultaneous independent training processes on
one GPU at N=1,2,4,8. Each process owns its own frozen base model, adapter, optimizer,
and CUDA context. It reuses the unchanged controlled worker/update engine. All
previous experiments and the FIFO platform service remain separate.

## Initial workload and controls

The initial screening run uses 500 updates (one epoch) per job, rather than the older
1,500-update experiments. That workload is identical across concurrency levels and
solo controls. Eight logical tenants clone the same tenant-A Bitext profile, with
distinct fixed seeds. Each clone has its own manifest and identity. This is a
homogeneous capacity experiment, not eight independently sourced customer datasets.

Each tenant first trains alone to create its own reference adapter and timing. Then
three blocks each run N=1,2,4,8 in seeded randomized order. A trial uses the first N
profiles; all jobs have equal token counts and settings. The model revision is
pinned; initialization, permutation, checkpointing, logging, and validation come
from the common engine. Each resulting adapter is compared against its own solo
reference, including initialization/order hashes and prespecified weight tolerances
(atol=1e-6, rtol=1e-5). Exact equality is recorded separately. No new quality claim is
inferred from the older three-epoch accuracy numbers.

Intra-op and inter-op CPU threads are fixed to one per process for every N and every
reference. Each Ray task declares 0.5 logical CPUs and 1/N GPUs. Fractional resource
reservations permit all eight tasks to launch on this host; they do not impose
hardware compute/VRAM partitions. The worker records actual thread settings, PID,
and GPU UUID. A successful trial requires N distinct workers on the same GPU and
an adequate intersection of their training phases (at least 10 seconds and 50
completed updates per job). No CUDA MPS is enabled.

## Measurements

Each trial measures:

- First submission to last observed result, including loading, training, validation,
  integrity checks, and export; total input tokens divided by that interval.
- Each tenant's submission-to-result latency, training-only tokens/s, and slowdown
  relative to that tenant's solo reference.
- Aggregate completed-token throughput during the intersection of all N training
  phases. Token counts are reconstructed from the exact seeded order and tokenizer;
  the time of each completed update comes from the worker's event log. This interval
  includes checkpoint/logging time and is not pure kernel time. Updates crossing
  its boundary introduce at most one update's token uncertainty per tenant.
- Whole-device utilization, memory, temperature, and power, sampled only within the
  matched execution window. Samples and errors are retained; missing is not zero.
- Container cgroup CPU usage, quota, throttling, RSS, and memory usage. The latter
  includes filesystem cache and is not model RAM. Other container processes are
  included. Both cgroup v1 and v2 are supported.

Each tenant has one solo timing reference; repeated N=1 cells estimate additional
profile-level variability. Per-tenant baseline noise remains a limitation, especially
for decisions close to the slowdown threshold. The three repeats are exploratory;
report ranges as well as medians. Do not claim a universal tenants-per-A40 number.

## Prespecified decision rules

The default tenant-friendly capacity is the largest successful tested N where every
tenant's median completion latency is at most 1.2 times its solo reference latency.
Training-only slowdown is reported independently. All repeats must succeed, adapter
checks must pass, and resource sampling/common overlap must be valid.

A candidate throughput plateau is an adjacent tested interval where the median
all-job-overlap throughput gain is at most 10%. This is separate from the largest
N meeting the tenant slowdown target and from the N with highest throughput. If
throughput still rises materially at 8, the knee has not been found within this
range. A bracket such as 4–8 does not establish an exact limit at 4 or 8.

## Run and artifacts

Stop other GPU workloads/dispatchers, activate the project environment, and set
HF_HOME. Use the existing two-tenant specification; its first profile supplies the
homogeneous workload. Baseline source data/config are validated before cloning.

```bash
python scripts/run_concurrency_sweep.py \
  --spec /workspace/post-training-data/experiments/shared-base-001.spec.json \
  --output-dir /workspace/post-training-data/experiments/concurrency-sweep-001 \
  --model-revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --levels 1 2 4 8 --repeats 3 --epochs 1
```

Use a new output directory. A longer workload can be selected with --epochs, but
must be used for both references and every trial in a fresh sweep. The driver does
not overwrite old experiments or silently retry failed trials. A failed reference
or solo-equivalence failure stops the sweep. Ordinary resource failures are recorded
and exclude that concurrency level from the eligible capacity result.

report.json is replaced atomically after each reference and trial. It includes the
randomized plan, solo references, all per-job results, failures, and the evolving
summary. summary.json is written on completion. inputs/ contains the cloned manifests
and effective configs; references/ and trials/ contain complete worker artifacts,
resource_samples.jsonl, and per-trial reports. These administrator artifacts belong
outside Git. Model, input, and thread settings must accompany any published capacity
statement. CPU/RAM contention and process/context overhead are part of this host's
measured behavior; whole-device readings do not decompose those causes.

To refine a completed sweep without retraining the solo controls, use a fresh output
directory with `--references-from <completed-sweep-dir>`. The runner requires exact
engine, thread settings, effective config, and dataset-manifest matches, and records
the original report hash. This does not modify the parent experiment. Use refinement
levels such as 5,6,7 only when they help narrow a capacity boundary; interpret marginal
throughput gains between these levels separately from the original doubling tests.

After both sweeps complete, combine them in a third, new artifact directory:

```bash
python scripts/combine_concurrency_sweeps.py \
  --coarse-dir /workspace/post-training-data/experiments/concurrency-sweep-001 \
  --refinement-dir /workspace/post-training-data/experiments/concurrency-sweep-refinement-001 \
  --output-dir /workspace/post-training-data/experiments/concurrency-sweep-combined-001
python scripts/plot_concurrency_sweep.py \
  --experiment-dir /workspace/post-training-data/experiments/concurrency-sweep-combined-001
```

The combination validates experiment controls and the original solo-reference report
hash. It preserves the original coarse intervals for plateau decisions; the 10% rule
must not be applied as if an increment from six to seven jobs were a doubling.
Neither source experiment is changed. The plot requires the analysis-only dependency
`matplotlib`; install it after timing runs finish to avoid adding setup work to the
measured host load. Input tokens/s counts prompt and completion tokens processed by
training, rather than autoregressively generated output tokens.

## Measured screening results (2026-10-16)

The coarse sweep and N=5,6,7 refinement completed 21 successful trials: three per
level. The combined administrator artifacts are under
`/workspace/post-training-data/experiments/concurrency-sweep-combined-001/`.
The original sweeps remain intact. The refinement reused the original solo controls
and ran in a subsequent session, rather than being randomized among the original
coarse cells. Session effects remain a possible confound for small differences.

All values below are medians across three repeats, except the completion range.
Slowdown is the largest per-tenant median latency ratio to that tenant's own solo
reference. VRAM is the median of whole-device sampled peaks, not allocated tensor
memory or a sum of independently observed process peaks.

| Jobs N | All jobs finish, seconds (range) | End-to-end input tokens/s | Training-overlap input tokens/s | Worst tenant completion ratio | Peak VRAM, GiB |
| --- | --- | --- | --- | --- | --- |
| 1 | 122.8 (122.8–126.2) | 267 | 341 | 0.94× | 0.86 |
| 2 | 128.3 (127.9–130.5) | 510 | 682 | 1.03× | 1.70 |
| 4 | 131.1 (129.8–139.2) | 999 | 1,336 | 1.05× | 3.42 |
| 5 | 133.6 (130.6–137.1) | 1,226 | 1,676 | 1.04× | 4.25 |
| 6 | 132.1 (131.9–134.9) | 1,488 | 1,996 | 1.08× | 5.09 |
| 7 | 136.3 (135.9–136.8) | 1,682 | 2,296 | 1.12× | 5.95 |
| 8 | 152.3 (152.3–154.3) | 1,720 | 2,404 | 1.24× | 6.81 |

Seven is the largest tested N satisfying the provisional 1.20× worst-tenant median
completion target. It is an exploratory capacity estimate for this exact 135M-model,
rank-8 LoRA, batch-one, 500-update workload and host. It is not a production guarantee
or a general A40 tenant limit. The repeated N=1 runs were faster than their single
solo timing reference, illustrating why baseline variability matters near a threshold.

The original doubling comparisons show no <=10% throughput plateau through eight.
The refinement reveals diminishing returns between seven and eight: training-overlap
throughput increases 4.7%, and end-to-end throughput increases 2.3%, while completion
latency increases. This suggests a practical knee around seven to eight jobs; testing
beyond eight would be needed to establish a sustained plateau.

During common training overlap, median container CPU demand reaches 91.3% of its
7.65-core quota at seven jobs and 99.7% at eight. These readings include other
container processes. CPU quota pressure is consistent with the observed slowdown;
it does not isolate causation or establish GPU compute saturation. GPU memory has
substantial headroom: sampled peak usage at eight jobs is 6.81 GiB on this roughly
45-GiB device. Repeating with more CPU capacity would help distinguish host and GPU
limits.

All 99 trial adapters exactly match their own solo adapter tensors, with matching
initialization and data-order hashes. Every frozen base remained unchanged. All
common training overlaps were adequate, and GPU sampling recorded zero query
errors. These are final-weight reproducibility checks for the tested workload, not
proof of platform-wide tenant isolation or a new held-out accuracy claim.

`report.json` retains each tenant's completion time, training speed, solo comparison,
and raw artifact paths. `summary.json`, `audit.json`, `overlap_cpu.json`,
`host_summary.json`, and `scaling.png` provide the aggregate evidence. Regression
validation after measurements: 49 tests passed, four skipped. Hourly-cost arithmetic
is omitted from the comparison because a fixed $0.59/hour simply rescales elapsed
time and adds no independent performance evidence.
