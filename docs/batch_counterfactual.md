# One-job batch-size counterfactual

This separate experiment asks whether increasing a single job's physical batch size
uses this GPU more efficiently than running multiple batch-one jobs. Historical FIFO,
shared-base, concurrency-sweep code and artifacts remain unchanged.

## Protocol

Run one adapter at a time with physical batch sizes 1, 2, 4, 8, 16, and 32. Use
three randomized blocks, with each batch size appearing once per block. Every run
uses the same 500-example tenant-A clone, seed 42, pinned SmolLM2-135M revision,
rank-8 BF16 LoRA, learning rate 0.001, one epoch, one CPU thread, and no gradient
accumulation. The sample order and initial adapter hash must match the historical
solo reference. Every run processes exactly 32,749 non-padding input tokens and
2,400 supervised completion tokens.

The batch engine is a small, separately named derivative of the historical controlled
engine. It reuses the same optimizer-update, tokenization, and export helpers. Its
batch-one adapter must exactly reproduce the historical adapter in every repeat.
The historical engine file hash is validated before launch. Complete source copies,
hashes, and a protocol file are saved before timing begins.

The final partial batch is retained. Optimizer update counts are respectively 500,
250, 125, 63, 32, and 16. There is no learning-rate or epoch tuning. These are
equal-example-budget comparisons, not equal-update-budget comparisons, and larger
batches need not produce the same adapter. Their test quality must be reported
alongside speed. Timing repeats at the same seed are not independent quality trials.

Logging records every optimizer update. Checkpoints are written at each crossed
100-example boundary, rounded up to the next complete batch, retaining the latest
two. This produces five checkpoint writes at every tested batch size. Validation
and final export are identical across runs. Rounding the checkpoint positions is
a small remaining difference between physical batch sizes.

## Measurements and interpretation

Measure first task submission to observed result for end-to-end latency, plus the
worker's full training phase for training throughput. Both useful input tokens/s
metrics exclude padding tokens; padding efficiency is reported separately. Prompt
and completion tokens both count. Whole-device GPU samples and container CPU/RAM
samples cover the same end-to-end interval; CUDA peak allocation/reservation counters
provide additional memory evidence for short training phases. GPU query errors are
fatal rather than treated as zero.

After all timing runs finish, evaluate each distinct batch-size adapter on the full
100-example held-out test split with identical greedy generation and exact-label
scoring. Also report held-out validation loss. Repeated adapters at a given batch
size must be bit-for-bit equal, allowing one full generation evaluation per batch
size. No batch is selected using the test set; this is a descriptive sweep, with
no prespecified accuracy noninferiority margin or statistical quality guarantee.

Historical independent concurrency uses the same model, data profile, sample budget
per job, and CPU-thread setting, but its timings come from an earlier session.
Report that limitation. Its aggregate training throughput processes multiple
independent adapters; a single larger-batch job processes one. Token throughput
can be compared, but raw makespan and completed adapter count are different units.
For example, one 500-example job cannot be described as completing the work of
seven distinct 500-example tenant jobs. Per-tenant personalization is preserved
only in the concurrent case.

A larger batch can improve speed while degrading quality because it performs fewer
updates. Any finding applies to this fixed learning-rate, fixed-data-budget recipe.
Further tuning might change the quality/speed frontier. Batching and concurrency
can also be combined; this experiment does not establish that they are mutually
exclusive alternatives.

## Run

```bash
python scripts/run_batch_counterfactual.py \
  --reference-sweep /workspace/post-training-data/experiments/concurrency-sweep-001 \
  --output-dir /workspace/post-training-data/experiments/batch-counterfactual-001 \
  --batches 1 2 4 8 16 32 --repeats 3
```

Use a new output directory. `protocol.json` records choices before trials,
`report.json` records atomic progress, and `summary.json` is written after timing
and quality evaluation complete. Raw trial samples, worker telemetry, optimizer
checkpoints, adapters, and held-out predictions remain outside Git.

## Memory audit amendment for run 001

Inspection after timing found that a few end-to-end device-memory peaks captured
the preceding Ray worker's CUDA context during teardown. Those raw samples are
preserved but are not used as the new job's peak VRAM. The analysis instead reports
whole-device samples restricted to that job's training phase, together with the
worker's own CUDA allocator peak counters. Training phases had at least five GPU
samples in this run; allocator counters remain useful because sampling can miss
short peaks. This correction was chosen after inspecting the timing data and is
recorded as an amendment, rather than described as prespecified.

The runner now waits for the GPU to have no compute processes before starting a
trial's sampler and timer, rejecting a busy GPU after 20 seconds. Run 001's source
snapshot predates this guard; its timings are not presented as reruns with that
guard. Training throughput and end-to-end timings remain the original measurements.

After training and quality evaluation complete:

```bash
python scripts/analyze_batch_counterfactual.py \
  --experiment-dir /workspace/post-training-data/experiments/batch-counterfactual-001 \
  --concurrency-dir /workspace/post-training-data/experiments/concurrency-sweep-combined-001
python scripts/plot_batch_counterfactual.py \
  --experiment-dir /workspace/post-training-data/experiments/batch-counterfactual-001 \
  --concurrency-dir /workspace/post-training-data/experiments/concurrency-sweep-combined-001
```

## Results from run 001 (2026-10-16)

All 18 timing trials and six full held-out evaluations completed. Times and rates
are medians of three repeats; quality uses one deterministic adapter per batch size
and the same 100 test examples. Every repeat reproduced its batch-size adapter
exactly. All three batch-one adapters also exactly matched the historical reference,
and every frozen base remained unchanged.

| Physical batch | Optimizer updates | End-to-end seconds (range) | Training input tokens/s | Test accuracy | Sampled training VRAM, GiB |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 127.6 (127.3–127.7) | 346 | 98/100 | 0.86 |
| 2 | 250 | 82.7 (80.0–82.9) | 665 | 98/100 | 1.18 |
| 4 | 125 | 58.4 (56.3–59.1) | 1,228 | 100/100 | 1.85 |
| 8 | 63 | 44.1 (42.8–45.4) | 2,305 | 99/100 | 2.97 |
| 16 | 32 | 37.1 (35.0–40.7) | 3,949 | 94/100 | 4.81 |
| 32 | 16 | 35.8 (34.0–37.5) | 6,026 | 60/100 | 6.53 |

The memory column is the median sampled training-window peak after the amendment
above. It is not a guaranteed instantaneous peak; worker CUDA allocator peaks are
retained in `analysis.json` and plotted separately.

Batch eight processes training input tokens 6.7 times faster than fresh batch one
and finishes its one-adapter job 2.9 times faster, with 99/100 observed test accuracy
versus 98/100. That small accuracy difference does not establish superiority or
statistical noninferiority. Batch sixteen and thirty-two improve speed further but
lose observed test accuracy under this fixed one-epoch, fixed-learning-rate recipe.
The result does not show that larger batches cannot recover quality with tuning.

One batch-eight job's training-phase throughput (2,305 input tokens/s) approximately
matches the earlier aggregate all-job-overlap throughput of seven batch-one jobs
(2,296 input tokens/s). This supports larger batches as a way to reduce per-job GPU
underuse and demand for many CPU processes. It does not replace multi-tenant
concurrency: those seven jobs produce seven separate adapters and process seven
times as many examples in total.

For these short jobs, the end-to-end comparison also matters: one batch-eight job
processes about 742 input tokens/s including startup, validation and export; seven
concurrent batch-one jobs achieved 1,682 aggregate input tokens/s. Their separate
startup work overlaps. Thus the single-job training-throughput result alone does
not establish higher platform-level end-to-end throughput. A useful subsequent
experiment would vary concurrency with a larger per-job batch, subject to validation
quality, rather than treating batching and concurrency as exclusive choices.

Artifacts: `/workspace/post-training-data/experiments/batch-counterfactual-001/`
contains the raw report, amended `analysis.json`, `batch_scaling.png`, snapshots,
adapters, test predictions, and resource samples. Regression validation: 50 passed,
four skipped. The historical FIFO and concurrency artifacts were preserved.
