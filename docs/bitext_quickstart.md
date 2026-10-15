# Bitext intent SFT: a measurable task for two tenants

Source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
Pinned revision: 430d1a89bd93bd1fa23c16f29dd53e73f0087443
Dataset card license: CDLA-Sharing-1.0. Preserve source attribution and review the
source license before redistributing derived datasets. This synthetic public dataset
simulates customer ownership; it is not private customer data.

## Task and data sizes

The model reads a customer request and generates exactly one of five labels:
cancel_order, track_order, get_refund, change_shipping_address, recover_password.
We list those labels in the prompt. Training still uses the existing Question/Answer
wrapper, answer-only causal loss, and LoRA q/v adapters.

Per tenant: 500 training, 100 validation, 100 test examples, balanced across labels.
Training changes model weights, validation supports choosing settings, test is reserved
for the final comparison. Validation loss is measured after training; this increment
does not automatically select an earlier checkpoint or run generated-label evaluation.

Requests are normalized (case, punctuation, placeholders), deduplicated globally,
and grouped by unordered normalized word sets. Each such family belongs to only one
split and tenant. Unused members of a family are discarded instead of leaking across
splits. This conservative policy forced a smaller budget than the initially proposed
1,000 training examples per tenant. It does not catch all semantic paraphrases;
held-out accuracy on synthetic data can still be optimistic.

## 1. Prepare files

From the repository root, in tiny-gpt:

```bash
python scripts/prepare_bitext.py --output-root runs/datasets/bitext-v1
```

This downloads the pinned source using Hugging Face Datasets. The output directory
must be new. If it was already prepared in this session, skip this command.
Each tenant directory contains train.jsonl, validation.jsonl, test.jsonl, and a
manifest.json recording source revision, labels, seed, counts, and file checksums.
The files stay out of Git because they live under runs/. Keep the prepared files
unchanged; new data requires a new output version and registration ID.

On RunPod, use /workspace/post-training-data/datasets/bitext-v1 instead. The manifest
and data must be accessible to the dispatcher on the same machine. Local file URIs
are not automatically transferred when moving from your Mac to RunPod.

## 2. Register new dataset IDs

Keep DATABASE_URL set to your existing application database. Do not recreate tenants:

```bash
python scripts/register_dataset.py --tenant-id tenant-a --dataset-id bitext_a_v1 --manifest runs/datasets/bitext-v1/tenant-a/manifest.json
python scripts/register_dataset.py --tenant-id tenant-b --dataset-id bitext_b_v1 --manifest runs/datasets/bitext-v1/tenant-b/manifest.json
```

Registration validates checksums and tenant ownership. It refuses existing IDs rather
than changing data referenced by previous jobs. Only this admin command accepts paths;
customer API requests still contain dataset IDs. Original dataset_a/dataset_b remain
available for the tiny demo. New registration requires no database schema change.

Restart Uvicorn if it was running while the code changed.

## 3. Submit jobs using /docs

Authorize as tenant A and POST /training/jobs:

```json
{
  "dataset_id": "bitext_a_v1",
  "base_model_id": "smollm2-135m",
  "learning_rate": 0.0003,
  "epochs": 1,
  "lora_rank": 8
}
```

Use tenant B's key and bitext_b_v1 for its job. These are starting hyperparameters,
not tuned settings or a promise of improvement. Customers cannot change GPU placement.

The CPU dispatcher can run this workload, but use RunPod for repeated experiments.
Do not launch the dispatcher unless you intend to process all queued jobs:

```bash
python scripts/run_dispatcher.py --device cpu --data-root runs/platform --drain
```

On RunPod, after preparing/registering accessible data and configuring the database:

```bash
python scripts/run_dispatcher.py --device cuda --data-root /workspace/post-training-data --drain
```

The registered loader reads only train/validation into the training task. Test examples
are checked for integrity but never passed to the Trainer. Every successful run saves
dataset.json for lineage, validation.json for held-out completion loss, and its adapter.
For datasets larger than the tiny demo, checkpoint spacing is at least 100 updates
instead of every four; the final adapter is always exported and checkpoint retention
is still limited to two. Logging remains every update.

## 4. Compare base model and saved adapter

Before training you can check the baseline with:

```bash
python scripts/evaluate_intents.py --manifest runs/datasets/bitext-v1/tenant-a/manifest.json --split validation --device cpu --limit 5 --output runs/evaluation/base-smoke-a.json
```

After training, replace ADAPTER_DIR below with execution.result.adapter_dir:

```bash
python scripts/evaluate_intents.py --manifest runs/datasets/bitext-v1/tenant-a/manifest.json --adapter-dir ADAPTER_DIR --split validation --device cpu --output runs/evaluation/validation-a.json
```

With an adapter, the evaluator generates with it disabled for the base comparison,
then enabled, using identical tokenizer, examples, and greedy decoding. It checks that
the adapter's saved dataset lineage matches the selected manifest. Both modes use FP32.
Reports include exact-label accuracy, per-intent accuracy, invalid-label rate, and all
predictions. Only whitespace and case are normalized; verbose or multi-label answers
are counted as wrong. The generation budget is 16 new tokens.

--limit is only for execution checks and is marked partial in the report. Omit it for
accuracy measurements. Use validation while choosing settings; evaluate test once after
selecting the recipe, using --split test. Existing output reports are never overwritten.
The evaluator runs separately, not inside the queued job's success transition yet.
No adapter quality result is implied by successful training or lower training loss.

## Checks performed during implementation

- Prepared all 1,400 tenant examples from the pinned source with no normalized overlap.
- Longest prepared sequence was 81 tokens, below the current 256-token limit.
- Tested split balance/reproducibility, checksum failures, tenant mismatch, and scoring.
- Ran a five-example CPU training/validation/reload smoke check. This validates the
  plumbing only; full 500-example training and quality comparison remain to be run.
