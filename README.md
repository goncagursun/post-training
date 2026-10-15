# post-training

Learning MVP: first establish one PEFT LoRA SFT run on RunPod.

## First increment

- `config.py` describes serializable settings without initializing a GPU.
- `training/sft.py` owns one model, adapters, Trainer, and artifact export.
- `scripts/run_sft.py` translates CLI arguments into a training call.
- The Ray worker launches training; the dispatcher and storage modules coordinate queued work and artifact paths; evaluation remains a later increment.

The four original QA examples are a mechanics smoke test, not evidence of generalization.
We train SmolLM2-135M q/v adapters with rank 8, alpha 16, completion-only loss,
and a constant learning rate. Inputs use `Question: <question>\nAnswer:`;
completions start with a space and end with EOS. Use that prompt format at inference.
TRL handles tokenization, padding masks, and causal label shifting.

## Run on RunPod

From the repository root, with the existing CUDA-enabled PyTorch environment active:

```bash
python -m pip install -e .
export HF_HOME=/workspace/post-training-data/cache/huggingface
python scripts/run_sft.py --config configs/sft_lora.json --run-dir /workspace/post-training-data/customers/customer-a/jobs/sft-001/attempts/001
```

The configuration requires CUDA and BF16 explicitly. There is no silent CPU fallback.
For a deliberate CPU smoke run, change both `device` to `cpu` and `precision` to `fp32`
in a separate config file. A run directory must not already exist; use a new attempt
path for each invocation. Failed attempts may leave partial files.

Each run produces resolved config, Trainer checkpoints, a final `adapter/` containing
adapter weights/config plus tokenizer files, summary metrics, and `result.json`.
The adapter must be loaded together with the base model, not as a standalone full model.
Checkpoint recovery and artifact reload evaluation are the next increments.

Dependencies are bounded for the TRL 0.23 API, but this is not yet a tested environment
lock. Preserve the existing CUDA PyTorch installation and inspect pip's resolution.
After a successful GPU run, capture exact package versions and pin the model revision
(currently `main`) for reproducibility. GPU execution has not been validated locally.

## Run through Ray (single node)

Install the updated project with `python -m pip install -e .` in your active environment.
The original `scripts/run_sft.py` still runs training directly.

CPU smoke run from the repository root:

```bash
python scripts/run_sft_ray.py --config configs/sft_lora_cpu.json --run-dir runs/ray-cpu-001
```

RunPod GPU run:

```bash
export HF_HOME=/workspace/post-training-data/cache/huggingface
python scripts/run_sft_ray.py --config configs/sft_lora.json --run-dir /workspace/post-training-data/customers/customer-a/jobs/ray-sft-001/attempts/001
```

The CLI starts local Ray, validates declared capacity, and submits one task. The worker
requests one CPU and one GPU (zero GPUs for the CPU config). It calls the same `run_sft`
function and returns adapter location, step count, and loss. Ray handles task placement;
TRL still owns the training loop. `max_calls=1` releases the worker process after each
attempt, and `max_retries=0` leaves recovery decisions to the future platform.

This is Ray Core task execution, not distributed Ray Train. It assumes the driver and
worker share this machine's filesystem. The CLI waits for completion and must remain
running; it is not yet a durable job service. No PostgreSQL queue, API, cancellation,
or resume handling is introduced here. The Ray dependency range is not an environment
lock; record the exact working version after validation on RunPod.

## Tenant API and durable job records

See [the API quickstart](docs/api_quickstart.md) for PostgreSQL setup, onboarding two
tenants, API-key authentication, and submit/status requests. Jobs remain QUEUED until
you run the separate dispatcher. Install the optional backend tools with
`python -m pip install -e '.[platform,dev]'`.

## Dispatch the queued jobs

See [the dispatcher quickstart](docs/dispatcher_quickstart.md) to add the execution
table, run the CPU dispatcher, and inspect results through the existing status API.

## Reload an adapter and generate

In a fresh terminal/process, use the adapter_dir returned by your job:

```bash
python scripts/generate_adapter.py --adapter-dir "<adapter_dir>" --prompt "What is LoRA?" --device cpu
```

The script reloads the saved tokenizer, base model, and LoRA weights independently
of training. It shares the training prompt format, uses greedy generation with no
gradients, and prints only the newly generated answer with artifact metadata as JSON.
The default budget is 64 new tokens; override with --max-new-tokens. CUDA is explicit;
the smoke check uses FP32 on either device. It does not write artifacts or job state.
Base weights are downloaded if not cached. Existing runs use the mutable revision
main, so exact long-term reproducibility still requires pinning a model commit.
This verifies artifact loading and generation, not held-out quality or improvement.

## Bitext intent fine-tuning

See [the Bitext quickstart](docs/bitext_quickstart.md) for separate tenant datasets,
registration, SFT jobs, held-out loss, and base-versus-adapter accuracy evaluation.
