# post-training

Learning MVP: first establish one PEFT LoRA SFT run on RunPod.

## First increment

- `config.py` describes serializable settings without initializing a GPU.
- `training/sft.py` owns one model, adapters, Trainer, and artifact export.
- `scripts/run_sft.py` translates CLI arguments into a training call.
- Worker, orchestration, storage, and evaluation modules remain empty for later increments.

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
