"""Ray execution boundary for the existing SFT training function."""
from dataclasses import asdict

import ray

from post_training.config import SFTJobConfig
from post_training.training.sft import run_sft


@ray.remote(num_cpus=1, num_gpus=1, max_calls=1, max_retries=0)
def train_sft_worker(config: dict, run_dir: str,
                     examples: list[tuple[str, str]] | None = None,
                     validation_examples: list[tuple[str, str]] | None = None,
                     dataset_metadata: dict | None = None) -> dict:
    """Own one training run in a fresh process and return small metadata only.

    GPU allocation can be overridden to zero for an explicit CPU smoke test.
    Automatic retries are disabled: run_sft requires a fresh output directory,
    so rerunning a partially written attempt needs platform recovery logic.
    """
    result = run_sft(SFTJobConfig(**config), run_dir, examples=examples,
                     validation_examples=validation_examples)
    if dataset_metadata is not None:
        import json
        from pathlib import Path
        (Path(run_dir) / "dataset.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    return asdict(result)
