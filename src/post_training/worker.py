"""Ray execution boundary for the existing SFT training function."""
from dataclasses import asdict

import ray

from post_training.config import SFTJobConfig
from post_training.training.sft import run_sft


@ray.remote(num_cpus=1, num_gpus=1, max_calls=1, max_retries=0)
def train_sft_worker(config: dict, run_dir: str,
                     examples: list[tuple[str, str]] | None = None,
                     validation_examples: list[tuple[str, str]] | None = None,
                     dataset_metadata: dict | None = None, context: dict | None = None,
                     resume_checkpoint: str | None = None,
                     interrupt_after_checkpoint: int | None = None) -> dict:
    """Own one training run in a fresh process and return small metadata only.

    GPU allocation can be overridden to zero for an explicit CPU smoke test.
    Automatic retries are disabled: run_sft requires a fresh output directory,
    so rerunning a partially written attempt needs platform recovery logic.
    """
    context = {**(context or {}), "ray_task_id": ray.get_runtime_context().get_task_id()}
    def execute():
        result = run_sft(SFTJobConfig(**config), run_dir, examples=examples,
                         validation_examples=validation_examples, context=context,
                         resume_checkpoint=resume_checkpoint,
                         interrupt_after_checkpoint=interrupt_after_checkpoint)
        if dataset_metadata is not None:
            import json
            from pathlib import Path
            (Path(run_dir) / "dataset.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
        return asdict(result)

    if "job_id" in context:
        from post_training.recovery import worker_guard
        with worker_guard(run_dir):
            return execute()
    return execute()
