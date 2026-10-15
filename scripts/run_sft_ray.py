"""Launch one training task on a local, single-node Ray runtime."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import ray

from post_training.config import SFTJobConfig
from post_training.worker import train_sft_worker


def main():
    parser = argparse.ArgumentParser(description="Run SFT LoRA through Ray")
    parser.add_argument("--config", default="configs/sft_lora.json")
    parser.add_argument("--run-dir", required=True, help="New directory for this attempt")
    args = parser.parse_args()
    config = SFTJobConfig.from_json(args.config)
    # Resolve before dispatch so the worker does not depend on its working directory.
    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.exists():
        parser.error(f"Run directory already exists: {run_dir}")
    num_gpus = 1 if config.device == "cuda" else 0

    # Start a local runtime even if RAY_ADDRESS points at another cluster.
    # This increment assumes one machine and its local/persistent filesystem.
    ray.init(address="local", include_dashboard=False)
    try:
        resources = ray.cluster_resources()
        if resources.get("CPU", 0) < 1 or resources.get("GPU", 0) < num_gpus:
            raise RuntimeError(f"Ray cannot satisfy 1 CPU / {num_gpus} GPU; resources: {resources}")
        print(f"Ray resources: {resources}", flush=True)
        result_ref = train_sft_worker.options(num_gpus=num_gpus).remote(asdict(config), str(run_dir))
        print("Training submitted to Ray; waiting for the result.", flush=True)
        # A later dispatcher will poll results while continuing to manage the queue.
        result = ray.get(result_ref)
        print(json.dumps(result, indent=2))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
