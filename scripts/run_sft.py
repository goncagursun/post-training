"""Run from the repository root after installing the package."""
import argparse
from dataclasses import asdict
import json

from post_training.config import SFTJobConfig
from post_training.training.sft import run_sft


def main():
    parser = argparse.ArgumentParser(description="Train one SmolLM2 PEFT LoRA adapter")
    parser.add_argument("--config", default="configs/sft_lora.json")
    parser.add_argument("--run-dir", required=True, help="New directory for this run's artifacts")
    args = parser.parse_args()
    result = run_sft(SFTJobConfig.from_json(args.config), args.run_dir)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
