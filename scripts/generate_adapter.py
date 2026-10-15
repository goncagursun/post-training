"""Reload a saved LoRA adapter and print its generated continuation."""
import argparse
from dataclasses import asdict
import json

from post_training.evaluation import generate_adapter


def main():
    parser = argparse.ArgumentParser(description="Check a saved adapter with generation")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    result = generate_adapter(args.adapter_dir, args.prompt, args.device, args.max_new_tokens)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
