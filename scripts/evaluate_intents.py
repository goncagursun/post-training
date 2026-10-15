"""Measure base/adapter accuracy on a prepared Bitext held-out split."""
import argparse
import json
from pathlib import Path

from post_training.intent_evaluation import evaluate_intents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--adapter-dir", help="Omit for a base-only baseline")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, help="Partial smoke check only; omit for full accuracy")
    parser.add_argument("--output", required=True, help="New report file outside Git")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error("Output already exists; use a new report path")
    report = evaluate_intents(args.manifest, args.split, args.adapter_dir, args.device, args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("base", "adapter") if key in report}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
