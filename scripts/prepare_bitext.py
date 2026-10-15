"""Download the pinned public dataset and create two immutable tenant bundles."""
import argparse

from post_training.bitext import REVISION, SOURCE, prepare_bitext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, help="New version directory, outside Git")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from datasets import load_dataset
    rows = load_dataset(SOURCE, revision=REVISION, split="train")
    paths = prepare_bitext(rows, args.output_root, args.train_size,
                          args.validation_size, args.test_size, args.seed)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
