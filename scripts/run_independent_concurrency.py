"""Run scenario 2: two independent LoRA workers on one GPU, preserving FIFO."""
import argparse
import json
from post_training.concurrent_scenario import run_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='JSON with exactly two tenant configs and manifests')
    parser.add_argument('--output-dir', required=True, help='New directory outside the FIFO experiment')
    parser.add_argument('--gpu-hourly-rate-usd', type=float)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    result = run_scenario(args.spec, args.output_dir, args.gpu_hourly_rate_usd, args.gpu_uuid)
    print(json.dumps({'status': result['status'], **result['measurements']}, indent=2))


if __name__ == '__main__':
    main()
