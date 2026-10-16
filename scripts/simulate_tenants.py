"""Submit tenant A then tenant B through the real API for a FIFO experiment."""
import argparse
import json
import os
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-url', default='http://127.0.0.1:8000')
    parser.add_argument('--dataset-a', required=True)
    parser.add_argument('--dataset-b', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--output', required=True, help='New JSON file recording both submitted jobs')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10:
        parser.error('epochs must be between 1 and 10')
    keys = {letter: os.environ[f'TENANT_{letter.upper()}_API_KEY'] for letter in ('a', 'b')}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to submit if the experiment file already exists. Persist each response
    # immediately so an interruption after the first request does not hide that job.
    with output.open('x') as file:
        file.write('[]\n')
    jobs = []
    with httpx.Client(base_url=args.api_url, timeout=30) as client:
        for letter, dataset in (('a', args.dataset_a), ('b', args.dataset_b)):
            response = client.post('/training/jobs', headers={'X-API-Key': keys[letter]}, json={
                'dataset_id': dataset, 'base_model_id': 'smollm2-135m',
                'learning_rate': 0.001, 'epochs': args.epochs, 'lora_rank': 8})
            response.raise_for_status()
            job = response.json()
            jobs.append({'tenant_id': f'tenant-{letter}', **job})
            output.write_text(json.dumps(jobs, indent=2) + '\n')
            print(f"tenant-{letter}: {job['id']} {job['status']}", flush=True)


if __name__ == '__main__':
    main()
