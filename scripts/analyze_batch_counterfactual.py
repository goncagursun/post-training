"""Audit a completed batch counterfactual; retain raw timing/memory records unchanged."""
import argparse
import json
from pathlib import Path
import statistics

from post_training.concurrency_sweep import summarize_resources, training_events


def analyze(root, concurrency_root):
    report = json.loads((root/'report.json').read_text())
    concurrent = json.loads((concurrency_root/'report.json').read_text())
    if report['status'] != 'COMPLETE' or concurrent['status'] != 'COMPLETE':
        raise ValueError('Both experiments must be complete')
    if (root/'analysis.json').exists():
        raise FileExistsError(root/'analysis.json')
    def spread(values):
        return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    details = []
    for trial in report['trials']:
        worker = Path(trial['job']['run_dir']).parents[3]
        start, end, _ = training_events(worker)
        raw = worker.parent/'resource_samples.jsonl'
        first = json.loads(raw.read_text().splitlines()[0])
        gpu = first['gpus'][0]['gpu_uuid']
        resources = summarize_resources(raw, start, end, gpu)
        if not resources['gpu']['sample_count'] or resources['gpu_query_errors']:
            raise RuntimeError('Training-window GPU samples unavailable')
        details.append({'batch_size': trial['batch_size'], 'repeat': trial['repeat'],
            'training_resources': resources,
            'training_peak_sampled_vram_gib': resources['gpu']['maximum']['device_memory_used_bytes']/1024**3,
            'end_to_end_peak_sampled_vram_gib': trial['resources']['gpu']['maximum']['device_memory_used_bytes']/1024**3,
            'cuda_peak_allocated_gib': trial['cuda_peak_allocated_bytes']/1024**3,
            'cuda_peak_reserved_gib': trial['cuda_peak_reserved_bytes']/1024**3})
    rows = []
    baseline = next(row for row in report['summary'] if row['batch_size'] == 1)
    for row in report['summary']:
        batch = row['batch_size']
        group = [d for d in details if d['batch_size'] == batch]
        row = dict(row)
        for key in ('training_peak_sampled_vram_gib', 'cuda_peak_allocated_gib', 'cuda_peak_reserved_gib'):
            row[key] = spread([d[key] for d in group])
        row['test_quality'] = report['quality'][str(batch)]
        row['training_speedup_vs_fresh_batch_one'] = (
            row['training_input_tokens_per_second']['median']/baseline['training_input_tokens_per_second']['median'])
        row['end_to_end_speedup_vs_fresh_batch_one'] = baseline['execution_seconds']['median']/row['execution_seconds']['median']
        rows.append(row)
    checks = [t['historical_batch_one_equivalence'] for t in report['trials'] if t['batch_size'] == 1]
    repeat_checks = [t['repeat_equivalence'] for t in report['trials'] if 'repeat_equivalence' in t]
    result = {'status': 'AUDITED', 'rows': rows, 'training_window_details': details,
        'audit': {'trials': len(report['trials']), 'all_trials_succeeded': all(t['status']=='SUCCEEDED' for t in report['trials']),
            'historical_batch_one_checks': len(checks), 'historical_batch_one_exact': all(c['exact'] for c in checks),
            'repeat_checks': len(repeat_checks), 'repeat_weights_exact': all(c['exact'] for c in repeat_checks),
            'all_bases_unchanged': all(t['frozen_base_unchanged'] for t in report['trials']),
            'every_batch_has_expected_repeats': all(r['repeats']==report['repeats'] for r in rows),
            'all_quality_splits_full': all(q['count']==100 for q in report['quality'].values())},
        'concurrency_reference': {'path': str(concurrency_root), 'levels': concurrent['summary']['levels']},
        'analysis_amendment': 'End-to-end VRAM peaks sometimes include an exiting previous Ray worker. '
            'Use training-window whole-device samples and worker-local CUDA peak counters for batch memory comparison. '
            'Raw end-to-end memory measurements are retained; this correction was made after inspecting timing data.',
        'interpretation': 'Fixed examples and learning rate, fewer updates at larger batches. '
            'One adapter versus multiple independent adapters; compare normalized throughput, not raw makespan as equal work. '
            'Historical concurrency timings come from an earlier session; no statistical quality guarantee.'}
    (root/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    return result['audit']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True)
    parser.add_argument('--concurrency-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.experiment_dir), Path(args.concurrency_dir)), indent=2))
