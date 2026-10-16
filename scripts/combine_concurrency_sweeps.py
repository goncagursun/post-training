"""Combine a completed coarse sweep and its capacity refinement without changing either."""
import argparse
import hashlib
import json
from pathlib import Path

from post_training.concurrency_sweep import summarize_sweep


def combine(coarse_dir, refinement_dir, output):
    roots = [Path(coarse_dir).resolve(), Path(refinement_dir).resolve()]
    reports = [json.loads((root / 'report.json').read_text()) for root in roots]
    coarse, refinement = reports
    if any(report['status'] != 'COMPLETE' for report in reports):
        raise ValueError('Both sweeps must be complete')
    for key in ('engine_sha256', 'model_revision', 'thread_settings',
                'epochs_per_job', 'updates_per_job', 'workload', 'repeats'):
        if coarse[key] != refinement[key]:
            raise ValueError(f'Incompatible sweep control: {key}')
    if Path(refinement['references_source']).resolve() != roots[0]:
        raise ValueError('Refinement must reuse this coarse sweep’s solo references')
    digest = hashlib.sha256((roots[0] / 'report.json').read_bytes()).hexdigest()
    if refinement['references_report_sha256'] != digest:
        raise ValueError('Original reference report changed')
    if set(coarse['levels']) & set(refinement['levels']):
        raise ValueError('Overlapping levels require explicit repeat reconciliation')
    trials = coarse['trials'] + refinement['trials']
    levels = sorted(coarse['levels'] + refinement['levels'])
    limit = coarse['summary']['slowdown_limit']
    plateau_gain = coarse['summary']['plateau_gain_threshold']
    if refinement['summary']['slowdown_limit'] != limit:
        raise ValueError('Different slowdown limits')
    summary = summarize_sweep(trials, levels, limit, plateau_gain)
    # A 10% marginal gain from N=6 to N=7 is not comparable to a 10% gain
    # when doubling N. Preserve the prespecified coarse comparisons.
    coarse_summary = summarize_sweep(coarse['trials'], coarse['levels'], limit, plateau_gain)
    summary['candidate_plateau_intervals'] = coarse_summary['candidate_plateau_intervals']
    summary['plateau_comparison_levels'] = coarse['levels']
    summary['plateau_scope'] = 'Original coarse intervals only; refinement narrows tenant latency capacity'
    jobs = [job for trial in trials for job in trial['jobs']]
    audit = {
        'trials': len(trials), 'adapters': len(jobs),
        'successful_trials': sum(t['status'] == 'SUCCEEDED' for t in trials),
        'all_adapter_weights_exact': all(j['solo_equivalence']['exact'] for j in jobs),
        'all_solo_checks_passed': all(j['solo_equivalence']['passed'] and
            j['solo_equivalence']['initial_adapter_equal'] and
            j['solo_equivalence']['order_equal'] for j in jobs),
        'all_frozen_bases_unchanged': all(j['frozen_base_unchanged'] for j in jobs),
        'gpu_query_errors': sum(t.get('resources', {}).get('gpu_query_errors', 0) for t in trials),
        'all_overlaps_adequate': all(t.get('steady_window', {}).get('adequate_overlap', False) for t in trials),
        'all_levels_have_expected_successful_repeats': all(
            row['eligible'] and row['successful_repeats'] == coarse['repeats'] for row in summary['levels']),
    }
    result = {**coarse, 'levels': levels, 'trials': trials, 'summary': summary,
              'audit': audit, 'source_sweeps': [str(root) for root in roots],
              'source_report_sha256': [hashlib.sha256((root / 'report.json').read_bytes()).hexdigest()
                                       for root in roots]}
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in [('report.json', result), ('summary.json', summary), ('audit.json', audit)]:
        (destination / name).write_text(json.dumps(value, indent=2) + '\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse-dir', required=True)
    parser.add_argument('--refinement-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(combine(args.coarse_dir, args.refinement_dir, args.output_dir), indent=2))
