"""Audit and plot a completed independent batch/concurrency matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def analyze(root):
    report = json.loads((root/'report.json').read_text())
    if report['status'] != 'COMPLETE':
        raise ValueError('Training and held-out quality must be complete')
    output = root/'analysis.json'
    if output.exists():
        raise FileExistsError(output)
    trials = report['trials']
    successful = [t for t in trials if t['status'] == 'SUCCEEDED']
    jobs = [j for t in successful for j in t['jobs']]
    quality = report['quality']
    for path, expected in report['historical_report_hashes'].items():
        if hashlib.sha256((Path(path)/'report.json').read_bytes()).hexdigest() != expected:
            raise ValueError('Historical reference report changed during the experiment')
    from verify_controlled_isolation import compare_weights
    prior = json.loads((Path(report['prior_batch_counterfactual'])/'report.json').read_text())
    reference_controls = {}
    for batch in (4, 8):
        fresh = report['references'][str(batch)]['sweep-tenant-01']
        old = next(t['job'] for t in prior['trials'] if t['batch_size']==batch)
        reference_controls[str(batch)] = compare_weights(old['adapter_dir'], fresh['adapter_dir'])
    golden = {str(batch): json.loads((root/'quality'/f'b{batch}-sweep-tenant-01.json').read_text())[
        'historical_prediction_equivalence'] for batch in (1, 4, 8)}
    audit = {'planned_trials': len(report['plan']), 'attempted_trials': len(trials),
        'successful_trials': len(successful), 'failed_trials': len(trials)-len(successful),
        'trial_adapters': len(jobs),
        'all_trial_adapters_exact': all(j['solo_equivalence']['exact'] for j in jobs),
        'all_initial_states_orders_and_tokens_equal': all(j['solo_equivalence']['initial_adapter_equal'] and
            j['solo_equivalence']['order_equal'] and j['solo_equivalence']['token_budget_equal'] for j in jobs),
        'all_frozen_bases_unchanged': all(j['frozen_base_unchanged'] for j in jobs),
        'all_successful_overlaps_valid': all(t['steady_window']['adequate_overlap'] for t in successful),
        'gpu_query_errors_successful_trials': sum(t['resources']['gpu_query_errors']+
            t['training_resources']['gpu_query_errors'] for t in successful),
        'quality_adapters': quality['completed_adapters'],
        'all_24_quality_evaluations_complete': quality['status']=='COMPLETE' and quality['completed_adapters']==24,
        'all_test_sets_full': all(q['count']==100 for group in quality['batches'].values() for q in group.values())}
    audit['fresh_batch_references_match_historical_weights'] = all(c['exact'] for c in reference_controls.values())
    audit['historical_generated_predictions_match'] = all(golden.values())
    def spread(values):
        return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    rows = []
    for batch, summary in report['summary'].items():
        for row in summary['levels']:
            row = {'batch_size': int(batch), 'is_endpoint_control': batch=='1', **row}
            group = [t for t in successful if t['batch_size']==int(batch) and t['n']==row['n']]
            if group:
                row['training_cpu_quota_percent'] = spread([
                    t['training_resources']['host']['mean_cpu_quota_percent'] for t in group])
                row['training_gpu_utilization_percent'] = spread([
                    t['training_resources']['gpu']['mean']['gpu_utilization_percent'] for t in group])
                row['training_device_peak_gib'] = spread([
                    t['training_resources']['gpu']['maximum']['device_memory_used_bytes']/1024**3 for t in group])
                row['worst_tenant_median_training_slowdown'] = max(
                    tenant['training_slowdown']['median'] for tenant in row['tenants'].values())
                scores = [quality['batches'][batch][name]['accuracy'] for name in row['tenants']]
                row['test_accuracy_across_tenants'] = {**spread(scores), 'mean': statistics.mean(scores)}
            rows.append(row)
    result = {'status': 'AUDITED', 'audit': audit, 'rows': rows,
        'historical_reference_controls': reference_controls, 'historical_prediction_controls': golden,
        'capacity_by_batch': {b:s['largest_tested_n_meeting_slowdown_target']
            for b,s in report['summary'].items() if b!='1'},
        'limitations': report['limitations'] + [
            'Batch sizes were chosen after earlier inspection of this same 100-example test set; '
            'quality is descriptive on reused test data, not independent confirmatory evidence'],
        'measurement_scope': 'Synchronized training starts; end-to-end latency includes loading and barrier wait; '
            'training throughput and CPU/GPU metrics use the all-worker phase intersection. '
            'Quality is inherited only through exact solo adapter equality.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def plot(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {1:'#6b7280', 4:'#2563eb', 8:'#d97706'}
    for batch in (1, 4, 8):
        rows = sorted([r for r in result['rows'] if r['batch_size']==batch and r['eligible'] and r['complete']], key=lambda r:r['n'])
        if not rows:
            continue
        xs = [r['n'] for r in rows]
        label = f'Batch {batch}' + (' endpoint controls' if batch==1 else '')
        def line(ax, key):
            vals = [r[key] for r in rows]
            ys = [v['median'] for v in vals]
            ax.errorbar(xs, ys, yerr=[[v['median']-v['min'] for v in vals],
                [v['max']-v['median'] for v in vals]], marker='o', capsize=3,
                linestyle='None' if batch==1 else '-', color=colors[batch], label=label)
        line(axes[0,0], 'steady_input_tokens_per_second')
        line(axes[0,1], 'aggregate_input_tokens_per_second')
        line(axes[0,2], 'execution_seconds')
        axes[1,0].plot(xs, [r['worst_tenant_median_completion_slowdown'] for r in rows],
            marker='o', linestyle='None' if batch==1 else '-', color=colors[batch], label=label)
        line(axes[1,1], 'training_device_peak_gib')
        line(axes[1,2], 'training_cpu_quota_percent')
    titles = [('All-worker training overlap','Useful input tokens/s'),
        ('End-to-end platform throughput','Useful input tokens/s'),('Time until all jobs finish','Seconds'),
        ('Worst tenant median completion slowdown','Latency / own solo reference'),
        ('Sampled peak VRAM in training overlap','GiB'),('Container CPU in training overlap','Usage / quota (%)')]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set(title=title, ylabel=ylabel, xlabel='Concurrent tenants N')
        ax.set_xscale('log', base=2)
        ax.set_xticks([1,2,4,8], labels=['1','2','4','8'])
        ax.grid(alpha=.2)
    axes[1,0].axhline(1.2, color='#dc2626', linestyle='--')
    axes[1,2].axhline(100, color='gray', linestyle=':')
    axes[0,0].legend()
    fig.suptitle('Independent batching + concurrency: 500 examples/job, three repeats; medians and repeat ranges\n'
        'Synchronized training starts; one CPU thread/worker; batch one has endpoint controls only', fontsize=12)
    destination = root/'batch_concurrency.png'
    if destination.exists():
        raise FileExistsError(destination)
    fig.savefig(destination, dpi=160)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    result = analyze(root)
    print(json.dumps(result['audit'], indent=2))
    print(plot(root, result))
