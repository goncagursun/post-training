"""Plot a completed one-job batch sweep against historical independent concurrency."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True)
    parser.add_argument('--concurrency-dir', required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    batch = json.loads((root/'report.json').read_text())
    concurrent = json.loads((Path(args.concurrency_dir)/'report.json').read_text())
    if batch['status'] != 'COMPLETE' or concurrent['status'] != 'COMPLETE':
        raise ValueError('Both experiments must be complete')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    analysis = json.loads((root/'analysis.json').read_text())
    rows = analysis['rows']
    bs = [r['batch_size'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    def line(ax, key, label, color):
        values = [r[key] for r in rows]
        medians = [v['median'] for v in values]
        ax.errorbar(bs, medians, yerr=[[v['median']-v['min'] for v in values],
            [v['max']-v['median'] for v in values]], marker='o', capsize=4, label=label, color=color)
    line(axes[0, 0], 'training_input_tokens_per_second', 'One job: training phase', '#2563eb')
    line(axes[0, 1], 'execution_seconds', 'One job: end-to-end', '#2563eb')
    axes[0, 0].set(title='Training input throughput; repeat ranges', ylabel='Useful input tokens/s')
    axes[0, 1].set(title='One 500-example adapter job', ylabel='Seconds to result')
    for n, color in [(7, '#d97706'), (8, '#059669')]:
        r = next(r for r in concurrent['summary']['levels'] if r['n'] == n)
        axes[0, 0].axhline(r['steady_input_tokens_per_second']['median'], color=color,
            linestyle='--', label=f'{n} independent batch-one jobs (earlier session)')
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].plot(bs, [100*batch['quality'][str(b)]['accuracy'] for b in bs],
        marker='o', color='#7c3aed')
    axes[1, 0].set(title='Held-out exact-label accuracy (100 examples)', ylabel='Accuracy (%)', ylim=(0, 102))
    line(axes[1, 1], 'training_peak_sampled_vram_gib', 'Training-window device peak', '#dc2626')
    line(axes[1, 1], 'cuda_peak_reserved_gib', 'Worker CUDA peak reserved', '#7c3aed')
    axes[1, 1].set(title='Median peak memory; repeat ranges', ylabel='GiB')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xscale('log', base=2)
        ax.set_xticks(bs, labels=[str(b) for b in bs])
        ax.set_xlabel('Physical batch size (one adapter)')
        ax.grid(alpha=.2)
    fig.suptitle('Batch-size counterfactual: same 500 examples, learning rate and one CPU thread\n'
        'Three timing repeats at one seed; larger batches perform fewer optimizer updates', fontsize=12)
    output = root/'batch_scaling.png'
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    main()
