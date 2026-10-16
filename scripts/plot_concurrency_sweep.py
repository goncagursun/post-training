"""Plot completed sweep results; requires matplotlib (analysis-only dependency)."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',required=True)
    args=parser.parse_args()
    root=Path(args.experiment_dir)
    report=json.loads((root/'report.json').read_text())
    if report['status']!='COMPLETE':raise RuntimeError('Wait for the sweep to complete')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in report['summary']['levels'] if r['eligible']]
    if not rows:raise RuntimeError('No eligible levels to plot')
    ns=[r['n'] for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    def error_line(ax,key,label,color):
        values=[r[key] for r in rows]
        median=[v['median'] for v in values]
        ax.errorbar(ns,median,yerr=[[v['median']-v['min'] for v in values],
                    [v['max']-v['median'] for v in values]],marker='o',capsize=4,label=label,color=color)
    error_line(axes[0,0],'aggregate_input_tokens_per_second','End-to-end','#2563eb')
    error_line(axes[0,0],'steady_input_tokens_per_second','All-job training overlap','#d97706')
    axes[0,0].set(ylabel='Aggregate input tokens/s',title='Throughput: medians and repeat ranges')
    axes[0,0].legend()
    axes[0,1].plot(ns,[r['worst_tenant_median_completion_slowdown'] for r in rows],marker='o',color='#2563eb')
    axes[0,1].axhline(report['summary']['slowdown_limit'],color='#dc2626',linestyle='--',label='Completion slowdown target')
    axes[0,1].set(ylabel='Completion time / own solo reference',title='Worst tenant median slowdown')
    axes[0,1].legend()
    error_line(axes[1,0],'peak_vram_gib','Peak sampled device VRAM','#7c3aed')
    axes[1,0].set(ylabel='GiB',title='Whole-device VRAM')
    host=[]
    for n in ns:
        trials=[t for t in report['trials'] if t['n']==n and t['status']=='SUCCEEDED']
        cpu=[t['resources']['host'].get('mean_cpu_quota_percent') for t in trials]
        cpu=[v for v in cpu if v is not None]
        rss=[t['resources']['host'].get('peak_memory_rss_bytes') for t in trials]
        rss=[v/1024**3 for v in rss if v is not None]
        throttling=[t['resources']['host'].get('cpu_throttled_seconds') for t in trials]
        throttling=[v for v in throttling if v is not None]
        host.append({'n':n,'median_cpu_quota_percent':statistics.median(cpu) if cpu else None,
            'median_peak_container_rss_gib':statistics.median(rss) if rss else None,
            'median_cpu_throttled_seconds':statistics.median(throttling) if throttling else None})
    axes[1,1].plot(ns,[h['median_cpu_quota_percent'] for h in host],marker='o',color='#059669')
    axes[1,1].axhline(100,color='gray',linestyle=':')
    axes[1,1].set(ylabel='Container CPU usage / quota (%)',title='Container CPU (includes other processes)')
    for ax in axes.flat:
        ax.set_xscale('log',base=2)
        ax.set_xticks(ns,labels=[str(n) for n in ns])
        ax.set_xlabel('Concurrent jobs N')
        ax.grid(alpha=.2)
    fig.suptitle(f"Independent concurrency sweep — {report['updates_per_job']} updates/job, {report['repeats']} repeats\nOne CPU thread per worker; homogeneous simulated tenants",fontsize=12)
    output=root/'scaling.png'
    if output.exists():raise FileExistsError(output)
    fig.savefig(output,dpi=160)
    (root/'host_summary.json').write_text(json.dumps(host,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
