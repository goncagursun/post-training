"""Finish a controlled benchmark with trajectory checks and separate quality evaluation."""

def main():
    import argparse
    from pathlib import Path
    import json
    import subprocess
    import sys
    import time
    parser=argparse.ArgumentParser(description='Verify cross-mode trajectories and evaluate every controlled adapter before publishing a comparison.')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    root=Path(args.output_dir).expanduser().resolve()
    report=json.loads((root/'report.json').read_text())
    if report['status'] != 'TRAINING_COMPLETE_ISOLATION_PASSED':
        raise RuntimeError('Training must complete and pass solo/shared isolation first')
    if (root/'comparison.json').exists():
        raise FileExistsError('Comparison already exists; retain the completed experiment')
    if report['status']=='FAILED':raise RuntimeError('Controlled training failed; no comparison will be published')
    if not report['isolation']['passed']:raise RuntimeError('Isolation did not pass')
    sys.path.insert(0,str(repo/'scripts'))
    from verify_controlled_isolation import compare_weights
    checks={}
    for name in ('tenant-a','tenant-b'):
        reference=next(j for j in report['modes']['fifo']['jobs'] if j['tenant_id']==name)
        checks[name]={}
        for mode in ('shared','independent'):
            job=next(j for j in report['modes'][mode]['jobs'] if j['tenant_id']==name)
            check=compare_weights(reference['adapter_dir'],job['adapter_dir'])
            check['initial_adapter_equal']=reference['initial_adapter_sha256']==job['initial_adapter_sha256']
            check['batch_order_equal']=reference['order_sha256']==job['order_sha256']
            checks[name][mode]=check
    (root/'trajectory_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(c['passed'] and c['initial_adapter_equal'] and c['batch_order_equal'] for v in checks.values() for c in v.values()):
        raise RuntimeError('Cross-mode trajectories differ beyond prespecified tolerances; investigate before publishing')
    quality={}
    for mode in ('shared','fifo','independent'):
        quality[mode]={}
        for job in report['modes'][mode]['jobs']:
            name=job['tenant_id']
            output=root/mode/f'{name}-test-evaluation.json'
            started=time.perf_counter()
            with (root/mode/f'{name}-evaluation.log').open('x') as log:
                subprocess.run([sys.executable,str(repo/'scripts/evaluate_intents.py'),'--manifest',job['manifest_path'],
                    '--split','test','--adapter-dir',job['adapter_dir'],'--device','cuda','--output',str(output)],
                    cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
            evaluation=json.loads(output.read_text())
            quality[mode][name]={'accuracy':evaluation['adapter']['accuracy'],
                'invalid_label_rate':evaluation['adapter']['invalid_label_rate'],
                'count':evaluation['adapter']['count'],'separate_evaluation_seconds':time.perf_counter()-started}
            print(mode,name,'accuracy',quality[mode][name]['accuracy'],flush=True)
            (root/'quality_report.json').write_text(json.dumps(quality,indent=2)+'\n')
    rows=[]
    for mode in ('fifo','independent','shared'):
        summary=report['modes'][mode]
        metrics=summary['execution_window_gpu_measurements']
        rows.append({'mode':mode,'execution_seconds':summary['execution_seconds'],
            'aggregate_input_tokens_per_second':summary['aggregate_input_tokens_per_second'],
            'gpu_seconds':summary['execution_seconds'],'estimated_cost_usd':summary['estimated_cost_usd'],
            'mean_gpu_utilization_percent':metrics['mean'].get('gpu_utilization_percent'),
            'peak_sampled_vram_gib':metrics['maximum'].get('device_memory_used_bytes',0)/1024**3,
            'gpu_samples':summary['gpu_sample_count'],'quality':quality[mode]})
    comparison={'status':'VERIFIED','isolation_passed':True,'trajectory_checks_passed':True,
        'repetitions':1,'timing_scope':'first task submission to last observed result, all modes',
        'model_revision':report['model_revision'],'results':rows,
        'limitations':['one fixed-order repetition per mode','whole-device memory includes contexts, caches, activations and weights',
        'cost is elapsed GPU allocation time multiplied by hourly price','no SM or Tensor Core metrics available']}
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
