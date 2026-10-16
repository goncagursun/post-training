from dataclasses import asdict
import json
from pathlib import Path

import pytest

from post_training.bitext import prepare_bitext
from post_training.concurrent_scenario import load_scenario, summarize_jobs
from post_training.config import SFTJobConfig
from test_bitext import source_rows


def test_scenario_reuses_exact_baseline_data_and_configuration(tmp_path):
    manifests = prepare_bitext(source_rows(), tmp_path/'datasets', 20, 10, 10)
    tenants = []
    for letter, manifest in zip(('a', 'b'), manifests):
        run = tmp_path/letter
        (run/'adapter').mkdir(parents=True)
        (run/'dataset.json').write_text(manifest.read_text())
        (run/'config.json').write_text(json.dumps(asdict(SFTJobConfig(num_train_epochs=3))))
        tenants.append({'tenant_id': f'tenant-{letter}', 'manifest_path': str(manifest),
                        'config_path': str(run/'config.json'), 'baseline_adapter_dir': str(run/'adapter')})
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps({'tenants': tenants}))
    prepared = load_scenario(spec)
    assert len(prepared) == 2
    assert len(prepared[0]['examples']) == 20
    assert prepared[0]['config']['num_train_epochs'] == 3
    original = manifests[0].read_text()
    altered = json.loads(original)
    altered['seed'] = 999
    manifests[0].write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='Dataset differs'):
        load_scenario(spec)
    manifests[0].write_text(original)
    other_config = tmp_path/'other.json'
    other_config.write_text(json.dumps(asdict(SFTJobConfig(num_train_epochs=1))))
    tenants[0]['config_path'] = str(other_config)
    spec.write_text(json.dumps({'tenants': tenants}))
    with pytest.raises(ValueError, match='Configuration differs'):
        load_scenario(spec)


def result_job(tmp_path, name, start, end, pid, gpu='GPU-a'):
    run = tmp_path/name
    run.mkdir()
    events = [{'timestamp': t, 'event': event, 'phase': 'training'} for t, event in
              [(start, 'phase_started'), (end, 'phase_finished')]]
    (run/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return {'status': 'SUCCEEDED', 'run_dir': str(run), 'result': {'telemetry': {
        'pid': pid, 'gpu_activity': {'gpu_uuid': gpu},
        'training': {'input_tokens': 100, 'samples': 10}}}}


def test_overlap_and_single_gpu_cost_not_summed_per_worker(tmp_path):
    a = result_job(tmp_path, 'a', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:10+00:00', 10)
    b = result_job(tmp_path, 'b', '2026-01-01T00:00:02+00:00', '2026-01-01T00:00:12+00:00', 11)
    report = summarize_jobs([a, b], 20, 3.6)
    assert report['concurrency_verified']
    assert report['training_overlap_seconds'] == 8
    assert report['aggregate_input_tokens_per_second'] == 10
    assert report['aggregate_samples_per_second'] == 1
    assert report['estimated_execution_cost_usd'] == pytest.approx(.02)
    b['result']['telemetry']['gpu_activity']['gpu_uuid'] = 'GPU-b'
    assert not summarize_jobs([a, b], 20, None)['concurrency_verified']


def test_sequential_training_is_not_reported_as_concurrent(tmp_path):
    a = result_job(tmp_path, 'a', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:10+00:00', 10)
    b = result_job(tmp_path, 'b', '2026-01-01T00:00:10+00:00', '2026-01-01T00:00:20+00:00', 11)
    report = summarize_jobs([a, b], 20, None)
    assert report['training_overlap_seconds'] == 0
    assert not report['concurrency_verified']
