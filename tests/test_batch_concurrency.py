import json

from post_training.batch_concurrency import matrix_plan, update_token_counts
from post_training.concurrency_sweep import steady_window


def test_matrix_randomization_has_full_factorial_and_fresh_controls_each_block():
    plan = matrix_plan()
    expected = {(b, n) for b in (4, 8) for n in (1, 2, 4, 8)} | {(1, 1), (1, 8)}
    assert plan == matrix_plan()
    assert len(plan) == 30
    for repeat in (1, 2, 3):
        cells = [(r['batch_size'], r['n']) for r in plan if r['repeat'] == repeat]
        assert len(cells) == len(set(cells))
        assert set(cells) == expected


def test_overlap_counts_useful_tokens_in_partial_physical_batch(tmp_path):
    # Five samples, two batches of four/one; the last update has only 11 tokens.
    counts = update_token_counts({'tenant': [2, 3, 5, 7, 11]}, 4)
    directory = tmp_path/'worker'
    directory.mkdir()
    records = [
        {'timestamp': '2026-01-01T00:00:00+00:00', 'event': 'phase_started', 'phase': 'training'},
        {'timestamp': '2026-01-01T00:00:01+00:00', 'event': 'tenant_update', 'step': 1},
        {'timestamp': '2026-01-01T00:00:02+00:00', 'event': 'tenant_update', 'step': 2},
        {'timestamp': '2026-01-01T00:00:03+00:00', 'event': 'phase_finished', 'phase': 'training'},
    ]
    (directory/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    window = steady_window([{'tenant_id':'tenant','worker_dir':str(directory)}], counts)
    assert window['input_tokens'] == 28
    assert window['completed_updates_per_tenant'] == {'tenant': 2}
