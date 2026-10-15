import json
from pathlib import Path

import pytest

from post_training.bitext import INTENTS, build_splits, load_manifest, normalize_request, prepare_bitext
from post_training.datasets import load_registered_dataset
from post_training.intent_evaluation import score_predictions


def source_rows():
    return [{'instruction': f'{label} distinct request {chr(97+i//26)}{chr(97+i%26)}', 'intent': label}
            for label in INTENTS for i in range(50)]


def test_splits_reproducible_balanced_and_disjoint(tmp_path):
    rows = source_rows()
    rows.append({**rows[0], 'instruction': rows[0]['instruction'].upper()+'!'})
    a=build_splits(rows,20,10,10)
    assert a==build_splits(rows,20,10,10)
    seen=set()
    families=set()
    for (tenant,split),examples in a.items():
        assert len(examples)==(20 if split=='train' else 10)
        for label in INTENTS:
            assert sum(r['intent']==label for r in examples)==len(examples)//5
        for row in examples:
            key=normalize_request(row['instruction'])
            family=' '.join(sorted(set(key.split())))
            assert key not in seen
            assert family not in families
            seen.add(key);families.add(family)


def test_verified_loader_and_tenant_checks(tmp_path):
    paths=prepare_bitext(source_rows(),tmp_path/'bundle',20,10,10)
    m,s=load_manifest(paths[0],'tenant-a')
    train,val,metadata=load_registered_dataset(paths[0].as_uri(),'tenant-a')
    assert len(train)==20 and len(val)==10 and metadata==m
    assert 'Allowed labels:' in train[0][0]
    with pytest.raises(ValueError,match='another tenant'):
        load_registered_dataset(paths[0].as_uri(),'tenant-b')
    (paths[0].parent/'train.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='Checksum'):
        load_manifest(paths[0])


def test_bad_budgets_and_label_scoring():
    with pytest.raises(ValueError):
        build_splits(source_rows(),21,10,10)
    rows=[{'intent':'cancel_order'},{'intent':'track_order'},{'intent':'get_refund'}]
    report=score_predictions(rows,[' CANCEL_ORDER ', 'track_order\nextra', 'get_refund'])
    assert report['accuracy']==2/3
    assert report['invalid_label_rate']==1/3
