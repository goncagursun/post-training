"""Admin registration: customers submit IDs; only admins select storage paths."""
import argparse

from post_training.bitext import load_manifest
from post_training.database import DatasetRecord, Tenant, make_engine, make_session_factory


def register_dataset(session, tenant_id, dataset_id, manifest_path):
    from pathlib import Path
    manifest_path = Path(manifest_path).expanduser().resolve()
    load_manifest(manifest_path, tenant_id)
    if session.get(Tenant, tenant_id) is None:
        raise ValueError("Tenant does not exist")
    if session.get(DatasetRecord, dataset_id) is not None:
        raise ValueError("Dataset ID already exists; register a new version")
    session.add(DatasetRecord(id=dataset_id, tenant_id=tenant_id, uri=manifest_path.as_uri()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if not 1 <= len(args.dataset_id) <= 64:
        parser.error("dataset-id must contain 1 to 64 characters")
    engine = make_engine()
    try:
        with make_session_factory(engine).begin() as session:
            register_dataset(session, args.tenant_id, args.dataset_id, args.manifest)
        print(f"Registered {args.dataset_id} for {args.tenant_id}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
