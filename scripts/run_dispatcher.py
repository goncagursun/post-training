"""Run one local platform training slot; keep this process alive while jobs execute."""
import argparse
import json
from pathlib import Path
import time

from post_training.database import make_engine, make_session_factory
from post_training.orchestration import claim_next_job, execute_claimed_job


def main():
    parser = argparse.ArgumentParser(description="Dispatch PostgreSQL jobs to local Ray")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--drain", action="store_true", help="Exit when the queue becomes empty")
    args = parser.parse_args()
    import ray
    engine = make_engine()
    if engine.dialect.name != "postgresql":
        parser.error("Dispatcher requires a PostgreSQL DATABASE_URL")
    sessions = make_session_factory(engine)
    ray.init(address="local", include_dashboard=False)
    try:
        resources = ray.cluster_resources()
        if resources.get("CPU", 0) < 1 or (args.device == "cuda" and resources.get("GPU", 0) < 1):
            raise RuntimeError(f"Insufficient Ray resources: {resources}")
        if args.device == "cuda":
            import torch
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA/BF16 required. Use --device cpu on your Mac.")
        while True:
            claimed = claim_next_job(sessions, Path(args.data_root).expanduser().resolve(), args.device)
            if claimed is None:
                if args.drain:
                    break
                time.sleep(2)
                continue
            print(json.dumps({"job_id": claimed["job_id"], "status": "CLAIMED"}), flush=True)
            status = execute_claimed_job(sessions, claimed)
            print(json.dumps({"job_id": claimed["job_id"], "status": status}), flush=True)
    finally:
        ray.shutdown()
        engine.dispose()


if __name__ == "__main__":
    main()
