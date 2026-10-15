"""Run one local platform training slot; keep this process alive while jobs execute."""
import argparse
import json
import math
from pathlib import Path
import time

from post_training.database import make_engine, make_session_factory
from post_training.orchestration import claim_next_job, execute_claimed_job, prepare_recovery


def main():
    parser = argparse.ArgumentParser(description="Dispatch PostgreSQL jobs to local Ray")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--drain", action="store_true", help="Exit when the queue becomes empty")
    parser.add_argument("--gpu-hourly-rate-usd", type=float, help="Optional configured GPU allocation rate")
    parser.add_argument("--resume-job", help="Explicitly requeue an interrupted job from its latest complete checkpoint")
    parser.add_argument("--interrupt-after-checkpoint", type=int,
                        help="Recovery demo: interrupt after a completed checkpoint at/after this global step")
    args = parser.parse_args()
    if args.interrupt_after_checkpoint is not None and args.interrupt_after_checkpoint <= 0:
        parser.error("Interruption checkpoint step must be positive")
    if args.gpu_hourly_rate_usd is not None and (
            not math.isfinite(args.gpu_hourly_rate_usd) or args.gpu_hourly_rate_usd < 0):
        parser.error("GPU hourly rate must be finite and nonnegative")
    import ray
    engine = make_engine()
    if engine.dialect.name != "postgresql":
        parser.error("Dispatcher requires a PostgreSQL DATABASE_URL")
    sessions = make_session_factory(engine)
    if args.resume_job:
        print(json.dumps(prepare_recovery(sessions, args.resume_job)), flush=True)
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
            status = execute_claimed_job(sessions, claimed, args.gpu_hourly_rate_usd,
                                         args.interrupt_after_checkpoint)
            print(json.dumps({"job_id": claimed["job_id"], "status": status}), flush=True)
            if status == "INTERRUPTED":
                break
    finally:
        ray.shutdown()
        engine.dispose()


if __name__ == "__main__":
    main()
