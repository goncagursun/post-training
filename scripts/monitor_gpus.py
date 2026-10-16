"""Monitor platform GPUs continuously, including when no training jobs are running."""
import argparse
import json
import math
import signal
import threading

from post_training.monitoring import PlatformMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New administrator-only monitoring session directory")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds; otherwise run until Ctrl+C")
    parser.add_argument("--gpu-hourly-rate-usd", type=float, help="Configured rate per visible allocated GPU")
    parser.add_argument("--with-database", action="store_true", help="Attribute the single CUDA slot using DATABASE_URL")
    parser.add_argument("--gpu-uuid", help="GPU assigned to the platform slot; required for attribution on multi-GPU hosts")
    args = parser.parse_args()
    for name in ("interval", "duration"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if args.gpu_hourly_rate_usd is not None and (not math.isfinite(args.gpu_hourly_rate_usd) or args.gpu_hourly_rate_usd < 0):
        parser.error("GPU hourly rate must be finite and nonnegative")
    engine = None
    sessions = None
    if args.with_database:
        from post_training.database import make_engine, make_session_factory
        engine = make_engine()
        sessions = make_session_factory(engine)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    import time
    try:
        monitor = PlatformMonitor(args.output_dir, args.gpu_hourly_rate_usd, sessions, args.gpu_uuid)
        deadline = time.monotonic() + args.duration if args.duration else None
        try:
            while not stop.is_set():
                monitor.poll()
                remaining = deadline - time.monotonic() if deadline else args.interval
                if remaining <= 0:
                    break
                stop.wait(min(args.interval, remaining))
        finally:
            summary = monitor.stop()
        print(json.dumps({"summary": str(monitor.directory / "summary.json"),
                          "gpu_count": len(summary["gpus"]), "query_error_count": summary["query_error_count"]}))
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
