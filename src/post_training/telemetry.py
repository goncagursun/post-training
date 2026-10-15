"""Small per-run measurements; no monitoring service or GPU import at module load."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def process_peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def cost_estimate(seconds, device, hourly_rate):
    return {"gpu_seconds": seconds if device == "cuda" else 0.0,
            "hourly_rate_usd": hourly_rate,
            "estimated_cost_usd": (0.0 if device == "cpu" else
                seconds / 3600 * hourly_rate if hourly_rate is not None else None)}


class RunTelemetry:
    def __init__(self, directory, context=None):
        self.directory = Path(directory)
        self.context = context or {}
        self.started_at = timestamp()
        self.start = time.perf_counter()
        self.phases = {}
        self.values = {}
        self._lock = threading.Lock()

    def event(self, name, **values):
        record = {"timestamp": timestamp(), **self.context, "event": name, **values}
        with self._lock:
            with (self.directory / "events.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")

    @contextmanager
    def phase(self, name, synchronize=None):
        if synchronize:
            synchronize()
        start = time.perf_counter()
        self.event("phase_started", phase=name)
        try:
            yield
        finally:
            if synchronize:
                synchronize()
            seconds = time.perf_counter() - start
            self.phases[name] = self.phases.get(name, 0.0) + seconds
            self.event("phase_finished", phase=name, seconds=seconds)

    def finish(self, status, error_type=None):
        summary = {"schema_version": 1, "status": status, "context": self.context,
                   "pid": os.getpid(), "run_started_at": self.started_at, "run_finished_at": timestamp(),
                   "run_seconds": time.perf_counter() - self.start,
                   "phase_seconds": self.phases,
                   "process_lifetime_peak_rss_bytes": process_peak_rss_bytes(),
                   **self.values}
        submitted = self.context.get("ray_submission_requested_at")
        if submitted:
            summary["ray_submission_to_run_start_seconds"] = (
                datetime.fromisoformat(self.started_at) - datetime.fromisoformat(submitted)).total_seconds()
        if error_type:
            summary["error_type"] = error_type
        self.event("run_finished", status=status, error_type=error_type)
        (self.directory / "telemetry.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


class NvidiaSampler:
    """Optional whole-device activity samples, identified by CUDA device UUID."""
    def __init__(self, recorder, uuid, interval=1.0):
        self.recorder = recorder
        self.uuid = str(uuid) if uuid else None
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.samples = []
        self.status = "unavailable"

    def start(self):
        if not self.uuid or not shutil.which("nvidia-smi"):
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.run([
                    "nvidia-smi", f"--id={self.uuid}",
                    "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, timeout=3, check=True)
                util, memory = [float(value.strip()) for value in output.stdout.strip().split(",")]
                sample = {"gpu_utilization_percent": util, "device_memory_used_bytes": int(memory * 1024**2)}
                self.samples.append(sample)
                self.recorder.event("gpu_sample", **sample)
                self.status = "available"
            except (OSError, ValueError, subprocess.SubprocessError):
                self.status = "sampling_failed"
                return
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)
        return {"status": self.status, "sample_count": len(self.samples),
                "scope": "whole device; includes other processes",
                "interval_seconds": self.interval,
                "mean_utilization_percent": (
                    sum(s["gpu_utilization_percent"] for s in self.samples)/len(self.samples)
                    if self.samples else None),
                "max_device_memory_used_bytes": (
                    max(s["device_memory_used_bytes"] for s in self.samples) if self.samples else None)}
