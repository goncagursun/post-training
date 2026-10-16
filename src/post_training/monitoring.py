"""Single-host GPU monitoring, independent of worker and dispatcher lifetimes."""
import csv
import json
import math
from pathlib import Path
import subprocess
import time

from post_training.telemetry import timestamp


GPU_FIELDS = "uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def query_gpus():
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3, check=True)
    rows = []
    for values in csv.reader(output.stdout.splitlines(), skipinitialspace=True):
        if len(values) != 7:
            raise ValueError("Unexpected nvidia-smi output")
        uuid, name, *metrics = values
        def number(value):
            try:
                result = float(value)
                return result if math.isfinite(result) else None
            except ValueError:
                return None
        util, used, total, temperature, power = map(number, metrics)
        rows.append({"gpu_uuid": uuid.strip(), "name": name.strip(),
                     "gpu_utilization_percent": util,
                     "device_memory_used_bytes": int(used * 1024**2) if used is not None else None,
                     "device_memory_total_bytes": int(total * 1024**2) if total is not None else None,
                     "temperature_celsius": temperature, "power_watts": power})
    if not rows:
        raise ValueError("No GPUs returned")
    return rows


def active_reservation(sessions):
    """Only the platform's single CUDA slot; this does not identify GPU processes."""
    from sqlalchemy import select
    from post_training.database import Job, JobExecution
    with sessions() as session:
        rows = session.execute(select(Job, JobExecution).join(JobExecution).where(
            Job.status.in_(("CLAIMED", "RUNNING", "CANCEL_REQUESTED")))).all()
        if len(rows) > 1:
            raise ValueError("Multiple active slots violate monitor assumptions")
        if not rows or rows[0][1].config["device"] != "cuda":
            return None
        job, execution = rows[0]
        return {"tenant_id": job.tenant_id, "job_id": job.id,
                "run_dir": execution.run_dir, "status": job.status}


class Measurements:
    """Bounded-memory aggregates. Missing observations never count as zero."""
    def __init__(self):
        self.count = 0
        self.sums = {}
        self.counts = {}
        self.maxima = {}
        self.zero_utilization = 0

    def add(self, sample):
        self.count += 1
        for key in ("gpu_utilization_percent", "device_memory_used_bytes",
                    "device_memory_total_bytes", "temperature_celsius", "power_watts"):
            value = sample.get(key)
            if value is None:
                continue
            self.sums[key] = self.sums.get(key, 0) + value
            self.counts[key] = self.counts.get(key, 0) + 1
            self.maxima[key] = max(self.maxima.get(key, value), value)
        self.zero_utilization += sample.get("gpu_utilization_percent") == 0

    def report(self):
        valid = self.counts.get("gpu_utilization_percent", 0)
        return {"sample_count": self.count, "valid_sample_counts": dict(self.counts),
                "mean": {k: v / self.counts[k] for k, v in self.sums.items()},
                "maximum": dict(self.maxima),
                "zero_utilization_sample_fraction": self.zero_utilization / valid if valid else None}


class PlatformMonitor:
    def __init__(self, directory, gpu_hourly_rate_usd=None, sessions=None, gpu_uuid=None):
        if gpu_hourly_rate_usd is not None and (
                not math.isfinite(gpu_hourly_rate_usd) or gpu_hourly_rate_usd < 0):
            raise ValueError("Hourly rate must be finite and nonnegative")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rate = gpu_hourly_rate_usd
        self.sessions = sessions
        self.selected_uuid = gpu_uuid
        self.started_at = timestamp()
        self.started = time.monotonic()
        self.previous = None
        self.gpus = {}
        self.tenants = {}
        self.jobs = {}
        self.errors = 0
        self.last_error = None
        self.reserved_seconds = 0.0
        self.unreserved_seconds = 0.0
        self.unknown_seconds = 0.0

    def _event(self, record):
        with (self.directory / "gpu_samples.jsonl").open("a") as file:
            file.write(json.dumps({"timestamp": timestamp(), **record}) + "\n")

    def _account(self, now):
        if self.previous is None:
            return
        before, inventory, owner, attribution = self.previous
        seconds = max(0.0, now - before)
        for sample in inventory:
            gpu = self.gpus[sample["gpu_uuid"]]
            gpu["observed_allocation_seconds"] += seconds
            if sample["gpu_uuid"] != attribution or owner == "unknown":
                self.unknown_seconds += seconds
            elif owner is None:
                self.unreserved_seconds += seconds
            else:
                self.reserved_seconds += seconds
                for bucket in (self.tenants[owner["tenant_id"]], self.jobs[owner["run_dir"]]):
                    bucket["reserved_gpu_seconds"] += seconds

    def poll(self):
        now = time.monotonic()
        self._account(now)
        # A query gap is unknown coverage, not a zero or an idle sample.
        self.previous = None
        try:
            inventory = query_gpus()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.errors += 1
            self.last_error = type(exc).__name__
            self._event({"event": "gpu_query_failed", "error_type": self.last_error})
            return self.write_summary()
        attribution = self.selected_uuid
        if attribution is None and len(inventory) == 1:
            attribution = inventory[0]["gpu_uuid"]
        if attribution not in {s["gpu_uuid"] for s in inventory}:
            attribution = None
        owner = "unknown"
        if self.sessions is not None:
            try:
                owner = active_reservation(self.sessions)
            except Exception as exc:
                self.errors += 1
                self.last_error = type(exc).__name__
                self._event({"event": "reservation_query_failed", "error_type": self.last_error})
        if isinstance(owner, dict) and attribution is not None:
            self.tenants.setdefault(owner["tenant_id"], {"reserved_gpu_seconds": 0.0, "measurements": Measurements()})
            self.jobs.setdefault(owner["run_dir"], {**owner, "reserved_gpu_seconds": 0.0, "measurements": Measurements()})
        for sample in inventory:
            gpu = self.gpus.setdefault(sample["gpu_uuid"], {"name": sample["name"],
                "observed_allocation_seconds": 0.0, "measurements": Measurements()})
            gpu["measurements"].add(sample)
            reservation = owner if sample["gpu_uuid"] == attribution and isinstance(owner, dict) else None
            if reservation is not None:
                self.tenants[owner["tenant_id"]]["measurements"].add(sample)
                self.jobs[owner["run_dir"]]["measurements"].add(sample)
            self._event({"event": "gpu_sample", **sample, "reservation": reservation,
                         "attribution_known": owner != "unknown" and sample["gpu_uuid"] == attribution})
        self.previous = (now, inventory, owner, attribution)
        return self.write_summary()

    def write_summary(self, stopped=False):
        def report(bucket, seconds_key):
            result = {k: v for k, v in bucket.items() if k != "measurements"}
            result["measurements"] = bucket["measurements"].report()
            result["estimated_cost_usd"] = bucket[seconds_key] / 3600 * self.rate if self.rate is not None else None
            return result
        elapsed = time.monotonic() - self.started
        allocated = sum(g["observed_allocation_seconds"] for g in self.gpus.values())
        result = {"schema_version": 1, "started_at": self.started_at, "updated_at": timestamp(),
            "status": "STOPPED" if stopped else "RUNNING", "monitor_seconds": elapsed,
            "scope": "visible whole GPUs during this monitor session; not process utilization or provider billing",
            "attribution": "single CUDA slot reservation sampled from PostgreSQL; whole-device activity may include unrelated processes",
            "time_accounting": "previous observation held until next poll; query gaps excluded; boundary error up to polling interval",
            "gpu_hourly_rate_usd": self.rate, "query_error_count": self.errors,
            "last_error_type": self.last_error,
            "observed_allocation_gpu_seconds": allocated,
            "reserved_gpu_seconds": self.reserved_seconds,
            "unreserved_gpu_seconds": self.unreserved_seconds,
            "unattributed_gpu_seconds": self.unknown_seconds,
            "estimated_cost_usd": allocated / 3600 * self.rate if self.rate is not None else None,
            "estimated_unreserved_cost_usd": self.unreserved_seconds / 3600 * self.rate if self.rate is not None else None,
            "gpus": {k: report(v, "observed_allocation_seconds") for k, v in self.gpus.items()},
            "tenants": {k: report(v, "reserved_gpu_seconds") for k, v in self.tenants.items()},
            "attempts": {k: report(v, "reserved_gpu_seconds") for k, v in self.jobs.items()}}
        temporary = self.directory / ".summary.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(self.directory / "summary.json")
        return result

    def stop(self):
        self._account(time.monotonic())
        self.previous = None
        return self.write_summary(stopped=True)
