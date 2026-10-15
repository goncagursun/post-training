"""Completed checkpoints and single-host, fenced attempt recovery."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path


class TrainingInterrupted(RuntimeError):
    """Controlled interruption used to demonstrate checkpoint recovery."""


def fingerprint(config, examples, validation, model_revision):
    from dataclasses import asdict
    data = {"config": asdict(config), "examples": examples, "validation": validation,
            "resolved_model_revision": model_revision}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_checkpoint(path, run_fingerprint):
    path = Path(path)
    required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                "scheduler.pt", "trainer_state.json", "rng_state.pth"]
    if not all((path / name).is_file() for name in required):
        raise ValueError("Checkpoint is missing required training state")
    files = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name != "complete.json"}
    data = {"fingerprint": run_fingerprint, "files": files,
            "global_step": json.loads((path / "trainer_state.json").read_text())["global_step"]}
    temporary = path / ".complete.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path / "complete.json")


def validate_checkpoint(path, expected_fingerprint=None):
    path = Path(path).resolve()
    marker = json.loads((path / "complete.json").read_text())
    if expected_fingerprint is not None and marker["fingerprint"] != expected_fingerprint:
        raise ValueError("Checkpoint model, data, or training configuration differs")
    required = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                "scheduler.pt", "trainer_state.json", "rng_state.pth"}
    if not required.issubset(marker["files"]):
        raise ValueError("Incomplete checkpoint manifest")
    for name, digest in marker["files"].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError("Checkpoint checksum mismatch")
    state = json.loads((path / "trainer_state.json").read_text())
    if state["global_step"] != marker["global_step"]:
        raise ValueError("Checkpoint step mismatch")
    return marker


def latest_checkpoint(run_dir):
    candidates = sorted((Path(run_dir) / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        reverse=True)
    for path in candidates:
        try:
            validate_checkpoint(path)
            return path.resolve()
        except (OSError, ValueError, KeyError):
            continue
    raise ValueError("No completed, checksum-valid checkpoint is available")


@contextmanager
def job_lock(run_dir):
    """Nonblocking OS lock shared by recovery and workers on this filesystem."""
    job_dir = Path(run_dir).resolve().parent.parent
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / ".execution.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A worker still holds the job lock; stop it before recovery") from None
        try:
            yield job_dir
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def fence_attempt(run_dir):
    with job_lock(run_dir) as job_dir:
        temporary = job_dir / ".attempt.tmp"
        temporary.write_text(str(Path(run_dir).resolve()))
        temporary.replace(job_dir / ".current-attempt")


@contextmanager
def worker_guard(run_dir):
    with job_lock(run_dir) as job_dir:
        if (job_dir / ".current-attempt").read_text() != str(Path(run_dir).resolve()):
            raise RuntimeError("Stale attempt refused")
        yield
