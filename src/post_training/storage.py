"""Platform-owned artifact paths, never customer-provided output paths."""
from pathlib import Path
import re


def job_run_dir(data_root: str | Path, tenant_id: str, job_id: str, attempt: int = 1) -> Path:
    for value in (tenant_id, job_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Unsafe tenant or job identifier")
    if attempt < 1:
        raise ValueError("Attempt must be positive")
    root = Path(data_root).expanduser().resolve()
    path = root / "customers" / tenant_id / "jobs" / job_id / "attempts" / f"{attempt:03d}"
    if not path.resolve().is_relative_to(root):
        raise ValueError("Output path escapes data root")
    return path
