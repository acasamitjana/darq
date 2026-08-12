from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """Return the current local time in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


def read_meta(meta_path: Path) -> dict[str, Any]:
    """Read and return one job metadata file."""
    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    """Write one metadata file atomically.

    The data is first written to a temporary file and then moved over the
    original meta.json. This reduces the risk of leaving a partially written
    JSON file if the process is interrupted.
    """
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = meta_path.with_suffix(".json.tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    temp_path.replace(meta_path)


def append_log(job_dir: Path, message: str) -> None:
    """Append one timestamped message to logs/worker.log."""
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "worker.log"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"[{now_iso()}] {message}\n")


def create_meta(
    *,
    job_id: str,
    pipeline: str,
    inputs: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the initial metadata dictionary for a queued job."""
    meta: dict[str, Any] = {
        "job_id": job_id,
        "pipeline": pipeline,
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "inputs": inputs,
        "outputs": {},
        "error": None,
            "progress": {
        "step": 0,
        "total_steps": 0,
        "message": "Waiting",
        "updated_at": now_iso(),
        },
    }

    if extra:
        meta.update(extra)

    return meta

def update_progress(
    meta_path: Path,
    *,
    step: int,
    total_steps: int,
    message: str,
) -> dict[str, Any]:
    """Update the current processing step of one job."""

    meta = read_meta(meta_path)

    meta["progress"] = {
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "updated_at": now_iso(),
    }

    write_meta(meta_path, meta)

    return meta

def mark_running(meta_path: Path) -> dict[str, Any]:
    """Mark one job as running and persist the updated metadata."""
    meta = read_meta(meta_path)
    meta["status"] = "running"
    meta["started_at"] = now_iso()
    meta["finished_at"] = None
    meta["error"] = None
    write_meta(meta_path, meta)
    return meta


def mark_completed(
    meta_path: Path,
    outputs: dict[str, str],
) -> dict[str, Any]:
    """Mark one job as completed and save its generated outputs."""
    meta = read_meta(meta_path)
    meta["status"] = "completed"
    meta["finished_at"] = now_iso()
    meta["outputs"] = outputs
    meta["error"] = None
    write_meta(meta_path, meta)
    return meta


def mark_failed(
    meta_path: Path,
    error_message: str,
) -> dict[str, Any] | None:
    """Mark one job as failed.

    Returns None when the metadata file does not exist.
    """
    if not meta_path.exists():
        return None

    meta = read_meta(meta_path)
    meta["status"] = "failed"
    meta["finished_at"] = now_iso()
    meta["error"] = error_message
    write_meta(meta_path, meta)
    return meta
