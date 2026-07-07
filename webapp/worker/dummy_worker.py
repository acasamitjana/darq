from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT_DIR / "webapp" / "jobs"

POLL_INTERVAL_SECONDS = 2
DUMMY_PROCESSING_SECONDS = 8


def now_iso() -> str:
    """Return current time as ISO text."""
    return datetime.now().isoformat(timespec="seconds")


def read_meta(meta_path: Path) -> dict:
    """Read one job metadata file."""
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_meta(meta_path: Path, meta: dict) -> None:
    """Write one job metadata file safely."""
    temp_path = meta_path.with_suffix(".json.tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    temp_path.replace(meta_path)


def append_log(job_dir: Path, message: str) -> None:
    """Append one message to the job log."""
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_path = logs_dir / "worker.log"

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {message}\n")


def find_queued_jobs() -> list[Path]:
    """Find job folders whose meta.json has status='queued'."""
    if not JOBS_DIR.exists():
        return []

    queued_jobs = []

    for job_dir in sorted(JOBS_DIR.iterdir()):
        if not job_dir.is_dir():
            continue

        meta_path = job_dir / "meta.json"

        if not meta_path.exists():
            continue

        try:
            meta = read_meta(meta_path)
        except Exception:
            continue

        if meta.get("status") == "queued":
            queued_jobs.append(job_dir)

    return queued_jobs


def process_dummy_job(job_dir: Path) -> None:
    """Simulate the processing of one job."""
    meta_path = job_dir / "meta.json"
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    meta = read_meta(meta_path)
    job_id = meta.get("job_id", job_dir.name)

    print(f"[Worker] Starting dummy job: {job_id}")
    append_log(job_dir, "Dummy worker picked up the job.")

    meta["status"] = "running"
    meta["started_at"] = now_iso()
    meta["error"] = None
    write_meta(meta_path, meta)

    append_log(job_dir, "Status changed to running.")
    append_log(job_dir, f"Simulating processing for {DUMMY_PROCESSING_SECONDS} seconds.")

    time.sleep(DUMMY_PROCESSING_SECONDS)

    dummy_result_path = output_dir / "dummy_result.txt"
    with dummy_result_path.open("w", encoding="utf-8") as f:
        f.write("Dummy result generated successfully.\n")
        f.write("The real DARQ pipeline has not been executed yet.\n")
        f.write(f"Job ID: {job_id}\n")
        f.write(f"Finished at: {now_iso()}\n")

    meta = read_meta(meta_path)
    meta["status"] = "completed"
    meta["finished_at"] = now_iso()
    meta["outputs"] = {
        "dummy_result": "output/dummy_result.txt",
    }
    write_meta(meta_path, meta)

    append_log(job_dir, "Dummy result created in output/dummy_result.txt.")
    append_log(job_dir, "Status changed to completed.")

    print(f"[Worker] Completed dummy job: {job_id}")


def main() -> None:
    """Continuously look for queued jobs and process them."""
    print("[Worker] DARQ dummy worker started.")
    print(f"[Worker] Watching folder: {JOBS_DIR}")
    print("[Worker] Press Ctrl+C to stop.\n")

    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        queued_jobs = find_queued_jobs()

        if not queued_jobs:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        for job_dir in queued_jobs:
            try:
                process_dummy_job(job_dir)
            except Exception as exc:
                meta_path = job_dir / "meta.json"

                if meta_path.exists():
                    meta = read_meta(meta_path)
                    meta["status"] = "failed"
                    meta["finished_at"] = now_iso()
                    meta["error"] = f"{type(exc).__name__}: {exc}"
                    write_meta(meta_path, meta)

                append_log(job_dir, f"Job failed: {type(exc).__name__}: {exc}")
                print(f"[Worker] Failed job {job_dir.name}: {exc}")


if __name__ == "__main__":
    main()