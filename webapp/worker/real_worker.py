from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT_DIR / "webapp" / "jobs"
PIPELINE_SCRIPT = ROOT_DIR / "scripts" / "dat2mri.py"

POLL_INTERVAL_SECONDS = 2

# False = run the local script with the current Python interpreter.
# True = run the installed DARQ command.
#
# Recommended for now: False.
# If you set it to True, the worker will run:
# darq --dat ... --mri ... --seg ... --o ... --cpu --force
USE_DARQ_COMMAND = False


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
    """Append one message to the worker log."""
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


def build_pipeline_command(job_dir: Path) -> list[str]:
    """Build the command that runs the real DARQ pipeline."""
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"

    dat_path = input_dir / "dat.nii.gz"
    mri_path = input_dir / "mri.nii.gz"
    seg_path = input_dir / "seg.nii.gz"

    if not dat_path.exists():
        raise FileNotFoundError(f"Missing DAT file: {dat_path}")

    if not mri_path.exists():
        raise FileNotFoundError(f"Missing MRI file: {mri_path}")

    if not seg_path.exists():
        raise FileNotFoundError(f"Missing SynthSeg file: {seg_path}")

    if USE_DARQ_COMMAND:
        return [
            "darq",
            "--dat",
            str(dat_path),
            "--mri",
            str(mri_path),
            "--seg",
            str(seg_path),
            "--o",
            str(output_dir),
            "--cpu",
            "--force",
        ]

    return [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--dat",
        str(dat_path),
        "--mri",
        str(mri_path),
        "--seg",
        str(seg_path),
        "--o",
        str(output_dir),
        "--cpu",
        "--force",
    ]


def collect_output_files(job_dir: Path) -> dict:
    """Collect generated output files as relative paths."""
    output_dir = job_dir / "output"

    outputs = {}

    if not output_dir.exists():
        return outputs

    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(job_dir).as_posix()

        if path.name.endswith("_sbr.tsv"):
            outputs["sbr_tsv"] = relative_path

        elif path.name.endswith("_symm.tsv"):
            outputs["symmetry_tsv"] = relative_path

        elif path.suffix.lower() == ".pdf":
            outputs["pdf_report"] = relative_path

        else:
            key = f"file_{len(outputs) + 1}"
            outputs[key] = relative_path

    return outputs


def write_pipeline_log(job_dir: Path, command: list[str], completed: subprocess.CompletedProcess[str]) -> None:
    """Save pipeline command, stdout and stderr into logs/pipeline.log."""
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_path = logs_dir / "pipeline.log"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("DARQ real worker pipeline execution\n")
        f.write("=" * 40 + "\n\n")

        f.write("Started command:\n")
        f.write(" ".join(command) + "\n\n")

        f.write(f"Return code: {completed.returncode}\n\n")

        f.write("STDOUT:\n")
        f.write(completed.stdout or "")
        f.write("\n\n")

        f.write("STDERR:\n")
        f.write(completed.stderr or "")
        f.write("\n")


def process_real_job(job_dir: Path) -> None:
    """Process one job by running the real DARQ pipeline."""
    meta_path = job_dir / "meta.json"
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    meta = read_meta(meta_path)
    job_id = meta.get("job_id", job_dir.name)

    print(f"[Real worker] Starting job: {job_id}")
    append_log(job_dir, "Real worker picked up the job.")

    meta["status"] = "running"
    meta["started_at"] = now_iso()
    meta["finished_at"] = None
    meta["error"] = None
    write_meta(meta_path, meta)

    command = build_pipeline_command(job_dir)

    append_log(job_dir, "Status changed to running.")
    append_log(job_dir, "Running real DARQ pipeline.")
    append_log(job_dir, "Command: " + " ".join(command))

    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )

    write_pipeline_log(job_dir, command, completed)

    if completed.returncode != 0:
        meta = read_meta(meta_path)
        meta["status"] = "failed"
        meta["finished_at"] = now_iso()
        meta["error"] = (
            "DARQ pipeline failed. "
            "Check logs/pipeline.log for details."
        )
        write_meta(meta_path, meta)

        append_log(job_dir, "Pipeline failed.")
        print(f"[Real worker] Failed job: {job_id}")
        return

    outputs = collect_output_files(job_dir)

    meta = read_meta(meta_path)
    meta["status"] = "completed"
    meta["finished_at"] = now_iso()
    meta["outputs"] = outputs
    meta["error"] = None
    write_meta(meta_path, meta)

    append_log(job_dir, "Pipeline finished successfully.")
    append_log(job_dir, "Status changed to completed.")

    print(f"[Real worker] Completed job: {job_id}")


def main() -> None:
    """Continuously look for queued jobs and process them."""
    print("[Real worker] DARQ real worker started.")
    print(f"[Real worker] Watching folder: {JOBS_DIR}")
    print(f"[Real worker] Project root: {ROOT_DIR}")
    print(f"[Real worker] Use DARQ command: {USE_DARQ_COMMAND}")
    print("[Real worker] Press Ctrl+C to stop.\n")

    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        queued_jobs = find_queued_jobs()

        if not queued_jobs:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        for job_dir in queued_jobs:
            try:
                process_real_job(job_dir)
            except Exception as exc:
                meta_path = job_dir / "meta.json"

                if meta_path.exists():
                    meta = read_meta(meta_path)
                    meta["status"] = "failed"
                    meta["finished_at"] = now_iso()
                    meta["error"] = f"{type(exc).__name__}: {exc}"
                    write_meta(meta_path, meta)

                append_log(job_dir, f"Job failed: {type(exc).__name__}: {exc}")
                print(f"[Real worker] Failed job {job_dir.name}: {exc}")


if __name__ == "__main__":
    main()