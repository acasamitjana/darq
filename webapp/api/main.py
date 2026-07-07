from __future__ import annotations

import json
import shutil
import uuid
import re as regex
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Form


app = FastAPI(
    title="DARQ API",
    version="0.1.0",
)

# Carpeta donde se guardarán los trabajos locales
JOBS_DIR = Path("webapp/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "DARQ API",
    }

def sanitize_job_text(text: str) -> str:
    """Convert user text into a safe folder name."""
    text = text.strip().lower()
    text = regex.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    text = text.strip("_")

    if not text:
        return "subject"

    return text


def create_job_id(subject_id: str) -> str:
    """Create a readable and unique job ID."""
    safe_subject = sanitize_job_text(subject_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]

    return f"{safe_subject}_{timestamp}_{short_uuid}"

def now_iso() -> str:
    """Return current time as text."""
    return datetime.now().isoformat(timespec="seconds")


def is_valid_nifti(filename: str | None) -> bool:
    """Check if uploaded file looks like a NIfTI image."""
    if filename is None:
        return False

    return filename.endswith(".nii") or filename.endswith(".nii.gz")


def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save one uploaded file to disk."""
    if not is_valid_nifti(upload_file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format: {upload_file.filename}. Expected .nii or .nii.gz",
        )

    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


@app.post("/api/jobs")
def create_job(
    dat_file: UploadFile = File(...),
    mri_file: UploadFile = File(...),
    seg_file: UploadFile = File(...),
    subject_id: str = Form("subject"),
) -> dict:
    """
    Create a new DARQ job.

    For now, this only saves the uploaded files.
    It does not run the pipeline yet.
    """
    job_id = create_job_id(subject_id)

    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    logs_dir = job_dir / "logs"

    input_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(exist_ok=False)
    logs_dir.mkdir(exist_ok=False)

    save_upload_file(dat_file, input_dir / "dat.nii.gz")
    save_upload_file(mri_file, input_dir / "mri.nii.gz")
    save_upload_file(seg_file, input_dir / "seg.nii.gz")

    meta = {
        "job_id": job_id,
        "pipeline": "darq",
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "inputs": {
            "dat": "input/dat.nii.gz",
            "mri": "input/mri.nii.gz",
            "seg": "input/seg.nii.gz",
        },
        "outputs": {},
        "error": None,
    }

    meta_path = job_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Files uploaded correctly. Job created but not processed yet.",
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Return information about one job."""
    job_dir = JOBS_DIR / job_id
    meta_path = job_dir / "meta.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

