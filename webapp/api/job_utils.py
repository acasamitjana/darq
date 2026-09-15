from __future__ import annotations

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile


def sanitize_job_text(text: str) -> str:
    """Convert user text into a safe folder-name component."""
    text = text.strip().lower()
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    text = text.strip("_")

    return text or "subject"


def create_job_id(subject_id: str) -> str:
    """Create a readable and unique job identifier."""
    safe_subject = sanitize_job_text(subject_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]

    return f"{safe_subject}_{timestamp}_{short_uuid}"


def is_valid_nifti(filename: str | None) -> bool:
    """Return whether a filename has a supported NIfTI extension."""
    if filename is None:
        return False

    filename = filename.lower()
    return filename.endswith(".nii") or filename.endswith(".nii.gz")


def save_upload_file(
    upload_file: UploadFile,
    destination: Path,
) -> None:
    """Validate and save one uploaded NIfTI file."""
    if not is_valid_nifti(upload_file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file format: {upload_file.filename}. "
                "Expected .nii or .nii.gz"
            ),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
