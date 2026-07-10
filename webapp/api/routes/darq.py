from __future__ import annotations

import shutil

from fastapi import APIRouter, File, Form, UploadFile

from webapp.api.config import JOBS_DIR
from webapp.api.job_utils import create_job_id, save_upload_file
from webapp.common.job_storage import create_meta, write_meta


router = APIRouter(
    prefix="/api/jobs",
    tags=["DARQ"],
)


@router.post("/darq")
def create_darq_job(
    dat_file: UploadFile = File(...),
    mri_file: UploadFile = File(...),
    seg_file: UploadFile = File(...),
    subject_id: str = Form("subject"),
) -> dict:
    """Create and queue one DARQ processing job."""
    job_id = create_job_id(subject_id)

    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    logs_dir = job_dir / "logs"

    try:
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(exist_ok=False)
        logs_dir.mkdir(exist_ok=False)

        save_upload_file(
            dat_file,
            input_dir / "dat.nii.gz",
        )
        save_upload_file(
            mri_file,
            input_dir / "mri.nii.gz",
        )
        save_upload_file(
            seg_file,
            input_dir / "seg.nii.gz",
        )

        meta = create_meta(
            job_id=job_id,
            pipeline="darq",
            inputs={
                "dat": "input/dat.nii.gz",
                "mri": "input/mri.nii.gz",
                "seg": "input/seg.nii.gz",
            },
            extra={
                "subject_id": subject_id,
            },
        )

        write_meta(
            job_dir / "meta.json",
            meta,
        )

    except Exception:
        if job_dir.exists():
            shutil.rmtree(
                job_dir,
                ignore_errors=True,
            )
        raise

    return {
        "job_id": job_id,
        "pipeline": "darq",
        "status": "queued",
        "message": "Files uploaded correctly. Job queued for processing.",
    }
