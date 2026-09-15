from __future__ import annotations

from fastapi import APIRouter, HTTPException

from webapp.api.config import JOBS_DIR
from webapp.common.job_storage import read_meta


router = APIRouter(
    prefix="/api/jobs",
    tags=["Jobs"],
)


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    """Return the metadata of one job."""
    meta_path = JOBS_DIR / job_id / "meta.json"

    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return read_meta(meta_path)
