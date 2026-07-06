from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests


API_URL = "http://127.0.0.1:8000"


def _get_file_path(file_obj: Any) -> Path:
    """
    Convert a Gradio uploaded file into a Path.

    In your current Gradio app, gr.File(..., type="filepath") returns a string path.
    This helper also supports objects with .name just in case the Gradio behaviour changes.
    """
    if file_obj is None:
        raise ValueError("Missing uploaded file.")

    if isinstance(file_obj, str):
        return Path(file_obj)

    if hasattr(file_obj, "name"):
        return Path(file_obj.name)

    raise TypeError(f"Unsupported file object type: {type(file_obj)}")


def submit_darq_job(
    dat_file: Any,
    mri_file: Any,
    seg_file: Any,
    subject_id: str | None = None,
) -> dict:
    """
    Send DAT, MRI and SynthSeg files from Gradio to FastAPI.

    This does not run the DARQ pipeline yet.
    It only asks FastAPI to create a job and store the uploaded files.
    """
    dat_path = _get_file_path(dat_file)
    mri_path = _get_file_path(mri_file)
    seg_path = _get_file_path(seg_file)

    with ExitStack() as stack:
        files = {
            "dat_file": stack.enter_context(dat_path.open("rb")),
            "mri_file": stack.enter_context(mri_path.open("rb")),
            "seg_file": stack.enter_context(seg_path.open("rb")),
        }

        data = {
            "subject_id": subject_id or "subject",
        }

        response = requests.post(
            f"{API_URL}/api/jobs",
            files=files,
            data=data,
            timeout=60,
        )

    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str) -> dict:
    """
    Ask FastAPI for the current metadata of a job.
    """
    response = requests.get(
        f"{API_URL}/api/jobs/{job_id}",
        timeout=30,
    )

    response.raise_for_status()
    return response.json()