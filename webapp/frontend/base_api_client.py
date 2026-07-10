from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests


class BaseApiClient:
    """Common HTTP client for pipeline jobs."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = (
            api_url
            or os.getenv("DARQ_API_URL", "http://127.0.0.1:8000")
        ).rstrip("/")

    @property
    def pipeline_name(self) -> str:
        """Return the pipeline identifier."""
        raise NotImplementedError

    @staticmethod
    def _get_file_path(file_obj: Any) -> Path:
        """Return the local path of an uploaded file."""
        if file_obj is None:
            raise ValueError("Missing uploaded file.")

        if isinstance(file_obj, (str, Path)):
            return Path(file_obj)

        if hasattr(file_obj, "name"):
            return Path(file_obj.name)

        raise TypeError(
            f"Unsupported file object type: {type(file_obj)}"
        )

    def submit_job(
        self,
        files: dict[str, Any],
        data: dict[str, str] | None = None,
    ) -> dict:
        """Submit one job to the pipeline endpoint."""

        endpoint = f"{self.api_url}/api/jobs/{self.pipeline_name}"

        with ExitStack() as stack:
            request_files = {}

            for field_name, file_obj in files.items():
                file_path = self._get_file_path(file_obj)

                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Input file not found: {file_path}"
                    )

                request_files[field_name] = stack.enter_context(
                    file_path.open("rb")
                )

            response = requests.post(
                endpoint,
                files=request_files,
                data=data or {},
                timeout=60,
            )

        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> dict:
        """Return the metadata of one job."""

        response = requests.get(
            f"{self.api_url}/api/jobs/{job_id}",
            timeout=30,
        )

        response.raise_for_status()
        return response.json()