from __future__ import annotations

from typing import Any

from webapp.frontend.base_api_client import BaseApiClient


class DarqApiClient(BaseApiClient):
    """API client for DARQ jobs."""

    @property
    def pipeline_name(self) -> str:
        return "darq"

    def submit(
        self,
        dat_file: Any,
        mri_file: Any,
        seg_file: Any,
        subject_id: str | None = None,
    ) -> dict:
        """Submit one DARQ job."""

        return self.submit_job(
            files={
                "dat_file": dat_file,
                "mri_file": mri_file,
                "seg_file": seg_file,
            },
            data={
                "subject_id": subject_id or "subject",
            },
        )