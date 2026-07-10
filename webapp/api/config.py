from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

JOBS_DIR = Path(
    os.getenv(
        "DARQ_JOBS_DIR",
        str(ROOT_DIR / "webapp" / "jobs"),
    )
)

JOBS_DIR.mkdir(parents=True, exist_ok=True)
