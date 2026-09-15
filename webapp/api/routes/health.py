from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(
    prefix="/api",
    tags=["Health"],
)


@router.get("/health")
def health() -> dict:
    """Return the API health status."""
    return {
        "status": "ok",
        "service": "DARQ API",
    }
