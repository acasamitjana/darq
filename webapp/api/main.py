from __future__ import annotations

from fastapi import FastAPI

from webapp.api.routes.darq import router as darq_router
from webapp.api.routes.health import router as health_router
from webapp.api.routes.jobs import router as jobs_router
from webapp.api.routes.results import router as results_router


app = FastAPI(
    title="DARQ API",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(darq_router)
app.include_router(jobs_router)
app.include_router(results_router)
