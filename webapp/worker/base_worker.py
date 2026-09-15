from __future__ import annotations

import os
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from webapp.common.job_storage import (
    append_log,
    mark_completed,
    mark_failed,
    mark_running,
    read_meta,
)

class BaseWorker(ABC):
    """Base worker containing the common job-processing logic.

    Subclasses only need to define:

    - pipeline_name
    - build_pipeline_command()
    - collect_output_files()
    """

    poll_interval_seconds = 2

    def __init__(self, jobs_dir: Path | None = None) -> None:
        """Initialize the worker paths."""

        self.root_dir = Path(__file__).resolve().parents[2]
        default_jobs_dir = self.root_dir / "webapp" / "jobs"

        self.jobs_dir = jobs_dir or Path(
            os.getenv("DARQ_JOBS_DIR", str(default_jobs_dir))
        )

    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Return the pipeline name handled by this worker."""

        raise NotImplementedError

    @abstractmethod
    def build_pipeline_command(
        self,
        job_dir: Path,
        execution_resources: Any = None,
    ) -> list[str]:
        """Build the command used to execute the concrete pipeline."""

        raise NotImplementedError

    @abstractmethod
    def collect_output_files(self, job_dir: Path) -> dict[str, str]:
        """Collect the outputs produced by the concrete pipeline."""

        raise NotImplementedError

    def find_queued_jobs(self) -> list[Path]:
        """Find queued jobs belonging to this pipeline."""

        if not self.jobs_dir.exists():
            return []

        queued_jobs = []

        for job_dir in sorted(self.jobs_dir.iterdir()):
            if not job_dir.is_dir():
                continue

            meta_path = job_dir / "meta.json"

            if not meta_path.exists():
                continue

            try:
                meta = read_meta(meta_path)
            except Exception:
                continue

            if meta.get("status") != "queued":
                continue

            if meta.get("pipeline") != self.pipeline_name:
                continue

            queued_jobs.append(job_dir)

        return queued_jobs

    def write_pipeline_log(
        self,
        job_dir: Path,
        command: list[str],
        completed: subprocess.CompletedProcess[str],
    ) -> None:
        """Save command, stdout and stderr into logs/pipeline.log."""

        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_path = logs_dir / "pipeline.log"

        with log_path.open("w", encoding="utf-8") as file:
            file.write(
                f"{self.pipeline_name.upper()} pipeline execution\n"
            )
            file.write("=" * 40 + "\n\n")

            file.write("Started command:\n")
            file.write(" ".join(command) + "\n\n")

            file.write(f"Return code: {completed.returncode}\n\n")

            file.write("STDOUT:\n")
            file.write(completed.stdout or "")
            file.write("\n\n")

            file.write("STDERR:\n")
            file.write(completed.stderr or "")
            file.write("\n")

    def process_job(self, job_dir: Path) -> None:
        """Execute one job using the concrete pipeline."""

        meta_path = job_dir / "meta.json"
        output_dir = job_dir / "output"

        output_dir.mkdir(parents=True, exist_ok=True)

        meta = read_meta(meta_path)
        job_id = meta.get("job_id", job_dir.name)

        print(
            f"[{self.pipeline_name} worker] Starting job: {job_id}"
        )

        append_log(
            job_dir,
            f"{self.pipeline_name} worker picked up the job.",
        )

        mark_running(meta_path)

        execution_resources = None

        try:
            execution_resources = self.acquire_execution_resources(
                job_dir
            )

            command = self.build_pipeline_command(
                job_dir,
                execution_resources=execution_resources,
            )

            append_log(job_dir, "Status changed to running.")

            append_log(
                job_dir,
                f"Running {self.pipeline_name} pipeline.",
            )

            append_log(
                job_dir,
                "Command: " + " ".join(command),
            )

            process = subprocess.Popen(
                command,
                cwd=self.root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            stdout_lines = []

            if process.stdout is not None:

                for line in process.stdout:

                    stdout_lines.append(line)

                    print(
                        line,
                        end="",
                    )

                    self.on_pipeline_output(
                        job_dir,
                        line,
                    )

            return_code = process.wait()

            completed = subprocess.CompletedProcess(
                args=command,
                returncode=return_code,
                stdout="".join(stdout_lines),
                stderr="",
            )

            self.write_pipeline_log(
                job_dir,
                command,
                completed,
            )

            if completed.returncode != 0:
                error_message = (
                    f"{self.pipeline_name.upper()} pipeline failed. "
                    "Check logs/pipeline.log for details."
                )

                mark_failed(
                    meta_path,
                    error_message,
                )

                append_log(
                    job_dir,
                    f"Job failed: {error_message}",
                )

                print(
                    f"[{self.pipeline_name} worker] "
                    f"Failed job: {job_id}"
                )

                return

            outputs = self.collect_output_files(job_dir)

            mark_completed(
                meta_path,
                outputs,
            )

            append_log(
                job_dir,
                "Pipeline finished successfully.",
            )

            append_log(
                job_dir,
                "Status changed to completed.",
            )

            print(
                f"[{self.pipeline_name} worker] "
                f"Completed job: {job_id}"
            )

        finally:
            try:
                self.release_execution_resources(
                    job_dir,
                    execution_resources,
                )

            except Exception as exc:
                append_log(
                    job_dir,
                    (
                        "Could not release execution resources: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

                print(
                    f"[{self.pipeline_name} worker] "
                    f"Resource release failed for {job_id}: {exc}"
                )

    def on_pipeline_output(
        self,
        job_dir: Path,
        line: str,
        ) -> None:
        """Handle one line produced by the running pipeline."""

        pass
                
    def run(self) -> None:
        """Continuously search for and process queued jobs."""

        print(
            f"[{self.pipeline_name} worker] Worker started."
        )
        print(
            f"[{self.pipeline_name} worker] "
            f"Watching folder: {self.jobs_dir}"
        )
        print(
            f"[{self.pipeline_name} worker] "
            f"Project root: {self.root_dir}"
        )
        print(
            f"[{self.pipeline_name} worker] "
            "Press Ctrl+C to stop.\n"
        )

        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                queued_jobs = self.find_queued_jobs()

                if not queued_jobs:
                    time.sleep(self.poll_interval_seconds)
                    continue

                for job_dir in queued_jobs:
                    try:
                        self.process_job(job_dir)

                    except Exception as exc:
                        error_message = (
                            f"{type(exc).__name__}: {exc}"
                        )

                        meta_path = job_dir / "meta.json"

                        mark_failed(
                            meta_path,
                            error_message,
                        )

                        append_log(
                            job_dir,
                            f"Job failed: {error_message}",
                        )

                        print(
                            f"[{self.pipeline_name} worker] "
                            f"Failed job {job_dir.name}: {exc}"
                        )

        except KeyboardInterrupt:
            print(
                f"\n[{self.pipeline_name} worker] Worker stopped."
            )