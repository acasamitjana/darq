from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from webapp.common.gpu_manager import SharedGPUManager
from webapp.common.job_storage import (
    append_log,
    read_meta,
    write_meta,
)
from webapp.worker.base_worker import BaseWorker
from webapp.worker.gpu_policy import DeviceDecision


class DarqWorker(BaseWorker):
    """Worker responsible for executing DARQ jobs."""

    def __init__(self, jobs_dir: Path | None = None) -> None:
        super().__init__(jobs_dir=jobs_dir)

        default_gpu_state_dir = (
            self.root_dir / "webapp" / "gpu_state"
        )

        gpu_state_dir = Path(
            os.getenv(
                "GPU_STATE_DIR",
                str(default_gpu_state_dir),
            )
        )

        reservation_ttl_seconds = float(
            os.getenv(
                "GPU_RESERVATION_TTL_SECONDS",
                "21600",
            )
        )

        lock_timeout_seconds = float(
            os.getenv(
                "GPU_LOCK_TIMEOUT_SECONDS",
                "10",
            )
        )

        self.gpu_manager = SharedGPUManager(
            state_dir=gpu_state_dir,
            reservation_ttl_seconds=reservation_ttl_seconds,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    @property
    def pipeline_name(self) -> str:
        """Return the pipeline identifier."""

        return "darq"

    def acquire_execution_resources(
        self,
        job_dir: Path,
    ) -> DeviceDecision:
        """Reserve GPU capacity or choose CPU."""

        meta_path = job_dir / "meta.json"
        meta = read_meta(meta_path)

        job_id = meta.get("job_id", job_dir.name)

        required_gib = float(
            os.getenv("DARQ_GPU_REQUIRED_GIB", "6.0")
        )

        safety_gib = float(
            os.getenv("DARQ_GPU_SAFETY_GIB", "1.0")
        )

        gpu_index = int(
            os.getenv("DARQ_GPU_INDEX", "0")
        )

        decision = self.gpu_manager.reserve(
            job_id=job_id,
            pipeline=self.pipeline_name,
            required_gib=required_gib,
            safety_gib=safety_gib,
            gpu_index=gpu_index,
        )

        meta = read_meta(meta_path)

        meta["requested_device"] = "auto"
        meta["execution_device"] = decision.execution_device
        meta["device_decision"] = asdict(decision)
        meta["gpu_reservation_active"] = bool(
            decision.reservation_id
        )

        write_meta(meta_path, meta)

        append_log(
            job_dir,
            (
                f"Selected device: {decision.execution_device}. "
                f"{decision.reason}"
            ),
        )

        return decision

    def release_execution_resources(
        self,
        job_dir: Path,
        execution_resources: Any,
    ) -> None:
        """Release the GPU reservation associated with this job."""

        if not isinstance(
            execution_resources,
            DeviceDecision,
        ):
            return

        reservation_id = execution_resources.reservation_id

        if not reservation_id:
            return

        released = self.gpu_manager.release(reservation_id)

        meta_path = job_dir / "meta.json"
        meta = read_meta(meta_path)

        meta["gpu_reservation_active"] = False
        meta["gpu_reservation_released"] = released

        write_meta(meta_path, meta)

        if released:
            append_log(
                job_dir,
                (
                    "GPU reservation released: "
                    f"{reservation_id}"
                ),
            )
        else:
            append_log(
                job_dir,
                (
                    "GPU reservation was not found when releasing: "
                    f"{reservation_id}"
                ),
            )

    def build_pipeline_command(
        self,
        job_dir: Path,
        execution_resources: Any = None,
    ) -> list[str]:
        """Build the DARQ command using the reserved device."""

        input_dir = job_dir / "input"
        output_dir = job_dir / "output"

        dat_path = input_dir / "dat.nii.gz"
        mri_path = input_dir / "mri.nii.gz"
        seg_path = input_dir / "seg.nii.gz"

        if not dat_path.exists():
            raise FileNotFoundError(
                f"Missing DAT file: {dat_path}"
            )

        if not mri_path.exists():
            raise FileNotFoundError(
                f"Missing MRI file: {mri_path}"
            )

        if not seg_path.exists():
            raise FileNotFoundError(
                f"Missing SynthSeg file: {seg_path}"
            )

        if not isinstance(
            execution_resources,
            DeviceDecision,
        ):
            raise RuntimeError(
                "DARQ execution resources were not acquired."
            )

        pipeline_script = (
            self.root_dir / "scripts" / "dat2mri.py"
        )

        command = [
            sys.executable,
            str(pipeline_script),
            "--dat",
            str(dat_path),
            "--mri",
            str(mri_path),
            "--seg",
            str(seg_path),
            "--o",
            str(output_dir),
            "--force",
        ]

        if not execution_resources.use_gpu:
            command.append("--cpu")

        return command

    def collect_output_files(
        self,
        job_dir: Path,
    ) -> dict[str, str]:
        """Collect files generated by DARQ."""

        output_dir = job_dir / "output"
        outputs = {}

        if not output_dir.exists():
            return outputs

        generic_file_index = 1

        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue

            relative_path = path.relative_to(
                job_dir
            ).as_posix()

            if path.name.endswith("_sbr.tsv"):
                outputs["sbr_tsv"] = relative_path

            elif path.name.endswith("_symm.tsv"):
                outputs["symmetry_tsv"] = relative_path

            elif path.suffix.lower() == ".pdf":
                outputs["pdf_report"] = relative_path

            else:
                key = f"file_{generic_file_index}"
                outputs[key] = relative_path
                generic_file_index += 1

        return outputs


def main() -> None:
    """Start the DARQ worker."""

    worker = DarqWorker()
    worker.run()


if __name__ == "__main__":
    main()