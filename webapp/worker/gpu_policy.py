from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


GIB = 1024 ** 3


@dataclass(frozen=True)
class DeviceDecision:
    """Result of GPU/CPU selection and optional reservation."""

    execution_device: str
    cuda_available: bool

    gpu_index: int | None
    gpu_name: str | None

    free_gib: float | None
    total_gib: float | None

    required_gib: float
    safety_gib: float
    reason: str

    reservation_id: str | None = None

    # Amount reserved for the new job.
    reserved_gib: float = 0.0

    # Amount already reserved before selecting this job.
    reserved_gib_before: float = 0.0

    real_available_gib: float | None = None
    reservation_budget_gib: float | None = None
    effective_available_gib: float | None = None
    effective_free_gib: float | None = None

    checked_gpu_indices: tuple[int, ...] = ()

    @property
    def use_gpu(self) -> bool:
        """Return True when a CUDA device was selected."""

        return self.execution_device.startswith("cuda:")


def choose_execution_device(
    required_gib: float,
    safety_gib: float = 1.0,
    gpu_index: int | None = None,
    reserved_gib_by_gpu: Mapping[int, float] | None = None,
) -> DeviceDecision:
    """
    Select an execution device.

    Behaviour
    ---------
    gpu_index is an integer:
        Strict manual selection. Only that GPU is checked.
        If it cannot be used, an exception is raised.

    gpu_index is None:
        Automatic selection. All visible GPUs are checked in order.
        The first suitable GPU is selected. CPU is used when no GPU fits.
    """

    if required_gib < 0:
        raise ValueError(
            "required_gib must be greater than or equal to zero."
        )

    if safety_gib < 0:
        raise ValueError(
            "safety_gib must be greater than or equal to zero."
        )

    reserved_gib_by_gpu = reserved_gib_by_gpu or {}

    # CUDA unavailable.
    if not torch.cuda.is_available():
        if gpu_index is not None:
            raise RuntimeError(
                f"cuda:{gpu_index} was explicitly requested, "
                "but CUDA is not available."
            )

        return DeviceDecision(
            execution_device="cpu",
            cuda_available=False,
            gpu_index=None,
            gpu_name=None,
            free_gib=None,
            total_gib=None,
            required_gib=required_gib,
            safety_gib=safety_gib,
            reason="CUDA is not available. CPU selected.",
        )

    gpu_count = torch.cuda.device_count()

    if gpu_count <= 0:
        if gpu_index is not None:
            raise RuntimeError(
                f"cuda:{gpu_index} was explicitly requested, "
                "but no CUDA GPUs were detected."
            )

        return DeviceDecision(
            execution_device="cpu",
            cuda_available=True,
            gpu_index=None,
            gpu_name=None,
            free_gib=None,
            total_gib=None,
            required_gib=required_gib,
            safety_gib=safety_gib,
            reason="No CUDA GPUs were detected. CPU selected.",
        )

    # Manual strict selection.
    if gpu_index is not None:
        if gpu_index < 0 or gpu_index >= gpu_count:
            raise ValueError(
                f"GPU index {gpu_index} does not exist. "
                f"Available GPU indices: 0 to {gpu_count - 1}."
            )

        candidate_indices = [gpu_index]

    # Automatic selection.
    else:
        candidate_indices = list(range(gpu_count))

    checked_indices: list[int] = []
    inspection_errors: list[str] = []

    for current_index in candidate_indices:
        checked_indices.append(current_index)

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                current_index
            )

            gpu_name = torch.cuda.get_device_name(
                current_index
            )

        except RuntimeError as exc:
            if gpu_index is not None:
                raise RuntimeError(
                    f"Could not inspect cuda:{current_index}: {exc}"
                ) from exc

            inspection_errors.append(
                f"cuda:{current_index}: {exc}"
            )
            continue

        free_gib = free_bytes / GIB
        total_gib = total_bytes / GIB

        reserved_gib_before = max(
            0.0,
            float(
                reserved_gib_by_gpu.get(
                    current_index,
                    0.0,
                )
            ),
        )

        # Physical memory available after preserving the new
        # job's safety margin.
        real_available_gib = max(
            free_gib - safety_gib,
            0.0,
        )

        # Logical capacity not already committed to other jobs.
        reservation_budget_gib = max(
            total_gib
            - reserved_gib_before
            - safety_gib,
            0.0,
        )

        effective_available_gib = min(
            real_available_gib,
            reservation_budget_gib,
        )

        if required_gib <= effective_available_gib:
            return DeviceDecision(
                execution_device=f"cuda:{current_index}",
                cuda_available=True,
                gpu_index=current_index,
                gpu_name=gpu_name,
                free_gib=round(free_gib, 3),
                total_gib=round(total_gib, 3),
                required_gib=required_gib,
                safety_gib=safety_gib,
                reserved_gib_before=round(
                    reserved_gib_before,
                    3,
                ),
                real_available_gib=round(
                    real_available_gib,
                    3,
                ),
                reservation_budget_gib=round(
                    reservation_budget_gib,
                    3,
                ),
                effective_available_gib=round(
                    effective_available_gib,
                    3,
                ),
                effective_free_gib=round(
                    effective_available_gib,
                    3,
                ),
                checked_gpu_indices=tuple(
                    checked_indices
                ),
                reason=(
                    f"GPU {current_index} selected. "
                    f"{free_gib:.2f} GiB are physically free, "
                    f"{reserved_gib_before:.2f} GiB are already reserved "
                    f"and {effective_available_gib:.2f} GiB are "
                    "effectively available after the safety margin."
                ),
            )

        # In strict manual mode, do not try another GPU or CPU.
        if gpu_index is not None:
            raise RuntimeError(
                f"cuda:{gpu_index} was explicitly requested, "
                "but it does not have enough available VRAM. "
                f"Required: {required_gib:.2f} GiB plus a "
                f"{safety_gib:.2f} GiB safety margin. "
                f"Effective available capacity: "
                f"{effective_available_gib:.2f} GiB."
            )

    # Automatic mode: no suitable GPU found.
    error_suffix = ""

    if inspection_errors:
        error_suffix = (
            " Inspection errors: "
            + "; ".join(inspection_errors)
        )

    return DeviceDecision(
        execution_device="cpu",
        cuda_available=True,
        gpu_index=None,
        gpu_name=None,
        free_gib=None,
        total_gib=None,
        required_gib=required_gib,
        safety_gib=safety_gib,
        checked_gpu_indices=tuple(checked_indices),
        reason=(
            "No visible GPU has enough available VRAM. "
            "CPU selected."
            + error_suffix
        ),
    )