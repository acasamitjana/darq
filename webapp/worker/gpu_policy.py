from __future__ import annotations

from dataclasses import dataclass

import torch


GIB = 1024 ** 3


@dataclass(frozen=True)
class DeviceDecision:
    """Result of the automatic GPU/CPU device selection."""

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
    reserved_gib_before: float = 0.0
    real_available_gib: float | None = None
    reservation_budget_gib: float | None = None
    effective_available_gib: float | None = None

    @property
    def use_gpu(self) -> bool:
        """Return True when the job should run on GPU."""

        return self.execution_device.startswith("cuda")


def choose_execution_device(
    required_gib: float,
    safety_gib: float = 1.0,
    gpu_index: int = 0,
) -> DeviceDecision:
    """Choose GPU when CUDA exists and enough VRAM is available."""

    if required_gib < 0:
        raise ValueError("required_gib must be greater than or equal to zero.")

    if safety_gib < 0:
        raise ValueError("safety_gib must be greater than or equal to zero.")

    required_with_margin = required_gib + safety_gib

    if not torch.cuda.is_available():
        return DeviceDecision(
            execution_device="cpu",
            cuda_available=False,
            gpu_index=None,
            gpu_name=None,
            free_gib=None,
            total_gib=None,
            required_gib=required_gib,
            safety_gib=safety_gib,
            reason="CUDA is not available inside the worker container.",
        )

    if gpu_index >= torch.cuda.device_count():
        return DeviceDecision(
            execution_device="cpu",
            cuda_available=True,
            gpu_index=None,
            gpu_name=None,
            free_gib=None,
            total_gib=None,
            required_gib=required_gib,
            safety_gib=safety_gib,
            reason=(
                f"GPU index {gpu_index} does not exist. "
                f"Available GPUs: {torch.cuda.device_count()}."
            ),
        )

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)

        free_gib = free_bytes / GIB
        total_gib = total_bytes / GIB
        gpu_name = torch.cuda.get_device_name(gpu_index)

    except RuntimeError as exc:
        return DeviceDecision(
            execution_device="cpu",
            cuda_available=True,
            gpu_index=gpu_index,
            gpu_name=None,
            free_gib=None,
            total_gib=None,
            required_gib=required_gib,
            safety_gib=safety_gib,
            reason=f"GPU memory could not be inspected: {exc}",
        )

    if free_gib >= required_with_margin:
        return DeviceDecision(
            execution_device=f"cuda:{gpu_index}",
            cuda_available=True,
            gpu_index=gpu_index,
            gpu_name=gpu_name,
            free_gib=round(free_gib, 3),
            total_gib=round(total_gib, 3),
            required_gib=required_gib,
            safety_gib=safety_gib,
            reason=(
                f"{free_gib:.2f} GiB are free. "
                f"{required_with_margin:.2f} GiB are required "
                "including the safety margin."
            ),
        )

    return DeviceDecision(
        execution_device="cpu",
        cuda_available=True,
        gpu_index=gpu_index,
        gpu_name=gpu_name,
        free_gib=round(free_gib, 3),
        total_gib=round(total_gib, 3),
        required_gib=required_gib,
        safety_gib=safety_gib,
        reason=(
            f"Only {free_gib:.2f} GiB are free. "
            f"{required_with_margin:.2f} GiB are required "
            "including the safety margin."
        ),
    )