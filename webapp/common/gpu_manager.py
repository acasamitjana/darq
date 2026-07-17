from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import torch

from webapp.worker.gpu_policy import DeviceDecision, GIB


class AtomicDirectoryLock:
    """Cross-process lock implemented with an atomically created directory."""

    def __init__(
        self,
        lock_dir: Path,
        timeout_seconds: float = 10.0,
        stale_after_seconds: float = 60.0,
        poll_seconds: float = 0.05,
    ) -> None:
        self.lock_dir = lock_dir
        self.timeout_seconds = timeout_seconds
        self.stale_after_seconds = stale_after_seconds
        self.poll_seconds = poll_seconds

    def __enter__(self) -> AtomicDirectoryLock:
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)

        deadline = time.monotonic() + self.timeout_seconds

        while True:
            try:
                self.lock_dir.mkdir()

            except FileExistsError:
                self._remove_stale_lock()

                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire GPU lock: {self.lock_dir}"
                    )

                time.sleep(self.poll_seconds)
                continue

            owner_path = self.lock_dir / "owner.json"
            owner_path.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_timestamp": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        shutil.rmtree(self.lock_dir, ignore_errors=True)

    def _remove_stale_lock(self) -> None:
        """Remove a lock left behind by a crashed worker."""

        try:
            age_seconds = (
                time.time() - self.lock_dir.stat().st_mtime
            )
        except FileNotFoundError:
            return

        if age_seconds > self.stale_after_seconds:
            shutil.rmtree(self.lock_dir, ignore_errors=True)


class SharedGPUManager:
    """Coordinate GPU reservations between different worker processes."""

    def __init__(
        self,
        state_dir: Path,
        reservation_ttl_seconds: float = 21600.0,
        lock_timeout_seconds: float = 10.0,
    ) -> None:
        self.state_dir = state_dir
        self.state_path = state_dir / "reservations.json"
        self.lock_dir = state_dir / ".gpu_lock"

        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.lock_timeout_seconds = lock_timeout_seconds

        self.state_dir.mkdir(parents=True, exist_ok=True)

    def reserve(
        self,
        *,
        job_id: str,
        pipeline: str,
        required_gib: float,
        safety_gib: float,
        gpu_index: int,
    ) -> DeviceDecision:
        """Reserve GPU capacity atomically or return a CPU decision."""

        if required_gib < 0:
            raise ValueError("required_gib cannot be negative.")

        if safety_gib < 0:
            raise ValueError("safety_gib cannot be negative.")

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

        device_count = torch.cuda.device_count()

        if gpu_index < 0 or gpu_index >= device_count:
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
                    f"Available GPUs: {device_count}."
                ),
            )

        lock = AtomicDirectoryLock(
            self.lock_dir,
            timeout_seconds=self.lock_timeout_seconds,
        )

        with lock:
            state = self._read_state()
            state_changed = self._remove_expired_reservations(state)

            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(
                    gpu_index
                )
                gpu_name = torch.cuda.get_device_name(gpu_index)

            except RuntimeError as exc:
                if state_changed:
                    self._write_state(state)

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

            free_gib = free_bytes / GIB
            total_gib = total_bytes / GIB

            active_reservations = [
                reservation
                for reservation in state["reservations"].values()
                if reservation.get("gpu_index") == gpu_index
            ]

            reserved_gib_before = sum(
                float(reservation.get("required_gib", 0.0))
                for reservation in active_reservations
            )

            # Real free VRAM, discounting the safety margin.
            real_available_gib = max(
                free_gib - safety_gib,
                0.0,
            )

            # Virtual capacity not already committed to another worker.
            reservation_budget_gib = max(
                total_gib
                - safety_gib
                - reserved_gib_before,
                0.0,
            )

            # Both conditions must be satisfied.
            effective_available_gib = min(
                real_available_gib,
                reservation_budget_gib,
            )

            if required_gib > effective_available_gib:
                if state_changed:
                    self._write_state(state)

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
                        f"GPU reservation denied. "
                        f"{free_gib:.2f} GiB are physically free, "
                        f"{reserved_gib_before:.2f} GiB are already reserved "
                        f"and only {effective_available_gib:.2f} GiB are "
                        "available after applying the safety margin."
                    ),
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
                )

            reservation_id = uuid.uuid4().hex

            state["reservations"][reservation_id] = {
                "reservation_id": reservation_id,
                "job_id": job_id,
                "pipeline": pipeline,
                "gpu_index": gpu_index,
                "required_gib": required_gib,
                "created_timestamp": time.time(),
                "worker_pid": os.getpid(),
            }

            self._write_state(state)

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
                    f"GPU reservation granted. "
                    f"{free_gib:.2f} GiB are physically free, "
                    f"{reserved_gib_before:.2f} GiB were already reserved "
                    f"and {required_gib:.2f} GiB have been reserved "
                    f"for job {job_id}."
                ),
                reservation_id=reservation_id,
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
            )

    def release(self, reservation_id: str | None) -> bool:
        """Release a previously granted GPU reservation."""

        if not reservation_id:
            return False

        lock = AtomicDirectoryLock(
            self.lock_dir,
            timeout_seconds=self.lock_timeout_seconds,
        )

        with lock:
            state = self._read_state()

            reservation = state["reservations"].pop(
                reservation_id,
                None,
            )

            if reservation is None:
                return False

            self._write_state(state)
            return True

    def _read_state(self) -> dict:
        """Read the shared reservation state."""

        if not self.state_path.exists():
            return {
                "version": 1,
                "reservations": {},
            }

        try:
            state = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return {
                "version": 1,
                "reservations": {},
            }

        if not isinstance(state.get("reservations"), dict):
            state["reservations"] = {}

        state.setdefault("version", 1)

        return state

    def _write_state(self, state: dict) -> None:
        """Write reservation state atomically."""

        self.state_dir.mkdir(parents=True, exist_ok=True)

        temporary_path = (
            self.state_dir
            / f".reservations-{uuid.uuid4().hex}.tmp"
        )

        temporary_path.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        os.replace(temporary_path, self.state_path)

    def _remove_expired_reservations(self, state: dict) -> bool:
        """Remove reservations left by workers that stopped unexpectedly."""

        now = time.time()
        expired_ids = []

        for reservation_id, reservation in (
            state["reservations"].items()
        ):
            created_timestamp = float(
                reservation.get("created_timestamp", 0.0)
            )

            age_seconds = now - created_timestamp

            if age_seconds > self.reservation_ttl_seconds:
                expired_ids.append(reservation_id)

        for reservation_id in expired_ids:
            state["reservations"].pop(reservation_id, None)

        return bool(expired_ids)