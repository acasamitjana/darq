import os

import torch

# ── Directories ──────────────────────────────────────────────────────────────
BIDS_DIR   = os.environ.get('BIDS_DIR')
SEG_DIR    = os.environ.get('SEG_DIR')

# ── Device ───────────────────────────────────────────────────────────────────
def get_device(
    cpu_flag: bool,
    requested_device: str | None = None,
) -> str:
    """Resolve and validate the PyTorch execution device."""

    if cpu_flag:
        return "cpu"

    # Direct execution without --device.
    if requested_device is None:
        if torch.cuda.is_available():
            return "cuda:0"

        return "cpu"

    try:
        device = torch.device(requested_device)

    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid execution device: {requested_device!r}."
        ) from exc

    if device.type == "cpu":
        return "cpu"

    if device.type != "cuda":
        raise ValueError(
            "DARQ only supports CPU and CUDA devices. "
            f"Received: {requested_device!r}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{requested_device} was requested, "
            "but CUDA is not available."
        )

    gpu_index = (
        device.index
        if device.index is not None
        else 0
    )

    gpu_count = torch.cuda.device_count()

    if gpu_index < 0 or gpu_index >= gpu_count:
        raise ValueError(
            f"GPU index {gpu_index} does not exist. "
            f"Available GPU indices: 0 to {gpu_count - 1}."
        )

    return f"cuda:{gpu_index}"

# ── Training / optimisation ───────────────────────────────────────────────────
REGISTRATION_DEFAULTS = {
    'save_model_frequency': 10,
    'starting_epoch': 0,
    'num_epochs': 30,
    'patience': 5,
    'results_dir': '/tmp/datscan_reg',
    'freq_print': 10
}

# ── DaT simulation hyperparameters ───────────────────────────────────────────
BLUR_MIN, BLUR_MAX  = 5, 5
