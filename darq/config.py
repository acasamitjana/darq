import os

# ── Directories ──────────────────────────────────────────────────────────────
BIDS_DIR   = os.environ.get('BIDS_DIR')
SEG_DIR    = os.environ.get('SEG_DIR')

# ── Device ───────────────────────────────────────────────────────────────────
def get_device(cpu_flag: bool) -> str:
    return 'cpu' if cpu_flag else 'cuda'

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
