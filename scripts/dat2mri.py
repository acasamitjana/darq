import os
from joblib import delayed, Parallel

from darq.cli import parse_args
from darq.config import get_device, BIDS_DIR, SEG_DIR, REGISTRATION_DEFAULTS
from darq.utils.loader import build_subject
from darq.src.preprocessing import get_preprocessing_transforms
from darq.processing import process_subject, process_fn_parallel, get_dat_transforms
from darq.src.datasets import MRI_DaT


def main() -> None:
    """Run the command-line entry point for the DaTSCAN-to-MRI pipeline.

    The function parses user arguments, builds the subject/session dataframe, creates
    the dataset and transforms, and launches either sequential or parallel subject
    processing.
    
    :return: None. Results are written to the output directory selected by the user.
    """
    
    os.system('clear')
    print('\n\n# ------------------------- #')
    print('# Running Dat2MRI algorithm #')
    print('# ------------------------- #\n\n')

    args   = parse_args()
    device = get_device(args.cpu)

    # ── Data ─────────────────────────────────────────────────────────────────
    subject_list = build_subject(args)
    dataset      = MRI_DaT(subject_list, transforms=[], crop_labels=True, crop_dat=False)

    print(f'Total sessions to process: N={len(subject_list)}\n')

    # ── Transforms ───────────────────────────────────────────────────────────
    preprocessing_tf = get_preprocessing_transforms(device)
    dat_tf           = get_dat_transforms()

    # ── Training config ───────────────────────────────────────────────────────
    main_dict = {
        **REGISTRATION_DEFAULTS,
        'device':     device,
        'output_dir': args.o,
    }

    # ── Processing loop ───────────────────────────────────────────────────────
    failed = []
    for it, index in enumerate(subject_list.index):
        print(f'Subject: {index}  ({it}/{len(dataset)})')
        try:
            result = process_subject(dataset[index], preprocessing_tf, dat_tf, main_dict, args)
        except Exception as e:
            print(f'  [Error] {e}')
            result = None
        if result is None:
            failed.append(str(index))

    print('\n')
    print('Failed subjects:', len(failed), '/', len(subject_list))
    print('\n - '.join(failed) if failed else '  (none)')
    print('\n# ---------#')
    print('# All DONE #')
    print('# ---------#\n\n')


if __name__ == '__main__':
    main()
