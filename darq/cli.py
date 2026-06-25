import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DARQ processing pipeline.

    :return: argparse namespace containing input paths, output path, optimizer
        settings, execution flags and hardware options.
    """
    
    parser = argparse.ArgumentParser(
        description="Automatic quantification pipeline of DaTSCAN images. If a companion MRI scan is availabe, it"
                    "registers both modalities together for subject-specific biomarkers; otherwise, it defaults to"
                    "using the MNI template.",
        epilog='\n',
    )

    # paths
    parser.add_argument("--dat",  help="Input DAT file (target).")

    parser.add_argument("--mri",  default=None, help="Input MRI file (reference).")


    parser.add_argument("--seg",  default=None, help="Input SynthSeg file (striatum segmentation).")

    parser.add_argument("--bids_dir", default=None,
                        help="Input BIDS directory for serial batch processing.")

    parser.add_argument("--seg_dir", default=None,
                        help="Directory containing SynthSeg segmentations for BIDS mode.")

    parser.add_argument("--o",   help="Output directory.")


    # algorithm options
    parser.add_argument("--opt_str", default='lbfgs',
                        choices=['lbfgs', 'adam', 'sgd'],
                        help="Optimiser to use for registration.")


    # execution flags
    parser.add_argument('--num_cores', default=1, type=int,
                        help="Parallel workers (1 = sequential).")

    parser.add_argument("--force",     action='store_true',
                        help="Recompute even if output already exists.")

    # hardware / debug
    parser.add_argument("--cpu",   action='store_true')

    return parser.parse_args()
