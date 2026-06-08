import argparse
from os import listdir
from os.path import join, exists, isdir

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

def build_subject(args: argparse.Namespace) -> pd.DataFrame:
    """Build a single-subject dataframe from explicit command-line paths.

    :param args: Command-line arguments containing DaT, MRI and segmentation paths.

    :raises NotImplementedError: Template-based registration is not implemented when MRI or
        segmentation
        
    :return: DataFrame containing one row with paths required by the dataset.
    """

    if args.mri is None or args.seg is None:
        raise NotImplementedError("Template-based registration still not implementes")
        #TO DO

    df = {
        'subject': [0],
        'id': [str(0)],
        'mri_path': [args.mri],
        'label_path': [args.seg],
        'dat_path': [args.dat],
    }

    df = pd.DataFrame(df)
    df.set_index('subject', inplace=True, drop=False)
    return df
