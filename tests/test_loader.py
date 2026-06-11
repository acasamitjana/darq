from argparse import Namespace

import pandas as pd
import pytest 

from darq.utils.loader import build_subject

def test_build_subject_raises_without_mri_or_seg():
    args = Namespace(mri=None, seg=None, dat="dat.nii.gz")

    with pytest.raises(NotImplementedError):
        build_subject(args)

def test_build_subject_creates_one_row_dataframe():
    args = Namespace(mri="mri.nii.gz", seg="seg.nii.gz", dat="dat.nii.gz")

    df = build_subject(args)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["subject"] == 0
    assert df.iloc[0]['id'] == "0"
    assert df.iloc[0]['mri_path'] == "mri.nii.gz"
    assert df.iloc[0]['label_path'] == "seg.nii.gz"
    assert df.iloc[0]['dat_path'] == "dat.nii.gz"