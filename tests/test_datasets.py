import numpy as np
import pandas as pd
import nibabel as nib


from darq.src.datasets import MRI_DaT

def test_dataset_len_matches_dataframe_rows():
    subject_df = pd.DataFrame({"subject": [0, 1, 2]}).set_index("subject", drop=False)
    dataset = MRI_DaT(subject_df)

    assert len(dataset) == 3

def test_get_roi_masks_detects_striatum_and_occipital_labels():
    subject_df = pd.DataFrame({"subject": [0]}).set_index("subject", drop=False)
    dataset = MRI_DaT(subject_df)

    lab = np.zeros((4, 4, 4), dtype=int)
    lab[0, 0, 0] = 11
    lab[0, 0, 1] = 12
    lab[0, 1, 0] = 50
    lab[0, 1, 1] = 51

    lab[2, 0, 0] = 1005
    lab[2, 0, 1] = 1011
    lab[2, 1, 0] = 2005
    lab[2, 1, 1] = 2011

    mask_str, mask_occ = dataset._get_ROI_masks(lab)

    assert mask_str.shape == lab.shape
    assert mask_occ.shape == lab.shape
    assert mask_str.sum() == 4
    assert mask_occ.sum() == 4
    assert mask_str.dtype == float
    assert mask_occ.dtype == float

def test_getitem_returns_none_for_unknown_index():
    subject_df = pd.DataFrame({"subject": [0]}).set_index("subject", drop=False)
    dataset = MRI_DaT(subject_df)

    assert dataset[99] is None


def test_getitem_loads_synthetic_nifti_subject(tmp_path):
    shape = (12, 12, 12)
    affine = np.eye(4, dtype=np.float32)

    # MRI sintética
    mri = np.random.default_rng(0).normal(size=shape).astype(np.float32)

    # Label sintética con algunas etiquetas de estriado
    label = np.zeros(shape, dtype=np.int16)
    label[3:5, 3:5, 3:5] = 11   # caudado izquierdo
    label[6:8, 6:8, 6:8] = 12   # putamen izquierdo
    label[2:4, 7:9, 7:9] = 50   # caudado derecho
    label[7:9, 2:4, 7:9] = 51   # putamen derecho

    # DaT sintética con señal de fondo + zona más intensa
    rng = np.random.default_rng(1)
    dat = rng.normal(loc=1.0, scale=0.1, size=shape).astype(np.float32)
    dat[4:8, 4:8, 4:8] += 5.0

    mri_path = tmp_path / "mri.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    dat_path = tmp_path / "dat.nii.gz"

    nib.save(nib.Nifti1Image(mri, affine), mri_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)
    nib.save(nib.Nifti1Image(dat, affine), dat_path)

    subject_df = pd.DataFrame(
        {
            "subject": [0],
            "mri_path": [str(mri_path)],
            "label_path": [str(label_path)],
            "dat_path": [str(dat_path)],
        }
    ).set_index("subject", drop=False)

    dataset = MRI_DaT(
        subject_df,
        transforms=[],
        crop_labels=False,
        crop_dat=False,
    )

    item = dataset[0]

    assert item is not None
    assert item["id"] == "0"

    assert item["mri_image"].shape == shape
    assert item["dat_image"].shape == shape
    assert item["label_image"].shape == shape

    assert item["mask_str"].shape == shape
    assert item["mask_occ"].shape == shape
    assert item["mask_brain"].shape == shape

    assert item["mask_cau"].shape == shape
    assert item["mask_pu"].shape == shape

    assert item["dat_brain"].shape == shape + (1,)
    assert item["dat_str"].shape == shape + (1,)

    assert item["dat_v2r"].shape == (4, 4)
    assert item["label_v2r"].shape == (4, 4)

    assert item["mask_str"].sum() > 0
    assert item["mask_brain"].sum() > 0
    assert item["dat_brain"].sum() > 0

def _make_synthetic_subject(tmp_path, shape=(20, 20, 20), mri_affine=None, label_affine=None):
    if mri_affine is None:
        mri_affine = np.eye(4, dtype=np.float32)
    if label_affine is None:
        label_affine = np.eye(4, dtype=np.float32)

    rng = np.random.default_rng(0)

    mri = rng.normal(size=shape).astype(np.float32)

    label = np.zeros(shape, dtype=np.int16)
    label[8:10, 8:10, 8:10] = 11
    label[10:12, 10:12, 10:12] = 12
    label[12:14, 8:10, 8:10] = 50
    label[8:10, 12:14, 8:10] = 51

    dat = rng.normal(loc=1.0, scale=0.1, size=shape).astype(np.float32)
    dat[8:12, 8:12, 8:12] += 5.0

    mri_path = tmp_path / "mri.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    dat_path = tmp_path / "dat.nii.gz"

    nib.save(nib.Nifti1Image(mri, mri_affine), mri_path)
    nib.save(nib.Nifti1Image(label, label_affine), label_path)
    nib.save(nib.Nifti1Image(dat, np.eye(4, dtype=np.float32)), dat_path)

    return pd.DataFrame(
        {
            "subject": [0],
            "mri_path": [str(mri_path)],
            "label_path": [str(label_path)],
            "dat_path": [str(dat_path)],
        }
    ).set_index("subject", drop=False)

def test_getitem_with_crop_labels_true_adds_label_crop(tmp_path):
    subject_df = _make_synthetic_subject(tmp_path)

    dataset = MRI_DaT(
        subject_df,
        transforms=[],
        crop_labels=True,
        crop_dat=False,
    )

    item = dataset[0]

    assert item is not None
    assert "label_crop" in item
    assert item["label_crop"].shape == (4, 4)
    assert item["label_image"].ndim == 3
    assert item["mri_image"].ndim == 3
    assert item["label_v2r"].shape == (4, 4)

def test_getitem_prints_when_label_and_mri_affines_differ(tmp_path, capsys):
    mri_affine = np.eye(4, dtype=np.float32)

    label_affine = np.eye(4, dtype=np.float32)
    label_affine[0, 3] = 5.0

    subject_df = _make_synthetic_subject(
        tmp_path,
        mri_affine=mri_affine,
        label_affine=label_affine,
    )

    dataset = MRI_DaT(
        subject_df,
        transforms=[],
        crop_labels=False,
        crop_dat=False,
    )

    item = dataset[0]

    captured = capsys.readouterr()

    assert item is not None
    assert captured.out != ""

