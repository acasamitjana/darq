import numpy as np
import pandas as pd
import pytest
import nibabel as nib
import darq.src.results as results

from darq.src.results import (
    _bilateral,
    _build_mask,
    _load_prev_tsv,
    _write_sbr_tsv,
)


def test_build_mask_selects_only_requested_labels():
    lab = np.array([
        [0, 11, 12],
        [50, 51, 99],
    ])

    mask = _build_mask(lab, (11, 12))

    expected = np.array([
        [False, True, True],
        [False, False, False],
    ])

    np.testing.assert_array_equal(mask, expected)


def test_bilateral_combines_right_and_left_masks():
    masks = {
        "R": np.array([True, False, False]),
        "L": np.array([False, True, False]),
    }

    out = _bilateral(masks)

    expected = np.array([True, True, False])

    np.testing.assert_array_equal(out, expected)


def test_load_prev_tsv_returns_empty_dataframe_when_file_missing(tmp_path):
    path = tmp_path / "missing.tsv"

    df = _load_prev_tsv(
        str(path),
        cols=["id", "hemi", "value"],
        index_cols=["id", "hemi"],
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.index.names) == ["id", "hemi"]


def test_write_sbr_tsv_creates_expected_file(tmp_path):
    shape = (4, 4, 4)

    dat_image = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1
    dat_fov = np.ones(shape, dtype=bool)

    def single_voxel(x, y, z):
        mask = np.zeros(shape, dtype=bool)
        mask[x, y, z] = True
        return mask

    mask_cau = {
        "R": single_voxel(0, 0, 0),
        "L": single_voxel(0, 0, 1),
    }

    mask_put = {
        "R": single_voxel(0, 1, 0),
        "L": single_voxel(0, 1, 1),
    }

    mask_str = {
        "R": mask_cau["R"] | mask_put["R"],
        "L": mask_cau["L"] | mask_put["L"],
    }

    mask_occ = {
        "R": single_voxel(3, 3, 2),
        "L": single_voxel(3, 3, 3),
    }

    for mask_dict in (mask_cau, mask_put, mask_str, mask_occ):
        mask_dict["Both"] = mask_dict["R"] | mask_dict["L"]

    _write_sbr_tsv(
        dat_image=dat_image,
        dat_fov=dat_fov,
        mask_cau=mask_cau,
        mask_put=mask_put,
        mask_str=mask_str,
        mask_occ=mask_occ,
        loss=0.5,
        results_dir=str(tmp_path),
        tag="sub-test",
    )

    out_file = tmp_path / "sub-test_sbr.tsv"

    assert out_file.exists()

    df = pd.read_csv(out_file, sep="\t")

    assert not df.empty
    assert len(df) == 21

    expected_columns = {
        "id",
        "processing",
        "loss",
        "hemi",
        "aggregate",
        "dat_str",
        "dat_cau",
        "dat_put",
        "dat_occ",
    }

    assert expected_columns.issubset(df.columns)
    assert set(df["id"]) == {"sub-test"}
    assert set(df["processing"]) == {"raw"}
    assert set(df["hemi"]) == {"R", "L", "Both"}
    assert set(df["aggregate"]) == {
        "sum",
        "mean",
        "median",
        "std",
        "p10",
        "p30",
        "p70",
    }

    assert np.allclose(df["loss"], 0.5)


@pytest.mark.gpu
def test_compute_symmetry_maps_gpu_only():
    pytest.skip(
        "_compute_symmetry_maps usa cuda:0 internamente; "
        "este test se deja marcado como GPU para moverlo a integración "
        "cuando haya GPU disponible."
    )

def test_load_prev_tsv_loads_existing_file_and_filters_columns(tmp_path):
    path = tmp_path / "existing.tsv"

    pd.DataFrame(
        {
            "id": ["sub-01"],
            "hemi": ["R"],
            "value": [1.5],
            "extra_column": ["ignore_me"],
        }
    ).to_csv(path, sep="\t", index=False)

    df = _load_prev_tsv(
        str(path),
        cols=["id", "hemi", "value"],
        index_cols=["id", "hemi"],
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.index.names) == ["id", "hemi"]
    assert "value" in df.columns
    assert "extra_column" not in df.columns
    assert df.loc[("sub-01", "R"), "value"] == 1.5


def test_write_sbr_tsv_force_replaces_previous_rows(tmp_path):
    shape = (4, 4, 4)

    dat_image = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1
    dat_fov = np.ones(shape, dtype=bool)

    def single_voxel(x, y, z):
        mask = np.zeros(shape, dtype=bool)
        mask[x, y, z] = True
        return mask

    mask_cau = {
        "R": single_voxel(0, 0, 0),
        "L": single_voxel(0, 0, 1),
    }

    mask_put = {
        "R": single_voxel(0, 1, 0),
        "L": single_voxel(0, 1, 1),
    }

    mask_str = {
        "R": mask_cau["R"] | mask_put["R"],
        "L": mask_cau["L"] | mask_put["L"],
    }

    mask_occ = {
        "R": single_voxel(3, 3, 2),
        "L": single_voxel(3, 3, 3),
    }

    for mask_dict in (mask_cau, mask_put, mask_str, mask_occ):
        mask_dict["Both"] = mask_dict["R"] | mask_dict["L"]

    _write_sbr_tsv(
        dat_image=dat_image,
        dat_fov=dat_fov,
        mask_cau=mask_cau,
        mask_put=mask_put,
        mask_str=mask_str,
        mask_occ=mask_occ,
        loss=0.1,
        results_dir=str(tmp_path),
        tag="sub-force",
    )

    _write_sbr_tsv(
        dat_image=dat_image,
        dat_fov=dat_fov,
        mask_cau=mask_cau,
        mask_put=mask_put,
        mask_str=mask_str,
        mask_occ=mask_occ,
        loss=0.2,
        results_dir=str(tmp_path),
        tag="sub-force",
        force_flag=True,
    )

    out_file = tmp_path / "sub-force_sbr.tsv"
    df = pd.read_csv(out_file, sep="\t")

    assert len(df) == 21
    assert set(df["id"]) == {"sub-force"}
    assert np.allclose(df["loss"], 0.2)

def test_write_symmetry_tsv_creates_expected_file_with_mocked_maps(tmp_path, monkeypatch):
    shape = (4, 4, 4)

    dat_image_raw = np.ones(shape, dtype=float)
    dat_v2r = np.eye(4)
    dat_rot = np.eye(4)
    intensity_norm = 1.0
    dat_fov = np.ones(shape, dtype=bool)

    def single_voxel(x, y, z):
        mask = np.zeros(shape, dtype=bool)
        mask[x, y, z] = True
        return mask

    mask_cau = {
        "R": single_voxel(0, 0, 0),
        "L": single_voxel(0, 0, 1),
    }

    mask_put = {
        "R": single_voxel(0, 1, 0),
        "L": single_voxel(0, 1, 1),
    }

    for mask_dict in (mask_cau, mask_put):
        mask_dict["Both"] = mask_dict["R"] | mask_dict["L"]

    def fake_compute_symmetry_maps(dat_norm, v2r_symm):
        symm_map = np.ones(shape, dtype=float)
        l2_map = np.ones(shape, dtype=float) * 2
        return symm_map, l2_map

    monkeypatch.setattr(
        results,
        "_compute_symmetry_maps",
        fake_compute_symmetry_maps,
    )

    results._write_symmetry_tsv(
        dat_image_raw=dat_image_raw,
        dat_v2r=dat_v2r,
        dat_rot=dat_rot,
        intensity_norm=intensity_norm,
        mask_cau=mask_cau,
        mask_put=mask_put,
        dat_fov=dat_fov,
        results_dir=str(tmp_path),
        tag="sub-symm",
    )

    out_file = tmp_path / "sub-symm_symm.tsv"

    assert out_file.exists()

    df = pd.read_csv(out_file, sep="\t")

    assert not df.empty
    assert len(df) == 30

    assert set(df["id"]) == {"sub-symm"}
    assert set(df["metric"]) == {"lncc", "l2"}
    assert set(df["hemi"]) == {"R", "L", "Both"}
    assert set(df["aggregate"]) == {"sum", "std", "mean", "min", "max"}

    expected_columns = {
        "id",
        "hemi",
        "metric",
        "aggregate",
        "dat_cau",
        "dat_put",
    }

    assert expected_columns.issubset(df.columns)

def test_save_session_results_returns_when_affine_missing(tmp_path, capsys):
    shape = (4, 4, 4)
    affine = np.eye(4)

    dat = np.ones(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.int16)

    dat_proxy = nib.Nifti1Image(dat, affine)
    labels_proxy = nib.Nifti1Image(labels, affine)

    results.save_session_results(
        dat_orig_proxy=dat_proxy,
        labels_orig_proxy=labels_proxy,
        loss=0.5,
        tag="sub-missing",
        results_dir=str(tmp_path),
    )

    captured = capsys.readouterr()

    assert "Affine matrix not available" in captured.out

def test_save_session_results_returns_when_affine_contains_nan(tmp_path, capsys):
    shape = (4, 4, 4)
    affine = np.eye(4)

    dat = np.ones(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.int16)

    dat_proxy = nib.Nifti1Image(dat, affine)
    labels_proxy = nib.Nifti1Image(labels, affine)

    affine_ras = np.eye(4)
    affine_ras[0, 0] = np.nan

    np.save(tmp_path / "sub-nan_space-symmetricT1w_aff.npy", affine_ras)

    results.save_session_results(
        dat_orig_proxy=dat_proxy,
        labels_orig_proxy=labels_proxy,
        loss=0.5,
        tag="sub-nan",
        results_dir=str(tmp_path),
    )

    captured = capsys.readouterr()

    assert "Affine matrix contains NaN" in captured.out

def test_save_session_results_with_synthetic_data_writes_outputs(tmp_path, monkeypatch):
    shape = (4, 4, 4)
    affine = np.eye(4)

    labels = np.zeros(shape, dtype=np.int16)

    labels[0, 0, 0] = 50
    labels[0, 0, 1] = 11

    labels[0, 1, 0] = 51
    labels[0, 1, 1] = 12

    labels[3, 3, 2] = 2005
    labels[3, 3, 3] = 1005

    dat = np.zeros(shape, dtype=np.float32)
    dat[labels > 0] = 2.0
    dat[1:3, 1:3, 1:3] = 5.0

    dat_proxy = nib.Nifti1Image(dat, affine)
    labels_proxy = nib.Nifti1Image(labels, affine)

    tag = "sub-full"

    np.save(tmp_path / f"{tag}_space-symmetricT1w_aff.npy", np.eye(4))
    np.save(tmp_path / f"{tag}_space-T1wdseg_rot.npy", np.eye(4))
    np.save(tmp_path / f"{tag}_space-dat_rot.npy", np.eye(4))

    def fake_vol_resample_fast(labels_orig_proxy, dat_proxy):
        return dat_proxy

    def fake_write_symmetry_tsv(
        dat_image_raw,
        dat_v2r,
        dat_rot,
        intensity_norm,
        mask_cau,
        mask_put,
        dat_fov,
        results_dir,
        tag,
        force_flag=False,
    ):
        pd.DataFrame(
            {
                "id": [tag],
                "hemi": ["Both"],
                "metric": ["fake"],
                "aggregate": ["mean"],
                "dat_cau": [1.0],
                "dat_put": [1.0],
            }
        ).to_csv(tmp_path / f"{tag}_symm.tsv", sep="\t", index=False)

    monkeypatch.setattr(results, "vol_resample_fast", fake_vol_resample_fast)
    monkeypatch.setattr(results, "_write_symmetry_tsv", fake_write_symmetry_tsv)

    results.save_session_results(
        dat_orig_proxy=dat_proxy,
        labels_orig_proxy=labels_proxy,
        loss=0.5,
        tag=tag,
        results_dir=str(tmp_path),
    )

    assert (tmp_path / f"{tag}_space-T1w_aff.npy").exists()
    assert (tmp_path / f"{tag}_dat.nii.gz").exists()
    assert (tmp_path / f"{tag}_desc-resampled_dat.nii.gz").exists()
    assert (tmp_path / f"{tag}_sbr.tsv").exists()
    assert (tmp_path / f"{tag}_symm.tsv").exists()

    sbr_df = pd.read_csv(tmp_path / f"{tag}_sbr.tsv", sep="\t")

    assert not sbr_df.empty
    assert len(sbr_df) == 21
    assert set(sbr_df["hemi"]) == {"R", "L", "Both"}

def test_save_session_results_returns_when_required_labels_missing(tmp_path, monkeypatch, capsys):
    shape = (4, 4, 4)
    affine = np.eye(4)

    # Labels sin caudado, putamen ni occipital válidos
    labels = np.zeros(shape, dtype=np.int16)

    dat = np.ones(shape, dtype=np.float32)

    dat_proxy = nib.Nifti1Image(dat, affine)
    labels_proxy = nib.Nifti1Image(labels, affine)

    tag = "sub-missing-labels"

    np.save(tmp_path / f"{tag}_space-symmetricT1w_aff.npy", np.eye(4))
    np.save(tmp_path / f"{tag}_space-T1wdseg_rot.npy", np.eye(4))
    np.save(tmp_path / f"{tag}_space-dat_rot.npy", np.eye(4))

    def fake_vol_resample_fast(labels_orig_proxy, dat_proxy):
        return dat_proxy

    monkeypatch.setattr(results, "vol_resample_fast", fake_vol_resample_fast)

    results.save_session_results(
        dat_orig_proxy=dat_proxy,
        labels_orig_proxy=labels_proxy,
        loss=0.5,
        tag=tag,
        results_dir=str(tmp_path),
    )

    captured = capsys.readouterr()

    assert "Label *caudate* not found" in captured.out
    assert not (tmp_path / f"{tag}_sbr.tsv").exists()

def test_write_symmetry_tsv_force_replaces_previous_rows(tmp_path, monkeypatch):
    shape = (4, 4, 4)

    dat_image_raw = np.ones(shape, dtype=float)
    dat_v2r = np.eye(4)
    dat_rot = np.eye(4)
    intensity_norm = 1.0
    dat_fov = np.ones(shape, dtype=bool)

    def single_voxel(x, y, z):
        mask = np.zeros(shape, dtype=bool)
        mask[x, y, z] = True
        return mask

    mask_cau = {
        "R": single_voxel(0, 0, 0),
        "L": single_voxel(0, 0, 1),
    }

    mask_put = {
        "R": single_voxel(0, 1, 0),
        "L": single_voxel(0, 1, 1),
    }

    for mask_dict in (mask_cau, mask_put):
        mask_dict["Both"] = mask_dict["R"] | mask_dict["L"]

    def fake_compute_symmetry_maps(dat_norm, v2r_symm):
        symm_map = np.ones(shape, dtype=float)
        l2_map = np.ones(shape, dtype=float) * 2
        return symm_map, l2_map

    monkeypatch.setattr(
        results,
        "_compute_symmetry_maps",
        fake_compute_symmetry_maps,
    )

    results._write_symmetry_tsv(
        dat_image_raw=dat_image_raw,
        dat_v2r=dat_v2r,
        dat_rot=dat_rot,
        intensity_norm=intensity_norm,
        mask_cau=mask_cau,
        mask_put=mask_put,
        dat_fov=dat_fov,
        results_dir=str(tmp_path),
        tag="sub-symm-force",
    )

    results._write_symmetry_tsv(
        dat_image_raw=dat_image_raw,
        dat_v2r=dat_v2r,
        dat_rot=dat_rot,
        intensity_norm=intensity_norm,
        mask_cau=mask_cau,
        mask_put=mask_put,
        dat_fov=dat_fov,
        results_dir=str(tmp_path),
        tag="sub-symm-force",
        force_flag=True,
    )

    out_file = tmp_path / "sub-symm-force_symm.tsv"
    df = pd.read_csv(out_file, sep="\t")

    assert len(df) == 30
    assert set(df["id"]) == {"sub-symm-force"}