from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from darq import processing


def _eye():
    return np.eye(4, dtype="float32")


def _small_3d(value=0.0):
    return np.full((2, 2, 2), value, dtype="float32")


def test_get_dat_transforms_returns_expected_transforms():
    transforms = processing.get_dat_transforms()

    assert len(transforms) == 2
    assert transforms[0].keys == ["simulated_dat"]
    assert transforms[1].keys == ["simulated_dat"]


def test_label_blobs_detects_connected_components():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[3, 3, 3] = True

    blobs, n_blobs = processing._label_blobs(mask)

    assert blobs.shape == mask.shape
    assert n_blobs == 2


def test_blob_centre_returns_ras_coordinate():
    blobs = np.zeros((4, 4, 4), dtype=int)
    blobs[1:3, 1, 1] = 1

    v2r = np.eye(4, dtype="float32")
    v2r[:3, 3] = np.array([10, 20, 30], dtype="float32")

    centre = processing._blob_centre(blobs, 1, v2r)

    np.testing.assert_allclose(centre, np.array([11.5, 21.0, 31.0, 1.0]))


def test_flip_dat_aggregation_modes(monkeypatch):
    dat_image = np.ones((2, 2, 2), dtype="float32")

    def fake_interp(tensor, di, dj, dk, mode="linear"):
        return torch.full_like(tensor, 2.0)

    monkeypatch.setattr(processing.fn_utils, "fast_3D_interp_torch", fake_interp)

    mean_out = processing._flip_dat(dat_image, _eye(), agg="mean")
    max_out = processing._flip_dat(dat_image, _eye(), agg="max")
    sum_out = processing._flip_dat(dat_image, _eye(), agg="sum")
    flip_out = processing._flip_dat(dat_image, _eye(), agg="flip")

    np.testing.assert_allclose(mean_out, np.full_like(dat_image, 1.5))
    np.testing.assert_allclose(max_out, np.full_like(dat_image, 2.0))
    np.testing.assert_allclose(sum_out, np.full_like(dat_image, 3.0))
    np.testing.assert_allclose(flip_out, np.full_like(dat_image, 2.0))


def test_get_dat_mask_keeps_large_plausible_blob(monkeypatch):
    dat_image = np.zeros((50, 50, 50), dtype="float32")
    dat_image[10:20, 10:20, 10:16] = 10.0

    class FakeKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x):
            self.labels_ = (x.reshape(-1) > 0).astype(int)
            return self

    monkeypatch.setattr(processing, "KMeans", FakeKMeans)
    monkeypatch.setattr(processing, "binary_opening", lambda mask, footprint: mask)

    mask = processing._get_dat_mask(dat_image, _eye())

    assert mask.shape == dat_image.shape
    assert mask.dtype == bool
    assert np.sum(mask) == 600


def test_get_dat_mask_prior_keeps_large_mask_inside_prior(monkeypatch):
    dat_image = np.zeros((10, 10, 10), dtype="float32")
    dat_image.reshape(-1)[:600] = 10.0
    prior = np.ones_like(dat_image)

    class FakeKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x):
            self.labels_ = (x.reshape(-1) > 0).astype(int)
            return self

    monkeypatch.setattr(processing, "KMeans", FakeKMeans)

    mask = processing._get_dat_mask_prior(dat_image, prior, percentage=100)

    assert mask.shape == dat_image.shape
    assert np.sum(mask) == 600


def test_get_dat_mask_prior_removes_small_blobs(monkeypatch):
    dat_image = np.zeros((10, 10, 10), dtype="float32")
    dat_image.reshape(-1)[:400] = 10.0
    prior = np.ones_like(dat_image)

    class FakeKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x):
            self.labels_ = (x.reshape(-1) > 0).astype(int)
            return self

    monkeypatch.setattr(processing, "KMeans", FakeKMeans)

    mask = processing._get_dat_mask_prior(dat_image, prior, percentage=100)

    assert mask.shape == dat_image.shape
    assert np.sum(mask) == 0


def test_build_optimizer_returns_adam_lbfgs_and_sgd():
    model = nn.Linear(1, 1)

    adam = processing._build_optimizer(model, "adam")
    lbfgs = processing._build_optimizer(model, "lbfgs")
    sgd = processing._build_optimizer(model, "sgd")

    assert isinstance(adam, torch.optim.Adam)
    assert isinstance(lbfgs, torch.optim.LBFGS)
    assert isinstance(sgd, torch.optim.SGD)


def test_prepare_subject_run_creates_output_directory(tmp_path):
    data_dict = {"id": "sub-001"}
    main_dict = {
        "device": "cpu",
        "output_dir": str(tmp_path / "out"),
    }

    run_info = processing._prepare_subject_run(data_dict, main_dict)

    assert run_info["device"] == "cpu"
    assert run_info["tag"] == "sub-001"
    assert run_info["output_dir"] == str(tmp_path / "out")
    assert run_info["temp_dir"].endswith("tmp")
    assert (tmp_path / "out").exists()


def test_should_skip_subject_when_sbr_exists_and_force_false(tmp_path):
    tag = "sub-001"
    sbr_file = tmp_path / f"{tag}_sbr.tsv"
    sbr_file.write_text("already processed")

    should_skip = processing._should_skip_subject(
        output_dir=str(tmp_path),
        tag=tag,
        force_flag=False,
    )

    assert should_skip is True


def test_should_not_skip_subject_when_force_true(tmp_path):
    tag = "sub-001"
    sbr_file = tmp_path / f"{tag}_sbr.tsv"
    sbr_file.write_text("already processed")

    should_skip = processing._should_skip_subject(
        output_dir=str(tmp_path),
        tag=tag,
        force_flag=True,
    )

    assert should_skip is False


def test_run_preprocessing_step_applies_transforms_and_saves_rotations(tmp_path):
    def add_rotations(data_dict):
        data_dict["rot_label_v2r"] = np.eye(4, dtype="float32")
        data_dict["rot_dat_v2r"] = np.eye(4, dtype="float32") * 2
        data_dict["was_transformed"] = True
        return data_dict

    data_dict = {"id": "sub-001"}
    preproc_tf = {"dummy": add_rotations}

    out = processing._run_preprocessing_step(
        data_dict=data_dict,
        preproc_tf=preproc_tf,
        output_dir=str(tmp_path),
        tag="sub-001",
    )

    assert out["was_transformed"] is True
    assert (tmp_path / "sub-001_space-T1wdseg_rot.npy").exists()
    assert (tmp_path / "sub-001_space-dat_rot.npy").exists()


def test_compute_symmetric_dat_averages_raw_and_flipped(monkeypatch):
    dat_raw = np.ones((2, 2, 2), dtype="float32")

    monkeypatch.setattr(
        processing,
        "_flip_dat",
        lambda dat_image, v2r_symm, agg="flip": np.full_like(dat_image, 3.0),
    )

    out = processing._compute_symmetric_dat(dat_raw, _eye())

    np.testing.assert_allclose(out, np.full_like(dat_raw, 2.0))


def test_build_dat_prior_cuboid_uses_detected_mask(monkeypatch):
    dat_symm = np.zeros((20, 20, 20), dtype="float32")

    mask = np.zeros_like(dat_symm, dtype=bool)
    mask[8:10, 8:10, 8:10] = True

    monkeypatch.setattr(processing, "_get_dat_mask", lambda dat_image, v2r: mask)

    cuboid = processing._build_dat_prior_cuboid(dat_symm, _eye())

    assert cuboid.shape == dat_symm.shape
    assert np.sum(cuboid) > np.sum(mask)


def test_build_symmetric_dat_mask_combines_raw_and_flipped_masks(monkeypatch):
    dat_raw = np.zeros((2, 2, 2), dtype="float32")
    cuboid = np.ones_like(dat_raw)

    raw_mask = np.zeros_like(dat_raw, dtype=bool)
    raw_mask[0, 0, 0] = True

    flipped_mask = np.zeros_like(dat_raw, dtype=bool)
    flipped_mask[1, 1, 1] = True

    monkeypatch.setattr(
        processing,
        "_get_dat_mask_prior",
        lambda dat_image, prior, percentage=1: raw_mask,
    )

    monkeypatch.setattr(
        processing,
        "_flip_dat",
        lambda dat_image, v2r_symm, agg="flip": flipped_mask,
    )

    out = processing._build_symmetric_dat_mask(dat_raw, _eye(), cuboid)

    assert out[0, 0, 0] is True or bool(out[0, 0, 0])
    assert out[1, 1, 1] is True or bool(out[1, 1, 1])
    assert np.sum(out) == 2


def test_normalize_dat_inside_mask_sets_outside_to_zero():
    dat_raw = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ],
        dtype="float32",
    )

    dat_mask = dat_raw >= 2.0

    out = processing._normalize_dat_inside_mask(dat_raw, dat_mask)

    assert out.shape == dat_raw.shape
    assert np.all(out[~dat_mask] == 0)
    assert np.max(out[dat_mask]) > 0


def test_estimate_dat_brain_foreground_with_fake_gmm(monkeypatch):
    dat_symm = np.arange(6, dtype="float32").reshape(3, 2, 1)
    reference_brain_mask = np.zeros_like(dat_symm)
    reference_brain_mask[0, 0, 0] = 1

    class FakeGMM:
        def __init__(self, *args, **kwargs):
            pass

        def fit_predict(self, x):
            return np.arange(x.shape[0])

    monkeypatch.setattr(processing, "GMM", FakeGMM)

    brain_dat = processing._estimate_dat_brain_foreground(
        dat_symm=dat_symm,
        reference_brain_mask=reference_brain_mask,
    )

    assert brain_dat.shape == dat_symm.shape
    assert np.sum(brain_dat) == 3


def test_build_mri_dat_mask_with_fake_kmeans(monkeypatch):
    simulated_dat = np.array(
        [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0],
        dtype="float32",
    ).reshape(2, 2, 2)

    class FakeKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x):
            self.labels_ = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            self.cluster_centers_ = np.array([[0.0], [6.5]])
            return self

    monkeypatch.setattr(processing, "KMeans", FakeKMeans)

    mri_mask, sim_dat = processing._build_mri_dat_mask(simulated_dat)

    assert mri_mask.shape == simulated_dat.shape
    assert sim_dat.shape == simulated_dat.shape
    assert np.sum(mri_mask) == 4
    assert np.isclose(np.max(sim_dat[mri_mask]), 1.0)


def test_estimate_initial_translation():
    mri_mask = np.zeros((4, 4, 4), dtype=bool)
    dat_mask = np.zeros((4, 4, 4), dtype=bool)

    mri_mask[0, 0, 0] = True
    dat_mask[1, 2, 3] = True

    tx_init = processing._estimate_initial_translation(mri_mask, dat_mask)

    np.testing.assert_allclose(tx_init, np.array([1.0, 2.0, 3.0]))


def test_simulate_dat_from_mri_applies_transforms(monkeypatch):
    data_dict = {
        "template_mask_str": np.ones((2, 2, 2), dtype="float32"),
    }

    dat_mask = np.zeros((2, 2, 2), dtype=bool)
    dat_mask[1, 1, 1] = True

    def transform(data):
        data["simulated_dat"] = data["simulated_dat"] * 2
        data["transform_applied"] = True
        return data

    mri_mask = np.zeros((2, 2, 2), dtype=bool)
    mri_mask[0, 0, 0] = True
    sim_dat = np.ones((2, 2, 2), dtype="float32") * 5

    monkeypatch.setattr(
        processing,
        "_build_mri_dat_mask",
        lambda simulated_dat: (mri_mask, sim_dat),
    )

    monkeypatch.setattr(
        processing,
        "_estimate_initial_translation",
        lambda mri, dat: np.array([1.0, 2.0, 3.0]),
    )

    out_data, mri_context = processing._simulate_dat_from_mri(
        data_dict=data_dict,
        dat_tf=[transform],
        template_v2r=_eye(),
        dat_mask=dat_mask,
    )

    assert out_data["transform_applied"] is True
    assert "v2r" in out_data
    assert "simulated_dat" in out_data
    np.testing.assert_allclose(mri_context["tx_init"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(mri_context["sim_dat"], sim_dat)


def test_build_registration_tensors_shapes_and_device():
    data_dict = {
        "template_mask_brain": np.ones((2, 2, 2), dtype="float32"),
        "template_mask_occ": np.zeros((2, 2, 2), dtype="float32"),
    }

    dat_context = {
        "template_v2r": _eye(),
        "dat_mask": np.ones((2, 2, 2), dtype=bool),
        "brain_dat": np.ones((2, 2, 2), dtype="float32"),
    }

    mri_context = {
        "mri_mask": np.zeros((2, 2, 2), dtype=bool),
    }

    tensors = processing._build_registration_tensors(
        data_dict=data_dict,
        dat_context=dat_context,
        mri_context=mri_context,
        device="cpu",
    )

    assert tensors["ref_image"].shape == (1, 1, 2, 2, 2)
    assert tensors["ref_mask"].shape == (1, 2, 2, 2, 2)
    assert tensors["flo_image"].shape == (1, 1, 2, 2, 2)
    assert tensors["flo_mask"].shape == (1, 2, 2, 2, 2)
    assert tensors["ref_image"].device.type == "cpu"
    np.testing.assert_allclose(tensors["template_v2r"], _eye())


def test_run_registration_step_uses_model_optimizer_loss_and_session(monkeypatch):
    calls = {}

    class FakeRigidModel:
        def __init__(self, *args, **kwargs):
            calls["model_kwargs"] = kwargs

        def to(self, device):
            calls["model_device"] = device
            return self

    class FakeSession:
        def __init__(self, loss_dict, main_dict, da, trainable_keys, verbose):
            calls["loss_dict"] = loss_dict
            calls["main_dict"] = main_dict
            calls["trainable_keys"] = trainable_keys
            calls["verbose"] = verbose

        def register(self, tensor_dict, model_dict, optimizer_dict):
            tensor_dict["loss"] = 1.23
            tensor_dict["registered"] = True
            calls["optimizer_dict"] = optimizer_dict
            return tensor_dict

    monkeypatch.setattr(processing.models, "InstanceRigidModelClassic", FakeRigidModel)
    monkeypatch.setattr(processing.models, "JointInstanceReg", FakeSession)
    monkeypatch.setattr(processing, "_build_optimizer", lambda model, opt_str: "optimizer")
    monkeypatch.setattr(
        processing,
        "_build_loss_dict",
        lambda device, v2r: {"reg": {"loss": SimpleNamespace(name="reg"), "weight": 1}},
    )

    tensor_dict = {"input": True}
    data_dict = {
        "template_space": np.zeros((2, 2, 2), dtype="float32"),
    }
    dat_context = {
        "template_v2r": _eye(),
    }
    mri_context = {
        "tx_init": np.array([1.0, 2.0, 3.0]),
    }

    args = SimpleNamespace(opt_str="sgd")

    out = processing._run_registration_step(
        tensor_dict=tensor_dict,
        data_dict=data_dict,
        dat_context=dat_context,
        mri_context=mri_context,
        main_dict={"num_epochs": 1},
        args=args,
        device="cpu",
    )

    assert out["loss"] == 1.23
    assert out["registered"] is True
    assert calls["model_device"] == "cpu"
    assert calls["model_kwargs"]["device"] == "cpu"
    assert calls["optimizer_dict"]["reg"] == "optimizer"
    assert calls["trainable_keys"] == {"reg": "reg"}


def test_save_subject_outputs_saves_affine_and_calls_results(monkeypatch, tmp_path):
    calls = {}

    class FakeToNumpy:
        def __init__(self, keys, to_nibabel=False):
            self.keys = keys
            self.to_nibabel = to_nibabel

        def __call__(self, tensor_dict):
            tensor_dict["to_numpy_called"] = True
            return tensor_dict

    def fake_save_session_results(dat, label, loss, tag, results_dir, force_flag=False):
        calls["dat"] = dat
        calls["label"] = label
        calls["loss"] = loss
        calls["tag"] = tag
        calls["results_dir"] = results_dir
        calls["force_flag"] = force_flag

    monkeypatch.setattr(processing, "ToNumpy", FakeToNumpy)
    monkeypatch.setattr(processing, "save_session_results", fake_save_session_results)

    data_dict = {
        "dat": "dat_proxy",
        "label": "label_proxy",
        "id": "sub-001",
    }

    tensor_dict = {
        "ref_image": torch.zeros((1, 1, 2, 2, 2)),
        "flo_image": torch.zeros((1, 1, 2, 2, 2)),
        "reg_image": torch.zeros((1, 1, 2, 2, 2)),
        "affine_ras": torch.eye(4).unsqueeze(0),
    }

    processing._save_subject_outputs(
        data_dict=data_dict,
        tensor_dict=tensor_dict,
        output_dir=str(tmp_path),
        tag="sub-001",
        loss=0.5,
        force_flag=True,
    )

    affine_file = tmp_path / "sub-001_space-symmetricT1w_aff.npy"

    assert affine_file.exists()
    np.testing.assert_allclose(np.load(affine_file), np.eye(4))
    assert calls["dat"] == "dat_proxy"
    assert calls["label"] == "label_proxy"
    assert calls["loss"] == 0.5
    assert calls["tag"] == "sub-001"
    assert calls["results_dir"] == str(tmp_path)
    assert calls["force_flag"] is True


def test_clean_temp_dir_removes_directory(tmp_path):
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("temporary")

    processing._clean_temp_dir(str(temp_dir))

    assert not temp_dir.exists()


def test_clean_temp_dir_does_nothing_if_directory_does_not_exist(tmp_path):
    temp_dir = tmp_path / "missing_tmp"

    processing._clean_temp_dir(str(temp_dir))

    assert not temp_dir.exists()


def test_build_loss_dict_contains_expected_losses():
    loss_dict = processing._build_loss_dict("cpu", _eye())

    assert set(loss_dict.keys()) == {
        "reg",
        "reg_label",
        "reg_uptake",
        "reg_lr",
        "regularizer",
    }

    assert loss_dict["reg"]["weight"] == 1
    assert loss_dict["reg_label"]["weight"] == 2
    assert loss_dict["reg_uptake"]["weight"] == 0.0
    assert loss_dict["reg_lr"]["weight"] == 0.5
    assert loss_dict["regularizer"]["weight"] == 1


def test_process_fn_parallel_returns_function_result():
    def fn(a, b, scale=1):
        return {"value": (a + b) * scale}

    out = processing.process_fn_parallel(fn, 2, 3, scale=10)

    assert out == {"value": 50}


def test_process_fn_parallel_returns_none_on_exception():
    def fn():
        raise RuntimeError("boom")

    out = processing.process_fn_parallel(fn)

    assert out is None


def test_process_subject_returns_none_when_data_dict_is_none():
    args = SimpleNamespace(force=False, opt_str="sgd")

    out = processing.process_subject(
        data_dict=None,
        preproc_tf={},
        dat_tf=[],
        main_dict={"device": "cpu", "output_dir": "unused"},
        args=args,
    )

    assert out is None


def test_process_subject_skips_when_outputs_exist(tmp_path):
    tag = "sub-001"
    (tmp_path / f"{tag}_sbr.tsv").write_text("already processed")

    args = SimpleNamespace(force=False, opt_str="sgd")

    out = processing.process_subject(
        data_dict={"id": tag},
        preproc_tf={},
        dat_tf=[],
        main_dict={"device": "cpu", "output_dir": str(tmp_path)},
        args=args,
    )

    assert out == {"exit": 0}


def test_process_subject_orchestrates_all_steps(monkeypatch, tmp_path):
    calls = []

    args = SimpleNamespace(force=True, opt_str="sgd")

    data_dict = {
        "id": "sub-001",
    }

    run_info = {
        "device": "cpu",
        "output_dir": str(tmp_path),
        "tag": "sub-001",
        "temp_dir": str(tmp_path / "tmp"),
    }

    dat_context = {
        "template_v2r": _eye(),
        "dat_mask": np.ones((2, 2, 2), dtype=bool),
        "brain_dat": np.ones((2, 2, 2), dtype="float32"),
    }

    mri_context = {
        "mri_mask": np.ones((2, 2, 2), dtype=bool),
        "tx_init": np.array([0.0, 0.0, 0.0]),
    }

    tensor_dict = {
        "loss": 0.25,
        "affine_ras": torch.eye(4).unsqueeze(0),
    }

    def fake_prepare_subject_run(data, main):
        calls.append("prepare")
        return run_info

    def fake_should_skip_subject(output_dir, tag, force_flag):
        calls.append("skip_check")
        return False

    def fake_run_preprocessing_step(data_dict, preproc_tf, output_dir, tag):
        calls.append("preprocess")
        data_dict["preprocessed"] = True
        return data_dict

    def fake_build_dat_symmetry_and_masks(data_dict):
        calls.append("dat_context")
        return dat_context

    def fake_simulate_dat_from_mri(data_dict, dat_tf, template_v2r, dat_mask):
        calls.append("simulate")
        return data_dict, mri_context

    def fake_build_registration_tensors(data_dict, dat_context, mri_context, device):
        calls.append("build_tensors")
        return {"built": True}

    def fake_run_registration_step(tensor_dict, data_dict, dat_context,
                                   mri_context, main_dict, args, device):
        calls.append("register")
        return {"loss": 0.25, "affine_ras": torch.eye(4).unsqueeze(0)}

    def fake_save_subject_outputs(data_dict, tensor_dict, output_dir, tag,
                                  loss, force_flag=False):
        calls.append("save")
        assert loss == 0.25
        assert force_flag is True

    def fake_clean_temp_dir(temp_dir):
        calls.append("clean")

    monkeypatch.setattr(processing, "_prepare_subject_run", fake_prepare_subject_run)
    monkeypatch.setattr(processing, "_should_skip_subject", fake_should_skip_subject)
    monkeypatch.setattr(processing, "_run_preprocessing_step", fake_run_preprocessing_step)
    monkeypatch.setattr(processing, "_build_dat_symmetry_and_masks", fake_build_dat_symmetry_and_masks)
    monkeypatch.setattr(processing, "_simulate_dat_from_mri", fake_simulate_dat_from_mri)
    monkeypatch.setattr(processing, "_build_registration_tensors", fake_build_registration_tensors)
    monkeypatch.setattr(processing, "_run_registration_step", fake_run_registration_step)
    monkeypatch.setattr(processing, "_save_subject_outputs", fake_save_subject_outputs)
    monkeypatch.setattr(processing, "_clean_temp_dir", fake_clean_temp_dir)

    out = processing.process_subject(
        data_dict=data_dict,
        preproc_tf={},
        dat_tf=[],
        main_dict={"device": "cpu", "output_dir": str(tmp_path)},
        args=args,
    )

    assert out == {"exit": 0}
    assert calls == [
        "prepare",
        "skip_check",
        "preprocess",
        "dat_context",
        "simulate",
        "build_tensors",
        "register",
        "save",
        "clean",
    ]