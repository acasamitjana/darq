import numpy as np
import pytest
import torch

import darq.src.preprocessing as preprocessing

from darq.src.preprocessing import (
    AlignLR,
    CreateTemplateSpace,
    GaussianBlur,
    MorphologicalOperator,
    ToNumpy,
    Transform,
    get_preprocessing_transforms,
)


def test_to_numpy_converts_tensor_values():
    data = {
        "image": torch.ones((1, 1, 3, 3, 3))
    }

    tf = ToNumpy(keys=["image"])

    out = tf(data)

    assert isinstance(out["image"], np.ndarray)
    assert out["image"].shape == (3, 3, 3)
    assert np.all(out["image"] == 1)


def test_gaussian_blur_keeps_shape_for_3d_array():
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, 3, 3] = 1

    data = {
        "image": image
    }

    tf = GaussianBlur(
        keys=["image"],
        sigma=np.array([1, 1, 1]),
        normalize_area=True,
    )

    out = tf(data)

    assert out["image"].shape == image.shape
    assert np.max(out["image"]) > 0


def test_morphological_operator_opening_keeps_shape():
    image = np.ones((5, 5, 5), dtype=bool)

    data = {
        "mask": image
    }

    tf = MorphologicalOperator(
        keys=["mask"],
        operator="opening",
        struct_fn=lambda x: np.ones((3, 3, 3), dtype=bool),
    )

    out = tf(data)

    assert out["mask"].shape == image.shape


def test_get_preprocessing_transforms_has_expected_keys():
    transforms = get_preprocessing_transforms(device="cpu")

    assert list(transforms.keys()) == [
        "align_dat",
        "align_mri",
        "template",
        "numpy",
    ]

    assert all(callable(tf) for tf in transforms.values())

def test_transform_base_stores_keys_and_call_raises():
    tf = Transform(keys=["image"])

    assert tf.keys == ["image"]

    with pytest.raises(NotImplementedError):
        tf({})

def test_morphological_operator_accepts_5d_input():
    image = np.ones((1, 1, 5, 5, 5), dtype=bool)

    data = {
        "mask": image
    }

    tf = MorphologicalOperator(
        keys=["mask"],
        operator="opening",
        struct_fn=lambda x: np.ones((3, 3, 3), dtype=bool),
    )

    out = tf(data)

    assert isinstance(out["mask"], np.ndarray)
    assert out["mask"].shape == (5, 5, 5)

def test_gaussian_blur_accepts_callable_sigma():
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, 3, 3] = 1.0

    data = {
        "image": image
    }

    tf = GaussianBlur(
        keys=["image"],
        sigma=lambda: np.array([1, 1, 1]),
        normalize_area=True,
    )

    out = tf(data)

    assert out["image"].shape == image.shape
    assert np.max(out["image"]) > 0

def test_gaussian_blur_with_mask_key_only_updates_masked_voxels():
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, 3, 3] = 1.0

    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[3, 3, 3] = True

    data = {
        "image": image.copy(),
        "mask": mask,
    }

    tf = GaussianBlur(
        keys=["image"],
        sigma=np.array([1, 1, 1]),
        mask_key="mask",
        normalize_area=True,
    )

    out = tf(data)

    assert out["image"].shape == image.shape
    assert out["image"][3, 3, 3] > 0
    assert np.count_nonzero(out["image"][~mask]) == 0

def test_alignlr_init_converts_scalar_factors_to_arrays():
    tf = AlignLR(
        keys=["dat_v2r"],
        ref_im="dat_brain",
        ref_v2r="dat_v2r",
        angle_factor=2.0,
        tx_factor=0.5,
        device="cpu",
    )

    np.testing.assert_array_equal(tf.angle_factor, np.array([2.0, 2.0, 2.0]))
    np.testing.assert_array_equal(tf.tx_factor, np.array([0.5, 0.5, 0.5]))

def test_alignlr_compute_cog_returns_expected_translation():
    tf = AlignLR(
        keys=["dat_v2r"],
        ref_im="dat_brain",
        ref_v2r="dat_v2r",
        device="cpu",
    )

    mask = np.zeros((5, 5, 5), dtype=np.float32)
    mask[2, 3, 4] = 1.0

    v2r = np.eye(4)

    cog = tf._compute_cog(mask, v2r)

    expected = np.eye(4)
    expected[0, 3] = -2
    expected[1, 3] = -3
    expected[2, 3] = -4

    np.testing.assert_allclose(cog, expected)

def test_alignlr_call_adds_rotation_and_aligned_affines(monkeypatch):
    tf = AlignLR(
        keys=["dat_v2r"],
        ref_im=["dat_brain"],
        ref_v2r="dat_v2r",
        device="cpu",
    )

    def fake_align_lr(proxy):
        return np.eye(4), np.eye(4)

    monkeypatch.setattr(tf, "_align_LR", fake_align_lr)

    data = {
        "dat_brain": np.ones((4, 4, 4), dtype=np.float32),
        "dat_v2r": np.eye(4),
    }

    out = tf(data)

    assert "rot_dat_v2r" in out
    assert "aligned_dat_v2r" in out
    np.testing.assert_allclose(out["rot_dat_v2r"], np.eye(4))
    np.testing.assert_allclose(out["aligned_dat_v2r"], np.eye(4))

def test_create_template_space_run_array_handles_transpose_and_no_transpose(monkeypatch):
    tf = CreateTemplateSpace(
        keys={
            "image_a": "affine_a",
            "image_b": "affine_b",
        },
        name="template",
    )

    def fake_create_template_space(proxy_list, mode, resolution):
        images = [
            np.ones((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32) * 2,
        ]
        return images, np.eye(4)

    monkeypatch.setattr(
        preprocessing.fn_utils,
        "create_template_space",
        fake_create_template_space,
    )

    proxy_list = [
        {
            "data": np.ones((2, 3, 4)),
            "affine": np.eye(4),
            "shape": (2, 3, 4),
            "undo_tr": (1, 0, 2),
        },
        {
            "data": np.ones((2, 3, 4)),
            "affine": np.eye(4),
            "shape": (2, 3, 4),
            "undo_tr": False,
        },
    ]

    out = tf.run_array(proxy_list)

    assert "template_v2r" in out
    assert "template_space" in out
    assert "template_image_a" in out
    assert "template_image_b" in out
    assert out["template_image_a"].shape == (3, 2, 4)
    assert out["template_image_b"].shape == (2, 3, 4)

def test_create_template_space_run_tensor_handles_transpose_and_no_transpose(monkeypatch):
    tf = CreateTemplateSpace(
        keys={
            "image_a": "affine_a",
            "image_b": "affine_b",
        },
        name="template",
    )

    def fake_create_template_space_tensor(proxy_list, mode, resolution):
        images = [
            torch.ones((2, 3, 4)),
            torch.ones((2, 3, 4)) * 2,
        ]
        return images, np.eye(4)

    monkeypatch.setattr(
        preprocessing.fn_utils,
        "create_template_space_tensor",
        fake_create_template_space_tensor,
    )

    proxy_list = [
        {
            "data": torch.ones((2, 3, 4)),
            "affine": np.eye(4),
            "shape": (2, 3, 4),
            "undo_tr": (1, 0, 2),
        },
        {
            "data": torch.ones((2, 3, 4)),
            "affine": np.eye(4),
            "shape": (2, 3, 4),
            "undo_tr": False,
        },
    ]

    out = tf.run_tensor(proxy_list)

    assert "template_v2r" in out
    assert "template_space" in out
    assert "template_image_a" in out
    assert "template_image_b" in out
    assert out["template_image_a"].shape == (3, 2, 4)
    assert out["template_image_b"].shape == (2, 3, 4)

def test_create_template_space_call_with_numpy_arrays_uses_run_array(monkeypatch):
    tf = CreateTemplateSpace(
        keys={
            "image_4d": "affine_4d",
            "image_5d": "affine_5d",
            "image_3d": "affine_3d",
        },
        name="template",
    )

    captured = {}

    def fake_run_array(proxy_list):
        captured["proxy_list"] = proxy_list
        return {
            "template_v2r": np.eye(4),
            "template_space": np.zeros((2, 3, 4)),
        }

    monkeypatch.setattr(tf, "run_array", fake_run_array)

    data = {
        "image_4d": np.ones((1, 2, 3, 4), dtype=np.float32),
        "affine_4d": np.eye(4),
        "image_5d": np.ones((1, 1, 2, 3, 4), dtype=np.float32),
        "affine_5d": np.eye(4),
        "image_3d": np.ones((2, 3, 4), dtype=np.float32),
        "affine_3d": np.eye(4),
    }

    out = tf(data)

    assert "template_v2r" in out
    assert "template_space" in out

    proxy_list = captured["proxy_list"]

    assert proxy_list[0]["undo_tr"] == (3, 0, 1, 2)
    assert proxy_list[1]["undo_tr"] == (3, 4, 0, 1, 2)
    assert proxy_list[2]["undo_tr"] is False

def test_create_template_space_call_with_tensors_uses_run_tensor(monkeypatch):
    tf = CreateTemplateSpace(
        keys={
            "image_4d": "affine_4d",
            "image_5d": "affine_5d",
            "image_3d": "affine_3d",
        },
        name="template",
    )

    captured = {}

    def fake_run_tensor(proxy_list):
        captured["proxy_list"] = proxy_list
        return {
            "template_v2r": np.eye(4),
            "template_space": torch.zeros((2, 3, 4)),
        }

    monkeypatch.setattr(tf, "run_tensor", fake_run_tensor)

    data = {
        "image_4d": torch.ones((1, 2, 3, 4)),
        "affine_4d": np.eye(4),
        "image_5d": torch.ones((1, 1, 2, 3, 4)),
        "affine_5d": np.eye(4),
        "image_3d": torch.ones((2, 3, 4)),
        "affine_3d": np.eye(4),
    }

    out = tf(data)

    assert "template_v2r" in out
    assert "template_space" in out

    proxy_list = captured["proxy_list"]

    assert proxy_list[0]["undo_tr"] == (3, 0, 1, 2)
    assert proxy_list[1]["undo_tr"] == (3, 4, 0, 1, 2)
    assert proxy_list[2]["undo_tr"] is False

def test_to_numpy_with_to_nibabel_transposes_channel_first_tensor():
    data = {
        "image": torch.ones((1, 2, 3, 4, 5))
    }

    tf = ToNumpy(keys=["image"], to_nibabel=True)

    out = tf(data)

    assert isinstance(out["image"], np.ndarray)
    assert out["image"].shape == (3, 4, 5, 2)

