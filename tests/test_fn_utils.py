import numpy as np
import pytest
import torch
import nibabel as nib

from darq.utils.fn_utils import (
    get_type_dict,
    convert_to_type,
    convert_to_tensor,
    convert_to_numpy,
    crop_label,
    rescale_voxel_factor,
    fast_3D_interp_torch,
    fast_3D_interp_field_torch,
    vol_resample_fast,
    create_template_space,
    create_template_space_tensor,
    L1Loss,
    L2Loss,
    NCCLoss,
    DiceLoss,
    DiceOverTrueLoss,
    Symmetry,
    MaxUptake,
    DICT_LOSSES,
)


def test_get_type_dict_numpy_array():
    arr = np.array([1, 2, 3], dtype=np.float32)

    info = get_type_dict(arr)

    assert info["type"] is np.ndarray
    assert info["dtype"] == np.float32


def test_get_type_dict_torch_tensor():
    tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

    info = get_type_dict(tensor)

    assert info["type"] is torch.Tensor
    assert info["dtype"] == torch.float32
    assert info["device"] == tensor.device


def test_convert_to_type_numpy_to_tensor():
    arr = np.array([1, 2, 3], dtype=np.float32)

    tensor = convert_to_type(arr, torch.Tensor)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32


def test_convert_to_type_tensor_to_numpy():
    tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

    arr = convert_to_type(tensor, np.ndarray)

    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.array([1, 2, 3], dtype=np.float32))


def test_convert_to_type_unknown_type_returns_original():
    value = "hola"

    out = convert_to_type(value, str)

    assert out == "hola"


def test_convert_to_tensor_from_numpy():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

    tensor = convert_to_tensor(arr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 2)
    assert tensor.dtype == torch.float32


def test_convert_to_tensor_recursive_structures():
    data = {
        "a": [np.array([1, 2], dtype=np.float32)],
        "b": (np.array([3], dtype=np.float32),),
        "c": "text",
    }

    out = convert_to_tensor(data)

    assert isinstance(out["a"][0], torch.Tensor)
    assert isinstance(out["b"][0], torch.Tensor)
    assert out["c"] == "text"


def test_convert_to_numpy_from_tensor():
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

    arr = convert_to_numpy(tensor)

    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_convert_to_numpy_recursive_structures():
    data = {
        "a": [torch.tensor([1, 2], dtype=torch.float32)],
        "b": (torch.tensor([3], dtype=torch.float32),),
        "c": "text",
    }

    out = convert_to_numpy(data)

    assert isinstance(out["a"][0], np.ndarray)
    assert isinstance(out["b"][0], np.ndarray)
    assert out["c"] == "text"


def test_crop_label_crops_nonzero_region_with_margin():
    mask = np.zeros((5, 5, 5), dtype=int)
    mask[2, 2, 1] = 1

    cropped, coords = crop_label(mask, margin=1, threshold=0)

    assert cropped.shape == (2, 2, 2)
    assert coords == [[1, 3], [1, 3], [0, 2]]
    assert cropped.sum() == 1


def test_crop_label_accepts_list_margin():
    mask = np.zeros((6, 6, 6), dtype=int)
    mask[3, 3, 3] = 1

    cropped, coords = crop_label(mask, margin=[1, 2, 1], threshold=0)

    assert cropped.sum() == 1
    assert coords == [[2, 4], [1, 5], [2, 4]]


def test_rescale_voxel_factor_identity_factor_keeps_shape():
    volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    affine = np.eye(4, dtype=np.float32)

    out, out_aff = rescale_voxel_factor(volume, affine, factor=1, method="nearest")

    assert out.shape == volume.shape
    np.testing.assert_allclose(out_aff, affine)


def test_rescale_voxel_factor_upsampling_changes_shape():
    volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    affine = np.eye(4, dtype=np.float32)

    out, out_aff = rescale_voxel_factor(volume, affine, factor=2, method="nearest")

    assert out.shape == (4, 4, 4)
    assert out_aff.shape == (4, 4)


def test_fast_3d_interp_torch_nearest_identity_grid():
    image = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    grid = torch.meshgrid(
        torch.arange(2),
        torch.arange(2),
        torch.arange(2),
        indexing="ij",
    )

    out = fast_3D_interp_torch(image, grid[0], grid[1], grid[2], mode="nearest")

    assert torch.equal(out, image)


def test_fast_3d_interp_torch_linear_center_value():
    image = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    II = torch.tensor([[[0.5]]])
    JJ = torch.tensor([[[0.5]]])
    KK = torch.tensor([[[0.5]]])

    out = fast_3D_interp_torch(image, II, JJ, KK, mode="linear")

    assert torch.isclose(out[0, 0, 0], torch.tensor(3.5))


def test_fast_3d_interp_torch_invalid_mode_raises():
    image = torch.zeros((2, 2, 2))
    grid = torch.zeros((2, 2, 2))

    with pytest.raises(Exception):
        fast_3D_interp_torch(image, grid, grid, grid, mode="invalid")


def test_fast_3d_interp_field_torch_nearest_identity_grid():
    image = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2, 1)
    grid = torch.meshgrid(
        torch.arange(2),
        torch.arange(2),
        torch.arange(2),
        indexing="ij",
    )

    out = fast_3D_interp_field_torch(image, grid[0], grid[1], grid[2], mode="nearest")

    assert out.shape == image.shape
    assert torch.equal(out, image)


def test_fast_3d_interp_field_torch_linear_center_value():
    image = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2, 1)

    II = torch.tensor([[[0.5]]])
    JJ = torch.tensor([[[0.5]]])
    KK = torch.tensor([[[0.5]]])

    out = fast_3D_interp_field_torch(image, II, JJ, KK, mode="linear")

    assert out.shape == (1, 1, 1, 1)
    assert torch.isclose(out[0, 0, 0, 0], torch.tensor(3.5))


def test_vol_resample_fast_identity_returns_same_numpy():
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    affine = np.eye(4, dtype=np.float32)

    ref_proxy = nib.Nifti1Image(data, affine)
    flo_proxy = nib.Nifti1Image(data, affine)

    out = vol_resample_fast(ref_proxy, flo_proxy, mode="nearest", return_np=True)

    assert out.shape == data.shape
    np.testing.assert_allclose(out, data)


def test_vol_resample_fast_identity_returns_nifti():
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    affine = np.eye(4, dtype=np.float32)

    ref_proxy = nib.Nifti1Image(data, affine)
    flo_proxy = nib.Nifti1Image(data, affine)

    out = vol_resample_fast(ref_proxy, flo_proxy, mode="nearest", return_np=False)

    assert isinstance(out, nib.Nifti1Image)
    assert out.shape == data.shape


def test_create_template_space_single_proxy():
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    affine = np.eye(4, dtype=np.float32)

    proxy_list = [
        {
            "data": data,
            "affine": affine,
            "shape": data.shape,
        }
    ]

    images, v2r = create_template_space(proxy_list, resolution=1, mode="nearest")

    assert len(images) == 1
    assert images[0].ndim == 3
    assert v2r.shape == (4, 4)


def test_create_template_space_tensor_single_proxy():
    data = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
    affine = np.eye(4, dtype=np.float32)

    proxy_list = [
        {
            "data": data,
            "affine": affine,
            "shape": data.shape,
        }
    ]

    images, v2r = create_template_space_tensor(proxy_list, resolution=1, mode="nearest")

    assert len(images) == 1
    assert isinstance(images[0], torch.Tensor)
    assert v2r.shape == (4, 4)


def test_l1_and_l2_losses_are_zero_for_equal_tensors():
    x = torch.ones((1, 1, 4, 4))

    assert L1Loss()(x, x).item() == 0
    assert L2Loss()(x, x).item() == 0


def test_l1_loss_with_mask():
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    loss = L1Loss()(prediction, target, mask=mask)

    assert torch.isclose(loss, torch.tensor(2.5))


def test_l2_loss_with_mask():
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    loss = L2Loss()(prediction, target, mask=mask)

    assert torch.isclose(loss, torch.tensor(8.5))


def test_dice_loss_is_zero_for_perfect_overlap():
    x = torch.ones((1, 1, 4, 4))

    loss = DiceLoss()(x, x)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_dice_loss_with_classes_compute():
    prediction = torch.zeros((1, 2, 3, 3))
    target = torch.zeros((1, 2, 3, 3))

    prediction[:, 1] = 1
    target[:, 1] = 1

    loss = DiceLoss()(prediction, target, classes_compute=[1])

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_dice_over_true_loss_is_zero_for_perfect_overlap():
    x = torch.ones((1, 1, 4, 4))

    loss = DiceOverTrueLoss()(x, x)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_ncc_loss_identical_images_is_negative():
    x = torch.zeros((1, 1, 6, 6))
    x[:, :, 2:4, 2:4] = 2

    loss = NCCLoss(device="cpu", kernel_var=[3, 3])(x, x)

    assert loss.item() < 0


def test_ncc_loss_zero_mask_returns_zero():
    x = torch.zeros((1, 1, 6, 6))
    x[:, :, 2:4, 2:4] = 2
    mask = torch.zeros_like(x)

    loss = NCCLoss(device="cpu", kernel_var=[3, 3])(x, x, mask=mask)

    assert loss.item() == 0


def test_ncc_loss_with_mask_is_negative():
    x = torch.zeros((1, 1, 6, 6))
    x[:, :, 2:4, 2:4] = 2
    mask = torch.ones_like(x)

    loss = NCCLoss(device="cpu", kernel_var=[3, 3])(x, x, mask=mask)

    assert loss.item() < 0


def test_ncc_loss_gaussian_kernel_runs():
    x = torch.zeros((1, 1, 8, 8))
    x[:, :, 3:5, 3:5] = 2

    loss = NCCLoss(device="cpu", kernel_type="gaussian")(x, x)

    assert torch.isfinite(loss)


def test_ncc_loss_linear_kernel_not_implemented():
    ncc = NCCLoss(device="cpu", kernel_type="linear")

    with pytest.raises(NotImplementedError):
        ncc._get_kernel("linear", [3, 3])


def test_ncc_loss_1d_and_3d_inputs_run():
    x1 = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    loss1 = NCCLoss(device="cpu", kernel_var=[3])(x1, x1)
    assert torch.isfinite(loss1)

    x3 = torch.zeros((1, 1, 4, 4, 4))
    x3[:, :, 1:3, 1:3, 1:3] = 2
    loss3 = NCCLoss(device="cpu", kernel_var=[3, 3, 3])(x3, x3)
    assert torch.isfinite(loss3)


def test_symmetry_get_flip_r_axis():
    sym = Symmetry(v2r=np.eye(4), axis="r", loss="l1", device="cpu")

    flip = sym._get_flip()

    expected = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(flip.cpu(), expected)


def test_symmetry_get_flip_a_axis():
    sym = Symmetry(v2r=np.eye(4), axis="a", loss="l1", device="cpu")

    flip = sym._get_flip()

    expected = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0]))
    assert torch.equal(flip.cpu(), expected)


def test_symmetry_get_flip_s_axis():
    sym = Symmetry(v2r=np.eye(4), axis="s", loss="l1", device="cpu")

    flip = sym._get_flip()

    expected = torch.diag(torch.tensor([1.0, 1.0, -1.0, 1.0]))
    assert torch.equal(flip.cpu(), expected)


def test_symmetry_get_grid_shape():
    sym = Symmetry(v2r=np.eye(4), axis="r", loss="l1", device="cpu")

    grid = sym._get_grid((2, 3, 4))

    assert grid.shape == (3, 2, 3, 4)


def test_max_uptake_returns_positive_difference():
    prediction = torch.tensor([[[[1.0, 2.0, 5.0]]]])
    target = torch.ones_like(prediction)

    value = MaxUptake()(prediction, target)

    assert value.item() > 0


def test_dict_losses_contains_expected_losses():
    assert DICT_LOSSES["l1"] is L1Loss
    assert DICT_LOSSES["l2"] is L2Loss
    assert DICT_LOSSES["ncc"] is NCCLoss
    assert DICT_LOSSES["dice"] is DiceLoss
    assert DICT_LOSSES["dice_over_true"] is DiceOverTrueLoss
    assert DICT_LOSSES["symmetry"] is Symmetry
    assert DICT_LOSSES["max_uptake"] is MaxUptake