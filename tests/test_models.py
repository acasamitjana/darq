import numpy as np
import pytest
import torch
import torch.nn as nn

from darq.src.models import (
    InstanceModelClassic,
    InstanceAlignModelClassic,
    InstanceRigidModelClassic,
    JointInstanceAlign,
    JointInstanceReg,
)


class DummyInstanceModel(InstanceModelClassic):
    def _compute_ras_matrix(self, *args, **kwargs):
        eye = torch.eye(4, dtype=self.ref_v2r.dtype, device=self.ref_v2r.device)
        return eye.unsqueeze(0).repeat(self.batchsize, 1, 1)

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return (torch.ones((self.batchsize, 3), device=self.ref_v2r.device),)


class DummyLoss(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, prediction, target=None, mask=None, *args, **kwargs):
        if target is None:
            return prediction.mean()

        diff = prediction - target

        if mask is not None and torch.is_tensor(mask):
            diff = diff * mask

        return torch.mean(diff ** 2)


class FakeAlignReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.offset, x - self.offset

    def get_params(self):
        return (self.offset.reshape(1, 1),)

    def get_params_scaled(self):
        return (self.offset.reshape(1, 1),)

    def get_matrix(self):
        return torch.eye(4).unsqueeze(0)

    def get_ras_matrix(self):
        return torch.eye(4).unsqueeze(0)


class FakeReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.offset

    def get_params(self):
        return (self.offset.reshape(1, 1),)

    def get_params_scaled(self):
        return (self.offset.reshape(1, 1),)

    def get_matrix(self):
        return torch.eye(4).unsqueeze(0)

    def get_ras_matrix(self):
        return torch.eye(4).unsqueeze(0)


def _eye():
    return np.eye(4, dtype="float32")


def _image():
    return torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)


def _train_config(num_epochs=1):
    return {
        "starting_epoch": 0,
        "num_epochs": num_epochs,
        "patience": 1,
        "freq_print": 1,
    }


def _align_loss_dict():
    return {
        "symmetry": {
            "loss": DummyLoss("symmetry"),
            "weight": 1,
        },
        "symmetry_im": {
            "loss": DummyLoss("symmetry_im"),
            "weight": 1,
        },
        "regularizer": {
            "loss": DummyLoss("regularizer"),
            "weight": 1,
        },
    }


def _reg_loss_dict():
    return {
        "reg": {
            "loss": DummyLoss("reg"),
            "weight": 1,
        },
        "reg_label": {
            "loss": DummyLoss("reg_label"),
            "weight": 1,
        },
        "reg_uptake": {
            "loss": DummyLoss("reg_uptake"),
            "weight": 1,
        },
        "reg_lr": {
            "loss": DummyLoss("reg_lr"),
            "weight": 1,
        },
        "regularizer": {
            "loss": DummyLoss("regularizer"),
            "weight": 1,
        },
    }


def _reg_data():
    return {
        "flo_image": torch.ones((1, 1, 3, 3, 3), dtype=torch.float32),
        "flo_mask": torch.ones((1, 1, 3, 3, 3), dtype=torch.float32),
        "ref_image": torch.zeros((1, 2, 3, 3, 3), dtype=torch.float32),
        "ref_mask": torch.ones((1, 1, 3, 3, 3), dtype=torch.float32),
        "template_v2r": _eye(),
    }


def test_base_model_abstract_methods_raise():
    model = InstanceModelClassic(
        image_shape=(1, 1, 3, 3, 3),
        ref_v2r=_eye(),
        device="cpu",
    )

    assert model.image_shape == (3, 3, 3)

    with pytest.raises(NotImplementedError):
        model._compute_ras_matrix()

    with pytest.raises(NotImplementedError):
        model.set_params(None)

    with pytest.raises(NotImplementedError):
        model.get_params()

    with pytest.raises(NotImplementedError):
        model.get_params_scaled()


def test_base_model_subclass_matrix_and_forward():
    model = DummyInstanceModel(
        image_shape=(1, 1, 3, 3, 3),
        ref_v2r=_eye(),
        flo_v2r=_eye(),
        device="cpu",
    )

    matrix = model.get_matrix()
    params = model.get_params_scaled()
    image = _image()
    output = model(image)

    assert matrix.shape == (1, 4, 4)
    assert params[0].shape == (1, 3)
    assert output.shape == image.shape


def test_instance_rigid_model_initial_params_are_zero():
    model = InstanceRigidModelClassic(
        image_shape=(3, 3, 3),
        ref_v2r=_eye(),
        flo_v2r=_eye(),
        tx_factor=np.ones(3, dtype="float32"),
        angle_factor=np.ones(3, dtype="float32"),
        device="cpu",
    )

    angle, translation = model.get_params()

    assert angle.shape == (1, 3)
    assert translation.shape == (1, 3)
    assert torch.allclose(angle, torch.zeros_like(angle))
    assert torch.allclose(translation, torch.zeros_like(translation))


def test_instance_rigid_model_identity_ras_matrix():
    model = InstanceRigidModelClassic(
        image_shape=(3, 3, 3),
        ref_v2r=_eye(),
        flo_v2r=_eye(),
        tx_factor=np.ones(3, dtype="float32"),
        angle_factor=np.ones(3, dtype="float32"),
        device="cpu",
    )

    matrix = model.get_ras_matrix()
    expected = torch.eye(4, dtype=matrix.dtype, device=matrix.device)

    assert matrix.shape == (1, 4, 4)
    assert torch.allclose(matrix[0], expected, atol=1e-6)


def test_instance_rigid_model_accepts_numpy_initial_params_and_cog():
    model = InstanceRigidModelClassic(
        image_shape=(3, 3, 3),
        ref_v2r=_eye(),
        flo_v2r=_eye(),
        cog=np.array([1, 2, 3], dtype="float32"),
        angle_init=np.array([[0.1, 0.2, 0.3]], dtype="float32"),
        tx_init=np.array([[1, 2, 3]], dtype="float32"),
        tx_factor=np.array([2, 3, 4], dtype="float32"),
        angle_factor=np.array([5, 6, 7], dtype="float32"),
        device="cpu",
    )

    angle, translation = model.get_params()
    scaled_angle, scaled_translation = model.get_params_scaled()
    matrix = model.get_ras_matrix()

    assert torch.allclose(angle, torch.tensor([[0.1, 0.2, 0.3]]))
    assert torch.allclose(translation, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(scaled_angle, angle * torch.tensor([5, 6, 7]))
    assert torch.allclose(scaled_translation, translation * torch.tensor([2, 3, 4]))
    assert matrix.shape == (1, 4, 4)


def test_instance_rigid_model_accepts_tensor_initial_params_and_set_params():
    model = InstanceRigidModelClassic(
        image_shape=(3, 3, 3),
        ref_v2r=_eye(),
        flo_v2r=_eye(),
        angle_init=torch.ones((1, 3), dtype=torch.float32),
        tx_init=torch.ones((1, 3), dtype=torch.float32),
        tx_factor=np.ones(3, dtype="float32"),
        angle_factor=np.ones(3, dtype="float32"),
        device="cpu",
    )

    new_angle = torch.zeros((1, 3), dtype=torch.float32)
    new_translation = torch.full((1, 3), 2.0, dtype=torch.float32)

    model.set_params((new_angle, new_translation))
    angle, translation = model.get_params()

    assert torch.allclose(angle, new_angle)
    assert torch.allclose(translation, new_translation)


def test_instance_align_forward_returns_regular_and_flipped_images():
    model = InstanceAlignModelClassic(
        image_shape=(3, 3, 3),
        v2r=_eye(),
        tx_factor=torch.ones(3, dtype=torch.float32),
        angle_factor=torch.ones(3, dtype=torch.float32),
        device="cpu",
    )

    image = _image()

    reg, flipped = model(image)

    assert reg.shape == image.shape
    assert flipped.shape == image.shape


def test_instance_align_model_accepts_numpy_initial_params_set_params_and_cog():
    model = InstanceAlignModelClassic(
        image_shape=(3, 3, 3),
        v2r=_eye(),
        cog=np.array([1, 2, 3], dtype="float32"),
        angle_init=np.array([[0.1, 0.2, 0.3]], dtype="float32"),
        tx_init=np.array([[1, 2, 3]], dtype="float32"),
        tx_factor=torch.tensor([2.0, 3.0, 4.0]),
        angle_factor=torch.tensor([5.0, 6.0, 7.0]),
        device="cpu",
    )

    new_angle = torch.zeros((1, 3), dtype=torch.float32)
    new_translation = torch.ones((1, 3), dtype=torch.float32)

    model.set_params((new_angle, new_translation))
    scaled_angle, scaled_translation = model.get_params_scaled()
    matrix = model._compute_ras_matrix(flip_lr=True)

    assert torch.allclose(scaled_angle, new_angle * torch.tensor([5.0, 6.0, 7.0]))
    assert torch.allclose(scaled_translation, new_translation * torch.tensor([2.0, 3.0, 4.0]))
    assert matrix.shape == (1, 4, 4)


def test_instance_align_model_accepts_tensor_initial_params():
    model = InstanceAlignModelClassic(
        image_shape=(3, 3, 3),
        v2r=_eye(),
        angle_init=torch.ones((1, 3), dtype=torch.float32),
        tx_init=torch.ones((1, 3), dtype=torch.float32),
        tx_factor=torch.ones(3, dtype=torch.float32),
        angle_factor=torch.ones(3, dtype=torch.float32),
        device="cpu",
    )

    angle, translation = model.get_params()

    assert torch.allclose(angle, torch.ones((1, 3)))
    assert torch.allclose(translation, torch.ones((1, 3)))


def test_joint_instance_align_initialization_adds_val_freq_and_callbacks():
    config = _train_config()

    session = JointInstanceAlign(
        _align_loss_dict(),
        config,
        device="cpu",
        trainable_keys="reg",
        verbose=True,
        freq_print=1,
    )

    assert session.trainable_keys == {"reg": "reg"}
    assert config["val_freq"] == 5
    assert len(session.callbacks) == 1


def test_joint_instance_align_forward_with_image_and_mask():
    session = JointInstanceAlign(
        _align_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    data = {
        "image": _image(),
        "mask": _image(),
    }

    output = session.forward(data, {"reg": FakeAlignReg()})

    assert output["reg_image"].shape == _image().shape
    assert output["reg_image_flip"].shape == _image().shape
    assert output["reg_mask"].shape == _image().shape
    assert output["reg_mask_flip"].shape == _image().shape


def test_joint_instance_align_forward_with_only_mask():
    session = JointInstanceAlign(
        _align_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    data = {
        "mask": _image(),
    }

    output = session.forward(data, {"reg": FakeAlignReg()})

    assert output["reg_mask"].shape == _image().shape
    assert output["reg_mask_flip"].shape == _image().shape


def test_joint_instance_align_compute_loss_with_all_branches():
    session = JointInstanceAlign(
        _align_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    model = {"reg": FakeAlignReg()}
    data = {
        "image": _image(),
        "mask": _image(),
    }

    data = session.forward(data, model)
    loss, logs = session.compute_loss(data, model)

    assert torch.is_tensor(loss)
    assert "loss_symmetry" in logs
    assert "loss_symmetry_im" in logs
    assert "loss_regularizer" in logs


def test_joint_instance_align_iterate_with_sgd():
    session = JointInstanceAlign(
        _align_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    reg = FakeAlignReg()
    optimizer = torch.optim.SGD(reg.parameters(), lr=0.1)

    logs = session.iterate(
        {"mask": _image()},
        {"reg": reg},
        {"reg": optimizer},
    )

    assert "loss" in logs


def test_joint_instance_align_iterate_with_lbfgs():
    session = JointInstanceAlign(
        _align_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    reg = FakeAlignReg()
    optimizer = torch.optim.LBFGS(reg.parameters(), lr=0.1, max_iter=1)

    logs = session.iterate(
        {"mask": _image()},
        {"reg": reg},
        {"reg": optimizer},
    )

    assert "loss" in logs


def test_joint_instance_align_register_runs_training_and_returns_outputs():
    session = JointInstanceAlign(
        _align_loss_dict(),
        _train_config(num_epochs=2),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    reg = FakeAlignReg()
    optimizer = torch.optim.SGD(reg.parameters(), lr=0.1)

    output = session.register(
        {"mask": _image()},
        {"reg": reg},
        {"reg": optimizer},
    )

    assert "parameters" in output
    assert "loss" in output
    assert "affine" in output
    assert "affine_ras" in output
    assert "reg_mask" in output
    assert "reg_mask_flip" in output


def test_joint_instance_reg_forward_with_and_without_mask():
    session = JointInstanceReg(
        _reg_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    model = {"reg": FakeReg()}

    data_with_mask = session.forward(_reg_data(), model)
    assert data_with_mask["reg_image"].shape == _image().shape
    assert data_with_mask["reg_mask"].shape == _image().shape

    data_without_mask = session.forward(
        {
            "flo_image": _image(),
        },
        model,
    )

    assert data_without_mask["reg_image"].shape == _image().shape
    assert "reg_mask" not in data_without_mask


def test_joint_instance_reg_compute_loss_with_all_branches():
    session = JointInstanceReg(
        _reg_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    model = {"reg": FakeReg()}
    data = session.forward(_reg_data(), model)

    loss, logs = session.compute_loss(data, model)

    assert torch.is_tensor(loss)
    assert "loss_reg" in logs
    assert "loss_reg_label" in logs
    assert "loss_reg_uptake" in logs
    assert "loss_reg_lr" in logs
    assert "loss_regularizer" in logs
    assert "loss" in logs


def test_joint_instance_reg_iterate_with_sgd():
    session = JointInstanceReg(
        _reg_loss_dict(),
        _train_config(),
        device="cpu",
        trainable_keys={"reg": "reg"},
        verbose=False,
    )

    reg = FakeReg()
    optimizer = torch.optim.SGD(reg.parameters(), lr=0.1)

    logs = session.iterate(
        _reg_data(),
        {"reg": reg},
        {"reg": optimizer},
    )

    assert "loss" in logs

