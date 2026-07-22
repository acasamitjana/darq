import darq.config as config


def test_get_device_cpu():
    assert config.get_device(cpu_flag=True) == "cpu"


def test_get_device_cuda(monkeypatch):
    monkeypatch.setattr(
        config.torch.cuda,
        "is_available",
        lambda: True,
    )

    assert config.get_device(cpu_flag=False) == "cuda:0"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(
        config.torch.cuda,
        "is_available",
        lambda: False,
    )

    assert config.get_device(cpu_flag=False) == "cpu"


def test_get_device_explicit_cuda_index(monkeypatch):
    monkeypatch.setattr(
        config.torch.cuda,
        "is_available",
        lambda: True,
    )

    monkeypatch.setattr(
        config.torch.cuda,
        "device_count",
        lambda: 2,
    )

    device = config.get_device(
        cpu_flag=False,
        requested_device="cuda:1",
    )

    assert device == "cuda:1"