from darq.config import get_device

def test_get_device_cpu():
    assert get_device(True) == "cpu"


def test_get_device_cuda():
    assert get_device(False) == "cuda"