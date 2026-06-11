import numpy as np

from darq.utils import io


def test_create_dir_creates_default_subdirectories(tmp_path):
    out = tmp_path / "experiment"

    io.create_dir(str(out))

    assert (out / "checkpoints").exists()
    assert (out / "results").exists()


def test_create_dir_creates_custom_subdirectories(tmp_path):
    out = tmp_path / "experiment"

    io.create_dir(str(out), subdirs=["a", "b"])

    assert (out / "a").exists()
    assert (out / "b").exists()


def test_create_dir_adds_missing_subdirectory_when_root_exists(tmp_path):
    out = tmp_path / "experiment"
    out.mkdir()

    io.create_dir(str(out), subdirs=["checkpoints", "results"])

    assert out.exists()
    assert (out / "checkpoints").exists()
    assert (out / "results").exists()


def test_callback_base_methods_return_none():
    callback = io.Callback()

    assert callback.on_train_init(model={}) is None
    assert callback.on_train_fi(model={}) is None
    assert callback.on_epoch_init(model={}, epoch=0) is None
    assert callback.on_epoch_fi(logs_dict={}, model={}, epoch=0) is None
    assert callback.on_step_init(logs_dict={}, model={}, epoch=0) is None
    assert callback.on_step_fi(logs_dict={}, model={}, epoch=0) is None


def test_printer_callback_can_be_instantiated():
    callback = io.PrinterCallback(keys=["loss"], freq_print=1)

    assert callback.keys == ["loss"]
    assert callback.freq_print == 1
    assert callback.logs == {}


def test_printer_callback_on_train_init_prints_banner(capsys):
    callback = io.PrinterCallback(keys=["loss"], freq_print=1)

    callback.on_train_init(model={})

    captured = capsys.readouterr()

    assert "Training started" in captured.out


def test_printer_callback_on_epoch_init_prints_epoch(capsys):
    callback = io.PrinterCallback(keys=["loss"], freq_print=1)

    callback.on_epoch_init(model={}, epoch=3)

    captured = capsys.readouterr()

    assert "Epoch: 3" in captured.out


def test_printer_callback_on_step_fi_prints_selected_metrics(capsys):
    callback = io.PrinterCallback(keys=["loss"], freq_print=1)

    callback.on_step_fi(
        logs_dict={
            "loss": [1.2345, 2.3456],
            "ignored_metric": [99],
        },
        model={},
        epoch=0,
        iteration=2,
        N=10,
    )

    captured = capsys.readouterr()

    assert "Iteration: (2/10)" in captured.out
    assert "loss:" in captured.out
    assert "ignored_metric" not in captured.out


def test_printer_callback_on_step_fi_does_not_print_when_frequency_not_met(capsys):
    callback = io.PrinterCallback(keys=["loss"], freq_print=2)

    callback.on_step_fi(
        logs_dict={"loss": [1.0]},
        model={},
        epoch=0,
        iteration=1,
        N=10,
    )

    captured = capsys.readouterr()

    assert captured.out == ""


def test_printer_callback_on_epoch_fi_prints_summary(capsys):
    callback = io.PrinterCallback(keys=["loss"], freq_print=1)

    callback.on_epoch_fi(
        logs_dict={
            "loss": 1.2345,
            "ignored_metric": 99,
        },
        model={},
        epoch=0,
    )

    captured = capsys.readouterr()

    assert "Epoch summary" in captured.out
    assert "loss:" in captured.out
    assert "ignored_metric" not in captured.out


def test_printer_callback_on_train_fi_prints_banner(capsys):
    callback = io.PrinterCallback(keys=["loss"], freq_print=1)

    callback.on_train_fi(model={})

    captured = capsys.readouterr()

    assert "Training finished" in captured.out


def test_remove_synthseg_parcellation_relabels_context_labels(monkeypatch):
    monkeypatch.setattr(io, "ctx_labels", np.array([0, 1005, 2005]))

    seg = np.array([0, 1005, 2005])

    out = io.remove_synthseg_parcellation(seg.copy())

    np.testing.assert_array_equal(out, np.array([0, 3, 42]))


def test_remove_synthseg_parcellation_keeps_non_context_labels(monkeypatch):
    monkeypatch.setattr(io, "ctx_labels", np.array([0, 1005, 2005]))

    seg = np.array([11, 1005, 2005, 9999])

    out = io.remove_synthseg_parcellation(seg.copy())

    np.testing.assert_array_equal(out, np.array([11, 3, 42, 9999]))


def test_remove_synthseg_hemisphere_maps_right_labels_to_left(monkeypatch):
    monkeypatch.setattr(
        io,
        "SYNTHSEG_DICT_REV",
        {
            "left caudate": 11,
            "right caudate": 50,
        },
    )

    seg = np.array([11, 50])

    out = io.remove_synthseg_hemisphere(seg.copy())

    np.testing.assert_array_equal(out, np.array([11, 11]))


def test_remove_synthseg_hemisphere_keeps_left_labels(monkeypatch):
    monkeypatch.setattr(
        io,
        "SYNTHSEG_DICT_REV",
        {
            "left caudate": 11,
            "right caudate": 50,
            "left putamen": 12,
            "right putamen": 51,
        },
    )

    seg = np.array([11, 12])

    out = io.remove_synthseg_hemisphere(seg.copy())

    np.testing.assert_array_equal(out, np.array([11, 12]))