import sys
import pytest
from darq.cli import parse_args

def test_parse_args_minimal_required_arguments(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datreg",
            "--dat", "dat.nii.gz",
            "--mri", "mri.nii.gz",
            "--seg", "seg.nii.gz",
            "--o", "out"
        ],
    )
    
    args = parse_args()

    assert args.dat == "dat.nii.gz"
    assert args.mri == "mri.nii.gz"
    assert args.seg == "seg.nii.gz"
    assert args.o == "out"
    assert args.opt_str == "lbfgs"
    assert args.num_cores == 1
    assert args.force is False 
    assert args.cpu is False

def test_parse_args_accepts_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datreg",
            "--dat", "dat.nii.gz",
            "--mri", "mri.nii.gz",
            "--seg", "seg.nii.gz",
            "--o", "out",
            "--force",
            "--cpu",
            "--num_cores", "2",
            "--opt_str", "adam",

        ],
    )

    args = parse_args()

    assert args.dat == "dat.nii.gz"
    assert args.mri == "mri.nii.gz"
    assert args.seg == "seg.nii.gz"
    assert args.o == "out"
    assert args.force is True
    assert args.cpu is True
    assert args.num_cores == 2
    assert args.opt_str == "adam"

def test_parse_args_rejects_invalid_optimizer(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datreg",
            "--dat", "dat.nii.gz",
            "--mri", "mri.nii.gz",
            "--seg", "seg.nii.gz",
            "--o", "out",
            "--opt_str", "invalid_optimizer"
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()
