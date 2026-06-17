# darq

<p align="center">
  <strong>DaTSCAN Automatic Registration and Quantification</strong>
</p>

<p align="center">
   <img src="assets/darq_logo.png" alt="DARQ logo" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/testing-pytest-blue" alt="pytest" />
  <img src="https://img.shields.io/badge/docs-Sphinx%20%7C%20ReadTheDocs-blue" alt="Documentation" />
  <img src="https://img.shields.io/badge/status-development-orange" alt="Development status" />
  <img src="https://img.shields.io/badge/code%20style-clean-lightgrey" alt="Code style" />
</p>


---

## Overview

**darq** is a Python-based pipeline for the automatic registration and quantification of DaTSCAN images.

The project aims to provide an open-source tool for DaTSCAN quantification, supporting the analysis of striatal uptake biomarkers from DaTSCAN SPECT images. When available, MRI and SynthSeg segmentations are used to perform subject-specific registration and anatomical quantification.

The current pipeline focuses on code cleaning, documentation, unit testing, modular design and preparation for future deployment as a command-line tool and web application.

## Main Features

- Automatic DaTSCAN processing pipeline
- MRI-DaTSCAN rigid registration
- SynthSeg-based anatomical mask extraction
- Striatal uptake quantification
- Caudate and putamen biomarker extraction
- Symmetry metric computation
- Command-line interface
- Modular Python structure
- Unit testing with `pytest`
- Documentation with Sphinx and ReadTheDocs

## Getting Started

### Clone the repository

```bash
git clone https://github.com/your-username/darq.git
cd darq
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

DARQ can be executed from the command line by providing a DaTSCAN image, an MRI image, a SynthSeg segmentation and an output directory.

```bash
python -m darq.dat2mri \
  --dat path/to/dat.nii.gz \
  --mri path/to/mri.nii.gz \
  --seg path/to/synthseg.nii.gz \
  --o path/to/output_directory \
  --cpu
```

To recompute results even if previous outputs already exist, use:

```bash
python -m darq.dat2mri \
  --dat path/to/dat.nii.gz \
  --mri path/to/mri.nii.gz \
  --seg path/to/synthseg.nii.gz \
  --o path/to/output_directory \
  --cpu \
  --force
```

## Command-line Arguments

| Argument | Description |
|---|---|
| `--dat` | Input DaTSCAN image. |
| `--mri` | Input MRI image. |
| `--seg` | Input SynthSeg segmentation. |
| `--o` | Output directory. |
| `--opt_str` | Optimizer used for registration: `lbfgs`, `adam` or `sgd`. |
| `--num_cores` | Number of parallel workers. |
| `--force` | Recompute results even if they already exist. |
| `--cpu` | Force CPU execution. |

## Inputs

The pipeline currently expects the following medical imaging files:

- A DaTSCAN SPECT image.
- A T1-weighted MRI image.
- A SynthSeg segmentation associated with the MRI image.

The MRI and SynthSeg inputs are used to extract anatomical masks and perform subject-specific registration.

## Outputs

DARQ saves registration outputs and quantitative results for each processed subject/session.

Generated outputs include:

- Registered DaTSCAN images.
- Resampled DaTSCAN images.
- Affine transformation matrices.
- Striatal uptake metrics.
- Symmetry metrics.
- TSV result files.

Typical output files include:

```text
output_directory/
├── <subject>_dat.nii.gz
├── <subject>_desc-resampled_dat.nii.gz
├── <subject>_space-T1w_aff.npy
├── <subject>_space-symmetricT1w_aff.npy
├── <subject>_space-T1wdseg_rot.npy
├── <subject>_space-dat_rot.npy
├── <subject>_sbr.tsv
└── <subject>_symm.tsv
```

## Pipeline Overview

The main processing workflow is organized as follows:

1. Parse command-line arguments.
2. Select the computation device.
3. Build the subject/session dataframe.
4. Load the DaTSCAN, MRI and SynthSeg images.
5. Apply preprocessing transforms.
6. Estimate DaTSCAN and MRI masks.
7. Run MRI-DaTSCAN registration.
8. Save images, affine matrices and quantitative results.

## Project Structure

```text
darq/
├── cli.py
├── config.py
├── dat2mri.py
├── processing.py
├── src/
│   ├── datasets.py
│   ├── models.py
│   ├── preprocessing.py
│   └── results.py
└── utils/
    ├── fn_utils.py
    ├── io.py
    └── loader.py
```

## Documentation

The project documentation is generated with Sphinx and can be published using ReadTheDocs.

To build the documentation locally:

```bash
cd docs
make html
```

The generated HTML documentation will be available in:

```text
docs/build/html/index.html
```

## Testing

Run the test suite with:

```bash
pytest
```

To check test coverage:

```bash
pytest --cov=darq
```

## Development Status

DARQ is currently under active development as part of a project at VICOROB, University of Girona.

Current development goals include:

- Cleaning and simplifying the existing code.
- Improving documentation and docstrings.
- Increasing unit test coverage.
- Improving the modular design of the processing pipeline.
- Integrating GitHub Actions workflows.
- Preparing the project for future packaging and deployment.
- Exploring a future web interface using Gradio or FastAPI.

## Contributing

Contributions, suggestions and feedback are welcome.

Possible ways to contribute include:

- Reporting issues.
- Improving documentation.
- Adding tests.
- Refactoring code.
- Suggesting new features.
- Improving the processing pipeline.

## Acknowledgements

This project is developed in the context of a biomedical engineering internship at VICOROB, University of Girona.

## License

To be defined.
