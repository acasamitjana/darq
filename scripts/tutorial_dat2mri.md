# Dat2MRI tutorial

This tutorial explains how to run the `dat2mri.py` script from the DARQ pipeline.

The `dat2mri.py` script is the command-line entry point of the DaTSCAN-to-MRI pipeline. It reads the input paths, builds the subject dataset, applies the preprocessing transforms, runs the MRI-DaT registration and saves the final quantitative results. The script uses `parse_args()`, `build_subject()`, `MRI_DaT`, `get_preprocessing_transforms()`, `get_dat_transforms()` and `process_subject()` to run the full pipeline. 

---

## 1. Goal of the script

The goal of `dat2mri.py` is to process one DaTSCAN image together with a companion MRI image and a SynthSeg segmentation.

The pipeline uses:

- A DaTSCAN image.
- A T1-weighted MRI image.
- A SynthSeg segmentation image.
- An output directory.

The main outputs are:

- Registered DaTSCAN images.
- Affine transformation matrices.
- Regional DaT uptake tables.
- Symmetry metric tables.

---

## 2. Tutorial input files

For this tutorial, the example input files are stored in the `tutorial/` folder:

```text
tutorial/
├── T1w.nii.gz
├── datscan.nii.gz
└── synthseg.nii.gz
```

These files are medical images in NIfTI format.

### `T1w.nii.gz`

This is the T1-weighted MRI image.

It is the anatomical reference image used by the pipeline.

### `datscan.nii.gz`

This is the DaTSCAN image.

It is the functional image that will be registered and quantified.

### `synthseg.nii.gz`

This is the SynthSeg segmentation image.

It contains anatomical labels used to build the masks required for quantification.

---

## 3. Current limitation

The current implementation requires both the MRI image and the SynthSeg segmentation.

Running the script without `--mri` or `--seg` is not supported yet because template-based registration is not implemented in the current version.

Therefore, the following arguments are required:

```bash
--dat
--mri
--seg
--o
```

---

## 4. Set the project path

Before running the script, make sure that the project root is available in `PYTHONPATH`.

From the root of the repository, run:

```bash
export PYTHONPATH=$(pwd)
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

This is needed so that Python can correctly find the `darq` package and the files inside the `data/` directory.

---

## 5. Basic command

To run the tutorial example on CPU, use:

```bash
py scripts/dat2mri.py \
    --dat data/tutorial/datscan.nii.gz \
    --mri data/tutorial/T1w.nii.gz \
    --seg data/tutorial/synthseg.nii.gz \
    --o data/tutorial/output \
    --cpu
```

This command will process the example files and save the results inside:

```text
data/tutorial/output/
```

---

## 6. Windows command prompt version

If you are using Windows Command Prompt instead of Git Bash, use `^` instead of `\`:

```cmd
py scripts/dat2mri.py ^
    --dat data/tutorial/datscan.nii.gz ^
    --mri data/tutorial/T1w.nii.gz ^
    --seg data/tutorial/synthseg.nii.gz ^
    --o data/tutorial/output ^
    --cpu
```

If you are using Git Bash, use the previous version with `\`.

---

## 7. GPU execution

If CUDA is available and the required GPU libraries are correctly installed, the pipeline can be run without the `--cpu` flag:

```bash
py scripts/dat2mri.py \
    --dat data/tutorial/datscan.nii.gz \
    --mri data/tutorial/T1w.nii.gz \
    --seg data/tutorial/synthseg.nii.gz \
    --o data/tutorial/output
```

If there is any CUDA-related error, run the script on CPU by adding:

```bash
--cpu
```

---

## 8. Command-line arguments

The script accepts the following command-line arguments:

| Argument | Description |
|---|---|
| `--dat` | Path to the input DaTSCAN image. |
| `--mri` | Path to the input MRI image. |
| `--seg` | Path to the input SynthSeg segmentation image. |
| `--o` | Path to the output directory. |
| `--opt_str` | Optimizer used during registration. Available options are `lbfgs`, `adam` and `sgd`. |
| `--num_cores` | Number of parallel workers. Use `1` for sequential execution. |
| `--force` | Recompute the results even if output files already exist. |
| `--cpu` | Force the pipeline to run on CPU instead of CUDA. |

---

## 9. Optimizer options

By default, the pipeline uses the `lbfgs` optimizer.

### Default optimizer: LBFGS

```bash
py scripts/dat2mri.py \
    --dat data/tutorial/datscan.nii.gz \
    --mri data/tutorial/T1w.nii.gz \
    --seg data/tutorial/synthseg.nii.gz \
    --o data/tutorial/output \
    --cpu
```

### Adam optimizer

```bash
py scripts/dat2mri.py \
    --dat data/tutorial/datscan.nii.gz \
    --mri data/tutorial/T1w.nii.gz \
    --seg data/tutorial/synthseg.nii.gz \
    --o data/tutorial/output \
    --opt_str adam \
    --cpu
```

### SGD optimizer

```bash
py scripts/dat2mri.py \
    --dat data/tutorial/datscan.nii.gz \
    --mri data/tutorial/T1w.nii.gz \
    --seg data/tutorial/synthseg.nii.gz \
    --o data/tutorial/output \
    --opt_str sgd \
    --cpu
```

---

## 10. Recomputing existing results

If the output files already exist, the pipeline may skip the subject to avoid overwriting previous results.

To force the computation again, use:

```bash
py scripts/dat2mri.py \
    --dat data/tutorial/datscan.nii.gz \
    --mri data/tutorial/T1w.nii.gz \
    --seg data/tutorial/synthseg.nii.gz \
    --o data/tutorial/output \
    --force \
    --cpu
```

The `--force` flag is useful when testing changes in the code or rerunning the tutorial.

---

## 11. Expected terminal output

When the script starts correctly, the terminal should show something similar to:

```text
# ------------------------- #
# Running Dat2MRI algorithm #
# ------------------------- #

Total sessions to process: N=1

Subject: 0  (0/1)
 * Preprocessing.
 * DaT symmetry and mask.
 * MRI DaT simulation and mask.
 * MRI-DaT registration.
```

During registration, the terminal may also print training information such as the optimizer, losses and epoch summaries.

At the end, if the pipeline finishes correctly, the terminal should show:

```text
# ---------#
# All DONE #
# ---------#
```

---

## 12. Expected output files

After running the script, the output directory should contain files similar to:

```text
output/
├── 0_dat.nii.gz
├── 0_desc-resampled_dat.nii.gz
├── 0_sbr.tsv
├── 0_space-dat_rot.npy
├── 0_space-symmetricT1w_aff.npy
├── 0_space-T1w_aff.npy
├── 0_space-T1wdseg_rot.npy
└── 0_symm.tsv
```

---

## 13. Output description

| Output file | Description |
|---|---|
| `0_dat.nii.gz` | DaTSCAN image after applying the estimated affine transformation. |
| `0_desc-resampled_dat.nii.gz` | DaTSCAN image resampled into the MRI/SynthSeg space. |
| `0_sbr.tsv` | Table containing regional DaT uptake values. |
| `0_space-dat_rot.npy` | Rotation/alignment matrix estimated for the DaTSCAN image during preprocessing. |
| `0_space-symmetricT1w_aff.npy` | Affine matrix estimated in the symmetric T1-weighted space. |
| `0_space-T1w_aff.npy` | Final affine matrix that maps the DaTSCAN image to the T1-weighted MRI space. |
| `0_space-T1wdseg_rot.npy` | Rotation/alignment matrix estimated for the MRI/SynthSeg image during preprocessing. |
| `0_symm.tsv` | Table containing left-right symmetry metrics. |

---

## 14. What happens inside the pipeline?

The script runs several processing steps.

### Step 1: Argument parsing

The script reads the input paths and options from the command line.

Example:

```bash
--dat data/tutorial/datscan.nii.gz
--mri data/tutorial/T1w.nii.gz
--seg data/tutorial/synthseg.nii.gz
--o data/tutorial/output
--cpu
```

### Step 2: Subject preparation

The input paths are converted into a subject/session table.

In the current tutorial example, the table contains one subject.

### Step 3: Dataset loading

The dataset loads:

- The MRI image.
- The SynthSeg segmentation.
- The DaTSCAN image.

It also prepares anatomical masks from the SynthSeg labels.

### Step 4: Preprocessing

The preprocessing step aligns the DaTSCAN and MRI images to a common template space.

It also saves intermediate rotation matrices.

### Step 5: DaT mask creation

The pipeline estimates the high-uptake DaT region using intensity-based clustering and anatomical constraints.

### Step 6: MRI-DaT registration

The pipeline registers the DaTSCAN-derived mask to the MRI-derived mask using a rigid registration model.

### Step 7: Result saving

The final affine matrix, registered images and quantitative tables are saved in the output directory.

---