Dat2MRI Tutorial
================

Overview
--------

This tutorial explains how to run the ``dat2mri.py`` processing script from the
DARQ project.

The script parses the command-line arguments, selects the execution device,
builds the subject dataset, creates the preprocessing transforms, executes the
DaTSCAN-to-MRI processing pipeline and writes the generated results to the
selected output directory.

The current implementation supports processing a single study from explicit
input paths and can also build a subject list from BIDS-style inputs when the
corresponding command-line options are provided.

The processing loop in the current version of ``dat2mri.py`` is sequential.
Although ``joblib`` parallel-processing utilities are imported, they are kept
for future batch-processing support and are not currently used by the main
processing loop.


1. Goal of the script
---------------------

The goal of ``dat2mri.py`` is to process a DaTSCAN image together with a
T1-weighted MRI image and a SynthSeg segmentation.

For the single-subject workflow, the pipeline uses:

* A DaTSCAN image.
* A T1-weighted MRI image.
* A SynthSeg segmentation image.
* An output directory.

The main outputs include:

* Registered and resampled DaTSCAN images.
* Affine transformation matrices.
* Striatal Binding Ratio (SBR) tables.
* Left-right symmetry metric tables.


2. Tutorial input files
-----------------------

For this tutorial, the example input files are stored in the project tutorial
data directory:

.. code-block:: text

   data/tutorial/
   |-- T1w.nii.gz
   |-- datscan.nii.gz
   `-- synthseg.nii.gz

These files use the NIfTI medical-image format.

``T1w.nii.gz``
^^^^^^^^^^^^^^

T1-weighted MRI image used as the anatomical reference.

``datscan.nii.gz``
^^^^^^^^^^^^^^^^^^

Functional DaTSCAN image that will be registered and quantified.

``synthseg.nii.gz``
^^^^^^^^^^^^^^^^^^^

SynthSeg segmentation containing the anatomical labels used to build the masks
required by the DARQ pipeline.


3. Current input limitation
---------------------------

The current single-subject workflow requires both the MRI image and the
SynthSeg segmentation.

Template-based processing without MRI and segmentation is not supported by the
current implementation.

For the standard tutorial workflow, the required paths are therefore:

.. code-block:: text

   --dat
   --mri
   --seg
   --o


4. Running from the source repository
-------------------------------------

When running the script directly from the source repository, the project root
must be importable by Python.

If DARQ is already installed in the active Python environment, no additional
``PYTHONPATH`` configuration should normally be necessary.

If DARQ is not installed, run the following command from the repository root.

Linux / Git Bash
^^^^^^^^^^^^^^^^

.. code-block:: bash

   export PYTHONPATH=$(pwd)

Windows PowerShell
^^^^^^^^^^^^^^^^^^

.. code-block:: powershell

   $env:PYTHONPATH = (Get-Location).Path


5. Basic CPU command
--------------------

For a reproducible tutorial execution on CPU:

.. code-block:: bash

   py scripts/dat2mri.py \
       --dat data/tutorial/datscan.nii.gz \
       --mri data/tutorial/T1w.nii.gz \
       --seg data/tutorial/synthseg.nii.gz \
       --o data/tutorial/output \
       --cpu

The generated results are written to:

.. code-block:: text

   data/tutorial/output/


6. Windows Command Prompt
-------------------------

When using Windows Command Prompt instead of PowerShell or Git Bash, use ``^``
for line continuation:

.. code-block:: batch

   py scripts/dat2mri.py ^
       --dat data/tutorial/datscan.nii.gz ^
       --mri data/tutorial/T1w.nii.gz ^
       --seg data/tutorial/synthseg.nii.gz ^
       --o data/tutorial/output ^
       --cpu


7. Device selection and GPU execution
-------------------------------------

The current ``dat2mri.py`` obtains the execution device through
``darq.config.get_device()``.

The device-selection function receives both the ``--cpu`` flag and the device
request supplied by the command-line parser.

For a CPU-only tutorial run, use:

.. code-block:: text

   --cpu

For GPU-capable environments, the script can be executed without forcing CPU:

.. code-block:: bash

   py scripts/dat2mri.py \
       --dat data/tutorial/datscan.nii.gz \
       --mri data/tutorial/T1w.nii.gz \
       --seg data/tutorial/synthseg.nii.gz \
       --o data/tutorial/output

At startup, the script prints the selected execution device:

.. code-block:: text

   Execution device: <device>

The exact accepted values and default behaviour of the optional device
selection argument are defined by the current ``darq.cli`` implementation.

When DARQ is executed through the web application and worker infrastructure,
device selection is handled by that deployment workflow rather than by this
tutorial command.


8. Main command-line arguments
------------------------------

The single-subject tutorial uses the following arguments:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Description
   * - ``--dat``
     - Path to the input DaTSCAN image.
   * - ``--mri``
     - Path to the input T1-weighted MRI image.
   * - ``--seg``
     - Path to the input SynthSeg segmentation image.
   * - ``--o``
     - Path to the output directory.
   * - ``--opt_str``
     - Optimizer selected for rigid registration.
   * - ``--force``
     - Recompute a subject even when existing result files are detected.
   * - ``--cpu``
     - Force execution on CPU.

The current ``dat2mri.py`` also reads additional options such as device and
BIDS-related settings through ``parse_args()``. Their exact syntax is defined
in ``darq.cli``.

``--num_cores`` is intentionally not described as enabling parallel execution
here because the current main processing loop is sequential.


9. Optimizer options
--------------------

The registration optimizer is selected with ``--opt_str``.

The processing code supports:

* ``lbfgs``
* ``adam``
* ``sgd``

LBFGS
^^^^^

If LBFGS is the configured default, the tutorial can be run without specifying
``--opt_str``:

.. code-block:: bash

   py scripts/dat2mri.py \
       --dat data/tutorial/datscan.nii.gz \
       --mri data/tutorial/T1w.nii.gz \
       --seg data/tutorial/synthseg.nii.gz \
       --o data/tutorial/output \
       --cpu

Adam
^^^^

.. code-block:: bash

   py scripts/dat2mri.py \
       --dat data/tutorial/datscan.nii.gz \
       --mri data/tutorial/T1w.nii.gz \
       --seg data/tutorial/synthseg.nii.gz \
       --o data/tutorial/output \
       --opt_str adam \
       --cpu

SGD
^^^

.. code-block:: bash

   py scripts/dat2mri.py \
       --dat data/tutorial/datscan.nii.gz \
       --mri data/tutorial/T1w.nii.gz \
       --seg data/tutorial/synthseg.nii.gz \
       --o data/tutorial/output \
       --opt_str sgd \
       --cpu


10. Recomputing existing results
--------------------------------

DARQ checks whether a subject already has generated results.

If the SBR output already exists and ``--force`` is not enabled, the subject
can be skipped to avoid recomputation.

To force a new execution:

.. code-block:: bash

   py scripts/dat2mri.py \
       --dat data/tutorial/datscan.nii.gz \
       --mri data/tutorial/T1w.nii.gz \
       --seg data/tutorial/synthseg.nii.gz \
       --o data/tutorial/output \
       --force \
       --cpu


11. What happens when the script starts?
----------------------------------------

The current ``dat2mri.py`` performs the following high-level sequence:

1. Clear the terminal.
2. Parse command-line arguments with ``parse_args()``.
3. Select the execution device with ``get_device()``.
4. Build either a single subject or a BIDS subject list.
5. Create an ``MRI_DaT`` dataset.
6. Build the preprocessing transforms.
7. Build the transforms used to simulate DaT uptake from the MRI-derived mask.
8. Create the registration configuration.
9. Process each subject sequentially with ``process_subject()``.
10. Report any failed subjects.
11. Print the final completion message.


12. Expected terminal output
----------------------------

A successful start produces output similar to:

.. code-block:: text

   # ------------------------- #
   # Running Dat2MRI algorithm #
   # ------------------------- #

   Execution device: cpu
   Total sessions to process: N=1

   Subject: 0  (1/1)
    * Preprocessing.
    * DaT symmetry and mask.
    * MRI DaT simulation and mask.
    * MRI-DaT registration.

During registration, additional information about the optimizer, configured
losses and final registration loss can also be printed.

When processing ends, the script reports the number of failed subjects and
prints:

.. code-block:: text

   # ---------#
   # All DONE #
   # ---------#


13. Expected output files
-------------------------

For the tutorial subject, the output directory can contain files such as:

.. code-block:: text

   output/
   |-- 0_dat.nii.gz
   |-- 0_desc-resampled_dat.nii.gz
   |-- 0_sbr.tsv
   |-- 0_space-dat_rot.npy
   |-- 0_space-symmetricT1w_aff.npy
   |-- 0_space-T1w_aff.npy
   |-- 0_space-T1wdseg_rot.npy
   `-- 0_symm.tsv

The exact set of files depends on the current result-saving implementation.


14. Output description
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Output
     - Description
   * - ``0_dat.nii.gz``
     - DaTSCAN image after applying the estimated transformation.
   * - ``0_desc-resampled_dat.nii.gz``
     - DaTSCAN image resampled into the MRI / segmentation space.
   * - ``0_sbr.tsv``
     - Table containing regional Striatal Binding Ratio (SBR) measurements.
   * - ``0_space-dat_rot.npy``
     - Rotation / alignment matrix estimated for the DaTSCAN image during preprocessing.
   * - ``0_space-symmetricT1w_aff.npy``
     - Affine matrix saved from the MRI-DaT registration in the symmetric T1-weighted space.
   * - ``0_space-T1w_aff.npy``
     - Final affine transformation associated with the DaTSCAN-to-T1w result.
   * - ``0_space-T1wdseg_rot.npy``
     - Rotation / alignment matrix estimated for the MRI / SynthSeg data during preprocessing.
   * - ``0_symm.tsv``
     - Table containing left-right symmetry measurements.


15. Pipeline details
--------------------

Step 1: Argument parsing
^^^^^^^^^^^^^^^^^^^^^^^^

``parse_args()`` reads the paths and execution options supplied from the
command line.

Step 2: Device selection
^^^^^^^^^^^^^^^^^^^^^^^^

``get_device()`` chooses the execution device using the CPU flag and the device
request returned by the command-line parser.

Step 3: Subject preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For explicit input paths, ``build_subject()`` creates the subject/session table.

When BIDS-related input options are provided, ``build_subject_list()`` is used
instead.

Step 4: Dataset loading
^^^^^^^^^^^^^^^^^^^^^^^

``MRI_DaT`` loads the MRI, SynthSeg segmentation and DaTSCAN image and prepares
the anatomical and DaT-derived masks required by the pipeline.

Step 5: Preprocessing
^^^^^^^^^^^^^^^^^^^^^

``get_preprocessing_transforms()`` builds the transformations used to align the
DaTSCAN and MRI-derived data and create the common template space.

Intermediate rotation matrices are saved during this stage.

Step 6: DaT symmetry and mask generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The processing pipeline builds a left-right symmetric representation of the
DaTSCAN data and estimates the high-uptake mask using clustering and anatomical
constraints.

Step 7: MRI-DaT preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MRI-derived striatal mask is transformed into a DaT-like representation.
This is used together with the DaT mask to estimate the initial translation for
registration.

Step 8: MRI-DaT registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A rigid registration model aligns the DaT-derived and MRI-derived information
using the optimizer selected by ``--opt_str``.

Step 9: Result saving
^^^^^^^^^^^^^^^^^^^^^

The final affine matrix, registered images and quantitative results are written
to the selected output directory.


16. Relationship with the web application
-----------------------------------------

This tutorial describes direct execution of ``scripts/dat2mri.py``.

The deployed DARQ web application uses additional infrastructure around the
processing pipeline:

* Gradio provides the user interface.
* FastAPI creates and exposes processing jobs.
* A DARQ worker executes the pipeline.
* Job metadata stores processing state and progress.

Therefore, the command-line tutorial is useful for understanding and testing
the core processing workflow independently of the web deployment.
