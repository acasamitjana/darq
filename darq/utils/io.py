"""Save per-session registration outputs and compute SBR / symmetry metrics."""
from os import makedirs
from os.path import join, exists
from importlib.resources import files, as_file

import numpy as np


def create_dir(results_dir: str, subdirs: list | None = None) -> None:
    """Create a results directory and its expected subdirectories.
    :param results_dir: Root directory to create.
    :param subdirs: Optional list of subdirectories created inside the root directory.

    :return: None. Directories are created on disk if they do not already exist.
    """

    if subdirs is None:
        subdirs = ['checkpoints', 'results']

    if not exists(results_dir):
        for sd in subdirs:
            makedirs(join(results_dir, sd))
    else:
        for sd in subdirs:
            if not exists(join(results_dir, sd)):
                makedirs(join(results_dir, sd))

class Callback(object):
    """Base callback interface used by the training and registration loops.
    Subclasses can override any hook to add logging, checkpointing or custom side
    effects.
    """

    def on_train_init(self, model, **kwargs) -> None:
        """Hook called when training starts.

        :param model: Model dictionary used by the training loop.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        pass

    def on_train_fi(self, model, **kwargs) -> None:
        """Hook called when training finishes.

        :param model: Model dictionary used by the training loop.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        pass

    def on_epoch_init(self, model, epoch, **kwargs) -> None:
        """Hook called at the beginning of an epoch.

        :param model: Model dictionary used by the training loop.
        :param epoch: Current epoch index.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        pass

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs) -> None:
        """Hook called at the end of an epoch.

        :param logs_dict: Dictionary of metrics collected during the epoch.
        :param model: Model dictionary used by the training loop.
        :param epoch: Current epoch index.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        pass

    def on_step_init(self, logs_dict, model, epoch, **kwargs) -> None:
        """Hook called at the beginning of an iteration step.

        :param logs_dict: Dictionary of metrics collected during the step.
        :param model: Model dictionary used by the training loop.
        :param epoch: Current epoch index.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        pass

    def on_step_fi(self, logs_dict, model, epoch,**kwargs) -> None:
        """Hook called at the end of an iteration step.

        :param logs_dict: Dictionary of metrics collected during the step.
        :param model: Model dictionary used by the training loop.
        :param epoch: Current epoch index.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        pass

class PrinterCallback(Callback):
    """Callback that prints training progress and selected metrics to the console."""
    def __init__(self, keys: list | None = None, freq_print: int = 1) -> None:
        """Initialize the console printer callback.

        :param keys: Metric names that should be printed.
        :param freq_print: Frequency, in iterations, used when printing step-level logs.
        """

        self.keys = keys
        self.freq_print = freq_print
        self.logs = {}

    def on_train_init(self, model, **kwargs) -> None:
        """Print a training-start banner.

        :param model: Model dictionary used by the training loop.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        print('    ####################')
        print('    # Training started #')
        print('    ####################')
        print('\n')

    def on_epoch_init(self, model, epoch, **kwargs) -> None:
        """Print the current epoch index.

        :param model: Model dictionary used by the training loop.
        :param epoch: Current epoch index.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        print('    * Epoch: ' + str(epoch))

    def on_step_fi(self, logs_dict, model, epoch, **kwargs) -> None:
        """Print selected metrics at the end of a training step.

        :param logs_dict: Dictionary of step metrics.
        :param model: Model dictionary used by the training loop.
        :param epoch: Current epoch index.
        :param kwargs: Additional context such as iteration number and total number of
            iterations.

        :return: None.
        """

        if np.mod(kwargs['iteration'], self.freq_print) == 0:
            to_print = '      o Iteration: (' + str(kwargs['iteration']) + '/' + str(kwargs['N']) + '). '
            to_print += ', '.join([k + ': ' + str(round(np.mean(v), 3)) for k, v in logs_dict.items() if k in self.keys])
            print(to_print)

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs) -> None:
        """Print a summary of metrics for the current epoch.

        :param logs_dict: Dictionary of epoch metrics.
        :param model: Model dictionary used by the training loop.
        :param epoch: Current epoch index.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """
        to_print = '      Epoch summary: ' + ','.join([k + ': ' + str(round(v, 3)) for k, v in logs_dict.items() if k in self.keys])
        print(to_print)
        print('\n')


    def on_train_fi(self, model, **kwargs) -> None:
        """Print a training-finished banner.

        :param model: Model dictionary used by the training loop.
        :param kwargs: Additional context provided by the caller.

        :return: None.
        """

        print('    #####################')
        print('    # Training finished #')
        print('    #####################')
        print('\n')

# ── Labels ─────────────────────────────────────────────────────────────────
def _load_label_resource(filename: str) -> np.ndarray:
    """Load a packaged SynthSeg label resource.

    :param filename: Name of the packaged .npy resource.

    :return: Loaded NumPy array.
    """

    resource = files("darq.data.labels_classes_priors").joinpath(filename)

    with as_file(resource) as path:
        return np.load(path, allow_pickle=True)


ctx_labels = _load_label_resource("synthseg_parcellation_labels.npy")
ctx_names = _load_label_resource("synthseg_parcellation_names.npy")

subcortical_labels = _load_label_resource("synthseg_segmentation_labels.npy")
subcortical_labels = np.concatenate((subcortical_labels, [24]))

subcortical_names = _load_label_resource("synthseg_segmentation_names.npy")
subcortical_names = np.concatenate((subcortical_names, ["csf"]))

SYNTHSEG_DICT = {k: v for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}
SYNTHSEG_DICT_REV = {v: k for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}

def remove_synthseg_parcellation(seg_array: np.ndarray) -> np.ndarray:
    """Collapse SynthSeg cortical parcellation labels into hemisphere-level labels.

    :param seg_array: SynthSeg segmentation array modified in place.

    :return: Segmentation array with cortical parcellation labels replaced by broad
        hemisphere labels.
    """

    for k in ctx_labels:
        if k == 0:
            continue
        elif k < 2000:
            seg_array[np.where(seg_array==k)] = 3
        else:
            seg_array[np.where(seg_array==k)] = 42

    return seg_array

def remove_synthseg_hemisphere(seg_array: np.ndarray) -> np.ndarray:
    """Map right-hemisphere SynthSeg labels to their left-hemisphere equivalents.
    
    :param seg_array: SynthSeg segmentation array modified in place.

    :return: Segmentation array with right labels converted to corresponding left
        labels.
    """
    
    for k, v in SYNTHSEG_DICT_REV.items():
        if 'right' in k:
            seg_array[seg_array == v] = SYNTHSEG_DICT_REV[k.replace('right', 'left')]

    return seg_array
