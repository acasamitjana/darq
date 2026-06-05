"""Save per-session registration outputs and compute SBR / symmetry metrics."""
import os
from os import makedirs
from os.path import join, exists

import numpy as np


def create_dir(results_dir, subdirs=None):
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

    def on_train_init(self, model, **kwargs):
        pass

    def on_train_fi(self, model, **kwargs):
        pass

    def on_epoch_init(self, model, epoch, **kwargs):
        pass

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_init(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_fi(self, logs_dict, model, epoch,**kwargs):
        pass

class PrinterCallback(Callback):

    def __init__(self, keys=None, freq_print=1):
        self.keys = keys
        self.freq_print = freq_print
        self.logs = {}

    def on_train_init(self, model, **kwargs):
        print('    ####################')
        print('    # Training started #')
        print('    ####################')
        print('\n')

    def on_epoch_init(self, model, epoch, **kwargs):
        print('    * Epoch: ' + str(epoch))

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        if np.mod(kwargs['iteration'], self.freq_print) == 0:
            to_print = '      o Iteration: (' + str(kwargs['iteration']) + '/' + str(kwargs['N']) + '). '
            to_print += ', '.join([k + ': ' + str(round(np.mean(v), 3)) for k, v in logs_dict.items() if k in self.keys])
            print(to_print)

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        to_print = '      Epoch summary: ' + ','.join([k + ': ' + str(round(v, 3)) for k, v in logs_dict.items() if k in self.keys])
        print(to_print)
        print('\n')


    def on_train_fi(self, model, **kwargs):
        # print('#####################')
        print('    #####################')
        print('    # Training finished #')
        print('    #####################')
        print('\n')
        # print('#####################')

# ── Labels ─────────────────────────────────────────────────────────────────
repo_home = os.environ.get('PYTHONPATH')
ctx_labels = np.load(join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_labels.npy'))
ctx_names = np.load(join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_names.npy'))

subcortical_labels = np.load(join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_labels.npy'))
subcortical_labels = np.concatenate((subcortical_labels, [24]))
subcortical_names = np.load(join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_names.npy'))
subcortical_names = np.concatenate((subcortical_names, ['csf']))

SYNTHSEG_DICT = {k: v for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}
SYNTHSEG_DICT_REV = {v: k for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}

def remove_synthseg_parcellation(seg_array):
    for k in ctx_labels:
        if k == 0:
            continue
        elif k < 2000:
            seg_array[np.where(seg_array==k)] = 3
        else:
            seg_array[np.where(seg_array==k)] = 42

    return seg_array

def remove_synthseg_hemisphere(seg_array):
    for k, v in SYNTHSEG_DICT_REV.items():
        if 'right' in k:
            seg_array[seg_array == v] = SYNTHSEG_DICT_REV[k.replace('right', 'left')]

    return seg_array

