import argparse
import os
import pdb
from os import listdir
from os.path import join, exists, isdir

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# from data import labels                          # shared library: ImageSynthesis/data/labels.py
# from darqsrc.datasets import MRI_DaT                # shared library: ImageSynthesis/src/datasets.py


def _normalize_subject_id(subject: str) -> str:
    """Ensure subject ID has the 'sub-PPMI' prefix."""
    if 'PPMI' not in subject:
        return 'sub-PPMI' + subject
    return subject


def _read_subject_sessions(subject: str, part_df: pd.DataFrame,
                            input_dir: str, output_dir: str,
                            seg_dir: str, args: argparse.Namespace,
                            dice_constraint: bool = False) -> dict | None:
    """Return session-level file paths for one subject, or None if skipped."""
    subject_info = {
        'mri_path': [], 'label_path': [], 'dat_path': [],
        'subject': [], 'dx': [], 'sess': [],
    }

    if not isdir(join(input_dir, subject)):
        return None

    sess_df = pd.read_csv(join(input_dir, subject, subject + '_sessions.tsv'), sep='\t')
    sess_df = sess_df.drop_duplicates(subset=['session_id']).set_index('session_id')

    for sess in listdir(join(input_dir, subject)):
        anat_dir  = join(input_dir, subject, sess, 'anat')
        spect_dir = join(input_dir, subject, sess, 'spect')
        if not exists(anat_dir) or not exists(spect_dir):
            continue
        if not exists(join(seg_dir, subject, sess, 'anat')):
            continue

        # Skip already-processed sessions unless --force
        if exists(join(output_dir, subject, subject + '_desc-sbr_sessions.tsv')):
            res_df = pd.read_csv(
                join(output_dir, subject, subject + '_desc-sbr_sessions.tsv'), sep='\t')
            if sess in res_df.session.to_list() and not args.force:
                continue

        # DaTSCAN file
        dat_files = [f for f in listdir(spect_dir) if 'DaTSCAN' in f and f.endswith('.nii.gz')]
        if not dat_files:
            continue
        dat_file = join(spect_dir, dat_files[0])

        # MRI file (prefer T1w, fallback to FLAIR then T2w)
        t1w   = [f for f in listdir(anat_dir) if 'T1w'   in f and f.endswith('.nii.gz')]
        flair = [f for f in listdir(anat_dir) if 'FLAIR'  in f and f.endswith('.nii.gz')]
        t2w   = [f for f in listdir(anat_dir) if 't2w'    in f and f.endswith('.nii.gz')]
        if t1w:
            mri_file = t1w[0]
        elif flair:
            mri_file = flair[0]
        elif t2w:
            mri_file = t2w[0]
        else:
            continue

        # Segmentation file derived from MRI filename
        parts = mri_file.split('_')
        ext_parts = parts[-1].split('.')
        ext_parts[0] += 'dseg'
        seg_fname = '_'.join(parts[:-1]) + '_' + '.'.join(ext_parts)

        im_file  = join(anat_dir,  mri_file)
        seg_file = join(seg_dir, subject, sess, 'anat', seg_fname)

        dat_proxy = nib.load(dat_file)
        mri_proxy = nib.load(im_file)
        if len(mri_proxy.shape) != 3 or len(dat_proxy.shape) != 3:
            continue
        if not all(s > 10 for s in dat_proxy.shape):
            continue

        subject_info['mri_path'].append(im_file)
        subject_info['label_path'].append(seg_file)
        subject_info['dat_path'].append(dat_file)
        subject_info['subject'].append(subject)
        subject_info['sess'].append(sess)
        subject_info['dx'].append(part_df.loc[subject[4:]].cohort)

    return subject_info if any(subject_info['subject']) else None

def build_subject(args: argparse.Namespace) -> pd.DataFrame:
    """Scan BIDS directory and return a DataFrame of all sessions to process."""

    if args.mri is None or args.seg is None:
        raise NotImplementedError("Template-based registration still not implementes")

    df = {
        'subject': [0],
        'id': [str(0)],
        'mri_path': [args.mri],
        'label_path': [args.seg],
        'dat_path': [args.dat],
    }

    df = pd.DataFrame(df)
    df.set_index('subject', inplace=True, drop=False)
    return df

def build_subject_list(args: argparse.Namespace,
                       bids_dir: str,
                       seg_dir: str,
                       dat_reg_dir: str) -> pd.DataFrame:
    """Scan BIDS directory and return a DataFrame of all sessions to process."""
    part_df = pd.read_csv(join(bids_dir, 'participants.tsv'), sep='\t')
    part_df = part_df.set_index('participant_id')

    candidates = args.subjects if args.subjects is not None else listdir(bids_dir)

    rows = {k: [] for k in ('mri_path', 'label_path', 'dat_path', 'subject', 'dx', 'sess')}
    for subject in tqdm(candidates):
        sid = _normalize_subject_id(str(subject))
        info = _read_subject_sessions(sid, part_df, bids_dir, dat_reg_dir, seg_dir, args)
        if info is None:
            continue
        for k in rows:
            rows[k].extend(info[k])

    df = pd.DataFrame(rows)
    df.set_index(['subject', 'sess'], inplace=True, drop=False)
    return df

#
# def build_dataset(subject_list: pd.DataFrame) -> MRI_DaT:
#     """Wrap subject list in the MRI_DaT PyTorch Dataset."""
#     return MRI_DaT(subject_list, transforms=[], crop_labels=True, crop_dat=False)


def build_synthseg_lut(labels_module) -> dict:
    """Integer LUT mapping SynthSeg label IDs to contiguous indices."""
    all_labels = np.sort(np.array([0] + list(labels_module.SYNTHSEG_DICT.keys())))
    return {k: it_k for it_k, k in enumerate(all_labels)}
