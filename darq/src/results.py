"""Save per-session registration outputs and compute SBR / symmetry metrics."""
from os.path import join, exists
from functools import partial

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from darq.utils.fn_utils import NCCLoss, fast_3D_interp_torch, vol_resample_fast


# ── Label indices for striatum / occipital cortex ────────────────────────────
_STR_LABELS  = {'R': (50, 51), 'L': (11, 12)}
_CAU_LABELS  = {'R': (50,),    'L': (11,)}
_PUT_LABELS  = {'R': (51,),    'L': (12,)}
_OCC_LABELS  = {
    'R': (2005, 2011, 2013, 2021),
    'L': (1005, 1011, 1013, 1021),
}


def _build_mask(lab_image: np.ndarray, label_ids: tuple) -> np.ndarray:
    """Build a binary mask from selected label IDs.

    :param lab_image: Input label image.
    :param label_ids: Tuple of integer label IDs included in the mask.

    :return: Boolean mask where selected labels are True.
    """

    return np.isin(lab_image, label_ids)


def _bilateral(d: dict) -> np.ndarray:
    """Combine right and left hemisphere masks into a bilateral mask.

    :param d: Dictionary containing ``'R'`` and ``'L'`` masks.

    :return: Boolean mask representing the union of right and left masks.
    """

    return d['R'] | d['L']


def save_session_results(dat_orig_proxy: nib.Nifti1Image,
                        labels_orig_proxy: nib.Nifti1Image,
                        loss,
                        tag: str,
                        results_dir: str,
                        force_flag: bool = False,) -> None:
    """Resample the registered DaT image to MRI space and save session metrics.

    :param dat_orig_proxy: Original DaT NIfTI image proxy.
    :param labels_orig_proxy: Original SynthSeg label NIfTI image proxy used as MRI-space
        reference.
    :param loss: Final registration loss value.
    :param tag: Subject/session identifier used in output filenames.
    :param results_dir: Directory where images and TSV files are written.
    :param force_flag: If True, replace previous rows for the same subject/session when
        writing TSV files.

    :return: None. Output images and tables are written to disk.
    """

    sess_dir  = results_dir

    affine_path = join(sess_dir, tag + '_space-symmetricT1w_aff.npy')
    if not exists(affine_path):
        print('[Warning] Affine matrix not available.')
        return

    affine_ras = np.load(affine_path)
    if np.any(np.isnan(affine_ras)):
        print('[Warning] Affine matrix contains NaN.')
        return

    mri_rot = np.load(join(sess_dir, tag + '_space-T1wdseg_rot.npy'))
    dat_rot = np.load(join(sess_dir, tag + '_space-dat_rot.npy'))

    dat_image = np.squeeze(np.array(dat_orig_proxy.dataobj).astype('float32'))
    dat_v2r   = dat_orig_proxy.affine
    affine    = np.linalg.inv(dat_rot) @ affine_ras @ mri_rot

    np.save(join(sess_dir, tag + '_space-T1w_aff.npy'), np.linalg.inv(affine))

    # Resample DaT to MRI space
    dat_proxy = nib.Nifti1Image(dat_image, np.linalg.inv(affine) @ dat_v2r)
    nib.save(dat_proxy, join(sess_dir, tag + '_dat.nii.gz'))
    dat_proxy = vol_resample_fast(labels_orig_proxy, dat_proxy)
    nib.save(dat_proxy, join(sess_dir, tag + '_desc-resampled_dat.nii.gz'))

    dat_image = np.squeeze(np.array(dat_proxy.dataobj))
    dat_image -= min(0, np.min(dat_image))

    lab_image = np.array(labels_orig_proxy.dataobj)

    # Build region masks
    mask_cau = {h: _build_mask(lab_image, _CAU_LABELS[h]) for h in ('R', 'L')}
    mask_put = {h: _build_mask(lab_image, _PUT_LABELS[h]) for h in ('R', 'L')}
    mask_str = {h: _build_mask(lab_image, _STR_LABELS[h]) for h in ('R', 'L')}
    mask_occ = {h: _build_mask(lab_image, _OCC_LABELS[h]) for h in ('R', 'L')}
    for m_dict in (mask_cau, mask_put, mask_str, mask_occ):
        m_dict['Both'] = _bilateral(m_dict)

    for name, m in (('caudate', mask_cau), ('putamen', mask_put), ('occipital', mask_occ)):
        if any(np.sum(v) == 0 for v in m.values()):
            print(f'[Warning] Label *{name}* not found.')
            return

    # Foreground FOV via KMeans
    km = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(dat_image.reshape(-1, 1))
    seg = km.labels_.reshape(dat_image.shape)
    means_order = np.argsort([np.mean(dat_image[seg == u]) for u in np.unique(seg)])
    dat_fov = (seg == means_order[-2]) | (seg == means_order[-1])

    intensity_norm = np.mean(dat_image[mask_occ['Both']])

    _write_sbr_tsv(dat_image, dat_fov, mask_cau, mask_put, mask_str, mask_occ, loss, results_dir, tag)
    _write_symmetry_tsv(dat_image, dat_v2r, dat_rot, intensity_norm,  mask_cau, mask_put, dat_fov, results_dir, tag,
                        force_flag=force_flag)


def _write_sbr_tsv(dat_image: np.ndarray,
                    dat_fov: np.ndarray,
                    mask_cau: dict,
                    mask_put: dict,
                    mask_str: dict,
                    mask_occ: dict,
                    loss,
                    results_dir: str, tag: str,
                    force_flag: bool = False,) -> None:
    """Compute regional DaT uptake statistics and write them to a TSV file.

    :param dat_image: DaT image resampled to MRI/label space.
    :param dat_fov: Foreground field-of-view mask used to restrict valid voxels.
    :param mask_cau: Hemisphere masks for the caudate nucleus.
    :param mask_put: Hemisphere masks for the putamen.
    :param mask_str: Hemisphere masks for the whole striatum.
    :param mask_occ: Hemisphere masks for the occipital reference region.
    :param loss: Final registration loss stored with the metrics.
    :param results_dir: Directory where the TSV file is written.
    :param tag: Subject/session identifier.
    :param force_flag: If True, remove previous rows for the same tag before writing.

    :return: None. The SBR-like regional statistics table is written to disk.
    """

    out_file = join(results_dir, tag + '_sbr.tsv') 
    cols = ['id', 'processing', 'loss',  'hemi', 'aggregate', 'dat_str', 'dat_cau', 'dat_put', 'dat_occ']

    prev = _load_prev_tsv(out_file, cols, ['id', 'processing', 'hemi', 'aggregate'])
    if force_flag and tag in (prev.index.get_level_values('id') if len(prev) else []):
        prev = prev.drop(tag, level='id')

    rows = []
    for hemi in ('R', 'L', 'Both'):
        for agg, fn in (('sum', np.sum), ('mean', np.mean), ('median', np.median), ('std', np.std),
                        ('p10', partial(np.percentile, q=10)), ('p30', partial(np.percentile, q=30)),
                        ('p70', partial(np.percentile, q=70))):
            fov = dat_fov
            rows.append({
                'processing': 'raw', 'id': tag,
                'loss': loss, 'hemi': hemi, 'aggregate': agg,
                'dat_cau': fn(dat_image[mask_cau[hemi] & fov]),
                'dat_put': fn(dat_image[mask_put[hemi] & fov]),
                'dat_str': fn(dat_image[mask_str[hemi] & fov]),
                'dat_occ': fn(dat_image[mask_occ[hemi] & fov]),
            })

    df = pd.concat([prev.reset_index(), pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(out_file, sep='\t', index=False)


def _write_symmetry_tsv(dat_image_raw: np.ndarray,
                        dat_v2r: np.ndarray,
                        dat_rot: np.ndarray,
                        intensity_norm: float,
                        mask_cau: dict,
                        mask_put: dict,
                        dat_fov: np.ndarray,
                        results_dir: str,
                        tag: str,
                        force_flag: bool = False) -> None:
    """Compute DaT symmetry metrics and write them to a TSV file.

    :param dat_image_raw: DaT image in MRI/label space before symmetry normalization.
    :param dat_v2r: Original DaT voxel-to-RAS affine matrix.
    :param dat_rot: Rotation/alignment matrix estimated for the DaT image.
    :param intensity_norm: Reference intensity used to normalize DaT values before symmetry
        computation.
    :param mask_cau: Hemisphere masks for the caudate nucleus.
    :param mask_put: Hemisphere masks for the putamen.
    :param dat_fov: Foreground field-of-view mask used to restrict valid voxels.
    :param results_dir: Directory where the TSV file is written.
    :param tag: Subject/session identifier.
    :param force_flag: If True, remove previous rows for the same tag before writing.

    :return: None. The symmetry metrics table is written to disk.
    """

    out_file = join(results_dir, tag + '_symm.tsv')
    cols = ['id', 'hemi', 'metric', 'aggregate', 'dat_cau', 'dat_put']

    prev = _load_prev_tsv(out_file, cols, ['id', 'metric', 'hemi', 'aggregate'])
    if force_flag and tag in (prev.index.get_level_values('id') if len(prev) else []):
        prev = prev.drop(tag, level='id')

    v2r_symm  = dat_rot @ dat_v2r
    symm_map, l2_map = _compute_symmetry_maps(dat_image_raw / intensity_norm, v2r_symm)

    rows = []
    for metric, arr in (('lncc', symm_map), ('l2', l2_map)):
        for hemi in ('R', 'L', 'Both'):
            for agg, fn in (('sum', np.sum), ('std', np.std),
                             ('mean', np.mean), ('min', np.min), ('max', np.max)):
                fov = dat_fov
                rows.append({
                    'id': tag, 'hemi': hemi, 'metric': metric, 'aggregate': agg,
                    'dat_cau': fn(arr[mask_cau[hemi] & fov]),
                    'dat_put': fn(arr[mask_put[hemi] & fov]),
                })

    df = pd.concat([prev.reset_index(), pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(out_file, sep='\t', index=False)


def _compute_symmetry_maps(dat_norm: np.ndarray, v2r_symm: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    """Compute local symmetry maps between a DaT image and its left-right flipped version.

    :param dat_norm: Intensity-normalized DaT image.
    :param v2r_symm: Affine matrix defining the symmetry space used for the flip.

    :return: Tuple containing the local NCC symmetry map and the voxelwise L2
             difference map.
    """

    T_flip      = np.diag([-1., 1., 1., 1.])
    T           = torch.from_numpy(np.linalg.inv(v2r_symm) @ T_flip @ v2r_symm).float()
    shape       = dat_norm.shape

    grid = torch.meshgrid([torch.arange(s) for s in shape], indexing='ij')
    di = T[0, 0]*grid[0] + T[0, 1]*grid[1] + T[0, 2]*grid[2] + T[0, 3]
    dj = T[1, 0]*grid[0] + T[1, 1]*grid[1] + T[1, 2]*grid[2] + T[1, 3]
    dk = T[2, 0]*grid[0] + T[2, 1]*grid[1] + T[2, 2]*grid[2] + T[2, 3]

    dat_t    = torch.from_numpy(dat_norm)
    dat_flip = fast_3D_interp_torch(dat_t, di, dj, dk, mode='linear')

    dat_t4    = dat_t.unsqueeze(0).unsqueeze(0).to('cuda:0')
    dat_flip4 = dat_flip.unsqueeze(0).unsqueeze(0).to('cuda:0')

    lncc_fn  = NCCLoss(name='lncc-9', kernel_var=[9, 9, 9], device='cuda:0')
    symm_map = np.squeeze(lncc_fn.ncc(dat_t4, dat_flip4).detach().cpu().numpy())
    l2_map   = np.squeeze(((dat_t4 - dat_flip4)**2).sqrt().detach().cpu().numpy())

    return symm_map, l2_map


def _load_prev_tsv(path: str, cols: list, index_cols: list) -> pd.DataFrame:
    """Load an existing TSV table and set its index, or create an empty table.

    :param path: Path to the TSV file.
    :param cols: Columns to keep in the loaded table.
    :param index_cols: Columns used as dataframe index.

    :return: Indexed dataframe containing previous rows or an empty dataframe with the
    expected structure.
    """
    
    empty = pd.DataFrame(columns=cols).set_index(index_cols)
    if not exists(path):
        return empty
    df = pd.read_csv(path, sep='\t')
    df = df[[c for c in df.columns if c in cols]]
    return df.set_index(index_cols)
