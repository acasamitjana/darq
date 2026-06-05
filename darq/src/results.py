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
    return np.isin(lab_image, label_ids)


def _bilateral(d: dict) -> np.ndarray:
    return d['R'] | d['L']


def save_session_results(dat_orig_proxy, labels_orig_proxy, loss, tag: str, results_dir: str, force_flag: bool = False):
    """Resample DaT into MRI space and write SBR / symmetry TSV files."""
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


def _write_sbr_tsv(dat_image, dat_fov, mask_cau, mask_put, mask_str, mask_occ, loss, results_dir, tag, force_flag=False):
    out_file = join(results_dir, tag + '_sbr.tsv') #poner el mismo que en el otro 
    cols = ['id', 'processing', 'loss',  'hemi', 'aggregate', 'dat_str', 'dat_cau', 'dat_put', 'dat_occ']

    prev = _load_prev_tsv(out_file, cols, ['id', 'processing', 'hemi', 'aggregate'])
    if force_flag and tag in (prev.get_level_values('id') if len(prev) else []):
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


def _write_symmetry_tsv(dat_image_raw, dat_v2r, dat_rot, intensity_norm, mask_cau, mask_put, dat_fov, results_dir,
                        tag, force_flag=False):

    out_file = join(results_dir, tag + '_symm.tsv')
    cols = ['id', 'hemi', 'metric', 'aggregate', 'dat_cau', 'dat_put']

    prev = _load_prev_tsv(out_file, cols, ['id', 'metric', 'hemi', 'aggregate'])
    if force_flag and tag in (prev.get_level_values('id') if len(prev) else []):
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


def _compute_symmetry_maps(dat_norm, v2r_symm):
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
    empty = pd.DataFrame(columns=cols).set_index(index_cols)
    if not exists(path):
        return empty
    df = pd.read_csv(path, sep='\t')
    df = df[[c for c in df.columns if c in cols]]
    return df.set_index(index_cols)
