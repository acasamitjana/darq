"""Per-subject registration pipeline: preprocessing → simulation → alignment → save."""
import copy
import time
from os import makedirs
from os.path import join, exists
import shutil
import numpy as np
import torch
from skimage.morphology import binary_dilation, binary_opening, ball
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from skimage import measure

from darq.src import models     # shared library
from darq.utils import fn_utils      # shared library
from darq.src.preprocessing import MorphologicalOperator, GaussianBlur, ToNumpy   # shared library
from darq.config import BLUR_MIN, BLUR_MAX

from darq.src.results import save_session_results


# ── DaT mask utilities ───────────────────────────────────────────────────────

def get_dat_transforms() -> list:
    """Build the transforms used to simulate DaT uptake from the MRI striatal mask.

    :return: List of transforms applied to the ``simulated_dat`` entry.
    """

    return [
        MorphologicalOperator(
            keys=['simulated_dat'],
            operator='opening',
            struct_fn=lambda x: ball(3),
        ),
        GaussianBlur(
            keys=['simulated_dat'],
            sigma=lambda: BLUR_MIN + (BLUR_MAX - BLUR_MIN) * np.random.rand(3),
            normalize_area=True,
        ),
    ]


def _get_dat_mask(dat_image: np.ndarray, v2r: np.ndarray) -> np.ndarray:
    """Estimate a striatal DaT uptake mask using KMeans and anatomical filtering.

    :param dat_image: DaT image in template space.
    :param v2r: Voxel-to-RAS affine matrix used to reject anatomically implausible blobs.

    :return: Binary mask containing the retained high-uptake DaT regions.
    """

    img = copy.deepcopy(dat_image)
    n_clusters = 2
    while True:
        km  = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(100 * img.reshape(-1, 1))
        seg = km.labels_.reshape(img.shape)
        means   = [np.mean(img[seg == u]) for u in np.unique(seg)]
        ordered = np.argsort(means)

        pre_mask = (seg == np.unique(seg)[ordered[-1]]) & (img > 0)
        blobs, n = _label_blobs(pre_mask)
        for nb in range(1, n + 1):
            coord = _blob_centre(blobs, nb, v2r)
            if np.abs(coord[0]) > 35 or coord[2] < -40:
                img[blobs == nb] = np.min(dat_image)
                pre_mask[blobs == nb] = 0

        roi_pct = np.sum(pre_mask) / np.prod(seg.shape) * 100
        if roi_pct > 1 or roi_pct == 0:
            n_clusters += 1
        else:
            break

    mask = (seg == np.unique(seg)[ordered[-1]]) & (img > 0)
    mask = binary_opening(mask, ball(3))
    blobs, n = _label_blobs(mask)
    counts = np.bincount(blobs.reshape(-1))
    for nb in range(1, n + 1):
        coord = _blob_centre(blobs, nb, v2r)
        if counts[nb] < 500 or np.abs(coord[0]) > 35 or coord[2] < -40:
            mask[blobs == nb] = 0
    return mask


def _get_dat_mask_prior(dat_image: np.ndarray, prior: np.ndarray,
                         percentage: float = 1.0) -> np.ndarray:
    """Estimate a high-uptake DaT mask inside a prior region.

    :param dat_image: DaT image in template space.
    :param prior: Binary prior mask restricting where the high-uptake region can be found.
    :param percentage: Maximum target percentage of image voxels assigned to the mask before
        increasing the number of clusters.

    :return: Binary high-uptake mask restricted to the prior region.
    """

    img = copy.deepcopy(dat_image) * prior
    n_clusters = 2
    while True:
        km  = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(100 * img.reshape(-1, 1))
        seg = km.labels_.reshape(img.shape)
        means   = [np.mean(img[seg == u]) for u in np.unique(seg)]
        ordered = np.argsort(means)
        pre_mask = (seg == np.unique(seg)[ordered[-1]]) & (img > 0)
        if np.sum(pre_mask) / np.prod(seg.shape) * 100 > percentage or np.sum(pre_mask) == 0:
            n_clusters += 1
        else:
            break

    mask = (seg == np.unique(seg)[ordered[-1]]) & (img > 0)
    blobs, n = _label_blobs(mask)
    counts = np.bincount(blobs.reshape(-1))
    for nb in range(1, n + 1):
        if counts[nb] < 500:
            mask[blobs == nb] = 0
    return mask


def _flip_dat(dat_image: np.ndarray, v2r_symm: np.ndarray,
               agg: str = 'mean') -> np.ndarray:
    """Flip or symmetrize a DaT image through the left-right symmetry plane.

    :param dat_image: Input DaT image.
    :param v2r_symm: Affine matrix defining the symmetry space used for the flip.
    :param agg: Aggregation mode: ``'mean'``, ``'max'``, ``'sum'`` or ``'flip'``.

    :return: Flipped or aggregated DaT image.
    """

    T_flip = np.diag([-1., 1., 1., 1.])
    T = torch.from_numpy(np.linalg.inv(v2r_symm) @ T_flip @ v2r_symm).float()
    grid = torch.meshgrid([torch.arange(s) for s in dat_image.shape], indexing='ij')
    di = T[0, 0]*grid[0] + T[0, 1]*grid[1] + T[0, 2]*grid[2] + T[0, 3]
    dj = T[1, 0]*grid[0] + T[1, 1]*grid[1] + T[1, 2]*grid[2] + T[1, 3]
    dk = T[2, 0]*grid[0] + T[2, 1]*grid[1] + T[2, 2]*grid[2] + T[2, 3]
    t  = torch.from_numpy(dat_image)
    tf = fn_utils.fast_3D_interp_torch(t, di, dj, dk, mode='linear')
    if agg == 'mean':
        return 0.5 * (t + tf).numpy()
    if agg == 'max':
        return torch.max(torch.stack([t, tf]), 0).values.numpy()
    if agg == 'sum':
        return (t + tf).numpy()
    return tf.numpy()  # 'flip'


def _label_blobs(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components in a binary mask.

    :param mask: Binary mask whose connected components will be labeled.

    :return: Tuple containing the labeled image and the number of detected components.
    """

    return measure.label(mask, connectivity=2, return_num=True)


def _blob_centre(blobs: np.ndarray, nb: int, v2r: np.ndarray) -> np.ndarray:
    """Compute the RAS-space center of one connected component.

    :param blobs: Connected-component label image.
    :param nb: Component label whose center will be computed.
    :param v2r: Voxel-to-RAS affine matrix used to convert voxel coordinates to RAS
        coordinates.

    :return: Four-element homogeneous RAS coordinate of the component center.
    """

    x, y, z = np.where(blobs == nb)
    return v2r @ np.array([np.median(x), np.median(y), np.median(z), 1])


# ── Optimiser factory ─────────────────────────────────────────────────────────

def _build_optimizer(model, opt_str: str) -> torch.optim.Optimizer:
    """Create the optimizer used for rigid registration.

    :param model: Registration model whose trainable parameters will be optimized.
    :param opt_str: Optimizer name selected by the user: ``'lbfgs'``, ``'adam'`` or
        ``'sgd'``.

    :return: Configured PyTorch optimizer.
    """

    if opt_str == 'adam':
        lr = 1e-3
        print(f'    Optimizer: ADAM  lr={lr}')
        return torch.optim.Adam(model.parameters(), lr=lr)
    if opt_str == 'lbfgs':
        lr, max_iter = 1e-1, 10
        print(f'    Optimizer: LBFGS lr={lr}  max_iter={max_iter}')
        return torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter,
                                  line_search_fn='strong_wolfe')
    lr = 1e-3
    print(f'    Optimizer: SGD   lr={lr}')
    return torch.optim.SGD(model.parameters(), lr=lr)


# ── Main subject pipeline ─────────────────────────────────────────────────────

def process_subject(data_dict: dict, preproc_tf: dict, dat_tf: list,
                    main_dict: dict, args) -> dict | None:
    """Run the complete registration pipeline for one subject/session."""
    if data_dict is None:
        return None

    run_info = _prepare_subject_run(data_dict, main_dict)

    if _should_skip_subject(run_info["output_dir"], run_info["tag"], args.force):
        return {"exit": 0}

    t0 = time.time()

    data_dict = _run_preprocessing_step(
        data_dict=data_dict,
        preproc_tf=preproc_tf,
        output_dir=run_info["output_dir"],
        tag=run_info["tag"],
    )

    dat_context = _build_dat_symmetry_and_masks(data_dict)

    data_dict, mri_context = _simulate_dat_from_mri(
        data_dict=data_dict,
        dat_tf=dat_tf,
        template_v2r=dat_context["template_v2r"],
        dat_mask=dat_context["dat_mask"],
    )

    tensor_dict = _build_registration_tensors(
        data_dict=data_dict,
        dat_context=dat_context,
        mri_context=mri_context,
        device=run_info["device"],
    )

    tensor_dict = _run_registration_step(
        tensor_dict=tensor_dict,
        data_dict=data_dict,
        dat_context=dat_context,
        mri_context=mri_context,
        main_dict=main_dict,
        args=args,
        device=run_info["device"],
    )

    _save_subject_outputs(
        data_dict=data_dict,
        tensor_dict=tensor_dict,
        output_dir=run_info["output_dir"],
        tag=run_info["tag"],
        loss=tensor_dict["loss"],
        force_flag=args.force,
        device=run_info["device"],
    )

    _clean_temp_dir(run_info["temp_dir"])

    print(f'Done in {round(time.time() - t0, 2)}s.\n')
    return {"exit": 0}

def _prepare_subject_run(data_dict: dict, main_dict: dict) -> dict:
    """Collect common paths and metadata required to process one session."""
    output_dir = main_dict["output_dir"]

    makedirs(output_dir, exist_ok=True)

    return {
        "device": main_dict["device"],
        "output_dir": output_dir,
        "tag": data_dict["id"],
        "temp_dir": join(output_dir, "tmp"),
    }


def _should_skip_subject(output_dir: str, tag: str, force_flag: bool) -> bool:
    """Return True if this session already has outputs and recomputation is disabled."""
    sbr_file = join(output_dir, tag + "_sbr.tsv")

    if exists(sbr_file) and not force_flag:
        print(f" * Skipping {tag}: results already exist. Use --force to recompute.")
        return True

    return False


def _run_preprocessing_step(data_dict: dict, preproc_tf: dict,
                            output_dir: str, tag: str) -> dict:
    """Apply preprocessing transforms and save rotation matrices."""
    print(" * Preprocessing.")

    for transform in preproc_tf.values():
        data_dict = transform(data_dict)

    np.save(join(output_dir, tag + "_space-T1wdseg_rot.npy"),
            data_dict["rot_label_v2r"])

    np.save(join(output_dir, tag + "_space-dat_rot.npy"),
            data_dict["rot_dat_v2r"])

    return data_dict

def _build_dat_symmetry_and_masks(data_dict: dict) -> dict:
    """Create symmetric DaT image, striatal DaT mask and DaT brain foreground."""
    print(" * DaT symmetry and mask.")

    template_v2r = data_dict["template_v2r"]
    dat_raw = data_dict["template_dat_image"]

    dat_symm = _compute_symmetric_dat(dat_raw, template_v2r)
    cuboid = _build_dat_prior_cuboid(dat_symm, template_v2r)

    dat_mask = _build_symmetric_dat_mask(dat_raw, template_v2r, cuboid)
    dat_image = _normalize_dat_inside_mask(dat_raw, dat_mask)

    brain_dat = _estimate_dat_brain_foreground(
        dat_symm=dat_symm,
        reference_brain_mask=data_dict["template_mask_brain"],
    )

    return {
        "template_v2r": template_v2r,
        "dat_raw": dat_raw,
        "dat_symm": dat_symm,
        "dat_mask": dat_mask,
        "dat_image": dat_image,
        "brain_dat": brain_dat,
    }


def _compute_symmetric_dat(dat_raw: np.ndarray, template_v2r: np.ndarray) -> np.ndarray:
    """Average the DaT image with its left-right flipped version."""
    dat_flip = _flip_dat(dat_raw, template_v2r, agg="flip")
    return 0.5 * (dat_raw + dat_flip)


def _build_dat_prior_cuboid(dat_symm: np.ndarray, template_v2r: np.ndarray) -> np.ndarray:
    """Build a cuboid prior around the high-uptake symmetric DaT region."""
    mask_symm = _get_dat_mask(dat_symm, template_v2r)
    mask_dilated = binary_dilation(mask_symm, np.ones((10, 10, 10)))

    _, crop = fn_utils.crop_label(mask_dilated, margin=5)

    cuboid = np.zeros_like(mask_dilated)
    cuboid[
        crop[0][0]:crop[0][1],
        crop[1][0]:crop[1][1],
        crop[2][0]:crop[2][1],
    ] = 1

    return cuboid


def _build_symmetric_dat_mask(dat_raw: np.ndarray,
                              template_v2r: np.ndarray,
                              cuboid: np.ndarray) -> np.ndarray:
    """Create a DaT mask using the raw image and its flipped counterpart."""
    mask_raw = _get_dat_mask_prior(dat_raw, cuboid, percentage=1)
    mask_flip = _flip_dat(mask_raw, template_v2r, agg="flip")

    return (mask_raw + mask_flip) > 0


def _normalize_dat_inside_mask(dat_raw: np.ndarray, dat_mask: np.ndarray) -> np.ndarray:
    """Robustly normalize DaT intensities inside the detected mask."""
    high = np.percentile(dat_raw[dat_mask], 99.5)
    low = np.percentile(dat_raw[dat_mask], 0.5)

    dat_image = np.clip((dat_raw - low) / (high - low), 0, None)
    dat_image[~dat_mask] = 0

    return dat_image


def _estimate_dat_brain_foreground(dat_symm: np.ndarray,
                                   reference_brain_mask: np.ndarray) -> np.ndarray:
    """Estimate DaT foreground using GMM clusters and the MRI brain-mask size."""
    n_clusters = 6

    dat_seg = GMM(n_components=n_clusters, random_state=0).fit_predict(
        dat_symm.reshape(-1, 1)
    )
    dat_seg = dat_seg.reshape(dat_symm.shape)

    labels = np.unique(dat_seg)
    means = [np.mean(dat_symm[dat_seg == label]) for label in labels]
    ordered_labels = labels[np.argsort(means)]

    brain_dat = np.zeros_like(dat_symm)

    for label in ordered_labels[1:][::-1]:  # skip background cluster
        brain_dat[dat_seg == label] = 1

        if np.sum(brain_dat) > 2 * np.sum(reference_brain_mask):
            break

    return brain_dat

def _simulate_dat_from_mri(data_dict: dict, dat_tf: list,
                           template_v2r: np.ndarray,
                           dat_mask: np.ndarray) -> tuple[dict, dict]:
    """Simulate a DaT-like MRI mask and estimate the initial translation."""
    print(" * MRI DaT simulation and mask.")

    data_dict = {
        **data_dict,
        "simulated_dat": data_dict["template_mask_str"],
        "v2r": template_v2r,
    }

    for transform in dat_tf:
        data_dict = transform(data_dict)

    mri_mask, sim_dat = _build_mri_dat_mask(data_dict["simulated_dat"])
    tx_init = _estimate_initial_translation(mri_mask, dat_mask)

    return data_dict, {
        "mri_mask": mri_mask,
        "sim_dat": sim_dat,
        "tx_init": tx_init,
    }


def _build_mri_dat_mask(simulated_dat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment the simulated DaT image and keep the high-intensity cluster."""
    km = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(
        simulated_dat.reshape(-1, 1)
    )

    mri_seg = km.labels_.reshape(simulated_dat.shape)
    hi_idx = np.argmax(km.cluster_centers_)

    mri_mask = mri_seg == hi_idx

    high = np.max(simulated_dat[mri_mask])
    low = np.min(simulated_dat[mri_mask])

    sim_dat = (simulated_dat - low) / (high - low)

    return mri_mask, sim_dat


def _estimate_initial_translation(mri_mask: np.ndarray,
                                  dat_mask: np.ndarray) -> np.ndarray:
    """Estimate initial translation from the centers of gravity of both masks."""
    mri_cog = np.asarray([np.mean(idx) for idx in np.where(mri_mask)])
    dat_cog = np.asarray([np.mean(idx) for idx in np.where(dat_mask)])

    return dat_cog - mri_cog

def _build_registration_tensors(data_dict: dict,
                                dat_context: dict,
                                mri_context: dict,
                                device: str) -> dict:
    """Build PyTorch tensors required by the registration model."""
    template_v2r = dat_context["template_v2r"]

    ref_mask = (
        np.stack(
            [
                data_dict["template_mask_brain"],
                data_dict["template_mask_occ"],
            ],
            axis=0,
        )[np.newaxis] > 0.5
    ).astype("float")

    flo_mask = (
        np.stack(
            [
                dat_context["brain_dat"],
                dat_context["brain_dat"],
            ],
            axis=0,
        )[np.newaxis] > 0.5
    ).astype("float")

    return {
        "ref_image": torch.as_tensor(
            mri_context["mri_mask"][np.newaxis, np.newaxis],
            dtype=torch.float,
        ).to(device),

        "ref_mask": torch.as_tensor(
            ref_mask,
            dtype=torch.float,
        ).to(device),

        "flo_image": torch.as_tensor(
            dat_context["dat_mask"][np.newaxis, np.newaxis],
            dtype=torch.float,
        ).to(device),

        "flo_mask": torch.as_tensor(
            flo_mask,
            dtype=torch.float,
        ).to(device),

        "template_v2r": template_v2r,
    }


def _run_registration_step(tensor_dict: dict,
                           data_dict: dict,
                           dat_context: dict,
                           mri_context: dict,
                           main_dict: dict,
                           args,
                           device: str) -> dict:
    """Run rigid MRI-DaT registration and return the registered tensors."""
    print(" * MRI-DaT registration.")

    template_v2r = dat_context["template_v2r"]

    reg_model = models.InstanceRigidModelClassic(
        data_dict["template_space"].shape,
        ref_v2r=template_v2r.astype("float32"),
        flo_v2r=template_v2r.astype("float32"),
        tx_factor=np.array([10, 1 / 1000, 1 / 1000]),
        angle_factor=np.array([1 / 100, 1, 1]),
        tx_init=mri_context["tx_init"],
        device=device,
    ).to(device)

    optimizer = _build_optimizer(reg_model, args.opt_str)
    loss_dict = _build_loss_dict(device, template_v2r)

    print(
        "    Losses: "
        + "; ".join(f'{key}(w={value["weight"]})'
                    for key, value in loss_dict.items())
    )

    session = models.JointInstanceReg(
        loss_dict,
        main_dict,
        da=[],
        trainable_keys={"reg": "reg"},
        verbose=True,
    )

    tensor_dict = session.register(
        tensor_dict,
        {"reg": reg_model},
        {"reg": optimizer},
    )

    print(f'    Final loss: {tensor_dict["loss"]}')

    return tensor_dict


def _save_subject_outputs(data_dict: dict,
                          tensor_dict: dict,
                          output_dir: str,
                          tag: str,
                          loss: float,
                          force_flag: bool = False,
                          device: str = "cpu") -> None:
    """Save affine matrix, registered outputs and quantitative results."""
    tensor_dict = ToNumpy(
        keys=["ref_image", "flo_image", "reg_image"],
        to_nibabel=True,
    )(tensor_dict)

    affine_ras = np.squeeze(
        tensor_dict["affine_ras"].cpu().detach().numpy()
    )

    np.save(
        join(output_dir, tag + "_space-symmetricT1w_aff.npy"),
        affine_ras,
    )

    save_session_results(
        data_dict["dat"],
        data_dict["label"],
        loss=loss,
        tag=data_dict["id"],
        results_dir=output_dir,
        force_flag=force_flag,
        device=device,
    )


def _clean_temp_dir(temp_dir: str) -> None:
    """Remove temporary files generated during the processing of one session."""
    if exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

def _build_loss_dict(device: str, v2r: np.ndarray) -> dict:
    """Build the loss configuration used during MRI-DaT registration.

    :param device: PyTorch device where tensor losses are computed.
    :param v2r: Template-space voxel-to-RAS affine used by the symmetry loss.

    :return: Dictionary mapping loss names to loss objects and scalar weights.
    """

    return {
        'reg':         {'loss': fn_utils.DiceLoss(name='reg'),             'weight': 1},
        'reg_label':   {'loss': fn_utils.DiceOverTrueLoss(name='reg_label'), 'weight': 2},
        'reg_uptake':  {'loss': fn_utils.MaxUptake(device=device, name='reg_uptake'), 'weight': 0.},
        'reg_lr':      {'loss': fn_utils.Symmetry(name='reg_lr', v2r=v2r, device=device, loss='l1'), 'weight': 0.5},
        'regularizer': {'loss': fn_utils.L2Loss(name='regularizer'),       'weight': 1},
    }


# ── Parallel wrapper ──────────────────────────────────────────────────────────

def process_fn_parallel(fn, *args, **kwargs) -> dict | None:
    """Safely execute a processing function inside a parallel worker.
    
    :param fn: Function to call in the worker process.
    :param args: Positional arguments forwarded to the processing function.
    :param kwargs: Keyword arguments forwarded to the processing function.

    :return: Function result, or None if the function raises an exception.
    """

    try:
        return fn(*args, **kwargs)
    except Exception:
        return None
