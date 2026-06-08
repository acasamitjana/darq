import re
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator as rgi


# ── Type- based functions ────────────────────────────────────────────────


def get_type_dict(data) -> dict:
    """Return basic type metadata needed to restore an object after conversion.

    :param data: Input object, typically a NumPy array or PyTorch tensor.

    :return: Dictionary containing the object type, dtype and optional tensor device.
    """

    t_dict = {}
    t_dict['type'] = type(data)
    t_dict['dtype'] = data.dtype
    if t_dict['type'] == torch.Tensor:
        t_dict['device'] = data.device

    return t_dict


def convert_to_type(data: Any, type: type, dtype: Any = None, device: Any = None) -> Any:
    """Convert data to a requested array or tensor type.

    :param data: Input data to convert.
    :param type: Target Python type, usually ``np.ndarray`` or ``torch.Tensor``.
    :param dtype: Optional target dtype.
    :param device: Optional target device when converting to a tensor.

    :return: Converted data with the requested type when supported.
    """

    if type == np.ndarray:
        return convert_to_numpy(data, dtype=dtype)
    elif type == torch.Tensor:
        return convert_to_tensor(data, dtype=dtype, device=device)

    return data


def convert_to_tensor(data: Any, dtype: Any = None, device: Any = None) -> torch.Tensor:
    """Recursively convert NumPy arrays and containers to PyTorch tensors.

    :param data: Input data, container or scalar to convert.
    :param dtype: Optional target tensor dtype.
    :param device: Optional target tensor device.

    :return: Converted tensor or container with tensor values where conversion is
        possible.
    """
    
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)

            return torch.as_tensor(data, dtype=dtype, device=device)

    elif isinstance(data, list):
        return [convert_to_tensor(i, dtype=dtype, device=device) for i in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_tensor(i, dtype=dtype, device=device) for i in data)
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device) for k, v in data.items()}

    return data


def convert_to_numpy(data: Any, dtype: Any = None) -> Any:
    """Recursively convert tensors and numeric values to NumPy arrays.

    :param data: Input data, container or scalar to convert.
    :param dtype: Optional target NumPy dtype.

    :return: Converted NumPy array or container with NumPy values where conversion is
        possible.
    """

    if isinstance(data, torch.Tensor):
        data = np.asarray(data.detach().to(device="cpu").numpy())

    elif isinstance(data, (np.ndarray, float, int, bool)):
        # Convert into a contiguous array first if the current dtype's size is smaller than the target dtype's size.
        # This help improve the performance because (convert to contiguous array) -> (convert dtype) is faster
        # than (convert dtype) -> (convert to contiguous array) when src dtype (e.g., uint8) is smaller than
        # target dtype(e.g., float32) and we are going to convert it to contiguous array anyway later in this
        # method.
        if isinstance(data, np.ndarray) and data.ndim > 0 and data.dtype.itemsize < np.dtype(dtype).itemsize:
            data = np.ascontiguousarray(data)
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        return [convert_to_numpy(i, dtype=dtype) for i in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_numpy(i, dtype=dtype) for i in data)
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data

# ── Deformation-based functions ──────────────────────────────────────────

def crop_label(mask: np.ndarray, margin: int = 10, threshold: float = 0) -> tuple[np.ndarray, list]:
    """Crop a mask around all voxels above a threshold.

    :param mask: Input mask or label image.
    :param margin: Number of voxels added around the detected bounding box, either scalar or
        per-axis list.
    :param threshold: Minimum value considered foreground.

    :return: Tuple containing the cropped mask and the crop coordinate ranges.
    """

    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin=[margin]*ndim

    crop_coord = []
    idx = np.where(mask>threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   crop_coord[2][0]: crop_coord[2][1]
                   ]

    return mask_cropped, crop_coord

def rescale_voxel_factor(volume: np.ndarray,
                        aff: np.ndarray,
                        factor,
                        not_aliasing: bool = False,
                        method: str = "linear",) -> tuple[np.ndarray, np.ndarray]:
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS

    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm

    :return: new volume and affine matrix
    """

    if isinstance(factor, (int, float)):
        factor = np.asarray([factor]*3)

    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    if len(volume.shape) > 3:
        sigmas = np.concatenate((sigmas, [0]*(len(volume.shape) - 3)))

    if all(sigmas == 0) or not_aliasing or method=='nearest':
        volume_filt = volume
    else:
        volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt, method)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape[:3] * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def fast_3D_interp_torch(X: torch.Tensor,
                        II: torch.Tensor,
                        JJ: torch.Tensor,
                        KK: torch.Tensor,
                        mode: str,) -> torch.Tensor:
    """Resize a volume by a voxel scaling factor and update its affine matrix.

    :param volume: Input image volume as a NumPy array.
    :param aff: 4x4 affine matrix of the input volume.
    :param factor: Scaling factor applied to the voxel grid. It can be a scalar or a
        3-element sequence.
    :param not_aliasing: If True, disables Gaussian anti-aliasing before resampling.
    :param method: Interpolation method used during resampling, such as ``'linear'`` or
        ``'nearest'``.

    :return: Tuple containing the resampled volume and the updated affine matrix.
    """

    if mode == 'nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        Y = X[IIr, JJr, KKr]

    elif mode == 'linear':
        ok = (II>=0) & (JJ>=0) & (KK>=0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]
        #
        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx
        #
        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy
        #
        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz
        #
        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]
        #
        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx
        #
        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy
        #
        c = c0 * wfz + c1 * wcz
        #
        Y = torch.zeros(II.shape, device=II.device)
        Y[ok] = c.float()

    else:
        raise Exception('mode must be linear or nearest')

    return Y


def fast_3D_interp_field_torch(X: torch.Tensor, II: torch.Tensor, JJ: torch.Tensor, KK: torch.Tensor, mode: str = 'linear', pad: float = 0.0) -> torch.Tensor:
    """Interpolate a 3D tensor at arbitrary voxel coordinates.

    :param X: Input 3D tensor.
    :param II: Sampling coordinates along the first image axis.
    :param JJ: Sampling coordinates along the second image axis.
    :param KK: Sampling coordinates along the third image axis.
    :param mode: Interpolation mode, either ``'linear'`` or ``'nearest'``.
    :param pad: Value used to initialize samples outside valid coordinates.

    :return: Interpolated tensor with the same spatial shape as the coordinate grids.
    """

    num_channels = X.shape[-1]
    if mode == 'nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if isinstance(pad, (int, float)):
            Y = pad*torch.ones([*II.shape, num_channels], device=X.device)
        else:
            Y = torch.zeros([*II.shape, num_channels], device=X.device)
        for channel in range(num_channels):
            #
            Xc = X[..., channel]
            Y[..., channel] = Xc[IIr, JJr, KKr]

    elif mode == 'linear':
        #
        ok = (II > 0) & (JJ > 0) & (KK > 0) & (II <= X.shape[0] - 1) & (JJ <= X.shape[1] - 1) & (KK <= X.shape[2] - 1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]
        #
        del JJ, KK
        #
        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx
        #
        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy
        #
        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz
        #
        Y = torch.zeros([*II.shape, num_channels], device=X.device)
        for channel in range(num_channels):
            #
            Xc = X[..., channel]
            #
            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]
            #
            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx
            #
            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy
            #
            c = c0 * wfz + c1 * wcz
            #
            if isinstance(pad, (int, float)):
                Yc = pad * torch.ones(II.shape, device=X.device)
            else:
                Yc = torch.zeros(II.shape, device=X.device)

            Yc[ok] = c.float()
            #
            Y[..., channel] = Yc
        #
    return Y


def vol_resample_fast(ref_proxy: nib.Nifti1Image,
                    flo_proxy: nib.Nifti1Image,
                    proxyflow=None,
                    mode: str = "linear",
                    return_np: bool = False,
                    **kwargs,) -> nib.Nifti1Image | np.ndarray:
    """Resample a floating NIfTI image into the voxel grid of a reference image.

    :param ref_proxy: Reference NIfTI proxy defining the output shape and affine.
    :param flo_proxy: Floating NIfTI proxy whose data will be resampled.
    :param proxyflow: Optional deformation field proxy applied before sampling the floating
        image.
    :param mode: Interpolation mode used for scalar images.
    :param return_np: If True, return only the NumPy array instead of a NIfTI image.
    :param kwargs: Reserved keyword arguments for future extensions.

    :return: Resampled NIfTI image, or NumPy array when ``return_np`` is True.
    """
    ref_v2r = (ref_proxy.affine).astype('float32')
    target_v2r = (flo_proxy.affine).astype('float32')

    ii = np.arange(0, ref_proxy.shape[0], dtype='int32')
    jj = np.arange(0, ref_proxy.shape[1], dtype='int32')
    kk = np.arange(0, ref_proxy.shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    del ii, jj, kk

    II = torch.tensor(II, device='cpu')
    JJ = torch.tensor(JJ, device='cpu')
    KK = torch.tensor(KK, device='cpu')

    if proxyflow is not None:

        flow_v2r = proxyflow.affine
        flow_v2r = flow_v2r.astype('float32')

        affine = torch.tensor(np.linalg.inv(flow_v2r) @ ref_v2r)
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        flow = np.array(proxyflow.dataobj)
        if flow.shape[-1] == 3: flow = np.transpose(flow, axes=(3, 0, 1, 2))
        flow = torch.tensor(flow)

        FIELD = fast_3D_interp_field_torch(flow, II2, JJ2, KK2)
        II3 = II2 + FIELD[:, :, :, 0]
        JJ3 = JJ2 + FIELD[:, :, :, 1]
        KK3 = KK2 + FIELD[:, :, :, 2]


        affine = torch.tensor(np.linalg.inv(target_v2r) @ flow_v2r)
        II4 = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
        JJ4 = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
        KK4 = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]


    else:
        affine = torch.tensor(np.linalg.inv(target_v2r) @ ref_v2r)
        II4 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ4 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK4 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]


    image = np.array(flo_proxy.dataobj)
    if len(flo_proxy.shape) == 3:
        reg_image = fast_3D_interp_torch(torch.tensor(image), II4, JJ4, KK4, mode)
    else:
        reg_image = fast_3D_interp_field_torch(torch.tensor(image), II4, JJ4, KK4)

    reg_image = reg_image.numpy()

    if return_np:
        return reg_image
    else:
        return nib.Nifti1Image(reg_image, ref_proxy.affine)


def create_template_space(proxy_list: list,
                        resolution=None,
                        mode: str | list = "linear",) -> tuple[np.ndarray, np.ndarray]:
    """Create a common NumPy template space covering multiple proxy images.

    :param proxy_list: List of proxy dictionaries containing data, shape and affine entries.
    :param resolution: Output voxel spacing. If None, the first proxy resolution is reused.
    :param mode: Interpolation mode or list of modes used for each proxy.

    :return: Tuple containing resampled images and the template-space affine matrix.
    """

    if resolution is None:
        ref_v2r = proxy_list[0]['affine'].astype('float32')
        resolution = np.sqrt(np.sum(ref_v2r * ref_v2r, axis=0))[:-1]

    elif isinstance(resolution, (int, float)):
        resolution = [resolution]*3

    if not isinstance(mode, list):
        mode = [mode]*len(proxy_list)

    boundaries_min = np.zeros((len(proxy_list), 3))
    boundaries_max = np.zeros((len(proxy_list), 3))
    for it_p, proxy in enumerate(proxy_list):
        if isinstance(proxy, str):
            proxy = nib.load(proxy)

        header = proxy['affine']
        vox_min = [0, 0, 0, 1]
        vox_max = list(proxy['shape'][:3]) + [1]

        minR, minA, minS = np.inf, np.inf, np.inf
        maxR, maxA, maxS = -np.inf, -np.inf, -np.inf

        for i in [vox_min[0], vox_max[0] + 1]:
            for j in [vox_min[1], vox_max[1] + 1]:
                for k in [vox_min[2], vox_max[2] + 1]:
                    aux = np.dot(header, np.asarray([i, j, k, 1]).T)

                    minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                    minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                    minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        boundaries_min[it_p] = [minR, minA, minS]
        boundaries_max[it_p] = [maxR, maxA, maxS]

    # Get the corners of cuboid in RAS space
    minR = np.min(boundaries_min[..., 0])
    minA = np.min(boundaries_min[..., 1])
    minS = np.min(boundaries_min[..., 2])
    maxR = np.max(boundaries_max[..., 0])
    maxA = np.max(boundaries_max[..., 1])
    maxS = np.max(boundaries_max[..., 2])

    # Define header and size
    temp_v2r = np.asarray([[resolution[0], 0, 0, minR],#/(scale[0]*ref_res[0])],#
                          [0, resolution[1], 0, minA],#/(scale[1]*ref_res[1])],#
                          [0, 0, resolution[2], minS],#/(scale[2]*ref_res[2])],#
                          [0, 0, 0, 1]]).astype('float32')

    template_size = np.asarray([int(np.ceil(maxR - minR) / (resolution[0])),
                                int(np.ceil(maxA - minA) / (resolution[1])),
                                int(np.ceil(maxS - minS) / (resolution[2]))])

    II, JJ, KK = np.meshgrid(np.arange(template_size[0]), np.arange(template_size[1]), np.arange(template_size[2]), indexing='ij')
    II, JJ, KK = torch.from_numpy(II), torch.from_numpy(JJ), torch.from_numpy(KK)

    images_out = []
    for proxy, m in zip(proxy_list, mode):
        if isinstance(proxy, str):
            proxy = nib.load(proxy)

        v2r = proxy['affine']
        im_shape = proxy['shape']
        data = np.array(proxy['data'])
        if len(im_shape) > 3:
            data = data.reshape(im_shape[:3] + (-1,))
        else:
            data = data[..., np.newaxis]

        data = torch.from_numpy(data)
        #
        affine = torch.tensor(np.linalg.inv(v2r) @ temp_v2r, device='cpu')
        di = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        dj = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        dk = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
        #
        im_out = fast_3D_interp_field_torch(data, di, dj, dk, mode=m, pad=float(torch.min(data))).numpy()
        images_out += [np.squeeze(im_out)]

    return images_out, temp_v2r

def create_template_space_tensor(proxy_list: list, resolution=None, mode: str | list = "linear") -> tuple[torch.Tensor, torch.Tensor]:
    """Create a common tensor template space covering multiple proxy images.

    :param proxy_list: List of proxy dictionaries containing tensor data, shape and affine
        entries.
    :param resolution: Output voxel spacing. If None, the first proxy resolution is reused.
    :param mode: Interpolation mode or list of modes used for each proxy.

    :return: Tuple containing resampled tensors and the template-space affine matrix.
    """

    device = proxy_list[0]['data'].device
    if resolution is None:
        ref_v2r = proxy_list[0]['affine']
        resolution = np.sqrt(np.sum(ref_v2r * ref_v2r, axis=0))[:-1]

    elif isinstance(resolution, (int, float)):
        resolution = [resolution]*3

    if not isinstance(mode, list):
        mode = [mode]*len(proxy_list)

    boundaries_min = torch.zeros((len(proxy_list), 3))
    boundaries_max = torch.zeros((len(proxy_list), 3))
    for it_p, proxy in enumerate(proxy_list):
        header = proxy['affine']
        vox_min = [0, 0, 0, 1]
        vox_max = list(proxy['shape']) + [1]

        minR, minA, minS = 1e7, 1e7, 1e7
        maxR, maxA, maxS = -1e7, -1e7, -1e7

        for i in [vox_min[0], vox_max[0] + 1]:
            for j in [vox_min[1], vox_max[1] + 1]:
                for k in [vox_min[2], vox_max[2] + 1]:
                    aux = header @ np.asarray([i, j, k, 1]).T

                    minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                    minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                    minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        boundaries_min[it_p] = torch.tensor([minR, minA, minS])
        boundaries_max[it_p] = torch.tensor([maxR, maxA, maxS])

    # Get the corners of cuboid in RAS space
    minR = torch.min(boundaries_min[..., 0])
    minA = torch.min(boundaries_min[..., 1])
    minS = torch.min(boundaries_min[..., 2])
    maxR = torch.max(boundaries_max[..., 0])
    maxA = torch.max(boundaries_max[..., 1])
    maxS = torch.max(boundaries_max[..., 2])

    # Define header and size
    temp_v2r = np.asarray([[resolution[0], 0, 0, minR],#/(scale[0]*ref_res[0])],#
                          [0, resolution[1], 0, minA],#/(scale[1]*ref_res[1])],#
                          [0, 0, resolution[2], minS],#/(scale[2]*ref_res[2])],#
                          [0, 0, 0, 1]]).astype('float32')

    template_size = tuple([int(np.ceil(maxR - minR) / (resolution[0])),
                                int(np.ceil(maxA - minA) / (resolution[1])),
                                int(np.ceil(maxS - minS) / (resolution[2]))])

    II, JJ, KK = torch.meshgrid(torch.arange(template_size[0], device=device),
                                torch.arange(template_size[1], device=device),
                                torch.arange(template_size[2], device=device), indexing='ij')

    images_out = []
    for proxy, m in zip(proxy_list, mode):
        v2r = proxy['affine']
        im_shape = proxy['shape']
        data = proxy['data']

        affine = torch.tensor(np.linalg.inv(v2r) @ temp_v2r, device=data.device)
        di = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        dj = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        dk = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        im_out = fast_3D_interp_torch(data, di, dj, dk, mode=m)
        # if len(data.shape) == 5:
        #     data_shape = data.shape
        #     # data = torch.permute(data, (2, 3, 4, 0, 1)).reshape(tuple(im_shape) + (-1,))
        #     im_out = fast_3D_interp_field_torch(data, di, dj, dk, mode=m)
        #     # im_out = torch.permute(im_out, (3, 0, 1, 2)).reshape(data_shape[:2] + template_size)
        #
        # elif len(data.shape) == 4:
        #     # data = torch.permute(data, (3, 0, 1, 2))
        #     im_out = fast_3D_interp_field_torch(data, di, dj, dk, mode=m)
        #     # im_out = torch.permute(im_out, (3, 0, 1, 2))
        #
        # else:
        #     im_out = fast_3D_interp_torch(data, di, dj, dk, mode=m)

        images_out += [im_out]

    return images_out, temp_v2r


# ── Loss functions ───────────────────────────────────────────────────────

class _Loss(nn.Module):
    """Base class for named PyTorch loss functions.
    The name is used when logging individual and weighted loss values during
    registration.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the loss name used for logging.
        :param name: Optional human-readable loss name.
        """

        super().__init__()
        self.name = name

class SSIM(_Loss):
    """Base wrapper for voxelwise similarity losses with optional masking and weighting.
    Despite the class name, subclasses can implement simple L1 or L2 losses by
    overriding ``_ssim_loss``.
    """

    def __init__(self, name: str | None = None, reduction: str = "mean", *args, **kwargs) -> None:
        """Initialize the similarity loss wrapper.
        
        :param name: Optional loss name used for logging.
        :param reduction: Reduction mode used after computing the voxelwise loss.
        :param args: Additional positional arguments reserved for subclasses.
        :param kwargs: Additional keyword arguments reserved for subclasses.
        """

        if name is None:
            name='SSIM'
        super().__init__(name=name)
        self.reduction = reduction

    @NotImplementedError
    def _ssim_loss(self, prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        pass 

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, weight: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """Compute the similarity loss with optional mask and weight maps.

        :param prediction: Predicted image tensor.
        :param target: Target image tensor.
        :param mask: Optional binary mask restricting the loss computation.
        :param weight: Optional voxelwise weights applied to the loss.
        :param args: Additional positional arguments reserved for subclasses.
        :param kwargs: Additional keyword arguments reserved for subclasses.

        :return: Reduced loss tensor according to the configured reduction mode.
        """

        ndims = len(prediction.shape)
        if mask is None and weight is None:
            return self._ssim_loss(prediction, target, reduction=self.reduction)
        else:
            res = self._ssim_loss(prediction, target, reduction='none')
            if mask is not None:
                res = res * mask
            if weight is not None:
                res = res * weight

        if self.reduction == 'mean':
            return 1/torch.sum(mask)*torch.sum(res)
            # norm_factor = torch.sum(mask, dim=[it for it in range(1,ndims)], keepdim=True)
            # wk = 1 / norm_factor * torch.sum(mask, dim=[it for it in range(2,ndims)])
            # res = 1 / norm_factor * torch.sum(res, dim=[it for it in range(2,ndims)])
            # res = torch.sum(wk*res, dim=1)
            # res = torch.mean(res)

        elif self.reduction == 'sum':
            return torch.sum(res)
        else:
            return res

        return res

class L1Loss(SSIM):
    """L1 similarity loss implemented through the generic SSIM wrapper."""
    def _ssim_loss(self, prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor   :
        """Compute voxelwise or reduced L1 loss.

        :param prediction: Predicted image tensor.
        :param target: Target image tensor.
        :param reduction: Reduction mode passed to PyTorch L1 loss.

        :return: L1 loss tensor.
        """

        return F.l1_loss(prediction, target, reduction=reduction)


class L2Loss(SSIM):
    """Mean-squared-error similarity loss implemented through the generic SSIM wrapper."""

    def _ssim_loss(self, prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Compute voxelwise or reduced L2 loss.

        :param prediction: Predicted image tensor.
        :param target: Target image tensor.
        :param reduction: Reduction mode passed to PyTorch MSE loss.

        :return: L2 loss tensor.
        """

        return F.mse_loss(prediction, target, reduction=reduction)

class NCCLoss(_Loss):
    """Local normalized cross-correlation loss for image registration.

    The loss computes local statistics with a configurable kernel and returns a
    negative similarity value so that optimization can minimize it.
    """

    def __init__(self,
                device: str,
                kernel_var=None,
                name: str | None = None,
                kernel_type: str = "mean",
                eps: float = 1e-5,
                *args,
                **kwargs,) -> None:
        """Initialize the NCC loss.

        :param device: PyTorch device used to store convolution kernels.
        :param kernel_var: Kernel size or variance configuration. If None, defaults are selected
            from the kernel type.
        :param name: Optional loss name used for logging.
        :param kernel_type: Kernel family used to compute local sums: ``'mean'``, ``'gaussian'``
            or ``'linear'``.
        :param eps: Small constant used for numerical stability.
        :param args: Additional positional arguments reserved for compatibility.
        :param kwargs: Additional keyword arguments reserved for compatibility.
        """

        if name is None:
            name = 'ncc'
        super().__init__(name=name)
        self.device = device
        self.kernel_var = kernel_var
        self.kernel_type = kernel_type
        self.eps = eps

        assert kernel_type in ['mean', 'gaussian', 'linear']

    def _get_kernel(self, kernel_type: str, kernel_sigma: list) -> torch.Tensor:
        """Build the convolution kernel used to compute local NCC statistics.

        :param kernel_type: Kernel family used for local averaging.
        :param kernel_sigma: Kernel size or sigma configuration.

        :return: PyTorch tensor containing the convolution kernel.
        """

        if kernel_type == 'mean':
            kernel = torch.ones([1, 1, *kernel_sigma]).to(self.device)

        elif kernel_type == 'linear':
            raise NotImplementedError("Linear kernel for NCC still not implemented")

        elif kernel_type == 'gaussian':
            kernel_size = kernel_sigma[0] * 3
            kernel_size += np.mod(kernel_size + 1, 2)

            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            x_cord = torch.arange(kernel_size)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            mean = (kernel_size - 1) / 2.
            variance = kernel_sigma[0] ** 2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            # 2.506628274631 = sqrt(2 * pi)

            kernel = (1. / (2.506628274631 * kernel_sigma[0])) * \
                     torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

            # Make sure sum of values in gaussian kernel equals 1.
            # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            kernel = kernel.to(self.device)

        return kernel

    def _compute_local_sums(self,
                            I: torch.Tensor,
                            J: torch.Tensor,
                            filt: torch.Tensor,
                            stride,
                            padding,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute local sums and second-order terms required by NCC.

        :param I: First image tensor.
        :param J: Second image tensor.
        :param filt: Convolution kernel used for local sums.
        :param stride: Convolution stride.
        :param padding: Convolution padding.
        
        :return: Tuple containing local variance of both images and the local cross term.
        """

        ndims = len(list(I.size())) - 2

        I2 = I * I
        J2 = J * J
        IJ = I * J

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I_sum = conv_fn(I, filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

        win_size = torch.sum(filt)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def ncc(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the local normalized cross-correlation map between two tensors.

        :param prediction: First image tensor with shape ``[batch, channels, *spatial]``.
        :param target: Second image tensor with shape ``[batch, channels, *spatial]``.

        :return: Tensor containing local squared correlation coefficients.
        """

        ndims = len(list(prediction.size())) - 2

        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if self.kernel_var is None:
            if self.kernel_type == 'gaussian':
                kernel_var = [3] * ndims  # sigma=3, radius = 9
            else:
                kernel_var = [9] * ndims  # sigma=radius=9 for mean and linear filter

        else:
            kernel_var = self.kernel_var

        sum_filt = self._get_kernel(self.kernel_type, kernel_var)
        radius = sum_filt.shape[-1]
        pad_no = int(np.floor(radius / 2))

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Eugenio: bug fixed where cross was not squared when computing cc
        I_var, J_var, cross = self._compute_local_sums(prediction, target, sum_filt, stride, padding)
        I_var = torch.clamp(I_var, 1e-5, 1e12)
        J_var = torch.clamp(J_var, 1e-5, 1e12)
        cc = cross * cross / (I_var * J_var + self.eps)
        return cc

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor | None = None,
                *args,
                **kwargs,) -> torch.Tensor:
        """Compute the negative NCC loss, optionally inside a mask.

        :param prediction: Predicted or floating image tensor.
        :param target: Target or reference image tensor.
        :param mask: Optional mask used to restrict the NCC average.
        :param args: Additional positional arguments ignored for compatibility.
        :param kwargs: Additional keyword arguments ignored for compatibility.

        :return: Scalar negative NCC loss tensor.
        """

        # if mask is not None:
        #     prediction = prediction * mask
        #     target = target * mask

        cc = self.ncc(prediction, target)
        if mask is None:
            return -1.0 * torch.sqrt(torch.mean(cc))
        elif torch.sum(mask) == 0:
            return torch.tensor(0)
        else:
            norm_factor = 1 / (torch.sum(mask))
            return -1.0 * torch.sqrt(norm_factor * torch.sum(cc * mask))

class DiceLoss(_Loss):
    """Soft Dice loss for comparing probabilistic or binary masks."""

    def __init__(self, name: str | None = None, *args, **kwargs) -> None:
        """Initialize the Dice loss.

        :param name: Optional loss name used for logging.
        :param args: Additional positional arguments reserved for compatibility.
        :param kwargs: Additional keyword arguments reserved for compatibility.
        """

        if name is None:
            name='dice'
        super().__init__(name=name)

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                classes_compute=None,
                eps: float = 0.0000001,
                *args,
                **kwargs,) -> torch.Tensor:
        """Compute the soft Dice loss between prediction and target masks.

        :param prediction: Predicted mask tensor with batch and optional class dimensions.
        :param target: Target mask tensor with the same shape as prediction.
        :param classes_compute: Optional class indices used to restrict the computation.
        :param eps: Small constant used for numerical stability.
        :param args: Additional positional arguments ignored for compatibility.
        :param kwargs: Additional keyword arguments ignored for compatibility.
        
        :return: Scalar Dice loss tensor.
        """

        # smooth = eps #1.
        # pflat = prediction.view(-1)
        # tflat = target.view(-1)
        # intersection = (pflat * tflat).sum()
        #
        # return 1 - ((2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth))

        # prediction = torch.clip(prediction / torch.sum(prediction, dim=1, keepdims=True), 0, 1)
        # target = torch.clip(target / torch.sum(target, dim=1, keepdims=True), 0, 1)
        if classes_compute is not None:
            prediction = prediction[:, classes_compute]
            target = target[:, classes_compute]

        top = torch.sum(2 * prediction * target, dim=list(range(1, len(prediction.shape))))
        bottom = prediction**2 + target**2 + eps
        bottom = torch.sum(bottom, dim=list(range(1, len(prediction.shape))))

        last_tensor = top / bottom
        return torch.mean(1 - last_tensor)

class DiceOverTrueLoss(_Loss):
    """Coverage-style Dice loss normalized by the true mask only."""
    def __init__(self, name: str | None = None, *args, **kwargs) -> None:
        """Initialize the Dice-over-true loss.

        :param name: Optional loss name used for logging.
        :param args: Additional positional arguments reserved for compatibility.
        :param kwargs: Additional keyword arguments reserved for compatibility.
        """

        if name is None:
            name='dice'
        super().__init__(name=name)

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                classes_compute=None,
                eps: float = 0.0000001,
                *args,
                **kwargs,) -> torch.Tensor:
        """Compute a Dice-like loss normalized by target-mask volume.

        :param prediction: Predicted mask tensor.
        :param target: Target mask tensor.
        :param classes_compute: Optional class indices used to restrict the computation.
        :param eps: Small constant used for numerical stability.
        :param args: Additional positional arguments ignored for compatibility.
        :param kwargs: Additional keyword arguments ignored for compatibility.

        :return: Scalar Dice-over-true loss tensor.
        """

        # smooth = eps #1.
        # pflat = prediction.view(-1)
        # tflat = target.view(-1)
        # intersection = (pflat * tflat).sum()
        #
        # return 1 - ((2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth))

        # prediction = torch.clip(prediction / torch.sum(prediction, dim=1, keepdims=True), 0, 1)
        # target = torch.clip(target / torch.sum(target, dim=1, keepdims=True), 0, 1)
        if classes_compute is not None:
            prediction = prediction[:, classes_compute]
            target = target[:, classes_compute]

        top = torch.sum(prediction * target, dim=list(range(1, len(prediction.shape))))
        bottom = target**2 + eps
        bottom = torch.sum(bottom, dim=list(range(1, len(prediction.shape))))

        last_tensor = top / bottom
        return torch.mean(1 - last_tensor)

class Symmetry(_Loss):
    """Loss that penalizes differences between an image and its flipped version.
    The flip is defined in RAS space and converted into voxel coordinates before
    interpolation.
    """

    def __init__(self,
                v2r: np.ndarray,
                name: str | None = None,
                axis: str = "r",
                loss: str = "l2",
                device: str = "cpu",
                *args,
                **kwargs,) -> None:
        """Initialize the symmetry loss.

        :param v2r: Voxel-to-RAS affine matrix defining the image geometry.
        :param name: Optional loss name used for logging.
        :param axis: RAS axis used for flipping: ``'r'``, ``'a'`` or ``'s'``.
        :param loss: Base loss name used to compare the image and its flipped version.
        :param device: PyTorch device used for tensors.
        :param args: Additional positional arguments reserved for compatibility.
        :param kwargs: Additional keyword arguments reserved for compatibility.

        """
        if name is None:
            name = 'symmetry'
        super().__init__(name=name)
        self.v2r = convert_to_tensor(v2r, dtype=torch.float, device=device)
        self.axis = axis
        assert loss in DICT_LOSSES.keys()
        self.loss = DICT_LOSSES[loss](device=device)
        self.device = device

    def _get_flip(self) -> torch.Tensor:
        """Build the homogeneous flip matrix for the configured anatomical axis.

        :return: 4x4 flip matrix in RAS coordinates.
        """

        T_flip = torch.zeros((4, 4), dtype=torch.float).to(self.device)
        if self.axis == 'r':
            T_flip[0, 0] = -1
            T_flip[1, 1] = 1
            T_flip[2, 2] = 1
            T_flip[3, 3] = 1

        elif self.axis == 'a':
            T_flip[0, 0] = 1
            T_flip[1, 1] = -1
            T_flip[2, 2] = 1
            T_flip[3, 3] = 1

        elif self.axis == 's':
            T_flip[0, 0] = 1
            T_flip[1, 1] = 1
            T_flip[2, 2] = -1
            T_flip[3, 3] = 1

        else:
            T_flip[0, 0] = 1
            T_flip[1, 1] = 1
            T_flip[2, 2] = 1
            T_flip[3, 3] = 1

        return T_flip

    def _get_grid(self, image_shape: tuple) -> torch.Tensor:
        """Create a voxel coordinate grid for an image shape.

        :param image_shape: Spatial image shape used to create the grid.

        :return: Tensor grid with one coordinate channel per spatial axis.
        """

        vectors = [torch.arange(0, s) for s in image_shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids) # y, x, z
        return grid.to(self.device)

    def forward(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compute the symmetry loss between an image and its flipped version.

        :param image: Image tensor with batch and channel dimensions.
        :param args: Additional positional arguments ignored for compatibility.
        :param kwargs: Additional keyword arguments ignored for compatibility.

        :return: Scalar loss tensor comparing the original image with its flipped version.
        """

        T_flip = self._get_flip()
        T_flip = torch.linalg.inv(self.v2r) @  T_flip  @ self.v2r
        im = torch.permute(image[0], (1, 2, 3, 0))

        grid = self._get_grid(im.shape[:3]).to(im.device)
        T = T_flip
        di = T[0, 0] * grid[0] + T[0, 1] * grid[1] + T[0, 2] * grid[2] + T[0, 3]
        dj = T[1, 0] * grid[0] + T[1, 1] * grid[1] + T[1, 2] * grid[2] + T[1, 3]
        dk = T[2, 0] * grid[0] + T[2, 1] * grid[1] + T[2, 2] * grid[2] + T[2, 3]

        image_flip = fast_3D_interp_field_torch(im, di, dj, dk, mode='linear')
        image_flip = torch.unsqueeze(torch.permute(image_flip, (3, 0, 1, 2)), 0)

        # pdb.set_trace()
        # import nibabel as nib
        # img = nib.Nifti1Image(fn_utils.convert_to_numpy(image)[0, 0], self.v2r)
        # nib.save(img, 'prova.nii.gz')
        # img = nib.Nifti1Image(fn_utils.convert_to_numpy(image_flip)[0, 0], self.v2r)
        # nib.save(img, 'prova_flip.nii.gz')

        loss = self.loss(image, image_flip)
        return loss

class MaxUptake(_Loss):
    """Loss encouraging concentrated uptake inside a target region."""

    def __init__(self, name: str | None = None, *args, **kwargs) -> None:
        """Initialize the maximum-uptake loss.
        
        :param name: Optional loss name used for logging.
        :param args: Additional positional arguments reserved for compatibility.
        :param kwargs: Additional keyword arguments reserved for compatibility.
        """

        if name is None:
            name = 'max_uptake'
        super().__init__(name=name)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Measure the difference between near-maximum and mean predicted uptake inside a target
            mask.

        :param prediction: Predicted uptake image tensor.
        :param target: Binary mask defining the region where uptake is evaluated.
        :param args: Additional positional arguments ignored for compatibility.
        :param kwargs: Additional keyword arguments ignored for compatibility.

        :return: Scalar uptake concentration loss tensor.
        """
        
        return torch.quantile(prediction[target > 0], 0.999)-torch.mean(prediction[target > 0])



DICT_LOSSES = {
    'l1': L1Loss,
    'l2': L2Loss,
    'ncc': NCCLoss,
    'dice': DiceLoss,
    'dice_over_true': DiceOverTrueLoss,
    'symmetry': Symmetry,
    'max_uptake': MaxUptake,
}

