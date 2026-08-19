import nibabel as nib
import numpy as np
import torch

from skimage.morphology import opening

from darq.src import models
from darq.utils import fn_utils, io
from darq.config import REGISTRATION_DEFAULTS


class Transform(object):
    """Base class for dictionary-based preprocessing transforms.

    Derived classes implement ``__call__`` and modify one or more entries of the
    subject dictionary.
    """
    def __init__(self, keys: list | dict) -> None:
        """Store the dictionary keys that will be processed by the transform.

        :param keys: Key or list of keys identifying the entries modified by the transform.
        """

        self.keys = keys

    def __call__(self, data_dict: dict, *args, **kwargs) -> dict:
        raise NotImplementedError


class MorphologicalOperator(Transform):
    """Apply a morphological operation to selected dictionary entries.

    The current implementation supports binary opening and preserves the input data
    type whenever possible.
    """
    def __init__(self, keys: list, operator: str, struct_fn) -> None:
        """Initialize the morphological operator transform.

        :param keys: Key or list of keys identifying the entries modified by the transform.
        :param operator: Morphological operator to apply ('opening' or 'closing').
        :param struct_fn: Function to generate the structuring element.
        """

        self.keys = keys
        self.operator = operator
        self.struct_fn = struct_fn

    def __call__(self, data_dict: dict, *args, **kwargs) -> dict:
        """Apply the selected morphological operation to each configured key.

        :param data_dict: Subject dictionary containing the selected image or mask entries.
        :return: Updated subject dictionary with processed entries.
        """

        for k in self.keys:
            type_dict = fn_utils.get_type_dict(data_dict[k])
            image = fn_utils.convert_to_numpy(data_dict[k])
            image_shape = image.shape
            if len(image_shape) == 5:
                image = image[0, 0]

            if self.operator == 'opening':
                image = opening(image, self.struct_fn(data_dict[k]))

            if len(image_shape) == 5:
                data_dict[k] = data_dict[k][np.newaxis, np.newaxis]

            data_dict[k] = fn_utils.convert_to_type(image, **type_dict)

        return data_dict

class GaussianBlur(Transform):
    """Apply a 3D Gaussian blur to selected dictionary entries.

    The transform builds a depthwise 3D convolution kernel in PyTorch and can
    optionally update only voxels inside a mask.
    """

    def __init__(self, keys: list, sigma, mask_key: str | None = None, **kwargs) -> None:
        """Initialize the Gaussian smoothing transform.

        :param keys: Dictionary keys whose images will be blurred.
        :param sigma: Gaussian standard deviation or callable returning a 3-element sigma
            vector.
        :param mask_key: Optional dictionary key containing a mask that restricts where the
            blurred image is copied back.
        :param kwargs: Optional settings such as ``normalize_area``.
        """

        self.keys = keys
        self.mask_key = mask_key
        self.sigma = sigma
        self.normalize_area = kwargs['normalize_area'] if 'normalize_area' in kwargs.keys() else False

    def _gaussian_filter_3d(self,sigma: np.ndarray, channels: int = 1, truncate: int = 4,) -> torch.nn.Conv3d:
        """Create a fixed 3D Gaussian filter implemented as a PyTorch convolution.

        :param sigma: Gaussian standard deviation for each spatial axis.
        :param channels: Number of channels processed independently by the depthwise filter.
        :param truncate: Kernel radius expressed as a multiple of sigma.

        :return: Configured ``torch.nn.Conv3d`` module with frozen Gaussian weights.
        """

        # Set these to whatever you want for your gaussian filter
        kernel_size = tuple([2*round(s*truncate)+1 for s in sigma])

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        II, JJ, KK = torch.meshgrid([torch.arange(0, kernel_size[0]), torch.arange(0, kernel_size[1]), torch.arange(0, kernel_size[2])], indexing='ij')
        grid = torch.stack([II, JJ, KK], dim=-1)

        mean = torch.from_numpy(np.array([(k - 1) / 2. for k in kernel_size]).reshape((1, 1, 1, 3)))
        variance = torch.from_numpy(((sigma ** 2.).reshape((1, 1, 1, 3))))

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / ((2. * torch.pi * torch.prod(variance)))**(1/3)) * torch.exp(-torch.sum((grid - mean) ** 2. / (2 * variance), dim=-1))
        gaussian_kernel /= torch.max(gaussian_kernel)
        # Make sure sum of values in gaussian kernel equals 1.
        if self.normalize_area:
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 3d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, *kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        gaussian_filter = torch.nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                          padding='same', groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel.float()
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def _blur(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Blur an image tensor using the configured Gaussian kernel.

        :param image: Image tensor with shape compatible with 3D convolution.
        :param mask: Optional mask reserved for future masked blur logic.

        :return: Blurred image tensor.
        """

        if callable(self.sigma):
            sigma = self.sigma()
        else:
            sigma = self.sigma

        filt = self._gaussian_filter_3d(sigma)
        filt = filt.to(image.device)

        return filt(image.clone())

    def __call__(self, data_dict: dict, *args, **kwargs) -> dict:
        """Apply Gaussian smoothing to each configured dictionary entry.

        :param data_dict: Subject dictionary containing the selected image entries.

        :return: Updated subject dictionary with blurred images.
        """

        for k in self.keys:
            type_dict = fn_utils.get_type_dict(data_dict[k])
            image = fn_utils.convert_to_tensor(data_dict[k])
            image_shape = image.shape
            if len(image_shape) == 3:
                image = torch.unsqueeze(torch.unsqueeze(image, 0), 0)

            # pdb.set_trace()
            # filt = self._gaussian_filter_3d(np.asarray([3, 3, 3]))
            # filt = filt.to(image.device)
            # mri_proxy = nib.Nifti1Image(image.numpy()[0, 0], data_dict['v2r'])
            # nib.save(mri_proxy, 'orig.nii.gz')
            # dat_proxy = nib.Nifti1Image(filt(image.clone()).numpy()[0, 0], data_dict['v2r'])
            # nib.save(dat_proxy, 'gauss_blur.nii.gz')

            image = fn_utils.convert_to_type(self._blur(image), **type_dict)
            if len(image_shape) == 3:
                image = image[0, 0]

            if self.mask_key is not None:
                mask = data_dict[self.mask_key]
                data_dict[k][mask] = image[mask]
            else:
                data_dict[k] = image


        return data_dict

class AlignLR(Transform):
    """Estimate a left-right symmetry alignment for an image or mask.

    This transform optimizes a rigid transform that makes the selected reference mask
    more symmetric around the left-right axis. The resulting matrices are later used
    to create a common template space.
    """

    def __init__(self,
                keys: list,
                ref_im,
                ref_v2r: str,
                rescaling_factor: float = 1,
                w_reg: float = 0.01,
                angle_factor=1.0,
                tx_factor=1.0,
                device: str = "cpu",
                ) -> None:
        """Initialize the left-right alignment transform.

        :param keys: Affine keys that will be updated with the estimated alignment.
        :param ref_im: Image or list of image keys used to estimate the symmetry alignment.
        :param ref_v2r: Dictionary key containing the reference voxel-to-RAS affine.
        :param rescaling_factor: Optional spatial rescaling factor used before optimization.
        :param w_reg: Weight of the regularization loss during alignment.
        :param angle_factor: Scaling factor applied to rotation parameters.
        :param tx_factor: Scaling factor applied to translation parameters.
        :param device: PyTorch device used for optimization.
        """

        #TODO check that all hyperparameters are used.

        self.keys = keys
        self.ref_im = ref_im if isinstance(ref_im, list) else [ref_im]
        self.ref_v2r = ref_v2r
        self.w_reg = w_reg
        self.device = device
        self.rescaling_factor = rescaling_factor

        self.angle_factor = angle_factor
        self.tx_factor = tx_factor
        if isinstance(angle_factor, (float, int)):
            self.angle_factor = np.array([angle_factor] * 3)

        if isinstance(tx_factor, (float, int)):
            self.tx_factor = np.array([tx_factor]*3)

    def _compute_cog(self, mask: np.ndarray, v2r_init: np.ndarray) -> np.ndarray:
        """Compute a translation matrix that centers the mask around its center of gravity.

        :param mask: Binary mask used to estimate the center of gravity.
        :param v2r_init: Initial voxel-to-RAS affine matrix.

        :return: 4x4 affine translation matrix that recenters the mask in RAS space.
        """

        idx = np.where(mask > 0)
        mx, my, mz = np.median(idx[0]), np.median(idx[1]), np.median(idx[2])
        ref_cog = v2r_init @ np.array([mx, my, mz, 1])

        T_ref_cog = np.eye(4)
        T_ref_cog[0, -1] = -ref_cog[0]
        T_ref_cog[1, -1] = -ref_cog[1]
        T_ref_cog[2, -1] = -ref_cog[2]

        return T_ref_cog

    def _align_LR(self, proxy: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
        """Optimize a rigid left-right alignment for a NIfTI proxy.

        :param proxy: NIfTI image containing the mask or stacked masks used for symmetry
            alignment.

        :return: Tuple with the optimized RAS-space transform and the center-of-gravity
            transform.
        """

        # Read input image(s)
        # TODO: check what is the shape of the proxy 3D or 5D and go ahead with one option only.
        v2r_init = proxy.affine.astype('float32')
        image_shape = proxy.shape[:3]
        mask = (np.asarray(proxy.dataobj) > 0).astype('float32')

        # Compte RAS
        if len(proxy.shape) == 5:
            mask = mask.reshape(image_shape + (-1, ))
        elif len(proxy.shape) == 3:
            mask = mask[..., np.newaxis]

        if self.rescaling_factor != 1:
            mask, v2r_init = fn_utils.rescale_voxel_factor(mask, v2r_init, factor=self.rescaling_factor, method='nearest')  # it does not really matter the precision, so go for nearest in case it is a labelmap (most common)

        mask = np.transpose(mask, axes=(3, 0, 1, 2))  # because it comes from a proxy, channels are in the last dimension
        mask_tensor = fn_utils.convert_to_tensor(mask[np.newaxis], dtype=torch.float16).to(self.device)

        T_ref_cog = self._compute_cog(mask[0], v2r_init)
        v2r = T_ref_cog @ v2r_init

        model = {'reg': models.InstanceAlignModelClassic(image_shape,
                                                         device=self.device,
                                                         v2r=v2r.astype('float32'),
                                                         angle_factor=torch.from_numpy(self.angle_factor).to(self.device),
                                                         tx_factor=torch.from_numpy(self.tx_factor).to(self.device)).to(self.device)}

        # Optimizer
        reg_lr = 1
        max_iter = 10
        optimizer = {'reg': torch.optim.LBFGS(params=model['reg'].parameters(), lr=reg_lr,
                                              max_iter=max_iter, line_search_fn='strong_wolfe')}

        # Training

        io.create_dir(REGISTRATION_DEFAULTS['results_dir'])
        pdict = {**REGISTRATION_DEFAULTS}
        loss_dict = {
            'symmetry': {'loss': fn_utils.DiceLoss(name='symmetry', device=self.device), 'weight': 1},
            'regularizer': {'loss': fn_utils.L2Loss(name='regularizer'), 'weight': self.w_reg},
        }

        training_session = models.JointInstanceAlign(loss_dict, pdict, device=self.device, da=[], trainable_keys={'reg': 'reg'}, verbose=False)

        _ = training_session.register({'mask': mask_tensor}, model, optimizer)

        return model['reg'].get_ras_matrix()[0].detach().cpu().numpy(), T_ref_cog

    def __call__(self, data_dict: dict, *args, **kwargs) -> dict:
        """Estimate and store left-right alignment matrices in the subject dictionary.

        :param data_dict: Subject dictionary containing reference images and affine matrices.

        :return: Updated subject dictionary with ``rot_*`` and ``aligned_*`` affine
        entries.
        """
        
        v2r = data_dict[self.ref_v2r]
        proxy = nib.Nifti1Image(np.stack([data_dict[r] for r in self.ref_im], axis=-1), v2r)

        ref_rot, ref_cog = self._align_LR(proxy)#nib.Nifti1Image(data_dict[self.ref_im], v2r))
        for k in self.keys:
            data_dict['rot_' + k] = np.linalg.inv(ref_rot) @ ref_cog
            data_dict['aligned_' + k] = np.linalg.inv(ref_rot) @ ref_cog @ data_dict[k]

        return data_dict

class CreateTemplateSpace(Transform):
    """Create a shared template grid for several images and masks.

    The transform computes a bounding box that contains all selected images in RAS
    space and resamples them into the same voxel grid.
    """

    def __init__(self, keys: dict, resolution: float | list = 1, name: str = "template") -> None:
        """Initialize the template-space creation transform.

        :param keys: Mapping from image keys to their corresponding affine keys.
        :param resolution: Template voxel spacing in millimeters, either scalar or 3-element
            sequence.
        :param name: Prefix used for the generated template-space dictionary keys.
        """

        self.keys = keys
        self.name = name

        if isinstance(resolution, (int, float)):
            resolution = [resolution] * 3

        self.resolution = resolution

    def run_tensor(self, proxy_list: list) -> dict:
        """Create template-space images from tensor-based proxy dictionaries.
        
        :param proxy_list: List of proxy dictionaries containing tensor data, shapes, affines
            and transpose metadata.

        :return: Dictionary with template-space images and the template affine matrix.
        """

        return_dict = {}
        images, v2r = fn_utils.create_template_space_tensor(proxy_list, mode='linear', resolution=self.resolution)
        return_dict[self.name + '_v2r'] = v2r
        return_dict[self.name + '_space'] = torch.zeros_like(images[0])
        for i, im_str in enumerate(self.keys.keys()):
            if proxy_list[i]['undo_tr']:
                return_dict[self.name + '_' + im_str] = torch.permute(images[i], proxy_list[i]['undo_tr'])
            else:
                return_dict[self.name + '_' + im_str] = images[i]
        return return_dict

    def run_array(self, proxy_list: list) -> dict:
        """Create template-space images from NumPy-based proxy dictionaries.

        :param proxy_list: List of proxy dictionaries containing array data, shapes, affines and
            transpose metadata.

        :return: Dictionary with template-space images and the template affine matrix.
        """

        return_dict = {}

        images, v2r = fn_utils.create_template_space(proxy_list, mode='linear', resolution=self.resolution)
        return_dict[self.name + '_v2r'] = v2r
        return_dict[self.name + '_space'] = np.zeros_like(images[0])
        for i, im_str in enumerate(self.keys.keys()):
            if proxy_list[i]['undo_tr']:
                return_dict[self.name + '_' + im_str] = np.transpose(images[i], proxy_list[i]['undo_tr'])
            else:
                return_dict[self.name + '_' + im_str] = images[i]

        return return_dict

    def __call__(self, data_dict: dict) -> dict:
        """Resample selected images and masks into a common template space.

        :param data_dict: Subject dictionary containing images, masks and their affine matrices.

        :return: Updated subject dictionary with template-space images and affine matrix.
        """

        run_tensor = False
        proxy_list, transpose_list = [], []
        for im_str, v2r_str in self.keys.items():
            data_shape = data_dict[im_str].shape
            # TODO: choose whether it is numpy or tensor and use only one of those.
            if isinstance(data_dict[im_str], torch.Tensor):
                run_tensor = True
                transpose_fn = torch.permute
                if len(data_shape) == 4 and data_shape[0] < data_shape[-1]:
                    data = torch.permute(data_dict[im_str], (1, 2, 3, 0))
                    undo_tr = (3, 0, 1, 2)

                elif len(data_shape) == 5 and data_shape[0] < data_shape[-1]:
                    data = torch.permute(data_dict[im_str], (2, 3, 4, 0, 1))
                    undo_tr = (3, 4, 0, 1, 2)

                else:
                    data = data_dict[im_str]
                    undo_tr = False

                data_shape = data.shape[:3]
                proxy_list.append({'data': data, 'affine': data_dict[v2r_str], 'shape': data_shape, 'undo_tr': undo_tr})

            else:
                run_tensor = False
                transpose_fn = np.transpose
                if len(data_shape) == 4 and data_shape[0] < data_shape[-1]:
                    data = np.transpose(data_dict[im_str], (1, 2, 3, 0))
                    undo_tr = (3, 0, 1, 2)

                elif len(data_shape) == 5 and data_shape[0] < data_shape[-1]:
                    data = np.transpose(data_dict[im_str], (2, 3, 4, 0, 1))
                    undo_tr = (3, 4, 0, 1, 2)

                else:
                    data = data_dict[im_str]
                    undo_tr = False

                proxy_list.append({'data': data, 'affine': data_dict[v2r_str], 'shape': data_shape, 'undo_tr': undo_tr})

        if run_tensor:
            d = self.run_tensor(proxy_list)
            data_dict = {**data_dict, **d}

        else:
            d = self.run_array(proxy_list)
            data_dict = {**data_dict, **d}

        return data_dict

class ToNumpy(Transform):
    """Convert selected tensor entries in a dictionary to NumPy arrays.

    This transform is typically used after PyTorch-based preprocessing or registration
    before saving outputs with nibabel or pandas.
    """

    def __init__(self, keys: list, to_nibabel: bool = False) -> None:
        """Initialize the tensor-to-NumPy conversion transform.

        :param keys: Dictionary keys converted from tensors to NumPy arrays.
        :param to_nibabel: If True, convert channel-first 4D tensors to nibabel-compatible
            channel last arrays.
        """

        self.keys = keys
        self.to_nibabel = to_nibabel

    def __call__(self, data_dict: dict, *args, **kwargs) -> dict:
        """Convert selected tensor entries to NumPy arrays in place.

        :param data_dict: Dictionary containing tensor or array entries.

        :return: Updated dictionary with selected entries converted to NumPy arrays.
        """

        for k in self.keys:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = np.squeeze(data_dict[k].detach().cpu().numpy())
                if len(data_dict[k].shape) == 4 and self.to_nibabel:
                    data_dict[k] = np.transpose(data_dict[k], axes=(1, 2, 3, 0))

        return data_dict

def get_preprocessing_transforms(device: str) -> dict:
    """Build the ordered preprocessing transform dictionary used before registration.

    :param device: PyTorch device used by transforms that require optimization.

    :return: Dictionary of named preprocessing transforms executed in order.
    """
    
    return {
        'align_dat': AlignLR(
            keys=['dat_v2r'],
            ref_im=['dat_brain'],
            ref_v2r='dat_v2r',
            rescaling_factor=1,
            angle_factor=np.array([10, 1, 1]),
            tx_factor=0.1,
            w_reg=0.1,
            device=device,
        ),
        'align_mri': AlignLR(
            keys=['label_v2r'],
            ref_im=['mask_brain'],
            ref_v2r='label_v2r',
            device=device,
            rescaling_factor=1,
            angle_factor=np.array([10, 1, 1]),
            tx_factor=0.1,
            w_reg=0.1,
        ),
        'template': CreateTemplateSpace(
            keys={
                'dat_image':   'aligned_dat_v2r',
                'dat_brain':   'aligned_dat_v2r',
                'mri_image':   'aligned_label_v2r',
                'mask_str':    'aligned_label_v2r',
                'mask_occ':    'aligned_label_v2r',
                'mask_brain':  'aligned_label_v2r',
            },
            name='template',
        ),
        'numpy': ToNumpy(keys=[
            'template_dat_image', 'template_dat_brain',
            'template_mask_str',  'template_mask_occ',
            'template_mask_brain', 'template_mri_image',
        ]),
    }
