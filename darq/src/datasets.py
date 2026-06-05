# py

# third party imports
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
from skimage.morphology import binary_opening
from skimage import filters
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

# project imports
from darq.utils.io import remove_synthseg_parcellation, remove_synthseg_hemisphere
from darq.utils import fn_utils



class MRI_DaT(Dataset):
    def __init__(self,
                 subject_df,
                 transforms=None,
                 crop_dat=False,
                 crop_labels=False,
                 ):

        super().__init__()

        self.subject_df = subject_df

        self.N = len(subject_df)

        self.transforms = transforms if transforms is not None else []
        self.crop_dat = crop_dat
        self.crop_labels = crop_labels

    def _get_ROI_masks(self, lab_image):
        mask_str_L = ((lab_image == 11) + (lab_image == 12))
        mask_str_R = ((lab_image == 50) + (lab_image == 51))
        mask_str = mask_str_L + mask_str_R

        mask_occ_L = (lab_image == 1005) + (lab_image == 1011) + (lab_image == 1013) + (lab_image == 1021)
        mask_occ_R = (lab_image == 2005) + (lab_image == 2011) + (lab_image == 2013) + (lab_image == 2021)
        mask_occ = mask_occ_L + mask_occ_R

        return mask_str.astype('float'), mask_occ.astype('float')


    def __getitem__(self, index):
        if index not in self.subject_df.index:
            return None

        subject = self.subject_df.loc[index]
        subject = {**subject, 'mri': nib.load(subject['mri_path']), 'label': nib.load(subject['label_path']), 'dat': nib.load(subject['dat_path'])}

        if np.sum(np.abs(subject['label'].affine - subject['mri'].affine)) > 1:
            print(subject['label'].affine)
            print(subject['mri'].affine)

        subject['label_v2r'] = subject['label'].affine

        subject['label_image'] = np.array(subject['label'].dataobj)
        subject['label_image'] = remove_synthseg_parcellation(subject['label_image'])
        subject['label_image'] = remove_synthseg_hemisphere(subject['label_image'])

        if self.crop_labels:
            subject['label_image'], crop_coords = fn_utils.crop_label(subject['label_image'], margin=10, threshold=0)
            tx_crop = np.array([crop_coords[0][0], crop_coords[1][0], crop_coords[2][0], 1])
            T_crop = np.eye(4)
            T_crop[0, 3] = tx_crop[0]
            T_crop[1, 3] = tx_crop[1]
            T_crop[2, 3] = tx_crop[2]
            subject['label_v2r'] = subject['label_v2r'] @ T_crop
            subject['label_crop'] = T_crop

        subject['mri'] = fn_utils.vol_resample_fast(nib.Nifti1Image(subject['label_image'], subject['label_v2r']), subject['mri'])
        subject['mri_image'] = np.array(subject['mri'].dataobj)
        subject['mask_str'], subject['mask_occ'] = self._get_ROI_masks(subject['label_image'])
        non_cerebrum = (subject['label_image'] <= 0) | (subject['label_image'] == 7) | (subject['label_image'] == 8) | (subject['label_image'] == 46) | (subject['label_image'] == 47) | (subject['label_image'] == 15) | (subject['label_image'] == 16) | (subject['label_image'] == 24)
        subject['mask_brain'] = (1 - non_cerebrum).astype('float')
        subject['mask_cau'] = (subject['label_image'] == 11).astype('float')
        subject['mask_pu'] = (subject['label_image'] == 12).astype('float')


        dat_image = np.squeeze(np.array(subject['dat'].dataobj).astype('float32'))
        dat_v2r = subject['dat'].affine
        if self.crop_dat:
            Crop_th = filters.threshold_otsu(dat_image)
            mask_crop = binary_opening(dat_image > Crop_th, np.ones((3, 3, 3))).astype('float32')
            _, crop_coords = fn_utils.crop_label(mask_crop, margin=15, threshold=0)
            dat_image = fn_utils.apply_crop(dat_image, crop_coords)
            tx_crop = np.array([crop_coords[0][0], crop_coords[1][0], crop_coords[2][0], 1])
            T_crop = np.eye(4)
            T_crop[0, 3] = tx_crop[0]
            T_crop[1, 3] = tx_crop[1]
            T_crop[2, 3] = tx_crop[2]
            dat_v2r = dat_v2r @ T_crop
            subject['dat_crop'] = T_crop

        dat_res = np.sqrt(np.sum(dat_v2r * dat_v2r, axis=0))[:-1]
        mri_res = np.sqrt(np.sum(subject['label_v2r'] * subject['label_v2r'], axis=0))[:-1]
        n_clusters = 6
        dat_seg = GMM(n_components=n_clusters, random_state=0).fit_predict(dat_image.reshape((-1, 1)))
        dat_seg = dat_seg.reshape(dat_image.shape)
        dat_roi_means = [np.mean(dat_image[dat_seg == ul]) for ul in np.unique(dat_seg)]
        dat_rois_ordered = np.argsort(dat_roi_means)

        dat_brain = np.zeros(dat_image.shape + (1, ))
        dat_brain[dat_seg == dat_rois_ordered[-1], 0] = 1
        dat_rois_ordered = dat_rois_ordered[1:-1] 
        for i, i_k in enumerate(dat_rois_ordered[::-1]):
            dat_brain[dat_seg == i_k, 0] = 1
            if np.prod(dat_res) * np.sum(dat_brain[..., 0]) > 2 * np.prod(mri_res) * np.sum(subject['mask_brain']):
                break


        subject['dat_image'] = dat_image
        subject['dat_str'] = dat_brain
        subject['dat_brain'] = dat_brain
        subject['dat_v2r'] = dat_v2r

        for data_tf in self.transforms:
            subject = data_tf(subject)

        subject['id'] = str(subject['subject'])
        if 'sess' in subject.keys():
            subject['id'] += '_' + str(subject['sess'])

        return subject

    def __len__(self):
        return self.N
