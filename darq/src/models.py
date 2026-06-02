import time

import torch
import torch.nn as nn
import numpy as np

from darq.utils import fn_utils
from darq.utils.io import PrinterCallback


# ── Deformation models ─────────────────────────────────────────────────────────


class InstanceModelClassic(nn.Module):

    def __init__(self, image_shape, ref_v2r, flo_v2r=None, batchsize=1, device='cpu', **kwargs):
        super().__init__()

        self.device = device
        self.batchsize = batchsize
        self.image_shape = image_shape
        if len(image_shape) > 3:
            self.image_shape = image_shape[2:]

        self.ndims = 3
        self.ref_v2r = torch.from_numpy(ref_v2r).to(device)
        self.flo_v2r = torch.from_numpy(flo_v2r).to(device) if flo_v2r is not None else self.ref_v2r

        vectors = [torch.arange(0, s) for s in self.image_shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        self.grid = torch.stack(grids).to(device)  # y, x, z

    def _compute_ras_matrix(self, *args, **kwargs):
        raise NotImplementedError

    def set_params(self, *args, **kwargs):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def get_params_scaled(self):
        return self.get_params()

    def _compute_rotation(self, rotation):
        shape = rotation[..., 0].shape + (1,)

        Rx_row0 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([1., 0., 0.])), 0), shape),
                                  axis=1).to(self.device)
        Rx_row1 = torch.stack([torch.zeros(shape).to(self.device), torch.unsqueeze(torch.cos(rotation[..., 0]), -1),
                               torch.unsqueeze(-torch.sin(rotation[..., 0]), -1)], axis=-1)
        Rx_row2 = torch.stack([torch.zeros(shape).to(self.device), torch.unsqueeze(torch.sin(rotation[..., 0]), -1),
                               torch.unsqueeze(torch.cos(rotation[..., 0]), -1)], axis=-1)
        Rx = torch.cat([Rx_row0, Rx_row1, Rx_row2], axis=1)

        Ry_row0 = torch.stack([torch.unsqueeze(torch.cos(rotation[..., 1]), -1), torch.zeros(shape).to(self.device),
                               torch.unsqueeze(torch.sin(rotation[..., 1]), -1)], axis=-1)
        Ry_row1 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([0., 1., 0.])), 0), shape),
                                  axis=1).to(self.device)
        Ry_row2 = torch.stack([torch.unsqueeze(-torch.sin(rotation[..., 1]), -1), torch.zeros(shape).to(self.device),
                               torch.unsqueeze(torch.cos(rotation[..., 1]), -1)], axis=-1)
        Ry = torch.cat([Ry_row0, Ry_row1, Ry_row2], axis=1)

        Rz_row0 = torch.stack(
            [torch.unsqueeze(torch.cos(rotation[..., 2]), -1), torch.unsqueeze(-torch.sin(rotation[..., 2]), -1),
             torch.zeros(shape).to(self.device)], axis=-1)
        Rz_row1 = torch.stack(
            [torch.unsqueeze(torch.sin(rotation[..., 2]), -1), torch.unsqueeze(torch.cos(rotation[..., 2]), -1),
             torch.zeros(shape).to(self.device)], axis=-1)
        Rz_row2 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([0., 0., 1.])), 0), shape),
                                  axis=1).to(self.device)
        Rz = torch.cat([Rz_row0, Rz_row1, Rz_row2], axis=1)

        T_rot = torch.matmul(torch.matmul(Rx, Ry), Rz)

        return T_rot

    def _compute_matrix(self, *args, **kwargs):
        T_rig = self._compute_ras_matrix(**kwargs)
        T_rig_list = torch.unbind(T_rig, dim=0)
        T_rig_ras_list = [torch.linalg.inv(self.flo_v2r) @ T @ self.ref_v2r for T in T_rig_list]
        T = torch.stack(T_rig_ras_list, 0)

        return T.to(self.device)

    def get_ras_matrix(self):
        return self._compute_ras_matrix()

    def get_matrix(self):
        return self._compute_matrix()

    def forward(self, image_targ, **kwargs):
        T = self._compute_matrix()
        im = torch.permute(image_targ[0], (1, 2, 3, 0))

        T = T[0]
        di = T[0, 0] * self.grid[0] + T[0, 1] * self.grid[1] + T[0, 2] * self.grid[2] + T[0, 3]
        dj = T[1, 0] * self.grid[0] + T[1, 1] * self.grid[1] + T[1, 2] * self.grid[2] + T[1, 3]
        dk = T[2, 0] * self.grid[0] + T[2, 1] * self.grid[1] + T[2, 2] * self.grid[2] + T[2, 3]

        image_reg = fn_utils.fast_3D_interp_field_torch(im, di, dj, dk, mode='linear')

        return torch.unsqueeze(torch.permute(image_reg, (3, 0, 1, 2)), 0)

class InstanceAlignModelClassic(InstanceModelClassic):

    def __init__(self, image_shape, v2r, batchsize=1, device='cpu', cog=None,
                 tx_init=None, angle_init=None, tx_factor=1, angle_factor=1):
        super().__init__(image_shape, ref_v2r=v2r, batchsize=batchsize, device=device)

        self.tx_factor = tx_factor
        self.angle_factor = angle_factor
        self.cog = cog

        # Parameters
        if angle_init is not None:
            if torch.is_tensor(angle_init):
                self.angle = torch.nn.Parameter(angle_init)
            else:
                self.angle = torch.nn.Parameter(torch.from_numpy(angle_init))

        else:
            self.angle = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        if tx_init is not None:
            if torch.is_tensor(tx_init):
                self.translation = torch.nn.Parameter(tx_init)
            else:
                self.translation = torch.nn.Parameter(torch.from_numpy(tx_init))
        else:
            self.translation = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        self.angle.requires_grad = True
        self.translation.requires_grad = True

    def set_params(self, params):
        self.angle = torch.nn.Parameter(params[0]).to(self.device)
        self.translation = torch.nn.Parameter(params[1]).to(self.device)

    def get_params(self):
        return self.angle, self.translation

    def get_params_scaled(self):
        return self.angle * self.angle_factor, self.translation * self.tx_factor

    def _compute_matrix(self, flip_lr=False):
        T_rig = self._compute_ras_matrix(flip_lr=flip_lr)
        T_rig_list = torch.unbind(T_rig, dim=0)
        T_rig_ras_list = [torch.linalg.inv(self.flo_v2r) @ T @ self.ref_v2r for T in T_rig_list]
        T = torch.stack(T_rig_ras_list, 0)

        return T.to(self.device)

    def _compute_ras_matrix(self, flip_lr=False):

        angle, tx = self.get_params()

        T_center = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center[:, 0, 0] = 1
        T_center[:, 1, 1] = 1
        T_center[:, 2, 2] = 1
        T_center[:, 3, 3] = 1
        if self.cog is not None:
            T_center[:, :3, 3] = torch.from_numpy(-self.cog)

        T_center_inv = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center_inv[:, 0, 0] = 1
        T_center_inv[:, 1, 1] = 1
        T_center_inv[:, 2, 2] = 1
        T_center_inv[:, 3, 3] = 1
        if self.cog is not None:
            T_center_inv[:, :3, 3] = torch.from_numpy(self.cog)

        T_trans = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_trans[:, :3, 3] = tx
        T_trans[:, 0, 0] = 1
        T_trans[:, 1, 1] = 1
        T_trans[:, 2, 2] = 1
        T_trans[:, 3, 3] = 1

        T_rot = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_rot[:, :3, :3] = self._compute_rotation(angle)
        T_rot[:, 3, 3] = 1

        T_flip_lr = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_flip_lr[:, 0, 0] = -1
        T_flip_lr[:, 1, 1] = 1
        T_flip_lr[:, 2, 2] = 1
        T_flip_lr[:, 3, 3] = 1

        if flip_lr:
            T_rig = T_center_inv @ T_trans @ T_rot @ T_flip_lr @ T_center
        else:
            T_rig = T_center_inv @ T_trans @ T_rot @ T_center

        return T_rig

    def forward(self, image_targ, **kwargs):
        T = self._compute_matrix(flip_lr=False)
        T_flip = self._compute_matrix(flip_lr=True)
        im = torch.permute(image_targ[0], (1, 2, 3, 0))

        T = T[0]
        di = T[0, 0] * self.grid[0] + T[0, 1] * self.grid[1] + T[0, 2] * self.grid[2] + T[0, 3]
        dj = T[1, 0] * self.grid[0] + T[1, 1] * self.grid[1] + T[1, 2] * self.grid[2] + T[1, 3]
        dk = T[2, 0] * self.grid[0] + T[2, 1] * self.grid[1] + T[2, 2] * self.grid[2] + T[2, 3]

        image_reg = fn_utils.fast_3D_interp_field_torch(im, di, dj, dk, mode='linear')

        T = T_flip[0]
        di = T[0, 0] * self.grid[0] + T[0, 1] * self.grid[1] + T[0, 2] * self.grid[2] + T[0, 3]
        dj = T[1, 0] * self.grid[0] + T[1, 1] * self.grid[1] + T[1, 2] * self.grid[2] + T[1, 3]
        dk = T[2, 0] * self.grid[0] + T[2, 1] * self.grid[1] + T[2, 2] * self.grid[2] + T[2, 3]

        image_flip_reg = fn_utils.fast_3D_interp_field_torch(im, di, dj, dk, mode='linear')

        return torch.unsqueeze(torch.permute(image_reg, (3, 0, 1, 2)), 0), torch.unsqueeze(torch.permute(image_flip_reg, (3, 0, 1, 2)), 0)

class InstanceRigidModelClassic(InstanceModelClassic):

    def __init__(self, image_shape, ref_v2r, flo_v2r=None, batchsize=1, device='cpu', cog=None,
                 tx_init=None, angle_init=None, tx_factor=1, angle_factor=1):
        super().__init__(image_shape, ref_v2r=ref_v2r, flo_v2r=flo_v2r, batchsize=batchsize, device=device)

        self.tx_factor = torch.from_numpy(tx_factor).to(device)
        self.angle_factor = torch.from_numpy(angle_factor).to(device)
        self.cog = cog

        # Parameters
        if angle_init is not None:
            if torch.is_tensor(angle_init):
                self.angle = torch.nn.Parameter(angle_init)
            else:
                self.angle = torch.nn.Parameter(torch.from_numpy(angle_init))

        else:
            self.angle = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        if tx_init is not None:
            if torch.is_tensor(tx_init):
                self.translation = torch.nn.Parameter(tx_init)
            else:
                self.translation = torch.nn.Parameter(torch.from_numpy(tx_init))
        else:
            self.translation = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        self.angle.requires_grad = True
        self.translation.requires_grad = True

    def set_params(self, params):
        self.angle = torch.nn.Parameter(params[0])
        self.translation = torch.nn.Parameter(params[1])

    def get_params(self):
        return self.angle, self.translation

    def get_params_scaled(self):
        return self.angle * self.angle_factor, self.translation * self.tx_factor

    def _compute_ras_matrix(self):

        angle, tx = self.get_params()

        T_center = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center[:, 0, 0] = 1
        T_center[:, 1, 1] = 1
        T_center[:, 2, 2] = 1
        T_center[:, 3, 3] = 1
        if self.cog is not None:
            T_center[:, :3, 3] = torch.from_numpy(-self.cog)

        T_center_inv = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center_inv[:, 0, 0] = 1
        T_center_inv[:, 1, 1] = 1
        T_center_inv[:, 2, 2] = 1
        T_center_inv[:, 3, 3] = 1
        if self.cog is not None:
            T_center_inv[:, :3, 3] = torch.from_numpy(self.cog)

        T_trans = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_trans[:, :3, 3] = tx
        T_trans[:, 0, 0] = 1
        T_trans[:, 1, 1] = 1
        T_trans[:, 2, 2] = 1
        T_trans[:, 3, 3] = 1

        T_rot = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_rot[:, :3, :3] = self._compute_rotation(angle)
        T_rot[:, 3, 3] = 1

        T_rig = T_center_inv @ T_trans @ T_rot @ T_center

        return T_rig



# ── Optimising functions ─────────────────────────────────────────────────────────

class JointInstanceAlign(object):

    def __init__(self, loss_dict, p_dict, device='cpu', trainable_keys=None, verbose=True, **kwargs):
        self.loss_dict = loss_dict
        self.log_keys = ['loss_' + loss['loss'].name for loss in loss_dict.values()] + \
                        ['w_loss_' + loss['loss'].name for loss in loss_dict.values()] + \
                        ['val_loss_' + loss['loss'].name for loss in loss_dict.values()] + \
                        ['val_loss', 'loss', 'time_duration (s)']

        self.callbacks = []
        if verbose:
            self.callbacks = [PrinterCallback(keys=self.log_keys, freq_print=kwargs.get('freq_print'))]

        self.trainable_keys = trainable_keys if trainable_keys is not None else {}
        if not isinstance(self.trainable_keys, dict): self.trainable_keys = {self.trainable_keys: self.trainable_keys}

        if 'val_freq' not in p_dict.keys():
            p_dict['val_freq'] = 5

        self.main_dict = p_dict
        self.device = device
        self.kwargs = kwargs

    def train(self, data_dict, model_dict, optimizer_dict, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.main_dict['starting_epoch'])

        best_loss = 100000000
        logs_dict = {}
        persistent_epoch = 0
        for epoch in range(self.main_dict['starting_epoch'], self.main_dict['num_epochs']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            for m in self.trainable_keys.values():
                model_dict[m].train()

            logs_dict = self.iterate(data_dict, model_dict, optimizer_dict, **kwargs)


            if ((best_loss - logs_dict['loss']) / np.abs(best_loss)) * 100 < 1:  # if loss improves less than 0.1%
                persistent_epoch += 1
            else:
                persistent_epoch = 0
                best_loss = logs_dict['loss']

            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time
            logs_dict = {**logs_dict}

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

            if persistent_epoch >= self.main_dict['patience']:
                break

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        return logs_dict

    def register(self, data_dict, model_dict, optimizer_dict, **kwargs):
        logs_dict = self.train(data_dict, model_dict, optimizer_dict, **kwargs)
        model_dict['reg'].eval()
        data_dict['parameters'] = model_dict['reg'].get_params()
        with torch.no_grad():
            data_dict['loss'] = logs_dict['loss']
            data_dict['affine'] = model_dict['reg'].get_matrix()
            data_dict['affine_ras'] = model_dict['reg'].get_ras_matrix()
            data_dict = self.forward(data_dict, model_dict)

        return data_dict

    def forward(self, data_dict, model_dict):
        if 'image' in data_dict.keys():
            im, im_flip = model_dict['reg'](torch.cat((data_dict['image'], data_dict['mask']), axis=1))
            data_dict['reg_image'], data_dict['reg_image_flip'] = im[:, 0:1], im_flip[:, 0:1]
            data_dict['reg_mask'], data_dict['reg_mask_flip'] = im[:, 1:], im_flip[:, 1:]
        else:
            data_dict['reg_mask'], data_dict['reg_mask_flip'] = model_dict['reg'](data_dict['mask'])

        data_dict['reg_mask'] = torch.permute(data_dict['reg_mask'], (1, 0, 2, 3, 4))
        data_dict['reg_mask_flip'] = torch.permute(data_dict['reg_mask_flip'], (1, 0, 2, 3, 4))

        return data_dict

    def iterate(self, data_dict, model_dict, optimizer_dict, **kwargs):
        if type(optimizer_dict['reg']) == torch.optim.LBFGS:
            def closure():
                optimizer_dict['reg'].zero_grad()

                data_dict_closure = self.forward(data_dict, model_dict)

                loss, _ = self.compute_loss(data_dict_closure, model_dict)
                loss.backward()

                return loss

            optimizer_dict['reg'].step(closure=closure)

        else:
            data_dict = self.forward(data_dict, model_dict)
            loss, _ = self.compute_loss(data_dict, model_dict)
            loss.backward()

            optimizer_dict['reg'].step()

        with torch.no_grad():
            data_dict = self.forward(data_dict, model_dict)
            loss, log_dict = self.compute_loss(data_dict, model_dict)
            log_dict['loss'] = loss.item()

        return log_dict

    def compute_loss(self, data_dict, model_dict):
        log_dict = {}

        # Registration
        SYM_loss = 0.
        if self.loss_dict['symmetry']['weight'] > 0:
            SYM_loss = self.loss_dict['symmetry']['loss'](data_dict['reg_mask'], data_dict['reg_mask_flip'])
            log_dict['loss_' + self.loss_dict['symmetry']['loss'].name] = SYM_loss.item()
            SYM_loss = self.loss_dict['symmetry']['weight'] * SYM_loss
            log_dict['w_loss_' + self.loss_dict['symmetry']['loss'].name] = SYM_loss.item()

        SYM_im_loss = 0.
        if 'image' in data_dict.keys() and self.loss_dict['symmetry_im']['weight'] > 0:
            SYM_im_loss = self.loss_dict['symmetry_im']['loss'](data_dict['reg_image'], data_dict['reg_image_flip'], mask=data_dict['reg_mask'][0:1]*data_dict['reg_mask_flip'][0:1])
            log_dict['loss_' + self.loss_dict['symmetry_im']['loss'].name] = SYM_im_loss.item()
            SYM_im_loss = self.loss_dict['symmetry_im']['weight'] * SYM_im_loss
            log_dict['w_loss_' + self.loss_dict['symmetry_im']['loss'].name] = SYM_im_loss.item()


        REG_loss = 0.
        if self.loss_dict['regularizer']['weight'] > 0:
            params = model_dict['reg'].get_params_scaled()
            for p in params:
                REG_loss += self.loss_dict['regularizer']['loss'](p, torch.zeros_like(p))
            log_dict['loss_' + self.loss_dict['regularizer']['loss'].name] = REG_loss.item()
            REG_loss = self.loss_dict['regularizer']['weight'] * REG_loss
            log_dict['w_loss_' + self.loss_dict['regularizer']['loss'].name] = REG_loss.item()

        R_loss = SYM_loss + SYM_im_loss + REG_loss
        return R_loss, log_dict

class JointInstanceReg(JointInstanceAlign):

    def forward(self, data_dict, model_dict):
        if 'flo_mask' in data_dict.keys():
            im = model_dict['reg'](torch.cat((data_dict['flo_image'], data_dict['flo_mask']), axis=1))
            data_dict['reg_image'] = im[:, 0:1]
            data_dict['reg_mask'] = im[:, 1:]
        else:
            data_dict['reg_image'] = model_dict['reg'](data_dict['flo_image'])

        return data_dict

    def iterate(self, data_dict, model_dict, optimizer_dict, **kwargs):
        def closure():
            if torch.is_grad_enabled():
                for k in self.trainable_keys.values():
                    optimizer_dict[k].zero_grad()

            data_dict_closure = self.forward(data_dict, model_dict)

            loss, _ = self.compute_loss(data_dict_closure, model_dict)
            loss.backward()

            return loss

        for k in self.trainable_keys.values():
            optimizer_dict[k].step(closure=closure)

        with torch.no_grad():
            data_dict = self.forward(data_dict, model_dict)
            loss, log_dict = self.compute_loss(data_dict, model_dict)

        log_dict['loss'] = loss.item()

        return log_dict

    def compute_loss(self, data_dict, model_dict):
        log_dict = {}

        ref_mask = None
        if 'ref_mask' in data_dict.keys():
            ref_mask = data_dict['ref_mask']

        SIM_loss = 0.
        if self.loss_dict['reg']['weight'] > 0:
            for it_chan in range(data_dict['ref_image'].shape[1]):
                SIM_loss += self.loss_dict['reg']['loss'](data_dict['reg_image'],
                                                          data_dict['ref_image'][:, it_chan:it_chan+1],
                                                          v2r=data_dict['template_v2r'], mask=ref_mask) / data_dict['ref_image'].shape[1]
            log_dict['loss_' + self.loss_dict['reg']['loss'].name] = SIM_loss.item()
            SIM_loss = self.loss_dict['reg']['weight'] * SIM_loss
            log_dict['w_loss_' + self.loss_dict['reg']['loss'].name] = SIM_loss.item()

        UPTAKE_loss = 0.
        if self.loss_dict['reg_uptake']['weight'] > 0:
            for it_chan in range(data_dict['ref_image'].shape[1]):
                UPTAKE_loss += self.loss_dict['reg_uptake']['loss'](data_dict['reg_image'],
                                                                    data_dict['ref_image'][:, it_chan:it_chan + 1],
                                                                    v2r=data_dict['template_v2r'], mask=ref_mask) / data_dict['ref_image'].shape[1]
            log_dict['loss_' + self.loss_dict['reg_uptake']['loss'].name] = UPTAKE_loss.item()
            UPTAKE_loss = self.loss_dict['reg_uptake']['weight'] * UPTAKE_loss
            log_dict['w_loss_' + self.loss_dict['reg_uptake']['loss'].name] = UPTAKE_loss.item()

        # Registration
        SIM_label_loss = 0.
        if self.loss_dict['reg_label']['weight'] > 0:
            SIM_label_loss = self.loss_dict['reg_label']['loss'](data_dict['reg_mask'], data_dict['ref_mask'],
                                                                 v2r=data_dict['template_v2r'])
            log_dict['loss_' + self.loss_dict['reg_label']['loss'].name] = SIM_label_loss.item()
            SIM_label_loss = self.loss_dict['reg_label']['weight'] * SIM_label_loss
            log_dict['w_loss_' + self.loss_dict['reg_label']['loss'].name] = SIM_label_loss.item()

        # Registration
        SYM_loss = 0.
        if self.loss_dict['reg_lr']['weight'] > 0:
            SYM_loss = self.loss_dict['reg_lr']['loss'](data_dict['reg_image'])
            log_dict['loss_' + self.loss_dict['reg_lr']['loss'].name] = SYM_loss.item()
            SYM_loss = self.loss_dict['reg_lr']['weight'] * SYM_loss
            log_dict['w_loss_' + self.loss_dict['reg_lr']['loss'].name] = SYM_loss.item()

        REG_loss = 0.
        if self.loss_dict['regularizer']['weight'] > 0:
            params = model_dict['reg'].get_params_scaled()
            for p in params:
                REG_loss += self.loss_dict['regularizer']['loss'](p, torch.zeros_like(p))
            log_dict['loss_' + self.loss_dict['regularizer']['loss'].name] = REG_loss.item()
            REG_loss = self.loss_dict['regularizer']['weight'] * REG_loss
            log_dict['w_loss_' + self.loss_dict['regularizer']['loss'].name] = REG_loss.item()

        R_loss = REG_loss + SIM_loss + SIM_label_loss + SYM_loss + UPTAKE_loss
        log_dict['loss'] = R_loss.item()

        return R_loss, log_dict

