import os
import torch
import torch.nn as nn
from uavdet3d.model import backbone_2d, backbone_3d, dense_head_2d, dense_head_3d, roi_head_2d, roi_head_3d


class DetectorTemplate(nn.Module):
    def __init__(self, model_cfg, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'backbone_2d', 'backbone_3d', 'dense_head_2d',  'dense_head_3d',  'roi_head_2d',  'roi_head_3d'
        ]

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'image_shape': self.dataset.dataset_cfg.IM_RESIZE,
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict
        this_module = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        this_module = backbone_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_dense_head_2d(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_2D', None) is None:
            return None, model_info_dict
        this_module = dense_head_2d.__all__[self.model_cfg.DENSE_HEAD_2D.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_2D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_dense_head_3d(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_3D', None) is None:
            return None, model_info_dict
        this_module = dense_head_3d.__all__[self.model_cfg.DENSE_HEAD_3D.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_3D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_roi_head_2d(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD_2D', None) is None:
            return None, model_info_dict
        this_module = roi_head_2d.__all__[self.model_cfg.ROI_HEAD_2D.NAME](
            model_cfg=self.model_cfg.ROI_HEAD_2D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_roi_head_3d(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD_3D', None) is None:
            return None, model_info_dict
        this_module = roi_head_3d.__all__[self.model_cfg.ROI_HEAD_3D.NAME](
            model_cfg=self.model_cfg.ROI_HEAD_3D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        raise NotImplementedError

    def load_params_from_file(self, filename, to_cpu):
        loc_type = torch.device('cpu') if to_cpu else None
        dict_sta = torch.load(filename, map_location=loc_type) # weights_only=False 较高pytorch版本才可以用这个参数
        print('loading weights')
        self.load_state_dict(dict_sta['model_state'],strict=False)

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type) # weights_only=False 较高pytorch版本才可以用这个参数
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        return it, epoch

