from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, dataset):
    model = build_detector(
        model_cfg=model_cfg,  dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['batch_size', 'scene_id', 'seq_id', 'frame_id', 'brightness', 'intrinsic', 'extrinsic', 'distortion', 'raw_im_size', 'new_im_size', 'obj_size', 'stride','im_num','obj_num','gt_name','relative_im_path','setting_id', 'gt_diff']:
            continue

        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict = model(batch_dict)
        loss = ret_dict['loss']
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()
        
        return loss

    return model_func