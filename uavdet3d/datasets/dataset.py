from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import torch.utils.data as torch_data
import os
from .augmentor.data_augmentor import DataAugmentor
from .pre_processor.pre_processor import DataPreProcessor
from .pre_processor.pre_processor_laam6d import DataPreProcessorLAAm6d
from .pre_processor.pre_processor_lidar_fusion import DataPreProcessorLidarFusion
import copy
import time


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.logger = logger
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        if self.dataset_cfg.DATASET == 'LAA3D_Det_Dataset':
            self.data_pre_processor = DataPreProcessor(self.dataset_cfg, training=self.training)
        elif self.dataset_cfg.DATASET == 'LAAM6D_Det_Dataset':
            self.data_pre_processor = DataPreProcessorLAAm6d(self.dataset_cfg, training=self.training)
        elif self.dataset_cfg.DATASET == 'LidarBasedFusionDataset':
            self.data_pre_processor = DataPreProcessorLidarFusion(self.dataset_cfg, training=self.training)

        # self.data_augmentor = DataAugmentor(self.root_path, self.dataset_cfg.DATA_AUGMENTOR, logger=self.logger) if self.training else None


    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def collate_batch(self, batch_list, _unused=False):

        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}


        for key, val in data_dict.items():
            try:
                if key in ['image', 'gt_heatmap']:  # [B,K,3,W,H]
                    ret[key] = np.stack(val, axis=0)
                elif key in ['gt_box9d']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes9dof = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes9dof[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes9dof
                elif key in ['gt_name', 'gt_diff']:
                    batch_gt_name = []
                    for k in range(batch_size):
                        batch_gt_name.append(val[k])
                    ret[key] = batch_gt_name
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size

        return ret
