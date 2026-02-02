from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import torch.utils.data as torch_data
import os
from .augmentor.data_augmentor import DataAugmentor
from .pre_processor.pre_processor import DataPreProcessor
from .pre_processor.pre_processor_laam6d import DataPreProcessorLAAm6d
from .pre_processor.pre_processor_lidar_fusion1 import DataPreProcessorLidarFusion
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
            # 避免把 dict/list 强行 np.stack 成 object 数组（会导致内存暴涨）
            if isinstance(val[0], (dict, list, tuple)) and key not in ['gt_name', 'gt_diff', 'gt_names']:
                ret[key] = val
                continue

            # torch tensor
            if torch.is_tensor(val[0]):
                ret[key] = torch.stack(val, dim=0)
                continue

            # numpy array
            if isinstance(val[0], np.ndarray):
                ret[key] = np.stack(val, axis=0)
                continue

            # 其他类型（字符串、object等）
            ret[key] = val

        ret['batch_size'] = batch_size
        return ret
