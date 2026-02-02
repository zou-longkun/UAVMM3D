import numpy as np
import cv2
import random
import copy
from functools import partial

class DataAugmentor:
    def __init__(self, root_path, dataset_cfg, logger):
        self.root_path = root_path
        self.dataset_cfg = dataset_cfg
        self.logger = logger
        self.aug_config_list = dataset_cfg.AUG_CONFIG_LIST

        self.augmentor_queue = []

        for cur_cfg in self.aug_config_list:
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.augmentor_queue.append(cur_augmentor)

    def image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_flip, config=config)

        data_dict = copy.deepcopy(data_dict)
        flip_x = 'x' in config.ALONG_AXIS_LIST
        flip_y = 'y' in config.ALONG_AXIS_LIST

        for key in config.DATA_LIST:
            if key not in data_dict:
                continue
            img = data_dict[key]
            if flip_x:
                img = img[:, :, ::-1] if img.ndim == 3 else img[:, ::-1]
            if flip_y:
                img = img[::-1, :, :] if img.ndim == 3 else img[::-1, :]
            data_dict[key] = img

        return data_dict

    def image_rot(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_rot, config=config)

        rot_range = config.ROT_RANGE
        angle = np.random.uniform(rot_range[0], rot_range[1]) * 180 / np.pi

        for key in config.DATA_LIST:
            if key not in data_dict:
                continue
            img = data_dict[key]
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            data_dict[key] = img

        return data_dict

    def image_brightness(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_brightness, config=config)

        factor = np.random.uniform(0.7, 1.3)
        for key in config.DATA_LIST:
            if key not in data_dict:
                continue
            img = data_dict[key].astype(np.float32)
            img *= factor
            img = np.clip(img, 0, 255).astype(np.uint8)
            data_dict[key] = img

        return data_dict

    def image_gaussian_noise(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_gaussian_noise, config=config)

        stddev = config.get('STDDEV', 10)
        for key in config.DATA_LIST:
            if key not in data_dict:
                continue
            img = data_dict[key].astype(np.float32)
            noise = np.random.normal(0, stddev, img.shape)
            img += noise
            img = np.clip(img, 0, 255).astype(np.uint8)
            data_dict[key] = img

        return data_dict

    def __call__(self, data_dict):
        for func in self.augmentor_queue:
            data_dict = func(data_dict)
        return data_dict