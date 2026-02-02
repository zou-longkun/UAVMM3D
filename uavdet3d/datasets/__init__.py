from torch.utils.data import DataLoader

from .dataset import DatasetTemplate
from .mav6d.mav6d_det_dataset import MAV6D_Det_Dataset
from .mav6d.mav6d_pose_dataset import MAV6D_Pose_Dataset
from .carla.carla_det_dataset import CARLA_Det_Dataset
from .laa3d.laa3d_det_dataset import LAA3D_Det_Dataset
from .laam6d.laam6d_det_dataset import LAAM6D_Det_Dataset
from .laam6d.lidar_based_fusion_dataset import LidarBasedFusionDataset

import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from uavdet3d.utils import common_utils

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'MAV6D_Det_Dataset': MAV6D_Det_Dataset,
    'MAV6D_Pose_Dataset': MAV6D_Pose_Dataset,
    'CARLA_Det_Dataset': CARLA_Det_Dataset,
    'LAA3D_Det_Dataset': LAA3D_Det_Dataset,
    'LAAM6D_Det_Dataset': LAAM6D_Det_Dataset,
    'LidarBasedFusionDataset': LidarBasedFusionDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg,  batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        root_path=root_path,
        training=training,
        logger=logger,
    )


    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, 
        worker_init_fn=partial(common_utils.worker_init_fn, seed=seed),
        multiprocessing_context='spawn'  # 解决子进程初始化问题，但是会影响数据载入速度
    )

    return dataset, dataloader, sampler
