from .detector_template import DetectorTemplate
from .key_point_pose import KeyPoint2Pose
from .center_det import CenterDet
from .center_det_laam6d import CenterDetLaam6d
from .lidar_based_fusion_detector import LidarBasedFusionDetector
__all__ = {
    'DetectorTemplate': DetectorTemplate,
    'KeyPoint2Pose': KeyPoint2Pose,
    'CenterDet': CenterDet,
    'CenterDetLaam6d': CenterDetLaam6d,
    'LidarBasedFusionDetector': LidarBasedFusionDetector
}


def build_detector(model_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, dataset=dataset
    )
    return model
