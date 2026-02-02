from uavdet3d.model.backbone_2d.resnet8x import ResNet8x
from uavdet3d.model.backbone_2d.resnet50 import ResNet50
from uavdet3d.model.backbone_2d.resnet50_v2 import ResNet50V2
from uavdet3d.model.backbone_2d.resnet101 import ResNet101
from uavdet3d.model.backbone_2d.resnet152 import ResNet152
from uavdet3d.model.backbone_2d.convnext_small import ConvNeXtSmall
from uavdet3d.model.backbone_2d.convnext_base import ConvNeXtBase
from uavdet3d.model.backbone_2d.convnext_large import ConvNeXtLarge
from uavdet3d.model.backbone_2d.vit_l import VitL
from uavdet3d.model.backbone_2d.vit_b import VitB
from .resnet8x_attention import ResNet8xAttention
from .lightweight_attention_net import LightweightAttentionNet

__all__ = {
    'ResNet50': ResNet50,
    'ResNet50V2': ResNet50V2,
    'ResNet101': ResNet101,
    'ConvNeXtSmall': ConvNeXtSmall,
    'ConvNeXtBase': ConvNeXtBase,
    'ConvNeXtLarge': ConvNeXtLarge,
    'VitL': VitL,
    'VitB': VitB,
    'ResNet152': ResNet152,
    'ResNet8xAttention': ResNet8xAttention,
    'LightweightAttentionNet': LightweightAttentionNet
}