import torch
import torch.nn as nn
from .resnet8x import ResNet8x


class ResNet8xModalityAttention(ResNet8x):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self.modal_indices = {
            'rgb': slice(0, 3),
            'ir': slice(3, 6),
            'dvs': slice(6, 9),
            'depth': slice(9, 10)
        }

        # 可学习权重，初始化为 1
        self.modal_weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1)) for name in self.modal_indices
        })

    def forward(self, batch_dict):
        x = batch_dict['image']  # [B, K, 10, W, H]
        B, K, C, W, H = x.shape
        x = x.reshape(B * K, C, W, H)  # [BK, 10, W, H]

        modal_parts = []
        for name, sl in self.modal_indices.items():
            weight = self.modal_weights[name].sigmoid()  # 可学习权重 ∈ (0,1)
            part = x[:, sl, :, :] * weight
            modal_parts.append(part)

        x = torch.cat(modal_parts, dim=1)  # 还是 [BK, 10, W, H]
        batch_dict['image'] = x.reshape(B, K, C, W, H)

        return super().forward(batch_dict)
