import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base

class ConvNeXtBase(nn.Module):
    def __init__(self, model_cfg):
        super(ConvNeXtBase, self).__init__()
        self.in_channels = model_cfg.INPUT_CHANNELS
        self.out_channels = model_cfg.OUT_CHANNELS
        self.feature_channels = model_cfg.NUM_FILTERS

        backbone = convnext_base(weights=None)
        self.stem = backbone.features[0]
        self.stage1 = backbone.features[1]
        self.stage2 = backbone.features[2]
        self.stage3 = backbone.features[3]
        self.stage4 = backbone.features[4]

        self.lateral8 = nn.Conv2d(256, self.feature_channels, 1)
        self.lateral16 = nn.Conv2d(256, self.feature_channels, 1)
        self.lateral32 = nn.Conv2d(512, self.feature_channels, 1)

        # Final output layer to convert fused features to target channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(fused_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.output_conv_neak = self._make_block(self.out_channels, self.out_channels)

    def _make_block(self, in_channels, feature_channels):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feature_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = feature_channels

        return nn.Sequential(*layers)

    def forward(self, batch_dict):
        x = batch_dict['image']  # B, K, 3, H, W
        B, K, C, H, W = x.shape
        x = x.reshape(B * K, C, H, W)

        # Forward through ConvNeXt backbone
        x = self.stem(x)
        x = self.stage1(x)  # stride 4
        feat8 = self.stage2(x)  # stride 8
        feat16 = self.stage3(feat8)  # stride 16
        feat32 = self.stage4(feat16)  # stride 32

        feat8 = self.lateral8(feat8)
        feat16 = self.lateral16(feat16)
        feat32 = self.lateral32(feat32)

        feat16_up = F.interpolate(feat16, size=feat8.shape[-2:], mode='bilinear', align_corners=False)
        feat32_up = F.interpolate(feat32, size=feat8.shape[-2:], mode='bilinear', align_corners=False)

        fused = torch.cat([feat8, feat16_up, feat32_up], dim=1)

        fused = self.output_conv(fused)

        fused = self.output_conv_neak(fused)

        fused = fused.reshape(B, K, -1, fused.shape[-2], fused.shape[-1])
        batch_dict['features_2d'] = fused
        return batch_dict
