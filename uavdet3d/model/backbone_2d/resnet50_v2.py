import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet50V2(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet50V2, self).__init__()

        # Get parameters from config
        self.in_channels = model_cfg.INPUT_CHANNELS
        self.out_channels = model_cfg.OUT_CHANNELS
        self.feature_channels = model_cfg.NUM_FILTERS

        # Load pretrained ResNet50
        resnet = models.resnet50()

        # Separate ResNet50 layers for multi-scale feature extraction
        self.conv1 = resnet.conv1  # stride=2, output 2x downsampling
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride=2, output 4x downsampling

        self.layer1 = resnet.layer1  # no downsampling, output 4x downsampling
        self.layer2 = resnet.layer2  # stride=2, output 8x downsampling

        # Modify first layer if input channels != 3
        if self.in_channels != 3:
            original_conv1 = self.conv1
            new_conv1 = nn.Conv2d(
                self.in_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )

            with torch.no_grad():
                if self.in_channels == 1:
                    # Single channel: average RGB weights
                    new_conv1.weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
                elif self.in_channels < 3:
                    # Less than 3 channels: slice corresponding channels
                    new_conv1.weight = nn.Parameter(
                        original_conv1.weight[:, :self.in_channels, :, :]
                    )
                else:
                    # More than 3 channels: extend channels
                    new_weight = torch.zeros(
                        original_conv1.out_channels,
                        self.in_channels,
                        *original_conv1.kernel_size
                    )
                    new_weight[:, :3, :, :] = original_conv1.weight
                    # Initialize extra channels with Xavier uniform
                    nn.init.xavier_uniform_(new_weight[:, 3:, :, :])
                    new_conv1.weight = nn.Parameter(new_weight)

                # Copy bias
                if original_conv1.bias is not None:
                    new_conv1.bias = nn.Parameter(original_conv1.bias.clone())

            self.conv1 = new_conv1



        # Final output layer to convert fused features to target channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=True)
        )

    def forward(self, batch_dict):
        x = batch_dict['image']  # B, K, C, W, H
        B, K, C, W, H = x.shape

        # Reshape to (B*K, C, W, H) for batch processing
        x = x.reshape(B * K, C, W, H)

        # Forward pass to extract multi-level features
        # Stage 1: Input -> 4x downsampling
        x = self.conv1(x)  # 2x downsampling
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 4x downsampling

        # Stage 2: 4x -> 4x (layer1 no downsampling)
        x = self.layer1(x)  # 4x downsampling

        # Stage 3: 4x -> 8x downsampling
        feat_8x = self.layer2(x)  # 8x downsampling, 512 channels

        # Final output layer
        output_features = self.output_conv(feat_8x)  # (B*K, out_channels, H/8, W/8)

        # Reshape back to original batch structure
        BK, out_C, out_H, out_W = output_features.shape
        output_features = output_features.reshape(B, K, out_C, out_H, out_W)

        # Store features in batch_dict
        batch_dict['features_2d'] = output_features

        return batch_dict

