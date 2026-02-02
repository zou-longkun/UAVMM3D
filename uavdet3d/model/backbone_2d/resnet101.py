import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet101(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet101, self).__init__()

        # Get parameters from config
        self.in_channels = model_cfg.INPUT_CHANNELS
        self.out_channels = model_cfg.OUT_CHANNELS
        self.feature_channels = model_cfg.NUM_FILTERS

        # Load pretrained ResNet50
        resnet = models.resnet101()

        # Separate ResNet50 layers for multi-scale feature extraction
        self.conv1 = resnet.conv1  # stride=2, output 2x downsampling
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride=2, output 4x downsampling

        self.layer1 = resnet.layer1  # no downsampling, output 4x downsampling
        self.layer2 = resnet.layer2  # stride=2, output 8x downsampling
        self.layer3 = resnet.layer3  # stride=2, output 16x downsampling
        self.layer4 = resnet.layer4  # stride=2, output 32x downsampling

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

        # Get feature channels for different levels
        # ResNet50 output channels: layer2=512, layer3=1024, layer4=2048
        self.feat_8x_channels = 512  # layer2 output
        self.feat_16x_channels = 1024  # layer3 output
        self.feat_32x_channels = 2048  # layer4 output

        # Feature adaptation layers to unify channel dimensions
        self.adapt_8x = nn.Sequential(
            nn.Conv2d(self.feat_8x_channels, self.feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True)
        )

        self.adapt_16x = nn.Sequential(
            nn.Conv2d(self.feat_16x_channels, self.feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True)
        )

        self.adapt_32x = nn.Sequential(
            nn.Conv2d(self.feat_32x_channels, self.feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True)
        )

        # Upsampling layers
        self.upsample_16x_to_8x = nn.Sequential(
            nn.ConvTranspose2d(self.feature_channels, self.feature_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True)
        )

        self.upsample_32x_to_8x = nn.Sequential(
            nn.ConvTranspose2d(self.feature_channels, self.feature_channels,
                               kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True)
        )

        # Fused feature channels = 3 * feature_channels (8x + 16x + 32x)
        fused_channels = 3 * self.feature_channels

        # Final output layer to convert fused features to target channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(fused_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.output_conv_neak = self._make_block(self.out_channels,self.out_channels)


    def _make_block(self, in_channels, feature_channels):

        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feature_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = feature_channels

        return nn.Sequential(*layers)

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

        # Stage 4: 8x -> 16x downsampling
        feat_16x = self.layer3(feat_8x)  # 16x downsampling, 1024 channels

        # Stage 5: 16x -> 32x downsampling
        feat_32x = self.layer4(feat_16x)  # 32x downsampling, 2048 channels

        # Feature adaptation: unify channel dimensions
        feat_8x_adapted = self.adapt_8x(feat_8x)  # (B*K, feature_channels, H/8, W/8)
        feat_16x_adapted = self.adapt_16x(feat_16x)  # (B*K, feature_channels, H/16, W/16)
        feat_32x_adapted = self.adapt_32x(feat_32x)  # (B*K, feature_channels, H/32, W/32)

        # Upsample to 8x resolution
        feat_16x_upsampled = self.upsample_16x_to_8x(feat_16x_adapted)  # (B*K, feature_channels, H/8, W/8)
        feat_32x_upsampled = self.upsample_32x_to_8x(feat_32x_adapted)  # (B*K, feature_channels, H/8, W/8)

        # Ensure consistent dimensions (handle potential size mismatches)
        target_h, target_w = feat_8x_adapted.shape[2], feat_8x_adapted.shape[3]

        if feat_16x_upsampled.shape[2:] != (target_h, target_w):
            feat_16x_upsampled = F.interpolate(feat_16x_upsampled, size=(target_h, target_w),
                                               mode='bilinear', align_corners=False)

        if feat_32x_upsampled.shape[2:] != (target_h, target_w):
            feat_32x_upsampled = F.interpolate(feat_32x_upsampled, size=(target_h, target_w),
                                               mode='bilinear', align_corners=False)

        # Feature fusion: concatenate along channel dimension
        fused_features = torch.cat([
            feat_8x_adapted,  # Original 8x features
            feat_16x_upsampled,  # 16x upsampled to 8x
            feat_32x_upsampled  # 32x upsampled to 8x
        ], dim=1)  # (B*K, 3*feature_channels, H/8, W/8)

        # Final output layer
        output_features = self.output_conv(fused_features)  # (B*K, out_channels, H/8, W/8)

        output_features = self.output_conv_neak(output_features)

        # Reshape back to original batch structure
        BK, out_C, out_H, out_W = output_features.shape
        output_features = output_features.reshape(B, K, out_C, out_H, out_W)

        # Store features in batch_dict
        batch_dict['features_2d'] = output_features

        return batch_dict

