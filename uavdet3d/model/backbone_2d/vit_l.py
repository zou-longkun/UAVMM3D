import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_l_16
import math


class VitL(nn.Module):
    """
    Vision Transformer Large (ViT-L) based network for multi-level feature extraction.
    This network extracts features at multiple scales and returns 8x downsampled features.
    """

    def __init__(self, model_cfg):
        super(VitL, self).__init__()

        # Get input and output channels from config
        self.in_channels = model_cfg.INPUT_CHANNELS
        self.out_channels = model_cfg.OUT_CHANNELS

        # Initialize ViT-L backbone
        self.vit_backbone = vit_l_16()

        # Remove the classification head from ViT
        self.vit_backbone.heads = nn.Identity()

        # ViT-L parameters
        self.patch_size = 16  # ViT-L uses 16x16 patches
        self.embed_dim = 1024  # ViT-L embedding dimension
        self.num_heads = 16  # ViT-L number of attention heads
        self.num_layers = 24  # ViT-L number of transformer layers

        # Multi-level feature extraction layers
        # Extract features from different transformer layers for multi-scale representation
        self.feature_layers = [6, 12, 18, 24]  # Extract from these layers

        # Feature projection layers for different scales
        self.feature_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(len(self.feature_layers))
        ])

        # Feature fusion module
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256 * len(self.feature_layers), 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        # Input channel adaptation if needed
        if self.in_channels != 3:
            self.input_adapter = nn.Conv2d(self.in_channels, 3, kernel_size=1)
        else:
            self.input_adapter = nn.Identity()

        # Position embedding adaptation for different input sizes
        self.adaptive_pos_embed = True

        self.output_conv_neak = self._make_block(self.out_channels, self.out_channels)

    def _make_block(self, in_channels, feature_channels):

        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feature_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = feature_channels

        return nn.Sequential(*layers)

    def _reshape_patches_to_feature_map(self, patch_embeddings, H, W):
        """
        Reshape patch embeddings back to 2D feature map format.

        Args:
            patch_embeddings: (B, num_patches, embed_dim)
            H, W: Original height and width after patch embedding

        Returns:
            feature_map: (B, embed_dim, H, W)
        """
        B, num_patches, embed_dim = patch_embeddings.shape

        # Calculate grid size
        grid_size = int(math.sqrt(num_patches))

        # Reshape to feature map
        feature_map = patch_embeddings.view(B, grid_size, grid_size, embed_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)

        return feature_map

    def _extract_multi_level_features(self, x):
        """
        Extract features from multiple transformer layers for multi-scale representation.

        Args:
            x: Input tensor (B*K, 3, H, W)

        Returns:
            multi_level_features: List of feature maps at different scales
        """
        # Adapt input channels if necessary
        x = self.input_adapter(x)

        B, C, H, W = x.shape

        # Patch embedding
        x = self.vit_backbone.conv_proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        patch_H, patch_W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add class token
        class_token = self.vit_backbone.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)

        # Add position embeddings
        if self.adaptive_pos_embed and x.shape[1] != self.vit_backbone.encoder.pos_embedding.shape[1]:
            # Interpolate position embeddings for different input sizes
            pos_embed = F.interpolate(
                self.vit_backbone.encoder.pos_embedding.transpose(1, 2),
                size=x.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.vit_backbone.encoder.pos_embedding

        x = x + pos_embed
        x = self.vit_backbone.encoder.dropout(x)

        # Extract features from multiple layers
        multi_level_features = []

        for i, layer in enumerate(self.vit_backbone.encoder.layers):
            x = layer(x)

            # Extract features at specified layers
            if (i + 1) in self.feature_layers:
                # Remove class token and reshape to feature map
                patch_features = x[:, 1:]  # Remove class token
                feature_map = self._reshape_patches_to_feature_map(
                    patch_features, patch_H, patch_W
                )
                multi_level_features.append(feature_map)

        return multi_level_features

    def _get_8x_downsampled_features(self, multi_level_features, target_H, target_W):
        """
        Process multi-level features and generate 8x downsampled output features.

        Args:
            multi_level_features: List of feature maps from different layers
            target_H, target_W: Target output dimensions (8x downsampled)

        Returns:
            features_8x: 8x downsampled features
        """
        processed_features = []

        # Process each level of features
        for i, features in enumerate(multi_level_features):
            # Apply feature projector
            projected_features = self.feature_projectors[i](features)

            # Resize to target size (8x downsampling)
            if projected_features.shape[-2:] != (target_H, target_W):
                projected_features = F.interpolate(
                    projected_features,
                    size=(target_H, target_W),
                    mode='bilinear',
                    align_corners=False
                )

            processed_features.append(projected_features)

        # Concatenate multi-level features
        concatenated_features = torch.cat(processed_features, dim=1)

        # Fuse features
        fused_features = self.feature_fusion(concatenated_features)

        return fused_features

    def forward(self, batch_dict):
        """
        Forward pass of the network.

        Args:
            batch_dict: Dictionary containing input data
                - 'image': Input images (B, K, 3, W, H)

        Returns:
            batch_dict: Updated dictionary with extracted features
                - 'features_2d': 8x downsampled features (B, K, out_channels, W//8, H//8)
        """
        x = batch_dict['image']  # B, K, 3, W, H
        B, K, C, W, H = x.shape

        # Calculate target dimensions for 8x downsampling
        target_H, target_W = H // 8, W // 8

        # Reshape for batch processing
        x = x.reshape(B * K, C, W, H)

        # Extract multi-level features using ViT-L
        multi_level_features = self._extract_multi_level_features(x)

        # Generate 8x downsampled features
        features_8x = self._get_8x_downsampled_features(
            multi_level_features, target_H, target_W
        )

        features_8x = self.output_conv_neak(features_8x)

        # Reshape back to original batch structure
        BK, out_C, feat_H, feat_W = features_8x.shape
        features_8x = features_8x.reshape(B, K, out_C, feat_H, feat_W)

        # Store in batch dictionary
        batch_dict['features_2d'] = features_8x

        return batch_dict