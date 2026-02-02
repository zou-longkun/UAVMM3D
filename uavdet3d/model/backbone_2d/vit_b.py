import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
import math


class VitB(nn.Module):
    """
    Vision Transformer Base (ViT-B) based network for multi-level feature extraction.
    This network extracts features at multiple scales and returns 8x downsampled features.
    """

    def __init__(self, model_cfg):
        super(VitB, self).__init__()

        # Get input and output channels from config
        self.in_channels = model_cfg.INPUT_CHANNELS
        self.out_channels = model_cfg.OUT_CHANNELS

        # Initialize ViT-B backbone
        self.vit_backbone = vit_b_16()

        # Remove the classification head from ViT
        self.vit_backbone.heads = nn.Identity()

        # ViT-B parameters
        self.patch_size = 16  # ViT-B uses 16x16 patches
        self.embed_dim = 768  # ViT-B embedding dimension
        self.num_heads = 12  # ViT-B number of attention heads
        self.num_layers = 12  # ViT-B number of transformer layers

        # Multi-level feature extraction layers
        # Extract features from different transformer layers for multi-scale representation
        self.feature_layers = [3, 6, 9, 12]  # Extract from these layers (quarter intervals)

        # Feature projection layers for different scales
        # Each projector reduces the embedding dimension and adds spatial processing
        self.feature_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, 384, kernel_size=3, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True)
            ) for _ in range(len(self.feature_layers))
        ])

        # Feature fusion module to combine multi-level features
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(192 * len(self.feature_layers), 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        # Input channel adaptation if needed
        if self.in_channels != 3:
            self.input_adapter = nn.Conv2d(self.in_channels, 3, kernel_size=1)
        else:
            self.input_adapter = nn.Identity()

        # Enable adaptive position embedding for different input sizes
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
            patch_embeddings: (B, num_patches, embed_dim) - patch token embeddings
            H, W: Height and width of the patch grid

        Returns:
            feature_map: (B, embed_dim, H, W) - 2D feature map
        """
        B, num_patches, embed_dim = patch_embeddings.shape

        # Calculate grid size from number of patches
        grid_size = int(math.sqrt(num_patches))

        # Reshape from sequence to 2D grid format
        feature_map = patch_embeddings.view(B, grid_size, grid_size, embed_dim)
        # Permute to channel-first format (B, C, H, W)
        feature_map = feature_map.permute(0, 3, 1, 2)

        return feature_map

    def _extract_multi_level_features(self, x):
        """
        Extract features from multiple transformer layers for multi-scale representation.

        Args:
            x: Input tensor (B*K, C, H, W)

        Returns:
            multi_level_features: List of feature maps at different transformer depths
        """
        # Adapt input channels if necessary
        x = self.input_adapter(x)

        B, C, H, W = x.shape

        # Convert input to patches using convolutional projection
        x = self.vit_backbone.conv_proj(x)  # (B, embed_dim, H_patch, W_patch)
        patch_H, patch_W = x.shape[-2:]

        # Flatten patches and transpose for transformer input
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add class token at the beginning of sequence
        class_token = self.vit_backbone.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)  # (B, num_patches + 1, embed_dim)

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

        # Apply position embeddings and dropout
        x = x + pos_embed
        x = self.vit_backbone.encoder.dropout(x)

        # Extract features from multiple transformer layers
        multi_level_features = []

        for i, layer in enumerate(self.vit_backbone.encoder.layers):
            # Apply transformer layer
            x = layer(x)

            # Extract features at specified layers
            if (i + 1) in self.feature_layers:
                # Remove class token and keep only patch tokens
                patch_features = x[:, 1:]  # (B, num_patches, embed_dim)

                # Reshape patch tokens back to 2D feature map
                feature_map = self._reshape_patches_to_feature_map(
                    patch_features, patch_H, patch_W
                )
                multi_level_features.append(feature_map)

        return multi_level_features

    def _get_8x_downsampled_features(self, multi_level_features, target_H, target_W):
        """
        Process multi-level features and generate 8x downsampled output features.

        Args:
            multi_level_features: List of feature maps from different transformer layers
            target_H, target_W: Target output dimensions for 8x downsampling

        Returns:
            features_8x: Final 8x downsampled features (B*K, out_channels, target_H, target_W)
        """
        processed_features = []

        # Process each level of features through dedicated projectors
        for i, features in enumerate(multi_level_features):
            # Apply feature projector to reduce dimensions and add spatial processing
            projected_features = self.feature_projectors[i](features)

            # Resize to target size for 8x downsampling if needed
            if projected_features.shape[-2:] != (target_H, target_W):
                projected_features = F.interpolate(
                    projected_features,
                    size=(target_H, target_W),
                    mode='bilinear',
                    align_corners=False
                )

            processed_features.append(projected_features)

        # Concatenate multi-level features along channel dimension
        concatenated_features = torch.cat(processed_features, dim=1)

        # Apply feature fusion to combine information from different levels
        fused_features = self.feature_fusion(concatenated_features)

        return fused_features

    def forward(self, batch_dict):
        """
        Forward pass of the ViT-B network.

        Args:
            batch_dict: Dictionary containing input data
                - 'image': Input images (B, K, C, W, H)

        Returns:
            batch_dict: Updated dictionary with extracted features
                - 'features_2d': 8x downsampled features (B, K, out_channels, W//8, H//8)
        """
        x = batch_dict['image']  # (B, K, C, W, H)
        B, K, C, W, H = x.shape

        # Calculate target dimensions for 8x downsampling
        target_H, target_W = H // 8, W // 8

        # Reshape for batch processing: combine batch and sequence dimensions
        x = x.reshape(B * K, C, W, H)  # (B*K, C, W, H)

        # Extract multi-level features using ViT-B backbone
        multi_level_features = self._extract_multi_level_features(x)

        # Generate final 8x downsampled features
        features_8x = self._get_8x_downsampled_features(
            multi_level_features, target_H, target_W
        )

        features_8x = self.output_conv_neak(features_8x)

        # Reshape back to original batch structure
        BK, out_C, feat_H, feat_W = features_8x.shape
        features_8x = features_8x.reshape(B, K, out_C, feat_H, feat_W)

        # Store extracted features in batch dictionary
        batch_dict['features_2d'] = features_8x

        return batch_dict