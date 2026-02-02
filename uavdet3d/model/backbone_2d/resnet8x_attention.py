import torch.nn as nn
import torch
import torch.nn.functional as F


class ResNet8xAttention(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet8xAttention, self).__init__()

        self.in_channels = model_cfg.INPUT_CHANNELS  # Each modality input has 3 channels
        self.out_channels = model_cfg.OUT_CHANNELS
        self.feature_channels = model_cfg.NUM_FILTERS
        self.num_modalities = model_cfg.NUM_MODALITIES
        self.fuse_method = model_cfg.FUSE_METHOD

        # Indices of modalities that need to be converted to single channel
        self.single_channel_indices = getattr(model_cfg, 'SINGLE_CHANNEL_INDICES', set())

        # Calculate total input channels after channel merging (only for channel concatenation fusion)
        if self.fuse_method == 'channel_concat':
            multi_channel_count = self.num_modalities - len(self.single_channel_indices)
            self.concat_in_channels = multi_channel_count * 3 + len(self.single_channel_indices) * 1
            self.encoder = self._create_encoder(self.concat_in_channels)
            self.final_conv = nn.Conv2d(self.feature_channels[3], self.out_channels, kernel_size=3, padding=1)
        else:
            # Create independent encoders for each modality
            self.modal_encoders = nn.ModuleList([
                self._create_encoder(self.in_channels[i])
                for i in range(self.num_modalities)
            ])
            self.final_convs = nn.ModuleList([
                nn.Conv2d(self.feature_channels[3], self.out_channels, kernel_size=3, padding=1)
                for _ in range(self.num_modalities)
            ])

        self._init_fusion_layers()

    def _create_encoder(self, in_channels):
        """Create base encoder, including initial block, 4 residual blocks and 3 downsampling blocks"""
        layers = nn.ModuleDict()
        layers['init_block'] = nn.Conv2d(in_channels, self.feature_channels[0], kernel_size=1)
        layers['block1'] = self._make_block(self.feature_channels[0], self.feature_channels[0])
        layers['d1'] = self.down_block(self.feature_channels[0], self.feature_channels[1])
        layers['block2'] = self._make_block(self.feature_channels[1], self.feature_channels[1])
        layers['d2'] = self.down_block(self.feature_channels[1], self.feature_channels[2])
        layers['block3'] = self._make_block(self.feature_channels[2], self.feature_channels[2])
        layers['d3'] = self.down_block(self.feature_channels[2], self.feature_channels[3])
        layers['block4'] = self._make_block(self.feature_channels[3], self.feature_channels[3])
        return layers

    def _make_block(self, in_channels, feature_channels):
        """Create residual block containing 4 convolutional layers (with BN and ReLU)"""
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feature_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = feature_channels
        return nn.Sequential(*layers)

    def down_block(self, in_channels, feature_channels):
        """Create downsampling block containing max pooling and convolutional layer (with BN and ReLU)"""
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def _init_fusion_layers(self):
        """Initialize layers required for different fusion methods
        Supports: weighted_sum/attention/gating/concat four fusion methods
        """
        if self.fuse_method == 'weighted_sum':
            self.modal_weights = nn.Parameter(torch.ones(self.num_modalities))
        elif self.fuse_method == 'attention':
            self.attention_compress = nn.Conv2d(self.out_channels, 1, kernel_size=1)
            self.attention_conv = nn.Sequential(
                nn.Conv2d(self.num_modalities, self.num_modalities, kernel_size=3, padding=1),
                nn.Softmax(dim=1)
            )
        elif self.fuse_method == 'gating':
            self.gate_conv = nn.Sequential(
                nn.Conv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.Sigmoid()
            )
        elif self.fuse_method == 'concat':
            self.concat_compress = nn.Conv2d(
                self.num_modalities * self.out_channels, self.out_channels, kernel_size=1
            )

    def _fuse_features(self, stacked_features):
        """Fuse multi-modal features according to the specified fusion method
        Args:
            stacked_features: Stacked multi-modal features with shape (B, K, C, W, H)
        Returns:
            Fused features with shape (B, C, W, H)
        """
        B, K, C, W, H = stacked_features.shape
        if self.fuse_method == 'concat':
            concat_features = stacked_features.permute(0, 2, 1, 3, 4).reshape(B, C * K, W, H)
            return self.concat_compress(concat_features)
        elif self.fuse_method == 'weighted_sum':
            weights = F.softmax(self.modal_weights, dim=0).view(1, K, 1, 1, 1)
            return torch.sum(stacked_features * weights, dim=1)
        elif self.fuse_method == 'attention':
            attention_input = stacked_features.reshape(B * K, C, W, H)
            compressed = self.attention_compress(attention_input)
            attention_map = compressed.reshape(B, K, 1, W, H).squeeze(2)
            attention_weights = self.attention_conv(attention_map)
            return torch.sum(stacked_features * attention_weights.unsqueeze(2), dim=1)
        elif self.fuse_method == 'gating':
            gate_input = torch.mean(stacked_features, dim=2, keepdim=True)
            gate_coeffs = self.gate_conv(gate_input)
            return torch.sum(stacked_features * gate_coeffs, dim=1)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fuse_method}")

    def forward(self, batch_dict):
        """Forward pass through the ResNet8xAttention backbone.
        Args:
            batch_dict: Dictionary containing multi-modal input images, with 'image' as key and shape (B, K, 3, W_in, H_in)
        Returns:
            batch_dict: Dictionary containing fused features and downsampling rate
        """
        x = batch_dict['image']
        
        # Handle both 4D (B, C, W, H) and 5D (B, K, C, W, H) input
        if len(x.shape) == 4:
            # If 4D input, add modality dimension to make it 5D
            B, C, W_in, H_in = x.shape
            K = 1
            x = x.unsqueeze(1)  # Shape becomes (B, 1, C, W, H)
        else:
            # Original 5D input handling
            B, K, _, W_in, H_in = x.shape

        if self.fuse_method == 'channel_concat':
            # Process each modality's channels: convert specified modalities to single channel, keep 3 channels for others
            processed_modalities = []
            for k in range(K):
                modal = x[:, k, :, :, :]  # Extract k-th modality (B, 3, W, H)

                if k in self.single_channel_indices:
                    # Convert 2nd and 4th modalities to single channel (take first channel)
                    modal = modal[:, 0:1, :, :]  # Shape becomes (B, 1, W, H)

                processed_modalities.append(modal)

            # Concatenate channels of all modalities
            concatenated = torch.cat(processed_modalities, dim=1)

            # Encoder feature extraction
            x_init = self.encoder['init_block'](concatenated)
            x1 = self.encoder['block1'](x_init) + x_init
            x1 = self.encoder['d1'](x1)
            x2 = self.encoder['block2'](x1) + x1
            x2 = self.encoder['d2'](x2)
            x3 = self.encoder['block3'](x2) + x2
            x3 = self.encoder['d3'](x3)
            x4 = self.encoder['block4'](x3) + x3
            fused_features = self.final_conv(x4)
            stacked_features = None
        else:
            # Other fusion methods: fuse after independent encoding of each modality
            modal_features = []
            for k in range(K):
                modal = x[:, k, :, :, :]
                encoder = self.modal_encoders[k]
                x_init = encoder['init_block'](modal)
                x1 = encoder['block1'](x_init) + x_init
                x1 = encoder['d1'](x1)
                x2 = encoder['block2'](x1) + x1
                x2 = encoder['d2'](x2)
                x3 = encoder['block3'](x2) + x2
                x3 = encoder['d3'](x3)
                x4 = encoder['block4'](x3) + x3
                x_final = self.final_convs[k](x4)
                modal_features.append(x_final)
            stacked_features = torch.stack(modal_features, dim=1)
            fused_features = self._fuse_features(stacked_features)

        # Calculate downsampling rate
        W_out, H_out = (stacked_features.shape[-2:]) if stacked_features is not None else fused_features.shape[-2:]
        down_sample_ratio = (W_in // W_out + H_in // H_out) // 2
        batch_dict['down_sample_ratio'] = down_sample_ratio

        if stacked_features is not None:
            batch_dict['stacked_features'] = stacked_features
        batch_dict['features_2d'] = fused_features

        return batch_dict
