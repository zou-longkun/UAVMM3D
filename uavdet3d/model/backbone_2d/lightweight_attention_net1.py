import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightAttentionNet(nn.Module):
    def __init__(self, model_cfg):
        super(LightweightAttentionNet, self).__init__()

        self.in_channels = model_cfg.INPUT_CHANNELS  # 3
        self.out_channels = model_cfg.OUT_CHANNELS
        # 轻量化特征通道，平衡性能与显存
        self.feature_channels = [max(c // 16, 8) for c in model_cfg.NUM_FILTERS]  # [8, 16, 24]
        self.num_modalities = model_cfg.NUM_MODALITIES  # 3
        self.fuse_method = model_cfg.get('FUSE_METHOD', 'modality_attention')

        self.down_sample_ratio = model_cfg.get('STRIDE', 8)  # 与标签对齐

        if hasattr(model_cfg, 'REDUCED_OUT_CHANNELS'):
            self.out_channels = model_cfg.REDUCED_OUT_CHANNELS
            
        # LiDAR投影深度图专用处理（核心：突出深度数值特征）
        self.lidar_proj_dim = 10  # 轻量化深度特征维度
        # 提取LiDAR投影的深度梯度和距离特征（不做复杂变换，保留投影特性）
        self.lidar_proj_processor = nn.Sequential(
            nn.Conv2d(1, self.lidar_proj_dim, kernel_size=3, padding=1),  # 保留空间连续性
            nn.BatchNorm2d(self.lidar_proj_dim),
            nn.LeakyReLU(0.05, inplace=True)  # 弱非线性，减少深度数值破坏
        )

        # 模态输入通道（明确区分LiDAR投影与其他模态）
        self.modal_in_channels = [
            3,  # RGB+IR（纹理主导）
            1 + self.lidar_proj_dim,  # LiDAR投影（原始深度+梯度特征）
            1   # Radar（强度特征）
        ]

        # 分支架构：为LiDAR投影定制轻量化编码器，避免被纹理特征压制
        self.modal_encoders = nn.ModuleList()
        for i in range(self.num_modalities):
            if i == 1:  # LiDAR投影专用编码器（保留深度数值特性）
                encoder = self._create_lidar_encoder(self.modal_in_channels[i])
            else:  # RGB/IR和Radar（纹理/强度特征，轻量化压缩）
                encoder = self._create_encoder(self.modal_in_channels[i])
            self.modal_encoders.append(encoder)
        
        # 融合后输出卷积（保持轻量）
        self.final_conv = nn.Conv2d(self.feature_channels[-1], self.out_channels, kernel_size=1)

        # 备用拼接融合路径（同样轻量化）
        if self.fuse_method == 'channel_concat':
            self.concat_in_channels = sum(self.modal_in_channels)
            self.concat_encoder = self._create_encoder(self.concat_in_channels)
        else:
            self.concat_encoder = None

        self._init_fusion_layers()

    def _create_encoder(self, in_channels):
        """RGB/IR/Radar编码器：侧重纹理压缩，轻量化设计"""
        layers = nn.ModuleDict()
        # 快速压缩通道，减少计算量
        layers['init_block'] = nn.Conv2d(in_channels, self.feature_channels[0], kernel_size=3, padding=1, stride=1)
        # 深度可分离卷积减少参数，适合纹理特征
        layers['block1'] = self._make_texture_block(self.feature_channels[0], self.feature_channels[0])
        layers['d1'] = self.down_block(self.feature_channels[0], self.feature_channels[1])  # 1/2
        layers['block2'] = self._make_texture_block(self.feature_channels[1], self.feature_channels[1])
        layers['d2'] = self.down_block(self.feature_channels[1], self.feature_channels[2])  # 1/4
        layers['block3'] = self._make_texture_block(self.feature_channels[2], self.feature_channels[2])
        layers['d3'] = self.down_block(self.feature_channels[2], self.feature_channels[2])  # 1/8
        return layers

    def _create_lidar_encoder(self, in_channels):
        """LiDAR投影编码器：保留深度数值特征，弱化纹理压缩"""
        layers = nn.ModuleDict()
        # 初始层不压缩通道，保留LiDAR投影的深度分布
        layers['init_block'] = nn.Conv2d(in_channels, self.feature_channels[0], kernel_size=3, padding=1, stride=1)
        # 不用深度可分离卷积，避免破坏深度的空间连续性
        layers['block1'] = self._make_lidar_block(self.feature_channels[0], self.feature_channels[0])
        layers['d1'] = self.down_block(self.feature_channels[0], self.feature_channels[1])  # 1/2
        layers['block2'] = self._make_lidar_block(self.feature_channels[1], self.feature_channels[1])
        layers['d2'] = self.down_block(self.feature_channels[1], self.feature_channels[2])  # 1/4
        layers['block3'] = self._make_lidar_block(self.feature_channels[2], self.feature_channels[2])
        layers['d3'] = self.down_block(self.feature_channels[2], self.feature_channels[2])  # 1/8
        return layers

    def _make_texture_block(self, in_channels, feature_channels):
        """纹理特征块：深度可分离卷积+强非线性，适合RGB/IR"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),  # 强非线性提取纹理
        )

    def _make_lidar_block(self, in_channels, feature_channels):
        """LiDAR投影特征块：普通卷积+弱非线性，保留深度数值"""
        return nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),  # 普通卷积保留空间关联
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.05, inplace=True),  # 弱非线性，减少深度数值扭曲
        )

    def down_block(self, in_channels, feature_channels):
        """通用下采样块：统一 stride 确保对齐"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def _init_fusion_layers(self):
        """融合层：强化LiDAR投影特征的权重"""
        # 模态注意力：用简单卷积计算权重，侧重LiDAR
        self.modality_attention = nn.Conv2d(self.feature_channels[-1] * self.num_modalities, self.num_modalities, kernel_size=1)
        # LiDAR投影模态的静态权重偏向（非参数化，不增加显存）
        self.lidar_prior = 1.8  # 让LiDAR投影特征在融合时更受重视

    def _fuse_features(self, stacked_features):
        """融合策略：动态注意力+LiDAR偏向，确保投影深度被有效利用"""
        B, K, C, H, W = stacked_features.shape  # [B, 3, C, H, W]
        
        # 1. 特征展平用于注意力计算
        flattened = stacked_features.view(B, K*C, H, W)  # [B, 3*C, H, W]
        
        # 2. 动态模态权重（基于特征内容）
        dyn_weights = self.modality_attention(flattened)  # [B, 3, H, W]
        dyn_weights = dyn_weights.unsqueeze(2)  # [B, 3, 1, H, W]
        
        # 3. 强化LiDAR投影的权重（索引1为LiDAR模态）
        dyn_weights[:, 1] = dyn_weights[:, 1] * self.lidar_prior  # 静态偏向LiDAR投影
        dyn_weights = F.softmax(dyn_weights, dim=1)  # 归一化
        
        # 4. 加权融合（LiDAR投影特征被优先考虑）
        fused = torch.sum(stacked_features * dyn_weights, dim=1)  # [B, C, H, W]
        
        return self.final_conv(fused)

    def forward(self, batch_dict):
        image = batch_dict['image']
        B, C, H_in, W_in = image.shape
        device = image.device
        
        # 分离模态：明确区分LiDAR投影深度图
        rgb_ir_data = image[:, :3]  # RGB+IR（纹理）
        lidar_proj_raw = image[:, 3:4]  # LiDAR原始投影深度图
        radar_proj_data = image[:, 6:7]  # Radar（强度）
        
        # 处理LiDAR投影：提取深度梯度特征（保留投影特性）
        lidar_proj_features = self.lidar_proj_processor(lidar_proj_raw)
        lidar_proj_data = torch.cat([lidar_proj_raw, lidar_proj_features], dim=1)  # 原始投影+梯度特征
        
        modalities = [rgb_ir_data, lidar_proj_data, radar_proj_data]
        H_out = H_in // self.down_sample_ratio
        W_out = W_in // self.down_sample_ratio

        if self.fuse_method == 'channel_concat':
            # 备用拼接路径（轻量化）
            concatenated = torch.cat(modalities, dim=1)
            x = self.concat_encoder['init_block'](concatenated)
            x = self.concat_encoder['block1'](x) + x
            x = self.concat_encoder['d1'](x)
            x = self.concat_encoder['block2'](x) + x
            x = self.concat_encoder['d2'](x)
            x = self.concat_encoder['block3'](x) + x
            x = self.concat_encoder['d3'](x)
            x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
            fused_features = self.final_conv(x)
            
        else:
            # 分支架构+LiDAR强化融合（核心路径）
            stacked_features = torch.empty(
                B, self.num_modalities, self.feature_channels[-1], H_out, W_out, 
                device=device
            )
            
            for k in range(self.num_modalities):
                # 各模态独立提取特征（LiDAR投影用专用编码器）
                x = self.modal_encoders[k]['init_block'](modalities[k])
                x = self.modal_encoders[k]['block1'](x)
                x = self.modal_encoders[k]['d1'](x)  # 1/2
                x = self.modal_encoders[k]['block2'](x)
                x = self.modal_encoders[k]['d2'](x)  # 1/4
                x = self.modal_encoders[k]['block3'](x)
                x = self.modal_encoders[k]['d3'](x)  # 1/8
                
                # 对齐输出尺寸
                x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
                stacked_features[:, k] = x
            
            # 融合时强化LiDAR投影特征
            fused_features = self._fuse_features(stacked_features)

        batch_dict['down_sample_ratio'] = self.down_sample_ratio
        batch_dict['features_2d'] = fused_features
        return batch_dict