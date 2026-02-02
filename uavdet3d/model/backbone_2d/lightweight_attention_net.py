import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightAttentionNet(nn.Module):
    def __init__(self, model_cfg):
        super(LightweightAttentionNet, self).__init__()

        self.in_channels = model_cfg.BACKBONE_2D.INPUT_CHANNELS  # 3
        self.out_channels = model_cfg.BACKBONE_2D.OUT_CHANNELS
        # 降低特征通道数以减少显存占用
        self.feature_channels = [max(c // 8, 8) for c in model_cfg.BACKBONE_2D.NUM_FILTERS]  # [8, 16, 32]
        self.num_modalities = model_cfg.BACKBONE_2D.NUM_MODALITIES  # 3
        self.fuse_method = model_cfg.BACKBONE_2D.get('FUSE_METHOD', 'weighted_sum')
        self.use_original_fusion = model_cfg.FEATURE_FUSION.USE_ORIGINAL_FUSION

        # 关键：根据标签尺寸反推下采样率必须与配置中的STRIDE保持一致
        # 从配置中获取stride参数，确保与标签生成过程保持同步
        self.down_sample_ratio = model_cfg.BACKBONE_2D.get('STRIDE', 8)  # 与标签对齐

        if hasattr(model_cfg.BACKBONE_2D, 'REDUCED_OUT_CHANNELS'):
            self.out_channels = model_cfg.BACKBONE_2D.REDUCED_OUT_CHANNELS
            
        # 简化的LiDAR位置信息增强模块
        self.enhance_lidar_position = True
        self.lidar_position_dim = 32  # 降低位置增强特征维度
        self.lidar_linear_activation = True
        
        # 简化的LiDAR位置信息增强模块
        if self.enhance_lidar_position:
            # 使用单个1x1卷积简化位置信息提取
            self.lidar_position_encoder = nn.Sequential(
                nn.Conv2d(1, self.lidar_position_dim, kernel_size=1),
                nn.BatchNorm2d(self.lidar_position_dim),
                nn.ReLU(inplace=False)
            )
            
            # 简化的空间特征提取（只保留单个3x3卷积）
            self.spatial_extractor = nn.Sequential(
                nn.Conv2d(1, self.lidar_position_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.lidar_position_dim),
                nn.ReLU(inplace=False)
            )
            
            # 简化的特征融合层
            self.position_fusion = nn.Sequential(
                nn.Conv2d(self.lidar_position_dim * 2, self.lidar_position_dim, kernel_size=1),
                nn.BatchNorm2d(self.lidar_position_dim),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.lidar_position_dim, 1, kernel_size=1)  # 输出通道数修改为1，与LiDAR输入通道一致
            )

        # 各模态输入通道数
        self.rgb_ir_channels = 6 if self.use_original_fusion else 3  # 根据配置动态选择RGB+IR(6通道)或RGB(3通道)
        self.lidar_channels = 1  # LiDAR投影数据（1通道）
        self.radar_channels = 1  # Radar投影数据（1通道）

        # 初始化编码器（确保所有分支下采样率一致）
        self.modal_encoders = nn.ModuleList()
        # 模态0：RGB/IR（3通道）
        self.modal_encoders.append(self._create_encoder(self.rgb_ir_channels))
        # 模态1：LiDAR（1通道）
        self.modal_encoders.append(self._create_lidar_encoder(self.lidar_channels))
        # 模态2：Radar（1通道）
        self.modal_encoders.append(self._create_encoder(self.radar_channels))
        
        self.final_conv = nn.Conv2d(self.feature_channels[-1], self.out_channels, kernel_size=1)

        if self.fuse_method == 'channel_concat':
            # 计算串联后的总通道数
            # RGB/IR(3通道) + LiDAR(1通道) + Radar(1通道) = 5通道
            self.concat_in_channels = self.rgb_ir_channels + self.lidar_channels + self.radar_channels
            self.concat_encoder = self._create_encoder(self.concat_in_channels)
        else:
            self.concat_encoder = None

        self._init_fusion_layers()

    def _create_encoder(self, in_channels):
        """通用编码器（3次下采样，总下采样率8）"""
        layers = nn.ModuleDict()
        layers['init_block'] = nn.Conv2d(in_channels, self.feature_channels[0], kernel_size=1)
        layers['block1'] = self._make_block(self.feature_channels[0], self.feature_channels[0])
        layers['d1'] = self.down_block(self.feature_channels[0], self.feature_channels[1])  # 1/2
        layers['block2'] = self._make_block(self.feature_channels[1], self.feature_channels[1])
        layers['d2'] = self.down_block(self.feature_channels[1], self.feature_channels[2])  # 1/4
        layers['block3'] = self._make_block(self.feature_channels[2], self.feature_channels[2])
        layers['d3'] = self.down_block(self.feature_channels[2], self.feature_channels[2])  # 1/8（第3次下采样）
        return layers

    def _create_lidar_encoder(self, in_channels):
        """LiDAR编码器（同样3次下采样，确保尺寸匹配）"""
        layers = nn.ModuleDict()
        layers['init_block'] = nn.Conv2d(in_channels, self.feature_channels[0], kernel_size=1)
        layers['block1'] = self._make_lidar_block(self.feature_channels[0], self.feature_channels[0])
        layers['d1'] = self.down_block(self.feature_channels[0], self.feature_channels[1])  # 1/2
        layers['block2'] = self._make_lidar_block(self.feature_channels[1], self.feature_channels[1])
        layers['d2'] = self.down_block(self.feature_channels[1], self.feature_channels[2])  # 1/4
        layers['block3'] = self._make_lidar_block(self.feature_channels[2], self.feature_channels[2])
        layers['d3'] = self.down_block(self.feature_channels[2], self.feature_channels[2])  # 1/8
        return layers

    def _make_block(self, in_channels, feature_channels):
        # 使用深度可分离卷积减少参数量和显存占用
        return nn.Sequential(
            # 深度可分离卷积替代常规卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            # 简化网络结构，只保留一层卷积
        )

    def _make_lidar_block(self, in_channels, feature_channels):
        # 简化的LiDAR处理块
        return nn.Sequential(
            # 使用深度可分离卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            # 线性激活保留位置信息
            nn.Identity() if self.lidar_linear_activation else nn.ReLU(inplace=True),
        )

    def down_block(self, in_channels, feature_channels):
        # 使用深度可分离卷积降低下采样块的显存占用
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def _init_fusion_layers(self):
        if self.fuse_method == 'weighted_sum':
            self.modal_weights = nn.Parameter(torch.tensor([1.0, 1.5, 1.0]))
        else:
            self.fuse_method = 'weighted_sum'
            self.modal_weights = nn.Parameter(torch.tensor([1.0, 1.5, 1.0]))

    def _fuse_features(self, stacked_features):
        B, K, C, W, H = stacked_features.shape
        weights = F.softmax(self.modal_weights, dim=0).view(1, K, 1, 1, 1)
        fused = torch.sum(stacked_features * weights, dim=1)
        return self.final_conv(fused)

    def forward(self, batch_dict):
        image = batch_dict['image']
        B, C, H_in, W_in = image.shape
        device = image.device
        
        # 分离模态数据
        if C == 9:  # 两分支结构：RGB(3)/IR(3) + LiDAR(3) + Radar(3) 或其他两分支配置
            # 对于9通道输入，我们取前3通道作为RGB数据
            rgb_ir_data = image[:, :3]  # 前3通道为RGB
            # LiDAR和Radar的投影图像实际上是单层通道repeat到3层的，实际上就是3层的
            # 在网络中我们只取第一层（第一个通道）
            lidar_proj_data = image[:, 3:4]  # 只取LiDAR投影数据的第一个通道
            radar_proj_data = image[:, 6:7]  # 只取雷达投影数据的第一个通道
        elif C == 12:  # 融合4种模态：RGB(3) + IR(3) + LiDAR(3) + Radar(3)
            # 对于12通道输入，表示融合4种模态，每种模态3通道
            # 正确的通道分布：RGB(0-2) + IR(3-5) + LiDAR(6-8) + Radar(9-11)
            rgb_ir_data = image[:, :6]  # 前3通道为RGB (0-5)
            lidar_proj_data = image[:, 6:7]  # 从LiDAR的3通道中取第1通道 (6)
            radar_proj_data = image[:, 9:10]  # 从Radar的3通道中取第1通道 (9)
        else:
            raise ValueError(f"不支持的输入通道数: {C}，请使用9或12通道输入")
        
        # 增强LiDAR投影特征的位置信息
        if self.enhance_lidar_position:
            # 提取通道特征和空间特征
            channel_features = self.lidar_position_encoder(lidar_proj_data)
            spatial_features = self.spatial_extractor(lidar_proj_data)
            
            # 融合特征
            fused_features = torch.cat([channel_features, spatial_features], dim=1)
            enhanced_position_features = self.position_fusion(fused_features)
            
            # 结合增强特征与原始数据
            lidar_proj_data = lidar_proj_data * 0.8 + enhanced_position_features * 0.2
        
        modalities = [rgb_ir_data, lidar_proj_data, radar_proj_data]

        # 计算与标签匹配的输出尺寸（下采样率8）
        H_out = H_in // self.down_sample_ratio
        W_out = W_in // self.down_sample_ratio

        if self.fuse_method == 'channel_concat':
            concatenated = torch.cat(modalities, dim=1)
            x = self.concat_encoder['init_block'](concatenated)
            x = self.concat_encoder['block1'](x) + x
            x = self.concat_encoder['d1'](x)
            x = self.concat_encoder['block2'](x) + x
            x = self.concat_encoder['d2'](x)
            x = self.concat_encoder['block3'](x) + x
            x = self.concat_encoder['d3'](x)  # 第3次下采样到1/8
            # 强制对齐到标签尺寸
            x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
            fused_features = self.final_conv(x)
            
        else:
            stacked_features = torch.empty(
                B, self.num_modalities, self.feature_channels[-1], H_out, W_out, 
                device=device
            )
            
            for k in range(self.num_modalities):
                x = self.modal_encoders[k]['init_block'](modalities[k])
                x = self.modal_encoders[k]['block1'](x) + x
                x = self.modal_encoders[k]['d1'](x)  # 1/2
                x = self.modal_encoders[k]['block2'](x) + x
                x = self.modal_encoders[k]['d2'](x)  # 1/4
                x = self.modal_encoders[k]['block3'](x) + x
                x = self.modal_encoders[k]['d3'](x)  # 1/8
                # 强制对齐尺寸
                x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
                stacked_features[:, k] = x
            
            fused_features = self._fuse_features(stacked_features)

        # 确保下采样率与标签一致
        batch_dict['down_sample_ratio'] = self.down_sample_ratio
        batch_dict['features_2d'] = fused_features  # 尺寸已与标签匹配

        return batch_dict