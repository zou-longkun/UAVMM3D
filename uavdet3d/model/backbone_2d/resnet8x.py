import torch.nn as nn


class ResNet8x(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet8x, self).__init__()

        self.in_channels, self.out_channels, self.feature_channels = model_cfg.INPUT_CHANNELS, model_cfg.OUT_CHANNELS, model_cfg.NUM_FILTERS

        self.init_block = nn.Conv2d(self.in_channels, self.feature_channels[0], kernel_size=1)

        self.block1 = self._make_block(self.feature_channels[0], self.feature_channels[0])
        self.d1 = self.down_block(self.feature_channels[0], self.feature_channels[1])

        self.block2 = self._make_block(self.feature_channels[1], self.feature_channels[1])
        self.d2 = self.down_block(self.feature_channels[1], self.feature_channels[2])

        self.block3 = self._make_block(self.feature_channels[2], self.feature_channels[2])
        self.d3 = self.down_block(self.feature_channels[2], self.feature_channels[3])

        self.block4 = self._make_block(self.feature_channels[3], self.feature_channels[3])

        self.final_conv = nn.Conv2d(self.feature_channels[3], self.out_channels, kernel_size=3, padding=1)

    def _make_block(self, in_channels, feature_channels):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feature_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = feature_channels

        return nn.Sequential(*layers)

    def down_block(self, in_channels, feature_channels):
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(feature_channels),
                             nn.ReLU(inplace=True))

    def forward(self, batch_dict):
        x = batch_dict['image']  # B,K,3,W,H

        B, K, C, W, H = x.shape

        x = x.reshape(B * K, C, W, H)

        x = self.init_block(x)
        x1 = self.block1(x)
        x1 = x1 + x
        x1 = self.d1(x1)

        x2 = self.block2(x1)
        x2 = x2 + x1
        x2 = self.d2(x2)

        x3 = self.block3(x2)
        x3 = x3 + x2
        x3 = self.d3(x3)

        x4 = self.block4(x3)
        x4 = x4 + x3

        x = self.final_conv(x4)

        BK, C, W, H = x.shape

        x = x.reshape(B, K, C, W, H)

        batch_dict['features_2d'] = x

        return batch_dict


class ResNet50(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet8x, self).__init__()

        self.in_channels, self.out_channels, self.feature_channels = model_cfg.INPUT_CHANNELS, model_cfg.OUT_CHANNELS, model_cfg.NUM_FILTERS

    def forward(self, batch_dict):
        x = batch_dict['image']  # B,K,3,W,H

        B, K, C, W, H = x.shape

        x = x.reshape(B * K, C, W, H)

        # x = resnet50(x)

        BK, C, W, H = x.shape

        x = x.reshape(B, K, C, W, H)

        batch_dict['features_2d'] = x

        return batch_dict
