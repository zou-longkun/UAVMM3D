import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time

class DepthPredictionBlock(nn.Module):
    """专门的深度预测增强模块，针对LiDAR融合优化（新增抗过拟合机制）"""
    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.15):
        super(DepthPredictionBlock, self).__init__()
        # 深度特征增强网络，保留残差连接核心
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)  # 新增dropout，缓解过拟合（适配多场景数据分布差异）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # 残差连接（确保深度特征不丢失）
        self.shortcut = nn.Conv2d(in_channels, mid_channels, kernel_size=1) if in_channels != mid_channels else nn.Identity()
        # 瓶颈层（优化特征维度，提升融合效率）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels//2, kernel_size=1),
            nn.BatchNorm2d(mid_channels//2),
            nn.ReLU(inplace=True)
        )
        # 输出层（保持无偏置设计，避免训练震荡）
        self.out_conv = nn.Conv2d(mid_channels//2, out_channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out) + residual  # 残差融合，保留LiDAR原始深度特征
        out = self.bottleneck(out)
        out = self.out_conv(out)
        out = torch.sigmoid(out)  # 添加sigmoid激活函数，将输出限制在0-1之间，与GT值范围一致
        return out


class CenterHeadLidarFusion(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterHeadLidarFusion, self).__init__()

        self.model_cfg = model_cfg
        self.head_config = self.model_cfg.SEPARATE_HEAD_CFG
        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT
        self.in_c = self.model_cfg.INPUT_CHANNELS

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            # 深度预测任务（center_dis）使用增强模块
            if cur_head_name == 'center_dis':
                # 适配无人机场景：dropout_rate=0.15（平衡特征保留与抗过拟合）
                fc = DepthPredictionBlock(self.in_c, conv_mid, out_c, dropout_rate=0.15)
            else:
                # 其他任务：新增dropout和weight decay友好设计
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(conv_mid),
                    nn.ReLU(),
                    nn.Dropout2d(0.1),  # 轻微dropout，防止其他头过拟合
                    nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=False)
                )
            self.__setattr__(cur_head_name, fc)

        self.forward_loss_dict = dict()
        # 可配置损失权重（使用更合理的默认值）
        self.depth_loss_weight = 1.0  # 深度预测损失权重
        self.l1_ratio = 0.6  # 混合损失中L1占比
        self.l2_ratio = 0.4  # 混合损失中L2占比

    def get_loss(self):
        loss = 0
        total_valid_loss_terms = 0
        # 用于存储每个预测头的loss值
        head_losses = {}

        # 创建调试目录
        debug_dir = 'debug_hm_images_ne'
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 统一可视化参数
        FIG_SIZE = (15, 6)  # 增加宽度以确保子图和颜色条空间
        DPI = 100           # 固定分辨率
        TITLE_FONT_SIZE = 12
        COLOR_MAP = 'jet'
        
        for cur_name in self.head_keys:
            if cur_name in self.forward_loss_dict['pred_center_dict'] and cur_name in self.forward_loss_dict['gt_center_dict']:
                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]
                current_loss = 0

                # 可视化hm图（仅对hm任务）
                # if cur_name == 'hm':
                #     try:
                #         # 处理预测的hm
                #         pred_np = pred_h.cpu().detach().numpy()
                #         # 添加调试信息以分析数值差异
                #         print(f"pred_np dtype: {pred_np.dtype}, min: {pred_np.min()}, max: {pred_np.max()}")
                #         pred_np_sigmoid_1 = torch.sigmoid(torch.from_numpy(pred_np)).numpy()
                #         # 添加数值稳定性处理
                #         pred_np_clamped = np.clip(pred_np, -709, 709)  # 避免np.exp溢出
                #         pred_np_sigmoid_2 = 1.0 / (1.0 + np.exp(-pred_np_clamped))
                #         # 对比原始输入分布
                #         print(f"Original min: {pred_np.min()}, max: {pred_np.max()}, clamped min: {pred_np_clamped.min()}, clamped max: {pred_np_clamped.max()}")
                #         # 计算绝对差异和最大差异
                #         diff = np.abs(pred_np_sigmoid_1 - pred_np_sigmoid_2)
                #         print(f"Max absolute difference: {diff.max()}")
                #         print(f"Are values approximately equal: {np.allclose(pred_np_sigmoid_1, pred_np_sigmoid_2, atol=1e-6)}")
                #         gt_np = gt_h.cpu().detach().numpy()
                #         print(gt_np.shape, pred_np_sigmoid_1.shape)
                #         # 同时处理两种sigmoid结果进行对比
                #         pred_combined_1 = np.sum(pred_np_sigmoid_1[0], axis=0)
                #         pred_combined_2 = np.sum(pred_np_sigmoid_2[0], axis=0)
                        
                #         # 对比合并后的差异
                #         sum_diff = np.abs(pred_combined_1 - pred_combined_2)
                #         print(f"Summed difference max: {sum_diff.max()}, mean: {sum_diff.mean()}")
                        
                #         # 对比阈值过滤效果
                #         pred_filtered_1 = pred_combined_1.copy()
                #         pred_filtered_1[pred_filtered_1 < 0.2] = 0
                #         pred_filtered_2 = pred_combined_2.copy()
                #         pred_filtered_2[pred_filtered_2 < 0.2] = 0
                        
                #         filter_diff = np.abs(pred_filtered_1 - pred_filtered_2)
                #         print(f"Filtered difference count: {np.sum(filter_diff > 0)}")
                        
                #         # 保存标准化对比图像
                #         import matplotlib.pyplot as plt
                #         fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI, gridspec_kw={'width_ratios': [1, 1]})
                        
                #         # 绘制左侧图像 (PyTorch Sigmoid)
                #         im1 = axes[0].imshow(pred_combined_1, cmap=COLOR_MAP)
                #         axes[0].set_title('PyTorch Sigmoid', fontsize=TITLE_FONT_SIZE)
                #         axes[0].set_xlabel('X Axis')
                #         axes[0].set_ylabel('Y Axis')
                #         fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.02, pad=0.02)
                        
                #         # 绘制右侧图像 (NumPy Sigmoid)
                #         im2 = axes[1].imshow(pred_combined_2, cmap=COLOR_MAP)
                #         axes[1].set_title('NumPy Sigmoid', fontsize=TITLE_FONT_SIZE)
                #         axes[1].set_xlabel('X Axis')
                #         axes[1].set_ylabel('Y Axis')
                #         fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.02, pad=0.02)
                        
                #         # 调整布局并保存
                #         plt.subplots_adjust(right=0.75)
                #         plt.tight_layout()
                #         plt.savefig(f"{debug_dir}/sigmoid_comparison_{timestamp}.png", dpi=DPI, bbox_inches='tight')
                #         plt.close()
                        
                #         # 保留原始逻辑但明确使用其中一种实现
                #         pred_combined_sigmoid = pred_combined_2  # 或pred_combined_2
                #         gt_combined = np.sum(gt_np[0], axis=0)        
                #         # 对连续值预测图添加confidence>0.2的过滤
                #         pred_combined_filtered = pred_combined_sigmoid.copy()
                #         pred_combined_filtered[pred_combined_filtered < 0.2] = 0
                #         # 添加调试信息
                #         print(f"pred_combined_sigmoid max: {pred_combined_sigmoid.max():.4f}, min: {pred_combined_sigmoid.min():.4f}")
                #         print(f"pred_combined_filtered max: {pred_combined_filtered.max():.4f}, min: {pred_combined_filtered.min():.4f}")
                #         print(f"gt_combined max: {gt_combined.max():.4f}, min: {gt_combined.min():.4f}")
                #         # 这是正常现象，因为gt_heatmap使用高斯分布生成，不是简单的二值图
                #         # 直接将值乘以255转换为图像格式（因为最大值是1）
                #         pred_img_filtered = (pred_combined_filtered * 255).astype(np.uint8)
                #         gt_img = (gt_combined * 255).astype(np.uint8)    
                #         # 应用颜色映射
                #         pred_colored_filtered = cv2.applyColorMap(pred_img_filtered, cv2.COLORMAP_JET)
                #         # 标准化颜色映射范围到0-1
                #         pred_colored_filtered = cv2.normalize(pred_colored_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                #         gt_colored = cv2.applyColorMap(gt_img, cv2.COLORMAP_JET)
                #         # 确保所有图像尺寸相同
                #         target_shape = (max(pred_colored_filtered.shape[1], gt_colored.shape[1]), 
                #                         max(pred_colored_filtered.shape[0], gt_colored.shape[0]))
                #         pred_colored_filtered = cv2.resize(pred_colored_filtered, target_shape)
                #         gt_colored = cv2.resize(gt_colored, target_shape)
                #         # 添加标签文字（适配120*70低分辨率图像）
                #         def add_text_to_image(image, text, position=(5, 12)):
                #             # 确保图像是3通道的
                #             if len(image.shape) == 2:
                #                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                #             # 添加半透明背景（小尺寸）
                #             overlay = image.copy()
                #             cv2.rectangle(overlay, (position[0]-2, position[1]-10), 
                #                             (position[0]+len(text)*6+2, position[1]+2), (0, 0, 0), -1)
                #             alpha = 0.6  # 透明度
                #             cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                #             # 添加文字（小字体）
                #             cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                #             return image
                #         # 添加标签
                #         pred_colored_filtered = add_text_to_image(pred_colored_filtered, "Pred (>0.2)")
                #         gt_colored = add_text_to_image(gt_colored, "GT")
                #         # 创建分割线
                #         separator = np.ones((target_shape[1], 10, 3), dtype=np.uint8) * 255
                #         # 使用matplotlib创建标准化组合图像
                #         import matplotlib.pyplot as plt
                #         fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI, gridspec_kw={'width_ratios': [1, 1]})
                        
                #         # 绘制预测图
                #         im1 = axes[0].imshow(pred_combined_filtered, cmap=COLOR_MAP, vmin=0, vmax=1)
                #         axes[0].set_title('Pred (>0.2)', fontsize=TITLE_FONT_SIZE)
                #         axes[0].set_xlabel('X Axis')
                #         axes[0].set_ylabel('Y Axis')
                #         fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.02, pad=0.02)
                        
                #         # 绘制GT图
                #         im2 = axes[1].imshow(gt_combined, cmap=COLOR_MAP)
                #         axes[1].set_title('GT', fontsize=TITLE_FONT_SIZE)
                #         axes[1].set_xlabel('X Axis')
                #         axes[1].set_ylabel('Y Axis')
                #         fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.02, pad=0.02)
                        
                #         # 调整布局并保存
                #         plt.tight_layout()
                #         combined_filename = os.path.join(debug_dir, f'hm_visualization_{timestamp}.png')
                #         plt.savefig(combined_filename, dpi=DPI, bbox_inches='tight')
                #         plt.close()
                        
                #         # 保存单独图像（标准化格式）
                #         plt.figure(figsize=FIG_SIZE, dpi=DPI)
                #         im = plt.imshow(pred_combined_filtered, cmap=COLOR_MAP)
                #         plt.title('Pred (>0.2)', fontsize=TITLE_FONT_SIZE)
                #         plt.colorbar(im, label='Confidence', orientation='vertical', location='right', fraction=0.046, pad=0.04)
                #         plt.subplots_adjust(right=0.85)
                #         plt.tight_layout()
                #         plt.savefig(os.path.join(debug_dir, f'pred_filtered_{timestamp}.png'), dpi=DPI, bbox_inches='tight')
                #         plt.close()
                        
                #         plt.figure(figsize=FIG_SIZE, dpi=DPI)
                #         im = plt.imshow(gt_combined, cmap=COLOR_MAP)
                #         plt.title('GT', fontsize=TITLE_FONT_SIZE)
                #         plt.colorbar(im, label='Confidence', orientation='vertical', location='right', fraction=0.046, pad=0.04)
                #         plt.subplots_adjust(right=0.85)
                #         plt.tight_layout()
                #         plt.savefig(os.path.join(debug_dir, f'gt_{timestamp}.png'), dpi=DPI, bbox_inches='tight')
                #         plt.close()
                #         print(f'Debug: Saved heatmap visualizations to {debug_dir}')
                #     except Exception as e:
                #         print(f"Error visualizing {cur_name}: {e}")
                #         import traceback
                #         traceback.print_exc()
                
                if cur_name == 'hm':
                    pred = pred_h.reshape(-1)
                    gt = gt_h.reshape(-1)
                    mask = gt > 0
                    mask_sum = mask.sum().item()
                    
                    # 使用标准Focal Loss缓解类别不平衡
                    alpha = 0.25  # 正样本权重，标准值在[0,1]之间
                    gamma = 2.0  # Focal Loss的聚焦参数
                    loss_weight = 1.0  # 整体损失权重因子
                    
                    # 确保alpha是标量，避免维度不匹配
                    alpha_val = float(alpha)
                    
                    # Focal Loss计算
                    # 使用clamp避免数值不稳定
                    sigmoid_p = torch.sigmoid(pred).clamp(min=1e-6, max=1-1e-6)
                    log_p = sigmoid_p.log()
                    log_one_minus_p = (1 - sigmoid_p).clamp(min=1e-6, max=1-1e-6).log()
                    
                    # 标准Focal Loss实现
                    pos_loss = -alpha_val * ((1 - sigmoid_p) ** gamma) * log_p * gt
                    neg_loss = -(1 - alpha_val) * (sigmoid_p ** gamma) * log_one_minus_p * (1 - gt)
                    
                    focal_loss = pos_loss + neg_loss
                    
                    if mask_sum > 0:
                        # 使用标准mean()归一化
                        focal_loss_total = focal_loss.mean() * loss_weight
                        current_loss = focal_loss_total.item()
                        loss += focal_loss_total
                        total_valid_loss_terms += loss_weight
                    else:
                        # 增加负样本正则化权重
                        neg_loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean') * 0.1
                        current_loss = neg_loss.item()
                        loss += neg_loss
                        total_valid_loss_terms += 0.1
                else:
                    pred = pred_h.reshape(-1)
                    gt = gt_h.reshape(-1)
                    mask_x = torch.abs(gt) > 0
                    mask_count = mask_x.sum().item()
                    
                    if mask_count > 0:
                        if cur_name == 'center_dis':
                            # 优化混合损失：适配无人机深度噪声（雨雾、遮挡场景）
                            gt_valid = gt[mask_x]
                            pred_valid = pred[mask_x]
                            
                            # L1抗噪（应对突发深度偏差）+ L2平滑（稳定连续帧预测）
                            diff = gt_valid - pred_valid
                            actual_diff = diff * 100.0
                            l1_loss = torch.abs(actual_diff).mean()
                            l2_loss = (actual_diff ** 2).mean()
                            
                            # 联合优化：距离误差权重 + 动态平衡系数
                            loss_res = self.l1_ratio * l1_loss + self.l2_ratio * l2_loss
                            # 降低基础权重以抵消MAX_DIS放大效应
                            weighted_loss = loss_res * (self.depth_loss_weight / 100.0)
                            current_loss = weighted_loss.item()
                            
                            loss += weighted_loss
                            total_valid_loss_terms += self.depth_loss_weight
                        else:
                            # 其他回归任务：保留L1，确保定位精准
                            if cur_name == 'dim':
                                # 应用sigmoid激活函数，约束预测值在0-1范围内
                                pred_sigmoid = torch.sigmoid(pred)
                                loss_res = torch.abs(gt[mask_x] - pred_sigmoid[mask_x]).mean()
                                current_loss = loss_res.item()
                                loss += loss_res
                                total_valid_loss_terms += 1
                            elif cur_name == 'rot':
                                # 对于rot任务（cos/sin值）：增加权重并优化损失计算
                                gt_valid = gt[mask_x]
                                pred_valid = pred[mask_x]
                                
                                # 计算每个维度的绝对差异
                                abs_diff = torch.abs(gt_valid - pred_valid)
                                
                                # 为rot任务设置较大权重，帮助loss更快下降
                                rot_weight = 2.0
                                loss_res = abs_diff.mean() * rot_weight
                                current_loss = loss_res.item()
                                
                                loss += loss_res
                                total_valid_loss_terms += rot_weight
                                
                                # 可选：打印rot损失信息用于调试
                                print(f"rot_gt: {gt_valid[:4].tolist()}")
                                print(f"rot_pred: {pred_valid[:4].tolist()}")
                                print(f"rot_abs_diff_mean: {abs_diff.mean().item()}")
                            else:
                                # 其他回归任务保持原有设置
                                loss_res = torch.abs(gt[mask_x] - pred[mask_x]).mean()
                                current_loss = loss_res.item()
                                loss += loss_res
                                total_valid_loss_terms += 1
                    else:
                        # 无有效GT时：强化深度预测正则化（避免预测值漂移）
                        if cur_name == 'center_dis':
                            reg_loss = torch.abs(pred).mean() * 0.01
                            current_loss = reg_loss.item()
                            loss += reg_loss
                            total_valid_loss_terms += 0.01
                        else:
                            reg_loss = torch.abs(pred).mean() * 0.001
                            current_loss = reg_loss.item()
                            loss += reg_loss
                            total_valid_loss_terms += 0.001
                
                # 存储当前头的loss值
                head_losses[cur_name] = current_loss
        
        # 打印所有预测头的loss值，并找出最差的预测
        if head_losses:
            print("\n各预测头loss值:")
            for head_name, head_loss in head_losses.items():
                print(f"{head_name}: {head_loss:.6f}")
            
            # 找出loss最大的预测头
            worst_head = max(head_losses, key=head_losses.get)
            print(f"\n最差的预测是: {worst_head}, loss值: {head_losses[worst_head]:.6f}")
        
        # 兜底损失（确保梯度连续）
        if total_valid_loss_terms == 0:
            return torch.tensor(1e-6, requires_grad=True, device=next(self.parameters()).device)
        
        return loss

    def forward(self, batch_dict):
        pred_dict = {}
        gt_dict = {}

        x = batch_dict['features_2d']  # 兼容多模态特征输入（RGB/IR/LiDAR融合后特征）
        
        # 创建调试目录
        debug_dir = 'debug_hm_images'
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        for cur_name in self.head_keys:
            head_module = self.__getattr__(cur_name)
            pred_dict[cur_name] = head_module(x)

            # 可视化hm图（仅对hm任务）
            # if cur_name == 'hm':
            #     try:
            #         # 处理预测的hm
            #         pred_np = pred_dict[cur_name].cpu().detach().numpy()
            #         # 添加调试信息以分析数值差异
            #         print(f"pred_np dtype: {pred_np.dtype}, min: {pred_np.min()}, max: {pred_np.max()}")
            #         pred_np_sigmoid_1 = torch.sigmoid(torch.from_numpy(pred_np)).numpy()
            #         # 添加数值稳定性处理
            #         pred_np_clamped = np.clip(pred_np, -709, 709)  # 避免np.exp溢出
            #         pred_np_sigmoid_2 = 1.0 / (1.0 + np.exp(-pred_np_clamped))
            #         # 对比原始输入分布
            #         print(f"Original min: {pred_np.min()}, max: {pred_np.max()}, clamped min: {pred_np_clamped.min()}, clamped max: {pred_np_clamped.max()}")
            #         # 计算绝对差异和最大差异
            #         diff = np.abs(pred_np_sigmoid_1 - pred_np_sigmoid_2)
            #         print(f"Max absolute difference: {diff.max()}")
            #         print(f"Are values approximately equal: {np.allclose(pred_np_sigmoid_1, pred_np_sigmoid_2, atol=1e-6)}")
            #         gt_np = batch_dict[cur_name].cpu().detach().numpy()
            #         print(gt_np.shape, pred_np_sigmoid_1.shape)
            #         # 同时处理两种sigmoid结果进行对比
            #         pred_combined_1 = np.sum(pred_np_sigmoid_1[0], axis=0)
            #         pred_combined_2 = np.sum(pred_np_sigmoid_2[0], axis=0)
                    
            #         # 对比合并后的差异
            #         sum_diff = np.abs(pred_combined_1 - pred_combined_2)
            #         print(f"Summed difference max: {sum_diff.max()}, mean: {sum_diff.mean()}")
                    
            #         # 对比阈值过滤效果
            #         pred_filtered_1 = pred_combined_1.copy()
            #         pred_filtered_1[pred_filtered_1 < 0.2] = 0
            #         pred_filtered_2 = pred_combined_2.copy()
            #         pred_filtered_2[pred_filtered_2 < 0.2] = 0
                    
            #         filter_diff = np.abs(pred_filtered_1 - pred_filtered_2)
            #         print(f"Filtered difference count: {np.sum(filter_diff > 0)}")
                    
            #         # 保留原始逻辑但明确使用其中一种实现
            #         pred_combined_sigmoid = pred_combined_2  # 或pred_combined_2
            #         gt_combined = np.sum(gt_np[0], axis=0)        
            #         # 对连续值预测图添加confidence>0.2的过滤
            #         pred_combined_filtered = pred_combined_sigmoid.copy()
            #         pred_combined_filtered[pred_combined_filtered < 0.2] = 0
            #         # 添加调试信息
            #         print(f"pred_combined_sigmoid max: {pred_combined_sigmoid.max():.4f}, min: {pred_combined_sigmoid.min():.4f}")
            #         print(f"pred_combined_filtered max: {pred_combined_filtered.max():.4f}, min: {pred_combined_filtered.min():.4f}")
            #         print(f"gt_combined max: {gt_combined.max():.4f}, min: {gt_combined.min():.4f}")
            #         # 这是正常现象，因为gt_heatmap使用高斯分布生成，不是简单的二值图
            #         # 直接将值乘以255转换为图像格式（因为最大值是1）
            #         pred_img_filtered = (pred_combined_filtered * 255).astype(np.uint8)
            #         gt_img = (gt_combined * 255).astype(np.uint8)    
            #         # 应用颜色映射
            #         pred_colored_filtered = cv2.applyColorMap(pred_img_filtered, cv2.COLORMAP_JET)
            #         # 标准化颜色映射范围到0-1
            #         pred_colored_filtered = cv2.normalize(pred_colored_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #         gt_colored = cv2.applyColorMap(gt_img, cv2.COLORMAP_JET)
            #         # 确保所有图像尺寸相同
            #         target_shape = (max(pred_colored_filtered.shape[1], gt_colored.shape[1]), 
            #                         max(pred_colored_filtered.shape[0], gt_colored.shape[0]))
            #         pred_colored_filtered = cv2.resize(pred_colored_filtered, target_shape)
            #         gt_colored = cv2.resize(gt_colored, target_shape)
            #         # 添加标签文字（适配120*70低分辨率图像）
            #         def add_text_to_image(image, text, position=(5, 12)):
            #             # 确保图像是3通道的
            #             if len(image.shape) == 2:
            #                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            #             # 添加半透明背景（小尺寸）
            #             overlay = image.copy()
            #             cv2.rectangle(overlay, (position[0]-2, position[1]-10), 
            #                          (position[0]+len(text)*6+2, position[1]+2), (0, 0, 0), -1)
            #             alpha = 0.6  # 透明度
            #             cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            #             # 添加文字（小字体）
            #             cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            #             return image
            #         # 添加标签
            #         pred_colored_filtered = add_text_to_image(pred_colored_filtered, "Pred (>0.2)")
            #         gt_colored = add_text_to_image(gt_colored, "GT")
            #         # 创建分割线
            #         separator = np.ones((target_shape[1], 10, 3), dtype=np.uint8) * 255
            #         # 创建两图并排的组合图像
            #         combined_image = np.hstack((pred_colored_filtered, separator, gt_colored))
            #         # 保存可视化结果
            #         os.makedirs(debug_dir, exist_ok=True)
            #         combined_filename = os.path.join(debug_dir, f'hm_visualization_{timestamp}.png')
            #         cv2.imwrite(combined_filename, combined_image)
            #         # 也保存单独的图像以便查看
            #         cv2.imwrite(os.path.join(debug_dir, f'pred_filtered_{timestamp}.png'), pred_colored_filtered)
            #         cv2.imwrite(os.path.join(debug_dir, f'gt_{timestamp}.png'), gt_colored)
            #         print(f'Debug: Saved heatmap visualizations to {debug_dir}')
            #     except Exception as e:
            #         print(f"Error visualizing {cur_name}: {e}")
            #         import traceback
            #         traceback.print_exc()
            # if cur_name == 'center_dis':
            #     gt = batch_dict[cur_name]
            #     # 打印预测值与GT差距（训练/测试阶段均可）

            
            #     if cur_name in pred_dict:
            #         pred_val = pred_dict[cur_name]
            #         print(pred_val.shape)
            #         abs_diff = torch.abs(pred_val - gt)
            #         # 仅计算GT非零位置的差异并打印
            #         non_zero_mask = gt != 0
            #         non_zero_diff = abs_diff[non_zero_mask]
            #         # 计算实际距离误差（乘以MAX_DIS）
            #         actual_diff = non_zero_diff * 100
            #         print(f"GT non-zero differences (normalized): {non_zero_diff}")
            #         print(f"GT non-zero differences (meters): {actual_diff}")
            #         print(f"[{cur_name}] pred mean: {pred_val.mean().item():.4f}, "
            #               f"GT mean: {gt.mean().item():.4f}, "
            #               f"GT non-zero count: {torch.count_nonzero(gt)}, "
            #               f"GT non-zero samples: {gt[gt != 0].tolist()}, "
            #               f"abs_diff mean: {abs_diff.mean().item():.4f}, "
            #               f"max_diff: {abs_diff.max().item():.4f}")
                

            if self.training:
                if cur_name in batch_dict:
                    gt = batch_dict[cur_name]
                    gt_dict[cur_name] = gt

        batch_dict['pred_center_dict'] = pred_dict
        self.forward_loss_dict['pred_center_dict'] = pred_dict

        if self.training:
            batch_dict['gt_dict'] = gt_dict
            self.forward_loss_dict['gt_center_dict'] = gt_dict

        return batch_dict