# LAAM6D检测器配置 - 使用基于LiDAR的多模态特征融合
from easydict import EasyDict as edict

cfg = edict()

# 数据配置
cfg.DATASET = edict()
cfg.DATASET.DATA_PATH = 'D:\\LAAM6D_MINI\\data_collect'
cfg.DATASET.NAME = 'LidarBasedFusionDataset'
cfg.DATASET.DATASET = cfg.DATASET.NAME

# LiDAR相关配置
cfg.DATASET.LIDAR_FUSION = edict()
cfg.DATASET.LIDAR_FUSION.MIN_VALID_POINTS = 10  # 最小有效点数量
cfg.DATASET.LIDAR_FUSION.RANSAC_THRESHOLD = 3.0  # RANSAC阈值(像素)
cfg.DATASET.LIDAR_FUSION.USE_ALTERNATIVE = True  # 当LiDAR对齐失败时使用中心点对齐作为备选
# cfg.DATASET.CLASS_NAMES = ['DJI-avata2','DJI-phantom4','drone-unk3','m210-rtk','DJI-mavic-mini','Matrice-600-Pro', 'matrix-300-RTK']
cfg.DATASET.CLASS_NAMES = ['drone']  # 所有类别合并为一个'drone'类别
cfg.DATASET.SEQ_IDS = ['00001', '00002','00004']  # 可配置的无人机型号列表
cfg.DATASET.WEATHER_NAMES = ['clear_day']  # 可配置的天气条件列表
cfg.DATASET.MAX_NUM = 100000000
cfg.DATASET.OB_SIZE = [[1.3, 1.3, 1.3]]  # 目标尺寸 (长, 宽, 高)
cfg.DATASET.IM_NUM = 1
cfg.DATASET.IM_SIZE = [1280, 720]
cfg.DATASET.IM_RESIZE = [960, 560]
cfg.DATASET.LIDAR_OFFSET = 6
cfg.DATASET.RADAR_OFFSET = 8
cfg.DATASET.MAX_DIS = 150
cfg.DATASET.MAX_SIZE = 4 # 最大目标尺寸其实不超过2m，最好改成2
cfg.DATASET.CENTER_RAD = 4  # hm预测的半径，建议设置为2，与stride一致
cfg.DATASET.STRIDE = 4
cfg.DATASET.SAMPLED_INTERVAL = {  # 修改为正确的字典结构，与原始配置保持一致
    'train': 1,
    'test': 1,
    'val': 1
}
cfg.DATASET.DATA_SPLIT = {
    'train': 'train',
    'test': 'test'
}

# 添加LAA3D_ADS_METRIC配置，用于评估指标计算
cfg.DATASET.LAA3D_ADS_METRIC = edict()
cfg.DATASET.LAA3D_ADS_METRIC.DisMax = 200
cfg.DATASET.LAA3D_ADS_METRIC.DisMin = 0
cfg.DATASET.LAA3D_ADS_METRIC.MinPixel = 0
cfg.DATASET.LAA3D_ADS_METRIC.AP2D = edict()
cfg.DATASET.LAA3D_ADS_METRIC.AP2D.IoUThresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cfg.DATASET.LAA3D_ADS_METRIC.AP2D.RecallNum = 101
cfg.DATASET.LAA3D_ADS_METRIC.AP3D = edict()
cfg.DATASET.LAA3D_ADS_METRIC.AP3D.DisThresh = [1, 2, 4, 6]
cfg.DATASET.LAA3D_ADS_METRIC.AP3D.RecallNum = 101
cfg.DATASET.LAA3D_ADS_METRIC.Dof6 = edict()
cfg.DATASET.LAA3D_ADS_METRIC.Dof6.DisNormMax = 8
cfg.DATASET.LAA3D_ADS_METRIC.Dof6.OriNormMax = 30
cfg.DATASET.LAA3D_ADS_METRIC.Dof6.SizeNormMax = 1

# 数据预处理配置 - 注意：此处必须是列表格式，每个处理器都需要有NAME属性
cfg.DATASET.DATA_PRE_PROCESSOR = []

# 添加filter_box_outside处理器
filter_box_outside = edict()
filter_box_outside.NAME = 'filter_box_outside'
cfg.DATASET.DATA_PRE_PROCESSOR.append(filter_box_outside)

# 添加convert_box9d_to_centermap处理器
convert_box9d_to_centermap = edict()
convert_box9d_to_centermap.NAME = 'convert_box9d_to_centermap'
convert_box9d_to_centermap.OFFSET = [0.0, 0.0, 0.0]
convert_box9d_to_centermap.ENCODER = 'center_point_encoder'
cfg.DATASET.DATA_PRE_PROCESSOR.append(convert_box9d_to_centermap)

# 添加image_normalization处理器
image_normalization = edict()
image_normalization.NAME = 'image_normalization'
cfg.DATASET.DATA_PRE_PROCESSOR.append(image_normalization)

# 使用基于LiDAR的图像对齐方法
cfg.DATASET.ALIGN_METHOD = 'lidar_based'

# 网络配置
cfg.MODEL = edict()
cfg.MODEL.NAME = 'LidarBasedFusionDetector'  # 使用我们新创建的融合检测器

# 2D骨干网络配置
cfg.MODEL.BACKBONE_2D = edict()
cfg.MODEL.BACKBONE_2D.NAME = 'LightweightAttentionNet'
cfg.MODEL.BACKBONE_2D.STRIDE = cfg.DATASET.STRIDE  # 与数据集步长保持一致
cfg.MODEL.BACKBONE_2D.FUSE_METHOD = 'channel_concat'  # [channel_concat, concat, weighted_sum, attention, gatting]
cfg.MODEL.BACKBONE_2D.ATTENTION_HEADS = 1
cfg.MODEL.BACKBONE_2D.NUM_MODALITIES = 3  # 每个分支有3种模态：RGB/IR + LiDAR投影 + Radar投影
cfg.MODEL.BACKBONE_2D.NUM_FILTERS = [32,64,128,128]  # 轻量化网络使用更小的通道数
cfg.MODEL.BACKBONE_2D.INPUT_CHANNELS = 3  # 这个参数弃用
cfg.MODEL.BACKBONE_2D.OUT_CHANNELS = 512  # 轻量化网络输出通道减半
cfg.MODEL.BACKBONE_2D.STRIDES = [1, 2, 2, 2]
cfg.MODEL.BACKBONE_2D.DILATIONS = [1, 1, 1, 1]
cfg.MODEL.BACKBONE_2D.LIGHTWEIGHT_MODE = True  # 启用轻量级模式
cfg.MODEL.BACKBONE_2D.ULTRA_LIGHT = False  # 是否使用极轻量化模式

# 密集头配置
cfg.MODEL.DENSE_HEAD_2D = edict()
cfg.MODEL.DENSE_HEAD_2D.NAME = 'CenterHeadLidarFusion'
cfg.MODEL.DENSE_HEAD_2D.INPUT_CHANNELS = 512
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG = edict()
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_ORDER = ['hm', 'center_res', 'center_dis', 'dim', 'rot']
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT = edict()
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.hm = edict()
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.hm.out_channels = 1
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.hm.conv_dim = 128
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.center_res = edict()
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.center_res.out_channels = 2
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.center_res.conv_dim = 128
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.center_dis = edict()
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.center_dis.out_channels = 1
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.center_dis.conv_dim = 128
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.dim = edict()
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.dim.out_channels = 3
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.dim.conv_dim = 128
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.rot = edict()
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.rot.out_channels = 6
cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.rot.conv_dim = 128

# LiDAR-based对齐器配置参数
cfg.MODEL.LIDAR_ALIGNER = edict()
cfg.MODEL.LIDAR_ALIGNER.MIN_VALID_POINTS = 10  # 最小有效点数量
cfg.MODEL.LIDAR_ALIGNER.RANSAC_THRESHOLD = 3.0  # RANSAC阈值(像素)
cfg.MODEL.LIDAR_ALIGNER.USE_ALTERNATIVE = True  # 当LiDAR对齐失败时使用中心点对齐作为备选

# 特征融合配置
cfg.MODEL.FEATURE_FUSION = edict()
cfg.MODEL.FEATURE_FUSION.IN_CHANNELS = 512  # 输入特征通道数，应与轻量化骨干网络输出匹配
cfg.MODEL.FEATURE_FUSION.OUT_CHANNELS = 512  # 输出特征通道数
cfg.MODEL.FEATURE_FUSION.USE_RELATIONSHIP_MATRIX = True  # 是否使用LiDAR构建的关系矩阵
cfg.MODEL.FEATURE_FUSION.METHOD = 'concat'  # 融合方法: 'attention', 'concat', 'add', 'avg'，注意：与USE_RELATIONSHIP_MATRIX冲突
cfg.MODEL.FEATURE_FUSION.FEATURE_SCALE_FACTOR = 8  # 特征图相对于原图的缩放因子
cfg.MODEL.FEATURE_FUSION.USE_ORIGINAL_FUSION = False  # 设置为True以使用RGB和IR数据进行融合, False使用LGFusionNet

# 数据增强配置 - 保留AUGMENTATION名称并添加DATA_AUGMENTOR以确保兼容性
cfg.AUGMENTATION = edict()
cfg.AUGMENTATION.RANDOM_FLIP = True
cfg.AUGMENTATION.RANDOM_ROTATE = True
cfg.AUGMENTATION.RANDOM_SCALE = True
cfg.AUGMENTATION.BRIGHTNESS = 0.2
cfg.AUGMENTATION.CONTRAST = 0.2
# 修改AUG_CONFIG_LIST为正确的列表格式，每个元素包含NAME属性
cfg.AUG_CONFIG_LIST = []
# 添加随机翻转增强器配置
aug_flip = edict()
aug_flip.NAME = 'image_flip'
aug_flip.ALONG_AXIS_LIST = ['x']
aug_flip.DATA_LIST = ['img']
cfg.AUG_CONFIG_LIST.append(aug_flip)
# 添加随机旋转增强器配置
aug_rot = edict()
aug_rot.NAME = 'image_rot'
aug_rot.ROT_RANGE = [-0.1, 0.1]
aug_rot.DATA_LIST = ['img']
cfg.AUG_CONFIG_LIST.append(aug_rot)
# 添加亮度增强器配置
aug_brightness = edict()
aug_brightness.NAME = 'image_brightness'
aug_brightness.DATA_LIST = ['img']
cfg.AUG_CONFIG_LIST.append(aug_brightness)
# 添加高斯噪声增强器配置
aug_gaussian = edict()
aug_gaussian.NAME = 'image_gaussian_noise'
aug_gaussian.DATA_LIST = ['img']
cfg.AUG_CONFIG_LIST.append(aug_gaussian)
cfg.AUGMENTATION.AUG_CONFIG_LIST = cfg.AUG_CONFIG_LIST
cfg.DATASET.DATA_AUGMENTOR = cfg.AUGMENTATION

# 优化器配置 - 保持OPTIMAZATION拼写以匹配代码引用，并按照标准配置更新
cfg.OPTIMIZATION = edict()
cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU = 2
cfg.OPTIMIZATION.USE_AMP = False
cfg.OPTIMIZATION.NUM_EPOCHS = 100
cfg.OPTIMIZATION.OPTIMIZER = 'adam_onecycle'
cfg.OPTIMIZATION.LR = 0.001
cfg.OPTIMIZATION.WEIGHT_DECAY = 0.01
cfg.OPTIMIZATION.MOMENTUM = 0.9
cfg.OPTIMIZATION.MOMS = [0.95, 0.85]
cfg.OPTIMIZATION.PCT_START = 0.4
cfg.OPTIMIZATION.DIV_FACTOR = 10
cfg.OPTIMIZATION.DECAY_STEP_LIST = [5, 10]
cfg.OPTIMIZATION.LR_DECAY = 0.1
cfg.OPTIMIZATION.LR_CLIP = 0.0000001
cfg.OPTIMIZATION.LR_WARMUP = False
cfg.OPTIMIZATION.WARMUP_EPOCH = 1
cfg.OPTIMIZATION.GRAD_NORM_CLIP = 10

# 后处理配置 - 移动到MODEL下以匹配LidarBasedFusionDetector的访问方式
cfg.MODEL.POST_PROCESSING = edict()
cfg.MODEL.POST_PROCESSING.DECONDER = 'center_point_decoder'
cfg.MODEL.POST_PROCESSING.MAX_OBJ = 10 # 最大检测对象数
cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.5
cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST = [0.3, 0.5, 0.7]

# 可视化配置
cfg.VISUALIZATION = edict()
cfg.VISUALIZATION.ENABLE = True
cfg.VISUALIZATION.SAVE_DIR = '../output/vis'
cfg.VISUALIZATION.SHOW_ALIGNED_IMAGES = True  # 显示对齐后的图像
cfg.VISUALIZATION.SHOW_FUSED_FEATURES = True  # 显示融合后的特征（调试用）

# 测试配置
cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.NMS_THRESH = 0.2
cfg.TEST.SCORE_THRESH = 0.2

# 调试配置
cfg.DEBUG = edict()
cfg.DEBUG.ENABLE = False
cfg.DEBUG.SAVE_ALIGNMENT_DEBUG_INFO = False
cfg.DEBUG.SAVE_FUSION_DEBUG_INFO = False  # 保存融合过程的调试信息

# 输出配置
cfg.OUTPUT_DIR = '../output'

# 设置标签和实验路径
cfg.TAG = 'laam6d_lidar_fusion'
cfg.EXP_GROUP_PATH = 'models/uavdet_3d/'

# 日志配置
cfg.LOG_DIR = './logs'

# 其他配置
cfg.NUM_WORKERS = 1
cfg.SEED = 42

# 交换IR/RGB
cfg.SWAP_RGB_IR = True

# 返回配置
def get_config():
    return cfg
