import datetime
import glob
import os
import sys
from pathlib import Path

current_script = Path(__file__).resolve()
new_project_root = current_script.parent.parent
if str(new_project_root) not in sys.path:
    sys.path.insert(0, str(new_project_root))

import argparse
import datetime
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from uavdet3d.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.model import build_network, model_fn_decorator
from uavdet3d.utils import common_utils
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.train_utils.train_utils import train_model
from tools.test import repeat_eval_ckpt


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # 使用我们新创建的融合配置文件作为默认值
    parser.add_argument('--cfg_file', type=str,
                        default='../configs/laam6d_det_lidar_fusion_config.py',
                        help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=None, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='fusion', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')

    # NOTE:
    # 你的 torchrun 会设置环境变量 MASTER_PORT/MASTER_ADDR。
    # 原来这里默认 18888，容易导致 init_dist_pytorch 用的端口与 torchrun 的 --master_port 不一致。
    # 为了兼容与稳定，tcp_port 允许为 None，None 时优先从环境变量 MASTER_PORT 读取。
    parser.add_argument('--tcp_port', type=int, default=None, help='tcp port for distrbuted training')

    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False,
                        help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')

    args = parser.parse_args()

    # 加载Python格式的配置文件
    # 将相对路径转换为绝对路径
    cfg_file_path = os.path.abspath(args.cfg_file)
    # 提取模块名称
    module_name = os.path.basename(cfg_file_path).split('.')[0]
    # 创建模块规范
    spec = importlib.util.spec_from_file_location(module_name, cfg_file_path)
    # 创建模块
    config_module = importlib.util.module_from_spec(spec)
    # 添加到sys.modules
    sys.modules[module_name] = config_module
    # 执行模块
    spec.loader.exec_module(config_module)
    # 获取配置对象
    cfg = config_module.cfg

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    # 让 tcp_port 与 torchrun 保持一致（优先读环境变量 MASTER_PORT）
    if args.tcp_port is None:
        args.tcp_port = int(os.environ.get('MASTER_PORT', '18888'))

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
        cfg.LOCAL_RANK = 0
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.workers is None:
        args.workers = cfg.NUM_WORKERS

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # 调整输出目录结构以适应新的配置路径
    output_dir = Path(cfg.OUTPUT_DIR) / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATASET,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        seed=666 if args.fix_random_seed else None
    )

    # --------- 关键修复：显式绑定每个进程到自己的 GPU ----------
    if dist_train:
        # torchrun 会设置 LOCAL_RANK；这里 cfg.LOCAL_RANK 已由 init_dist_* 返回/设置
        torch.cuda.set_device(cfg.LOCAL_RANK)
        device = torch.device('cuda', cfg.LOCAL_RANK)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -----------------------------------------------------------

    model = build_network(model_cfg=cfg.MODEL, dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 原来是 model.cuda()（容易导致所有 rank 默认跑到 cuda:0）
    model = model.to(device)

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        # NOTE: 你原来是 to_cpu=dist_train（dist_train=True 时加载到 CPU）
        # 这里不强行改你底层函数语义，只保持原逻辑（如果你确认函数语义相反，再一起改）
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train,
                                                           optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters

    # 注意：梯度异常检测会显著减慢训练速度，仅在调试时启用
    # torch.autograd.set_detect_anomaly(True)  # 默认禁用

    if dist_train:
        # 关键修复：不要对 local_rank 取模，否则会掩盖错误并导致多个 rank 用同一块 GPU
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.LOCAL_RANK],
            output_device=cfg.LOCAL_RANK,
            find_unused_parameters=True
        )

    logger.info(
        f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATASET,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Only evaluate the last args.num_epochs_to_eval epochs
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
