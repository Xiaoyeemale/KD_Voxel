import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['baseline', 'sparse', 'sdr', 'sdr_mse', 'sdr_nokd', 'sdr_noms'],
                        help='一条命令切换实验组：\n'
                             '  baseline = 基线 Voxel R-CNN（稠密 64 线）\n'
                             '  sparse   = 基线模型 + 均匀抽线到 25%（等点数稀疏对照）\n'
                             '  sdr      = 完整 SDR-Voxel R-CNN（③+①+②）\n'
                             '  sdr_mse  = sdr 但蒸馏换成归一化 L2（② 的对照组）\n'
                             '  sdr_nokd = sdr 但关闭蒸馏（①+③ 的净贡献）\n'
                             '  sdr_noms = sdr 但 VFE 换回 MeanVFE（① 的净贡献）')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
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
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')

    parser.add_argument('--eval_interval', type=int, default=0,
                        help='run validation every N epochs while training (0 = disabled). '
                             'Also writes training_log.csv / val_metrics.csv into the exp dir')
    parser.add_argument('--extra_metrics', nargs='+', default=None,
                        help='extra eval metrics on KITTI: ap50_95 (AP averaged over IoU 0.5:0.95), '
                             'prf (3D P/R/F1 at IoU 0.7). Defaults to MODEL.POST_PROCESSING.EXTRA_METRICS')
    parser.add_argument('--no_plot_after_train', action='store_true', default=False,
                        help='skip the automatic curve plots written after training '
                             '(training_curves.png / sdr_diagnostics.png / sdr_summary.json)')
    parser.add_argument('--plot_smooth', type=int, default=20,
                        help='moving-average window (iterations) used by the post-training plots')
    parser.add_argument('--plot_skip_head', type=int, default=0,
                        help='ignore the first N iterations in sdr_summary.json (ramp-up warm-up)')
    parser.add_argument('--no_vis_after_train', action='store_true', default=False,
                        help='skip the random-frame detection images written after training '
                             '(detections_overview.png + det_<frame_id>.png)')
    parser.add_argument('--vis_num', type=int, default=3,
                        help='how many random val frames to draw after training (default 3). '
                             'Only used the first time: the chosen indices are then frozen '
                             'in --vis_frames_file and reused by every later run')
    parser.add_argument('--vis_seed', type=int, default=0,
                        help='seed of the one-off random frame sampling')
    parser.add_argument('--vis_frames', type=int, nargs='+', default=None,
                        help='explicit frame indices to draw, e.g. --vis_frames 192 238 318 '
                             '(overrides and rewrites the frozen list)')
    parser.add_argument('--vis_frames_file', type=str, default=None,
                        help='where the frozen frame indices live '
                             '(default: <repo>/output/vis_frames.json, shared by all presets)')
    parser.add_argument('--reset_frames', action='store_true', default=False,
                        help='discard the frozen frame list and sample --vis_num new frames '
                             '(otherwise every run reuses the same frames so results are comparable)')
    

    args = parser.parse_args()

    if args.preset is not None:
        if args.cfg_file is not None:
            raise ValueError('--preset 与 --cfg_file 只能二选一')
        # (配置文件, 实验标签后缀, 覆写项)
        # 覆写项只改"消融开关"，不改任何超参；有后缀时输出目录也会分开，
        # 避免同一份 yaml 的几组实验互相覆盖 ckpt。
        presets = {
            'baseline': ('tools/cfgs/kitti_models/voxel_rcnn_car.yaml', None, {}),
            'sparse': ('tools/cfgs/kitti_models/voxel_rcnn_car_sparse.yaml', None, {}),
            'sdr': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', None, {}),
            'sdr_mse': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', 'sdr_mse',
                        {'MODEL.KD.DIST_TYPE': 'l2n'}),
            'sdr_nokd': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', 'sdr_nokd',
                         {'MODEL.KD.ENABLE': False}),
            'sdr_noms': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', 'sdr_noms',
                         {'MODEL.VFE.NAME': 'MeanVFE'}),
        }
        cfg_file, tag_suffix, preset_overrides = presets[args.preset]
        args.cfg_file = cfg_file
        if not os.path.isfile(args.cfg_file):
            raise FileNotFoundError(
                '找不到 %s —— 请在项目根目录（KD_Voxel/）下执行该命令' % args.cfg_file)
    else:
        tag_suffix, preset_overrides = None, {}
    assert args.cfg_file is not None, '必须指定 --preset 或 --cfg_file'

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    if tag_suffix:
        cfg.TAG = cfg.TAG + '_' + tag_suffix
    for key, val in preset_overrides.items():
        _set_cfg_value(cfg, key, val)
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def _set_cfg_value(cfg, dotted_key, val):
    """按 'MODEL.KD.ENABLE' 这种点分路径写配置（预设用，避免重复整份 yaml）。"""
    keys = dotted_key.split('.')
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = val


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
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
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    eval_loader = None
    if args.eval_interval > 0:
        logger.info('----------- Create validation dataloader (validate every %d epochs) -----------'
                    % args.eval_interval)
        _, eval_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False, workers=args.workers,
            logger=logger,
            training=False,
        )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
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
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
        eval_loader=eval_loader,
        eval_interval=args.eval_interval,
        eval_output_dir=output_dir / 'eval' / 'eval_during_train',
        metrics_dir=output_dir,
        iter_log_path=output_dir / 'iter_log.csv',
        eval_args=args,
        dist_train=dist_train,
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # ------------------------------------------------------------------ #
    #  训练结束：随机抽 K 帧画检测效果图（BEV 俯视图，不需要 open3d/显示器）
    #  用的是 test_set（training=False），因此学生输入也是 EIP-BS 之后的稀疏
    #  点云 —— 与部署/评测口径一致，而不是"训练时的稠密输入"。
    #
    #  帧号只随机一次：首次跑完把 indices 写进 <repo>/output/vis_frames.json，
    #  之后 baseline / sparse / sdr / sdr_mse 各组训练都读这份清单 —— 这样
    #  不同实验组出的是同一批场景，图可以直接并排比较。想换一批就删清单或
    #  用 --vis_frames 显式指定。
    # ------------------------------------------------------------------ #
    if not args.no_vis_after_train and cfg.LOCAL_RANK == 0:
        try:
            from visualize_detections import generate as generate_vis
            vis_dir = output_dir / 'visualizations'
            vis_manifest = args.vis_frames_file or str(
                cfg.ROOT_DIR / 'output' / 'vis_frames.json')
            if args.reset_frames and os.path.isfile(vis_manifest):
                os.remove(vis_manifest)
                logger.info('已删除固定帧清单 %s，将重新随机抽 %d 帧'
                            % (vis_manifest, args.vis_num))
            vis_files = generate_vis(
                model=model.module if dist_train else model,
                dataset=test_set,
                out_dir=str(vis_dir),
                class_names=cfg.CLASS_NAMES,
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                num_samples=args.vis_num,
                seed=args.vis_seed,
                frames=args.vis_frames,
                manifest=vis_manifest,
                logger=logger,
            )
            if vis_files:
                logger.info('**********************Detection images**********************')
                logger.info('  frames fixed in %s' % vis_manifest)
                logger.info('  images  -> %s' % vis_dir)
            else:
                logger.info('未生成检测效果图（matplotlib 缺失或数据集为空）')
        except Exception as e:                              # 出图失败不影响训练结果
            logger.info('检测效果图生成失败（不影响训练结果）：%s: %s' % (type(e).__name__, e))

    # ------------------------------------------------------------------ #
    #  训练结束自动出图：training_curves.png +（SDR 组）sdr_diagnostics.png
    #  + sdr_summary.json。数据源是训练期间一直写着的 iter_log.csv /
    #  training_log.csv / val_metrics.csv，所以即便这里失败，也能事后手动跑
    #  `python tools/plot_curves.py --exp_dir <输出目录>` 补出来。
    # ------------------------------------------------------------------ #
    if not args.no_plot_after_train and cfg.LOCAL_RANK == 0:
        try:
            from plot_curves import generate as generate_plots
            files = generate_plots(
                str(output_dir),
                class_name=cfg.CLASS_NAMES[0],
                smooth=args.plot_smooth,
                skip_head=args.plot_skip_head,
                verbose=False,
            )
            if files:
                logger.info('**********************Training plots**********************')
                for p in files:
                    logger.info('  %s' % p)
            else:
                logger.info('未生成曲线：%s 下没有 csv 记录' % output_dir)
        except Exception as e:                              # 出图失败不影响主流程
            logger.info('曲线生成失败（不影响训练结果）：%s: %s' % (type(e).__name__, e))


if __name__ == '__main__':
    main()
