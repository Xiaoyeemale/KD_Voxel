# -*- coding: utf-8 -*-
"""随机抽 K 帧把检测结果画成 PNG（BEV 俯视图，不需要 open3d / 显示器）。

为什么不用 tools/demo.py：
    demo.py 走 open3d（或 mayavi）交互窗口，`draw_scenes` 不落盘，服务器上
    （无 X11 / 无 GPU 显示）根本跑不起来；而训练结束要的是"能直接看图、能贴进
    报告"的静态文件。这里只用 matplotlib + Agg 后端，保存 PNG 即可。

每帧画两张子图，共用同一批点云与坐标范围：
    左 = 学生实际输入的点云（EIP-BS 之后的 25% 稀疏点）+ GT 框
    右 = 同一批点云 + 模型预测框（按 score 深浅上色）
因此左右对比能同时看出两件事：稀疏输入还剩多少、检测框跟得准不准。

产出（默认写到 <exp_dir>/visualizations/）：
    det_<frame_id>.png        单帧大图（1x2）
    detections_overview.png   K 帧总览（2xK，上 GT 下 Pred）
    detections_summary.json   抽到哪几帧、各帧点数/保留率/预测数/最高分

用法：
    # 训练结束由 tools/train.py 自动调用（--no_vis_after_train 可关闭）
    # 事后单独出图：
    python tools/visualize_detections.py \
        --preset sdr \
        --ckpt output/cfgs/kitti_models/sdr_voxel_rcnn_car/default/ckpt/checkpoint_epoch_80.pth \
        --num_samples 3 --seed 0
"""

import argparse
import json
import os
import sys

import numpy as np

import _init_path  # noqa: F401  （把仓库根加入 sys.path）

# 注意：torch / pcdet 都在函数内部按需 import。
# 这样本文件可以在"没有 GPU / 没有 spconv"的机器上被单独 import，
# 用假数据验证画图逻辑（见 tests 里的自测用例）。

# 只画英文标签：matplotlib 默认字体没有中文字形，中文会变成一排方块
GT_COLOR = '#d62728'
PRED_COLOR = '#1f6fd6'


# --------------------------------------------------------------------------- #
#  几何与绘制工具
# --------------------------------------------------------------------------- #
def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'detach'):                 # torch.Tensor（不 import torch 也能判）
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _box_corners(box):
    """(7,) 3D 框 -> BEV 四个角点（按逆时针闭合）。"""
    x, y, _z, dx, dy, _dz, heading = [float(v) for v in box[:7]]
    c, s = np.cos(heading), np.sin(heading)
    lx = np.array([dx, dx, -dx, -dx], dtype=np.float64) / 2.0
    ly = np.array([dy, -dy, -dy, dy], dtype=np.float64) / 2.0
    return x + lx * c - ly * s, y + lx * s + ly * c


def _draw_boxes(ax, boxes, color, scores=None, lw=1.2, label=None):
    """在 ax 上画一组 BEV 框；scores 给出时按分数调深浅，并顺带画朝向短线。"""
    if boxes is None or len(boxes) == 0:
        return
    for i, box in enumerate(boxes):
        px, py = _box_corners(box)
        alpha = 0.95
        if scores is not None:
            s = float(np.clip(scores[i], 0.0, 1.0))
            alpha = 0.35 + 0.65 * s
        ax.plot(np.append(px, px[0]), np.append(py, py[0]),
                color=color, linewidth=lw, alpha=alpha,
                label=label if i == 0 else None)
        # 朝向：从中心沿 heading 画半条边的短线（检查方向有没有翻 180°）
        x, y, _z, dx, _dy, _dz, heading = [float(v) for v in box[:7]]
        ax.plot([x, x + np.cos(heading) * dx * 0.5],
                [y, y + np.sin(heading) * dx * 0.5],
                color=color, linewidth=0.8, alpha=alpha)


def _draw_points(ax, pts, size=1.0):
    """点云俯视图，按高度上色。"""
    if pts is None or len(pts) == 0:
        return
    z = pts[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    if hi - lo < 1e-3:
        lo, hi = lo - 0.5, lo + 0.5
    ax.scatter(pts[:, 0], pts[:, 1], c=z, cmap='viridis', s=size,
               linewidths=0, vmin=lo, vmax=hi, alpha=0.9, rasterized=True)


def _setup_ax(ax, pcr, title):
    ax.set_xlim(pcr[0], pcr[3])
    ax.set_ylim(pcr[1], pcr[4])
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#f7f7f7')
    ax.grid(True, color='white', linewidth=0.6)
    ax.set_xlabel('x (m, forward)', fontsize=8)
    ax.set_ylabel('y (m, left)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def _draw_pair(axes, frame, pcr):
    """在同一对子图上画 GT / Pred 两个面板。"""
    ax_gt, ax_pred = axes
    n_gt = 0 if frame['gt_boxes'] is None else len(frame['gt_boxes'])
    n_pr = 0 if frame['pred_boxes'] is None else len(frame['pred_boxes'])

    _draw_points(ax_gt, frame['points'])
    _draw_boxes(ax_gt, frame['gt_boxes'], GT_COLOR, lw=1.3, label='GT')
    _setup_ax(ax_gt, pcr, 'GT  (%d boxes)' % n_gt)
    if n_gt:
        ax_gt.legend(loc='upper right', fontsize=7, frameon=False)

    _draw_points(ax_pred, frame['points'])
    _draw_boxes(ax_pred, frame['pred_boxes'], PRED_COLOR,
                scores=frame['pred_scores'], lw=1.3, label='Pred')
    top = ('%.2f' % float(np.max(frame['pred_scores']))) if n_pr else '-'
    _setup_ax(ax_pred, pcr, 'Pred  (%d boxes, score>=%s, max %s)'
              % (n_pr, frame['threshold'], top))
    if n_pr:
        ax_pred.legend(loc='upper right', fontsize=7, frameon=False)


def _zoom_window(frame, pcr, margin=8.0, min_size=30.0):
    """以所有框的并集为中心取一个裁剪窗口；没有框就取场景中心。"""
    boxes = [b for b in (frame['gt_boxes'], frame['pred_boxes']) if b is not None and len(b)]
    if boxes:
        allb = np.concatenate(boxes, axis=0)
        cx = float(np.median(allb[:, 0]))
        cy = float(np.median(allb[:, 1]))
        span = max(float(np.ptp(allb[:, 0])), float(np.ptp(allb[:, 1]))) + 2 * margin
    else:
        cx, cy = (pcr[0] + pcr[3]) / 2.0, (pcr[1] + pcr[4]) / 2.0
        span = 50.0
    span = max(span, min_size)
    half_x = span / 2.0
    half_y = span / 2.0 * 0.8
    x0 = min(max(cx - half_x, pcr[0]), pcr[3] - 2 * half_x)
    y0 = min(max(cy - half_y, pcr[1]), pcr[4] - 2 * half_y)
    x1 = min(x0 + 2 * half_x, pcr[3])
    y1 = min(y0 + 2 * half_y, pcr[4])
    return x0, x1, y0, y1


def _draw_zoom(ax, frame, pcr):
    """放大叠加面板：GT(红) 与 Pred(蓝) 画在同一批点上，标出 score。"""
    x0, x1, y0, y1 = _zoom_window(frame, pcr)
    _draw_points(ax, frame['points'], size=1.6)
    _draw_boxes(ax, frame['gt_boxes'], GT_COLOR, lw=1.6, label='GT')
    _draw_boxes(ax, frame['pred_boxes'], PRED_COLOR, scores=frame['pred_scores'],
                lw=1.6, label='Pred')
    # 预测分数（只在放大面板里标，避免全图时糊成一团）
    if frame['pred_boxes'] is not None and len(frame['pred_boxes']):
        for b, s in zip(frame['pred_boxes'], frame['pred_scores']):
            ax.text(b[0], b[1] + b[4] / 2 + 0.6, '%.2f' % float(s),
                    fontsize=6.5, color=PRED_COLOR, ha='center', va='bottom')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#f7f7f7')
    ax.grid(True, color='white', linewidth=0.6)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title('Overlay (zoom %.0f x %.0f m)' % (x1 - x0, y1 - y0), fontsize=9)
    if (frame['gt_boxes'] is not None and len(frame['gt_boxes'])) or \
            (frame['pred_boxes'] is not None and len(frame['pred_boxes'])):
        ax.legend(loc='upper right', fontsize=7, frameon=False)


def _frame_suptitle(frame):
    return ('frame %s   |   student input %d pts (keep %.3f, beams %.0f)'
            % (frame['frame_id'], len(frame['points']), frame['keep_ratio'],
               frame['beams']))


# --------------------------------------------------------------------------- #
#  推理：对指定帧跑一次 eval 前向，返回画图用的数组
# --------------------------------------------------------------------------- #
def _collect_frame(model, dataset, index, score_thresh, max_boxes, class_names):
    from pcdet.models import load_data_to_gpu
    import torch                 # 只有真正推理时才需要

    data_dict = dataset[int(index)]
    batch = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch)

    with torch.no_grad():
        pred_dicts, _ = model(batch)
    pred = pred_dicts[0]

    # 学生分支的真实输入（EIP-BS 之后）；没有 *_kd 键时退回主点云
    pts = batch.get('points_kd', batch.get('points'))
    pts = _as_numpy(pts)
    if pts is not None and pts.shape[1] >= 5:
        pts = pts[pts[:, 0] == 0][:, 1:4]          # 去掉 collate 补的 batch 列
    elif pts is not None:
        pts = pts[:, :3]

    gt = _as_numpy(batch.get('gt_boxes'))
    if gt is not None:
        gt = gt.reshape(-1, gt.shape[-1])
        gt = gt[np.abs(gt).sum(axis=1) > 1e-6]
        gt = gt[:, :7]

    boxes = _as_numpy(pred['pred_boxes'])
    scores = _as_numpy(pred['pred_scores'])
    keep = scores >= score_thresh
    boxes, scores = boxes[keep], scores[keep]
    order = np.argsort(-scores)[:max_boxes]
    boxes, scores = boxes[order], scores[order]

    stats = _as_numpy(batch.get('eipbs_stats'))
    keep_ratio, beams = 0.0, 0.0
    if stats is not None:
        s = stats.reshape(-1, stats.shape[-1])[0]
        keep_ratio = float(s[0])
        beams = float(s[2]) if s.shape[0] > 2 else 0.0

    return {
        'frame_id': str(batch['frame_id'][0]),
        'points': pts,
        'gt_boxes': gt,
        'pred_boxes': boxes,
        'pred_scores': scores,
        'keep_ratio': keep_ratio,
        'beams': beams,
        'threshold': ('%.2f' % score_thresh),
    }


# --------------------------------------------------------------------------- #
#  帧号的"一次性随机 + 永久固定"
#
#  需求：随机只是**第一次**用来挑几张图；挑完之后要写进清单，以后每次训练
#  （baseline / sparse / sdr / sdr_mse ...）都用同一批帧出图，否则各组看到的
#  是不同场景，"改进模型框得更准"就成了看图说话。
#
#  三级优先：
#     1) 命令行显式给了 --frames / --vis_frames  -> 用它，并覆盖清单
#     2) 清单文件已存在                          -> 直接读，不重新随机
#     3) 两者都没有                              -> 用 seed 随机抽，写入清单
# --------------------------------------------------------------------------- #
DEFAULT_MANIFEST = os.path.join('output', 'vis_frames.json')


def _load_manifest(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    idx = data.get('indices')
    if not isinstance(idx, list) or not idx:
        return None
    try:
        return [int(v) for v in idx]
    except (TypeError, ValueError):
        return None


def _save_manifest(path, indices, seed, dataset_size, num_samples, frame_ids=None):
    if not path:
        return None
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {
        'indices': [int(i) for i in indices],
        'seed': int(seed),
        'dataset_size': int(dataset_size),
        'num_samples': int(len(indices)),
        'note': '首次随机抽取后固定；删除本文件才会重新随机，'
                '命令行给 --frames/--vis_frames 可直接覆盖。',
    }
    if frame_ids:
        payload['frame_ids'] = list(frame_ids)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def resolve_frames(dataset_size, num_samples=3, seed=0, frames=None,
                   manifest=None, log=None):
    """决定这次要画哪几帧，返回 (indices, source, manifest_path)。

    source ∈ {'explicit', 'manifest', 'random'}，会打进 summary.json，
    以后回看时能确认"这几张图是第一次随机抽的，还是这次手动指定的"。
    """
    log = log or (lambda m: None)

    if frames:
        idx = [int(i) for i in frames]
    else:
        idx = _load_manifest(manifest)
        if idx is not None:
            bad = [i for i in idx if i < 0 or i >= dataset_size]
            if bad:
                log('[vis] 清单里的帧号 %s 超出当前数据集范围 (size=%d)，重新随机'
                    % (bad, dataset_size))
                idx = None
            else:
                # 清单里的帧数优先于 --num_samples：既然固定了，就不该被别的组改掉
                _save_manifest(manifest, idx, seed, dataset_size, num_samples)
                return idx, 'manifest', manifest

    if idx is None:
        k = int(min(max(num_samples, 1), dataset_size))
        rng = np.random.default_rng(seed)
        idx = sorted(int(v) for v in rng.choice(dataset_size, size=k, replace=False))

    source = 'explicit' if frames else 'random'
    if source == 'random':
        log('[vis] 首次随机抽取 %d 帧并写入清单: %s  ->  %s'
            % (len(idx), idx, manifest))
    else:
        log('[vis] 使用命令行指定的帧: %s' % idx)
    _save_manifest(manifest, idx, seed, dataset_size, num_samples)
    return idx, source, manifest


# --------------------------------------------------------------------------- #
#  对外入口
# --------------------------------------------------------------------------- #
def generate(model, dataset, out_dir, class_names=('Car',), point_cloud_range=None,
             num_samples=3, seed=0, score_thresh=None, max_boxes=100,
             frames=None, manifest=None, logger=None, dpi=130, verbose=True):
    """随机（首次）抽 num_samples 帧出图，返回写出的文件路径列表。

    帧号会被写进 manifest（默认 output/vis_frames.json），下次训练直接复用：
    同一批帧、同一个比较基准。

    供 tools/train.py 在评测结束后直接调用；也可用本文件的 CLI 事后补图。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        msg = 'matplotlib is required: pip install matplotlib'
        (logger.info if logger is not None else print)(msg)
        return []

    os.makedirs(out_dir, exist_ok=True)
    n_total = len(dataset)
    log = logger.info if logger is not None else print
    indices, frame_source, manifest_path = resolve_frames(
        n_total, num_samples=num_samples, seed=seed, frames=frames,
        manifest=manifest, log=log)

    if point_cloud_range is None:
        point_cloud_range = getattr(dataset, 'point_cloud_range', None)
    pcr = np.asarray(point_cloud_range if point_cloud_range is not None
                     else [0, -40, -3, 70.4, 40, 1], dtype=np.float64)
    if score_thresh is None:
        pp = getattr(model, 'model_cfg', {}).get('POST_PROCESSING', None)
        score_thresh = float(pp.get('SCORE_THRESH', 0.3)) if pp is not None else 0.3

    log = logger.info if logger is not None else print
    was_training = model.training
    model.eval()
    written, records, frames_data = [], [], []

    try:
        for idx in indices:
            rec = _collect_frame(model, dataset, idx, score_thresh, max_boxes, class_names)
            frames_data.append(rec)
            records.append({
                'index': int(idx),
                'frame_id': rec['frame_id'],
                'num_points': int(len(rec['points'])),
                'eipbs_keep_ratio': rec['keep_ratio'],
                'eipbs_beams': rec['beams'],
                'num_gt': 0 if rec['gt_boxes'] is None else int(len(rec['gt_boxes'])),
                'num_pred': 0 if rec['pred_boxes'] is None else int(len(rec['pred_boxes'])),
                'max_score': (float(np.max(rec['pred_scores']))
                              if rec['pred_scores'] is not None and len(rec['pred_scores'])
                              else 0.0),
            })

            fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.6),
                                     gridspec_kw={'width_ratios': [1, 1, 1.25]})
            _draw_pair(axes[:2], rec, pcr)
            _draw_zoom(axes[2], rec, pcr)
            fig.suptitle(_frame_suptitle(rec), fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            path = os.path.join(out_dir, 'det_%s.png' % rec['frame_id'])
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)
            if verbose:
                log('  [vis] %s  -> %s' % (rec['frame_id'], path))

        # 总览：上排 GT、下排 Pred，共用列
        k = len(frames_data)
        if k:
            fig, axes = plt.subplots(2, k, figsize=(6.2 * k, 10.4), squeeze=False)
            for j, rec in enumerate(frames_data):
                _draw_pair((axes[0][j], axes[1][j]), rec, pcr)
            for j, rec in enumerate(frames_data):
                axes[0][j].set_title('frame %s  |  GT (%d)'
                                     % (rec['frame_id'],
                                        0 if rec['gt_boxes'] is None else len(rec['gt_boxes'])),
                                     fontsize=9)
                axes[1][j].set_title('Pred (%d, score>=%s)'
                                     % (0 if rec['pred_boxes'] is None else len(rec['pred_boxes']),
                                        rec['threshold']), fontsize=9)
            fig.suptitle('SDR-Voxel R-CNN  ·  fixed %d frames (indices %s, %s)'
                         '  ·  student input = EIP-BS sparse cloud'
                         % (k, ','.join(str(i) for i in indices), frame_source), fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            overview = os.path.join(out_dir, 'detections_overview.png')
            fig.savefig(overview, dpi=dpi)
            plt.close(fig)
            written.append(overview)

        # 把 frame_id 回填进清单：以后看清单就知道这几号对应哪几帧
        _save_manifest(manifest_path, indices, seed, n_total, len(indices),
                       frame_ids=[r['frame_id'] for r in records])

        json_path = os.path.join(out_dir, 'detections_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'num_samples': len(records), 'seed': seed,
                       'frame_source': frame_source,          # explicit / manifest / random
                       'manifest': manifest_path,
                       'indices': [int(i) for i in indices],
                       'score_thresh': score_thresh, 'frames': records},
                      f, ensure_ascii=False, indent=2)
        written.append(json_path)
    finally:
        if was_training:
            model.train()

    if verbose:
        for p in written:
            log('  [vis] saved: %s' % p)
    return written


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_config():
    parser = argparse.ArgumentParser(description='draw random K frames with GT / prediction boxes')
    parser.add_argument('--cfg_file', type=str, default=None, help='model config yaml')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['baseline', 'sparse', 'sdr', 'sdr_mse', 'sdr_nokd', 'sdr_noms'],
                        help='与 tools/train.py 的 --preset 完全一致')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to visualize')
    parser.add_argument('--num_samples', type=int, default=3, help='how many random frames')
    parser.add_argument('--seed', type=int, default=0, help='random seed for frame sampling')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='output dir (default: <exp_dir>/visualizations)')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='override POST_PROCESSING.SCORE_THRESH')
    parser.add_argument('--max_boxes', type=int, default=100)
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='显式指定帧号（数据集索引），会覆盖并重写清单')
    parser.add_argument('--manifest', type=str, default=None,
                        help='固定帧清单路径（默认 <repo>/output/vis_frames.json）；'
                             '存在就沿用里面的帧号，不会重新随机')
    parser.add_argument('--reset_frames', action='store_true',
                        help='忽略已有清单，按 --seed 重新随机抽一批并覆盖清单')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    if args.preset is not None:
        if args.cfg_file is not None:
            raise ValueError('--preset 与 --cfg_file 只能二选一')
        args.cfg_file, _tag, overrides = {
            'baseline': ('tools/cfgs/kitti_models/voxel_rcnn_car.yaml', None, {}),
            'sparse': ('tools/cfgs/kitti_models/voxel_rcnn_car_sparse.yaml', None, {}),
            'sdr': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', None, {}),
            'sdr_mse': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', 'sdr_mse',
                        {'MODEL.KD.DIST_TYPE': 'l2n'}),
            'sdr_nokd': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', 'sdr_nokd',
                         {'MODEL.KD.ENABLE': False}),
            'sdr_noms': ('tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml', 'sdr_noms',
                         {'MODEL.VFE.NAME': 'MeanVFE'}),
        }[args.preset]
    else:
        overrides = {}
    if args.cfg_file is None or args.ckpt is None:
        parser.error('必须提供 --ckpt，以及 --preset 或 --cfg_file')
    if not os.path.isfile(args.cfg_file):
        raise FileNotFoundError('找不到 %s —— 请在项目根目录（KD_Voxel/）下执行'
                                % args.cfg_file)

    from pcdet.config import cfg, cfg_from_yaml_file
    cfg_from_yaml_file(args.cfg_file, cfg)
    for dotted, val in overrides.items():        # 消融组结构与训练时保持一致
        keys = dotted.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return args, cfg


def main():
    args, cfg = parse_config()
    from pathlib import Path
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=args.workers, logger=logger, training=False,
    )
    logger.info('test frames: %d' % len(dataset))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    exp_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / Path(args.cfg_file).stem
    out_dir = args.out_dir or str(exp_dir / 'visualizations')
    manifest = args.manifest or str(cfg.ROOT_DIR / DEFAULT_MANIFEST)
    if args.reset_frames:
        # 明确要求重抽：把清单移走，让 resolve_frames 走"首次随机"分支
        if os.path.isfile(manifest):
            os.remove(manifest)
            logger.info('已删除旧清单，重新随机: %s' % manifest)

    files = generate(
        model=model, dataset=dataset, out_dir=out_dir, class_names=cfg.CLASS_NAMES,
        point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        num_samples=args.num_samples, seed=args.seed,
        score_thresh=args.score_thresh, max_boxes=args.max_boxes,
        frames=args.frames, manifest=manifest, logger=logger,
    )
    logger.info('detection images: %d file(s) in %s' % (len(files), out_dir))
    return 0 if files else 1


if __name__ == '__main__':
    sys.exit(main())
