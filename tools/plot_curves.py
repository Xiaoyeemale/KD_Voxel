"""Plot training / validation curves from the csv files written by tools/train.py.

Three files are consumed (all on the KITTI dataset):

    <exp_dir>/iter_log.csv       epoch, iter, loss, lr, <every loss component>
                                 (one row per iteration, always written)
    <exp_dir>/training_log.csv   epoch, train_loss, lr
                                 (one row per epoch)
    <exp_dir>/val_metrics.csv    epoch, all validation metrics (every N epochs)

Outputs, all written next to the csv files:

    training_curves.png     loss / lr / AP / P-R-F1 / loss components   (all models)
    sdr_diagnostics.png     ① 门控、② 蒸馏、③ 采样 的诊断面板          (SDR only)
    sdr_summary.json        上面每个指标的首值/末值/均值/极值            (SDR only)

Usage:
    python tools/plot_curves.py --exp_dir output/cfgs/kitti_models/voxel_rcnn_car/kitti500
    python tools/plot_curves.py --exp_dir ... --save curve.png --logy --smooth 50
    python tools/plot_curves.py --exp_dir ... --no_sdr      # 只出总图

也可以被 tools/train.py 在训练结束时直接调用（generate()）。
"""

import argparse
import csv
import json
import os
import sys

# non-loss columns of iter_log.csv
META_KEYS = {'epoch', 'iter', 'loss', 'lr'}

# --------------------------------------------------------------------------- #
#  SDR 诊断面板的分组定义： (面板标题, [列名], ylabel, 是否 log 轴)
#  列名对应 tb_dict 里的键，由 iter_log.csv 原样落盘。
# --------------------------------------------------------------------------- #
SDR_PANELS = [
    ('CG-RD loss terms', ['cgrd_loss_rod', 'cgrd_loss_ford', 'cgrd_loss_rho'],
     'loss', True),
    ('CG-RD lambda ramp-up / gate', ['cgrd_weight', 'cgrd_gate_eff', 'cgrd_gate_mean'],
     'value', False),
    ('MS-DGE gate entropy & per-scale weight',
     ['msdge_gate_entropy', 'msdge_gate_w_s1', 'msdge_gate_w_s2',
      'msdge_gate_w_s3', 'msdge_gate_w_s4'], 'normalised', False),
    ('CG-RD direction vs radial term', ['cgrd_theta_mean', 'cgrd_logmag_mean'],
     'mean distance', True),
    ('masks & EIP-BS sampling ratio', ['cgrd_roi_ratio', 'cgrd_fore_ratio',
                                'eipbs_keep_ratio', 'eipbs_fg_keep_ratio'],
     'ratio', False),
    ('EIP-BS beams kept', ['eipbs_beams'], 'beams', False),
]


def read_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def col(rows, key):
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key, '')))
        except (TypeError, ValueError):
            out.append(float('nan'))
    return out


def has_values(values):
    """True if the column has at least one finite non-zero value."""
    return any(v == v and abs(v) > 0 for v in values)  # nan-safe


def smooth(values, window):
    """Centred moving average that ignores nans (window=1 means no smoothing)."""
    if window is None or window <= 1:
        return list(values)
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = [v for v in values[lo:hi] if v == v]
        out.append(sum(chunk) / len(chunk) if chunk else float('nan'))
    return out


def plot_group(ax, rows, keys, title, labels=None, ylabel=None, smooth_window=1,
               x_key='epoch', xlabel='epoch'):
    plotted = False
    x = col(rows, x_key) if rows else []
    for i, key in enumerate(keys):
        if key not in (rows[0].keys() if rows else []):
            continue
        vals = col(rows, key)
        if not has_values(vals):
            continue
        label = labels[i] if labels and i < len(labels) else key
        ax.plot(x, smooth(vals, smooth_window), linewidth=1.4, label=label)
        plotted = True
    ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if plotted:
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')


def plot_loss(ax, iter_rows, epoch_rows, smooth_window, logy):
    """Total training loss: per-iteration curve + smoothed curve, in real iter steps."""
    if iter_rows:
        x = col(iter_rows, 'iter')
        vals = col(iter_rows, 'loss')
        ax.plot(x, vals, color='tab:blue', linewidth=0.6, alpha=0.25, label='loss (per iter)')
        ax.plot(x, smooth(vals, smooth_window), color='tab:blue', linewidth=1.5,
                label='loss (avg %d it)' % smooth_window)
        ax.set_xlabel('iteration', fontsize=9)
    elif epoch_rows:
        ax.plot(col(epoch_rows, 'epoch'), col(epoch_rows, 'train_loss'),
                marker='o', markersize=3, linewidth=1.5, label='epoch avg loss')
        ax.set_xlabel('epoch', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_xlabel('epoch', fontsize=9)

    if logy:
        ax.set_yscale('log')
    ax.set_title('Train loss', fontsize=11)
    ax.set_ylabel('loss', fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, frameon=False)


def component_keys(iter_rows):
    """Loss component keys present in iter_log.csv, e.g. rpn_loss_cls, rcnn_loss_reg."""
    if not iter_rows:
        return []
    keys = []
    for key in iter_rows[0].keys():
        if key in META_KEYS:
            continue
        # SDR 的诊断量（cgrd_* / msdge_* / eipbs_*）有专门的九格面板，
        # 混进"Loss components"会把真正的 rpn_loss / rcnn_loss 压成一条直线。
        if key.startswith(('cgrd_', 'msdge_', 'eipbs_')):
            continue
        if not has_values(col(iter_rows, key)):
            continue
        keys.append(key)
    # group by prefix so related curves share a colour cycle: rpn_*, rcnn_*
    return keys


def plot_components(ax, iter_rows, keys, smooth_window):
    if not keys:
        ax.text(0.5, 0.5, 'no loss components\n(tb_dict was empty)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
    else:
        x = col(iter_rows, 'iter')
        totals = [k for k in keys if k.count('_') == 1]  # rpn_loss / rcnn_loss
        for key in keys:
            # aggregates drawn thicker so the stage totals stand out from the sub-terms
            lw = 1.8 if key in totals else 1.0
            ax.plot(x, smooth(col(iter_rows, key), smooth_window), linewidth=lw, label=key)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title('Loss components', fontsize=11)
    ax.set_ylabel('loss', fontsize=9)
    ax.set_xlabel('iteration', fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)


# --------------------------------------------------------------------------- #
#  单模型曲线（与原来一致）
# --------------------------------------------------------------------------- #
def plot_training_curves(iter_rows, train_rows, val_rows, cls, smooth_window, logy, dpi,
                         save_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))

    plot_loss(axes[0][0], iter_rows, train_rows, smooth_window, logy)

    lr_rows = iter_rows or train_rows
    plot_group(axes[0][1], lr_rows, ['lr'], 'Learning rate', ylabel='lr',
               x_key='iter' if iter_rows else 'epoch',
               xlabel='iteration' if iter_rows else 'epoch')

    plot_group(
        axes[1][0], val_rows,
        ['%s_3d/easy_R40' % cls, '%s_3d/moderate_R40' % cls, '%s_3d/hard_R40' % cls],
        'KITTI 3D AP (R40, IoU 0.7)', ['easy', 'moderate', 'hard'], ylabel='AP (%)'
    )

    plot_group(
        axes[1][1], val_rows,
        ['ap50_95/%s_3d/easy' % cls, 'ap50_95/%s_3d/moderate' % cls, 'ap50_95/%s_3d/hard' % cls],
        'KITTI 3D AP @ IoU 0.50:0.95 (avg)', ['easy', 'moderate', 'hard'], ylabel='AP (%)'
    )

    plot_group(
        axes[2][0], val_rows,
        ['prf/%s_3d/precision' % cls, 'prf/%s_3d/recall' % cls, 'prf/%s_3d/f1' % cls],
        'KITTI 3D P / R / F1 @ IoU 0.7', ['precision', 'recall', 'F1'], ylabel='score (0-1)'
    )

    plot_components(axes[2][1], iter_rows, component_keys(iter_rows), smooth_window)

    fig.suptitle(os.path.basename(os.path.normpath(os.path.dirname(save_path))), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
#  SDR 诊断面板：三个创新点各自的健康度
# --------------------------------------------------------------------------- #
def has_sdr_columns(iter_rows):
    if not iter_rows:
        return False
    keys = {k for k in iter_rows[0].keys() if k.startswith(('cgrd_', 'msdge_', 'eipbs_'))}
    return len(keys) >= 3


def sdr_figure(iter_rows, val_rows, cls, smooth_window, dpi, save_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.reshape(-1)
    xkey = 'iter'

    # 1) 三段蒸馏损失（log 轴：三项量级差别可能很大）
    title, keys, ylabel, logy = SDR_PANELS[0]
    plot_group(axes[0], iter_rows, keys, title, ylabel=ylabel,
               smooth_window=smooth_window, x_key=xkey, xlabel='iteration')
    if logy:
        axes[0].set_yscale('log')

    # 2) λ 爬升与门控强度
    title, keys, ylabel, _ = SDR_PANELS[1]
    plot_group(axes[1], iter_rows, keys, title, ylabel=ylabel,
               smooth_window=smooth_window, x_key=xkey, xlabel='iteration')

    # 3) MS-DGE 门控熵 + 各尺度权重
    title, keys, ylabel, _ = SDR_PANELS[2]
    plot_group(axes[2], iter_rows, keys, title, ylabel=ylabel,
               smooth_window=smooth_window, x_key=xkey, xlabel='iteration')
    axes[2].set_ylim(-0.02, 1.02)

    # 4) 方向项 vs 径向项
    title, keys, ylabel, logy = SDR_PANELS[3]
    plot_group(axes[3], iter_rows, keys, title, ylabel=ylabel,
               smooth_window=smooth_window, x_key=xkey, xlabel='iteration')
    if logy:
        axes[3].set_yscale('log')

    # 5) 掩码覆盖率 + ③ 的保留率
    title, keys, ylabel, _ = SDR_PANELS[4]
    plot_group(axes[4], iter_rows, keys, title, ylabel=ylabel,
               smooth_window=smooth_window, x_key=xkey, xlabel='iteration')
    axes[4].set_ylim(-0.02, 1.02)

    # 6) 存活线束数（近 16 = "16 线保真"口径，近 60 = 全局花预算）
    title, keys, ylabel, _ = SDR_PANELS[5]
    plot_group(axes[5], iter_rows, keys, title, ylabel=ylabel,
               smooth_window=smooth_window, x_key=xkey, xlabel='iteration')

    # 7) 蒸馏项相对检测损失（> 1 说明 KD 压过检测，应降 LOSS_WEIGHT）
    ratio = []
    for r in iter_rows:
        try:
            det = float(r.get('rpn_loss', 0.0)) + float(r.get('rcnn_loss', 0.0))
            kd = float(r.get('cgrd_loss_total', 0.0))
            ratio.append(kd / det if det > 1e-9 else float('nan'))
        except (TypeError, ValueError):
            ratio.append(float('nan'))
    if iter_rows and any(v == v for v in ratio):
        axes[6].plot(col(iter_rows, xkey), ratio, linewidth=0.8, alpha=0.5, label='per iter')
        axes[6].plot(col(iter_rows, xkey), smooth(ratio, smooth_window), linewidth=1.5,
                     label='avg %d it' % smooth_window)
        axes[6].axhline(1.0, color='tab:red', linestyle='--', linewidth=1.0,
                        label='warn: KD > det')
        axes[6].set_yscale('log')
        axes[6].legend(fontsize=8, frameon=False)
        axes[6].set_title('KD / detection loss', fontsize=11)
        axes[6].set_ylabel('ratio', fontsize=9)
    else:
        axes[6].text(0.5, 0.5, 'no data', ha='center', va='center',
                     transform=axes[6].transAxes, fontsize=10, color='gray')
    axes[6].set_xlabel('iteration', fontsize=9)
    axes[6].grid(True, alpha=0.3, linewidth=0.5)

    # 8) 单帧采样耗时（dataloader 侧真实开销，对应文档修订 15）
    plot_group(axes[7], iter_rows, ['eipbs_ms'], 'EIP-BS sampler time per frame (ms)',
               ylabel='ms', smooth_window=smooth_window, x_key=xkey, xlabel='iteration')

    # 9) 验证 AP moderate（有 eval_interval 时才有）
    plot_group(
        axes[8], val_rows,
        ['%s_3d/easy_R40' % cls, '%s_3d/moderate_R40' % cls, '%s_3d/hard_R40' % cls],
        'KITTI 3D AP (R40)', ['easy', 'moderate', 'hard'], ylabel='AP (%)'
    )

    fig.suptitle('%s  ·  SDR diagnostics (MS-DGE / CG-RD / EIP-BS)'
                 % os.path.basename(os.path.normpath(os.path.dirname(save_path))), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
#  数值汇总：写 json，便于直接抄进实验记录/论文表格
# --------------------------------------------------------------------------- #
SUMMARY_KEYS = [
    'cgrd_loss_rod', 'cgrd_loss_ford', 'cgrd_loss_rho', 'cgrd_loss_total',
    'cgrd_weight', 'cgrd_gate_eff', 'cgrd_gate_mean',
    'cgrd_theta_mean', 'cgrd_logmag_mean',
    'cgrd_roi_ratio', 'cgrd_fore_ratio', 'cgrd_to_det_ratio',
    'msdge_gate_entropy', 'msdge_gate_w_s1', 'msdge_gate_w_s2',
    'msdge_gate_w_s3', 'msdge_gate_w_s4',
    'eipbs_keep_ratio', 'eipbs_fg_keep_ratio', 'eipbs_beams', 'eipbs_ms',
]


def _stats(values):
    v = [x for x in values if x == x]        # drop nan
    if not v:
        return None
    return {'first': v[0], 'last': v[-1], 'mean': sum(v) / len(v),
            'min': min(v), 'max': max(v), 'n': len(v)}


def sdr_summary(iter_rows, val_rows, cls, skip_head=0):
    """首值/末值/均值/极值。skip_head 用来跳过 ramp-up 与 EMA 未收敛的前若干 iter。"""
    out = {'class': cls, 'num_iters': len(iter_rows),
           'skip_head': skip_head, 'metrics': {}}
    rows = iter_rows[skip_head:] if skip_head > 0 else iter_rows
    for key in SUMMARY_KEYS:
        if not iter_rows or key not in iter_rows[0]:
            continue
        st = _stats(col(rows, key))
        if st is not None:
            out['metrics'][key] = st
    ap = {}
    for split in ('easy', 'moderate', 'hard'):
        k = '%s_3d/%s_R40' % (cls, split)
        st = _stats(col(val_rows, k)) if val_rows and k in (val_rows[0] if val_rows else {}) else None
        if st:
            ap[split] = st
    if ap:
        out['val_ap_3d_r40'] = ap
    return out


def generate(exp_dir, class_name='Car', smooth=20, logy=False, dpi=150,
             save=None, make_sdr=True, skip_head=0, verbose=True):
    """读取 exp_dir 下的 csv，生成图片与 json；返回写出的文件路径列表。

    供 tools/train.py 在训练结束后调用（训练完立刻出图，不必再手动跑一遍）。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        if verbose:
            print('matplotlib is required: pip install matplotlib', file=sys.stderr)
        return []

    iter_rows, _ = read_csv(os.path.join(exp_dir, 'iter_log.csv'))
    train_rows, _ = read_csv(os.path.join(exp_dir, 'training_log.csv'))
    val_rows, _ = read_csv(os.path.join(exp_dir, 'val_metrics.csv'))

    if not iter_rows and not train_rows and not val_rows:
        if verbose:
            print('no csv found under %s' % exp_dir, file=sys.stderr)
        return []

    written = []
    main_path = save or os.path.join(exp_dir, 'training_curves.png')
    written.append(plot_training_curves(iter_rows, train_rows, val_rows, class_name,
                                        smooth, logy, dpi, main_path))

    if make_sdr and has_sdr_columns(iter_rows):
        sdr_path = os.path.join(exp_dir, 'sdr_diagnostics.png')
        written.append(sdr_figure(iter_rows, val_rows, class_name, smooth, dpi, sdr_path))

        summary = sdr_summary(iter_rows, val_rows, class_name, skip_head=skip_head)
        json_path = os.path.join(exp_dir, 'sdr_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        written.append(json_path)

    if verbose:
        for p in written:
            print('saved: %s' % p)
    return written


def main():
    parser = argparse.ArgumentParser(description='plot training curves from csv logs')
    parser.add_argument('--exp_dir', required=True,
                        help='experiment dir, e.g. output/cfgs/kitti_models/voxel_rcnn_car/kitti500')
    parser.add_argument('--save', default=None, help='output png path (default: <exp_dir>/training_curves.png)')
    parser.add_argument('--class_name', default='Car', help='class to plot (default: Car)')
    parser.add_argument('--smooth', type=int, default=20,
                        help='moving-average window for loss curves, in iterations (default: 20)')
    parser.add_argument('--logy', action='store_true', help='log scale for the loss axis')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--no_sdr', action='store_true',
                        help='skip the SDR diagnostics panel even if cgrd_*/msdge_*/eipbs_* exist')
    parser.add_argument('--skip_head', type=int, default=0,
                        help='ignore the first N iterations of the summary (ramp-up warm-up)')
    args = parser.parse_args()

    written = generate(args.exp_dir, class_name=args.class_name, smooth=args.smooth,
                       logy=args.logy, dpi=args.dpi, save=args.save,
                       make_sdr=not args.no_sdr, skip_head=args.skip_head)
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
