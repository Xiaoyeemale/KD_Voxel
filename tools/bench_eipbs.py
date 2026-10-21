# -*- coding: utf-8 -*-
"""EIP-BS 采样策略真实数据实测（纯 numpy，不需要 torch / spconv）

为什么需要这个脚本
    文档 3.3 节给的"前景保留率 0.76"是合成场景数字，而"按平均分丢线"在真实 KITTI
    帧上经常劣于"均匀抽线"。这个脚本把"哪种策略真的保住了车"这件事变成一张可复现
    的表，用来回答三个审稿级问题：
      1. 保住了多少点？（keep_ratio —— 与基线等密度才可比）
      2. 保住了哪些点？（fg_keep / car_cov —— 车还剩不剩得下）
      3. 稀疏模式和真实低线束雷达像不像？（beam_gap —— 仰角均匀性）

指标
    keep_ratio   整体保留率
    fg_keep      车框内点的保留率
    car_cov30/50 单车保留率 >=30% / >=50% 的车占比（可检测性代理）
    car_low5     保留点数 <5 的车占比（基本不可检测）
    near_cov50   30m 以内车的 car_cov50
    gap_mean/max 存活线束相邻间隔（以线束为单位，均匀抽线恒为 4）

用法
    python tools/bench_eipbs.py                        # 40 帧，全部策略
    python tools/bench_eipbs.py --frames 20 --csv out.csv
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcdet.utils.eipbs_utils import EIPBS  # noqa: E402

POINT_CLOUD_RANGE = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
SRC_BEAMS, DST_BEAMS = 64, 16


# --------------------------------------------------------------------------- #
#  KITTI 读取
# --------------------------------------------------------------------------- #
def _read_calib(path):
    """读 R0_rect 与 Tr_velo_to_cam，返回 4x4 的 T_velo2cam。"""
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            data[k.strip()] = np.array([float(x) for x in v.split()], dtype=np.float64)
    r0 = data['R0_rect'].reshape(3, 3)
    tr = data['Tr_velo_to_cam'].reshape(3, 4)
    t = np.eye(4)
    t[:3, :4] = tr
    rect = np.eye(4)
    rect[:3, :3] = r0
    return rect @ t


def _read_gt_cars(path):
    """返回 (M, 7) [xc, yc, zc, h, w, l, ry]（相机坐标系，**框中心**）。

    KITTI 标注的 (x,y,z) 是框的**底面中心**，这里换算成中心：yc = y - h/2。
    （用底面当中心会让判定框整体上移 h/2，只能框到路面、漏掉车身。）
    """
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            p = line.split()
            if len(p) < 15 or p[0] != 'Car':
                continue
            if float(p[1]) > 0.3 or int(float(p[2])) > 2:   # 截断/遮挡（≈ Moderate 口径）
                continue
            h, w, l, x, y, z, ry = [float(v) for v in p[8:15]]
            boxes.append([x, y - h / 2.0, z, h, w, l, ry])
    return np.array(boxes, dtype=np.float64).reshape(-1, 7)


def _points_in_cam_boxes(pts_velo, boxes_cam, T_velo2cam):
    """(N,3) 激光点是否落在任一相机系 3D 框内 -> (N, M) bool。

    只在相机系里判，避免把"相机框转激光框"的欧拉角约定搞错：
    p_cam = T_velo2cam · p_velo，再按 -ry 绕 y 轴转回轴对齐。
    """
    n, m = len(pts_velo), len(boxes_cam)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=bool)
    hom = np.concatenate([pts_velo, np.ones((n, 1))], axis=1)
    p_cam = hom @ T_velo2cam.T                                       # (N,4)
    rel = p_cam[:, None, :3] - boxes_cam[None, :, :3]                # (N,M,3)
    ry = boxes_cam[None, :, 6]
    c, s = np.cos(-ry), np.sin(-ry)                                  # Ry(-ry)
    lx = rel[..., 0] * c + rel[..., 2] * s
    lz = -rel[..., 0] * s + rel[..., 2] * c
    ly = rel[..., 1]
    h, w, l = boxes_cam[None, :, 3], boxes_cam[None, :, 4], boxes_cam[None, :, 5]
    inside = (np.abs(lx) <= w / 2) & (np.abs(ly) <= h / 2) & (np.abs(lz) <= l / 2)
    return inside


def _cam_boxes_to_lidar(boxes_cam, T_velo2cam):
    """相机系框（中心口径）-> 激光系 (M, 8) [x,y,z,dx,dy,dz,heading,label=1]。

    EIP-BS 的类别先验与线束分配都在激光系里做，所以必须真实转一次；
    heading 由"框的长度轴"在激光系的朝向给出（Ry(ry)·ẑ_cam 转过去再取 atan2）。
    """
    if len(boxes_cam) == 0:
        return np.zeros((0, 8), dtype=np.float64)
    T_cam2velo = np.linalg.inv(T_velo2cam)
    out = np.zeros((len(boxes_cam), 8), dtype=np.float64)
    for i, (xc, yc, zc, h, w, l, ry) in enumerate(boxes_cam):
        c = T_cam2velo @ np.array([xc, yc, zc, 1.0])
        axis = np.array([np.sin(ry), 0.0, np.cos(ry)])           # Ry(ry)·[0,0,1]
        v = T_velo2cam[:3, :3].T @ axis                          # 旋转部分转回激光系
        yaw = np.arctan2(v[1], v[0])
        out[i] = [c[0], c[1], c[2], l, w, h, yaw, 1.0]
    return out


def _load_frame(idx, root):
    pcd = np.fromfile(os.path.join(root, 'training', 'velodyne', '%06d.bin' % idx),
                      dtype=np.float32).reshape(-1, 4)
    calib = _read_calib(os.path.join(root, 'training', 'calib', '%06d.txt' % idx))
    boxes = _read_gt_cars(os.path.join(root, 'training', 'label_2', '%06d.txt' % idx))

    # 与 OpenPCDet 数据管线一致：先裁到点云范围（range 的 x 起点是 0，即只保留前向）
    pcr = np.array(POINT_CLOUD_RANGE)
    m = np.all((pcd[:, :3] >= pcr[:3]) & (pcd[:, :3] <= pcr[3:]), axis=1)
    pcd = pcd[m]

    keep_box = []
    T_cam2velo = np.linalg.inv(calib)
    for b in boxes:
        hom = np.array([[b[0], b[1], b[2], 1.0]]) @ T_cam2velo.T
        if pcr[0] <= hom[0, 0] <= pcr[3] and pcr[1] <= hom[0, 1] <= pcr[4]:
            keep_box.append(b)
    boxes = np.array(keep_box, dtype=np.float64).reshape(-1, 7)
    return pcd, boxes, calib


# --------------------------------------------------------------------------- #
#  指标
# --------------------------------------------------------------------------- #
def _beam_gaps(beam_ids_all, keep_mask, num_beams=SRC_BEAMS, thresh=0.5):
    """"存活线束"在 64 条网格上的间隔（含两端空缺）。均匀抽线恒为 4。

    注意不能用"这条线束还剩不剩点"来判存活：EIP-BS 的救援机制会给每条被丢线束
    留 20% 的高分点，于是 64 条线束条条都有点，按"有点"算间隔永远是 1。这里改用
    "保留比例 >= thresh 的线束"作为存活线束 —— 被丢线束只留了 20%，不算存活。
    """
    cin = np.bincount(beam_ids_all, minlength=num_beams).astype(np.float64)
    cout = np.bincount(beam_ids_all[keep_mask], minlength=num_beams).astype(np.float64)
    ratio = cout / np.maximum(cin, 1.0)
    alive = ratio >= thresh
    if alive.sum() < 2:                      # 退化：全算
        alive = cout > 0
    pos = np.where(alive)[0]
    if pos.size == 0:
        return 0.0, 0.0
    gaps = list(np.diff(pos)) + [pos[0], num_beams - 1 - pos[-1]]
    return float(np.mean(gaps)), float(np.max(gaps))


def evaluate(sampler, frames, tag, use_gt=True):
    acc = dict(keep=0.0, fg=0.0, cov30=[], cov50=[], low5=[], near50=[],
               gap_mean=[], gap_max=[], ms=[], fg_pts_in=[], fg_pts_out=[])
    for pcd, boxes, calib in frames:
        t0 = time.time()
        gt_lidar = _cam_boxes_to_lidar(boxes, calib)
        kept = sampler(pcd, (gt_lidar if len(gt_lidar) else None) if use_gt else None)
        acc['ms'].append((time.time() - t0) * 1000.0)
        mask_kept = sampler.last_keep

        acc['keep'] += len(kept) / max(len(pcd), 1)
        inside = _points_in_cam_boxes(pcd[:, :3], boxes, calib)
        if inside.size and inside.shape[1] > 0:
            tot_in = inside.sum(axis=0)
            kept_in = (inside & mask_kept[:, None]).sum(axis=0)
            ok = tot_in > 0
            if ok.any():
                acc['fg'] += kept_in.sum() / max(tot_in.sum(), 1)
                cov = kept_in[ok] / tot_in[ok]
                acc['cov30'].append(cov >= 0.30)
                acc['cov50'].append(cov >= 0.50)
                acc['low5'].append(kept_in[ok] < 5)
                dist = np.linalg.norm(boxes[ok][:, :2], axis=1)
                near = dist <= 30.0
                if near.any():
                    acc['near50'].append(cov[near] >= 0.50)
                acc['fg_pts_in'].append(tot_in[ok].mean())
                acc['fg_pts_out'].append(kept_in[ok].mean())

        bid = sampler.assign_beams(pcd, SRC_BEAMS)
        gm, gx = _beam_gaps(bid, mask_kept)
        acc['gap_mean'].append(gm)
        acc['gap_max'].append(gx)

    n = len(frames)
    cat = lambda k: (np.concatenate(acc[k]) if len(acc[k]) else np.zeros(0, bool))

    def _mean_bool(k):
        a = cat(k)
        return float(a.mean()) if a.size else float('nan')

    return {
        'tag': tag,
        'keep_ratio': acc['keep'] / n,
        'fg_keep': acc['fg'] / n,
        'car_cov30': _mean_bool('cov30'),
        'car_cov50': _mean_bool('cov50'),
        'car_low5': _mean_bool('low5'),
        'near_cov50': _mean_bool('near50'),
        'gap_mean': float(np.mean(acc['gap_mean'])),
        'gap_max': float(np.mean(acc['gap_max'])),
        'ms': float(np.mean(acc['ms'])),
        'fg_pts_in': float(np.mean(acc['fg_pts_in'])) if acc['fg_pts_in'] else 0.0,
        'fg_pts_out': float(np.mean(acc['fg_pts_out'])) if acc['fg_pts_out'] else 0.0,
    }


def build_strategies():
    base = dict(src_beams=SRC_BEAMS, dst_beams=DST_BEAMS, stages=[64, 48, 32, 16],
                curvature_k=8, curvature_cell_size=0.5, density_radius=1.0,
                class_priority=(0.6, 1.0, 1.0), near_margin=0.5,
                use_numba=False, seed=0)
    s = []
    s.append(('uniform_beam', '均匀抽 16 线（真实低线束代理）', EIPBS(mode='uniform_beam', **base)))
    s.append(('random', '随机保留 25%（最弱对照）', EIPBS(mode='random', **base)))
    # 不做等点数（保留救援，实际 39%~45% 点）—— 只用来复现"文档原式"的口径
    for score in ('mean', 'topk'):
        s.append(('flat_%s' % score, '不分层 %s（无等点数）' % score,
                  EIPBS(mode='importance', stratified=False, beam_score=score, **base)))
        s.append(('strat_%s' % score, '分层 %s（无等点数，靠救援）' % score,
                  EIPBS(mode='importance', stratified=True, beam_score=score, **base)))
    # 等点数 25%：全局按重要性花预算（importance sampling）
    for score in ('mean', 'topk', 'max', 'count'):
        s.append(('strat_%s_eq' % score, '分层 %s + 等点数 25%%（全局花预算）' % score,
                  EIPBS(mode='importance', stratified=True, beam_score=score,
                        target_keep_ratio=0.25, align_keep_structure=False, **base)))
    # 等点数 25%：不分层，纯逐点重要性采样（检验"线束"这一步到底有没有用）
    for score in ('mean', 'topk'):
        s.append(('flat_%s_eq' % score, '不分层 %s + 等点数 25%%（纯逐点重要性）' % score,
                  EIPBS(mode='importance', stratified=False, beam_score=score,
                        target_keep_ratio=0.25, align_keep_structure=False, **base)))
    # 等点数 25%：保线束结构（近似真 16 线）
    for score in ('mean', 'topk'):
        s.append(('strat_%s_beam' % score, '分层 %s + 等点数 25%%（保线束结构）' % score,
                  EIPBS(mode='importance', stratified=True, beam_score=score,
                        target_keep_ratio=0.25, align_keep_structure=True, **base)))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/kitti')
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--only', default=None, help='只跑 tag 含该子串的策略')
    ap.add_argument('--no_gt', action='store_true',
                    help='按**测试期**行为打分（不给 GT 框，类别先验失效）。'
                         '用来量化"训练靠 GT 挑点、测试没有 GT"这个分布偏移有多大')
    args = ap.parse_args()

    ids = sorted(int(os.path.basename(p)[:-4])
                 for p in glob.glob(os.path.join(args.data_root, 'training', 'velodyne', '*.bin')))
    if not ids:
        raise SystemExit('没找到 %s/training/velodyne/*.bin' % args.data_root)
    ids = ids[:args.frames]
    print('读取 %d 帧真实 KITTI ...' % len(ids))
    frames = [_load_frame(i, args.data_root) for i in ids]
    n_box = sum(len(b) for _, b, _ in frames)
    print('共 %d 个 Car 框，平均每帧 %.1f 个；平均 %.0f 点/帧'
          % (n_box, n_box / len(frames), np.mean([len(p) for p, _, _ in frames])))

    rows = []
    for tag, desc, sampler in build_strategies():
        if args.only and args.only not in tag:
            continue
        r = evaluate(sampler, frames, tag, use_gt=not args.no_gt)
        r['desc'] = desc
        rows.append(r)
        print('  %-16s keep=%.3f fg=%.3f cov50=%.3f low5=%.3f gap_max=%.1f %5.1fms'
              % (tag, r['keep_ratio'], r['fg_keep'], r['car_cov50'],
                 r['car_low5'], r['gap_max'], r['ms']))

    hdr = ['策略', '保留率', '前景保留', '车≥30%', '车≥50%', '车<5点', '近车≥50%',
           '线束间隔均值', '最大间隔', 'ms/帧']
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for r in rows:
        lines.append('| `%s` | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.1f | %.1f |'
                     % (r['tag'], r['keep_ratio'], r['fg_keep'], r['car_cov30'],
                        r['car_cov50'], r['car_low5'], r['near_cov50'],
                        r['gap_mean'], r['gap_max'], r['ms']))
    table = '\n'.join(lines)
    print('\n' + table)

    if args.csv:
        keys = ['tag', 'desc', 'keep_ratio', 'fg_keep', 'car_cov30', 'car_cov50',
                'car_low5', 'near_cov50', 'gap_mean', 'gap_max', 'ms', 'fg_pts_in', 'fg_pts_out']
        with open(args.csv, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r[k]) for k in keys) + '\n')
        print('\n已写出 %s' % args.csv)


if __name__ == '__main__':
    main()
