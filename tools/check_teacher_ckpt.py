# -*- coding: utf-8 -*-
"""教师 checkpoint 体检：在真正开跑蒸馏之前，确认这个 .pth 能当教师。

为什么需要单独查一遍
--------------------
`SDRVoxelRCNN.load_teacher_from_baseline()` 的加载策略是"按键名 + 形状匹配，
匹配不上的静默跳过"。这个策略对结构一致的 ckpt 是对的，但对结构不同的权重会
**大面积静默跳过**，结果是：教师被冻结在随机初始化上，蒸馏损失能正常下降
（学生在拟合噪声），`cgrd_*` 曲线看起来一切正常，最终 AP 却只等于稀疏基线。
这个失败模式很贵（一次 80 epoch），所以先花 2 秒查一下。

用法
----
    # 只查 ckpt 自身（不建模型、不需要 torch、不需要 spconv）
    python tools/check_teacher_ckpt.py --ckpt checkpoints/voxel_rcnn_84.54.pth

    # 连教师段配置一起查：核对教师需要哪些张量、通道是否对得上
    python tools/check_teacher_ckpt.py --ckpt ... --config tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml

判定
----
    PASS  教师侧模块的张量全部命中，可以开跑
    WARN  有可疑点（自蒸馏 ckpt / 通道对不上 / 缺模块），会逐条列出
    FAIL  ckpt 里根本没有 backbone 张量，或该文件读不出来

注意：`vfe`（MeanVFE）与 `map_to_bev`（HeightCompression）**本身没有可学习参数**，
所以 ckpt 里没有 `vfe.*` / `map_to_bev.*` 是正常的，不是缺模块。第一版脚本把这两条
当成缺失而误判 FAIL，已修正。
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# 教师分支（BEVTeacher）的子模块 → (state_dict 前缀, 是否有可学习参数)
# 与 pcdet/models/kd/teacher_branch.py 的 self.vfe / backbone_3d / map_to_bev / backbone_2d 对应。
SEGMENTS = (
    ('VFE', 'vfe.', {
        'MeanVFE': False,            # 逐体素取均值，无参数
        'PillarVFE': True,
        'DynamicPillarVFE': True,
    }),
    ('BACKBONE_3D', 'backbone_3d.', {
        'VoxelBackBone8x': True,
        'VoxelResBackBone8x(unused)': True,
    }),
    ('MAP_TO_BEV', 'map_to_bev.', {
        'HeightCompression': False,  # dense + reshape
        'PointPillarScatter': False,
    }),
    ('BACKBONE_2D', 'backbone_2d.', {
        'BaseBEVBackbone': True,
        'BaseBEVResBackbone': True,
    }),
)
HEAD_PREFIXES = ('dense_head.', 'roi_head.', 'point_head.')
SDR_PREFIXES = ('kd_criterion.', 'teacher.')


# --------------------------------------------------------------------------- #
#  读 ckpt：优先 torch，装不好就退到纯 pickle 解析（只读键名与形状）
# --------------------------------------------------------------------------- #
def _load_with_torch(path):
    import torch
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                       # torch < 1.13 没有 weights_only
        ckpt = torch.load(path, map_location='cpu')
    meta = {}
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
        for k in ('epoch', 'it', 'best_score', 'best_epoch', 'start_epoch', 'global_step'):
            if k in ckpt:
                meta[k] = ckpt[k]
        for k in ('optimizer_state', 'lr_scheduler'):
            if k in ckpt:
                meta[k] = '<已保存>'
        wrapped = True
    else:
        state = ckpt
        wrapped = False
    shapes = {k: tuple(v.shape) for k, v in state.items() if hasattr(v, 'shape')}
    return shapes, meta, wrapped, 'torch'


def _load_without_torch(path):
    sys.path.insert(0, _HERE)
    from _pth_reader import read_state_dict
    sd, meta, fmt = read_state_dict(path)
    wrapped = bool(meta) or 'global_step' not in sd
    shapes = {k: tuple(v.shape) for k, v in sd.items()}
    return shapes, meta, wrapped, '%s(纯 pickle)' % fmt


def load_ckpt(path):
    try:
        return _load_with_torch(path)
    except Exception as e:
        try:
            shapes, meta, wrapped, mode = _load_without_torch(path)
            print('[INFO] torch 不可用（%s），已切换到纯 pickle 解析：只校验键名与形状，'
                  '不读取权重数值。' % str(e).split('\n')[0][:60])
            return shapes, meta, wrapped, mode
        except Exception as e2:
            raise RuntimeError('两种方式都读不出来：torch(%s) / pickle(%s)' % (e, e2))


# --------------------------------------------------------------------------- #
#  配置
# --------------------------------------------------------------------------- #
def _load_teacher_cfg(yaml_path):
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = (cfg or {}).get('MODEL', {}) or {}
    base = None
    dc = (cfg or {}).get('DATA_CONFIG', {}) or {}
    if dc.get('_BASE_CONFIG_'):
        try:
            with open(os.path.join(_REPO, 'tools', dc['_BASE_CONFIG_']), 'r', encoding='utf-8') as f:
                base = yaml.safe_load(f)
        except Exception:
            base = None
    return model.get('TEACHER', None), model.get('KD', None), model, base


def _fmt_shape(s):
    return 'x'.join(str(v) for v in s) if s else 'scalar'


def main():
    ap = argparse.ArgumentParser(description='教师 ckpt 体检')
    ap.add_argument('--ckpt', required=True, help='教师用的 .pth')
    ap.add_argument('--config', default=None, help='sdr 模型 yaml，用于核对教师段配置')
    ap.add_argument('--strict', action='store_true', help='WARN 也返回非 0')
    ap.add_argument('--list', action='store_true', help='打印全部键名与形状')
    args = ap.parse_args()

    ck = args.ckpt
    if not os.path.isabs(ck) and not os.path.isfile(ck):
        alt = os.path.join(_REPO, ck)
        if os.path.isfile(alt):
            ck = alt
    if not os.path.isfile(ck):
        print('[FAIL] 文件不存在: %s' % args.ckpt)
        return 2

    try:
        shapes, meta, wrapped, mode = load_ckpt(ck)
    except Exception as e:
        print('[FAIL] %s' % e)
        return 2

    keys = list(shapes)
    print('=' * 70)
    print('教师 ckpt 体检: %s' % ck)
    print('=' * 70)
    print('格式            : %s%s' % (mode, '，含 model_state 包装' if wrapped else '，裸 state_dict'))
    print('张量总数        : %d' % len(keys))
    if meta:
        for k, v in meta.items():
            print('%-15s : %s' % (k, v))

    print('\n按顶层前缀分布:')
    pref = {}
    for k in keys:
        pref[k.split('.')[0]] = pref.get(k.split('.')[0], 0) + 1
    for k, v in sorted(pref.items(), key=lambda x: -x[1]):
        print('  %-16s %d' % (k, v))

    if args.list:
        print('\n全部键名:')
        for k in keys:
            print('  %-52s %s' % (k, _fmt_shape(shapes[k])))

    problems, notes = [], []

    # --- 1. 是不是改进模型自己的 ckpt（自蒸馏） ---
    sdr_keys = [k for k in keys if k.startswith(SDR_PREFIXES)]
    if sdr_keys:
        problems.append('含蒸馏张量 %s：这是改进模型自身，自蒸馏口径不成立'
                        % sorted({k.split('.')[0] for k in sdr_keys}))

    # --- 2. 教师侧模块是否齐（只有"有参数"的模块才要求出现） ---
    required_prefixes = []
    cfg_teacher = None
    if args.config:
        cfg_teacher, kd_cfg, model, _ = _load_teacher_cfg(args.config)
        if cfg_teacher is None:
            problems.append('%s 里没有 MODEL.TEACHER 段' % args.config)
        if kd_cfg is not None and not bool(kd_cfg.get('ENABLE', True)):
            notes.append('该 yaml 的 MODEL.KD.ENABLE=False，教师不会被构建，本 ckpt 用不上')
    for seg, prefix, table in SEGMENTS:
        name = None
        if cfg_teacher is not None and cfg_teacher.get(seg):
            name = (cfg_teacher[seg] or {}).get('NAME')
        has = table.get(name, True) if name else True       # 未知模块按"有参数"处理
        hit = sum(1 for k in keys if k.startswith(prefix))
        if has:
            required_prefixes.append(prefix)
            if hit == 0:
                problems.append('教师模块 %s=%s 需要参数，但 ckpt 里没有任何 %s* 张量'
                                % (seg, name, prefix))
        else:
            notes.append('%s=%s 无可学习参数（正常，ckpt 里没有 %s* 不算缺）'
                         % (seg, name, prefix))
        if has and name:
            print('\n%-12s %-18s 张量 %3d 个' % (seg, name, hit))

    if not required_prefixes:
        required_prefixes = ['backbone_3d.', 'backbone_2d.']
    if not any(k.startswith(tuple(required_prefixes)) for k in keys):
        print('\n[FAIL] 没有任何 %s 张量，该文件不是 Voxel R-CNN 权重。'
              % ' / '.join(required_prefixes))
        return 2

    # --- 3. 通道核对（教师第一层吃的就是这两处） ---
    print('\n关键通道核对:')
    first3d = shapes.get('backbone_3d.conv_input.0.weight')
    if first3d is not None:
        in_ch = first3d[-2]        # spconv: (kD,kH,kW,in,out)
        ok = (in_ch == 4)
        print('  backbone_3d 首层输入通道 = %d  %s' % (in_ch, 'OK（MeanVFE 输出 4：x,y,z,i）'
                                                       if ok else '异常，MeanVFE 应输出 4'))
        if not ok:
            problems.append('backbone_3d 首层输入通道 %d ≠ 4' % in_ch)
    first2d = shapes.get('backbone_2d.blocks.0.1.weight') or shapes.get('backbone_2d.blocks.0.0.weight')
    bev_cfg = (cfg_teacher or {}).get('MAP_TO_BEV') or {}
    expect_bev = bev_cfg.get('NUM_BEV_FEATURES')
    if first2d is not None:
        in_bev = first2d[1]
        print('  backbone_2d 首层输入通道 = %d  （配置 NUM_BEV_FEATURES=%s）' % (in_bev, expect_bev))
        if expect_bev is not None and int(expect_bev) != int(in_bev):
            problems.append('backbone_2d 首层输入 %d 与 NUM_BEV_FEATURES=%s 不符'
                            % (in_bev, expect_bev))
    last3d = shapes.get('backbone_3d.conv_out.0.weight')
    if last3d is not None:
        print('  backbone_3d 输出通道     = %d  （HeightCompression 的 C）' % last3d[-1])

    # --- 4. 检测头的类别数（教师不用，但能告诉我们这 ckpt 的口径） ---
    cls_w = shapes.get('dense_head.conv_cls.weight')
    if cls_w is not None:
        n_out = cls_w[0]
        n_cls = n_out // 2 if n_out % 2 == 0 else None
        print('  dense_head.conv_cls 输出 = %d  → %s 类 × 2 锚（教师不使用检测头）'
              % (n_out, n_cls if n_cls else '?'))
        notes.append('该 ckpt 是 %s 类模型；教师只用 backbone，类别数不影响加载，'
                     '但与学生同口径时 baseline 数字最可比' % (n_cls if n_cls else '?'))

    # --- 5. 结论 ---
    print('\n' + '-' * 70)
    for n in notes:
        print('  说明: %s' % n)
    if problems:
        for p in problems:
            print('  [WARN] %s' % p)
        print('-' * 70)
        return 1 if args.strict else 0

    n_teacher = sum(1 for k in keys if k.startswith(tuple(required_prefixes)))
    print('  教师侧张量 %d 个，全部命中。' % n_teacher)
    print('[PASS] 可以填入 MODEL.TEACHER.PRETRAINED_PATH。')
    print('       训练开始时应看到: teacher 权重已载入: ... (%d/%d 个张量匹配, 100%%)'
          % (n_teacher, n_teacher))
    print('       若实际匹配数明显小于 %d，说明加载路径或结构有问题。' % n_teacher)
    print('-' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
