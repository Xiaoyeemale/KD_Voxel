# -*- coding: utf-8 -*-
"""SDR-Voxel R-CNN 链路自检（环境装好后先跑这个，再开训练）。

它按顺序验证 5 件事，任何一步失败都会直接打印"下一步该干什么"：

    1. 配置能加载          —— 两份 yaml 是否存在、_BASE_CONFIG_ 是否解析对
    2. 数据能读            —— infos / dbinfos / gt_database 是否已生成
    3. 三个创新点是否生效  —— EIP-BS 是否产出 points_kd、压缩比是多少
    4. 网络能构建 + 前反向  —— MSDGE / 教师分支 / CG-RD 是否接通，损失是否有限
    5. 推理路径能走        —— post_processing 是否正常（不跑教师）

用法（在项目根目录 KD_Voxel/ 下执行）：

    python tools/check_sdr_setup.py --preset sdr          # 查改进模型
    python tools/check_sdr_setup.py --preset baseline     # 查基线（对照组）
    python tools/check_sdr_setup.py --preset sdr --cpu    # 没有 GPU 时只查配置+数据

只想看配置不碰数据：加 --config_only
"""

import argparse
import os
import sys

import numpy as np

try:
    import torch
except Exception:            # 没装/装坏 torch 时仍可只用 --config_only 检查配置
    torch = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file          # noqa: E402

PRESETS = {
    'baseline': 'tools/cfgs/kitti_models/voxel_rcnn_car.yaml',
    'sparse': 'tools/cfgs/kitti_models/voxel_rcnn_car_sparse.yaml',
    'sdr': 'tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml',
}

OK = '[ OK ]'
BAD = '[FAIL]'


def _fail(msg, hint=None):
    print('%s %s' % (BAD, msg))
    if hint:
        print('        → %s' % hint)
    sys.exit(1)


def load_config(args):
    print('=' * 78)
    print('1/5  配置加载')
    print('=' * 78)

    if args.preset:
        cfg_file = PRESETS[args.preset]
    else:
        cfg_file = args.cfg_file
    if not os.path.isfile(cfg_file):
        _fail('找不到配置文件 %s' % cfg_file,
              '请在项目根目录 KD_Voxel/ 下执行本脚本')

    cfg_from_yaml_file(cfg_file, cfg)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs)

    print('  配置文件   : %s' % cfg_file)
    print('  模型       : %s' % cfg.MODEL.NAME)
    print('  VFE        : %s' % cfg.MODEL.VFE.NAME)
    procs = [p.NAME for p in cfg.DATA_CONFIG.DATA_PROCESSOR]
    print('  数据处理器 : %s' % ' -> '.join(procs))

    is_sdr = cfg.MODEL.NAME == 'SDRVoxelRCNN'
    if is_sdr:
        kd_cfg0 = cfg.MODEL.get('KD', None)
        kd_on = bool(kd_cfg0 and kd_cfg0.get('ENABLE', True))
        if kd_on and not hasattr(cfg.MODEL, 'TEACHER'):
            _fail('MODEL.KD.ENABLE=True 但缺少 MODEL.TEACHER 段')
        if 'eip_bs_voxelize' not in procs:
            _fail('SDR 配置的 DATA_PROCESSOR 里没有 eip_bs_voxelize',
                  '创新点③ 不会生效；请确认 DATA_CONFIG 指向 kitti_dataset_sdr.yaml')
        print('  KD 段      : %s' % ('已启用' if kd_on else '未启用（消融 A/B 组）'))
        if kd_on:
            print('  教师       : %s / BACKBONE_2D=%s / FREEZE=%s' % (
                cfg.MODEL.TEACHER.VFE.NAME,
                cfg.MODEL.TEACHER.BACKBONE_2D.NAME,
                cfg.MODEL.TEACHER.get('FREEZE', True)))
            tp = cfg.MODEL.TEACHER.get('PRETRAINED_PATH', None)
            if not tp:
                print('  教师权重   : 未设置（调试可跑，正式训练必须先给 ckpt）')
            else:
                import os as _os
                _p = tp if _os.path.isabs(tp) else _os.path.join(ROOT_DIR, tp)
                if _os.path.isfile(_p):
                    print('  教师权重   : %s  (%.1f MB ✓)' % (tp, _os.path.getsize(_p) / 1e6))
                else:
                    print('  教师权重   : %s  [缺失] 先跑 python tools/check_teacher_ckpt.py --ckpt %s'
                          % (tp, tp))
            print('  门控归一化 : %s' % (kd_cfg0.get('GATE_NORM', 'mean')))
        ep = [p for p in cfg.DATA_CONFIG.DATA_PROCESSOR if p.NAME == 'eip_bs_voxelize'][0]
        print('  EIP-BS     : %d 线 -> %d 线, stages=%s, numba=%s' % (
            ep.SRC_BEAMS, ep.DST_BEAMS, ep.STAGES, ep.get('USE_NUMBA', True)))
    else:
        print('  （基线配置：不应出现 MSDGE / TEACHER / KD / eip_bs_voxelize）')
        if cfg.MODEL.VFE.NAME != 'MeanVFE':
            _fail('基线配置的 VFE 不是 MeanVFE: %s' % cfg.MODEL.VFE.NAME)

    print('%s 配置检查通过' % OK)
    return cfg_file, is_sdr


def check_data_files():
    print()
    print('=' * 78)
    print('2/5  数据文件')
    print('=' * 78)

    data_path = cfg.DATA_CONFIG.DATA_PATH
    if not os.path.isabs(data_path):
        data_path = os.path.join(cfg.ROOT_DIR, data_path)
    missing = []

    def _check(name, required=True):
        path = os.path.join(data_path, name)
        if os.path.exists(path):
            print('  %s %s' % (OK, name))
            return True
        if required:
            print('  %s %s 不存在' % (BAD, name))
            missing.append(name)
        else:
            print('  [ -- ] %s 不存在（该阶段不需要）' % name)
        return False

    info_name = cfg.DATA_CONFIG.INFO_PATH['train'][0]
    _check(info_name)
    _check('kitti_dbinfos_train.pkl')
    _check(os.path.join('gt_database'), required=False)

    if missing:
        _fail('缺少 %s' % missing,
              '先执行：python tools/gen_kitti_infos_local.py '
              '（生成 infos 与 gt_database），再回来跑本脚本')
    print('%s 数据文件齐备（%s）' % (OK, data_path))


def check_dataset_batch(is_sdr, batch_size):
    print()
    print('=' * 78)
    print('3/5  取一个 batch（验证 EIP-BS / 体素化）')
    print('=' * 78)

    from pcdet.datasets import build_dataloader          # 延迟导入：依赖 torch/spconv
    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=batch_size, dist=False, workers=0,
        logger=None, training=True)
    print('  训练集样本数: %d' % len(train_set))
    if len(train_set) == 0:
        _fail('训练集为空（ImageSets 或 pkl 里的样本数为 0）',
              '确认 data/kitti/ImageSets/train.txt 与 kitti_infos_train.pkl 是否配套')

    batch = next(iter(train_loader))
    print('  points        : %s' % (batch['points'].shape,))
    print('  voxels        : %s' % (batch['voxels'].shape,))
    print('  gt_boxes      : %s' % (batch['gt_boxes'].shape,))
    if not is_sdr:
        print('%s 基线数据流正常（只有稠密点云）' % OK)
        return batch, train_set

    for key in ('points_kd', 'voxels_kd', 'voxel_coords_kd', 'voxel_num_points_kd'):
        if key not in batch:
            _fail('batch 里没有 %s' % key,
                  'EIP-BS 没有生效：检查 DATA_PROCESSOR 里是否启用了 eip_bs_voxelize')
        print('  %-14s: %s' % (key, (batch[key].shape,)))

    ratio = batch['points_kd'].shape[0] / max(batch['points'].shape[0], 1)
    print('  点数压缩比    : %.3f（%.0f%%）' % (ratio, 100 * ratio))
    if not (0.05 < ratio < 0.95):
        _fail('压缩比 %.3f 不合理' % ratio, '检查 EIP-BS 的 SRC_BEAMS / DST_BEAMS / STAGES')

    # 采样器实况：确认等点数预算 / 分层 / 采样模式真的生效
    sampler = getattr(getattr(train_set, 'data_processor', None), 'eip_bs_sampler', None)
    if sampler is not None and getattr(sampler, 'last_info', None):
        info = sampler.last_info
        print('  采样器        : mode=%s  等预算=%s  分层=%s  保结构=%s' % (
            info.get('mode'), sampler.target_keep_ratio,
            sampler.stratified, sampler.align_keep_structure))
        print('  线束          : %d -> %d 条（末帧）' % (
            info.get('n_beam_in', 0), info.get('n_beam_out', 0)))
        if sampler.target_keep_ratio is not None and \
                abs(info.get('keep_ratio', 0) - sampler.target_keep_ratio) > 0.02:
            print('  [警告] 实际保留率与 TARGET_KEEP_RATIO 偏差 >2%，检查 STAGES 是否可达')

    if 'voxel_num_points_true_kd' in batch:
        tc = batch['voxel_num_points_true_kd']
        cc = batch['voxel_num_points_kd']
        print('  真实点数/截断 : max %.0f vs %.0f , mean %.2f vs %.2f' % (
            float(tc.max()), float(cc.max()), float(tc.mean()), float(cc.mean())))
        if float(tc.max()) <= float(cc.max()) + 1e-6:
            print('  [警告] 真实点数没有超过截断值，密度先验修复可能未生效')
    else:
        print('  [警告] 缺少 voxel_num_points_true_kd，MS-DGE 将退回被截断的点数')
    print('%s 三个创新点的数据侧已生效' % OK)
    return batch, train_set


def check_model(batch, train_set, is_sdr, device):
    if torch is None:
        _fail('当前 Python 环境没有 torch',
              '请先按 docs/SDR-VoxelRCNN_使用说明.md 装好 torch + spconv；'
              '只想查配置可加 --config_only')
    print()
    print('=' * 78)
    print('4/5  构建网络 + 单步前反向')
    print('=' * 78)

    from pcdet.models import build_network, model_fn_decorator
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                          dataset=train_set)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  参数量        : 总计 %.2fM / 可训练 %.2fM' % (n_all / 1e6, n_train / 1e6))
    if is_sdr:
        print('  VFE 输出通道  : 学生=%d（%s）' % (
            model.vfe.get_output_feature_dim(), model.model_cfg.VFE.NAME), end='')
        if model.teacher is not None:
            print(' 教师=%d（%s）' % (
                model.teacher.vfe.get_output_feature_dim(),
                model.model_cfg.TEACHER.VFE.NAME))
        else:
            print(' 教师=未构建（KD 关闭）')
        if model.teacher is not None:
            print('  BEV 通道      : 学生=%d 教师=%d %s' % (
                model.backbone_2d.num_bev_features, model.teacher.num_bev_features,
                '（一致，CG-RD 适配器 1x1 直通）'
                if model.backbone_2d.num_bev_features == model.teacher.num_bev_features
                else '（不一致，由 CG-RD 的 1x1 适配器对齐）'))
            print('  教师分支      : %.2fM（FREEZE=%s）' % (
                sum(p.numel() for p in model.teacher.parameters()) / 1e6,
                model.teacher_frozen))
            print('  教师权重加载  : %s' % ('成功' if model.teacher_pretrained_loaded else '未加载'))
        if model.kd_criterion is None:
            print('  CG-RD         : 未启用（MODEL.KD.ENABLE=False，属消融 A/B 组配置）')
        else:
            print('  CG-RD 权重    : rod=%.2f ford=%.2f rho=%.2f  lambda=%.2f  gate_norm=%s' % (
                model.w_rod, model.w_ford, model.w_rho, model.kd_loss_weight,
                model.kd_criterion.gate_norm))

    model.to(device).train()
    model_func = model_fn_decorator()

    try:
        ret = model_func(model, batch)
    except Exception as exc:                                   # noqa: BLE001
        _fail('前反向失败: %r' % (exc,),
              '看上面的 traceback；多为 spconv/CUDA 算子未编译，或通道配置不匹配')

    loss = ret.loss
    print('  总损失        : %.4f' % float(loss.detach()))
    if not np.isfinite(float(loss.detach())):
        _fail('损失是 nan / inf', '先确认教师权重已加载，或把 MODEL.KD.LOSS_WEIGHT 调小')
    loss.backward()
    print('  反传          : 通过')

    if is_sdr and model.kd_criterion is not None:
        kd_items = {k: v for k, v in ret.tb_dict.items() if k.startswith('cgrd')}
        if not kd_items:
            _fail('tb_dict 里没有 cgrd_* 项', '蒸馏损失没有参与总损失')
        for key in sorted(kd_items):
            val = kd_items[key]
            print('  %-22s %.4f' % (key, float(val.detach() if torch.is_tensor(val) else val)))
        print('%s 教师分支 + CG-RD 已接通' % OK)

    # ---- 文档 5.4 的 6 个数：确认一次前向之后它们都真的产生了 ----
    if is_sdr:
        required = ['cgrd_roi_ratio', 'cgrd_gate_eff', 'cgrd_theta_mean',
                    'cgrd_logmag_mean', 'cgrd_weight', 'msdge_gate_entropy']
        missing = [k for k in required if k not in ret.tb_dict]
        print()
        print('  5.4 六个监控量 : %s' % ('齐全' if not missing
                                        else '缺少 %s' % missing))
        if missing:
            _fail('训练日志缺 %s' % missing,
                  '这些是文档 5.4 要求必须能看到的量，缺了就无法验证创新点是否在工作')

        vfe = model.vfe
        if hasattr(vfe, 'last_gate_entropy') and vfe.last_gate_entropy is not None:
            print('  MS-DGE 门控熵  : %.3f（0=硬选一档 / 1=各尺度等权；健康 0.5~0.9）'
                  % float(vfe.last_gate_entropy))
            if vfe.last_gate_weights is not None:
                print('  各尺度平均权重 : %s' % ['%.3f' % v for v in
                                                vfe.last_gate_weights.tolist()])
            if vfe.last_gate_dist_profile is not None:
                prof = vfe.last_gate_dist_profile.tolist()
                print('  近/中/远权重   : 近 %s | 中 %s | 远 %s'
                      % (['%.3f' % v for v in prof[0]],
                         ['%.3f' % v for v in prof[1]],
                         ['%.3f' % v for v in prof[2]]))
                print('                   （健康的趋势：近处偏 s1，远处逐级偏 s3/s4）')

        if 'eipbs_keep_ratio' in ret.tb_dict:
            print('  EIP-BS 保留率  : %.3f  前景保留 %s  %.1f ms/帧'
                  % (float(ret.tb_dict['eipbs_keep_ratio']),
                     ('%.3f' % float(ret.tb_dict['eipbs_fg_keep_ratio']))
                     if 'eipbs_fg_keep_ratio' in ret.tb_dict else '无 GT',
                     float(ret.tb_dict.get('eipbs_ms', 0.0))))
        else:
            print('  EIP-BS 统计    : 未出现在 tb_dict（检查 DATA_PROCESSOR 的 LOG_STATS 与 '
                  'eip_bs_voxelize 是否启用）')

    print('%s 网络构建与前反向通过' % OK)
    return model


def check_inference(model, batch, is_sdr, device):
    print()
    print('=' * 78)
    print('5/5  推理路径（不跑教师分支）')
    print('=' * 78)

    if not is_sdr or torch is None:
        print('  （无可检查项）')
        return

    model.eval()
    with torch.no_grad():
        preds, _ = model(batch)
    print('  预测帧数      : %d' % len(preds))
    if len(preds) > 0:
        print('  第 0 帧 box   : %s  score: %s' % (
            tuple(preds[0]['pred_boxes'].shape), tuple(preds[0]['pred_scores'].shape)))
    print('%s 推理路径通过' % OK)


def main():
    parser = argparse.ArgumentParser('SDR-Voxel R-CNN 链路自检')
    parser.add_argument('--preset', type=str, default='sdr', choices=list(PRESETS))
    parser.add_argument('--cfg_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='强制用 CPU（只查配置/数据）')
    parser.add_argument('--config_only', action='store_true', help='只查配置，不碰数据')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg_file, is_sdr = load_config(args)
    if args.config_only:
        return

    check_data_files()
    batch, train_set = check_dataset_batch(is_sdr, args.batch_size)

    device = 'cpu' if (args.cpu or torch is None) else ('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n  运行设备      : %s' % device)
    if device == 'cpu':
        print('  注意：CPU 下 spconv 通常不可用，下一步大概率失败；这是环境问题，不是代码问题。')

    model = check_model(batch, train_set, is_sdr, device)
    check_inference(model, batch, is_sdr, device)

    print()
    print('=' * 78)
    print('全部通过 —— 可以开训了：')
    print('  python tools/train.py --preset %s' % ('sdr' if is_sdr else 'baseline'))
    print('=' * 78)


if __name__ == '__main__':
    main()
