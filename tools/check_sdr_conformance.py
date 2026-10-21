# -*- coding: utf-8 -*-
"""文档一致性检查：`scp-innovations/最终方案.html`（v2） ↔ KD_Voxel 代码/配置

不依赖 torch / spconv，纯 yaml + 源码文本检查，因此在没有 GPU 环境时也能跑。
每个检查项对应文档里的一处修订（编号）或 5.3 开关表的一行，输出 PASS/FAIL。

用法（在 KD_Voxel/ 下）：
    python tools/check_sdr_conformance.py
    python tools/check_sdr_conformance.py -v            # 连带打印实际取值
    python tools/check_sdr_conformance.py --json out.json

退出码：0 = 全部通过；1 = 有 FAIL（WARN 不影响退出码）。
"""

import argparse
import copy
import json
import os
import re
import sys
import unicodedata

try:
    import yaml
except ImportError:
    print('需要 PyYAML：pip install pyyaml', file=sys.stderr)
    raise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')

MODEL_CFG = 'cfgs/kitti_models/sdr_voxel_rcnn_car.yaml'
DATA_CFG = 'cfgs/dataset_configs/kitti_dataset_sdr.yaml'
SPARSE_MODEL_CFG = 'cfgs/kitti_models/voxel_rcnn_car_sparse.yaml'
BASE_MODEL_CFG = 'cfgs/kitti_models/voxel_rcnn_car.yaml'

FILES = {
    'vfe': 'pcdet/models/backbones_3d/vfe/msdge_vfe.py',
    'cgrd': 'pcdet/models/kd/cgrd_loss.py',
    'det': 'pcdet/models/detectors/sdr_voxel_rcnn.py',
    'eipbs': 'pcdet/utils/eipbs_utils.py',
    'proc': 'pcdet/datasets/processor/data_processor.py',
    'train': 'tools/train.py',
    'train_utils': 'tools/train_utils/train_utils.py',
    'plot': 'tools/plot_curves.py',
    'vis': 'tools/visualize_detections.py',
    'dataset': 'pcdet/datasets/dataset.py',
}


# --------------------------------------------------------------------------- #
#  配置加载（递归解析 _BASE_CONFIG_）
# --------------------------------------------------------------------------- #
def _deep_merge(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(node, root):
    if isinstance(node, list):
        return [_resolve(v, root) for v in node]
    if not isinstance(node, dict):
        return node
    node = dict(node)
    base_path = node.pop('_BASE_CONFIG_', None)
    own = {k: _resolve(v, root) for k, v in node.items()}
    if base_path is None:
        return own
    base = load_cfg(os.path.join(root, base_path), root)
    return _deep_merge(base, own)


def load_cfg(path, root=None):
    root = root or TOOLS
    full = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.isfile(full):
        raise FileNotFoundError(full)
    with open(full, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return _resolve(raw, root)


def _width(text):
    """终端显示宽度（CJK 全角算 2 列），用于对齐检查表。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in text)


class Checker(object):
    def __init__(self):
        self.rows = []

    def add(self, item, expect, actual, ok, note='', level='FAIL'):
        # level 的默认值必须是 FAIL：原写法默认 PASS，会让任何 true/eq 的失败
        # 都被渲染成 [PASS] 并计入通过数（等于把 50 条断言变成永远绿）。
        self.rows.append(dict(item=item, expect=expect, actual=actual,
                              ok=bool(ok), note=note,
                              level=level if not ok else 'PASS'))

    def eq(self, item, expect, actual, note=''):
        self.add(item, expect, actual, actual == expect, note)

    def in_(self, item, allowed, actual, note=''):
        self.add(item, 'in %s' % (allowed,), actual, actual in allowed, note)

    def true(self, item, cond, expect='成立', actual=None, note=''):
        self.add(item, expect, '成立' if cond else '不成立', cond, note)

    def warn(self, item, cond, expect, actual, note=''):
        """不通过时记为 WARN（文档已知/待决策项），不计入 FAIL。"""
        self.add(item, expect, actual, cond, note, level='WARN')

    def report(self, verbose=False):
        n_fail = sum(1 for r in self.rows if not r['ok'] and r['level'] == 'FAIL')
        n_warn = sum(1 for r in self.rows if not r['ok'] and r['level'] == 'WARN')
        n_pass = len(self.rows) - n_fail - n_warn
        w = max(_width(r['item']) for r in self.rows) + 2
        for r in self.rows:
            mark = 'PASS' if r['ok'] else r['level']
            pad = ' ' * max(0, w - _width(r['item']))
            line = '[%s] %s%s%s' % (mark, r['item'], pad, '' if r['ok'] else
                                    ' 期望 %s ，实际 %s' % (r['expect'], r['actual']))
            print(line)
            if verbose and r['note']:
                print('       └ %s' % r['note'])
        print('\n合计 %d 项：PASS %d / FAIL %d / WARN %d'
              % (len(self.rows), n_pass, n_fail, n_warn))
        return n_fail


def read_src(key):
    with open(os.path.join(ROOT, FILES[key]), encoding='utf-8') as f:
        return f.read()


# --------------------------------------------------------------------------- #
#  检查主体
# --------------------------------------------------------------------------- #
def run(verbose=False):
    ck = Checker()
    model = load_cfg(MODEL_CFG)
    data = load_cfg(DATA_CFG)
    sparse = load_cfg(SPARSE_MODEL_CFG)
    src = {k: read_src(k) for k in FILES}

    m = model['MODEL']
    vfe = m['VFE']
    kd = m['KD']
    # 数据集配置直接以 DATA_PROCESSOR 为顶层键（模型配置里才是 DATA_CONFIG.DATA_PROCESSOR）
    data_proc = data.get('DATA_PROCESSOR', data.get('DATA_CONFIG', {}).get('DATA_PROCESSOR', []))
    proc = {p['NAME']: p for p in data_proc}
    eip = proc['eip_bs_voxelize']
    voxel_size = [0.05, 0.05, 0.1]
    for p in data_proc:
        if p['NAME'] == 'transform_points_to_voxels':
            voxel_size = p['VOXEL_SIZE']

    # ---------------- ① MS-DGE ----------------
    factors = vfe['SCALE_FACTORS']
    abs_scales = [round(voxel_size[0] * f, 3) for f in factors]
    ck.eq('① 尺度数=4（修订 1）', 4, len(factors),
          '文档图 1-1：原生 + 0.4/0.8/1.6 共 4 支')
    ck.eq('① 首尺度倍率=1（原生 0.05m 恒等分支，修订 1）', 1, factors[0],
          'SCALE_FACTORS 是相对基础体素的倍率，不是绝对米数')
    ck.eq('① 绝对尺度 = 0.05/0.4/0.8/1.6m（修订 1）', [0.05, 0.4, 0.8, 1.6], abs_scales,
          'voxel_size=%s' % voxel_size)
    ck.eq('① 输出通道=64（1.3(3) 接口）', 64, vfe['OUTPUT_CHANNELS'])
    ck.true('① FINE_RESIDUAL=True（修订 1）', vfe.get('FINE_RESIDUAL') is True,
            'True', vfe.get('FINE_RESIDUAL'))
    ck.true('① 精细分支 BN 权重零初始化（修订 1，代码级）',
            'nn.init.zeros_(bn.weight)' in src['vfe'])
    ck.true('① GATE_PRIOR_DIRECT=True（修订 3）',
            vfe.get('GATE_PRIOR_DIRECT') is True, 'True', vfe.get('GATE_PRIOR_DIRECT'))
    ck.true('① 先验直连层零初始化（修订 3，代码级）',
            'nn.init.zeros_(self.prior_film.weight)' in src['vfe'])
    ck.in_('① DENSITY_PRIOR 消融档位（1.5）',
           ['full', 'no_count', 'count_only', 'none'], vfe.get('DENSITY_PRIOR'))
    ck.eq('① DENSITY_STATS_MOMENTUM=0.01（修订 3）', 0.01,
          float(vfe.get('DENSITY_STATS_MOMENTUM', -1)))
    ck.true('① 用真实点数做池化权重（修订 2）',
            '_resolve_counts' in src['vfe'] and 'voxel_num_points_true' in src['vfe'])
    ck.true('① 门控熵归一化到 [0,1]（修订 6）',
            '/ np.log(len(self.scale_factors))' in src['vfe'])
    ck.true('① 硬切换消融开关 GATE_HARD（1.5）',
            'GATE_HARD' in src['vfe'] and 'GATE_HARD' in src['vfe'])
    ck.warn('① GATE_HARD 默认关闭（1.5，消融项应为 False）',
            vfe.get('GATE_HARD') is False, 'False', vfe.get('GATE_HARD'))

    # ---------------- ② CG-RD ----------------
    ck.eq('② GATE_NORM=mean（修订 8）', 'mean', kd.get('GATE_NORM'))
    ck.true('② ℒ_rod 分母是 Σ(M_roi·w)（修订 8，代码级）',
            'rod_numer / w_sum.clamp_min(self.denom_floor)' in src['cgrd'])
    ck.eq('② DIST_TYPE=riemann（修订 10）', 'riemann', kd.get('DIST_TYPE'))
    ck.eq('② RADIAL_WEIGHT=1.0（修订 10）', 1.0, float(kd.get('RADIAL_WEIGHT')))
    ck.eq('② DENOM_FLOOR=1.0（修订 9）', 1.0, float(kd.get('DENOM_FLOOR')))
    ck.eq('② RAMPUP_ITERS=500（修订 10）', 500, int(kd.get('RAMPUP_ITERS', -1)))
    ck.eq('② 损失权重 1.0/0.5/0.2（4 节建议初值）',
          [1.0, 0.5, 0.2],
          [float(kd['LOSS_WEIGHTS']['rod']), float(kd['LOSS_WEIGHTS']['ford']),
           float(kd['LOSS_WEIGHTS']['rho'])])
    ck.true('② asin clamp 留 1e-4 余量（修订 7，代码级）',
            '2.0 - 1e-4' in src['cgrd'])
    ck.true('② θ 与 |log r| 对所有 DIST_TYPE 都计算（修订 10）',
            re.search(r'theta = _geodesic_angle\(f_t, f_s, eps\)\n', src['cgrd']) is not None
            and src['cgrd'].index('theta = _geodesic_angle') < src['cgrd'].index("kind == 'l2n'"))
    ck.true('② 曲率门控开关 CURVATURE_GATE（2.5 可消融项）',
            'use_curvature_gate' in src['cgrd'] and 'CURVATURE_GATE' in src['det'])
    ck.true('② gate_eff 以 0.5 为基准归一到 ≈1（5.4）',
            'n_roi.clamp_min(1.0) * 0.5' in src['cgrd'], note='否则日志从 0.5 起步，看不出塌缩')
    ck.true('② 双掩码 M_roi / M_fore 均由 GT 几何构造（2.3(2)）',
            'bev_roi_mask' in src['cgrd'] and 'bev_foreground_mask' in src['cgrd'])
    ck.true('② 旋转框四角/半平面判定而非轴对齐（2.3(2)）',
            'cos_h = torch.cos(-h)' in src['cgrd'] or 'cos_h = torch.cos(-boxes[:, 6])' in src['cgrd'])

    # ---------------- ③ EIP-BS ----------------
    ck.in_('③ MODE=importance（5.3）', ['importance', 'uniform_beam', 'random'], eip.get('MODE'))
    ck.eq('③ TARGET_KEEP_RATIO=0.25（修订 11）', 0.25, float(eip.get('TARGET_KEEP_RATIO')))
    ck.true('③ STRATIFIED=False（修订 12）', eip.get('STRATIFIED') is False,
            'False', eip.get('STRATIFIED'))
    ck.true('③ ALIGN_KEEP_STRUCTURE=False（5.3）', eip.get('ALIGN_KEEP_STRUCTURE') is False,
            'False', eip.get('ALIGN_KEEP_STRUCTURE'))
    ck.eq('③ BEAM_SCORE=topk（修订 12 结论）', 'topk', eip.get('BEAM_SCORE'))
    ck.eq('③ 阶段序列 [64,48,32,16]（3.3(2)）', [64, 48, 32, 16], list(eip.get('STAGES')))
    ck.true('③ SAVE_TRUE_COUNTS=True（修订 2）', eip.get('SAVE_TRUE_COUNTS') is True)
    ck.eq('③ 学生 MAX_POINTS_PER_VOXEL=8（稠密仍为 5）', 8, int(eip.get('MAX_POINTS_PER_VOXEL')))
    ck.true('③ 测试期不使用 GT（修订 14，代码级）',
            "config.get('USE_GT_IN_TEST', False)" in src['proc'],
            note='否则评测时用 GT 框保车 = 信息泄漏')
    # 默认值也必须和文档一致（"只写 NAME 不写参数"时不能走回旧行为）
    ck.true('③ data_processor 默认值与 5.3 表一致（代码级）',
            "config.get('STRATIFIED', False)" in src['proc']
            and "config.get('ALIGN_KEEP_STRUCTURE', False)" in src['proc']
            and "config.get('TARGET_KEEP_RATIO', 0.25)" in src['proc']
            and "config.get('BEAM_SCORE', 'topk')" in src['proc'])
    ck.true('③ 对照组 uniform_beam / random 可跑（5.3）',
            "'uniform_beam'" in src['eipbs'] and "'random'" in src['eipbs'])

    # ---------------- 训练可见性 ----------------
    ck.true('6 个数进 tb_dict（5.4）',
            all(k in src['cgrd'] for k in ('cgrd_roi_ratio', 'cgrd_gate_eff',
                                           'cgrd_theta_mean', 'cgrd_logmag_mean'))
            and 'cgrd_weight' in src['det'])
    ck.true('msdge_gate_entropy 与 KD 开关解耦（5.4）',
            '_msdge_tb' in src['det'] and 'if self._kd_loss is not None' in src['det'])
    ck.true('控制台周期日志 [SDR]（5.4）',
            '_log_sdr_diagnostics' in src['train_utils'] and 'SDR_LOG_KEYS' in src['train_utils'])
    ck.true('训练后自动出图（training_curves / sdr_diagnostics）',
            'generate_plots' in src['train'] and 'sdr_diagnostics' in src['plot'])

    # ---------------- 训练后检测效果图（5.6 新增） ----------------
    ck.true('训练后输出检测效果图（train.py 挂载 visualize_detections）',
            'visualize_detections' in src['train'] and 'no_vis_after_train' in src['train'],
            note='否则 80 epoch 跑完只有曲线，没有一张能看框的图')
    ck.true('检测效果图不需要 open3d（matplotlib Agg 落盘）',
            "matplotlib.use('Agg')" in src['vis'] and 'savefig' in src['vis'])
    ck.true('效果图用的是 test_set（与评测同口径的稀疏输入）',
            'dataset=test_set' in src['train'],
            note='用训练集出图会看到稠密输入，与部署不一致')
    ck.true('帧号一次性随机、之后固定（跨实验组可比）',
            'resolve_frames' in src['vis'] and 'vis_frames.json' in src['train']
            and 'manifest' in src['vis'],
            note='同一批帧才能把 baseline / sparse / sdr 的图并排比较')
    ck.true('随机帧清单支持显式指定与越界重抽（vis_frames / --frames）',
            '--vis_frames' in src['train'] and '超出当前数据集范围' in src['vis'])
    ck.true('collate 支持 voxel_num_points_true（sparse 预设的 WRITE_TO_MAIN）',
            "'voxel_num_points_true'" in src['dataset'],
            note='漏掉会走 np.stack 分支，第一帧就 TypeError')

    # BEV 通道链：VoxelBackBone8x 输出 128ch × D(z 下采样后) == HeightCompression
    # 的 NUM_BEV_FEATURES == BaseBEVBackbone 的输入通道。这里按 spconv 公式静态复算。
    pcr = [float(v) for v in data['POINT_CLOUD_RANGE']]
    z = int(round((pcr[5] - pcr[2]) / voxel_size[2])) + 1    # sparse_shape z = grid+1
    for _ in range(3):                                       # conv2/3/4: k3 s2 p1
        z = (z + 2 * 1 - 3) // 2 + 1
    z = (z - 3) // 2 + 1                                     # conv_out: k3 s2 p0
    bev_ch = 128 * max(z, 1)
    stu_bev = int(m.get('MAP_TO_BEV', {}).get('NUM_BEV_FEATURES', -1))
    tch_bev = int(m.get('TEACHER', {}).get('MAP_TO_BEV', {}).get('NUM_BEV_FEATURES', -1))
    ck.eq('① 通道链：128×%d(z) == NUM_BEV_FEATURES（学生）' % max(z, 1), bev_ch, stu_bev,
          note='HeightCompression 用 C*D 动态拼接，对不上会在 BaseBEVBackbone 首次卷积报错')
    ck.eq('② 师生 BEV 通道一致（蒸馏逐格比较的前提）', stu_bev, tch_bev)

    # 师生 2D 骨干结构必须一致，否则 f_t / f_s 不是同一层的特征
    stu_2d = m.get('BACKBONE_2D', {})
    tch_2d = m.get('TEACHER', {}).get('BACKBONE_2D', {})
    keys_2d = ('LAYER_NUMS', 'LAYER_STRIDES', 'NUM_FILTERS',
               'UPSAMPLE_STRIDES', 'NUM_UPSAMPLE_FILTERS')
    ck.true('② 师生 2D 骨干超参一致（f_t/f_s 同层对齐）',
            all(stu_2d.get(k) == tch_2d.get(k) for k in keys_2d),
            note='不一致时 f_t 与 f_s 感受野不同，黎曼距离没有意义')
    ck.eq('② 学生 2D 骨干输出通道 = 256（= KD STUDENT_CHANNELS 默认）',
          sum(stu_2d.get('NUM_UPSAMPLE_FILTERS', [])), 256)

    # ---------------- 预设与对照 ----------------
    presets = set(re.findall(r"'(\w+)':\s*\('tools/cfgs", src['train']))
    ck.eq('6 个一键预设（修订 16）',
          {'baseline', 'sparse', 'sdr', 'sdr_mse', 'sdr_nokd', 'sdr_noms'}, presets)
    sp = sparse['DATA_CONFIG']['DATA_PROCESSOR']
    sp_eip = next(p for p in sp if p['NAME'] == 'eip_bs_voxelize')
    ck.eq('sparse 预设 = 均匀抽线（等点数对照）', 'uniform_beam', sp_eip.get('MODE'))
    ck.true('sparse 预设写回主键（基线也吃稀疏点云）', sp_eip.get('WRITE_TO_MAIN') is True)
    ck.eq('sparse 预设与 EIP-BS 等点数', 0.25, float(sp_eip.get('TARGET_KEEP_RATIO')))
    ck.true('sdr_mse 预设 = DIST_TYPE l2n（② 的对照组）',
            "'MODEL.KD.DIST_TYPE': 'l2n'" in src['train'])
    ck.true('sdr_nokd 预设 = 关闭蒸馏', "'MODEL.KD.ENABLE': False" in src['train'])

    # ---------------- 对照组只有"输入"这一个自变量 ----------------
    base = load_cfg(BASE_MODEL_CFG)

    def _aug_sig(c):
        g = c['DATA_CONFIG']['DATA_AUGMENTOR']['AUG_CONFIG_LIST'][0]
        return (list(g['PREPARE']['filter_by_min_points']),
                list(g['SAMPLE_GROUPS']), bool(g.get('LIMIT_WHOLE_SCENE')))

    ck.true('baseline / sparse / sdr 的 gt_sampling 完全一致',
            _aug_sig(base) == _aug_sig(sparse) == _aug_sig(model),
            note='差异会让"稀疏 vs EIP-BS"混入"贴了多少个车"这个第二自变量')
    ck.true('baseline 不含 EIP-BS 处理器，sparse/sdr 含',
            all(p['NAME'] != 'eip_bs_voxelize'
                for p in base['DATA_CONFIG']['DATA_PROCESSOR'])
            and all(any(p['NAME'] == 'eip_bs_voxelize'
                        for p in c['DATA_CONFIG']['DATA_PROCESSOR'])
                    for c in (sparse, model)))

    # ---------------- 文档自身声明的不一致（必须保持可见） ----------------
    w_class = float(eip.get('W_CLASS', 0))
    ck.warn('③ 训练/测试一致性：W_CLASS 应为 0（训练期不用 GT 挑点）',
            w_class == 0.0, 'W_CLASS=0（一致）', 'W_CLASS=%.2f（不一致，需在论文说明）' % w_class,
            note='文档 3.4/5.5 第 1 条：三个选项任选其一')
    prio = list(eip.get('CLASS_PRIORITY', []))
    ck.true('③ CLASS_PRIORITY 长度与 CLASS_NAMES 一致（Car-only 口径）',
            len(prio) == len(model.get('CLASS_NAMES', [])),
            'CLASS_PRIORITY=%s / CLASS_NAMES=%s' % (prio, model.get('CLASS_NAMES')),
            note='长度不一致时多出的项永不触发，实际权重=W_CLASS×prio')

    # ---------------- 教师 ckpt（CG-RD 的前置条件，5.7） ----------------
    tp = ((model.get('MODEL') or {}).get('TEACHER') or {}).get('PRETRAINED_PATH', None)
    tp_abs = tp if (tp and os.path.isabs(tp)) else (os.path.join(ROOT, tp) if tp else '')
    ck.eq('② 教师 ckpt 已配置且文件存在（5.7）',
          True, bool(tp) and os.path.isfile(tp_abs),
          note='PRETRAINED_PATH=%s；空值会导致教师随机初始化，蒸馏信号无意义' % tp)
    ck.warn('③ Car-only 时行人/骑行者先验不触发（5.5 第 3 条）',
            len(model.get('CLASS_NAMES', [])) > 1, '多类训练',
            'CLASS_NAMES=%s' % model.get('CLASS_NAMES'),
            note='要报行人 AP 必须切三类')
    return ck


def main():
    ap = argparse.ArgumentParser(description='最终方案.html ↔ 代码 一致性检查')
    ap.add_argument('-v', '--verbose', action='store_true', help='打印每项的补充说明')
    ap.add_argument('--json', default=None, help='把结果写到 json')
    args = ap.parse_args()

    ck = run(verbose=args.verbose)
    n_fail = ck.report(verbose=args.verbose)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(ck.rows, f, ensure_ascii=False, indent=2)
        print('已写入 %s' % args.json)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
