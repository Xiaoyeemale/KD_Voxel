# -*- coding: utf-8 -*-
"""
SDR-Voxel R-CNN
===============

改进模型：**S**parse-input · **D**ensity-gated encoding · **R**iemannian distillation
（对应三个创新点：EIP-BS → MS-DGE → CG-RD）

    ┌──────────────── 教师分支（冻结，64 线稠密点云） ────────────────┐
    │  MeanVFE → VoxelBackBone8x → HeightCompression → BaseBEVBackbone │ → f_t
    └──────────────────────────────────────────────────────────────────┘
                                  ↓  CG-RD（创新点②）
    ┌──────────────── 学生分支（可训练，16 线稀疏点云） ─────────────┐
    │  MSDGE（创新点①）→ VoxelBackBone8x → HeightCompression          │
    │                    → BaseBEVBackbone → AnchorHead → VoxelRCNNHead│ → 检测结果
    └──────────────────────────────────────────────────────────────────┘

    学生输入点云由创新点③ EIP-BS 在 DataProcessor 里生成（points_kd / voxels_kd）。

与基线的隔离方式
    - 全部新增代码放在新文件里，基线 `VoxelRCNN` 的 module_list 与超参一字未改；
    - 基线用 `voxel_rcnn_car.yaml` + `kitti_dataset.yaml`，改进模型用
      `sdr_voxel_rcnn_car.yaml` + `kitti_dataset_sdr.yaml`，两套配置互不引用；
    - 只有改进模型的配置里才会出现 `voxels_kd` 等键，基线数据流完全不受影响。
"""

import os

import torch

from .detector3d_template import Detector3DTemplate
from ..kd.cgrd_loss import CGRDLoss
from ..kd.teacher_branch import BEVTeacher

# 仓库根目录（本文件在 <repo>/pcdet/models/detectors/ 下）。
# 只用于把配置里的相对 ckpt 路径解析成绝对路径，避免"从别的目录启动就找不到权重"。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


class SDRVoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()          # 学生分支（与基线同构）

        self.kd_cfg = self.model_cfg.get('KD', None)
        self.teacher_cfg = self.model_cfg.get('TEACHER', None)
        kd_enabled = self.kd_cfg is not None and bool(self.kd_cfg.get('ENABLE', True))

        # ---------------------------- 教师分支 ---------------------------- #
        # KD 关闭时（消融矩阵里的 A / B 组：只要 ③ + ①，不要 ②）不再构建教师，
        # 省掉一整份主干参数与显存；此时 MODEL.TEACHER 也可以整段省略。
        self.teacher = None
        self.teacher_frozen = False
        self.teacher_pretrained_loaded = False
        if kd_enabled:
            assert self.teacher_cfg is not None, \
                'MODEL.KD.ENABLE=True 时必须提供 MODEL.TEACHER 段，' \
                '可参考 tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml'
            self.teacher = BEVTeacher(
                model_cfg=self.teacher_cfg,
                num_rawpoint_features=self.dataset.point_feature_encoder.num_point_features,
                grid_size=self.dataset.grid_size,
                point_cloud_range=self.dataset.point_cloud_range,
                voxel_size=self.dataset.voxel_size,
                depth_downsample_factor=self.dataset.depth_downsample_factor,
            )
            self.teacher_frozen = bool(self.teacher_cfg.get('FREEZE', True))
            if self.teacher_frozen:
                for p in self.teacher.parameters():
                    p.requires_grad_(False)
                self.teacher.eval()

            pretrained = self.teacher_cfg.get('PRETRAINED_PATH', None)
            if pretrained:
                self.teacher_pretrained_loaded = self.load_teacher_from_baseline(pretrained)

        # ---------------------------- KD 损失 ----------------------------- #
        # 注意：kd_criterion 只在这里赋值一次（None 或 Module），
        # 避免"先赋 None 再赋 Module"导致旧 PyTorch 版本属性查找拿到 None。
        self.kd_cell_wh = None
        self.w_rod = self.w_ford = self.w_rho = 0.0
        self.kd_loss_weight = 0.0
        self._kd_loss = None
        self._kd_tb = {}

        if kd_enabled:
            t_ch = int(self.kd_cfg.get('TEACHER_CHANNELS', self.teacher.num_bev_features))
            s_ch = int(self.kd_cfg.get('STUDENT_CHANNELS', self.backbone_2d.num_bev_features))
            self.kd_criterion = CGRDLoss(
                teacher_channels=t_ch,
                student_channels=s_ch,
                adapter_channels=self.kd_cfg.get('ADAPTER_CHANNELS', None),
                gate_hidden_channels=self.kd_cfg.get('GATE_HIDDEN_CHANNELS', 32),
                roi_dilate=self.kd_cfg.get('ROI_DILATE', 0.5),
                gate_norm=self.kd_cfg.get('GATE_NORM', 'mean'),
                dist_type=self.kd_cfg.get('DIST_TYPE', 'riemann'),
                radial_weight=self.kd_cfg.get('RADIAL_WEIGHT', 1.0),
                denom_floor=self.kd_cfg.get('DENOM_FLOOR', 1.0),
                use_curvature_gate=self.kd_cfg.get('CURVATURE_GATE', True),
            )
            weights = self.kd_cfg.get('LOSS_WEIGHTS', None)
            self.w_rod = float(weights.get('rod', 1.0)) if weights is not None else 1.0
            self.w_ford = float(weights.get('ford', 0.5)) if weights is not None else 0.5
            self.w_rho = float(weights.get('rho', 0.2)) if weights is not None else 0.2
            self.kd_loss_weight = float(self.kd_cfg.get('LOSS_WEIGHT', 1.0))
            # 蒸馏权重 ramp-up：前 N 次迭代把 λ 从 0 线性升到设定值。
            # 动机：训练初期学生特征还是随机的，D_rod 的量级由"随机初始化"决定、
            # 与检测任务无关，此时若 λ 全量生效，KD 项会压过 RPN/RCNN 的定位梯度。
            # 0 = 关闭（原行为）。
            self.kd_rampup_iters = int(self.kd_cfg.get('RAMPUP_ITERS', 0))
            cell_wh = self.kd_cfg.get('CELL_WH', None)
            if cell_wh is not None:
                self.kd_cell_wh = tuple(float(v) for v in cell_wh)
        else:
            self.kd_criterion = None

        self._kd_iters = 0
        self._kd_scale = 0.0
        self._log_build_summary()

    def _log_build_summary(self):
        """构建期打一行结构摘要：参数量、KD 开关、消融口径。

        蒸馏实验最容易出错的不是代码而是"跑了哪一版配置"，把关键开关打出来，
        日志里一眼能看出这一组是不是真的开了 ②。
        """
        def _n(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0

        msg = ['[SDR-VoxelRCNN] 学生 %.2fM' % (_n(self) / 1e6 - _n(self.teacher) / 1e6)]
        if self.teacher is not None:
            msg.append('教师 %.2fM(%s)' % (_n(self.teacher) / 1e6,
                                           '冻结' if self.teacher_frozen else '协同训练'))
        else:
            msg.append('教师 未构建')
        if self.kd_criterion is not None:
            msg.append('KD=%s λ=%.2f(w=%.2f/%.2f/%.2f) 归一化=%s 径向权重=%.2f 曲率门控=%s'
                       % (self.kd_criterion.dist_type, self.kd_loss_weight,
                          self.w_rod, self.w_ford, self.w_rho,
                          self.kd_criterion.gate_norm, self.kd_criterion.radial_weight,
                          '开' if self.kd_criterion.use_curvature_gate else '关'))
        else:
            msg.append('KD=关闭')
        vfe = self.vfe
        if hasattr(vfe, 'scale_factors'):
            # 文档 1.5 修订 5 要求"参数量以构建期日志为准"，这里把 MS-DGE 自身
            # 参数与"相对 MeanVFE 的净增量"一起打出来（净增量含第一层稀疏卷积
            # 因输入通道 4→64 而多出的权重，这才是和基线对比时真正的差值）。
            n_vfe = _n(vfe)
            base_in = int(vfe.get_output_feature_dim())
            msg.append('MSDGE 尺度=%s(%s m) 先验=%s 精细残差=%s 硬门控=%s 参数=%.3fM'
                       % (vfe.scale_factors,
                          '/'.join('%.2f' % (vfe.voxel_size[0] * f) for f in vfe.scale_factors),
                          vfe.density_prior, getattr(vfe, 'fine_residual', False),
                          getattr(vfe, 'gate_hard', False), n_vfe / 1e6))
            # 第一层稀疏卷积：MeanVFE 输入 4 通道，MSDGE 输入 OUTPUT_CHANNELS
            extra_in = 0
            for m in self.backbone_3d.modules():
                w = getattr(m, 'weight', None)
                if w is not None and w.dim() == 5:      # SubMConv3d
                    per_in = w.numel() // max(w.shape[1], 1)
                    extra_in = per_in * max(base_in - 4, 0)
                    break
            msg.append('净增参数≈%.3fM(含首层 %d→%d 通道)' % ((n_vfe + extra_in) / 1e6, 4, base_in))
        print(' | '.join(msg))

        if self.kd_criterion is not None and not self.teacher_pretrained_loaded:
            path = self.teacher_cfg.get('PRETRAINED_PATH', None)
            why = '未设置' if not path else '加载失败（%s）' % path
            print('[SDR-VoxelRCNN] WARNING: MODEL.TEACHER.PRETRAINED_PATH %s，'
                  '教师分支%s 初始化。' % (why, '随机' if self.teacher_frozen else '随机且参与训练'))
            print('[SDR-VoxelRCNN]          此时蒸馏信号没有意义（学生在拟合噪声），'
                  'cgrd_* 曲线照样会下降、AP 却只等于稀疏基线。')
            print('[SDR-VoxelRCNN]          正确顺序：先用基线配置训练一个 ckpt →'
                  ' python tools/check_teacher_ckpt.py --ckpt <path> 体检 →'
                  ' 填进 PRETRAINED_PATH。')

    # ------------------------------------------------------------------ #
    #  教师权重加载（基线 / 官方预训练 ckpt 都可用，键名一致）
    # ------------------------------------------------------------------ #
    def load_teacher_from_baseline(self, path, logger=None):
        log = logger.info if logger is not None else (lambda m: print('[SDR-VoxelRCNN] ' + m))
        if not os.path.isfile(path) and not os.path.isabs(path):
            # 允许写相对仓库根的路径（默认配置里的 checkpoints/xxx.pth 就是这种），
            # 这样从 `python tools/train.py` 之外的目录启动也不会找不到。
            alt = os.path.join(_REPO_ROOT, path)
            if os.path.isfile(alt):
                path = alt
        if not os.path.isfile(path):
            log('teacher ckpt 不存在，跳过加载: %s' % path)
            return False

        # torch>=2.6 的 torch.load 默认 weights_only=True，会拒绝带 optimizer/epoch
        # 的 OpenPCDet ckpt；这里显式关闭（本文件只读自己训练出来的权重）。
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:                       # torch<1.13 没有 weights_only 参数
            ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            state = ckpt['model_state']
        else:
            state = ckpt

        target = self.teacher.state_dict()
        matched, skipped = {}, 0
        for key, val in state.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            if key in target and target[key].shape == val.shape:
                matched[key] = val
            else:
                skipped += 1
        target.update(matched)
        self.teacher.load_state_dict(target)
        ratio = len(matched) / max(len(target), 1)
        log('teacher 权重已载入: %s  (%d/%d 个张量匹配, %.0f%%)'
            % (path, len(matched), len(target), 100.0 * ratio))
        if skipped > 0:
            log('  （%d 个张量属于检测头/ROI 头等教师分支不包含的模块，已忽略）' % skipped)
        if ratio < 0.9:
            # 教师分支不含检测头，所以从**本仓库基线 ckpt** 加载时应当接近 100%。
            # 命中率偏低说明结构/通道对不上，剩下的张量会停在随机初始化上 ——
            # 这种"半随机教师"是最难察觉的失败模式，故不认为加载成功。
            log('  WARNING: 仅命中 %.0f%% 的教师张量，剩余部分仍是随机初始化，'
                '蒸馏目标不可信。请用 tools/check_teacher_ckpt.py 核对结构与通道。'
                % (100.0 * ratio))
            return False
        return True

    # ------------------------------------------------------------------ #
    #  输入切换：把学生（EIP-BS 稀疏）点云换进来
    # ------------------------------------------------------------------ #
    @staticmethod
    def swap_to_student_inputs(batch_dict):
        """若数据管线产出了 *_kd 键，则把主键替换为学生分支的输入。

        注意：必须在教师分支前向之后调用，教师分支读的是稠密点云。
        """
        for dense_key, sparse_key in (
                ('voxels', 'voxels_kd'),
                ('voxel_coords', 'voxel_coords_kd'),
                ('voxel_num_points', 'voxel_num_points_kd'),
                ('voxel_num_points_true', 'voxel_num_points_true_kd'),
                ('points', 'points_kd')):
            if sparse_key in batch_dict:
                batch_dict[dense_key] = batch_dict[sparse_key]
        return batch_dict

    # ------------------------------------------------------------------ #
    #  前向
    # ------------------------------------------------------------------ #
    def teacher_forward(self, batch_dict):
        """教师前向。

        冻结时：eval + no_grad + detach（BN 用 running stats，不产出梯度，
                也不影响 DDP 的 bucket 划分）；
        不冻结时：正常前向且**不 detach**，教师参数会通过蒸馏损失得到梯度，
                这才是真正的协同训练。
        """
        if self.teacher_frozen:
            self.teacher.eval()
            with torch.no_grad():
                f_t = self.teacher(batch_dict)
            return f_t.detach()
        return self.teacher(batch_dict)

    def _compute_kd_loss(self, f_t, batch_dict):
        f_s = batch_dict.get('spatial_features_2d', None)
        gt_boxes = batch_dict.get('gt_boxes', None)
        if f_s is None or gt_boxes is None or gt_boxes.numel() == 0:
            return

        pts_key = 'points_kd' if 'points_kd' in batch_dict else 'points'
        pts = batch_dict.get(pts_key, None)

        pcr = [float(v) for v in self.dataset.point_cloud_range]
        h, w = int(f_s.shape[-2]), int(f_s.shape[-1])
        if self.kd_cell_wh is None:
            cell_wh = ((pcr[3] - pcr[0]) / w, (pcr[4] - pcr[1]) / h)
        else:
            cell_wh = self.kd_cell_wh
        xy_min = (pcr[0], pcr[1])

        l_rod, l_ford, l_rho, tb_dict = self.kd_criterion(
            f_t, f_s, gt_boxes, pts, cell_wh, xy_min)
        self._kd_loss = self.w_rod * l_rod + self.w_ford * l_ford + self.w_rho * l_rho
        self._kd_tb = tb_dict

    # ------------------------------------------------------------------ #
    #  MS-DGE 的监控量（与 KD 是否开启无关）
    # ------------------------------------------------------------------ #
    def _msdge_tb(self, batch_dict=None):
        """把 ① 的门控状态与 ③ 的采样统计取出来进日志。

        必须独立于 ② 调用：`--preset sdr_nokd` 与 `--preset sdr_noms` 这两组消融
        里 KD 是关的，但"门控有没有学到东西"恰恰是这两组要回答的问题。原先这段
        逻辑写在 _compute_kd_loss 里，KD 一关就没有 msdge_gate_entropy，
        等于把 ① 的验证证据绑死在 ② 的开关上。
        """
        out = {}
        vfe = getattr(self, 'vfe', None)
        if hasattr(vfe, 'scale_factors'):
            ent = getattr(vfe, 'last_gate_entropy', None)
            gw = getattr(vfe, 'last_gate_weights', None)
            if ent is not None:
                out['msdge_gate_entropy'] = ent.detach()
            if gw is not None:
                for k in range(gw.numel()):
                    out['msdge_gate_w_s%d' % (k + 1)] = gw[k].detach()
            scale = getattr(vfe, 'last_density_scale', None)
            if scale is not None and scale[0] > 0 and ent is not None:
                out['msdge_density_scale_fine'] = torch.tensor(scale[0], device=ent.device)
                out['msdge_density_scale_coarse'] = torch.tensor(scale[1], device=ent.device)
        out.update(self._eipbs_stats(batch_dict))
        return out

    @staticmethod
    def _eipbs_stats(batch_dict):
        """③ 的运行时统计：保留率 / 前景保留率 / 存活线束数 / 单帧耗时。"""
        if batch_dict is None or 'eipbs_stats' not in batch_dict:
            return {}
        s = batch_dict['eipbs_stats']
        s = s.reshape(-1, s.shape[-1]).float()
        names = ('eipbs_keep_ratio', 'eipbs_fg_keep_ratio', 'eipbs_beams', 'eipbs_ms')
        out = {}
        for i, n in enumerate(names):
            if i >= s.shape[-1]:
                break
            col = s[:, i]
            if n == 'eipbs_fg_keep_ratio':
                col = col[col >= 0]        # -1 = 本帧无 GT（测试期），不计入均值
                if col.numel() == 0:
                    continue
            out[n] = col.mean()
        return out

    def _kd_weight_now(self):
        """当前生效的蒸馏权重 λ（含 ramp-up）。"""
        if self.kd_rampup_iters <= 0:
            return self.kd_loss_weight
        return self.kd_loss_weight * min(1.0, (self._kd_iters + 1) / float(self.kd_rampup_iters))

    def forward(self, batch_dict):
        self._kd_loss = None
        self._kd_tb = {}

        f_t = None
        if self.training and self.kd_criterion is not None:
            f_t = self.teacher_forward(batch_dict)      # 教师看稠密点云
            self._kd_iters += 1

        batch_dict = self.swap_to_student_inputs(batch_dict)   # 换成学生稀疏点云
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            if f_t is not None:
                self._compute_kd_loss(f_t, batch_dict)
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            return {'loss': loss}, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    # ------------------------------------------------------------------ #
    #  总损失 = RPN + RCNN + λ · (w1·ℒ_rod + w2·ℒ_ford + w3·ℒ_ρ)
    # ------------------------------------------------------------------ #
    def get_training_loss(self, batch_dict=None):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_rcnn

        # ① / ③ 的监控量：与 ② 是否开启无关，一律记录。
        # （文档 5.4 的 6 个数里，msdge_gate_entropy 与 eipbs_* 都属于这一类）
        tb_dict.update(self._msdge_tb(batch_dict))
        if 'rpn_loss' not in tb_dict:               # 极端情况下检测头没回填
            tb_dict['rpn_loss'] = float(loss_rpn.detach())
        if 'rcnn_loss' not in tb_dict:
            tb_dict['rcnn_loss'] = float(loss_rcnn.detach())

        if self._kd_loss is not None:
            self._kd_scale = self._kd_weight_now()
            weighted = self._kd_scale * self._kd_loss
            loss = loss + weighted
            tb_dict.update(self._kd_tb)
            tb_dict['cgrd_loss_total'] = weighted.detach()
            tb_dict['cgrd_weight'] = torch.tensor(self._kd_scale, device=loss.device)
            # KD 项相对检测损失的量级（文档 5.4 第 6 行）：> 1 说明蒸馏压过检测，
            # 应该降 LOSS_WEIGHT。用 detach 后的标量，不进反向。
            det = (loss_rpn.detach() + loss_rcnn.detach()).clamp_min(1e-3)
            tb_dict['cgrd_to_det_ratio'] = weighted.detach() / det

            # ---- 控制台/进度条：只放 4 个短键，避免把 tqdm 撑爆 ----
            # 完整数值由 train_utils 在每 logger_iter_interval 打一行 [SDR] 日志。
            disp_dict['kd'] = float(weighted.detach())
            disp_dict['rod'] = float(self._kd_tb.get('cgrd_loss_rod', 0.0))
            disp_dict['g_eff'] = float(self._kd_tb.get('cgrd_gate_eff', 0.0))
            disp_dict['H'] = float(self._kd_tb.get('msdge_gate_entropy', 0.0))

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d

        return loss, tb_dict, disp_dict
