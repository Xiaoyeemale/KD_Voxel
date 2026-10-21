# -*- coding: utf-8 -*-
"""
创新点② CG-RD —— 曲率门控黎曼蒸馏（OpenPCDet 工程化版本）

本文件是 `创新点代码/创新点2_CG_RD.py` 的框架内适配版：数学定义、三段损失、
双掩码、曲率门控与黎曼距离公式与原版完全一致，仅在实现上做了两点工程优化：

    1. `points_in_rboxes` / `bev_roi_mask` 由"逐框 Python for 循环"改为按 batch
       向量化（KITTI 一帧 ~10 个框、BEV 200x176，向量化后单次前向省几十毫秒）；
    2. 所有几何量（cell_wh / xy_min）可由模型 cfg 传入，缺省时由调用方从
       voxel_size × feature_map_stride 与 point_cloud_range 推导。

三段损失（v2 口径，分母见下文第 1 条）：
    ℒ_rod  : Σ(M_roi × 曲率门控 w × D_rod) / Σ(M_roi × w)   主对齐项
    ℒ_ford : Σ(M_fore × D_rod) / ΣM_fore                    稀疏前景强制对齐
    ℒ_ρ    : M_roi 内局部特征能量密度一致性                   密度正则

与独立版的**必要**差异（都是为了不引入数值/优化陷阱，或为了能跑消融）：

    1. `gate_norm='mean'`（默认）：ℒ_rod 的分母改为 Σ(M_roi×w) 而非 ΣM_roi。
       原式存在"把门控关掉即可让损失归零"的平凡解，会让 ② 在训练中被自动废弃。
       设 'count' 可严格复现原式（仅用于对照）。
    2. asin 的 clamp 上限留 1e-4 余量，避免方向相反的师生特征触发 inf 梯度。
    3. `denom_floor`：分母下限（默认 1 个格子），防止 ROI 极小时除出尖峰梯度。
    4. `dist_type` / `radial_weight`：距离从"固定黎曼"变成可切换。
       这不是为了效果，而是为了让"② 相对普通 MSE 蒸馏到底值多少"这一行消融能跑
       —— 同一份代码、同一个教师 ckpt、同样的掩码与门控，只换"罚多少"的定义。
    5. 曲率门控输出层小方差初始化：初始 w≈0.5（所有位置同等重要），
       让"在哪些位置罚得更重"从均匀分配开始学。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  特征距离：黎曼（测地）距离 / 对照组
# --------------------------------------------------------------------------- #
def _geodesic_angle(f_t, f_s, eps=1e-6):
    """θ = 2·asin(‖û_t − û_s‖/2)，两个特征向量方向的球面测地角，落在 [0, π]。"""
    u_t = f_t / f_t.norm(dim=1).clamp_min(eps).unsqueeze(1)
    u_s = f_s / f_s.norm(dim=1).clamp_min(eps).unsqueeze(1)
    # asin'(x) = 1/sqrt(1-x²) 在 x=1 处发散：若只 clamp 到 2.0，方向相反的师生
    # 特征（线性层输出可为负，完全可能出现）会得到 inf 梯度并污染整个 batch，
    # 因此留 1e-4 的安全余量（对距离本身的影响 < 1e-4·π，可忽略）。
    chord = (u_t - u_s).norm(dim=1).clamp(max=2.0 - 1e-4)
    return 2.0 * torch.asin(chord / 2.0)           # 与 acos 等价，但恒等输入精确为 0


def feature_distance(f_t, f_s, kind='riemann', radial_weight=1.0, eps=1e-6):
    """统一的"罚多少"入口，`kind` 同时充当消融开关。

    riemann —— 文档原式：方向（测地角）+ 模长，默认
    cosine  —— 只保留方向项 θ/π，去掉径向项（师生主干不同构、模长量纲不可比时的稳妥版）
    l2n     —— 归一化 L2：‖f_t−f_s‖/(‖f_t‖+eps)，与 θ/π 同量级
    l2      —— 普通 MSE 蒸馏（逐像素 RMS 距离），最常见的朴素对照
               —— 用来回答"② 相对普通 MSE 蒸馏到底值多少"这一行关键消融

    返回 (d, theta, log_mag)，后两个用于判断是哪一项在主导损失（训练日志里的
    cgrd_theta_mean / cgrd_logmag_mean）。

    注意：**无论 `kind` 是什么都计算 theta / log_mag**。文档 5.4 要求这两个数
    常驻日志，而 `--preset sdr_mse`（DIST_TYPE=l2n）恰恰是最需要看一眼"换成
    L2 之后方向项/径向项各自多大"的那一组；若只在 riemann/cosine 下才算，
    对照组跑完就没有这两列，消融表无法解释。
    """
    theta = _geodesic_angle(f_t, f_s, eps)
    log_mag = (torch.log(f_t.norm(dim=1).clamp_min(eps))
               - torch.log(f_s.norm(dim=1).clamp_min(eps))).abs()
    if kind == 'riemann':
        d = theta / np.pi + radial_weight * log_mag
    elif kind == 'cosine':
        d = theta / np.pi
    elif kind == 'l2n':
        d = (f_t - f_s).norm(dim=1) / (f_t.norm(dim=1) + eps)
    elif kind == 'l2':
        d = ((f_t - f_s) ** 2).mean(dim=1).clamp_min(0).sqrt()
    else:
        raise ValueError("DIST_TYPE 只能是 riemann/cosine/l2n/l2，收到 %s" % kind)
    return d, theta, log_mag


def riemannian_distance(f_t, f_s, eps=1e-6, radial_weight=1.0):
    """D_rod = θ/π + w_r·|log r_t − log r_s|（只返回距离，保留原接口）。"""
    return feature_distance(f_t, f_s, 'riemann', radial_weight, eps)[0]


# --------------------------------------------------------------------------- #
#  旋转框包含判定（纯 torch）
# --------------------------------------------------------------------------- #
def points_in_rboxes(points, gt_boxes):
    """(N, 1+C)[batch_idx, x, y, z] 是否落在 (B, M, 8) 任一 GT 框内。"""
    n = points.shape[0]
    inside = torch.zeros(n, dtype=torch.bool, device=points.device)
    if gt_boxes is None or n == 0:
        return inside

    for b in range(gt_boxes.shape[0]):
        m_b = points[:, 0] == b
        if not m_b.any():
            continue
        boxes = gt_boxes[b]
        boxes = boxes[boxes.abs().sum(dim=1) > 1e-6]
        if boxes.numel() == 0:
            continue
        pts = points[m_b, 1:4]
        c, d, h = boxes[:, :3], boxes[:, 3:6], boxes[:, 6]
        cos_h = torch.cos(-h).unsqueeze(0)
        sin_h = torch.sin(-h).unsqueeze(0)
        rel = pts.unsqueeze(1) - c.unsqueeze(0)                     # (P, K, 3)
        lx = rel[..., 0] * cos_h - rel[..., 1] * sin_h
        ly = rel[..., 0] * sin_h + rel[..., 1] * cos_h
        lz = rel[..., 2]
        cur = (lx.abs() <= d[:, 0] / 2) & (ly.abs() <= d[:, 1] / 2) & (lz.abs() <= d[:, 2] / 2)
        inside[m_b] = cur.any(dim=1)
    return inside


# --------------------------------------------------------------------------- #
#  双掩码
# --------------------------------------------------------------------------- #
def _cell_centers(feat_hw, cell_wh, xy_min, device):
    h, w = feat_hw
    cell_w, cell_h = cell_wh
    x_min, y_min = xy_min
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * cell_w + x_min
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * cell_h + y_min
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)      # (H*W, 2)


def bev_roi_mask(gt_boxes, feat_hw, cell_wh, xy_min, dilate=0.5):
    """M_roi：(B, H, W) bool，落在（外扩 dilate 米的）GT 3D 框内的 BEV 格子。"""
    b_size = gt_boxes.shape[0]
    h, w = feat_hw
    cells = _cell_centers(feat_hw, cell_wh, xy_min, gt_boxes.device)
    mask = torch.zeros(b_size, h * w, dtype=torch.bool, device=gt_boxes.device)

    for b in range(b_size):
        boxes = gt_boxes[b]
        boxes = boxes[boxes.abs().sum(dim=1) > 1e-6]
        if boxes.numel() == 0:
            continue
        c = boxes[:, :2]
        hx = boxes[:, 3] / 2 + dilate
        hy = boxes[:, 4] / 2 + dilate
        cos_h = torch.cos(-boxes[:, 6]).unsqueeze(0)
        sin_h = torch.sin(-boxes[:, 6]).unsqueeze(0)
        rel = cells.unsqueeze(1) - c.unsqueeze(0)                   # (HW, K, 2)
        lx = rel[..., 0] * cos_h - rel[..., 1] * sin_h
        ly = rel[..., 0] * sin_h + rel[..., 1] * cos_h
        mask[b] = ((lx.abs() <= hx) & (ly.abs() <= hy)).any(dim=1)
    return mask.view(b_size, h, w)


def bev_foreground_mask(student_points, gt_boxes, feat_hw, cell_wh, xy_min):
    """M_fore：(B, H, W) bool，学生前景点（GT 框内）落入的 BEV 格子。"""
    b_size = gt_boxes.shape[0]
    h, w = feat_hw
    cell_w, cell_h = cell_wh
    x_min, y_min = xy_min
    mask = torch.zeros(b_size, h, w, dtype=torch.bool, device=gt_boxes.device)
    if student_points is None or student_points.shape[0] == 0:
        return mask

    inside = points_in_rboxes(student_points, gt_boxes)
    if not bool(inside.any()):
        return mask
    pts = student_points[inside]
    cell_x = ((pts[:, 1] - x_min) / cell_w).long().clamp(0, w - 1)
    cell_y = ((pts[:, 2] - y_min) / cell_h).long().clamp(0, h - 1)
    b_idx = pts[:, 0].long().clamp(0, b_size - 1)
    mask[b_idx, cell_y, cell_x] = True
    return mask


# --------------------------------------------------------------------------- #
#  曲率门控
# --------------------------------------------------------------------------- #
class CurvatureGate(nn.Module):
    """曲率门控 w = σ(Conv([f_s, κ])) ∈ (0,1)，κ 为特征场梯度幅值（边缘处大）。"""

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    @staticmethod
    def feature_curvature(f):
        grad_x = F.pad(f[:, :, :, 1:] - f[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')
        grad_y = F.pad(f[:, :, 1:, :] - f[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')
        kappa = grad_x.abs().mean(dim=1, keepdim=True) + grad_y.abs().mean(dim=1, keepdim=True)
        k_max = kappa.amax(dim=(2, 3), keepdim=True)
        return kappa / (k_max + 1e-6)

    def forward(self, f_s):
        kappa = self.feature_curvature(f_s)
        return torch.sigmoid(self.conv2(F.relu(self.conv1(torch.cat([f_s, kappa], dim=1)))))


# --------------------------------------------------------------------------- #
#  CG-RD 损失
# --------------------------------------------------------------------------- #
class CGRDLoss(nn.Module):
    """ℒ_rod + ℒ_ford + ℒ_ρ，内含学生→教师 1x1 通道适配器与曲率门控。"""

    def __init__(self,
                 teacher_channels=256,
                 student_channels=256,
                 adapter_channels=None,
                 gate_hidden_channels=32,
                 roi_dilate=0.5,
                 gate_norm='mean',
                 dist_type='riemann',
                 radial_weight=1.0,
                 denom_floor=1.0,
                 use_curvature_gate=True):
        super().__init__()
        self.teacher_channels = int(teacher_channels)
        self.student_channels = int(student_channels)
        adapter_channels = int(adapter_channels) if adapter_channels else self.teacher_channels
        self.roi_dilate = float(roi_dilate)
        assert gate_norm in ('mean', 'count'), \
            "GATE_NORM 只能是 'mean'（推荐，防塌缩）或 'count'（文档原式）"
        self.gate_norm = gate_norm
        assert dist_type in ('riemann', 'cosine', 'l2n', 'l2'), \
            "DIST_TYPE 只能是 riemann/cosine/l2n/l2"
        self.dist_type = dist_type
        self.radial_weight = float(radial_weight)
        # 分母下限：M_roi 覆盖率在 KITTI Car 上只有 2~4%（一帧 4~8 个框），
        # 单帧 ROI 为空或极小（远车、测试集无标注）时 Σw 会接近 0，
        # 除法会把噪声放大成尖峰梯度。这里给分母一个"至少 1 个格子"的下限。
        self.denom_floor = float(denom_floor)
        # 文档 2.5 的可消融项包含"曲率门控开/关"。关掉时 w ≡ 1，ℒ_rod 退化为
        # "M_roi 内的平均黎曼距离" —— 这一行是"门控本身值多少"的唯一对照，
        # 与"换成 L2 距离值多少"（DIST_TYPE）是两件独立的事，必须能分开跑。
        self.use_curvature_gate = bool(use_curvature_gate)

        self.adapter = nn.Sequential(
            nn.Conv2d(self.student_channels, adapter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_channels),
        )
        self.curvature_gate = CurvatureGate(adapter_channels, int(gate_hidden_channels))
        # 门控输出层小方差初始化：初始 w ≈ σ(0) = 0.5，即"所有位置同等重要"，
        # 让"在哪些位置罚得更重"从均匀分配开始学，而不是从一个随机偏好开始。
        with torch.no_grad():
            self.curvature_gate.conv2.weight.mul_(0.1)
            if self.curvature_gate.conv2.bias is not None:
                self.curvature_gate.conv2.bias.zero_()

    def forward(self, f_t, f_s, gt_boxes, student_points, cell_wh, xy_min):
        """
        Args:
            f_t: (B, C_t, H, W) 教师 BEV 特征（应已 detach）
            f_s: (B, C_s, H, W) 学生 BEV 特征
            gt_boxes: (B, M, 8) [x,y,z,dx,dy,dz,heading,label]
            student_points: (N, 1+C) 学生输入点云
            cell_wh: (cell_w, cell_h) BEV 格子物理尺寸（米）
            xy_min: (x_min, y_min)
        Returns:
            loss_rod, loss_ford, loss_rho, tb_dict
        """
        if f_t.shape[-2:] != f_s.shape[-2:]:
            # 正常情况下师生共用同一 voxel_size / point_cloud_range，BEV 尺寸应当相同；
            # 走到这里说明教师 ckpt 的字段与学生的数据配置不一致（例如换了 voxel_size
            # 或换了数据集）。双线性对齐能跑通，但"逐格点比较距离"的前提已经被破坏，
            # 所以只提醒一次，不做静默降级。
            if not getattr(self, '_shape_warned', False):
                print('[CG-RD] WARNING: 教师 BEV %s 与学生 %s 不一致，已双线性对齐。'
                      '蒸馏目标会失真，请确认教师 ckpt 与数据集配置同源。'
                      % (tuple(f_t.shape[-2:]), tuple(f_s.shape[-2:])))
                self._shape_warned = True
            f_t = F.interpolate(f_t, size=f_s.shape[-2:], mode='bilinear', align_corners=False)
        f_s_adapted = self.adapter(f_s)

        # AMP 兼容：距离与掩码统一在 fp32 下计算。
        # （否则 autocast 会把 d_rod 升到 fp32、门控输出仍是 fp16，相乘直接报 dtype 不一致；
        #   同时 fp16 下 asin/log 在恒等附近会精度塌缩，破坏"师生一致 => 距离为 0"的性质）
        f_t_compute = f_t.float()
        f_s_compute = f_s_adapted.float()

        # 1) 罚多少：特征距离（默认黎曼测地距离；DIST_TYPE 可切成对照组的 L2）
        d_rod, theta, log_mag = feature_distance(
            f_t_compute, f_s_compute, kind=self.dist_type,
            radial_weight=self.radial_weight)

        # 2) 罚多重：曲率门控（空间内容自适应）
        if self.use_curvature_gate:
            w_gate = self.curvature_gate(f_s_compute).squeeze(1).float()
        else:
            # 消融口径：所有位置同等重要（w≡1），等价于"去掉门控、只留掩码与距离"
            w_gate = torch.ones_like(d_rod)

        # 3) 在哪对齐：双掩码
        feat_hw = f_s.shape[-2:]
        m_roi = bev_roi_mask(gt_boxes, feat_hw, cell_wh, xy_min,
                             dilate=self.roi_dilate).to(d_rod.dtype)
        m_fore = bev_foreground_mask(student_points, gt_boxes, feat_hw,
                                     cell_wh, xy_min).to(d_rod.dtype)

        eps = 1e-6
        # ------------------------------------------------------------------ #
        #  门控的归一化方式 —— 这一行决定了创新点② 会不会"自己关掉自己"
        #
        #  文档原式 ℒ_rod = Σ(M·w·D)/ΣM 中，w = σ(Conv([F_s,κ])) 是**由学生特征
        #  算出、且随蒸馏一起训练**的（原版代码注释明确写了"两者均为可学习模块"）。
        #  于是 ∂ℒ/∂b = Σ(M·w(1−w)·D)/ΣM > 0 恒成立，梯度下降只要把门控 bias 压到
        #  很小的负数就能让 w→0，从而 ℒ_rod→0 —— 训练会稳定收敛到"把门关掉"这个
        #  平凡解，主对齐项在几十个 epoch 后彻底消失，消融表上表现为"加了 CG-RD
        #  没提升"，而真相是它根本没在起作用。
        #
        #  gate_norm='mean'（默认）：分母换成权重和 Σ(M·w)，即退化为"以 w 为权的
        #  加权平均"。整体缩放 w 不改变损失，塌缩方向被精确消掉，而"在哪些位置
        #  罚得更重"这一真正想学的相对分配被完整保留（w 均匀时与 w≡1 完全等价）。
        #  gate_norm='count' ：严格照文档原式，仅用于复现/对照实验。
        # ------------------------------------------------------------------ #
        w_sum = (w_gate * m_roi).sum()
        rod_numer = (d_rod * w_gate * m_roi).sum()
        n_roi = m_roi.sum()
        if self.gate_norm == 'mean':
            loss_rod = rod_numer / w_sum.clamp_min(self.denom_floor)
            # 等效门控强度：以"初始化值 σ(0)=0.5"为 1.0 做归一，这样文档 5.4 的
            # "≈1 且平稳"才成立（直接把 w 的绝对值报出来会从 0.5 起步，看不出
            # 是不是在塌）。持续单调下降 = 门控把 ROI 内大多数位置压到 0，
            # 只剩极少数位置参与对齐 —— 那是塌缩的早期征兆。
            gate_eff = w_sum / (n_roi.clamp_min(1.0) * 0.5)
        else:
            loss_rod = rod_numer / n_roi.clamp_min(self.denom_floor)
            gate_eff = w_gate.mean() / 0.5
        n_fore = m_fore.sum()
        loss_ford = (d_rod * m_fore).sum() / n_fore.clamp_min(self.denom_floor)

        rho_t = F.avg_pool2d(f_t_compute.norm(dim=1, keepdim=True), kernel_size=3, stride=1, padding=1)
        rho_s = F.avg_pool2d(f_s_compute.norm(dim=1, keepdim=True), kernel_size=3, stride=1, padding=1)
        loss_rho = ((rho_t - rho_s).abs().squeeze(1) * m_roi).sum() / n_roi.clamp_min(self.denom_floor)

        tb_dict = {
            'cgrd_loss_rod': loss_rod.detach(),
            'cgrd_loss_ford': loss_ford.detach(),
            'cgrd_loss_rho': loss_rho.detach(),
            'cgrd_roi_ratio': m_roi.mean().detach(),
            'cgrd_fore_ratio': m_fore.mean().detach(),
            'cgrd_gate_mean': w_gate.mean().detach(),
            # 归一化后进入分母的等效平均权重。'mean' 模式下它应在 1 附近波动；
            # 若持续单调下降 → 门控仍在塌缩（此时 'mean' 已不影响损失大小，
            # 只说明 w 的分布变平，可接受）。
            'cgrd_gate_eff': gate_eff.detach(),
        }
        # 分项监控：θ/π 与 |log r| 谁在主导损失。
        # 若 logmag 比 theta/π 大一个量级，说明径向项在主导 —— 师生主干不同构
        # （教师 4 通道 MeanVFE vs 学生 64 通道 MSDGE）时这很常见，
        # 此时把 KD.RADIAL_WEIGHT 调到 0.5 或 0，或改用 DIST_TYPE: cosine。
        if theta is not None:
            tb_dict['cgrd_theta_mean'] = (theta / np.pi).mean().detach()
            tb_dict['cgrd_logmag_mean'] = log_mag.mean().detach()
        return loss_rod, loss_ford, loss_rho, tb_dict
