# -*- coding: utf-8 -*-
"""
创新点① MS-DGE —— 多尺度密度门控体素编码（OpenPCDet / Voxel R-CNN 工程化版本）

本文件是 `创新点代码/创新点1_MS_DGE.py` 的**框架内适配版**，数学与结构与原版一致，
仅把口径从 "Pillar（z 塌缩）" 改为 "3D Voxel（保留 z）"，以直接替换 Voxel R-CNN
的 VFE（MeanVFE），无需改动 3D 稀疏骨干、2D 骨干、检测头与 ROI 头。

▍与原版的差异（唯一差异，其余逐行对应）
    原版：自己按 (vx, vy) 做柱体化，输出 voxel_coords = [b, 0, y, x]（z 恒为 0）。
    本版：不自己做体素化，直接消费 data_processor 已经产生的
          `voxels / voxel_num_points / voxel_coords`（3D 体素，含 z），
          因此 voxel_coords **原样透传**，3D 稀疏骨干可以照常工作。
          多尺度池化只在 BEV 平面（x/y）上做，z 不变 —— 与 VoxelBackBone8x 的
          stride(2,2,2)/8x 下采样后 BEV 分辨率对齐。

▍流程（与原版一一对应）
    1. 基础编码：体素内逐点 PFN + 掩码最大池化 -> 基础体素特征 (M, num_base_filters[-1])
    2. 多尺度并行：每个尺度按 SCALE_FACTORS[k] 在 xy 上做"点数加权均值池化"后回填；
          factor = 1 时该分支即恒等（＝原版"精细分支 = 基础特征"）
    3. 密度-距离软门控：
         门控输入 [各尺度特征, log(体素点数), log(中间尺度点数), 归一化距离]
         -> softmax 多尺度权重（稀疏/远处自动偏向粗糙尺度）
    4. 加权融合 + 线性压缩 -> (M, output_channels)，写回 batch_dict['voxel_features']

▍尺度换算（关键，别照抄原版的 (1,2,4)）
    原版是 Pillar 口径：基础柱体 0.4m，scale_factors=(1,2,4) -> 0.4/0.8/1.6m。
    本仓库 Voxel R-CNN 的基础体素是 0.05m，倍率必须 ×8 才落到同一组绝对尺度：
        [1, 8, 16, 32] -> 0.05(原生恒等) / 0.4 / 0.8 / 1.6 m   ← 默认
        [8, 16, 32]    ->        0.4 / 0.8 / 1.6 m（无恒等分支）
    **不要**用 [8,16,32]：三个尺度全部被池化后，0.05m 以下的细节被完全均值掉，
    3D 稀疏骨干（stride 1/2/4/8 = 0.05/0.1/0.2/0.4m）与 ROI 的 x_conv2/3 池化
    拿到的都是 0.4m 平滑后的特征，小目标与行人会明显掉点。文档 1.5 节明确写了
    "尺度数（2/3/4）" 均可，默认取 4 档即"原生 + 文档三尺度"。

▍配置项（MODEL.VFE）
    USE_NORM             [True]   各层是否用 BatchNorm
    USE_ABSOLUTE_XYZ     [True]   是否把绝对坐标拼入逐点特征
    WITH_DISTANCE        [False]  是否附加到原点距离特征
    NUM_BASE_FILTERS     [[128]]  基础 PFN 输出通道
    SCALE_FACTORS        [[1,8,16,32]] 相对基础体素的 xy 尺度倍率（须递增）
    SCALE_CHANNELS       [=num_base_filters[-1]] 各尺度分支变换通道
    OUTPUT_CHANNELS      [64]     融合压缩后输出通道（= BACKBONE_3D 的 input_channels）
    GATE_HIDDEN_CHANNELS [64]     门控网络隐层通道
    FINE_RESIDUAL        [True]   精细分支走残差（初始化即恒等，见下）
    GATE_PRIOR_DIRECT    [True]   密度/距离先验经一个零初始化的线性层直连 softmax 逻辑值
    DENSITY_PRIOR        [full]   先验消融：full / no_count / count_only / none
    DENSITY_STATS_MOMENTUM [0.01] 自适应归一化的 EMA 动量
    GATE_HARD            [False]  消融项：门控改 argmax 硬切换（证明"软配比"有用）
    GATE_DIST_EDGES      [[.25,.5]] 距离分箱边界（归一化），用于记录"权重随距离变化"

▍训练期日志（文档 5.4 要求）
    msdge_gate_entropy   归一化门控熵，恒为 1 = 门控没学到东西，0 = 退化成硬选
    msdge_gate_w_s{k}    第 k 个尺度的全局平均权重
    msdge_density_scale_{fine,coarse}  p99 归一化除数的当前估计（应随数据自适应）

▍三处"初始化即基线"的安全设计（都是为了让改进项在 step 0 不破坏基线行为）
    1. `FINE_RESIDUAL`：factor==1 分支 = base_feat + transform(base_feat)，且该分支的
       BN 权重零初始化 => 初始化时精细分支**严格等于**基础特征，不是"被一个随机线性
       层重新混合后的特征"。训练中残差才逐步注入。
    2. `GATE_PRIOR_DIRECT` 的线性层零初始化 => 初始化时门控与"只用各尺度特征"完全等价，
       先验的贡献从 0 开始学，不会一开始就把 softmax 推偏。
    3. 门控最后一层也用小方差初始化（见 _init_gate），softmax 初始接近均匀，
       即初始化时 MS-DGE ≈ 多尺度特征的等权平均 + 残差细分支。

▍依赖：仅 torch（不依赖 torch_scatter，分组用 scatter_add_ 实现）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class _EMAQuantileNorm(nn.Module):
    """把 log1p(点数) 归一到 [0,1]，归一化除数用训练集 99 分位的 EMA 估计。

    为什么不能硬编码 /5：
        KITTI 0.05m 体素的点数 99 分位只有 ~5，log1p 后整体落在 1.0~1.6，
        除以 5 之后是 0.14~0.31 —— 门控拿到的这一维几乎是常数，"密度先验"名存实亡。
        用数据自身的分位数归一化后，这一维才真正张开到 [0,1]；换成 Waymo、更大体素
        或 EIP-BS 之后的稀疏点云，也不必重新手调除数（这是原实现最容易被质疑的点）。
    不引入可学习参数，只用 buffer 做统计，因此不改变参数量与 DDP 语义。
    """

    def __init__(self, momentum=0.01, init_scale=5.0, min_scale=1.0):
        super().__init__()
        self.momentum = float(momentum)
        self.min_scale = float(min_scale)
        self.register_buffer('scale', torch.tensor(float(np.log1p(init_scale))))

    def forward(self, counts):
        v = torch.log1p(counts.clamp_min(0).float())
        if self.training and v.numel() > 0:
            with torch.no_grad():
                k = max(1, int(round(0.99 * v.numel())))
                q = torch.kthvalue(v.reshape(-1), k).values
                q = q.clamp(min=self.min_scale)
                self.scale.mul_(1.0 - self.momentum).add_(self.momentum * q)
        return v / self.scale.clamp(min=1e-3)


class _PFNLayer(nn.Module):
    """体素内逐点线性编码 + 掩码最大池化。

    非末层输出 [逐点特征, 池化特征广播] 拼接；末层只输出池化特征。
    """

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        if use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        # nn.Linear 在行数过大时结果不稳定，分块计算（与 PillarVFE 的处理一致）
        self.part = 50000

    def forward(self, inputs, mask):
        """
        Args:
            inputs: (M, T, C) 体素内逐点特征（T = MAX_POINTS_PER_VOXEL）
            mask:   (M, T, 1) 有效点掩码（1 有效 / 0 填充）
        Returns:
            (M, T, C') 或 (M, C') 末层
        """
        m, t, c = inputs.shape
        flat = inputs.reshape(m * t, c)
        if flat.shape[0] > self.part:
            outs = [self.linear(flat[i:i + self.part])
                    for i in range(0, flat.shape[0], self.part)]
            x = torch.cat(outs, dim=0)
        else:
            x = self.linear(flat)
        if self.use_norm:
            x = self.norm(x)
        x = F.relu(x).reshape(m, t, -1)
        x = x * mask                       # 填充位置清零（ReLU 后 >=0，故 max 等价于 -inf）
        x_max = x.max(dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max.squeeze(1)
        return torch.cat([x, x_max.expand(-1, t, -1)], dim=2)


class MSDGE(VFETemplate):
    """多尺度密度门控体素编码（3D 体素版）。"""

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range,
                 grid_size, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = bool(self.model_cfg.get('USE_NORM', True))
        self.use_absolute_xyz = bool(self.model_cfg.get('USE_ABSOLUTE_XYZ', True))
        self.with_distance = bool(self.model_cfg.get('WITH_DISTANCE', False))

        # ---------- 基础编码：体素内逐点 -> 基础体素特征 ----------
        in_ch = int(num_point_features) + (6 if self.use_absolute_xyz else 3)
        if self.with_distance:
            in_ch += 1

        self.num_base_filters = list(self.model_cfg.get('NUM_BASE_FILTERS', [128]))
        assert len(self.num_base_filters) > 0
        num_filters = [in_ch] + self.num_base_filters
        layers = []
        for i in range(len(num_filters) - 1):
            layers.append(_PFNLayer(
                num_filters[i], num_filters[i + 1], self.use_norm,
                last_layer=(i >= len(num_filters) - 2)))
        self.base_pfn_layers = nn.ModuleList(layers)
        base_channels = self.num_base_filters[-1]

        # ---------- 多尺度分支 ----------
        self.scale_factors = [int(v) for v in self.model_cfg.get('SCALE_FACTORS', [1, 8, 16, 32])]
        assert self.scale_factors == sorted(self.scale_factors), 'SCALE_FACTORS 必须递增'
        assert self.scale_factors[0] >= 1, 'SCALE_FACTORS 必须 >= 1'
        self.scale_channels = int(self.model_cfg.get('SCALE_CHANNELS', base_channels) or base_channels)
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_channels, self.scale_channels, bias=False),
                nn.BatchNorm1d(self.scale_channels, eps=1e-3, momentum=0.01) if self.use_norm
                else nn.Identity(),
                nn.ReLU(inplace=True),
            ) for _ in self.scale_factors
        ])
        # 精细分支残差：只在"有 factor==1 分支 且 通道数一致"时启用
        self.fine_idx = self.scale_factors.index(1) if 1 in self.scale_factors else -1
        self.fine_residual = bool(self.model_cfg.get('FINE_RESIDUAL', True)) \
            and self.fine_idx >= 0 and self.scale_channels == base_channels
        if self.fine_residual:
            bn = self.scale_transforms[self.fine_idx][1]
            if isinstance(bn, nn.BatchNorm1d):
                # BN 权重零初始化 => 初始化时 transform(base_feat) ≡ 0，
                # 精细分支严格等于 base_feat（不是"被随机线性层混合过"的特征）
                nn.init.zeros_(bn.weight)

        # ---------- 密度-距离软门控 ----------
        gate_in_channels = len(self.scale_factors) * self.scale_channels + 3
        gate_hidden = int(self.model_cfg.get('GATE_HIDDEN_CHANNELS', 64))
        self.gate = nn.Sequential(
            nn.Linear(gate_in_channels, gate_hidden),
            nn.BatchNorm1d(gate_hidden, eps=1e-3, momentum=0.01) if self.use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, len(self.scale_factors)),
        )
        # 先验直连：把 [log点数, log粗尺度点数, 距离] 用一个**零初始化**的线性层
        # 直接加到 softmax 逻辑值上。原实现把三先验和 512 维尺度特征拼在一起喂给
        # 一个 64 维隐层，先验信噪比被淹没；这条旁路让"稀疏/远处该偏向粗尺度"这个
        # 归纳偏置有一条低噪声的直达路径，且初始化时为 0，不改变初始行为。
        self.gate_prior_direct = bool(self.model_cfg.get('GATE_PRIOR_DIRECT', True))
        if self.gate_prior_direct:
            self.prior_film = nn.Linear(3, len(self.scale_factors))
            nn.init.zeros_(self.prior_film.weight)
            nn.init.zeros_(self.prior_film.bias)

        # 硬切换（消融项，文档 1.5）：门控输出去 argmax 变成 one-hot。
        # 这不是方法的一部分，只用来证明"连续软配比"确实比"离散硬选"好——
        # 没有这一行，文档 1.4 第 2 条的"软配比优于硬切换"就只是断言。
        # 注意 one-hot 处梯度为 0，门控在本组里学不到东西（这本身就是对照结论）。
        self.gate_hard = bool(self.model_cfg.get('GATE_HARD', False))

        # 距离分箱的边界（归一化距离，0~1）。用于把"各尺度权重随距离的变化"
        # 落成可读的数字：默认 0.25/0.5 × max_range(≈80m) = 20m / 40m 两个切点，
        # 正好对应 KITTI 的 near / mid / far 三段。
        self.gate_dist_edges = [float(v) for v in
                                self.model_cfg.get('GATE_DIST_EDGES', [0.25, 0.5])]

        # ---------- 密度先验消融开关 ----------
        # full / no_count / count_only / none（对应审计报告 §8 第 7 项那行消融）
        prior_mode = str(self.model_cfg.get('DENSITY_PRIOR', 'full')).lower()
        assert prior_mode in ('full', 'no_count', 'count_only', 'none'), \
            "DENSITY_PRIOR 只能是 full/no_count/count_only/none"
        self.density_prior = prior_mode
        keep = {'full': (1.0, 1.0), 'no_count': (0.0, 1.0),
                'count_only': (1.0, 0.0), 'none': (0.0, 0.0)}[prior_mode]

        # 自适应归一化（替代硬编码 /5、/8）
        mom = float(self.model_cfg.get('DENSITY_STATS_MOMENTUM', 0.01))
        self.norm_fine = _EMAQuantileNorm(momentum=mom, init_scale=5.0)
        self.norm_coarse = _EMAQuantileNorm(momentum=mom, init_scale=8.0)
        self.register_buffer('prior_keep_mask',
                             torch.tensor(list(keep), dtype=torch.float32), persistent=False)

        # ---------- 融合压缩 ----------
        self.output_channels = int(self.model_cfg.get('OUTPUT_CHANNELS', 64))
        self.compress = nn.Sequential(
            nn.Linear(self.scale_channels, self.output_channels, bias=False),
            nn.BatchNorm1d(self.output_channels, eps=1e-3, momentum=0.01) if self.use_norm
            else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # ---------- 网格几何 ----------
        self.voxel_size = [float(v) for v in voxel_size]
        self.point_cloud_range = [float(v) for v in point_cloud_range]
        self.grid_size = [int(v) for v in grid_size]
        self.vx, self.vy = self.voxel_size[0], self.voxel_size[1]
        self.x_offset = self.vx / 2 + self.point_cloud_range[0]
        self.y_offset = self.vy / 2 + self.point_cloud_range[1]
        self.max_range = float(max(
            abs(self.point_cloud_range[0]), abs(self.point_cloud_range[1]),
            abs(self.point_cloud_range[3]), abs(self.point_cloud_range[4])))

        # 调试/可视化：最近一次前向的各尺度平均融合权重
        self.last_gate_weights = None
        self.last_gate_entropy = None
        self.last_gate_dist_profile = None    # (3, K) 近/中/远三段的平均权重
        self.last_density_scale = (0.0, 0.0)  # (fine, coarse) 的 p99 归一化除数

        # 门控最后一层小方差初始化：logits ≈ 0 => softmax 初始接近均匀，
        # 即初始化时 MS-DGE ≈ "多尺度等权融合 + 残差细分支"，不会一开始就押注某一档。
        with torch.no_grad():
            last = self.gate[-1]
            last.weight.mul_(0.1)
            if last.bias is not None:
                last.bias.zero_()

    def get_output_feature_dim(self):
        return self.output_channels

    # ------------------------------------------------------------------ #
    #  内部：基础编码
    # ------------------------------------------------------------------ #
    def _encode_base(self, voxels, voxel_num_points, voxel_coords):
        """体素内逐点特征 -> (M, base_channels)。"""
        dtype = voxels.dtype
        vn = voxel_num_points.clamp(min=1).to(dtype).view(-1, 1, 1)
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / vn
        f_cluster = voxels[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxels[:, :, :3])
        f_center[:, :, 0] = voxels[:, :, 0] - (
            voxel_coords[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (
            voxel_coords[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, 2] = voxels[:, :, 2] - (
            voxel_coords[:, 1].to(dtype).unsqueeze(1) * self.voxel_size[2]
            + self.voxel_size[2] / 2 + self.point_cloud_range[2])

        if self.use_absolute_xyz:
            features = [voxels, f_cluster, f_center]
        else:
            features = [voxels[..., 3:], f_cluster, f_center]
        if self.with_distance:
            features.append(torch.norm(voxels[:, :, :3], 2, dim=2, keepdim=True))
        features = torch.cat(features, dim=-1)

        # 有效点掩码（M, T, 1）
        max_pts = voxels.shape[1]
        idx = torch.arange(max_pts, device=voxels.device).view(1, max_pts)
        mask = (idx < voxel_num_points.view(-1, 1).long()).unsqueeze(-1).to(dtype)

        for pfn in self.base_pfn_layers:
            features = pfn(features, mask)
        return features

    # ------------------------------------------------------------------ #
    #  内部：粗尺度池化回填（只在 xy 上做，z 保持不变）
    # ------------------------------------------------------------------ #
    def _pool_to_scale(self, base_feat, counts, voxel_coords, factor):
        """把体素特征按 factor 在 xy 上做点数加权均值池化，再回填到每个体素。

        Returns:
            pooled_up:       (M, C) 所属粗 cell 的上下文（回填后）
            coarse_counts_up:(M,)   所属粗 cell 的总点数
        """
        nx, ny = self.grid_size[0], self.grid_size[1]
        bi = voxel_coords[:, 0].long()
        yi = voxel_coords[:, 2].long()
        xi = voxel_coords[:, 3].long()

        nxc = (nx + factor - 1) // factor
        nyc = (ny + factor - 1) // factor
        xc = torch.div(xi, factor, rounding_mode='floor')
        yc = torch.div(yi, factor, rounding_mode='floor')
        key = (bi * nxc + xc) * nyc + yc

        unq, inv = torch.unique(key, return_inverse=True)
        num_c = unq.numel()

        c = base_feat.shape[1]
        w = counts.to(base_feat.dtype).view(-1, 1)
        summed = base_feat.new_zeros((num_c, c)).scatter_add_(
            0, inv.view(-1, 1).expand(-1, c), base_feat * w)
        wsum = w.new_zeros((num_c, 1)).scatter_add_(0, inv.view(-1, 1), w).clamp(min=1e-6)
        pooled = summed / wsum

        return pooled[inv], wsum[inv].squeeze(-1)

    # ------------------------------------------------------------------ #
    #  内部：点数（真实值优先）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_counts(batch_dict, voxels):
        """返回每个体素的**真实**点数（M,），缺失时退回被截断的 voxel_num_points。

        `voxel_num_points` 被 MAX_POINTS_PER_VOXEL 截断（稠密分支 5、稀疏分支 8），
        于是 log1p(counts) 在近场几乎恒为 log1p(8)，密度先验退化成常数；点数加权
        池化的权重也不再等于"真实点数"，与文档 1.3(1) 的"等价于该大 cell 内全部
        原始点的均值、无信息偏置"不符。data_processor 的 eip_bs_voxelize 会把真实
        点数写进 voxel_num_points_true_kd，这里优先使用。
        """
        m = voxels.shape[0]
        for key in ('voxel_num_points_true', 'voxel_num_points_true_kd'):
            t = batch_dict.get(key, None)
            if t is not None and t.numel() == m:
                return t.reshape(-1).to(voxels.dtype).clamp(min=1.0)
        return batch_dict['voxel_num_points'].reshape(-1).to(voxels.dtype)

    # ------------------------------------------------------------------ #
    #  前向
    # ------------------------------------------------------------------ #
    def forward(self, batch_dict, **kwargs):
        voxels = batch_dict['voxels']                    # (M, T, C)
        voxel_num_points = batch_dict['voxel_num_points']  # (M,)
        voxel_coords = batch_dict['voxel_coords']        # (M, 4) [b, z, y, x]

        # 池化权重与密度先验都用真实点数（未被 MAX_POINTS_PER_VOXEL 截断）
        counts = self._resolve_counts(batch_dict, voxels)

        # 1. 基础编码
        base_feat = self._encode_base(voxels, voxel_num_points, voxel_coords)

        # 2. 多尺度分支：每个尺度都按自己的 factor 做点数加权池化后回填
        #    factor == 1 时退化为恒等（＝原版 SCALE_FACTORS 以 1 开头的情形）
        branch_feats = []
        coarse_counts = None
        # 门控里的"粗尺度点数"取绝对尺度最接近 0.4m（= 基础体素 ×8）的那一档，
        # 这样即使把 SCALE_FACTORS 换成 [8,16,32] 或 [4,8,16]，先验口径也不漂移。
        coarse_k = int(np.argmin([abs(np.log(f) - np.log(8.0)) for f in self.scale_factors]))
        for k, factor in enumerate(self.scale_factors):
            if factor == 1:
                src_feat, src_counts = base_feat, counts
            else:
                src_feat, src_counts = self._pool_to_scale(
                    base_feat, counts, voxel_coords, factor)
            out = self.scale_transforms[k](src_feat)
            if k == self.fine_idx and self.fine_residual:
                out = out + base_feat          # 初始化时 transform≡0，故 out ≡ base_feat
            branch_feats.append(out)
            if k == coarse_k:
                coarse_counts = src_counts
        if coarse_counts is None:                        # 只有 1 个尺度时退化为自身点数
            coarse_counts = counts

        # 3. 密度-距离软门控
        x_center = voxel_coords[:, 3].to(voxels.dtype) * self.vx + self.x_offset
        y_center = voxel_coords[:, 2].to(voxels.dtype) * self.vy + self.y_offset
        dist = torch.norm(torch.stack([x_center, y_center], dim=1), dim=1) / max(self.max_range, 1e-6)

        # 三个先验的量纲对齐：log1p(点数) 用训练集 99 分位的 EMA 归一化到 ~[0,1]
        # （硬编码 /5 会让这一维在 KITTI 上几乎恒为常数，见 _EMAQuantileNorm 注释）；
        # 距离本来就已除以雷达最大量程。
        keep_mask = self.prior_keep_mask.to(counts.dtype)
        density_feats = torch.stack([
            self.norm_fine(counts) * keep_mask[0],
            self.norm_coarse(coarse_counts) * keep_mask[1],
            dist,
        ], dim=1)
        # AMP 下 log1p/norm 被提升到 fp32，而分支特征是 fp16 时 torch.cat 会直接报
        # "expected all tensors to be the same type"，这里统一到分支特征的 dtype。
        density_feats = density_feats.to(branch_feats[0].dtype)

        gate_logits = self.gate(torch.cat(branch_feats + [density_feats], dim=1))
        if self.gate_prior_direct:
            # 先验直达 softmax 逻辑值（零初始化启动，见 __init__）
            gate_logits = gate_logits + self.prior_film(density_feats)
        if self.gate_hard:
            idx = gate_logits.argmax(dim=1)
            gate_weights = F.one_hot(idx, num_classes=len(self.scale_factors)).to(gate_logits.dtype)
        else:
            gate_weights = torch.softmax(gate_logits, dim=1)

        fused = sum(gate_weights[:, k:k + 1] * branch_feats[k]
                    for k in range(len(self.scale_factors)))

        # 4. 压缩输出
        #    .float()：USE_NORM=False 时 Linear 输出在 AMP 下是 fp16，而 spconv 的稀疏
        #    卷积要求 fp32（基线 MeanVFE 因 sum 被提升为 fp32 才没暴露），显式对齐。
        batch_dict['voxel_features'] = self.compress(fused).float()
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_num_points'] = voxel_num_points

        with torch.no_grad():
            gw = gate_weights.detach()
            self.last_gate_weights = gw.mean(dim=0)
            # 门控熵（归一化到 [0,1]）：接近 1 = 各尺度权重接近均匀（门控没学到东西），
            # 接近 0 = 每处都硬选某一档。训练时应缓慢下降但不塌到 0。
            p = gw.clamp_min(1e-8)
            ent = -(p * p.log()).sum(dim=1).mean() / np.log(len(self.scale_factors))
            self.last_gate_entropy = ent

            # 各尺度权重随距离的变化（近/中/远三段）。这是"密度-距离门控确实
            # 按距离改变配比"的唯一直接证据：健康情况应从近处的细尺度主导
            # 平滑过渡到远处的粗尺度主导，而不是三段几乎相同。
            d_np = dist.detach()
            edges = [0.0] + self.gate_dist_edges + [1.0 + 1e-6]
            rows = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (d_np >= lo) & (d_np < hi)
                if bool(m.any()):
                    rows.append(gw[m].mean(dim=0))
                else:
                    rows.append(torch.zeros_like(self.last_gate_weights))
            self.last_gate_dist_profile = torch.stack(rows, dim=0)     # (3, K)

        # 归一化除数的当前估计（换个数据集/体素大小后它应该自己变，而不是靠手调）
        self.last_density_scale = (float(self.norm_fine.scale.item()),
                                   float(self.norm_coarse.scale.item()))
        return batch_dict
