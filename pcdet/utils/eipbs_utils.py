# -*- coding: utf-8 -*-
"""
创新点③ EIP-BS —— 边缘重要性保持渐进波束采样（OpenPCDet 工程化版本）

本文件是 `创新点代码/创新点3_EIP_BS.py` 的框架内适配版，算法与超参完全一致，
仅补上"在 dataloader 里作为数据处理器被调用"所需的工程细节：

    - 单帧接口 __call__(points, gt_boxes) -> sampled_points  （与 dataloader 对齐）
    - 批量接口 EIPBS.apply_to_batch(...)                       （在线蒸馏用）
    - 打分权重、类优先级的向量化实现（原版逐框 for 循环 -> numpy 广播）
    - USE_NUMBA 开关：True 用 kNN 逐点曲率（原版精度），False 用网格协方差
      近似（全向量化，dataloader 多进程下更快）

被 `DataProcessor.eip_bs_voxelize` 调用，用于产生学生分支的"16 线"点云。
"""

import time

import numpy as np

try:
    import numba
    _NUMBA_IMPORTED = True
except Exception:
    _NUMBA_IMPORTED = False


# --------------------------------------------------------------------------- #
#  逐点 kNN 曲率 + 半径邻域密度（numba 版，与独立版一致）
# --------------------------------------------------------------------------- #
if _NUMBA_IMPORTED:
    @numba.njit(cache=True)
    def _knn_curvature_density(xyz, cell_size, k, radius, max_candidates):
        N = xyz.shape[0]
        curv = np.zeros(N, dtype=np.float64)
        dens = np.zeros(N, dtype=np.float64)
        if N < 4:
            return curv, dens

        inv_cell = 1.0 / cell_size
        gx = np.empty(N, dtype=np.int64)
        gy = np.empty(N, dtype=np.int64)
        gz = np.empty(N, dtype=np.int64)
        for i in range(N):
            gx[i] = int(np.floor(xyz[i, 0] * inv_cell))
            gy[i] = int(np.floor(xyz[i, 1] * inv_cell))
            gz[i] = int(np.floor(xyz[i, 2] * inv_cell))

        OFF = 1 << 20
        keys = np.empty(N, dtype=np.int64)
        for i in range(N):
            keys[i] = ((gx[i] + OFF) << 42) | ((gy[i] + OFF) << 21) | (gz[i] + OFF)
        order = np.argsort(keys)
        sorted_keys = keys[order]

        r2 = radius * radius
        dist2_best = np.empty(k, dtype=np.float64)
        nn_idx = np.empty(k, dtype=np.int64)
        cov = np.empty((3, 3))

        for i in range(N):
            n_best = 0
            cnt_radius = 0
            cand = 0
            for d in range(27):
                ddx = d // 9 - 1
                ddy = (d // 3) % 3 - 1
                ddz = d % 3 - 1
                ck = ((gx[i] + ddx + OFF) << 42) | ((gy[i] + ddy + OFF) << 21) | (gz[i] + ddz + OFF)
                lo = np.searchsorted(sorted_keys, ck, side='left')
                hi = np.searchsorted(sorted_keys, ck, side='right')
                for t in range(lo, hi):
                    j = order[t]
                    if j == i:
                        continue
                    ex = xyz[j, 0] - xyz[i, 0]
                    ey = xyz[j, 1] - xyz[i, 1]
                    ez = xyz[j, 2] - xyz[i, 2]
                    dd2 = ex * ex + ey * ey + ez * ez
                    if dd2 <= r2:
                        cnt_radius += 1
                    if n_best < k:
                        pos = n_best
                        while pos > 0 and dist2_best[pos - 1] > dd2:
                            dist2_best[pos] = dist2_best[pos - 1]
                            nn_idx[pos] = nn_idx[pos - 1]
                            pos -= 1
                        dist2_best[pos] = dd2
                        nn_idx[pos] = j
                        n_best += 1
                    elif dd2 < dist2_best[k - 1]:
                        pos = k - 1
                        while pos > 0 and dist2_best[pos - 1] > dd2:
                            dist2_best[pos] = dist2_best[pos - 1]
                            nn_idx[pos] = nn_idx[pos - 1]
                            pos -= 1
                        dist2_best[pos] = dd2
                        nn_idx[pos] = j
                    cand += 1
                    if cand >= max_candidates:
                        break
                if cand >= max_candidates:
                    break
            dens[i] = cnt_radius

            if n_best >= 3:
                sx = sy = sz = 0.0
                sxx = sxy = sxz = syy = syz = szz = 0.0
                for t in range(n_best):
                    px = xyz[nn_idx[t], 0]
                    py = xyz[nn_idx[t], 1]
                    pz = xyz[nn_idx[t], 2]
                    sx += px; sy += py; sz += pz
                    sxx += px * px; sxy += px * py; sxz += px * pz
                    syy += py * py; syz += py * pz; szz += pz * pz
                n = float(n_best)
                mx = sx / n; my = sy / n; mz = sz / n
                cov[0, 0] = sxx / n - mx * mx
                cov[0, 1] = sxy / n - mx * my
                cov[0, 2] = sxz / n - mx * mz
                cov[1, 0] = cov[0, 1]
                cov[1, 1] = syy / n - my * my
                cov[1, 2] = syz / n - my * mz
                cov[2, 0] = cov[0, 2]
                cov[2, 1] = cov[1, 2]
                cov[2, 2] = szz / n - mz * mz
                w_eig, _ = np.linalg.eigh(cov)
                s = w_eig[0] + w_eig[1] + w_eig[2]
                if s > 1e-12:
                    curv[i] = w_eig[0] / s
        return curv, dens


_NUMBA_RUNTIME_FAILED = False


def _grid_curvature_density(xyz, cell_size):
    """无 numba 时的全向量化回退：按网格单元统计协方差（同一单元内曲率相同）。"""
    N = xyz.shape[0]
    cells = np.floor(xyz[:, :3] / cell_size).astype(np.int64)
    cells -= cells.min(axis=0, keepdims=True)
    nx, ny, nz = (cells.max(axis=0) + 1).tolist()
    keys = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
    _, inv = np.unique(keys, return_inverse=True)
    num_cells = int(inv.max()) + 1

    # 用 bincount 做分组求和：比 np.add.at 快约一个数量级（11 万点 ~0.3s -> ~0.01s）
    cnts = np.bincount(inv, minlength=num_cells).astype(np.float64)
    xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    def _cell_mean(vals):
        return np.bincount(inv, weights=vals, minlength=num_cells) / cnts

    mx, my, mz = _cell_mean(xs), _cell_mean(ys), _cell_mean(zs)
    # 二阶中心矩（E[xy] - E[x]E[y]），只算 6 个独立分量
    cxx = _cell_mean(xs * xs) - mx * mx
    cyy = _cell_mean(ys * ys) - my * my
    czz = _cell_mean(zs * zs) - mz * mz
    cxy = _cell_mean(xs * ys) - mx * my
    cxz = _cell_mean(xs * zs) - mx * mz
    cyz = _cell_mean(ys * zs) - my * mz

    cov = np.empty((num_cells, 3, 3), dtype=np.float64)
    cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2] = cxx, cyy, czz
    cov[:, 0, 1] = cov[:, 1, 0] = cxy
    cov[:, 0, 2] = cov[:, 2, 0] = cxz
    cov[:, 1, 2] = cov[:, 2, 1] = cyz

    w_eig = np.linalg.eigvalsh(cov)                       # (num_cells, 3) 升序
    curv = w_eig[:, 0] / (w_eig.sum(axis=1) + 1e-12)
    return curv[inv], cnts[inv]


def _curvature_density(xyz, cell_size, k, radius, use_numba=True):
    """曲率 + 密度估计统一入口（kNN 优先，失败/关闭时回退网格统计）。"""
    global _NUMBA_RUNTIME_FAILED
    if use_numba and _NUMBA_IMPORTED and not _NUMBA_RUNTIME_FAILED:
        try:
            return _knn_curvature_density(
                np.ascontiguousarray(xyz[:, :3], dtype=np.float64),
                float(cell_size), int(k), float(radius), 128)
        except Exception:
            _NUMBA_RUNTIME_FAILED = True
    return _grid_curvature_density(
        np.ascontiguousarray(xyz[:, :3], dtype=np.float64), float(cell_size))


def _robust_norm01(x):
    """鲁棒归一化到 [0,1]：1%~99% 分位数截断 + min-max；常量输入映射 0.5。"""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 1e-8:
        return np.full_like(x, 0.5)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def true_point_counts(points, voxel_coords, voxel_size, point_cloud_range, grid_size=None):
    """统计体素化之后每个体素里的**真实**点数（不受 MAX_POINTS_PER_VOXEL 截断）。

    spconv 的 VoxelGenerator 返回的 num_points 被 max_points_per_voxel 截断，
    直接拿它做 MS-DGE 的密度先验与点数加权池化会损失"到底有多密"的信息。
    这里用原始点云反查坐标，重新计数。

    Args:
        points:            (N, C>=3) 参与体素化的点云（单帧）
        voxel_coords:      (M, 3) [z,y,x] 或 (M, 4) [batch,z,y,x]
        voxel_size:        [vx, vy, vz]
        point_cloud_range: [x0,y0,z0,x1,y1,z1]
        grid_size:         可选，(nx,ny,nz)，缺省由 range/voxel_size 推
    Returns:
        counts: (M,) int64
    """
    vc = np.asarray(voxel_coords)
    if vc.ndim != 2 or vc.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if vc.shape[1] == 4:                      # 兼容已带 batch 列的情形
        vc = vc[:, 1:]
    vc = vc.astype(np.int64, copy=False)

    vs = np.asarray(voxel_size, dtype=np.float64)[:3]
    pcr = np.asarray(point_cloud_range, dtype=np.float64)
    if grid_size is None:
        grid = np.round((pcr[3:6] - pcr[0:3]) / vs).astype(np.int64)
    else:
        grid = np.asarray(grid_size, dtype=np.int64)[:3]
    nx, ny, nz = int(grid[0]), int(grid[1]), int(grid[2])

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    pidx = np.floor((pts - pcr[0:3]) / vs).astype(np.int64)
    ok = np.all((pidx >= 0) & (pidx < np.array([nx, ny, nz], dtype=np.int64)), axis=1)
    if not ok.any():
        return np.zeros(vc.shape[0], dtype=np.int64)
    pidx = pidx[ok]

    # 用 (z,y,x) 行主序编码成唯一 key，与 voxel_coords 的列序一致
    key_p = (pidx[:, 2] * ny + pidx[:, 1]) * nx + pidx[:, 0]
    key_v = (vc[:, 0] * ny + vc[:, 1]) * nx + vc[:, 2]

    order = np.argsort(key_v, kind='stable')
    sorted_keys = key_v[order]
    pos = np.searchsorted(sorted_keys, key_p)
    np.clip(pos, 0, len(order) - 1, out=pos)
    hit = sorted_keys[pos] == key_p
    counts = np.bincount(order[pos[hit]], minlength=vc.shape[0]).astype(np.int64)
    return counts


# --------------------------------------------------------------------------- #
#  EIP-BS 主类
# --------------------------------------------------------------------------- #
class EIPBS(object):
    """边缘重要性保持渐进波束采样（64 线 -> 16 线）。

    打分：s(p) = w_curv * κ̂ + w_dens * ρ̂ + w_cls * ĉ_t
    渐进丢线：按 stages 逐阶段丢弃重要性最低的线束；
              被丢线束内打分最高的 keep_ratio_per_drop 比例点被救援保留。

    ▍四个"论文级"开关（相对文档原式新增，全部有真实 KITTI 实测支撑，
      见 docs/SDR-理论-实现对照与优化.md 与 tools/bench_eipbs.py）
    1. `target_keep_ratio` —— **等点数预算**。原实现保留 0.43 的点，而真 16 线只有
       0.25，多拿 1.7 倍数据足以解释掉全部增益。设为 0.25 后，"保住了哪些点"才是
       唯一自变量。（0.25 = 16/64，也便于和"均匀抽 16 线"逐点对比）
    2. `stratified=False`（默认）—— **不分层**。分层的动机是"让存活线束在仰角上
       均匀、更像真雷达"，但实测在等预算下反而有害：它强制保留"这一仰角组里最好
       的那条线束"，即使那条线上没有值得保留的东西，于是别的车被牺牲。
       实测（200 帧真实 KITTI，25% 预算，无 GT 即测试期行为）：
           均匀抽线      车≥50%点数覆盖 = 0.14
           分层 + topk   0.18
           不分层 + topk 0.35   ← 默认
       代价是不分层会打破"线束"结构（仰角间隙变大），所以**不能**再用
       "模拟 16 线雷达"来描述它，应描述为"等点预算的边缘重要性稀疏化"。
       若论文必须坚持 16 线口径，把 STRATIFIED 与 ALIGN_KEEP_STRUCTURE 同时设 True，
       接受覆盖率的下降（0.24）。
    3. `align_keep_structure` —— 预算怎么花。False（默认）= 全局按重要性花；
       True = 先砍被丢线束的救援点、再在存活线束内等比削减（保住"哪 16 条线"）。
    4. `mode` —— 对照组。`uniform_beam`（均匀抽线）/`random` 让"纯稀疏基线"与
       本方法走同一条代码路径，只差"选哪些点"。
    """

    def __init__(self,
                 src_beams=64,
                 dst_beams=16,
                 stages=None,
                 w_curvature=0.4,
                 w_density=0.3,
                 w_class=0.3,
                 curvature_k=8,
                 curvature_cell_size=0.5,
                 density_radius=1.0,
                 class_priority=(0.6, 1.0, 1.0),
                 keep_ratio_per_drop=0.2,
                 min_rescue_per_beam=1,
                 beam_score='mean',
                 beam_top_ratio=0.1,
                 beam_top_k=16,
                 beam_score_thresh=0.6,
                 near_margin=0.5,
                 use_numba=True,
                 seed=None,
                 mode='importance',
                 stratified=False,
                 target_keep_ratio=None,
                 align_keep_structure=False,
                 log_stats=True):
        self.src_beams = int(src_beams)
        self.dst_beams = int(dst_beams)
        if stages is None:
            s = [self.src_beams]
            while s[-1] > self.dst_beams:
                nxt = max(int(round(s[-1] * 0.7)), self.dst_beams)
                if nxt >= s[-1]:
                    nxt = s[-1] - 1
                s.append(max(nxt, self.dst_beams))
            if s[-1] != self.dst_beams:
                s.append(self.dst_beams)
            stages = s
        self.stages = [int(v) for v in stages]
        assert self.stages[0] >= self.stages[-1], 'stages 必须递减'
        assert self.stages[-1] == self.dst_beams, 'stages 末项必须等于 dst_beams'

        self.w_curvature = float(w_curvature)
        self.w_density = float(w_density)
        self.w_class = float(w_class)
        self.curvature_k = int(curvature_k)
        self.curvature_cell_size = float(curvature_cell_size)
        self.density_radius = float(density_radius)
        self.class_priority = [float(v) for v in class_priority]
        self.keep_ratio_per_drop = float(keep_ratio_per_drop)
        # 每条被丢线束的救援下限。设为 1 时（原版行为）最终保留率 > 16/64，
        # 与"真实 16 线只保留 25% 的点"不可比；要严格 25% 请用 target_keep_ratio=0.25
        # （推荐）或 keep_ratio_per_drop=0 且 min_rescue_per_beam=0。
        self.min_rescue_per_beam = int(min_rescue_per_beam)
        # 单条线束的重要性统计方式（文档写的是 mean，但实测 mean 会被整条线束上的
        # 地面/背景点稀释，见 docs/SDR-审计报告.md §1.1）：
        #   mean  —— 文档原式，线束内全部存活点的平均分
        #   top   —— 线束内得分最高的 beam_top_ratio 比例的均值（比例口径，
        #            点数多的线束天然拿到更极端的极值，存在尺寸偏置）
        #   topk  —— 线束内最高 k 个点的均值，k 为**绝对数** beam_top_k，
        #            与线束点数无关，消除尺寸偏置（推荐，默认）
        #   max   —— 线束内最高分（对单点噪声最敏感）
        #   count —— 线束内得分 >= beam_score_thresh 的点数（"这条线上有多少
        #            真正值得保留的结构"，语义最直白，但对近处密集线束有偏）
        assert beam_score in ('mean', 'top', 'topk', 'max', 'count'), \
            "BEAM_SCORE 只能是 mean/top/topk/max/count"
        self.beam_score = beam_score
        self.beam_top_ratio = float(beam_top_ratio)
        self.beam_top_k = int(beam_top_k)
        self.beam_score_thresh = float(beam_score_thresh)
        self.near_margin = float(near_margin)
        self.use_numba = bool(use_numba)

        # ---- 分层丢线（保仰角均匀性）----
        self.stratified = bool(stratified)

        # ---- 等点数对齐：None = 关闭（原始行为）----
        assert target_keep_ratio is None or 0.0 < float(target_keep_ratio) <= 1.0
        self.target_keep_ratio = None if target_keep_ratio is None else float(target_keep_ratio)
        # 预算怎么花（只在 target_keep_ratio 生效时起作用）：
        #   True  —— 保线束结构：先砍被丢线束的救援点，再在存活线束内等比削减。
        #            结果 gap≈7（近似真 16 线的均匀仰角间隔），但 25% 预算下
        #            前景保留只有 ~0.35（走的是"选哪 16 条线"，不是"保哪些点"）。
        #   False —— 全局按重要性花预算：等价于"等点数的边缘重要性采样"，
        #            线束结构基本被打破（gap 大），但 25% 预算下前景保留可达 ~0.73。
        # 实测见 docs/SDR-理论-实现对照与优化.md 第 2 节。
        self.align_keep_structure = bool(align_keep_structure)

        # ---- 采样模式（对照组）----
        #   importance   —— 本文方法
        #   uniform_beam —— 均匀抽线（每 src/dst 条留 1 条），模拟真实低线束雷达
        #   random       —— 随机保留同样比例的点（最弱对照，等价于随机降采样）
        assert mode in ('importance', 'uniform_beam', 'random'), \
            "MODE 只能是 importance/uniform_beam/random"
        self.mode = mode

        self.rng = np.random.RandomState(seed)
        # 统计最近一次调用的信息，便于日志/自检
        self.last_info = {}
        self.last_keep = None
        self._last_inside = None
        # 是否统计前景保留率（训练期有 GT 才有意义；多一次 (N,K) 判定的开销）
        self.log_stats = bool(log_stats)

    # ---------------- 三个打分成分 ----------------

    def compute_curvature(self, points):
        return _curvature_density(points, self.curvature_cell_size, self.curvature_k,
                                  self.density_radius, self.use_numba)[0]

    def compute_density(self, points):
        return _curvature_density(points, self.curvature_cell_size, self.curvature_k,
                                  self.density_radius, self.use_numba)[1]

    def _box_relations(self, points, gt_boxes):
        """一次算好 (inside, near, prios)；无有效框时返回 None。

        抽出来是为了让 `compute_class_scores` 与 `foreground_mask` 共用同一份
        (N, K) 广播结果 —— 训练期每帧都要报"前景保留率"，若为此再算一遍旋转框
        包含关系，等于把 60~90ms 的采样再拖慢 10~20ms。
        """
        self._last_inside = None
        n = points.shape[0]
        if gt_boxes is None or len(gt_boxes) == 0 or n == 0:
            return None

        xyz = np.asarray(points[:, :3], dtype=np.float64)
        boxes = np.asarray(gt_boxes, dtype=np.float64)
        if boxes.ndim == 3:
            boxes = boxes.reshape(-1, boxes.shape[-1])
        boxes = boxes[np.abs(boxes).sum(axis=1) > 1e-6]      # 去掉全零填充行
        if len(boxes) == 0:
            return None

        # 兼容两种输入：(M,8) 带标签列 / (M,7) 无标签列（一律当第 1 类）
        if boxes.shape[1] >= 8:
            labels = np.rint(boxes[:, 7]).astype(np.int64)
        else:
            labels = np.ones(len(boxes), dtype=np.int64)
        valid = (labels > 0) & (labels <= len(self.class_priority))
        if not valid.any():
            return None
        boxes, labels = boxes[valid], labels[valid]
        prios = np.asarray(self.class_priority, dtype=np.float64)[labels - 1]

        # (N, K) 广播：旋转到每个框的局部坐标系
        rel = xyz[:, None, :] - boxes[None, :, :3]
        cos_h = np.cos(-boxes[:, 6])[None, :]
        sin_h = np.sin(-boxes[:, 6])[None, :]
        lx = rel[:, :, 0] * cos_h - rel[:, :, 1] * sin_h
        ly = rel[:, :, 0] * sin_h + rel[:, :, 1] * cos_h
        lz = rel[:, :, 2]

        hx = boxes[:, 3][None, :] / 2.0
        hy = boxes[:, 4][None, :] / 2.0
        hz = boxes[:, 5][None, :] / 2.0
        m = self.near_margin
        inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)
        near = (np.abs(lx) <= hx + m) & (np.abs(ly) <= hy + m) & (np.abs(lz) <= hz + m) & (~inside)
        self._last_inside = inside.any(axis=1)      # 供 foreground_mask 复用
        return inside, near, prios

    def foreground_mask(self, points, gt_boxes):
        """(N,) bool：点是否落在任一 GT 框内；无 GT 时返回 None。

        只用于训练期统计（"前景点保留率"），不参与采样决策。同一次 `__call__`
        里 `compute_scores` 刚算过就复用，避免重复做 (N,K) 的框内判定。
        """
        if self._last_inside is not None and self._last_inside.shape[0] == points.shape[0]:
            return self._last_inside
        if self._box_relations(points, gt_boxes) is None:
            return None
        return self._last_inside

    def compute_class_scores(self, points, gt_boxes):
        """类别先验 ĉ_t：框内 = class_priority[label]，框外近邻 = 一半，背景 = 0。"""
        n = points.shape[0]
        rel = self._box_relations(points, gt_boxes)
        if rel is None:
            return np.zeros(n, dtype=np.float64)
        inside, near, prios = rel
        scores = np.max(np.where(inside, prios[None, :], 0.0), axis=1)
        near_score = np.max(np.where(near, prios[None, :] * 0.5, 0.0), axis=1)
        return np.maximum(scores, near_score)

    def compute_scores(self, points, gt_boxes=None):
        curv, dens = _curvature_density(points, self.curvature_cell_size, self.curvature_k,
                                        self.density_radius, self.use_numba)
        kappa = _robust_norm01(curv)
        rho = 1.0 - _robust_norm01(dens)          # 密度取反：稀疏区域得分高
        cls = self.compute_class_scores(points, gt_boxes)
        has_gt = gt_boxes is not None and len(gt_boxes) > 0

        if has_gt:
            # 注意：**不能**再对 cls 做 _robust_norm01。
            # compute_class_scores 的输出本来就落在 [0,1]（框内=prio、近邻=prio/2、
            # 背景=0），与文档 3.3(1) 的定义一致；再做 p1~p99 的 min-max 会出问题：
            # 当前景点占全帧 < 1% 时（KITTI 只有 1~2 辆远车时很常见）99 分位数仍为 0，
            # lo==hi -> _robust_norm01 返回全 0.5 常数，类别项于是退化成给所有点加
            # 同一个偏置 —— 对"线束平均打分"的排序毫无影响，即创新点③ 的语义线索
            # 被静默丢弃（原独立版 创新点3_EIP_BS.py 的 compute_scores 有这个问题）。
            c_hat = np.clip(cls, 0.0, 1.0)
            return self.w_curvature * kappa + self.w_density * rho + self.w_class * c_hat

        total = self.w_curvature + self.w_density
        return (self.w_curvature * kappa + self.w_density * rho) / max(total, 1e-6)

    # ---------------- 波束分配与渐进降采样 ----------------

    @staticmethod
    def assign_beams(points, num_beams):
        """按仰角把点分配到 num_beams 个模拟线束（均匀分 bin）。"""
        xyz = points[:, :3]
        elev = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
        lo, hi = elev.min(), elev.max()
        if hi - lo < 1e-8:
            return np.zeros(len(points), dtype=np.int64)
        beam_id = ((elev - lo) / (hi - lo) * num_beams).astype(np.int64)
        return np.clip(beam_id, 0, num_beams - 1)

    def beam_importance(self, s):
        """一条线束的重要性（s 为该线束内当前存活点的打分）。"""
        if s.size == 0:
            return -1.0
        if self.beam_score == 'max':
            return float(s.max())
        if self.beam_score == 'count':
            return float(np.count_nonzero(s >= self.beam_score_thresh))
        if self.beam_score == 'topk':
            k = min(self.beam_top_k, s.size)
            if k >= s.size:
                return float(s.mean())
            return float(np.partition(s, s.size - k)[s.size - k:].mean())
        if self.beam_score == 'top':
            k = max(1, int(np.ceil(s.size * self.beam_top_ratio)))
            if k >= s.size:
                return float(s.mean())
            return float(np.partition(s, s.size - k)[s.size - k:].mean())
        return float(s.mean())

    # ---------------- 渐进丢线 ----------------

    def _drop_beams(self, drop_ids, buckets, keep, alive, scores):
        """把 drop_ids 里的线束丢掉（线束内按分数救援 keep_ratio_per_drop）。"""
        for b in drop_ids:
            alive[b] = False
            m = buckets[b]
            m = m[keep[m]]
            if m.size == 0:
                continue
            n_keep = max(int(np.ceil(m.size * self.keep_ratio_per_drop)),
                         self.min_rescue_per_beam)
            n_keep = min(n_keep, m.size)
            order = np.argsort(-scores[m], kind='stable')
            keep[m] = False
            if n_keep > 0:
                keep[m[order[:n_keep]]] = True

    def progressive_reduce(self, points, scores, beam_ids):
        """渐进式丢线。

        `stratified=True`：每阶段把存活线束按**仰角顺序**切成若干连续组，
            每组只保留重要性最高的 ceil(len(group) * target/current) 条。
            64→48→32→16 时组数恒为 16（每组 4/3/2 留 3/2/1），
            等价于"每组只丢一个"，存活线束在仰角上近似均匀间隔。
        `stratified=False`：全局按重要性排序丢最差的若干条（文档原式）。
        """
        n = points.shape[0]
        keep = np.ones(n, dtype=bool)
        present = np.unique(beam_ids)
        cur_count = len(present)
        alive = np.ones(int(present.max()) + 1, dtype=bool)

        # 一次 argsort 建好"线束 -> 点索引"的桶（比逐线束 np.where 快一个数量级：
        # 后者是 64 次 O(N) 扫描，11 万点时要多花几十毫秒）
        order_all = np.argsort(beam_ids, kind='stable')
        sorted_b = beam_ids[order_all]
        lo = np.searchsorted(sorted_b, present, side='left')
        hi = np.searchsorted(sorted_b, present, side='right')
        buckets = {int(b): order_all[l:h] for b, l, h in zip(present, lo, hi)}

        for target in self.stages[1:]:
            if cur_count <= target:
                continue
            alive_list = [int(b) for b in present if alive[int(b)]]
            if not alive_list:                     # 理论不可达，防御
                break

            if self.stratified:
                ratio = float(target) / float(len(alive_list))
                # 组大小：让"每组保留数"尽量接近整数、且组数尽量多（间距均匀）
                group_size = self._pick_group_size(len(alive_list), ratio)
                drop_ids = []
                for i in range(0, len(alive_list), group_size):
                    grp = alive_list[i:i + group_size]
                    if len(grp) <= 1:
                        continue
                    k_keep = max(1, int(round(len(grp) * ratio)))
                    if k_keep >= len(grp):
                        continue
                    imp = [(self.beam_importance(scores[buckets[b][keep[buckets[b]]]]), b)
                           for b in grp]
                    imp.sort(key=lambda t: (t[0], t[1]))       # 分数升序，同分按 beam id
                    drop_ids.extend([b for _, b in imp[:len(grp) - k_keep]])
            else:
                imp = [(self.beam_importance(scores[buckets[b][keep[buckets[b]]]]), b)
                       for b in alive_list]
                imp.sort(key=lambda t: (t[0], t[1]))
                drop_ids = [b for _, b in imp[:cur_count - target]]

            self._drop_beams(drop_ids, buckets, keep, alive, scores)
            cur_count = len([b for b in alive_list if alive[b]])

            # 分层模式下组内取整可能多留 1~2 条：用全局最差补齐，保证逐阶段收敛
            extra = cur_count - target
            if extra > 0:
                alive_now = [b for b in present if alive[int(b)]]
                imp = [(self.beam_importance(scores[buckets[int(b)][keep[buckets[int(b)]]]]), int(b))
                       for b in alive_now]
                imp.sort(key=lambda t: (t[0], t[1]))
                self._drop_beams([b for _, b in imp[:extra]], buckets, keep, alive, scores)
                cur_count -= extra
        return keep, alive

    @staticmethod
    def _pick_group_size(n_alive, ratio):
        """选组大小：只考虑能整除 n_alive 的 g（否则各组点数不等、阶段数对不上），
        在其中取"每组保留数 len(group)*ratio 最接近整数"、且组数尽量多的那个。"""
        cands = [g for g in range(2, min(9, n_alive) + 1) if n_alive % g == 0]
        best, best_err = 1, 1e9
        for g in cands:
            k = g * ratio                          # 每组的保留数
            err = abs(k - round(k))
            if round(k) < 1:
                continue
            if err < best_err - 1e-9:
                best, best_err = g, err
            if best_err < 1e-6:
                break
        return best

    # ---------------- 对照组采样 ----------------

    def _uniform_beam_keep(self, points, beam_ids):
        """均匀抽线：每 period 条线束保留 1 条（period = src/dst）。"""
        period = max(1, int(round(self.src_beams / float(self.dst_beams))))
        offset = int(self.rng.randint(0, period))
        return (beam_ids % period) == offset

    def _random_keep(self, n):
        ratio = self.target_keep_ratio
        if ratio is None:
            ratio = self.dst_beams / float(self.src_beams)
        n_keep = max(1, int(round(n * ratio)))
        keep = np.zeros(n, dtype=bool)
        keep[self.rng.choice(n, min(n_keep, n), replace=False)] = True
        return keep

    def _align_ratio(self, keep, scores, beam_ids, alive_beams=None, use_scores=True):
        """把保留点数对齐到 round(n * target_keep_ratio)，两种预算花法。

        ▍alive_beams 为 None（ALIGN_KEEP_STRUCTURE=False）：全局按重要性花预算
            超了就从低分开始砍、不够就从高分开始补，**不区分线束**。
            语义是"等点数的边缘重要性采样"，前景保留最高（实测 25% 预算下 0.73），
            但保留下来的点散布在所有 64 条线束上，**不能**声称是 16 线雷达。
        ▍alive_beams 给定（ALIGN_KEEP_STRUCTURE=True）：保住线束结构
            1) 先砍"被丢线束里靠救援留下的点"（本来就是预算外的）；
            2) 还超，就在每条存活线束内部等比削减（线束相对密度不变）；
            3) 最后全局兜底，保证点数严格相等。
            为什么不能直接全局取 top-N（第一版就是这么写的，实测翻车）：
            分层丢线好不容易做出"存活线束在仰角上均匀间隔"，全局取 top-N 会把分数低
            的**整条存活线束**削到 50% 以下，仰角上又出现连续空洞 —— 实测 gap_max
            从 7 条反弹到 34 条，分层的意义被直接抹掉。
        """
        n = keep.size
        n_target = min(max(int(round(n * self.target_keep_ratio)), 1), n)
        n_cur = int(keep.sum())

        if alive_beams is None:
            # ---- 全局按重要性花预算 ----
            if n_cur == n_target:
                return keep
            if n_cur > n_target:
                idx = np.where(keep)[0]
                order = idx[np.argsort(scores[idx], kind='stable')] \
                    if use_scores else self.rng.permutation(idx)
                keep[order[:n_cur - n_target]] = False
            else:
                idx = np.where(~keep)[0]
                if idx.size == 0:
                    return keep
                order = idx[np.argsort(-scores[idx], kind='stable')] \
                    if use_scores else self.rng.permutation(idx)
                keep[order[:min(n_target - n_cur, order.size)]] = True
            return keep

        # ---- 保住线束结构 ----
        is_alive = alive_beams[beam_ids]
        if n_cur > n_target:
            # 1) 先砍被丢线束里的救援点
            need = n_cur - n_target
            dead_idx = np.where(keep & ~is_alive)[0]
            if dead_idx.size:
                order = dead_idx[np.argsort(scores[dead_idx], kind='stable')] \
                    if use_scores else self.rng.permutation(dead_idx)
                take = order[:min(need, order.size)]
                keep[take] = False
                need -= take.size

            # 2) 存活线束内部等比削减（保持"哪些线束存活"不变）
            if need > 0:
                alive_kept = np.where(keep & is_alive)[0]
                if alive_kept.size:
                    factor = max(0.0, 1.0 - float(need) / float(alive_kept.size))
                    alive_beam_ids = beam_ids[alive_kept]
                    for b in np.unique(alive_beam_ids):
                        idx = alive_kept[alive_beam_ids == b]
                        m = max(1, int(np.floor(idx.size * factor)))
                        if m >= idx.size:
                            continue
                        order = idx[np.argsort(scores[idx], kind='stable')] \
                            if use_scores else self.rng.permutation(idx)
                        keep[order[:idx.size - m]] = False

            # 3) 兜底：严格对齐（正常情况下走不到）
            n_cur = int(keep.sum())
            if n_cur > n_target:
                idx = np.where(keep)[0]
                order = idx[np.argsort(scores[idx], kind='stable')] \
                    if use_scores else self.rng.permutation(idx)
                keep[order[:n_cur - n_target]] = False
        elif n_cur < n_target:
            idx = np.where(~keep)[0]
            if idx.size == 0:
                return keep
            order = idx[np.argsort(-scores[idx], kind='stable')] \
                if use_scores else self.rng.permutation(idx)
            keep[order[:min(n_target - n_cur, order.size)]] = True
        return keep

    def __call__(self, points, gt_boxes=None):
        """单帧降线束采样。

        Args:
            points: (N, C>=3) 单帧点云，前三列 x,y,z，其余列原样保留
            gt_boxes: (M, 8) [x,y,z,dx,dy,dz,heading,label]，测试阶段可为 None
        Returns:
            sampled: (N', C)
        """
        points = np.asarray(points)
        self._last_inside = None                 # 防止复用上一帧的框内判定
        t_start = time.perf_counter()
        if len(points) == 0:
            self.last_info = {'n_in': 0, 'n_out': 0, 'keep_ratio': 0.0,
                              'fg_keep_ratio': 0.0, 'n_beam_out': 0, 'ms': 0.0}
            self.last_keep = np.zeros(0, dtype=bool)
            return points
        if self.dst_beams >= self.src_beams and self.mode == 'importance':
            self.last_keep = np.ones(len(points), dtype=bool)
            self.last_info = {'n_in': int(len(points)), 'n_out': int(len(points)),
                              'keep_ratio': 1.0, 'fg_keep_ratio': 1.0,
                              'n_beam_out': int(self.src_beams), 'mode': self.mode,
                              'ms': (time.perf_counter() - t_start) * 1000.0}
            return points                             # 不降线束，原样返回

        beam_ids = self.assign_beams(points, self.src_beams)
        alive_beams = None

        if self.mode == 'random':
            keep = self._random_keep(len(points))
        elif self.mode == 'uniform_beam':
            keep = self._uniform_beam_keep(points, beam_ids)
            if self.target_keep_ratio is not None:
                keep = self._align_ratio(keep, None, beam_ids, use_scores=False)
        else:
            scores = self.compute_scores(points, gt_boxes)
            keep, alive_beams = self.progressive_reduce(points, scores, beam_ids)
            if self.target_keep_ratio is not None:
                keep = self._align_ratio(keep, scores, beam_ids,
                                         alive_beams=alive_beams if self.align_keep_structure else None,
                                         use_scores=True)

        # 前景保留率：训练期（有 GT）才有意义，也是文档 5.1 两张表的同一口径。
        # 记录它才能回答"训练时学生看到的车，和部署时是不是同一批点"（修订 14）。
        fg_ratio = -1.0                          # -1 = 本帧无 GT / 未统计（不是 0）
        if self.log_stats and gt_boxes is not None:
            fg = self.foreground_mask(points, gt_boxes)
            if fg is not None and bool(fg.any()):
                fg_ratio = float(keep[fg].sum()) / float(fg.sum())

        self.last_info = {
            'n_in': int(len(points)),
            'n_out': int(keep.sum()),
            'keep_ratio': float(keep.sum()) / max(len(points), 1),
            'fg_keep_ratio': fg_ratio,
            'n_beam_in': int(len(np.unique(beam_ids))),
            'n_beam_out': int(len(np.unique(beam_ids[keep]))) if keep.any() else 0,
            'mode': self.mode,
            'ms': (time.perf_counter() - t_start) * 1000.0,
        }
        self.last_keep = keep
        return points[keep]

    @staticmethod
    def apply_to_batch(eipbs, batch_points, batch_gt_boxes):
        """按第 0 列 batch_idx 逐帧调用（在线蒸馏用）。

        Args:
            batch_points: (N, 1+C) tensor/ndarray，第 0 列 batch_idx
            batch_gt_boxes: (B, M, 8)
        Returns:
            (N', 1+C) ndarray
        """
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None and isinstance(batch_points, torch.Tensor):
            batch_points = batch_points.detach().cpu().numpy()
        if torch is not None and isinstance(batch_gt_boxes, torch.Tensor):
            batch_gt_boxes = batch_gt_boxes.detach().cpu().numpy()
        batch_points = np.asarray(batch_points)
        if batch_points.size == 0:
            return batch_points
        bs_idx = batch_points[:, 0].astype(np.int64)
        num_batch = int(bs_idx.max()) + 1
        if batch_gt_boxes is not None:
            batch_gt_boxes = np.asarray(batch_gt_boxes)

        outs = []
        for b in range(num_batch):
            m = bs_idx == b
            if not m.any():
                continue
            pts = batch_points[m][:, 1:]          # 去掉 batch 列
            gts = None
            if batch_gt_boxes is not None and b < len(batch_gt_boxes):
                gts = batch_gt_boxes[b]
                gts = gts[~np.all(gts == 0, axis=1)] if gts.size else gts
            kept = eipbs(pts, gts)
            if kept.size == 0:
                continue
            col = np.full((len(kept), 1), b, dtype=kept.dtype)
            outs.append(np.concatenate([col, kept], axis=1))
        if not outs:
            return np.zeros((0, batch_points.shape[1]), dtype=batch_points.dtype)
        return np.concatenate(outs, axis=0)
