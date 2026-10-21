from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        # 创新点③ EIP-BS：学生分支的降线束采样器与独立体素生成器（惰性创建，避免多进程 pickle）
        self.eip_bs_sampler = None
        self.voxel_generator_kd = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    # ------------------------------------------------------------------ #
    #  创新点③ EIP-BS：稠密点云 -> 降线束稀疏点云（学生分支输入）
    # ------------------------------------------------------------------ #
    def eip_bs_voxelize(self, data_dict=None, config=None):
        """对稠密点云执行 EIP-BS 降线束，并单独体素化为学生分支的输入。

        产物（仅当数据配置启用了本处理器时存在）：
            points_kd                (N', C)   学生输入点云（已含 intensity）
            voxels_kd                (M', T, C)
            voxel_coords_kd          (M', 4)   [batch_idx, z, y, x]（collate 时补 batch 列）
            voxel_num_points_kd      (M',)     被 MAX_POINTS_PER_VOXEL 截断的点数
            voxel_num_points_true_kd (M',)     真实点数（供 MS-DGE 的密度先验/池化权重）

        必须排在 transform_points_to_voxels 之后（需要它算好的 voxel_size）。
        """
        if data_dict is None:
            if self.voxel_size is None:
                raise RuntimeError(
                    'DATA_PROCESSOR 中 eip_bs_voxelize 必须排在 transform_points_to_voxels 之后')
            return partial(self.eip_bs_voxelize, config=config)

        if not config.get('ENABLED', True):
            return data_dict

        from ...utils.eipbs_utils import EIPBS
        if self.eip_bs_sampler is None:
            self.eip_bs_sampler = EIPBS(
                src_beams=config.get('SRC_BEAMS', 64),
                dst_beams=config.get('DST_BEAMS', 16),
                stages=config.get('STAGES', None),
                w_curvature=config.get('W_CURVATURE', 0.4),
                w_density=config.get('W_DENSITY', 0.3),
                w_class=config.get('W_CLASS', 0.3),
                curvature_k=config.get('CURVATURE_K', 8),
                curvature_cell_size=config.get('CURVATURE_CELL_SIZE', 0.5),
                density_radius=config.get('DENSITY_RADIUS', 1.0),
                class_priority=config.get('CLASS_PRIORITY', (0.6, 1.0, 1.0)),
                keep_ratio_per_drop=config.get('KEEP_RATIO_PER_DROP', 0.2),
                min_rescue_per_beam=config.get('MIN_RESCUE_PER_BEAM', 1),
                # 默认值与文档 5.3 的开关表一致（topk / 25% / 不分层 / 全局花预算）。
                # 这里保持和 yaml 相同，是为了"只写 NAME 不写参数"时行为也不走样。
                beam_score=config.get('BEAM_SCORE', 'topk'),
                beam_top_ratio=config.get('BEAM_TOP_RATIO', 0.1),
                beam_top_k=config.get('BEAM_TOP_K', 16),
                beam_score_thresh=config.get('BEAM_SCORE_THRESH', 0.6),
                near_margin=config.get('NEAR_MARGIN', 0.5),
                use_numba=config.get('USE_NUMBA', True),
                seed=config.get('SEED', None),
                # 采样模式与三个"论文级"约束（见 eipbs_utils.EIPBS 类注释）
                mode=config.get('MODE', 'importance'),
                stratified=config.get('STRATIFIED', False),
                target_keep_ratio=config.get('TARGET_KEEP_RATIO', 0.25),
                align_keep_structure=config.get('ALIGN_KEEP_STRUCTURE', False),
                log_stats=config.get('LOG_STATS', True),
            )
        if self.voxel_generator_kd is None:
            self.voxel_generator_kd = VoxelGeneratorWrapper(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.get('MAX_POINTS_PER_VOXEL', 5),
                max_num_voxels=config.get('MAX_NUMBER_OF_VOXELS')[self.mode],
            )

        points = data_dict['points']
        gt_boxes = data_dict.get('gt_boxes', None)
        if gt_boxes is not None and len(gt_boxes) == 0:
            gt_boxes = None

        # 测试期不得使用 GT（文档修订 14）：KITTI 的 val 划分是带标注的，而
        # `prepare_data` 会把 gt_boxes 一并放进 data_dict。如果这里不丢弃，评测时
        # 的学生输入会变成"用 GT 框重点保住车"的点云 —— 既与部署不一致，也是
        # 直接的信息泄漏（训练期 0.79 vs 部署 0.29 的差距就是这么来的）。
        if self.mode != 'train' and not config.get('USE_GT_IN_TEST', False):
            gt_boxes = None

        points_kd = self.eip_bs_sampler(points, gt_boxes)
        if len(points_kd) == 0:                     # 极端兜底：采样为空则退回稠密
            points_kd = points
        points_kd = np.ascontiguousarray(points_kd, dtype=np.float32)

        voxels, coords, num_points = self.voxel_generator_kd.generate(points_kd)

        # 真实点数（voxel_num_points 被 MAX_POINTS_PER_VOXEL 截断，MS-DGE 的密度
        # 先验与点数加权池化需要真实值；见 pcdet/models/backbones_3d/vfe/msdge_vfe.py）
        if config.get('SAVE_TRUE_COUNTS', True):
            from ...utils.eipbs_utils import true_point_counts
            data_dict['voxel_num_points_true_kd'] = true_point_counts(
                points_kd, coords, self.voxel_size, self.point_cloud_range)

        data_dict['points_kd'] = points_kd
        data_dict['voxels_kd'] = voxels
        data_dict['voxel_coords_kd'] = coords
        data_dict['voxel_num_points_kd'] = num_points

        # 采样器的运行时统计（保留率 / 前景保留率 / 存活线束数 / 单帧耗时）。
        # 逐样本写进 data_dict，靠 DataLoader 的默认 collate 堆成 (B, 4)，
        # 再由 SDRVoxelRCNN 取 batch 均值写进 tb_dict —— 这样"③ 在训练时到底
        # 保住了多少前景"才有连续的曲线，而不是只在离线 bench 里出现一次。
        if config.get('LOG_STATS', True):
            info = getattr(self.eip_bs_sampler, 'last_info', {}) or {}
            data_dict['eipbs_stats'] = np.asarray([
                float(info.get('keep_ratio', 0.0)),
                float(info.get('fg_keep_ratio', -1.0)),
                float(info.get('n_beam_out', 0.0)),
                float(info.get('ms', 0.0)),
            ], dtype=np.float32)

        # WRITE_TO_MAIN: 把稀疏结果直接写成主键。
        # 用途是"纯稀疏基线"对照组：让**未经改动的** Voxel R-CNN 也在同样密度的
        # 点云上训练，此时"保住了哪些点"是唯一自变量（见审计报告 §1.1 问题 B）。
        if config.get('WRITE_TO_MAIN', False):
            data_dict['points'] = points_kd
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coords
            data_dict['voxel_num_points'] = num_points
            if 'voxel_num_points_true_kd' in data_dict:
                data_dict['voxel_num_points_true'] = data_dict['voxel_num_points_true_kd']
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict
    
    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
