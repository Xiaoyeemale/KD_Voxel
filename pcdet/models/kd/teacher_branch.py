# -*- coding: utf-8 -*-
"""
教师分支 —— 只需产出 BEV 特征图的 Voxel R-CNN 前段（VFE + 3D 骨干 + MapToBEV + 2D 骨干）。

为什么不复用完整的 VoxelRCNN？
    CG-RD 只需要 f_t，不需要教师的检测结果；去掉 DENSE_HEAD / ROI_HEAD 后
    教师分支的显存与耗时都明显下降，且可以整体 freeze + no_grad 运行。

配置位置：MODEL.TEACHER.<模块名>，结构与 MODEL 下同名段完全一致，例如
    TEACHER:
        VFE:          {NAME: MeanVFE}
        BACKBONE_3D:  {NAME: VoxelBackBone8x}
        MAP_TO_BEV:   {NAME: HeightCompression, NUM_BEV_FEATURES: 256}
        BACKBONE_2D:  {NAME: BaseBEVBackbone, ...}
"""

import torch
import torch.nn as nn

from ..backbones_2d import __all__ as backbones_2d_all
from ..backbones_2d import map_to_bev
from ..backbones_3d import __all__ as backbones_3d_all
from ..backbones_3d import vfe


class BEVTeacher(nn.Module):
    def __init__(self, model_cfg, num_rawpoint_features, grid_size, point_cloud_range,
                 voxel_size, depth_downsample_factor=1):
        super().__init__()
        self.model_cfg = model_cfg

        # ---- VFE ----
        vfe_cfg = model_cfg.get('VFE', None)
        assert vfe_cfg is not None, 'MODEL.TEACHER.VFE 必须配置'
        vfe_module = vfe.__all__[vfe_cfg.NAME](
            model_cfg=vfe_cfg,
            num_point_features=num_rawpoint_features,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            grid_size=grid_size,
            depth_downsample_factor=depth_downsample_factor,
        )
        self.vfe = vfe_module
        num_point_features = vfe_module.get_output_feature_dim()

        # ---- 3D 骨干 ----
        bb3d_cfg = model_cfg.get('BACKBONE_3D', None)
        assert bb3d_cfg is not None, 'MODEL.TEACHER.BACKBONE_3D 必须配置'
        backbone_3d = backbones_3d_all[bb3d_cfg.NAME](
            model_cfg=bb3d_cfg,
            input_channels=num_point_features,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.backbone_3d = backbone_3d

        # ---- MapToBEV ----
        bev_cfg = model_cfg.get('MAP_TO_BEV', None)
        assert bev_cfg is not None, 'MODEL.TEACHER.MAP_TO_BEV 必须配置'
        map_to_bev_module = map_to_bev.__all__[bev_cfg.NAME](
            model_cfg=bev_cfg, grid_size=grid_size)
        self.map_to_bev = map_to_bev_module
        num_bev_features = map_to_bev_module.num_bev_features

        # ---- 2D 骨干 ----
        bb2d_cfg = model_cfg.get('BACKBONE_2D', None)
        assert bb2d_cfg is not None, 'MODEL.TEACHER.BACKBONE_2D 必须配置'
        backbone_2d = backbones_2d_all[bb2d_cfg.NAME](
            model_cfg=bb2d_cfg, input_channels=num_bev_features)
        self.backbone_2d = backbone_2d

        self.num_bev_features = backbone_2d.num_bev_features

    def forward(self, batch_dict):
        """输入 batch_dict 只需 voxels / voxel_coords / voxel_num_points / batch_size。

        Returns:
            (B, C, H, W) BEV 特征图
        """
        cur = {
            'batch_size': batch_dict['batch_size'],
            'voxels': batch_dict['voxels'],
            'voxel_coords': batch_dict['voxel_coords'],
            'voxel_num_points': batch_dict['voxel_num_points'],
        }
        for module in (self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d):
            cur = module(cur)
        return cur['spatial_features_2d']
