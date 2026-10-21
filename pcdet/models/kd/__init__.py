# -*- coding: utf-8 -*-
"""
蒸馏增强（Knowledge-Distillation）模块集合。

    teacher_branch.BEVTeacher   教师分支（VFE + 3D 骨干 + MapToBEV + 2D 骨干）
    cgrd_loss.CGRDLoss          创新点② CG-RD 曲率门控黎曼蒸馏损失

创新点① MS-DGE 在 `pcdet/models/backbones_3d/vfe/msdge_vfe.py`（需注册为 VFE）；
创新点③ EIP-BS 在 `pcdet/utils/eipbs_utils.py`（由 DataProcessor 调用）。
"""

from .cgrd_loss import CGRDLoss, riemannian_distance, CurvatureGate
from .teacher_branch import BEVTeacher

__all__ = ['CGRDLoss', 'BEVTeacher', 'riemannian_distance', 'CurvatureGate']
