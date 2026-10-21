"""Generate KITTI infos / gt_database WITHOUT compiling the CUDA extensions.

Official command (requires `python setup.py develop` to have built the ops):
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
        tools/cfgs/dataset_configs/kitti_dataset.yaml

`roiaware_pool3d_utils.points_in_boxes_cpu` is implemented in the compiled extension
`roiaware_pool3d_cuda`, so before building the ops the dataset module cannot even be
imported. This script replaces that single function with an equivalent NumPy version
(same z/x/y bounds and the same 1e-2 margin used by src/roiaware_pool3d.cpp
::check_pt_in_box3d_cpu) so kitti_infos_*.pkl and the gt_database can be produced
without CUDA. The output is numerically identical to the CUDA kernel.

Usage (from anywhere):
    python tools/gen_kitti_infos_local.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent


def _points_in_boxes_cpu(boxes, pts, pts_indices):
    """NumPy port of check_pt_in_box3d_cpu + points_in_boxes_cpu.

    boxes: (N, 7) [x, y, z, dx, dy, dz, heading] torch.Tensor
    pts:   (M, 3) torch.Tensor
    pts_indices: (N, M) int32 torch.Tensor, filled in-place with 0/1
    """
    b = boxes.detach().cpu().numpy().astype(np.float64)
    p = pts.detach().cpu().numpy().astype(np.float64)
    out = np.zeros((b.shape[0], p.shape[0]), dtype=np.int32)
    for i in range(b.shape[0]):
        dx, dy, dz, rz = b[i, 3], b[i, 4], b[i, 5], b[i, 6]
        sx = p[:, 0] - b[i, 0]
        sy = p[:, 1] - b[i, 1]
        cosa, sina = np.cos(-rz), np.sin(-rz)
        local_x = cosa * sx - sina * sy
        local_y = sina * sx + cosa * sy
        inside = (
            (np.abs(p[:, 2] - b[i, 2]) <= dz / 2.0)
            & (np.abs(local_x) < dx / 2.0 + 1e-2)
            & (np.abs(local_y) < dy / 2.0 + 1e-2)
        )
        out[i] = inside
    pts_indices[:] = torch.from_numpy(out).to(dtype=pts_indices.dtype, device=pts_indices.device)
    return 1


def _install_cuda_stub():
    """Register a stand-in for the not-yet-built extension before pcdet is imported."""
    name = 'pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda'
    mod = types.ModuleType(name)
    mod.points_in_boxes_cpu = _points_in_boxes_cpu

    def _need_build(*args, **kwargs):
        raise RuntimeError(
            'This CUDA op is only needed for training/eval. '
            'Build it with `python setup.py develop` in the project root.'
        )

    mod.forward = _need_build
    mod.backward = _need_build
    mod.points_in_boxes_gpu = _need_build
    sys.modules[name] = mod


def main():
    sys.path.insert(0, str(ROOT))
    _install_cuda_stub()

    from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos

    cfg_file = ROOT / 'tools' / 'cfgs' / 'dataset_configs' / 'kitti_dataset.yaml'
    with open(cfg_file, 'r', encoding='utf-8') as f:
        dataset_cfg = EasyDict(yaml.safe_load(f))

    data_path = ROOT / 'data' / 'kitti'
    print('data path :', data_path)
    print('ImageSets :', sorted(p.name for p in (data_path / 'ImageSets').glob('*.txt')))

    create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=data_path,
        save_path=data_path,
        workers=4,
    )


if __name__ == '__main__':
    main()
