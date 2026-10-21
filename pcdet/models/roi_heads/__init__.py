from .pvrcnn_head import PVRCNNHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PVRCNNHead': PVRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
}
