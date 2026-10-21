from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .sdr_voxel_rcnn import SDRVoxelRCNN
from .voxel_rcnn import VoxelRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'VoxelRCNN': VoxelRCNN,
    'SDRVoxelRCNN': SDRVoxelRCNN,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
