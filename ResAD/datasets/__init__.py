from .mvtec import MVTEC
from .mvtec import MVTECANO, get_normal_image_paths_mvtec
from .btad import BTAD, get_normal_image_paths_btad
from .mvtec_3d import MVTEC3D, get_normal_image_paths_mvtec3d
from .mvtec_loco import MVTECLOCO
from .visa import VISA, get_normal_image_paths_visa
from .dcase import DCASE2024


__all__ = ['MVTEC', 'MVTECANO', 'BTAD', 'MVTEC3D', 'MVTECLOCO', 'VISA', 'DCASE2024']


def get_normal_image_paths(root, class_name, dataset='mvtec'):
    if dataset == 'mvtec':
        return get_normal_image_paths_mvtec(root, class_name)
    elif dataset == 'btad':
        return get_normal_image_paths_btad(root, class_name)
    elif dataset == 'visa':
        return get_normal_image_paths_visa(root, class_name)
    elif dataset == 'mvtec3d':
        return get_normal_image_paths_mvtec3d(root, class_name)
    else:
        raise ValueError("Unrecognized!")