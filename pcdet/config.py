from pathlib import Path

import yaml
from easydict import EasyDict


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def _resolve_cfg_path(path_str, cfg_dir=None):
    """Resolve a path written inside a yaml config.

    Relative paths are tried against the current working directory first (upstream
    behaviour), then the yaml file's own directory, then the project root and the
    project's ``tools/`` directory. This way both of these work:
        python tools/train.py --cfg_file tools/cfgs/kitti_models/voxel_rcnn_car.yaml
        cd tools && python train.py --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    candidates = [p]
    if cfg_dir is not None:
        candidates.append(Path(cfg_dir) / path_str)
    candidates.append(Path(cfg.ROOT_DIR) / path_str)
    candidates.append(Path(cfg.ROOT_DIR) / 'tools' / path_str)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return p


def merge_new_config(config, new_config, cfg_dir=None):
    if '_BASE_CONFIG_' in new_config:
        with open(_resolve_cfg_path(new_config['_BASE_CONFIG_'], cfg_dir), 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, cfg_dir)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config, cfg_dir=Path(cfg_file).parent)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
