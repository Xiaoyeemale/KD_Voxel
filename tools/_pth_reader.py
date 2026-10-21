# -*- coding: utf-8 -*-
"""不依赖 torch 读取 .pth 的 state_dict 键名与形状。

为什么需要
----------
体检教师 ckpt 只需要"键名 + 形状"，但 torch 装不好（比如 Windows 上
`torch_python.dll` 的 WinError 126）时连 `torch.load` 都跑不了。而 .pth 本质是
pickle：张量的形状写在 `_rebuild_tensor(..., size, stride, ...)` 的参数里，
所以用标准库就能读出来，不需要真的分配内存。

支持两种格式
------------
- **legacy**（torch < 1.6 默认，magic `0x1950a86a20f9469cfc6c`）
- **zip**（torch >= 1.6 默认，内部是 `data.pkl` + `data/*`）

只读，不执行任何反序列化出来的代码：所有 torch 类型都替换成桩对象。
"""

import io
import os
import pickle
import struct
import zipfile

_LEGACY_MAGIC = 0x1950a86a20f9469cfc6c


class _TensorStub(object):
    """占位张量：只记 shape。"""

    __slots__ = ('shape', 'dtype')

    def __init__(self, size=None, dtype=None):
        self.shape = tuple(int(v) for v in size) if size is not None else ()
        self.dtype = dtype

    def __repr__(self):
        return 'TensorStub(shape=%s)' % (self.shape,)


class _StorageStub(object):
    __slots__ = ('numel',)

    def __init__(self, numel=0):
        self.numel = int(numel)

    def __len__(self):
        return self.numel


def _make_finder():
    """collections / builtins / numpy 用真的，torch 一律用桩。"""

    def find_class(module, name):
        if module in ('collections', 'builtins', '__builtin__'):
            return pickle.Unpickler.find_class(_Unpickler, module, name)
        if module == 'torch._utils' and name in (
                '_rebuild_tensor', '_rebuild_tensor_v2'):
            def _rebuild(storage, storage_offset, size, stride, *rest, **kw):
                return _TensorStub(size)
            return _rebuild
        if module == 'torch._utils' and name == '_rebuild_parameter':
            def _rebuild_param(data, requires_grad=False, backward_hooks=None):
                return data
            return _rebuild_param
        if module == 'torch._utils' and name == '_rebuild_parameter_with_state':
            def _rebuild_param2(data, *rest, **kw):
                return data
            return _rebuild_param2
        # 其余（torch.Size / torch.FloatStorage / numpy 标量 ……）一律桩，
        # 我们不需要它们的值，只需要不让 pickle 报错。
        def _any(*args, **kwargs):
            return _TensorStub()
        return _any

    return staticmethod(find_class)


class _Unpickler(pickle.Unpickler):
    """把 torch 的类型全换成桩；collections / builtins 保持原样。"""

    def find_class(self, module, name):
        if module in ('collections', 'builtins', '__builtin__'):
            return super(_Unpickler, self).find_class(module, name)
        if module == '_pth_reader':
            return super(_Unpickler, self).find_class(module, name)
        if module == 'torch._utils' and name in (
                '_rebuild_tensor', '_rebuild_tensor_v2'):
            def _rebuild(storage, storage_offset, size, stride, *rest, **kw):
                return _TensorStub(size)
            return _rebuild
        if module == 'torch._utils' and name in (
                '_rebuild_parameter', '_rebuild_parameter_with_state'):
            def _rebuild_param(data, *rest, **kw):
                return data
            return _rebuild_param
        # 其余（torch.Size / torch.FloatStorage / numpy 标量 ……）一律桩，
        # 我们不需要它们的值，只需要不让 pickle 报错。
        def _any(*args, **kwargs):
            return _TensorStub()
        return _any

    def persistent_load(self, pid):
        # ('storage', storage_type, root_key, location, numel, view_metadata?)
        if isinstance(pid, tuple) and len(pid) >= 5:
            try:
                return _StorageStub(pid[4])
            except Exception:
                return _StorageStub(0)
        return _StorageStub(0)


def _load_legacy(path):
    """legacy 格式有两种开头，都要认：
    - `torch.save` 老实现把 magic 当 Python int **pickle** 进去（首字节 0x80 = PROTO）
    - 也有直接 `struct.pack('<Q', magic)` 的裸 8 字节写法
    """
    with open(path, 'rb') as f:
        head = f.read(1)
        f.seek(0)
        if head == b'\x80':
            magic = pickle.load(f)
            if int(magic) != _LEGACY_MAGIC:
                raise ValueError('magic 不匹配: %x' % int(magic))
            pickle.load(f)                        # PROTOCOL_VERSION
            pickle.load(f)                        # sys_info
        else:
            raw = f.read(8)
            if len(raw) < 8 or struct.unpack('<Q', raw)[0] != _LEGACY_MAGIC:
                raise ValueError('不是 legacy 格式')
            struct.unpack('<i', f.read(4))        # protocol version
            pickle.load(f)                        # sys_info
        obj = _Unpickler(f).load()                # 主对象（state dict）
    return obj


def _load_zip(path):
    with zipfile.ZipFile(path) as z:
        # torch >= 1.13 会多一层 <name>/data.pkl
        names = [n for n in z.namelist() if n.endswith('data.pkl')]
        if not names:
            raise ValueError('zip 里没有 data.pkl')
        name = sorted(names, key=len)[0]
        with z.open(name) as f:
            obj = _Unpickler(io.BytesIO(f.read())).load()
    return obj


def read_state_dict(path):
    """返回 (state_dict, meta)。

    state_dict: {key: _TensorStub}
    meta:       从主对象里摘出来的 epoch / it / best_score 等标量（有就带上）
    """
    if not os.path.isfile(path):
        raise IOError('文件不存在: %s' % path)

    with open(path, 'rb') as f:
        head = f.read(1)
    is_legacy = head == b'\x80' or head == b'\x00' or head not in (b'PK',)
    if head == b'\x80':
        # 首字节 0x80 也可能是被 pickle 的 magic（legacy），也可能不是。用 magic 判定。
        try:
            with open(path, 'rb') as f:
                is_legacy = int(pickle.load(f)) == _LEGACY_MAGIC
        except Exception:
            is_legacy = False
    elif head == b'P':
        is_legacy = False
    else:
        is_legacy = True

    obj = _load_legacy(path) if is_legacy else _load_zip(path)

    meta = {}
    if isinstance(obj, dict) and 'model_state' in obj:
        state = obj['model_state']
        for k in ('epoch', 'it', 'best_score', 'best_epoch', 'start_epoch'):
            if k in obj:
                v = obj[k]
                meta[k] = v if isinstance(v, (int, float, str)) else type(v).__name__
        for k in ('optimizer_state', 'lr_scheduler'):
            if k in obj:
                meta[k] = '<已保存>'
    else:
        state = obj

    if not isinstance(state, dict):
        raise ValueError('无法识别的 ckpt 结构: %s' % type(state))

    out = {}
    for key, val in state.items():
        key = key[len('module.'):] if isinstance(key, str) and key.startswith('module.') else key
        out[key] = val
    return out, meta, ('legacy' if is_legacy else 'zip')


if __name__ == '__main__':
    import sys
    sd, meta, fmt = read_state_dict(sys.argv[1])
    print('格式: %s   张量数: %d' % (fmt, len(sd)))
    for k, v in meta.items():
        print('  %-15s %s' % (k, v))
    for k in list(sd)[:20]:
        print('  %-52s %s' % (k, getattr(sd[k], 'shape', '?')))
