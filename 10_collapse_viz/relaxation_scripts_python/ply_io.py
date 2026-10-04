"""Binary little-endian PLY writer: positions (double), triangle faces, and extra
per-vertex / per-face properties (one scalar per element, any numpy dtype below)."""
import os

import numpy as np

from obj_io import long_path

_PLY_TYPE = {np.dtype(np.float64): 'double', np.dtype(np.float32): 'float', np.dtype(np.int32): 'int',
             np.dtype(np.uint32): 'uint', np.dtype(np.uint8): 'uchar', np.dtype(np.int8): 'char',
             np.dtype(np.int16): 'short', np.dtype(np.uint16): 'ushort'}
_CAST = {np.dtype(np.int64): np.int32, np.dtype(np.bool_): np.uint8, np.dtype(np.uint64): np.uint32}


def _props(props, n, what):
    out = []
    for name, a in (props or {}).items():
        a = np.asarray(a)
        if a.shape != (n,):
            raise ValueError('[ply_io] %s property %s has shape %s, expected (%d,)' % (what, name, a.shape, n))
        if a.dtype in _CAST:
            a = a.astype(_CAST[a.dtype])
        if a.dtype not in _PLY_TYPE:
            raise ValueError('[ply_io] %s property %s: unsupported dtype %s' % (what, name, a.dtype))
        out.append((name, a.astype(a.dtype.newbyteorder('<'))))
    return out


def write_ply(path, V, F, vprops=None, fprops=None, comments=()):
    """V: n x 3, F: m x 3; vprops / fprops: {name: array}. Returns True on success."""
    V = np.asarray(V, dtype=np.float64)[:, :3]
    F = np.asarray(F, dtype=np.int64)
    n, m = V.shape[0], F.shape[0]
    vp = _props(vprops, n, 'vertex')
    fp = _props(fprops, m, 'face')
    head = ['ply', 'format binary_little_endian 1.0']
    head += ['comment ' + c.replace('\n', ' ') for c in comments]
    head += ['element vertex %d' % n, 'property double x', 'property double y', 'property double z']
    head += ['property %s %s' % (_PLY_TYPE[a.dtype.newbyteorder('=')], name) for name, a in vp]
    head += ['element face %d' % m, 'property list uchar int vertex_indices']
    head += ['property %s %s' % (_PLY_TYPE[a.dtype.newbyteorder('=')], name) for name, a in fp]
    head.append('end_header')

    vdt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')] + [(name, a.dtype) for name, a in vp])
    vrec = np.empty(n, dtype=vdt)
    vrec['x'], vrec['y'], vrec['z'] = V[:, 0], V[:, 1], V[:, 2]
    for name, a in vp:
        vrec[name] = a
    fdt = np.dtype([('k', 'u1'), ('i', '<i4', (3,))] + [(name, a.dtype) for name, a in fp])
    frec = np.empty(m, dtype=fdt)
    frec['k'] = 3
    frec['i'] = F
    for name, a in fp:
        frec[name] = a
    d = os.path.dirname(path)
    if d:
        os.makedirs(long_path(d), exist_ok=True)
    try:
        with open(long_path(path), 'wb') as f:
            f.write(('\n'.join(head) + '\n').encode('ascii'))
            f.write(vrec.tobytes())
            f.write(frec.tobytes())
    except OSError:
        return False
    return True


def read_ply(path):
    """Reads what write_ply writes: (V, F, vprops, fprops)."""
    with open(long_path(path), 'rb') as f:
        lines = []
        while True:
            s = f.readline().decode('ascii').rstrip('\n')
            lines.append(s)
            if s == 'end_header':
                break
        body = f.read()
    rev = {v: k for k, v in _PLY_TYPE.items()}
    elems, cur = [], None
    for s in lines:
        t = s.split()
        if t[0] == 'element':
            cur = [t[1], int(t[2]), []]
            elems.append(cur)
        elif t[0] == 'property' and t[1] != 'list':
            cur[2].append((t[2], rev[t[1]].newbyteorder('<')))
        elif t[0] == 'property':
            cur[2].append(('__list', None))
    (_, n, vfields), (_, m, ffields) = elems
    vdt = np.dtype(vfields)
    vrec = np.frombuffer(body, dtype=vdt, count=n)
    fdt = np.dtype([('k', 'u1'), ('i', '<i4', (3,))] + [x for x in ffields if x[0] != '__list'])
    frec = np.frombuffer(body, dtype=fdt, count=m, offset=vdt.itemsize * n)
    V = np.stack([vrec['x'], vrec['y'], vrec['z']], axis=1)
    vprops = {k: vrec[k] for k, _ in vfields if k not in ('x', 'y', 'z')}
    fprops = {k: frec[k] for k, _ in ffields if k != '__list'}
    return V, frec['i'].astype(np.int64), vprops, fprops
