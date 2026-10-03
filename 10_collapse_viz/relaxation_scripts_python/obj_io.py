"""OBJ reading / writing in the format the C++ side writes (igl::writeOBJ):

    v %.15g %.15g %.15g
    f a b c          (1-based), platform line endings

so files written here can be compared byte for byte with the C++ outputs.
"""
import os

import numpy as np


def long_path(path):
    """subdiv_long_path: on Windows, the absolute path in \\\\?\\ form (paths over 260 chars)."""
    if os.name != 'nt':
        return path
    a = os.path.normpath(os.path.abspath(path))
    if a.startswith('\\\\'):
        return a
    return '\\\\?\\' + a


def write_obj(path, V, F):
    """igl::writeOBJ(path, V, F). Returns False on an I/O error."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    try:
        # text mode, platform newlines: CRLF on Windows, as igl::writeOBJ's fopen(.., "w")
        with open(long_path(path), 'w') as fh:
            fh.write(''.join('v %.15g %.15g %.15g\n' % (x, y, z) for x, y, z in V[:, :3].tolist()))
            fh.write(''.join('f %d %d %d\n' % (a + 1, b + 1, c + 1) for a, b, c in F.tolist()))
        return True
    except OSError:
        return False


def read_obj(path):
    """Vertices (n x 3 float64) and triangles (m x 3 int64, 0-based) of an OBJ.
    Ignores every other element (l, vt, vn, ...); face corners may be v/vt/vn."""
    V, F = [], []
    with open(long_path(path), 'r') as fh:
        for line in fh:
            if line.startswith('v '):
                p = line.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                p = line.split()[1:]
                F.append(tuple(int(t.split('/')[0]) - 1 for t in p[:3]))
    return np.array(V, dtype=np.float64).reshape(-1, 3), np.array(F, dtype=np.int64).reshape(-1, 3)
