"""Shared helpers for the coarse-subdivision relaxation analysis scripts.

A run directory is the --output_dir of one collapse_viz_bin run with the coarse
subdivision relaxation on. It holds:
  coarse_subdiv_<stem>.obj                      subdivided coarse mesh (planar, coarse positions)
  coarse_subdiv_at_fine_pos_<stem>.obj          seed: every vertex at its coarse -> fine correspondence
  coarse_subdiv_at_fine_pos_relaxed_*_<stem>.obj relaxed result
  simplified_<stem>.obj                         the coarse mesh
  laplacian_graph/{sheets,curves,junctions}_coarse_subdiv_<stem>.ply
"""
import glob
import os
import re

import numpy as np


def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                F.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(V), np.array(F, dtype=np.int64)


def find(run_dir, pattern):
    hits = glob.glob(os.path.join(run_dir, pattern))
    return hits[0] if hits else None


def coarse_mesh(run_dir):
    """(Vc, F): the subdivided coarse mesh at coarse positions."""
    return load_obj(find(run_dir, 'coarse_subdiv_mat_*.obj') or find(run_dir, 'coarse_subdiv_*.obj'))


def seed_positions(run_dir):
    return load_obj(find(run_dir, 'coarse_subdiv_at_fine_pos_mat_*.obj'))[0]


SNAPSHOT_RE = re.compile(r'_it(\d+)\.obj$')


def relaxed_positions(run_dir):
    """Final relaxed OBJ of a run (not the explicit solver's _it<N> snapshots)."""
    hits = [p for p in glob.glob(os.path.join(run_dir, 'coarse_subdiv_at_fine_pos_relaxed_*.obj'))
            if not SNAPSHOT_RE.search(p)]
    return (load_obj(hits[0])[0], os.path.basename(hits[0])) if hits else (None, None)


def snapshots(run_dir):
    """[(iteration, path)] of the explicit solver's snapshots, sorted."""
    out = []
    for p in glob.glob(os.path.join(run_dir, 'coarse_subdiv_at_fine_pos_relaxed_*_it*.obj')):
        m = SNAPSHOT_RE.search(p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def tri_normals(V, F):
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def coarse_face_of(run_dir, Vc, F):
    """Coarse face of every subdivided face (cached in <run_dir>/_coarse_face_of.npy).

    The subdivision is planar at coarse positions, so each subdivided face's
    centroid lies inside exactly one coarse triangle of simplified_<stem>.obj.
    """
    cache = os.path.join(run_dir, '_coarse_face_of.npy')
    if os.path.exists(cache):
        best = np.load(cache)
        if len(best) == len(F):
            return best
    Vk, Fk = load_obj(find(run_dir, 'simplified_*.obj'))
    C = Vc[F].mean(1)
    A, B, Cc = Vk[Fk[:, 0]], Vk[Fk[:, 1]], Vk[Fk[:, 2]]
    best = np.full(len(F), -1)
    bd = np.full(len(F), np.inf)
    for k in range(len(Fk)):
        e0, e1 = B[k] - A[k], Cc[k] - A[k]
        n = np.cross(e0, e1)
        nn = n @ n
        d = C - A[k]
        u = (np.cross(d, e1) @ n) / nn
        v = (np.cross(e0, d) @ n) / nn
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        dist = np.abs(d @ n) / np.sqrt(nn)
        m = inside & (dist < bd)
        best[m] = k
        bd[m] = dist[m]
    if (best < 0).any():
        raise RuntimeError(f'{int((best < 0).sum())} subdivided faces not inside any coarse face')
    np.save(cache, best)
    return best


def folded(V, F, Vc, best):
    """Folded triangles: orientation disagrees with the majority of its coarse face.

    Comparing against the coarse normal directly is wrong: some coarse faces are
    oriented against the fine sheet under them, which flips all 1024 of their
    triangles without any fold.
    """
    nc = tri_normals(Vc, F)
    s = np.sign(np.einsum('ij,ij->i', tri_normals(V, F), nc))
    nCoarse = int(best.max()) + 1
    pos = np.bincount(best, s > 0, nCoarse)
    neg = np.bincount(best, s < 0, nCoarse)
    return s != np.where(pos >= neg, 1, -1)[best]


def degenerate(V, F, Vc):
    diag = np.linalg.norm(Vc.max(0) - Vc.min(0))
    return np.linalg.norm(tri_normals(V, F), axis=1) / 2 <= 1e-14 * diag * diag


def edge_cv(V, F):
    e = np.concatenate([np.linalg.norm(V[F[:, i]] - V[F[:, (i + 1) % 3]], axis=1) for i in range(3)])
    return e.std() / e.mean()


def edge_cv_per_coarse_face(V, F, best):
    """Median over coarse faces of the edge-length CV of that face's triangles:
    how evenly spaced the vertices are inside each coarse face, whatever its size."""
    L = np.stack([np.linalg.norm(V[F[:, i]] - V[F[:, (i + 1) % 3]], axis=1) for i in range(3)], 1)
    n = int(best.max()) + 1
    cnt = np.bincount(best, minlength=n) * 3
    s1 = np.bincount(best, L.sum(1), n)
    s2 = np.bincount(best, (L ** 2).sum(1), n)
    m = s1 / cnt
    cv = np.sqrt(np.maximum(s2 / cnt - m ** 2, 0)) / m
    return float(np.median(cv))


def read_graph_ply(path):
    """(vertices, edges) of a laplacian_graph PLY as numpy structured arrays."""
    b = open(path, 'rb').read()
    h = b.index(b'end_header\n') + 11
    hdr = b[:h].decode()
    nv = int(hdr.split('element vertex ')[1].split()[0])
    ne = int(hdr.split('element edge ')[1].split()[0])
    vt = np.dtype([('p', '<f4', 3), ('c', 'u1', 3), ('vid', '<i4'), ('fixed', 'u1')])
    et = np.dtype([('ab', '<i4', 2), ('c', 'u1', 3), ('id', '<i4')])
    if len(b) != h + nv * vt.itemsize + ne * et.itemsize:
        raise RuntimeError(f'{path}: size does not match the header')
    v = np.frombuffer(b, vt, nv, h)
    e = np.frombuffer(b, et, ne, h + nv * vt.itemsize)
    return v, e
