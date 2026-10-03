"""Self-test of the projector (port of verify_projector_bvh, coarse_subdiv_relax.cpp):
every tree's structure (boxes nest, leaves contain their primitives, each
primitive once) and closest-point queries against brute force (identical
distance and primitive, same tie rule), around random primitives at three
scales of the tree's box diagonal (on / near / far). Trees are built for every
struct id of the .ma_struct (sheets and curves).

  python test_projector.py --bundle <run>/correspondence_<stem>.c2f --matstruct_path <.ma_struct> [--n 30]
"""
import argparse
import os
import sys

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bundle_io import load_bundle_flat  # noqa: E402
from matstruct import load_matstruct  # noqa: E402
from projector import Projector, bvh_query, prim_closest  # noqa: E402
from struct_ids import RELAX_SHEET  # noqa: E402
from subdiv_mesh import subdiv_unique_edges  # noqa: E402


@njit(cache=True)
def brute_force(PJ, t, px, py, pz):
    VO = PJ[0]
    p0, p1 = PJ[19][t], PJ[19][t + 1]
    corner = PJ[20]; gid = PJ[21]
    bestD = np.inf
    bestGid = 2 ** 31 - 1
    for i in range(p0, p1):
        d, b0, b1, b2 = prim_closest(VO, corner[i, 0], corner[i, 1], corner[i, 2], px, py, pz)
        if d < bestD or (d == bestD and gid[i] < bestGid):
            bestD = d
            bestGid = gid[i]
    return bestD, bestGid


def structure_errors(tr, VO):
    n = len(tr.gid)
    err = 0
    order = list(tr.order)
    if sorted(order) != list(range(n)):
        err += 1
    if n == 0:
        return err
    nd = tr.nodes
    if nd[0][4] != 0 or nd[0][5] != n:
        err += 1
    covered = np.zeros(n, dtype=np.int64)
    for mn, mx, l, r, b, e in nd:
        if l < 0:
            for k in range(b, e):
                i = order[k]
                covered[i] += 1
                c = [x for x in tr.corner[i] if x >= 0]
                P = VO[c, :3]
                if (P.min(axis=0) < mn).any() or (P.max(axis=0) > mx).any():
                    err += 1
        else:
            L, R = nd[l], nd[r]
            for ch in (L, R):
                if (ch[0] < mn).any() or (ch[1] > mx).any():
                    err += 1
            if L[4] != b or L[5] != R[4] or R[5] != e:
                err += 1
    err += int((covered != 1).sum())
    return err


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--matstruct_path', required=True)
    p.add_argument('--n', type=int, default=30, help='queries per tree')
    a = p.parse_args()
    B = load_bundle_flat(a.bundle)
    ms = load_matstruct(a.matstruct_path, B.fineV, B.fineF)
    # palette = one set per struct id, so every tree is built
    from struct_ids import StructPalette, role_of, split_palette
    pal = StructPalette()
    for sid in sorted(ms.structType):
        pal.ids.append(sid)
        pal.offsets.append(len(pal.ids))
        pal.typeMask.append(1 << ms.structType[sid] if 0 <= ms.structType[sid] <= 3 else 0)
    S = split_palette(pal, ms)
    setRole = [role_of(m) for m in pal.typeMask]
    proj = Projector(B.fineV, B.fineF, subdiv_unique_edges(B.fineF), ms, pal, S, setRole)
    PJ = proj.pj
    structErr = mismatch = nq = 0
    for t, tr in enumerate(proj.trees):
        structErr += structure_errors(tr, proj.VO)
        if len(tr.gid) == 0:
            continue
        mn, mx = tr.nodes[0][0], tr.nodes[0][1]
        tdiag = float(np.sqrt(((mx - mn) ** 2).sum()))
        rng = np.random.default_rng(t)
        for s in range(a.n):
            c = [x for x in tr.corner[rng.integers(len(tr.gid))] if x >= 0]
            q = proj.VO[c].mean(axis=0) + (0.0, 0.02, 0.5)[s % 3] * tdiag * rng.normal(size=3)
            d0, g0 = brute_force(PJ, t, q[0], q[1], q[2])
            d1, g1, _, _, _ = bvh_query(PJ, t, q[0], q[1], q[2], np.inf, 2 ** 31 - 1, 1.0, 0.0, 0.0)
            mismatch += (d0 != d1) or (g0 != g1)
            nq += 1
    print('[test_projector] projector BVH check: %d trees, %d structure errors, %d of %d queries differ from brute force'
          % (len(proj.trees), structErr, mismatch, nq))
    return 0 if structErr == 0 and mismatch == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
