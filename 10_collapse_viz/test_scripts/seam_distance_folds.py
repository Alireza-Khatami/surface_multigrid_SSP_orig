"""Folded / degenerate triangles by ring distance from the nearest seam/boundary
(or junction) vertex, to see whether the damage concentrates at the curves.

usage: python seam_distance_folds.py <run_dir> [<run_dir> ...]
The first run must have laplacian_graph/ (curves_*.ply, junctions_*.ply).
A triangle's distance is that of its closest vertex, in mesh-edge rings.
"""
import os
import sys
from collections import deque

import numpy as np

from relax_metrics_common import (coarse_face_of, coarse_mesh, degenerate, find, folded,
                                  read_graph_ply, relaxed_positions, seed_positions)

runs = sys.argv[1:]
if not runs:
    sys.exit(__doc__)
Vc, F = coarse_mesh(runs[0])
best = coarse_face_of(runs[0], Vc, F)
Vs = seed_positions(runs[0])
nV = len(Vs)

src = set()
for kind in ('curves', 'junctions'):
    p = find(runs[0], f'laplacian_graph/{kind}_*.ply')
    if p is None:
        sys.exit(f'{runs[0]}: no laplacian_graph/{kind}_*.ply')
    src |= set(read_graph_ply(p)[0]['vid'].tolist())

adj = [[] for _ in range(nV)]
for a, b, c in F:
    adj[a] += [b, c]; adj[b] += [a, c]; adj[c] += [a, b]
dist = np.full(nV, -1)
q = deque()
for v in src:
    dist[v] = 0
    q.append(v)
while q:
    v = q.popleft()
    for w in adj[v]:
        if dist[w] < 0:
            dist[w] = dist[v] + 1
            q.append(w)
fd = dist[F].min(1)

bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 10 ** 9)]
labels = [f'{a}-{b}' if b < 10 ** 9 else f'{a}+' for a, b in bins]
masks = [(fd >= a) & (fd <= b) for a, b in bins]
print('ring from seam/boundary :', ' | '.join(f'{l:>14s}' for l in labels))
print('triangles               :', ' | '.join(f'{int(m.sum()):14d}' for m in masks))

rows = [('seed', Vs)] + [(os.path.basename(os.path.normpath(r)), relaxed_positions(r)[0]) for r in runs]
for name, V in rows:
    if V is None:
        continue
    fo, de = folded(V, F, Vc, best), degenerate(V, F, Vc)
    cells = [f'{100 * fo[m].mean():5.2f}% / {100 * de[m].mean():5.2f}%' for m in masks]
    print(f'{name[:24]:24s}:', ' | '.join(cells), f'  (folded / degenerate rate; total {int(fo.sum())} / {int(de.sum())})')
