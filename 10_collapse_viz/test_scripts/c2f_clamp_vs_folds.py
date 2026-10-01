"""Does the clamp-and-renormalize step of the coarse -> fine walk cause the seed's folds?

Reads <run_dir>/coarse_subdiv_c2f_clamp_<stem>.csv (one row per subdivided
vertex: walk steps, clamped steps, largest negative barycentric, largest snap)
and compares the fold rate of the seed's triangles that touch a clamped vertex
with those that do not, and by how hard their vertices were clamped.

usage: python c2f_clamp_vs_folds.py <run_dir>
"""
import sys

import numpy as np

from relax_metrics_common import coarse_face_of, coarse_mesh, find, folded, seed_positions

if len(sys.argv) != 2:
    sys.exit(__doc__)
run = sys.argv[1]
csv = find(run, 'coarse_subdiv_c2f_clamp_*.csv')
if csv is None:
    sys.exit(f'{run}: no coarse_subdiv_c2f_clamp_*.csv (run with a build that writes it)')
d = np.genfromtxt(csv, delimiter=',', names=True)
Vc, F = coarse_mesh(run)
best = coarse_face_of(run, Vc, F)
Vs = seed_positions(run)
fold = folded(Vs, F, Vc, best)

mapped = d['mapped'] > 0
print(f'vertices {len(d)}, mapped {int(mapped.sum())}; walk steps per vertex: mean {d["steps"][mapped].mean():.1f}, '
      f'max {int(d["steps"].max())}')
cl = d['clamped_steps'] > 0
print(f'vertices clamped at least once: {int(cl.sum())} ({100 * cl.mean():.1f}%); clamped steps per clamped vertex: '
      f'mean {d["clamped_steps"][cl].mean() if cl.any() else 0:.2f}; far-outside steps: {int(d["far_steps"].sum())}')

tri_cl = cl[F].any(1)
print(f'\nseed triangles: {len(F)}, folded {int(fold.sum())} ({100 * fold.mean():.2f}%)')
print(f'  touching a clamped vertex: {int(tri_cl.sum()):7d} triangles, folded {100 * fold[tri_cl].mean() if tri_cl.any() else 0:.2f}%')
print(f'  not touching one         : {int((~tri_cl).sum()):7d} triangles, folded {100 * fold[~tri_cl].mean() if (~tri_cl).any() else 0:.2f}%')

# by the largest negative barycentric clamped away at any of the triangle's vertices
neg = d['max_neg_bary'][F].max(1)
bins = [(0, 0), (1e-12, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, np.inf)]
print('\nby the largest clamp at the triangle\'s vertices (largest negative barycentric):')
for a, b in bins:
    m = (neg == 0) if b == 0 else ((neg > a) & (neg <= b))
    if m.any():
        label = 'never clamped' if b == 0 else f'({a:g}, {b:g}]'
        print(f'  {label:18s} {int(m.sum()):7d} triangles, folded {100 * fold[m].mean():.2f}%')
