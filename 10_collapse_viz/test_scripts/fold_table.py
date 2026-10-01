"""Absolute fold / degenerate / edge-CV table, seed vs each run's relaxed mesh,
plus how many of the seed's folds each relaxation removed and how many it added.

usage: python fold_table.py <run_dir> [<run_dir> ...]
All runs must come from the same decimation (same coarse mesh); the seed and the
coarse mesh are read from the first one.
"""
import os
import sys

from relax_metrics_common import (coarse_face_of, coarse_mesh, degenerate, edge_cv, folded,
                                  relaxed_positions, seed_positions)

runs = sys.argv[1:]
if not runs:
    sys.exit(__doc__)
Vc, F = coarse_mesh(runs[0])
best = coarse_face_of(runs[0], Vc, F)
Vs = seed_positions(runs[0])
f0 = folded(Vs, F, Vc, best)

print(f"{'run':40s} {'folded':>7s} {'removed':>8s} {'kept':>6s} {'new':>7s} {'coarse faces':>13s} {'degenerate':>10s} {'edge CV':>8s}")
print(f"{'seed (no relaxation)':40s} {int(f0.sum()):7d} {'-':>8s} {'-':>6s} {'-':>7s} "
      f"{int(len(set(best[f0]))):13d} {int(degenerate(Vs, F, Vc).sum()):10d} {edge_cv(Vs, F):8.3f}")
for r in runs:
    V, name = relaxed_positions(r)
    if V is None:
        print(f'{os.path.basename(r):40s} (no relaxed OBJ)')
        continue
    f1 = folded(V, F, Vc, best)
    print(f"{os.path.basename(os.path.normpath(r)):40s} {int(f1.sum()):7d} {int((f0 & ~f1).sum()):8d} "
          f"{int((f0 & f1).sum()):6d} {int((~f0 & f1).sum()):7d} {int(len(set(best[f1]))):13d} "
          f"{int(degenerate(V, F, Vc).sum()):10d} {edge_cv(V, F):8.3f}")
