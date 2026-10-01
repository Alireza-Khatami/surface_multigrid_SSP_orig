"""Folded / degenerate / edge CV of every snapshot of an explicit run
(coarse_subdiv_at_fine_pos_relaxed_explicit_*_it<N>.obj), plus seed and final.

usage: python snapshot_table.py <run_dir>
"""
import sys

from relax_metrics_common import (coarse_face_of, coarse_mesh, degenerate, edge_cv, folded, load_obj,
                                  relaxed_positions, seed_positions, snapshots)

if len(sys.argv) != 2:
    sys.exit(__doc__)
run = sys.argv[1]
Vc, F = coarse_mesh(run)
best = coarse_face_of(run, Vc, F)
Vs = seed_positions(run)
f0 = folded(Vs, F, Vc, best)
rows = [('seed', Vs)] + [(f'iteration {it}', load_obj(p)[0]) for it, p in snapshots(run)]
V, _ = relaxed_positions(run)
if V is not None:
    rows.append(('final', V))
print(f"{'state':16s} {'folded':>7s} {'removed':>8s} {'new':>7s} {'degenerate':>10s} {'edge CV':>8s}")
for name, P in rows:
    f1 = folded(P, F, Vc, best)
    print(f"{name:16s} {int(f1.sum()):7d} {int((f0 & ~f1).sum()):8d} {int((~f0 & f1).sum()):7d} "
          f"{int(degenerate(P, F, Vc).sum()):10d} {edge_cv(P, F):8.3f}")
