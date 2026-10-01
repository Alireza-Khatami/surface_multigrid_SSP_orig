"""Why "flipped vs coarse" / "flipped vs seed" mislead: counts triangles flipped
against the coarse mesh normal and against the seed, and shows that whole coarse
faces (all 1024 triangles) are flipped in the seed, i.e. oriented against the
fine sheet under them rather than folded.

usage: python orientation_vs_coarse.py <run_dir>
"""
import sys

import numpy as np

from relax_metrics_common import coarse_face_of, coarse_mesh, relaxed_positions, seed_positions, tri_normals

if len(sys.argv) != 2:
    sys.exit(__doc__)
run = sys.argv[1]
Vc, F = coarse_mesh(run)
best = coarse_face_of(run, Vc, F)
Vs = seed_positions(run)
nc, ns = tri_normals(Vc, F), tri_normals(Vs, F)
counts = np.bincount(best)
print('subdivided faces', len(F), '| per coarse face', counts.min(), '-', counts.max())
fs_c = np.einsum('ij,ij->i', ns, nc) < 0
print('seed flipped vs coarse normal:', int(fs_c.sum()))
ps = np.bincount(best[fs_c], minlength=len(counts))
print('  coarse faces with any: ', int((ps > 0).sum()), '| fully flipped (all triangles):',
      int((ps == counts).sum()), '| top counts', np.sort(ps)[::-1][:10])
V, name = relaxed_positions(run)
if V is not None:
    nr = tri_normals(V, F)
    fr_s = np.einsum('ij,ij->i', nr, ns) < 0
    fr_c = np.einsum('ij,ij->i', nr, nc) < 0
    print(name)
    print('relaxed flipped vs seed:', int(fr_s.sum()), '| vs coarse:', int(fr_c.sum()))
    print('  seed-flipped that the relaxation un-flipped (counted as "flipped vs seed"):', int((fs_c & ~fr_c).sum()))
