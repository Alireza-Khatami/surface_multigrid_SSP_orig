"""Validate the laplacian_graph PLYs of a run: parse, size vs header, structure
ids, fixed flags, and that every vertex sits at the seed position of its vid.

usage: python check_graph_ply.py <run_dir>
"""
import glob
import os
import sys

import numpy as np

from relax_metrics_common import read_graph_ply, seed_positions

if len(sys.argv) != 2:
    sys.exit(__doc__)
run = sys.argv[1]
Vs = seed_positions(run)
for p in sorted(glob.glob(os.path.join(run, 'laplacian_graph', '*.ply'))):
    v, e = read_graph_ply(p)
    ids = len(set(e['id'].tolist())) if len(e) else len(set(map(tuple, v['c'].tolist())))
    off = np.abs(v['p'] - Vs[v['vid']]).max() if len(v) else 0.0
    print(f"{os.path.basename(p)[:40]:40s} vertices {len(v):8d} edges {len(e):7d} ids {ids:3d} "
          f"fixed {int(v['fixed'].sum()):7d} max |pos - seed| {off:.3g}")
