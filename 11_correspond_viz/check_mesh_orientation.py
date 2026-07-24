#!/usr/bin/env python3
"""
check_mesh_orientation.py

Checks face-orientation consistency of a triangle mesh (.off / .obj).

For each pair of faces sharing a manifold interior edge, the shared edge
must be traversed in OPPOSITE directions (a->b in one face, b->a in the other)
for the mesh to be consistently oriented (outward-normals convention).

If the QMAT mesh has inconsistently oriented faces, joint_lscm's
vector_area_matrix receives mixed-sign area contributions, which causes
some UV triangles to flip during the LSCM solve.

Usage:
    python check_mesh_orientation.py [path_to_mesh.off]
    (defaults to the QMAT mat_initial.off used in the collapse visualizer)
"""

import sys
import os
import numpy as np
from collections import defaultdict, deque


DEFAULT_MESH = (
    r"C:\Users\alirz\Projects\Graphics\QMAT_old working version  exe file"
    r"\qmat_x64\qmat\output"
    r"\01_00040057_f8f78dbd17414efda75bc437_trimesh_000"
    r"\01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial.off"
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_off(path):
    with open(path, 'r') as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith('#')]

    idx = 0
    # Accept OFF, COFF, NOFF, etc.
    if lines[idx].upper().lstrip('CN').startswith('OFF'):
        idx += 1

    nV, nF, _ = map(int, lines[idx].split())
    idx += 1

    V = np.array([list(map(float, lines[idx + i].split()[:3])) for i in range(nV)])
    idx += nV

    F = []
    for i in range(nF):
        parts = list(map(int, lines[idx + i].split()))
        F.append(parts[1: parts[0] + 1])

    return V, F


def parse_obj(path):
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            t = line.strip().split()
            if not t:
                continue
            if t[0] == 'v':
                V.append(list(map(float, t[1:4])))
            elif t[0] == 'f':
                F.append([int(tok.split('/')[0]) - 1 for tok in t[1:]])
    return np.array(V), F


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.off':
        return parse_off(path)
    elif ext == '.obj':
        return parse_obj(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


# ---------------------------------------------------------------------------
# Edge table
# ---------------------------------------------------------------------------

def build_edge_table(F_tri):
    """
    Returns {(min_v, max_v): [(face_idx, is_forward), ...]}
    is_forward = True if the face traverses min_v -> max_v.
    """
    table = defaultdict(list)
    for fi, face in enumerate(F_tri):
        for k in range(3):
            a, b = face[k], face[(k + 1) % 3]
            key = (min(a, b), max(a, b))
            table[key].append((fi, a < b))
    return table


# ---------------------------------------------------------------------------
# Orientation BFS
# ---------------------------------------------------------------------------

def check_orientation(F_tri, edge_table):
    """
    BFS from face 0 assigns a consistent orientation to each reachable face.

    orientation[fi] = 0 : same winding as seed face 0
    orientation[fi] = 1 : opposite winding  (needs flip)
    orientation[fi] = -1: unreachable (isolated face — no manifold interior neighbours)

    bad_edges: list of (fi, fj) where the BFS-assigned orientations conflict,
               indicating a non-orientable loop in the mesh.

    Returns (orientation, bad_edges, n_components, n_flipped)
    """
    nF = len(F_tri)

    # Build undirected adjacency restricted to manifold interior edges
    adjacency = defaultdict(list)   # fi -> [(fj, is_consistent)]
    for edge, face_list in edge_table.items():
        if len(face_list) != 2:
            continue
        (fi, dir_i), (fj, dir_j) = face_list
        # Consistent = they traverse the shared edge in OPPOSITE directions
        consistent = (dir_i != dir_j)
        adjacency[fi].append((fj, consistent))
        adjacency[fj].append((fi, consistent))

    orientation   = np.full(nF, -1, dtype=int)
    bad_edges     = []
    n_components  = 0
    n_flipped     = 0

    for seed in range(nF):
        if orientation[seed] != -1:
            continue
        n_components += 1
        orientation[seed] = 0
        q = deque([seed])
        while q:
            fi = q.popleft()
            for fj, consistent in adjacency[fi]:
                expected = orientation[fi] if consistent else (1 - orientation[fi])
                if orientation[fj] == -1:
                    orientation[fj] = expected
                    if expected == 1:
                        n_flipped += 1
                    q.append(fj)
                elif orientation[fj] != expected:
                    bad_edges.append((fi, fj))

    return orientation, bad_edges, n_components, n_flipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MESH

    print(f"File : {os.path.basename(path)}")
    print(f"Path : {path}")

    if not os.path.exists(path):
        print("\nERROR: file not found.")
        sys.exit(1)

    print("Loading...", flush=True)
    V, F = load_mesh(path)

    F_tri   = [f for f in F if len(f) == 3]
    F_other = [f for f in F if len(f) != 3]

    # ------------------------------------------------------------------
    print(f"\n{'='*54}")
    print(f"  MESH STATISTICS")
    print(f"{'='*54}")
    print(f"  Vertices           : {len(V)}")
    print(f"  Total faces        : {len(F)}")
    print(f"  Triangle faces     : {len(F_tri)}")
    if F_other:
        counts = defaultdict(int)
        for f in F_other:
            counts[len(f)] += 1
        for k, v in sorted(counts.items()):
            print(f"  {k}-gon faces       : {v}")

    # ------------------------------------------------------------------
    edge_table  = build_edge_table(F_tri)
    n_interior  = sum(1 for fs in edge_table.values() if len(fs) == 2)
    n_boundary  = sum(1 for fs in edge_table.values() if len(fs) == 1)
    n_nonmanif  = sum(1 for fs in edge_table.values() if len(fs) > 2)
    nm_sizes    = [len(fs) for fs in edge_table.values() if len(fs) > 2]

    print(f"\n{'='*54}")
    print(f"  EDGE STATISTICS")
    print(f"{'='*54}")
    print(f"  Interior edges  (2 faces) : {n_interior}")
    print(f"  Boundary edges  (1 face)  : {n_boundary}")
    print(f"  Non-manifold edges (3+)   : {n_nonmanif}")
    if nm_sizes:
        print(f"  Non-manifold fan sizes    : max={max(nm_sizes)}  "
              f"mean={np.mean(nm_sizes):.1f}")
    is_closed_mfld = (n_boundary == 0 and n_nonmanif == 0)
    print(f"  Closed manifold?          : {'YES' if is_closed_mfld else 'NO'}")

    # ------------------------------------------------------------------
    print(f"\n{'='*54}")
    print(f"  FACE ORIENTATION CONSISTENCY")
    print(f"{'='*54}")
    print("  Running BFS orientation check...", flush=True)

    orientation, bad_edges, n_comp, n_flipped = check_orientation(F_tri, edge_table)

    n_consistent  = int((orientation == 0).sum())
    n_unreachable = int((orientation == -1).sum())

    print(f"  Connected components       : {n_comp}")
    print(f"  Consistent with seed face  : {n_consistent}")
    print(f"  FLIPPED (need re-winding)  : {n_flipped}")
    print(f"  Unreachable (isolated)     : {n_unreachable}")
    print(f"  Non-orientable conflicts   : {len(bad_edges)}")

    # ------------------------------------------------------------------
    print(f"\n{'='*54}")
    print(f"  VERDICT")
    print(f"{'='*54}")

    if n_flipped == 0 and len(bad_edges) == 0:
        print("  CONSISTENT — all faces have the same winding convention.")
        print("  joint_lscm vector_area_matrix will receive correct signs.")
        print("  Inconsistent orientation is NOT the cause of flipped UV triangles.")

    elif len(bad_edges) == 0:
        frac = 100.0 * n_flipped / len(F_tri)
        print(f"  INCONSISTENT — {n_flipped} / {len(F_tri)} faces ({frac:.1f}%) have")
        print(f"  reversed winding relative to the seed face.")
        print()
        print(f"  No non-orientable topology — can be fixed by re-winding.")
        print()
        print(f"  EFFECT ON joint_lscm:")
        print(f"    vector_area_matrix (flatten(), ~line 505) accumulates NEGATED")
        print(f"    area contributions for every reversed face.  The LSCM solver")
        print(f"    then tries to reflect those faces in UV space, producing the")
        print(f"    flipped triangles visible in the visualizer.")
        print(f"    check_valid_UV_lscm then rejects those collapses (signedArea<0),")
        print(f"    so they pile up with infinite cost in the priority queue.")

        # Sample reversed faces
        flipped_idx = np.where(orientation == 1)[0]
        show_n = min(20, len(flipped_idx))
        print(f"\n  Sample of reversed faces (first {show_n} of {n_flipped}):")
        for fi in flipped_idx[:show_n]:
            print(f"    face {fi:7d} : {F_tri[fi]}")

    else:
        print(f"  NON-ORIENTABLE — {len(bad_edges)} conflict(s) found.")
        print(f"  Mesh contains Mobius-like topology; cannot be globally re-wound.")

    # ------------------------------------------------------------------
    print(f"\n{'='*54}")
    print(f"  QUICK FIX HINT")
    print(f"{'='*54}")
    if n_flipped > 0 and len(bad_edges) == 0:
        print("  For each face fi where orientation[fi] == 1, swap vertices 1 and 2:")
        print("    F[fi] = [F[fi][0], F[fi][2], F[fi][1]]")
        print("  This re-winds without changing the mesh topology.")
    elif n_flipped == 0:
        print("  No face re-winding needed — look elsewhere for the UV flip cause.")
    else:
        print("  Non-orientable topology: check mesh export / seam handling.")


if __name__ == "__main__":
    main()
