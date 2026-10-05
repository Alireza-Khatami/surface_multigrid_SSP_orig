"""Retired (2026-10-05) from relax_viewer.py: the first concave mask.

Concave = a fine edge with exactly two faces whose dihedral bends inward by more than a
threshold (seams and boundaries skipped); points marked from their CURRENT positions
each step. Replaced by concave_parts.py (concave corners of the seam / boundary
structures of the .ma_struct, marked once on the relaxation input).
Not imported anywhere; kept for reference.
"""
import numpy as np


def concave_fine_edges(V, F, angle_deg):
    """Fine edges whose two faces meet at a concave dihedral angle larger than
    angle_deg. Normals made consistent across the edge (the second face's normal
    is flipped when both faces run the edge in the same direction); edges with
    other than two faces are skipped. Returns (edges (m,2), mask over origEdges order)."""
    F = np.asarray(F, dtype=np.int64)
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    opp = F[:, [2, 0, 1]].reshape(-1)
    fid = np.repeat(np.arange(F.shape[0]), 3)
    key = (np.minimum(a, b) << 32) | np.maximum(a, b)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[start, len(ks)])
    two = start[cnt == 2]
    h1, h2 = order[two], order[two + 1]
    n = np.cross(V[b] - V[a], V[opp] - V[a])
    nf = n[np.arange(0, len(a), 3)]
    nf = nf / np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-300)
    n1 = nf[fid[h1]]
    n2 = nf[fid[h2]] * np.where(a[h1] == a[h2], -1.0, 1.0)[:, None]  # same direction: inconsistent
    s = np.einsum('ij,ij->i', n1, V[opp[h2]] - V[a[h1]])
    bend = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2), -1.0, 1.0)))
    conc = (s > 0) & (bend > angle_deg)
    E = np.stack([np.minimum(a[h1], b[h1]), np.maximum(a[h1], b[h1])], axis=1)[conc]
    return E



# RelaxViewer.concave_mask (method, as it was):
#     def concave_mask(self):
#         """Points within k rings of a concave fine edge. Seeds: points whose current
#         fine face (from the relaxation state) has a concave edge closer than the mean
#         subdivided edge length; then k rings over the subdivided mesh."""
#         B = self.B
#         if self.concEdges is None or self.concEdges[0] != self.concAngle:
#             E = concave_fine_edges(B.fineV, B.fineF, self.concAngle)
#             self.concEdges = (self.concAngle, E)
#             # per fine face corner c: is edge (F[f,c], F[f,c+1]) concave
#             FO = B.fineF
#             ck = np.sort((E[:, 0] << 32) | E[:, 1])
#             self.concCorner = np.zeros((FO.shape[0], 3), bool)
#             for c in range(3):
#                 a, b = FO[:, c], FO[:, (c + 1) % 3]
#                 k = (np.minimum(a, b) << 32) | np.maximum(a, b)
#                 j = np.clip(np.searchsorted(ck, k), 0, max(len(ck) - 1, 0))
#                 self.concCorner[:, c] = (len(ck) > 0) & (ck[j] == k) if len(ck) else False
#             if len(E):
#                 nodes, edges = B.fineV, E
#                 self.ps.register_curve_network('concave fine edges', nodes, edges, color=(0.9, 0.1, 0.1),
#                                                radius=0.002).add_to_group('fine MAT')
#             elif self.ps.has_curve_network('concave fine edges'):
#                 self.ps.remove_curve_network('concave fine edges')
#         X, face = self.rel.X, self.rel.face
#         FO = B.fineF
#         ok = face >= 0
#         mask = np.zeros(self.Vs, bool)
#         if not self.concCorner.any():
#             return mask
#         fc = np.where(ok, face, 0)
#         Fe = self.F
#         el = np.linalg.norm(X[Fe[:, 0]] - X[Fe[:, 1]], axis=1).mean()
#         for c in range(3):
#             a, b = FO[fc, c], FO[fc, (c + 1) % 3]
#             isc = self.concCorner[fc, c] & ok
#             idx = np.nonzero(isc)[0]
#             d = seg_dist(X[idx], B.fineV[a[idx]], B.fineV[b[idx]])
#             mask[idx[d <= el]] = True
#         G = self.rel.G
#         rows = np.repeat(np.arange(self.Vs), np.diff(G.rowOffs))
#         for _ in range(self.concK):
#             grow = np.zeros(self.Vs, bool)
#             grow[rows[mask[G.cols]]] = True
#             mask |= grow
#         return mask
# 
#     # -------------------------------------------------------------- selection
