"""Port of build_relax_graph (subdiv_sample_tracker/subdiv_relax.cpp).

Row i = the vertices that pull on i (CSR, ascending within a row):
  JUNCTION row: empty (never moves)
  CURVE row   : CURVE/JUNCTION neighbours sharing a seam/boundary id
  SHEET row   : any neighbour sharing a sheet id (or both without sheet ids)
With ms = None every vertex is SHEET and the graph is the plain two-way mesh
adjacency (the "symmetric" graph of the explicit relaxation).
"""
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from log_util import log

from struct_ids import (RELAX_CURVE, RELAX_JUNCTION, RELAX_SHEET, intersects, role_of, split_palette)


@dataclass
class RelaxGraph:
    role: np.ndarray = None     # uint8, Vs
    rowOffs: np.ndarray = None  # int64, Vs+1
    cols: np.ndarray = None     # int64, nnz
    nUndirected: int = 0
    kind: list = field(default_factory=lambda: [0] * 5)  # S<-S, S<-C, S<-J, C<-C, C<-J
    droppedSheetSheet: int = 0
    sheetWithoutSheetId: int = 0
    isolatedSheet: int = 0
    isolatedCurve: int = 0
    nRole: list = field(default_factory=lambda: [0] * 3)


def build_relax_graph(F, Vs, pal, setId, ms):
    t0 = time.perf_counter()
    G = RelaxGraph()
    setId = np.asarray(setId, dtype=np.int64)
    if setId.shape[0] != Vs:
        raise RuntimeError('[subdiv_relax] setId size mismatch')
    S = split_palette(pal, ms)
    setRole = np.array([role_of(pal.typeMask[k]) if ms is not None else RELAX_SHEET for k in range(pal.size())],
                       dtype=np.uint8)
    G.role = setRole[setId]
    for r in range(3):
        G.nRole[r] = int((G.role == r).sum())
    if ms is not None:
        sheetEmpty = np.array([len(S.sheet[k]) == 0 for k in range(pal.size())], dtype=bool)
        G.sheetWithoutSheetId = int(((G.role == RELAX_SHEET) & sheetEmpty[setId]).sum())

    # unique undirected edges of F
    F = np.asarray(F, dtype=np.int64)
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    keys = np.unique((np.minimum(a, b) << 32) | np.maximum(a, b))
    G.nUndirected = len(keys)
    ea, eb = keys >> 32, keys & 0xffffffff

    if ms is None:
        keepAB = np.ones(len(keys), dtype=bool)
        keepBA = np.ones(len(keys), dtype=bool)
    else:
        # share(ka, kb) per palette pair: bit0 share sheet, bit1 share curve, bit2 both without sheet ids
        P = pal.size()
        share = np.zeros((P, P), dtype=np.uint8)
        for ka in range(P):
            for kb in range(ka, P):
                r = 0
                if intersects(S.sheet[ka], S.sheet[kb]):
                    r |= 1
                if intersects(S.curve[ka], S.curve[kb]):
                    r |= 2
                if not S.sheet[ka] and not S.sheet[kb]:
                    r |= 4
                share[ka, kb] = share[kb, ka] = r
        sAB = share[setId[ea], setId[eb]]
        ra, rb = G.role[ea], G.role[eb]

        def keep(ri, rj, s):
            return np.where(ri == RELAX_JUNCTION, False,
                            np.where(ri == RELAX_CURVE, (rj != RELAX_SHEET) & ((s & 2) != 0),
                                     ((s & 1) != 0) | ((s & 4) != 0)))
        keepAB = keep(ra, rb, sAB)
        keepBA = keep(rb, ra, sAB)

    def kind_of(ri, rj):
        return np.where(ri == RELAX_SHEET, np.where(rj == RELAX_SHEET, 0, np.where(rj == RELAX_CURVE, 1, 2)),
                        np.where(rj == RELAX_CURVE, 3, 4))
    ra, rb = G.role[ea], G.role[eb]
    kab = kind_of(ra, rb)[keepAB]
    kba = kind_of(rb, ra)[keepBA]
    for k in range(5):
        G.kind[k] = int((kab == k).sum() + (kba == k).sum())
    both = keepAB & keepBA
    G.droppedSheetSheet = int(((ra == RELAX_SHEET) & (rb == RELAX_SHEET) & ~both).sum())

    rows = np.concatenate([ea[keepAB], eb[keepBA]])
    cols = np.concatenate([eb[keepAB], ea[keepBA]])
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    G.cols = cols.astype(np.int64)
    G.rowOffs = np.zeros(Vs + 1, dtype=np.int64)
    np.add.at(G.rowOffs, rows + 1, 1)
    G.rowOffs = np.cumsum(G.rowOffs)
    deg = np.diff(G.rowOffs)
    G.isolatedSheet = int(((deg == 0) & (G.role == RELAX_SHEET)).sum())
    G.isolatedCurve = int(((deg == 0) & (G.role == RELAX_CURVE)).sum())

    log('[subdiv_relax] graph: %d vertices (sheet %d, curve %d, junction %d), %d undirected edges, '
          '%d directed entries (%.2f s)\n'
          '[subdiv_relax]   kept: S<-S %d, S<-C %d, S<-J %d, C<-C %d, C<-J %d | '
          'dropped sheet-sheet %d | sheet without sheet id %d | isolated sheet %d, curve %d'
          % (Vs, G.nRole[0], G.nRole[1], G.nRole[2], G.nUndirected, len(G.cols), time.perf_counter() - t0,
             *G.kind, G.droppedSheetSheet, G.sheetWithoutSheetId, G.isolatedSheet, G.isolatedCurve),
          )
    return G
