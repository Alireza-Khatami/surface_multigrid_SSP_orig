"""Structure IDs of the subdivided coarse vertices.

Ports of
  coarse_subdiv_relax.cpp      coarse_matstruct (struct IDs of the compact coarse mesh)
  subdiv_struct_ids.h/.cpp     StructPalette, build_struct_sets
  subdiv_relax_projector.h     SetIds, split_palette, intersects, role_of

The C++ takes the coarse vertices' struct IDs and ancestors (fine vertices
merged into each coarse vertex) from simp_viz_tracker. Here both come from the
bundle: every collapse d records (survivor, absorbed) in its sheets' b, so
ancestors[s] |= ancestors[d] in collapse order reproduces
simp_viz_tracker_on_collapse; struct IDs are the union over ancestors.
"""
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

from log_util import log

from matstruct import MatStruct, matstruct_edge_key
from subdiv_mesh import (SUBDIV_CARRIER_EDGE, SUBDIV_CARRIER_FACE, SUBDIV_CARRIER_VERTEX)

# Palette type-mask bits, one per .ma_struct type_id.
STRUCT_MASK_SHEET = 1 << 0
STRUCT_MASK_SEAM = 1 << 1
STRUCT_MASK_BOUNDARY = 1 << 2
STRUCT_MASK_JUNCTION = 1 << 3

RELAX_SHEET = 0
RELAX_CURVE = 1
RELAX_JUNCTION = 2


class StructError(RuntimeError):
    pass


@dataclass
class StructPalette:
    offsets: List[int] = field(default_factory=lambda: [0])
    ids: List[int] = field(default_factory=list)
    typeMask: List[int] = field(default_factory=list)

    def size(self):
        return len(self.typeMask)

    def set(self, k):
        return self.ids[self.offsets[k]:self.offsets[k + 1]]


@dataclass
class SetIds:
    sheet: List[List[int]] = field(default_factory=list)
    curve: List[List[int]] = field(default_factory=list)


def split_palette(pal, ms):
    S = SetIds(sheet=[[] for _ in range(pal.size())], curve=[[] for _ in range(pal.size())])
    if ms is None:
        return S
    for k in range(pal.size()):
        for a in range(pal.offsets[k], pal.offsets[k + 1]):
            i = pal.ids[a]
            t = ms.structType[i]
            if t == 0:
                S.sheet[k].append(i)
            elif t == 1 or t == 2:
                S.curve[k].append(i)
    return S


def intersects(a, b):
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            return True
        if a[i] < b[j]:
            i += 1
        else:
            j += 1
    return False


def role_of(mask):
    if mask & STRUCT_MASK_JUNCTION:
        return RELAX_JUNCTION
    if mask & (STRUCT_MASK_SEAM | STRUCT_MASK_BOUNDARY):
        return RELAX_CURVE
    return RELAX_SHEET


def bundle_ancestors(B):
    """Per SSP vertex (gV id): the set of fine vertices merged into it, from the
    collapse history (b = survivor / absorbed of each collapse's sheets)."""
    nV = max(int(B.nV_total), B.fineV.shape[0])
    anc = [set([v]) if v < B.fineV.shape[0] else set() for v in range(nV)]
    noPair = 0
    for d in range(B.nDec):
        pair = None
        for s in range(B.sh_off[d], B.sh_off[d + 1]):
            if B.sh_b[s, 0] >= 0 and B.sh_b[s, 1] >= 0:
                svo = B.sh_sv_off[s]
                p = (int(B.sv[svo + B.sh_b[s, 0]]), int(B.sv[svo + B.sh_b[s, 1]]))
                if pair is None:
                    pair = p
                elif pair != p:
                    raise StructError('collapse %d: sheets disagree on (survivor, absorbed): %s vs %s'
                                      % (d, pair, p))
        if pair is None:
            noPair += 1
            continue
        s, dv = pair
        anc[s] |= anc[dv]
    if noPair:
        raise StructError('%d collapses without a (survivor, absorbed) pair in the bundle' % noPair)
    return anc


def coarse_matstruct(B, ms, anc):
    """coarse_matstruct(cmc, ms): structure IDs of the compact coarse mesh as a
    MatStruct indexed like it. Raises like the C++ on inconsistent IDs."""
    NCE, FC, nFine = B.coarseV.shape[0], B.coarseF.shape[0], B.fineV.shape[0]
    cms = MatStruct()
    cms.nv, cms.nf = NCE, FC
    cms.structType = ms.structType
    cms.hasStructSection = ms.hasStructSection

    # Vertices: struct IDs tracked through the collapses = union over ancestors.
    cms.vertexIds = []
    idsDiffer = 0
    for i in range(NCE):
        gv = int(B.vtxMap[i])
        ids = set()
        for u in anc[gv]:
            if u < len(ms.vertexIds):
                ids |= ms.vertexIds[u]
        cms.vertexIds.append(ids)
        if gv < len(ms.vertexIds) and ids != ms.vertexIds[gv]:
            idsDiffer += 1

    cms.faceIds = [ms.faceIds[int(B.faceMap[f])] for f in range(FC)]

    owner = [-1] * nFine
    ownerConflicts = 0
    for i in range(NCE):
        for u in anc[int(B.vtxMap[i])]:
            if u < 0 or u >= nFine:
                continue
            if owner[u] >= 0 and owner[u] != i:
                ownerConflicts += 1
            owner[u] = i
    coarseEdges = set()
    CF = B.coarseF.tolist()
    for f in range(FC):
        for c in range(3):
            coarseEdges.add(matstruct_edge_key(CF[f][c], CF[f][(c + 1) % 3]))

    fineCurveEdges = inside = noOwner = notCoarseEdge = 0
    edgeIds = {}
    for e0, e1 in ms.maEdges:
        ids = ms.edgeIds.get(matstruct_edge_key(e0, e1))
        if not ids:
            continue
        fineCurveEdges += 1
        a = owner[e0] if 0 <= e0 < nFine else -1
        b = owner[e1] if 0 <= e1 < nFine else -1
        if a < 0 or b < 0:
            noOwner += 1
            continue
        if a == b:
            inside += 1
            continue
        key = matstruct_edge_key(a, b)
        if key not in coarseEdges:
            notCoarseEdge += 1
            continue
        edgeIds.setdefault(key, []).extend(ids)
    notOnEndpoints = 0
    for key in edgeIds:
        edgeIds[key] = sorted(set(edgeIds[key]))
        a, b = key >> 32, key & 0xffffffff
        for i in edgeIds[key]:
            if i not in cms.vertexIds[a] or i not in cms.vertexIds[b]:
                notOnEndpoints += 1
                break
    cms.edgeIds = edgeIds
    log('[coarse_subdiv_relax] coarse struct IDs: %d vertices (%d differ from the fine vertex\'s IDs), %d faces, '
          '%d curve edges from %d fine curve edges (%d collapsed inside a coarse vertex, %d without owner, '
          '%d not a coarse edge); %d owner conflicts, %d curve edges whose IDs are not on both endpoints'
          % (NCE, idsDiffer, FC, len(edgeIds), fineCurveEdges, inside, noOwner, notCoarseEdge,
             ownerConflicts, notOnEndpoints))
    if idsDiffer or noOwner or notCoarseEdge or ownerConflicts or notOnEndpoints:
        raise StructError('[coarse_subdiv_relax] coarse structure IDs are inconsistent (see counts above)')
    return cms


def build_struct_sets(M, FO, ms):
    """Palette + one set index per subdivided vertex (carrier rules):
    VERTEX v: vertexIds[v]; EDGE e: seam/boundary ids of e U sheet ids of its
    faces; FACE f: sheet ids of f. Sets interned in first-seen order."""
    pal = StructPalette()
    interned = {}

    def intern(ids):
        t = tuple(ids)
        k = interned.get(t)
        if k is not None:
            return k
        k = len(pal.typeMask)
        mask = 0
        for i in ids:
            typ = ms.structType[i] if ms is not None else -1
            if 0 <= typ <= 3:
                mask |= 1 << typ
        pal.ids.extend(ids)
        pal.offsets.append(len(pal.ids))
        pal.typeMask.append(mask)
        interned[t] = k
        return k

    Vs = len(M.carrierType)
    if ms is None:
        e = intern([])
        return pal, np.full(Vs, e, dtype=np.int64)

    FO = np.asarray(FO, dtype=np.int64)
    FOl = FO.tolist()
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(M.origEdges.tolist())}
    edgeFaces = [[] for _ in range(M.origEdges.shape[0])]
    for f in range(len(FOl)):
        for c in range(3):
            a, b = FOl[f][c], FOl[f][(c + 1) % 3]
            edgeFaces[eidx[(min(a, b), max(a, b))]].append(f)

    vCache, eCache, fCache = {}, {}, {}
    setId = np.full(Vs, -1, dtype=np.int64)
    ct, ci = M.carrierType.tolist(), M.carrierIndex.tolist()
    oE = M.origEdges.tolist()
    for v in range(Vs):
        idx = ci[v]
        t = ct[v]
        if t == SUBDIV_CARRIER_VERTEX:
            if idx not in vCache:
                vCache[idx] = intern(sorted(ms.vertexIds[idx]))
            setId[v] = vCache[idx]
        elif t == SUBDIV_CARRIER_EDGE:
            if idx not in eCache:
                ids = list(ms.edgeIds.get(matstruct_edge_key(oE[idx][0], oE[idx][1]), []))
                for f in edgeFaces[idx]:
                    ids.extend(ms.faceIds[f])
                eCache[idx] = intern(sorted(set(ids)))
            setId[v] = eCache[idx]
        elif t == SUBDIV_CARRIER_FACE:
            if idx not in fCache:
                fCache[idx] = intern(list(ms.faceIds[idx]))
            setId[v] = fCache[idx]
    maxSet = max((pal.offsets[k + 1] - pal.offsets[k] for k in range(pal.size())), default=0)
    log('[subdiv_struct_ids] palette: %d distinct sets, %d ids total, largest set %d'
          % (pal.size(), len(pal.ids), maxSet))
    return pal, setId
