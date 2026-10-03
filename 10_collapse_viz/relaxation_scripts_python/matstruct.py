"""Port of load_matstruct.h / load_matstruct.cpp: the .ma_struct reader.

File format:
  <nv> <ne> <nf>
  v x y z r          (nv lines)
  e v0 v1            (ne lines)
  f v0 v1 v2         (nf lines)
  <num_structures>
  <struct_id> <type_id> <count>
  <elem_id_0> ... <elem_id_N>
type_id: 0 SHEET (faces), 1 SEAM (edges), 2 BOUNDARY (edges), 3 JUNCTION (vertices).
"""
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from log_util import log


def matstruct_edge_key(a, b):
    """Order-independent key of edge (a, b)."""
    a, b = int(a), int(b)
    if a > b:
        a, b = b, a
    return ((a & 0xffffffff) << 32) | (b & 0xffffffff)


@dataclass
class MatStructEntry:
    id: int = -1
    type: int = -1
    elements: List[int] = field(default_factory=list)


@dataclass
class MatStruct:
    nv: int = 0
    ne: int = 0
    nf: int = 0
    maEdges: List[tuple] = field(default_factory=list)       # "e v0 v1" lines, file order
    structs: List[MatStructEntry] = field(default_factory=list)
    structType: Dict[int, int] = field(default_factory=dict)  # struct id -> type_id
    hasStructSection: bool = False
    # Derived, indexed like the fine mesh.
    vertexIds: List[set] = field(default_factory=list)       # std::set<int> per vertex
    faceIds: List[List[int]] = field(default_factory=list)   # per face: sheet ids, sorted
    edgeIds: Dict[int, List[int]] = field(default_factory=dict)  # edge key -> seam/boundary ids, sorted
    nEdgesWithoutFace: int = 0
    maxPosDiff: float = 0.0


class MatStructError(RuntimeError):
    pass


def load_matstruct(fname, VO, FO):
    """Reads fname and checks it describes the fine mesh (VO, FO). Raises
    MatStructError on any I/O or consistency error (the C++ returns false)."""
    def fail(msg):
        raise MatStructError('[load_matstruct] ' + fname + ': ' + msg)

    try:
        with open(fname, 'r') as fh:
            tok = fh.read().split()
    except OSError:
        fail('cannot open')
    pos = 0

    def nxt():
        nonlocal pos
        if pos >= len(tok):
            raise IndexError
        t = tok[pos]
        pos += 1
        return t

    out = MatStruct()
    try:
        nv, ne, nf = int(nxt()), int(nxt()), int(nxt())
    except (IndexError, ValueError):
        fail('bad header')
    if nv <= 0 or ne < 0 or nf < 0:
        fail('bad header')
    if nv != VO.shape[0] or nf != FO.shape[0]:
        fail('counts do not match the mesh: .ma has %d vertices / %d faces, mesh has %d / %d'
             % (nv, nf, VO.shape[0], FO.shape[0]))
    out.nv, out.ne, out.nf = nv, ne, nf

    for i in range(nv):
        try:
            ch = nxt(); x = float(nxt()); y = float(nxt()); z = float(nxt()); float(nxt())
        except (IndexError, ValueError):
            fail('bad vertex line %d' % i)
        if ch != 'v':
            fail('bad vertex line %d' % i)
        out.maxPosDiff = max(out.maxPosDiff, abs(x - VO[i, 0]), abs(y - VO[i, 1]), abs(z - VO[i, 2]))

    out.maEdges = []
    for i in range(ne):
        try:
            ch = nxt(); a = int(nxt()); b = int(nxt())
        except (IndexError, ValueError):
            fail('bad edge line %d' % i)
        if ch != 'e':
            fail('bad edge line %d' % i)
        if not (0 <= a < nv) or not (0 <= b < nv):
            fail('edge %d has an out-of-range vertex' % i)
        out.maEdges.append((a, b))

    for i in range(nf):
        try:
            ch = nxt(); a = [int(nxt()), int(nxt()), int(nxt())]
        except (IndexError, ValueError):
            fail('bad face line %d' % i)
        if ch != 'f':
            fail('bad face line %d' % i)
        if sorted(a) != sorted(int(x) for x in FO[i]):
            fail('face %d has different corners than mesh face %d' % (i, i))

    out.vertexIds = [set() for _ in range(nv)]
    out.faceIds = [[] for _ in range(nf)]

    try:
        numStructs = int(nxt())
    except (IndexError, ValueError):
        log('[load_matstruct] no struct section in %s' % fname)
        return out
    out.hasStructSection = True

    faceEdges = set()
    FOl = FO.tolist()
    for fi in range(nf):
        for c in range(3):
            faceEdges.add(matstruct_edge_key(FOl[fi][c], FOl[fi][(c + 1) % 3]))

    for s in range(numStructs):
        st = MatStructEntry()
        try:
            st.id = int(nxt()); st.type = int(nxt()); count = int(nxt())
        except (IndexError, ValueError):
            fail('truncated struct header %d' % s)
        if count < 0:
            fail('truncated struct header %d' % s)
        if st.id in out.structType:
            fail('duplicate struct id %d' % st.id)
        out.structType[st.id] = st.type
        st.elements = []
        for _ in range(count):
            try:
                el = int(nxt())
            except (IndexError, ValueError):
                fail('truncated struct %d' % st.id)
            st.elements.append(el)
            if st.type == 0:
                if el < 0 or el >= nf:
                    fail('sheet face out of range in struct %d' % st.id)
                out.faceIds[el].append(st.id)
                for c in range(3):
                    out.vertexIds[FOl[el][c]].add(st.id)
            elif st.type in (1, 2):
                if el < 0 or el >= ne:
                    fail('edge out of range in struct %d' % st.id)
                a, b = out.maEdges[el]
                out.vertexIds[a].add(st.id)
                out.vertexIds[b].add(st.id)
                k = matstruct_edge_key(a, b)
                out.edgeIds.setdefault(k, []).append(st.id)
                if k not in faceEdges:
                    out.nEdgesWithoutFace += 1
            elif st.type == 3:
                if el < 0 or el >= nv:
                    fail('junction vertex out of range in struct %d' % st.id)
                out.vertexIds[el].add(st.id)
        out.structs.append(st)

    out.faceIds = [sorted(set(v)) for v in out.faceIds]
    out.edgeIds = {k: sorted(set(v)) for k, v in out.edgeIds.items()}

    nFacesNoSheet = sum(1 for v in out.faceIds if not v)
    log('[load_matstruct] %s: %d structs, max |.ma pos - mesh pos| = %.3g, '
          '%d/%d faces in no sheet, %d seam/boundary edges on no face'
          % (fname, numStructs, out.maxPosDiff, nFacesNoSheet, nf, out.nEdgesWithoutFace))
    return out
