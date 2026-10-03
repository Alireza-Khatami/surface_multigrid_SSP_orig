"""Strict reader of the .c2f bundle (layout of coarse_fine_save_bundle in
coarse_fine_viz.cpp), returning flat arrays for the numba kernels.

The bundle holds everything the relaxation needs from the C++ run:
  coarseV / coarseF  = cmc.Vbase / cmc.Fout (compact coarse mesh)
  fineV / fineF      = gVO / gFO (fine mesh, oriented)
  vtxMap / faceMap   = cmc.newToOld / cmc.faceOrigIdx
  faceSheetID        = gFaceSheetID
  decIM / decInfo    = the SSP collapse history (query_coarse_to_fine input)

Unlike 11_correspond_viz/c2f_query.load_bundle, this reader checks that every
byte of the file is consumed. compare_with_c2f_query() checks that both readers
return the same data.
"""
import struct
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

from log_util import log


@dataclass
class FlatBundle:
    version: int = 0
    NC: int = 0
    NCE: int = 0
    coarseV: np.ndarray = None   # NCE x 3
    coarseF: np.ndarray = None   # FC x 3 (int64)
    fineV: np.ndarray = None     # NF x 3
    fineF: np.ndarray = None     # FF x 3 (int64)
    corrBC: np.ndarray = None    # NC x 3
    corrFV: np.ndarray = None    # NC x 3
    nV_total: int = 0
    vtxMap: np.ndarray = None    # NCE
    faceMap: np.ndarray = None   # FC
    faceSheetID: np.ndarray = None  # nFO
    staleChains: List[List[int]] = field(default_factory=list)
    # decIM as CSR: face f -> decIM_idx[decIM_off[f]:decIM_off[f+1]]
    decIM_off: np.ndarray = None
    decIM_idx: np.ndarray = None
    # decInfo flattened: collapse d -> sheets [sh_off[d], sh_off[d+1])
    sh_off: np.ndarray = None
    sh_gid: np.ndarray = None     # global_sheet_id per sheet
    sh_b: np.ndarray = None       # nSheets x 2 (survivor, absorbed local), -1 when absent
    sh_sv_off: np.ndarray = None  # subsetVIdx CSR
    sv: np.ndarray = None
    sh_uv_off: np.ndarray = None  # UV rows CSR (UV_pre / UV_post / V_pre / V_post share it)
    UV_pre: np.ndarray = None
    UV_post: np.ndarray = None
    V_pre: np.ndarray = None
    V_post: np.ndarray = None
    sh_fpre_off: np.ndarray = None  # FUV_pre / FIdx_pre rows CSR
    FUV_pre: np.ndarray = None
    FIdx_pre: np.ndarray = None
    sh_fpost_off: np.ndarray = None
    FUV_post: np.ndarray = None
    FIdx_post: np.ndarray = None

    @property
    def nDec(self):
        return len(self.sh_off) - 1

    @property
    def FC(self):
        return self.coarseF.shape[0]


class _R:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def u32(self):
        v, = struct.unpack_from('<I', self.d, self.p); self.p += 4; return v

    def i32(self):
        v, = struct.unpack_from('<i', self.d, self.p); self.p += 4; return v

    def arr(self, dtype, n):
        a = np.frombuffer(self.d, dtype=dtype, count=n, offset=self.p)
        self.p += a.itemsize * n
        return a.copy()


def load_bundle_flat(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    r = _R(data)
    magic = r.u32()
    if not 0xC2F50006 <= magic <= 0xC2F50007:
        raise ValueError('unsupported bundle magic 0x%08X (need v6 / v7)' % magic)
    ver = magic - 0xC2F50000
    b = FlatBundle(version=ver)
    NC, FC, NF, FF = r.u32(), r.u32(), r.u32(), r.u32()
    NCE = r.u32() if ver == 7 else NC
    b.NC, b.NCE = NC, NCE
    b.coarseV = r.arr('<f8', NCE * 3).reshape(NCE, 3)
    b.coarseF = r.arr('<u4', FC * 3).reshape(FC, 3).astype(np.int64)
    b.fineV = r.arr('<f8', NF * 3).reshape(NF, 3)
    b.fineF = r.arr('<u4', FF * 3).reshape(FF, 3).astype(np.int64)
    corr = np.frombuffer(data, dtype=np.dtype([('bc', '<f8', 3), ('fv', '<u4', 3)]), count=NC, offset=r.p)
    r.p += 36 * NC
    b.corrBC = corr['bc'].copy()
    b.corrFV = corr['fv'].astype(np.int64)

    b.nV_total = r.u32()
    nF_decIM = r.u32()
    nFO = r.u32()
    b.vtxMap = r.arr('<i4', NCE).astype(np.int64)
    if ver == 7:
        nStale = r.u32()
        for _ in range(nStale):
            n = r.u32()
            b.staleChains.append(r.arr('<u4', n).astype(np.int64).tolist())
    b.faceMap = r.arr('<i4', FC).astype(np.int64)
    b.faceSheetID = r.arr('<i4', nFO).astype(np.int64)

    off = [0]
    idx = []
    for _ in range(nF_decIM):
        cnt = r.u32()
        idx.append(r.arr('<i4', cnt))
        off.append(off[-1] + cnt)
    b.decIM_off = np.array(off, dtype=np.int64)
    b.decIM_idx = np.concatenate(idx).astype(np.int64) if idx else np.zeros(0, np.int64)

    nDec = r.u32()
    sh_off = [0]
    gid, bb = [], []
    sv_off, sv = [0], []
    uv_off, uvpre, uvpost, vpre, vpost = [0], [], [], [], []
    fpre_off, fuvpre, fidxpre = [0], [], []
    fpost_off, fuvpost, fidxpost = [0], [], []
    for _ in range(nDec):
        nS = r.u32()
        for _ in range(nS):
            gid.append(r.i32())
            n = r.u32(); sv.append(r.arr('<i4', n)); sv_off.append(sv_off[-1] + n)
            uvRows = r.u32()
            uvpre.append(r.arr('<f8', 2 * uvRows).reshape(uvRows, 2))
            uvpost.append(r.arr('<f8', 2 * uvRows).reshape(uvRows, 2))
            uv_off.append(uv_off[-1] + uvRows)
            n = r.u32()
            fuvpre.append(r.arr('<i4', 3 * n).reshape(n, 3)); fidxpre.append(r.arr('<i4', n))
            fpre_off.append(fpre_off[-1] + n)
            n = r.u32()
            fuvpost.append(r.arr('<i4', 3 * n).reshape(n, 3)); fidxpost.append(r.arr('<i4', n))
            fpost_off.append(fpost_off[-1] + n)
            bb.append((r.i32(), r.i32()))
            vpre.append(r.arr('<f8', 3 * uvRows).reshape(uvRows, 3))
            vpost.append(r.arr('<f8', 3 * uvRows).reshape(uvRows, 3))
            if r.u32():  # has_dc: not used by the query
                n = r.u32(); r.p += 2 * 16 * n
                n = r.u32(); r.p += 12 * n
                n = r.u32(); r.p += 12 * n
        sh_off.append(sh_off[-1] + nS)
    if r.p != len(data):
        raise ValueError('bundle %s: %d trailing bytes not consumed' % (path, len(data) - r.p))

    cat = lambda L, shape, dt: np.concatenate(L).astype(dt) if L else np.zeros(shape, dt)
    b.sh_off = np.array(sh_off, dtype=np.int64)
    b.sh_gid = np.array(gid, dtype=np.int64)
    b.sh_b = np.array(bb, dtype=np.int64).reshape(-1, 2)
    b.sh_sv_off = np.array(sv_off, dtype=np.int64); b.sv = cat(sv, (0,), np.int64)
    b.sh_uv_off = np.array(uv_off, dtype=np.int64)
    b.UV_pre = cat(uvpre, (0, 2), np.float64); b.UV_post = cat(uvpost, (0, 2), np.float64)
    b.V_pre = cat(vpre, (0, 3), np.float64); b.V_post = cat(vpost, (0, 3), np.float64)
    b.sh_fpre_off = np.array(fpre_off, dtype=np.int64)
    b.FUV_pre = cat(fuvpre, (0, 3), np.int64); b.FIdx_pre = cat(fidxpre, (0,), np.int64)
    b.sh_fpost_off = np.array(fpost_off, dtype=np.int64)
    b.FUV_post = cat(fuvpost, (0, 3), np.int64); b.FIdx_post = cat(fidxpost, (0,), np.int64)
    log('[bundle_io] %s: v%d NC=%d NCE=%d FC=%d NF=%d FF=%d nDec=%d sheets=%d (all %d bytes read)'
          % (path, ver, NC, NCE, FC, NF, FF, nDec, len(gid), len(data)))
    return b


def compare_with_c2f_query(flat, cq):
    """Field-by-field comparison of FlatBundle with c2f_query.load_bundle's
    Bundle. Returns a list of mismatch descriptions (empty = identical)."""
    bad = []

    def eq(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.array_equal(a, b):
            bad.append(name)

    eq('coarseV', flat.coarseV, cq.coarseV)
    eq('coarseF', flat.coarseF, cq.coarseF)
    eq('fineV', flat.fineV, cq.fineV)
    eq('fineF', flat.fineF, cq.fineF)
    eq('corrBC', flat.corrBC, cq.corrBC)
    eq('corrFV', flat.corrFV, cq.corrFV)
    eq('vtxMap', flat.vtxMap, cq.vtxMap)
    eq('faceMap', flat.faceMap, cq.faceMap)
    eq('faceSheetID', flat.faceSheetID, cq.faceSheetID)
    if flat.staleChains != [list(c) for c in cq.staleChains]:
        bad.append('staleChains')
    if len(cq.decIM) != len(flat.decIM_off) - 1:
        bad.append('decIM length')
    else:
        for f, L in enumerate(cq.decIM):
            if list(L) != flat.decIM_idx[flat.decIM_off[f]:flat.decIM_off[f + 1]].tolist():
                bad.append('decIM[%d]' % f)
                break
    if len(cq.decInfo) != flat.nDec:
        bad.append('decInfo length')
        return bad
    s = 0
    for d, cd in enumerate(cq.decInfo):
        if len(cd.sheets) != flat.sh_off[d + 1] - flat.sh_off[d]:
            bad.append('decInfo[%d] sheet count' % d)
            return bad
        for sd in cd.sheets:
            u0, u1 = flat.sh_uv_off[s], flat.sh_uv_off[s + 1]
            p0, p1 = flat.sh_fpre_off[s], flat.sh_fpre_off[s + 1]
            q0, q1 = flat.sh_fpost_off[s], flat.sh_fpost_off[s + 1]
            ok = (sd.global_sheet_id == flat.sh_gid[s]
                  and np.array_equal(sd.subsetVIdx, flat.sv[flat.sh_sv_off[s]:flat.sh_sv_off[s + 1]])
                  and np.array_equal(sd.UV_pre, flat.UV_pre[u0:u1]) and np.array_equal(sd.UV_post, flat.UV_post[u0:u1])
                  and np.array_equal(sd.V_pre, flat.V_pre[u0:u1]) and np.array_equal(sd.V_post, flat.V_post[u0:u1])
                  and np.array_equal(sd.FUV_pre, flat.FUV_pre[p0:p1]) and np.array_equal(sd.FIdx_pre, flat.FIdx_pre[p0:p1])
                  and np.array_equal(sd.FUV_post, flat.FUV_post[q0:q1]) and np.array_equal(sd.FIdx_post, flat.FIdx_post[q0:q1])
                  and np.array_equal(sd.b, flat.sh_b[s]))
            if not ok:
                bad.append('decInfo[%d] sheet %d' % (d, s))
                return bad
            s += 1
    return bad
