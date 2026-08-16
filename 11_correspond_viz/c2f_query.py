"""
c2f_query.py — Python loader and correspondence query for .c2f bundles.

Mirrors what 11_correspond_viz/ does in C++:
  - load_bundle()              reads the binary .c2f file
  - compute_barycentric_2d()   2-D barycentric coords in UV space
  - query_coarse_to_fine()     walks the SSP decimation sequence backwards
  - sample_face_correspondence() samples a coarse face and maps to fine mesh

Usage:
    python c2f_query.py bundle.c2f [--face 42] [--n 50]
"""

import struct
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures (mirrors bundle.h + single_collapse_data.h)
# ---------------------------------------------------------------------------

@dataclass
class SheetData:
    global_sheet_id: int
    subsetVIdx: np.ndarray   # (svSz,)  int32 — global vertex indices, sorted asc
    UV_pre:  np.ndarray      # (uvRows, 2) float64
    UV_post: np.ndarray      # (uvRows, 2) float64
    FUV_pre: np.ndarray      # (fuvRows, 3) int32 — local vertex indices, pre-collapse
    FIdx_pre: np.ndarray     # (fuvRows,)   int32 — global face indices, pre-collapse
    FUV_post: np.ndarray     # (fuvPostRows, 3) int32 — local vertex indices, post-collapse
    FIdx_post: np.ndarray    # (fuvPostRows,)   int32 — global face indices, post-collapse
    b: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    # b[0]=survivor_local, b[1]=absorbed_local  (v4+ only; empty for v2/v3)
    V_pre:  np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    V_post: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    # V_pre/V_post: 3D ring geometry in local index space (v5+ only; empty otherwise)


@dataclass
class CollapseData:
    sheets: List[SheetData] = field(default_factory=list)


@dataclass
class Bundle:
    # Compact coarse mesh
    coarseV: np.ndarray      # (NC, 3) float64
    coarseF: np.ndarray      # (FC, 3) int32

    # Original fine mesh
    fineV: np.ndarray        # (NF, 3) float64
    fineF: np.ndarray        # (FF, 3) int32

    # Vertex correspondence: fine_pos = coarseV[i] + corrVec[i]
    corrVec: np.ndarray      # (NC, 3) float64
    corrBC:  np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # (NC, 3) barycentric weights
    corrFV:  np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))  # (NC, 3) fine vertex indices

    # v2 SSP query data (present only if has_ssp_data)
    has_ssp_data: bool = False
    nV_total: int = 0        # global SSP vertex count (includes infVtx)
    nF_total: int = 0        # original face count (= nFO)
    vtxMap: Optional[np.ndarray] = None    # (NC,) int32  compact→global vertex
    faceMap: Optional[np.ndarray] = None   # (FC,) int32  compact→global face
    faceSheetID: Optional[np.ndarray] = None   # (nFO,) int32
    decIM: List[List[int]] = field(default_factory=list)
    decInfo: List[CollapseData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Binary reader helpers
# ---------------------------------------------------------------------------

class _Reader:
    """Minimal binary reader wrapping a bytes buffer."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def u32(self) -> int:
        v, = struct.unpack_from('<I', self._data, self._pos)
        self._pos += 4
        return v

    def i32(self) -> int:
        v, = struct.unpack_from('<i', self._data, self._pos)
        self._pos += 4
        return v

    def f64(self) -> float:
        v, = struct.unpack_from('<d', self._data, self._pos)
        self._pos += 8
        return v

    def u32_array(self, n: int) -> np.ndarray:
        arr = np.frombuffer(self._data, dtype='<u4', count=n, offset=self._pos)
        self._pos += 4 * n
        return arr.astype(np.int32)

    def i32_array(self, n: int) -> np.ndarray:
        arr = np.frombuffer(self._data, dtype='<i4', count=n, offset=self._pos)
        self._pos += 4 * n
        return arr.copy()

    def f64_array(self, n: int) -> np.ndarray:
        arr = np.frombuffer(self._data, dtype='<f8', count=n, offset=self._pos)
        self._pos += 8 * n
        return arr.copy()


# ---------------------------------------------------------------------------
# load_bundle
# ---------------------------------------------------------------------------

def load_bundle(path: str) -> Bundle:
    """
    Read a .c2f bundle file (v1 or v2) and return a Bundle.

    Raises:
        ValueError  on bad magic or read errors.
    """
    with open(path, 'rb') as fh:
        data = fh.read()

    r = _Reader(data)

    magic = r.u32()
    if magic not in (0xC2F50001, 0xC2F50002, 0xC2F50003, 0xC2F50004, 0xC2F50005):
        raise ValueError(f'Bad magic: 0x{magic:08X}')
    v2 = (magic == 0xC2F50002)
    v3 = (magic == 0xC2F50003)
    v4 = (magic == 0xC2F50004)
    v5 = (magic == 0xC2F50005)

    NC = r.u32()
    FC = r.u32()
    NF = r.u32()
    FF = r.u32()

    # Coarse mesh
    coarseV = r.f64_array(NC * 3).reshape(NC, 3)
    coarseF = r.u32_array(FC * 3).reshape(FC, 3)

    # Fine mesh
    fineV = r.f64_array(NF * 3).reshape(NF, 3)
    fineF = r.u32_array(FF * 3).reshape(FF, 3)

    # Correspondence table → corrVec, corrBC, corrFV
    corrVec = np.empty((NC, 3), dtype=np.float64)
    corrBC  = np.empty((NC, 3), dtype=np.float64)
    corrFV  = np.empty((NC, 3), dtype=np.int32)
    for i in range(NC):
        bc = np.array([r.f64(), r.f64(), r.f64()])
        fv = np.array([r.u32(), r.u32(), r.u32()], dtype=np.int32)
        fine_pos = bc[0] * fineV[fv[0]] + bc[1] * fineV[fv[1]] + bc[2] * fineV[fv[2]]
        corrVec[i] = fine_pos - coarseV[i]
        corrBC[i]  = bc
        corrFV[i]  = fv

    b = Bundle(coarseV=coarseV, coarseF=coarseF,
               fineV=fineV, fineF=fineF,
               corrVec=corrVec, corrBC=corrBC, corrFV=corrFV)

    if not v2 and not v3 and not v4 and not v5:
        print(f'[bundle] Loaded v1  NC={NC} FC={FC} NF={NF} FF={FF}')
        return b

    # ---- v2/v3/v4/v5 SSP data ----
    nV_total = r.u32()
    nF_decIM = r.u32()
    nFO      = r.u32()

    b.nV_total = nV_total
    b.nF_total = nF_decIM

    b.vtxMap      = r.i32_array(NC)
    b.faceMap     = r.i32_array(FC)
    b.faceSheetID = r.i32_array(nFO)

    # decIM: one variable-length list per original face
    b.decIM = []
    for _ in range(nF_decIM):
        cnt = r.u32()
        b.decIM.append(r.i32_array(cnt).tolist())

    # decInfo: one CollapseData per collapse
    nDec = r.u32()
    b.decInfo = []
    for _ in range(nDec):
        nSheets = r.u32()
        cd = CollapseData()
        for _ in range(nSheets):
            sid   = r.i32()
            svSz  = r.u32()
            subsetVIdx = r.i32_array(svSz)

            uvRows  = r.u32()
            UV_pre  = r.f64_array(uvRows * 2).reshape(uvRows, 2)
            UV_post = r.f64_array(uvRows * 2).reshape(uvRows, 2)

            fuvRows  = r.u32()
            FUV_pre  = r.i32_array(fuvRows * 3).reshape(fuvRows, 3)
            FIdx_pre = r.i32_array(fuvRows)

            if v3 or v4 or v5:
                fuvPostRows = r.u32()
                FUV_post  = r.i32_array(fuvPostRows * 3).reshape(fuvPostRows, 3)
                FIdx_post = r.i32_array(fuvPostRows)
            else:
                FUV_post  = np.empty((0, 3), dtype=np.int32)
                FIdx_post = np.empty((0,),   dtype=np.int32)

            if v4 or v5:
                b_arr = r.i32_array(2)  # [survivor_local, absorbed_local]
            else:
                b_arr = np.empty((0,), dtype=np.int32)

            if v5:
                V_pre  = r.f64_array(uvRows * 3).reshape(uvRows, 3)
                V_post = r.f64_array(uvRows * 3).reshape(uvRows, 3)
            else:
                V_pre  = np.empty((0, 3), dtype=np.float64)
                V_post = np.empty((0, 3), dtype=np.float64)

            cd.sheets.append(SheetData(
                global_sheet_id=sid,
                subsetVIdx=subsetVIdx,
                UV_pre=UV_pre,
                UV_post=UV_post,
                FUV_pre=FUV_pre,
                FIdx_pre=FIdx_pre,
                FUV_post=FUV_post,
                FIdx_post=FIdx_post,
                b=b_arr,
                V_pre=V_pre,
                V_post=V_post,
            ))
        b.decInfo.append(cd)

    b.has_ssp_data = True
    ver = 5 if v5 else (4 if v4 else (3 if v3 else 2))
    print(f'[bundle] Loaded v{ver}  NC={NC} FC={FC} NF={NF} FF={FF}'
          f'  nDec={nDec}  nFO={nFO}')

    # Confirm whether sd.b (survivor/absorbed vertex pair) is present
    total_sheets = sum(len(cd.sheets) for cd in b.decInfo)
    sheets_with_b = sum(1 for cd in b.decInfo for sd in cd.sheets if len(sd.b) >= 2)
    if sheets_with_b == 0:
        print(f'[bundle] sd.b NOT present (v{ver} bundle — need v4 to get vertex fixup data)')
    else:
        print(f'[bundle] sd.b present: {sheets_with_b}/{total_sheets} sheets have survivor/absorbed pair')

    return b


# ---------------------------------------------------------------------------
# compute_barycentric_2d
# ---------------------------------------------------------------------------

def compute_barycentric_2d(p: np.ndarray,
                            UV: np.ndarray,
                            F: np.ndarray) -> np.ndarray:
    """
    Barycentric coordinates of a 2-D point `p` with respect to every face in F.

    Parameters
    ----------
    p  : (2,)         query point in UV space
    UV : (nV, 2)      UV vertices
    F  : (nF, 3)      face vertex indices (into UV)

    Returns
    -------
    B  : (nF, 3)   barycentric coordinates (u, v, w) for each face.
                   Negative components indicate the point is outside that triangle.
    """
    a = UV[F[:, 0]]   # (nF, 2)
    b = UV[F[:, 1]]
    c = UV[F[:, 2]]

    v0 = b - a        # (nF, 2)
    v1 = c - a
    v2 = p - a        # broadcast: (nF, 2)

    d00 = (v0 * v0).sum(axis=1)
    d01 = (v0 * v1).sum(axis=1)
    d11 = (v1 * v1).sum(axis=1)
    d20 = (v2 * v0).sum(axis=1)
    d21 = (v2 * v1).sum(axis=1)

    denom = d00 * d11 - d01 * d01
    vv = (d11 * d20 - d01 * d21) / denom
    ww = (d00 * d21 - d01 * d20) / denom
    uu = 1.0 - (vv + ww)   # match C++: 1.0f - (v+w) avoids (1-vv)-ww rounding at boundaries

    return np.stack([uu, vv, ww], axis=1)   # (nF, 3)


# ---------------------------------------------------------------------------
# query_coarse_to_fine
# ---------------------------------------------------------------------------

def query_coarse_to_fine(
    decInfo:     List[CollapseData],
    decIM:       List[List[int]],
    faceSheetID: np.ndarray,
    BC:          np.ndarray,
    BF:          np.ndarray,
    FIdx:        np.ndarray,
) -> None:
    """
    Walk each query point backwards through the SSP decimation sequence.

    Modifies BC, BF, FIdx in-place so that after the call:
        fine_pos = BC[q,0]*fineV[BF[q,0]] + BC[q,1]*fineV[BF[q,1]] + BC[q,2]*fineV[BF[q,2]]
    gives the fine-mesh position corresponding to the original coarse-mesh query.

    Parameters
    ----------
    decInfo     : list of CollapseData (length = number of SSP collapses)
    decIM       : list of sorted collapse-index lists, one per original face
    faceSheetID : (nFO,) int32 — sheet ID per original face
    BC          : (N, 3) float64 — barycentric weights  [modified in-place]
    BF          : (N, 3) int32   — global vertex indices [modified in-place]
    FIdx        : (N,)   int32   — global face index     [modified in-place]
    """
    N    = len(FIdx)
    nDec = len(decInfo)
    nFO  = len(faceSheetID)

    for q in range(N):
        d_idx = nDec   # start from "present" (one past the last collapse)

        while True:
            face = int(FIdx[q])

            # Find the most recent collapse older than d_idx that touched face
            d_list = decIM[face]
            found = False
            for ii in range(len(d_list) - 1, -1, -1):
                if d_idx > d_list[ii]:
                    d_idx = d_list[ii]
                    found = True
                    break
            if not found:
                break

            # Find the sheet and pre-collapse face row by searching FIdx_pre directly
            # across all sheets (more robust than faceSheetID matching for seam collapses).
            cd      = decInfo[d_idx]
            sd      = None
            pre_row = -1
            for s in cd.sheets:
                for r in range(len(s.FIdx_pre)):
                    if int(s.FIdx_pre[r]) == face:
                        sd      = s
                        pre_row = r
                        break
                if sd is not None:
                    break
            if sd is None or pre_row < 0:
                continue

            # Project BC through UV_post → UV query point
            v0, v1, v2 = sd.FUV_pre[pre_row]
            uv_query = (BC[q, 0] * sd.UV_post[v0] +
                        BC[q, 1] * sd.UV_post[v1] +
                        BC[q, 2] * sd.UV_post[v2])

            # Compute barycentric coords in UV_pre for all local faces
            B = compute_barycentric_2d(uv_query, sd.UV_pre, sd.FUV_pre)

            # Pick the face whose minimum barycentric coord is largest (most inside)
            min_per_face = B.min(axis=1)          # (fuvRows,)
            best = int(np.argmax(min_per_face))   # largest min = most inside

            # Clamp small negatives and renormalize (no centroid fallback — match C++)
            b_row = np.maximum(0.0, B[best])
            s_row = b_row.sum()
            if s_row > 1e-12:
                b_row /= s_row

            new_fidx = int(sd.FIdx_pre[best])
            new_v0   = int(sd.subsetVIdx[sd.FUV_pre[best, 0]])
            new_v1   = int(sd.subsetVIdx[sd.FUV_pre[best, 1]])
            new_v2   = int(sd.subsetVIdx[sd.FUV_pre[best, 2]])

            BC[q]   = b_row
            BF[q]   = [new_v0, new_v1, new_v2]
            FIdx[q] = new_fidx


# ---------------------------------------------------------------------------
# query_point_with_intermediates
# ---------------------------------------------------------------------------

def query_point_with_intermediates(
    decInfo, decIM, faceSheetID,
    BC_init, BF_init, FIdx_init,
    fineV,
):
    """
    Like query_coarse_to_fine for a single point, but returns a list of
    3-D positions — one per collapse step visited — so callers can display
    the actual intermediate results of the C2F query walk.

    The first entry is the initial (coarse) position; each subsequent entry
    is the position after that collapse step has been applied.

    Parameters
    ----------
    BC_init  : (3,) float64  — initial barycentric weights
    BF_init  : list of 3 ints — initial global vertex indices
    FIdx_init: int — initial global face index
    fineV    : (NF, 3) fine-mesh vertex positions

    Returns
    -------
    List[np.ndarray]  — 3-D positions at each query step
    """
    BC   = np.array(BC_init, dtype=np.float64)
    BF   = list(BF_init)
    face = int(FIdx_init)
    nDec = len(decInfo)
    nFO  = len(faceSheetID)
    d_idx = nDec

    def pos3d():
        return (BC[0] * fineV[BF[0]] +
                BC[1] * fineV[BF[1]] +
                BC[2] * fineV[BF[2]])

    positions = [pos3d()]   # position[0] = coarse start

    while True:
        d_list = decIM[face]
        found  = False
        for ii in range(len(d_list) - 1, -1, -1):
            if d_idx > d_list[ii]:
                d_idx = d_list[ii]
                found = True
                break
        if not found:
            break

        # Direct FIdx_pre search across all sheets (fixes seam-collapse sheet mismatch)
        cd      = decInfo[d_idx]
        sd      = None
        pre_row = -1
        for s in cd.sheets:
            for r in range(len(s.FIdx_pre)):
                if int(s.FIdx_pre[r]) == face:
                    sd      = s
                    pre_row = r
                    break
            if sd is not None:
                break
        if sd is None or pre_row < 0:
            continue

        v0, v1, v2 = sd.FUV_pre[pre_row]
        uv_q = (BC[0] * sd.UV_post[v0] +
                BC[1] * sd.UV_post[v1] +
                BC[2] * sd.UV_post[v2])

        B     = compute_barycentric_2d(uv_q, sd.UV_pre, sd.FUV_pre)
        best  = int(np.argmax(B.min(axis=1)))
        b_row = np.maximum(0.0, B[best])
        s_row = b_row.sum()
        if s_row > 1e-12:   # no centroid fallback — match C++
            b_row /= s_row
        BC   = b_row
        BF   = [int(sd.subsetVIdx[sd.FUV_pre[best, 0]]),
                int(sd.subsetVIdx[sd.FUV_pre[best, 1]]),
                int(sd.subsetVIdx[sd.FUV_pre[best, 2]])]
        face = int(sd.FIdx_pre[best])

        positions.append(pos3d())

    return positions


# ---------------------------------------------------------------------------
# query_vertex_f2c_intermediates
# ---------------------------------------------------------------------------

def query_vertex_f2c_intermediates(vi, bundle, verbose=False, _fi_seed_override=None, _skip_log=None):
    """
    Track fine vertex vi to its coarse position using the correct F2C shadow cast.

    Implements fine_to_coarse_procedure.md exactly:
      Step 0 : seed one-hot BC at vi's corner on fi_seed
      Step 1+ : find next collapse ci_next > ci via decIM[face]
                embed BC into UV_pre  →  cast into UV_post (using FUV_post/FIdx_post)
                update BC, BF, face
      Final   : BC · coarseV[compact(BF)]  (SSP moves vertices; must use coarseV, not fineV)

    Parameters
    ----------
    vi               : int     — fine vertex index
    bundle           : Bundle  — must have v2/v3 SSP data
    verbose          : bool    — print per-step UV diagnostics
    _fi_seed_override: (fi, col) tuple to force a specific seeding face (for diagnostics)

    Returns
    -------
    positions   : List[np.ndarray(3,)]
        positions[0] = fine start;  positions[-1] = final coarse position
    steps       : List[Tuple[int, SheetData, int]]
        One entry per successful cast: (collapse_idx, SheetData, local_v).
        len(steps) == len(positions) - 1 always.
        local_v = local index (in SheetData.subsetVIdx) of the dominant corner.
    n_skips     : int
        number of collapses skipped (no sheet match or face not in FIdx_pre)
    final_bf    : List[int]  (3 global SSP vertex indices of the landing triangle)
    final_bc    : np.ndarray(3,)
    """
    fineV       = bundle.fineV
    fineF       = bundle.fineF
    coarseV     = bundle.coarseV
    vtxMap      = bundle.vtxMap
    decIM       = bundle.decIM
    decInfo     = bundle.decInfo
    faceSheetID = bundle.faceSheetID
    nFO         = len(faceSheetID) if faceSheetID is not None else 0

    # global SSP index → compact coarse index (for coarseV lookup)
    g2c = {int(vtxMap[i]): i for i in range(len(vtxMap))} if vtxMap is not None else {}

    # Step 0: find fi_seed (any face incident to vi) and initialize one-hot BC
    if _fi_seed_override is not None:
        fi_seed, col = _fi_seed_override
    else:
        fi_seed = col = -1
        for fi in range(fineF.shape[0]):
            row = fineF[fi]
            for c in range(3):
                if int(row[c]) == vi:
                    fi_seed, col = fi, c
                    break
            if fi_seed >= 0:
                break

    if fi_seed < 0:
        return [fineV[vi].copy()], [], 0, [int(fineF[0, i]) for i in range(3)], np.array([1/3, 1/3, 1/3])

    BC      = np.zeros(3, dtype=np.float64)
    BC[col] = 1.0
    BF      = [int(fineF[fi_seed, 0]), int(fineF[fi_seed, 1]), int(fineF[fi_seed, 2])]
    face    = fi_seed
    ci      = -1
    n_skips = 0

    def _v(idx):
        ci = g2c.get(idx)
        return coarseV[ci] if ci is not None else fineV[idx]

    def pos3d():
        return BC[0]*_v(BF[0]) + BC[1]*_v(BF[1]) + BC[2]*_v(BF[2])

    positions = [fineV[vi].copy()]  # positions[0] = exact fine-mesh start (not _v, which uses coarseV for survivors)
    steps     = []                  # parallel to positions[1:]: (ci, SheetData, local_v)

    while True:
        # Step 1: first collapse index > ci that touched the current face
        ci_next = None
        for d in decIM[face]:
            if d > ci:
                ci_next = d
                break
        if ci_next is None:
            if verbose:
                print(f"  [DONE] no more collapses for face={face} after ci={ci}  "
                      f"({len(positions)-1} steps, {n_skips} skips)")
            break

        # Find the sheet containing the current face by searching FIdx_pre directly.
        # This is more robust than faceSheetID matching for seam collapses, where a
        # face's stored sheet ID may not equal the collapse sheet's global_sheet_id.
        cd      = decInfo[ci_next]
        sd      = None
        pre_row = -1
        for s in cd.sheets:
            for r in range(len(s.FIdx_pre)):
                if int(s.FIdx_pre[r]) == face:
                    sd      = s
                    pre_row = r
                    break
            if sd is not None:
                break
        if sd is None:
            if verbose:
                sid_dbg = int(faceSheetID[face]) if faceSheetID is not None else -1
                print(f"  [SKIP] ci={ci_next} face={face} not found in any sheet's FIdx_pre "
                      f"(faceSheetID={sid_dbg}  sheets={[s.global_sheet_id for s in cd.sheets]})")
            ci = ci_next
            n_skips += 1
            continue

        # Relabel absorbed vertex global_d → survivor global_s in BF after UV cast.
        def _apply_vtx_fixup(sd):
            nonlocal BF
            if len(sd.b) >= 2 and sd.b[0] >= 0 and sd.b[1] >= 0:
                global_d = int(sd.subsetVIdx[sd.b[1]])
                global_s = int(sd.subsetVIdx[sd.b[0]])
                BF = [global_s if BF[k] == global_d else BF[k] for k in range(3)]

        # UV cast — always; no SNAP special-case (see snap_vs_uv_cast.md).
        # argmax(B.min(axis=1)) picks the best containing face even when the query
        # point lands inside the deleted-triangle region of UV_post.
        # Capture dominant corner in UV_pre for canonical-view local_v.
        local_v = int(sd.FUV_pre[pre_row, int(np.argmax(BC))])
        a = int(sd.FUV_pre[pre_row, 0])
        b = int(sd.FUV_pre[pre_row, 1])
        c = int(sd.FUV_pre[pre_row, 2])
        uv_q = BC[0]*sd.UV_pre[a] + BC[1]*sd.UV_pre[b] + BC[2]*sd.UV_pre[c]

        if verbose:
            print(f"  [CAST] ci={ci_next} face={face} sid={sd.global_sheet_id} pre_row={pre_row} "
                  f"uv_q=({uv_q[0]:.5f},{uv_q[1]:.5f})")

        if len(sd.FUV_post) > 0:
            B    = compute_barycentric_2d(uv_q, sd.UV_post, sd.FUV_post)
            best = int(np.argmax(B.min(axis=1)))
            b_row = np.maximum(0.0, B[best])
            s_row = b_row.sum()
            if s_row > 1e-12:
                b_row /= s_row
            BC   = b_row
            BF   = [int(sd.subsetVIdx[sd.FUV_post[best, i]]) for i in range(3)]
            face = int(sd.FIdx_post[best])
            _apply_vtx_fixup(sd)
        else:
            B    = compute_barycentric_2d(uv_q, sd.UV_post, sd.FUV_pre)
            best = int(np.argmax(B.min(axis=1)))
            b_row = np.maximum(0.0, B[best])
            s_row = b_row.sum()
            if s_row > 1e-12:
                b_row /= s_row
            BC   = b_row
            BF   = [int(sd.subsetVIdx[sd.FUV_pre[best, i]]) for i in range(3)]
            face = int(sd.FIdx_pre[best])
            _apply_vtx_fixup(sd)

        if verbose:
            print(f"         → new face={face}  BC=({BC[0]:.4f},{BC[1]:.4f},{BC[2]:.4f})"
                  f"  BF={BF}")

        # Compute next 3D position from V_post (exact, v5+) or pos3d() (approximate fallback).
        if sd.V_post.shape[0] > 0:
            fuv = sd.FUV_post if len(sd.FUV_post) > 0 else sd.FUV_pre
            pa, pb, pc = int(fuv[best, 0]), int(fuv[best, 1]), int(fuv[best, 2])
            pos_next = BC[0]*sd.V_post[pa] + BC[1]*sd.V_post[pb] + BC[2]*sd.V_post[pc]
        else:
            pos_next = pos3d()  # approximate: uses coarseV for survivors (v2/v3/v4 bundles)

        ci = ci_next
        steps.append((ci_next, sd, local_v))
        positions.append(pos_next)

    return positions, steps, n_skips, BF, BC.copy()


# ---------------------------------------------------------------------------
# compute_f2c_correspondences  (batch F2C for every fine vertex)
# ---------------------------------------------------------------------------

def compute_f2c_correspondences(bundle):
    """
    For every fine vertex, compute its final F2C position using the correct
    shadow-cast chain (fine_to_coarse_procedure.md).

    Parameters
    ----------
    bundle : Bundle  (must have v2/v3 SSP data)

    Returns
    -------
    f2c_v        : (NF, 3) float64
    tracked_mask : (NF,) bool
    """
    if not bundle.has_ssp_data:
        raise RuntimeError('Bundle has no SSP data (need a v2 .c2f file)')

    NF    = bundle.fineV.shape[0]
    fineV = bundle.fineV

    f2c_v        = fineV.copy()
    tracked_mask = np.zeros(NF, dtype=bool)
    n_moved      = 0
    n_stationary = 0
    n_zero       = 0
    step_counts  = []
    total_skips  = 0
    skip_log     = []

    for vi in range(NF):
        positions, _steps, n_sk, _, _ = query_vertex_f2c_intermediates(vi, bundle, _skip_log=skip_log)
        n_steps = len(positions) - 1
        total_skips += n_sk
        if n_steps > 0:
            final = positions[-1]
            f2c_v[vi]        = final
            tracked_mask[vi] = True
            step_counts.append(n_steps)
            if np.allclose(final, fineV[vi], atol=1e-10):
                n_stationary += 1
            else:
                n_moved += 1
        else:
            n_zero += 1

    sc       = np.array(step_counts, dtype=np.int32) if step_counts else np.array([0])
    n_in_map = int(tracked_mask.sum())

    print(f'\n[f2c_batch] ---- F2C batch stats (correct shadow-cast) ----')
    print(f'[f2c_batch] Fine vertices total  : {NF}')
    print(f'[f2c_batch] Zero steps           : {n_zero} / {NF}  ({100*n_zero/NF:.1f}%)')
    print(f'[f2c_batch] Tracked (≥1 step)    : {n_in_map} / {NF}  ({100*n_in_map/NF:.1f}%)')
    print(f'[f2c_batch]   stationary          : {n_stationary}')
    print(f'[f2c_batch]   moved               : {n_moved}')
    print(f'[f2c_batch] Total SKIPs (all verts): {total_skips}')
    if len(sc) > 0 and sc.max() > 0:
        nz = sc[sc > 0]
        print(f'[f2c_batch] Steps/vertex: '
              f'min={nz.min()}  median={int(np.median(nz))}  max={nz.max()}  mean={nz.mean():.1f}')
    print(f'[f2c_batch] --------------------------------------------------\n')

    # Write SKIP-pre log to file
    if skip_log:
        import os
        log_path = os.path.join(os.path.dirname(__file__), 'skip_pre_log.txt')
        n_bf_changed = sum(1 for e in skip_log if e['bf_changed'])
        with open(log_path, 'w') as lf:
            lf.write(f'SKIP-pre log — {len(skip_log)} events  '
                     f'({n_bf_changed} changed BF, {len(skip_log)-n_bf_changed} no BF change)\n')
            lf.write(f'{"vi":>6}  {"ci":>6}  {"face":>6}  {"FIdx_pre":<14}  '
                     f'{"global_d":>9}  {"global_s":>9}  {"BF_before":<20}  {"BF_after":<20}  changed\n')
            lf.write('-' * 110 + '\n')
            for e in skip_log:
                changed = 'YES' if e['bf_changed'] else 'no'
                lf.write(f'{e["vi"]:>6}  {e["ci"]:>6}  {e["face"]:>6}  '
                         f'{str(e["FIdx_pre"]):<14}  '
                         f'{e["global_d"]:>9}  {e["global_s"]:>9}  '
                         f'{str(e["BF_before"]):<20}  {str(e["BF_after"]):<20}  {changed}\n')
        print(f'[f2c_batch] SKIP-pre log written → {log_path}')
        print(f'[f2c_batch]   {len(skip_log)} SKIPs: {n_bf_changed} changed BF  '
              f'{len(skip_log)-n_bf_changed} no change')

    # Verbose trace of first tracked vertex to spot any remaining cast issues
    first_tracked = next((vi for vi in range(NF) if tracked_mask[vi]), None)
    if first_tracked is not None:
        print(f'[f2c_batch] Verbose trace for vi={first_tracked}:')
        query_vertex_f2c_intermediates(first_tracked, bundle, verbose=True)

    return f2c_v, tracked_mask


# ---------------------------------------------------------------------------
# compute_f2c_correspondences_incident  (same correct walk — alias kept for API compat)
# ---------------------------------------------------------------------------

def compute_f2c_correspondences_incident(bundle):
    """
    Same as compute_f2c_correspondences.

    With the correct face-tracking shadow cast, there is no distinct
    "incident faces only" vs "full one-ring" algorithm — the walk naturally
    follows the face topology via decIM[face] at each step.

    Returns
    -------
    f2c_v        : (NF, 3) float64
    tracked_mask : (NF,) bool
    """
    return compute_f2c_correspondences(bundle)


# ---------------------------------------------------------------------------
# sample_face_correspondence
# ---------------------------------------------------------------------------

def sample_face_correspondence(
    bundle: Bundle,
    coarse_face_idx: int,
    n_samples: int = 20,
    seed: int = 42,
) -> dict:
    """
    Sample `n_samples` uniformly random barycentric points inside a coarse
    triangle, then map each one to the fine mesh via SSP correspondence.

    Parameters
    ----------
    bundle          : loaded Bundle (must have v2 SSP data)
    coarse_face_idx : index into bundle.coarseF
    n_samples       : number of samples
    seed            : RNG seed for reproducibility

    Returns
    -------
    dict with:
        'BC'         : (N, 3) final barycentric coords on fine mesh
        'BF'         : (N, 3) fine global vertex indices
        'coarse_pts' : (N, 3) 3D positions on the coarse mesh
        'fine_pts'   : (N, 3) 3D positions on the fine mesh
        'displace'   : (N, 3) fine_pts - coarse_pts
    """
    if not bundle.has_ssp_data:
        raise RuntimeError('Bundle has no SSP data (need a v2 .c2f file)')

    cf = int(coarse_face_idx)
    ci0, ci1, ci2 = bundle.coarseF[cf]
    cv0 = bundle.coarseV[ci0]
    cv1 = bundle.coarseV[ci1]
    cv2 = bundle.coarseV[ci2]

    # Global SSP vertex indices for the coarse face corners
    gvi0 = int(bundle.vtxMap[ci0])
    gvi1 = int(bundle.vtxMap[ci1])
    gvi2 = int(bundle.vtxMap[ci2])

    # Global SSP face index
    gfi = int(bundle.faceMap[cf])

    # Fold-sampled uniform barycentric coordinates (square → triangle)
    rng = np.random.default_rng(seed)
    r   = rng.random((n_samples, 2))
    mask = r[:, 0] + r[:, 1] > 1.0
    r[mask] = 1.0 - r[mask]    # fold excess back inside

    BC = np.empty((n_samples, 3), dtype=np.float64)
    BC[:, 0] = r[:, 0]
    BC[:, 1] = r[:, 1]
    BC[:, 2] = 1.0 - r[:, 0] - r[:, 1]

    # Coarse 3D positions
    coarse_pts = (BC[:, 0:1] * cv0 +
                  BC[:, 1:2] * cv1 +
                  BC[:, 2:3] * cv2)

    # Initial BF and FIdx: all samples start on the same coarse face
    BF   = np.tile([gvi0, gvi1, gvi2], (n_samples, 1)).astype(np.int32)
    FIdx = np.full(n_samples, gfi, dtype=np.int32)

    # Walk backwards through SSP collapses
    query_coarse_to_fine(
        bundle.decInfo,
        bundle.decIM,
        bundle.faceSheetID,
        BC, BF, FIdx,
    )

    # Evaluate fine 3D positions from walked BC/BF
    fine_pts = (BC[:, 0:1] * bundle.fineV[BF[:, 0]] +
                BC[:, 1:2] * bundle.fineV[BF[:, 1]] +
                BC[:, 2:3] * bundle.fineV[BF[:, 2]])

    return {
        'BC':         BC,
        'BF':         BF,
        'coarse_pts': coarse_pts,
        'fine_pts':   fine_pts,
        'displace':   fine_pts - coarse_pts,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='C2F bundle loader and query demo')
    p.add_argument('bundle', help='Path to .c2f file')
    p.add_argument('--face', type=int, default=0,
                   help='Coarse face index to sample (default: 0)')
    p.add_argument('--n', type=int, default=20,
                   help='Number of samples per face (default: 20)')
    return p.parse_args()


def main():
    args = _parse_args()

    b = load_bundle(args.bundle)

    print(f'Coarse mesh:  {b.coarseV.shape[0]} vertices  {b.coarseF.shape[0]} faces')
    print(f'Fine   mesh:  {b.fineV.shape[0]} vertices  {b.fineF.shape[0]} faces')

    if not b.has_ssp_data:
        print('[v1 bundle] No SSP data — showing vertex correspondence only.')
        print(f'corrVec[0] = {b.corrVec[0]}')
        return

    print(f'SSP collapses: {len(b.decInfo)}')

    result = sample_face_correspondence(b, args.face, args.n)
    print(f'\nFace {args.face}  —  {args.n} samples:')
    print(f'  coarse_pts[0] = {result["coarse_pts"][0]}')
    print(f'  fine_pts[0]   = {result["fine_pts"][0]}')
    print(f'  displace[0]   = {result["displace"][0]}')
    print(f'  ||displace||  = {np.linalg.norm(result["displace"], axis=1).mean():.6f}  (mean over {args.n} samples)')


if __name__ == '__main__':
    main()
