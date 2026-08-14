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
    FUV_pre: np.ndarray      # (fuvRows, 3) int32 — local vertex indices
    FIdx_pre: np.ndarray     # (fuvRows,)   int32 — global original face indices


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
    if magic not in (0xC2F50001, 0xC2F50002):
        raise ValueError(f'Bad magic: 0x{magic:08X}')
    v2 = (magic == 0xC2F50002)

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

    # Correspondence table → corrVec
    corrVec = np.empty((NC, 3), dtype=np.float64)
    for i in range(NC):
        bc = np.array([r.f64(), r.f64(), r.f64()])
        fv = np.array([r.u32(), r.u32(), r.u32()], dtype=np.int32)
        fine_pos = bc[0] * fineV[fv[0]] + bc[1] * fineV[fv[1]] + bc[2] * fineV[fv[2]]
        corrVec[i] = fine_pos - coarseV[i]

    b = Bundle(coarseV=coarseV, coarseF=coarseF,
               fineV=fineV, fineF=fineF,
               corrVec=corrVec)

    if not v2:
        print(f'[bundle] Loaded v1  NC={NC} FC={FC} NF={NF} FF={FF}')
        return b

    # ---- v2 SSP data ----
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

            cd.sheets.append(SheetData(
                global_sheet_id=sid,
                subsetVIdx=subsetVIdx,
                UV_pre=UV_pre,
                UV_post=UV_post,
                FUV_pre=FUV_pre,
                FIdx_pre=FIdx_pre,
            ))
        b.decInfo.append(cd)

    b.has_ssp_data = True
    print(f'[bundle] Loaded v2  NC={NC} FC={FC} NF={NF} FF={FF}'
          f'  nDec={nDec}  nFO={nFO}')
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
    uu = 1.0 - vv - ww

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

            # Determine the sheet this face belongs to
            sid = int(faceSheetID[face]) if face < nFO else 0

            # Find the matching SheetData for this face's sheet.
            cd  = decInfo[d_idx]
            sd  = None
            for s in cd.sheets:
                if s.global_sheet_id == sid:
                    sd = s
                    break
            if sd is None and cd.sheets:
                if len(cd.sheets) == 1:
                    # Manifold collapse: single sheet, [0] is always correct.
                    sd = cd.sheets[0]
                # else: seam collapse — no matching sheet found.
                # Using a different sheet's UV would corrupt the sample; skip.
            if sd is None:
                continue

            # Find the pre-collapse face row matching the current face
            pre_row = -1
            for r in range(len(sd.FIdx_pre)):
                if sd.FIdx_pre[r] == face:
                    pre_row = r
                    break
            if pre_row < 0:
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

            # Clamp small negatives and renormalize
            b_row = np.maximum(0.0, B[best])
            s_row = b_row.sum()
            if s_row > 0:
                b_row /= s_row
            else:
                b_row = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

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

        sid = int(faceSheetID[face]) if face < nFO else 0
        cd  = decInfo[d_idx]
        sd  = None
        for s in cd.sheets:
            if s.global_sheet_id == sid:
                sd = s
                break
        if sd is None and cd.sheets:
            if len(cd.sheets) == 1:
                sd = cd.sheets[0]
        if sd is None:
            continue

        pre_row = -1
        for r in range(len(sd.FIdx_pre)):
            if sd.FIdx_pre[r] == face:
                pre_row = r
                break
        if pre_row < 0:
            continue

        v0, v1, v2 = sd.FUV_pre[pre_row]
        uv_q = (BC[0] * sd.UV_post[v0] +
                BC[1] * sd.UV_post[v1] +
                BC[2] * sd.UV_post[v2])

        B            = compute_barycentric_2d(uv_q, sd.UV_pre, sd.FUV_pre)
        best         = int(np.argmax(B.min(axis=1)))
        b_row        = np.maximum(0.0, B[best])
        s_row        = b_row.sum()
        BC           = b_row / s_row if s_row > 0 else np.full(3, 1.0 / 3)
        BF           = [int(sd.subsetVIdx[sd.FUV_pre[best, 0]]),
                        int(sd.subsetVIdx[sd.FUV_pre[best, 1]]),
                        int(sd.subsetVIdx[sd.FUV_pre[best, 2]])]
        face         = int(sd.FIdx_pre[best])

        positions.append(pos3d())

    return positions


# ---------------------------------------------------------------------------
# query_vertex_f2c_intermediates
# ---------------------------------------------------------------------------

def query_vertex_f2c_intermediates(
    collapse_steps,
    fineV,
    start_pos,
):
    """
    Track a fine vertex forward through F2C collapse steps and return the 3D
    position at each stage (fine → coarse direction).

    Parameters
    ----------
    collapse_steps : list of (collapse_idx, SheetData, local_v_idx)
                     sorted ascending by collapse_idx (as produced by
                     _find_vtx_collapse_steps in the viz script)
    fineV          : (NF, 3) fine mesh vertex positions
    start_pos      : (3,) starting 3D position (the selected fine vertex)

    Returns
    -------
    List[np.ndarray(3,)]
        positions[0]   = start_pos (fine mesh vertex)
        positions[i+1] = 3D position after the i-th collapse step
    """
    positions = [np.array(start_pos, dtype=np.float64)]

    for _ci, sheet, local_v in collapse_steps:
        if local_v >= len(sheet.UV_post):
            continue   # vertex has no UV slot in this sheet (boundary/seam case)
        # UV position of this vertex after the collapse
        uv_post = sheet.UV_post[local_v]

        # Express uv_post in barycentric coords within the UV_pre triangulation
        B    = compute_barycentric_2d(uv_post, sheet.UV_pre, sheet.FUV_pre)
        best = int(np.argmax(B.min(axis=1)))
        b_row = np.maximum(0.0, B[best])
        s_row = b_row.sum()
        BC    = b_row / s_row if s_row > 0 else np.full(3, 1.0 / 3)

        p0 = int(sheet.subsetVIdx[sheet.FUV_pre[best, 0]])
        p1 = int(sheet.subsetVIdx[sheet.FUV_pre[best, 1]])
        p2 = int(sheet.subsetVIdx[sheet.FUV_pre[best, 2]])
        pos = BC[0] * fineV[p0] + BC[1] * fineV[p1] + BC[2] * fineV[p2]
        positions.append(pos)

    return positions


# ---------------------------------------------------------------------------
# compute_f2c_correspondences  (batch F2C for every fine vertex)
# ---------------------------------------------------------------------------

def compute_f2c_correspondences(bundle):
    """
    For every fine vertex that appears in at least one collapse sheet, compute
    its final F2C position (where it ends up after all collapse steps that
    touch it, i.e. its coarse-mesh equivalent in fine-mesh space).

    Parameters
    ----------
    bundle : Bundle  (must have v2 SSP data)

    Returns
    -------
    f2c_v        : (NF, 3) float64
        Copy of fineV with tracked vertices moved to their F2C final position.
        Untracked vertices keep their original fineV position.
    tracked_mask : (NF,) bool
        True for vertices that went through at least one collapse step.
    """
    if not bundle.has_ssp_data:
        raise RuntimeError('Bundle has no SSP data (need a v2 .c2f file)')

    NF    = bundle.fineV.shape[0]
    fineV = bundle.fineV

    # Build map: global fine vertex index -> [(ci, SheetData, local_v_idx), ...]
    # Guard: local_v must be < uvRows (len(UV_post)) — subsetVIdx can be larger
    # than the UV arrays for boundary/seam vertices that have no UV slot here.
    # Dedup by (ci, v): seam collapses have multiple sheets; we use the first
    # sheet where the vertex has a valid UV position.
    vtx_steps = {}
    recorded  = set()   # (ci, v) pairs already committed
    for ci, cd in enumerate(bundle.decInfo):
        for sheet in cd.sheets:
            uvRows = len(sheet.UV_post)
            for local_v, gv in enumerate(sheet.subsetVIdx):
                if local_v >= uvRows:
                    continue   # no UV slot for this vertex in this sheet
                v = int(gv)
                if not (0 <= v < NF):
                    continue
                key = (ci, v)
                if key in recorded:
                    continue   # already have a sheet for this (collapse, vertex)
                recorded.add(key)
                if v not in vtx_steps:
                    vtx_steps[v] = []
                vtx_steps[v].append((ci, sheet, local_v))

    # Sort each vertex's steps ascending (fine→coarse)
    for v in vtx_steps:
        vtx_steps[v].sort(key=lambda x: x[0])

    # --- Diagnostic: seam duplicate pattern and multi-sheet cases ---
    # "local_v >= uvRows" entries: the user believes these mark seam vertices that
    # need merging after the collapse — the same vertex should also appear at
    # local_v < uvRows in the same sheet (the canonical copy).  Log both cases.
    n_over_uvrows      = 0   # (ci, v) skipped because local_v >= uvRows
    n_has_canonical    = 0   # of those: also appear at local_v < uvRows in same sheet
    n_multi_valid_sheet = 0  # (ci, v) with valid UV slot in >1 sheet of same collapse

    for ci, cd in enumerate(bundle.decInfo):
        # Multi-sheet check per collapse
        valid_sheet_count = {}  # v -> number of sheets with local_v < uvRows
        for sheet in cd.sheets:
            uvRows = len(sheet.UV_post)
            for local_v, gv in enumerate(sheet.subsetVIdx):
                if local_v >= uvRows:
                    continue
                v = int(gv)
                if 0 <= v < NF:
                    valid_sheet_count[v] = valid_sheet_count.get(v, 0) + 1
        n_multi_valid_sheet += sum(1 for cnt in valid_sheet_count.values() if cnt > 1)

        # Seam duplicate check per sheet
        for sheet in cd.sheets:
            uvRows = len(sheet.UV_post)
            svSz   = len(sheet.subsetVIdx)
            if svSz <= uvRows:
                continue
            # Vertices with a canonical UV slot in this sheet
            canonical_set = {int(sheet.subsetVIdx[lv]) for lv in range(uvRows)
                             if 0 <= int(sheet.subsetVIdx[lv]) < NF}
            for local_v in range(uvRows, svSz):
                gv = int(sheet.subsetVIdx[local_v])
                if not (0 <= gv < NF):
                    continue
                n_over_uvrows += 1
                if gv in canonical_set:
                    n_has_canonical += 1

    pct_can = (100 * n_has_canonical / n_over_uvrows) if n_over_uvrows else 0.0
    print(f'[f2c_seam] Entries at local_v >= uvRows (skipped in map build) : {n_over_uvrows}')
    print(f'[f2c_seam]   of those with canonical (local_v < uvRows) copy   : '
          f'{n_has_canonical}  ({pct_can:.1f}%)')
    print(f'[f2c_seam] (ci, v) with valid UV slot in >1 sheet of same coll : {n_multi_valid_sheet}')

    f2c_v        = fineV.copy()
    tracked_mask = np.zeros(NF, dtype=bool)

    # per-vertex stats
    n_zero_steps      = NF - len(vtx_steps)   # never appeared in any valid UV slot
    n_all_skipped     = 0   # had steps in map but query_vertex_f2c_intermediates skipped all
    n_moved           = 0   # actually displaced (positions[-1] != fineV[v])
    n_stationary      = 0   # had steps but ended exactly at start (degenerate UV)
    step_counts       = []  # valid step counts for vertices that had any

    for v, steps in vtx_steps.items():
        positions = query_vertex_f2c_intermediates(steps, fineV, fineV[v])
        step_counts.append(len(positions) - 1)   # excludes start
        final = positions[-1]
        f2c_v[v]        = final
        tracked_mask[v] = True

        if len(positions) == 1:
            # every step was skipped inside query_vertex_f2c_intermediates
            n_all_skipped += 1
        elif np.allclose(final, fineV[v], atol=1e-10):
            n_stationary += 1
        else:
            n_moved += 1

    n_in_map   = len(vtx_steps)
    n_tracked  = int(tracked_mask.sum())
    sc         = np.array(step_counts, dtype=np.int32) if step_counts else np.array([0])

    print(f'\n[f2c_batch] ---- F2C batch stats ----')
    print(f'[f2c_batch] Fine vertices total          : {NF}')
    print(f'[f2c_batch] Zero steps (never in map)   : {n_zero_steps} / {NF}  ({100*n_zero_steps/NF:.1f}%)')
    print(f'[f2c_batch] In map (≥1 valid UV step)   : {n_in_map} / {NF}  ({100*n_in_map/NF:.1f}%)')
    print(f'[f2c_batch]   of those — all skipped    : {n_all_skipped}  (local_v >= uvRows for every step)')
    print(f'[f2c_batch]   of those — stationary     : {n_stationary}  (steps ran but final == start)')
    print(f'[f2c_batch]   of those — moved          : {n_moved}')
    if len(sc) > 0 and sc.max() > 0:
        nz = sc[sc > 0]
        print(f'[f2c_batch] Steps per vertex (moved): '
              f'min={nz.min()}  median={int(np.median(nz))}  max={nz.max()}  mean={nz.mean():.1f}')
    print(f'[f2c_batch] ---------------------------\n')

    # expose vtx_steps on the bundle so callers reuse the same sheet selection
    bundle._f2c_vtx_steps = vtx_steps

    return f2c_v, tracked_mask


# ---------------------------------------------------------------------------
# compute_f2c_correspondences_incident  (batch F2C — incident faces method)
# ---------------------------------------------------------------------------

def compute_f2c_correspondences_incident(bundle):
    """
    Same interface as compute_f2c_correspondences, but finds collapse steps for
    each fine vertex using only its directly incident faces (via decIM), NOT the
    full one-ring scan.  This mirrors what _find_vtx_collapse_steps does interactively.

    Stores the per-vertex step lists on bundle._f2c_incident_vtx_steps so the
    interactive tracker can reuse them when "incident faces only" mode is active.

    Returns
    -------
    f2c_v        : (NF, 3) float64
    tracked_mask : (NF,) bool
    """
    if not bundle.has_ssp_data:
        raise RuntimeError('Bundle has no SSP data (need a v2 .c2f file)')

    NF    = bundle.fineV.shape[0]
    fineV = bundle.fineV
    fineF = bundle.fineF
    decIM   = bundle.decIM
    decInfo = bundle.decInfo
    nFO     = len(decIM)

    # Build vertex→incident-faces adjacency once (O(NF) instead of O(NV*NF))
    v2f = [[] for _ in range(NF)]
    for fi in range(fineF.shape[0]):
        for v in fineF[fi]:
            v2f[v].append(fi)

    vtx_steps = {}

    for v in range(NF):
        seen = set()
        for fi in v2f[v]:
            if fi < nFO:
                for ci in decIM[fi]:
                    seen.add(ci)
        if not seen:
            continue

        steps = []
        for ci in sorted(seen):
            cd = decInfo[ci]
            v_arr = np.int32(v)
            # First sheet with a valid UV slot for this vertex
            valid = None
            for sheet in cd.sheets:
                uvRows = len(sheet.UV_post)
                hits   = np.where(sheet.subsetVIdx == v_arr)[0]
                if len(hits) and int(hits[0]) < uvRows:
                    valid = (ci, sheet, int(hits[0]))
                    break
            if valid is None:
                # fallback: any sheet that contains the vertex (will be skipped in query)
                for sheet in cd.sheets:
                    hits = np.where(sheet.subsetVIdx == v_arr)[0]
                    if len(hits):
                        valid = (ci, sheet, int(hits[0]))
                        break
            if valid is not None:
                steps.append(valid)

        if steps:
            vtx_steps[v] = steps

    f2c_v        = fineV.copy()
    tracked_mask = np.zeros(NF, dtype=bool)

    n_zero_steps  = NF - len(vtx_steps)
    n_all_skipped = 0
    n_moved       = 0
    n_stationary  = 0
    step_counts   = []

    for v, steps in vtx_steps.items():
        positions = query_vertex_f2c_intermediates(steps, fineV, fineV[v])
        step_counts.append(len(positions) - 1)
        final = positions[-1]
        f2c_v[v]        = final
        tracked_mask[v] = True

        if len(positions) == 1:
            n_all_skipped += 1
        elif np.allclose(final, fineV[v], atol=1e-10):
            n_stationary += 1
        else:
            n_moved += 1

    n_in_map = len(vtx_steps)
    sc = np.array(step_counts, dtype=np.int32) if step_counts else np.array([0])

    print(f'\n[f2c_incident] ---- F2C incident-faces batch stats ----')
    print(f'[f2c_incident] Fine vertices total          : {NF}')
    print(f'[f2c_incident] Zero steps (no incident col) : {n_zero_steps} / {NF}  ({100*n_zero_steps/NF:.1f}%)')
    print(f'[f2c_incident] In map (≥1 step found)       : {n_in_map} / {NF}  ({100*n_in_map/NF:.1f}%)')
    print(f'[f2c_incident]   all skipped (no valid UV)  : {n_all_skipped}')
    print(f'[f2c_incident]   stationary (moved=0)       : {n_stationary}')
    print(f'[f2c_incident]   moved                      : {n_moved}')
    if len(sc) > 0 and sc.max() > 0:
        nz = sc[sc > 0]
        print(f'[f2c_incident] Steps/vertex (moved): '
              f'min={nz.min()}  median={int(np.median(nz))}  max={nz.max()}  mean={nz.mean():.1f}')
    print(f'[f2c_incident] ------------------------------------------\n')

    bundle._f2c_incident_vtx_steps = vtx_steps
    return f2c_v, tracked_mask


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
