#!/usr/bin/env python3
"""
fine_2_coarse_samples_viz_with_query copy.py
  Two modes for visualising fine → coarse correspondence:

  Mode 0 — Calculated (F2C query)
    Sample random interior points on fine mesh faces and map each to the
    coarse mesh via the SSP F2C correspondence (c2f_query.py).
    Per-vertex F2C map is computed once; each sample is interpolated from
    the surrounding fine-face vertices.

  Mode 1 — SSP Sample Tracker
    Read pre-saved samples_fine_*.txt / samples_coarse_*.txt produced by
    sample_tracker_save() in face_sample_tracker.cpp.
    Fine position  = BC on fine face (from the fine file).
    Coarse position = BC with global coarse vertex indices (from the coarse file).

Two sampling sub-modes for Mode 0 (UI checkbox):
  "Per face"     — exactly N samples on every fine face
  "Total (area)" — N total samples distributed by face area

Visualization:
  fine_mesh           grey at Z=z_offset      (source / lifted)
  coarse_mesh_src     green at Z=0            (target background)
  fine_samples        blue dots at Z=z_offset (sampled fine positions)
  coarse_landings     orange dots at Z=0      (F2C landed positions)
  sample_arrows       yellow ambient arrows   (fine sample → coarse landing)

Click a blue sample dot to inspect its details.
Click a fine mesh face in Pick-face mode to filter samples (Mode 0 only).

Usage:
    python "fine_2_coarse_samples_viz_with_query copy.py"

Dependencies:
    pip install polyscope numpy
    (c2f_query.py must be in the same directory or on PYTHONPATH)
"""

import sys
import os
import glob
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from c2f_query import load_bundle, compute_f2c_correspondences, compute_barycentric_2d

# ---------------------------------------------------------------------------
# Structure names
# ---------------------------------------------------------------------------

FINE_MESH         = "fine_mesh"
COARSE_MESH_VIZ   = "coarse_mesh_src"
FINE_SAMPLE_PC    = "fine_samples"
COARSE_LANDING_PC = "coarse_landings"
ARROWS_PC         = "sample_arrows"
SEL_FINE_PC       = "sel_fine_dot"
SEL_COARSE_PC     = "sel_coarse_dot"

# ---------------------------------------------------------------------------
# Mutable state — shared
# ---------------------------------------------------------------------------

_bundle     = None
_bundle_dir = ""
_z_offset   = 1.0
_mesh_span  = 1.0

# viz mode: 0 = calculated F2C query, 1 = SSP sample tracker files
_viz_mode = 0

# ---------------------------------------------------------------------------
# Mutable state — Mode 0 (Calculated)
# ---------------------------------------------------------------------------

_f2c_v        = None   # (NF, 3)  each fine vertex → coarse-side landing position
_tracked_mask = None   # (NF,)  bool

_n_samples      = 1000
_seed           = 42
_per_face_mode  = False

# ---------------------------------------------------------------------------
# Mutable state — Mode 1 (Tracker files)
# ---------------------------------------------------------------------------

# Paths to the tracker files (auto-detected from bundle directory)
_tracker_fine_path   = ""
_tracker_coarse_path = ""

# tracker-mode sample metadata (parallel with _fine_sample_pts/_coarse_land_pts)
_tracker_sample_ids   = None   # (N,) int
_tracker_is_vertex    = None   # (N,) bool
_tracker_fine_face_id = None   # (N,) int
_tracker_fine_bc      = None   # (N, 3)
_tracker_coarse_face_id = None  # (N,) int
_tracker_coarse_bc    = None   # (N, 3)
_tracker_coarse_bv    = None   # (N, 3) int — global coarse vertex indices

# ---------------------------------------------------------------------------
# Mutable state — per-sample data (both modes)
# ---------------------------------------------------------------------------

_sample_face_ids  = None   # (N,)   fine face index
_sample_bcs       = None   # (N, 3) barycentric coords on fine face
_fine_sample_pts  = None   # (N, 3) 3-D position on fine mesh at Z=0
_coarse_land_pts  = None   # (N, 3) F2C landed position at Z=0
_sample_tracked   = None   # (N,)   bool

# display
_show_arrows      = True
_show_fine_dots   = True
_show_coarse_dots = True
_selected_sample  = -1

# fine face filter (Mode 0 only)
_pick_face_mode  = False
_selected_face   = -1

# coarse face filter (all modes — available when _coarse_face_ids is not None)
_coarse_selected_face = -1   # compact coarse face index, -1 = no filter
_coarse_face_ids      = None  # (N,) compact coarse face index per sample; None if not available

# per-mode pre-computed state cache
_mode_cache = {}  # mode_int -> dict of snapshot globals

# global SSP face index → compact coarse face index
_inv_faceMap = None  # built at startup from bundle.faceMap


# ---------------------------------------------------------------------------
# Sampling helpers (Mode 0)
# ---------------------------------------------------------------------------

def _uniform_random_bc(n: int, rng: np.random.Generator) -> np.ndarray:
    r1 = rng.random(n)
    r2 = rng.random(n)
    sr = np.sqrt(r1)
    return np.stack([1.0 - sr, sr * (1.0 - r2), sr * r2], axis=1).astype(np.float64)


def _face_areas(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _area_weighted_face_ids(areas, N, rng):
    total = areas.sum()
    if total < 1e-30:
        areas = np.ones(len(areas))
        total = float(len(areas))
    exact     = areas * (N / total)
    counts    = np.floor(exact).astype(np.int64)
    remainder = exact - counts
    leftover  = N - int(counts.sum())
    counts[np.argsort(-remainder)[:leftover]] += 1
    return np.repeat(np.arange(len(areas), dtype=np.int32), counts)


# ---------------------------------------------------------------------------
# Per-sample forward UV-cast F2C walk  (used by Mode 3)
# ---------------------------------------------------------------------------

def _query_f2c_batch(bundle, face_ids: np.ndarray, bcs: np.ndarray):
    """
    Forward fine→coarse UV-cast for each sample independently.

    Generalises query_vertex_f2c_intermediates to arbitrary BC (not just
    one-hot vertex BCs): seed each sample at its fine face with its BC,
    walk forward through every SSP collapse that touches the face, and
    UV-cast through UV_pre → UV_post at each step.

    Parameters
    ----------
    bundle   : Bundle (must have SSP data)
    face_ids : (N,) int  — fine face index per sample
    bcs      : (N, 3)    — barycentric coords on that face

    Returns
    -------
    coarse_pts     : (N, 3) float64
    tracked        : (N,)   bool  — True if the sample moved through ≥1 collapse
    final_face_ids : (N,)   int   — SSP face index where the sample landed
    """
    if not bundle.has_ssp_data:
        raise RuntimeError("[mode3] Bundle has no SSP data")

    decInfo  = bundle.decInfo
    decIM    = bundle.decIM
    fineV    = bundle.fineV
    fineF    = bundle.fineF
    coarseV  = bundle.coarseV
    vtxMap   = bundle.vtxMap

    g2c = ({int(vtxMap[i]): i for i in range(len(vtxMap))}
           if vtxMap is not None else {})

    def _vpos(gid):
        ci = g2c.get(int(gid))
        return coarseV[ci] if ci is not None else fineV[int(gid)]

    N = len(face_ids)
    coarse_pts     = np.empty((N, 3), dtype=np.float64)
    tracked        = np.zeros(N, dtype=bool)
    final_face_ids = face_ids.copy().astype(np.int32)

    for q in range(N):
        face = int(face_ids[q])
        BC   = bcs[q].copy()
        BF   = [int(fineF[face, 0]), int(fineF[face, 1]), int(fineF[face, 2])]
        ci   = -1

        while True:
            # next collapse > ci that touched current face
            ci_next = None
            for d in decIM[face]:
                if d > ci:
                    ci_next = d
                    break
            if ci_next is None:
                break

            # find sheet containing this face in FIdx_pre
            cd = decInfo[ci_next]
            sd = None
            pre_row = -1
            for s in cd.sheets:
                for r in range(len(s.FIdx_pre)):
                    if int(s.FIdx_pre[r]) == face:
                        sd = s
                        pre_row = r
                        break
                if sd is not None:
                    break

            if sd is None:
                ci = ci_next
                continue

            # UV cast: embed BC into UV_pre → find landing in UV_post
            lv0, lv1, lv2 = sd.FUV_pre[pre_row]
            uv_q = (BC[0] * sd.UV_pre[lv0]
                  + BC[1] * sd.UV_pre[lv1]
                  + BC[2] * sd.UV_pre[lv2])

            if len(sd.FUV_post) > 0:
                B    = compute_barycentric_2d(uv_q, sd.UV_post, sd.FUV_post)
                best = int(np.argmax(B.min(axis=1)))
                b_row = np.maximum(0.0, B[best])
                s_b   = b_row.sum()
                if s_b > 1e-12:
                    b_row /= s_b
                BC   = b_row
                BF   = [int(sd.subsetVIdx[sd.FUV_post[best, i]]) for i in range(3)]
                face = int(sd.FIdx_post[best])
            else:
                B    = compute_barycentric_2d(uv_q, sd.UV_post, sd.FUV_pre)
                best = int(np.argmax(B.min(axis=1)))
                b_row = np.maximum(0.0, B[best])
                s_b   = b_row.sum()
                if s_b > 1e-12:
                    b_row /= s_b
                BC   = b_row
                BF   = [int(sd.subsetVIdx[sd.FUV_pre[best, i]]) for i in range(3)]
                face = int(sd.FIdx_pre[best])

            # absorbed→survivor vertex fixup (same as query_vertex_f2c_intermediates)
            if len(sd.b) >= 2 and sd.b[0] >= 0 and sd.b[1] >= 0:
                gd = int(sd.subsetVIdx[sd.b[1]])
                gs = int(sd.subsetVIdx[sd.b[0]])
                BF = [gs if BF[k] == gd else BF[k] for k in range(3)]

            ci = ci_next
            tracked[q] = True
            final_face_ids[q] = face

        coarse_pts[q] = (BC[0] * _vpos(BF[0])
                       + BC[1] * _vpos(BF[1])
                       + BC[2] * _vpos(BF[2]))

    return coarse_pts, tracked, final_face_ids


# ---------------------------------------------------------------------------
# Compute samples — Mode 0
# ---------------------------------------------------------------------------

def _compute_f2c_samples():
    global _sample_face_ids, _sample_bcs
    global _fine_sample_pts, _coarse_land_pts, _sample_tracked, _coarse_face_ids

    b = _bundle
    if b is None or _f2c_v is None:
        return False

    FF  = b.fineF.shape[0]
    rng = np.random.default_rng(_seed)

    if _per_face_mode:
        N        = _n_samples * FF
        face_ids = np.repeat(np.arange(FF, dtype=np.int32), _n_samples)
        print(f"[f2c_samples] {_n_samples} samples/face × {FF} faces = {N} total  seed={_seed}")
    else:
        N        = _n_samples
        areas    = _face_areas(b.fineV, b.fineF)
        face_ids = _area_weighted_face_ids(areas, N, rng)
        print(f"[f2c_samples] {N} total samples, area-weighted across {FF} faces  seed={_seed}")

    bc = _uniform_random_bc(N, rng)

    f0 = b.fineV[b.fineF[face_ids, 0]]
    f1 = b.fineV[b.fineF[face_ids, 1]]
    f2 = b.fineV[b.fineF[face_ids, 2]]
    fine_pts = bc[:, 0:1] * f0 + bc[:, 1:2] * f1 + bc[:, 2:3] * f2

    c0 = _f2c_v[b.fineF[face_ids, 0]]
    c1 = _f2c_v[b.fineF[face_ids, 1]]
    c2 = _f2c_v[b.fineF[face_ids, 2]]
    coarse_pts = bc[:, 0:1] * c0 + bc[:, 1:2] * c1 + bc[:, 2:3] * c2

    t0 = _tracked_mask[b.fineF[face_ids, 0]]
    t1 = _tracked_mask[b.fineF[face_ids, 1]]
    t2 = _tracked_mask[b.fineF[face_ids, 2]]
    tracked = t0 & t1 & t2

    _sample_face_ids  = face_ids
    _sample_bcs       = bc
    _fine_sample_pts  = fine_pts
    _coarse_land_pts  = coarse_pts
    _sample_tracked   = tracked
    _coarse_face_ids  = None  # no per-sample coarse face for Mode 0

    n_tracked = int(tracked.sum())
    print(f"[f2c_samples] {n_tracked}/{N} samples on fully-tracked faces  "
          f"({100*n_tracked/N:.1f}%)")
    return True


# ---------------------------------------------------------------------------
# Mode 2: tracker fine samples → F2C query
# ---------------------------------------------------------------------------

def _compute_tracker_fine_f2c() -> bool:
    """
    Read sample positions from samples_fine_*.txt (same as Mode 1) but
    compute the coarse landing via the pre-built per-vertex F2C map (_f2c_v),
    exactly like Mode 0 does for random samples.

    This lets you compare the tracker's coarse output (Mode 1) against the
    F2C query result (Mode 2) on the *identical* set of fine-mesh samples.
    """
    global _sample_face_ids, _sample_bcs
    global _fine_sample_pts, _coarse_land_pts, _sample_tracked, _coarse_face_ids
    global _tracker_sample_ids, _tracker_is_vertex, _tracker_fine_face_id, _tracker_fine_bc

    b = _bundle
    if b is None or _f2c_v is None:
        print("[mode2] ERROR: bundle or f2c map not ready")
        return False
    if not _tracker_fine_path:
        print("[mode2] ERROR: no tracker fine file path set")
        return False
    if not os.path.isfile(_tracker_fine_path):
        print(f"[mode2] ERROR: fine file not found: {_tracker_fine_path}")
        return False

    def _read_txt(path):
        with open(path, "r") as fh:
            lines = fh.readlines()
        data_lines = [l for l in lines if not l.strip().startswith("#")]
        count = int(data_lines[0].strip())
        return [l.split() for l in data_lines[1:count + 1]]

    fine_rows = _read_txt(_tracker_fine_path)
    N = len(fine_rows)
    print(f"[mode2] {N} samples from {os.path.basename(_tracker_fine_path)} — mapping via F2C query")

    sample_ids    = np.zeros(N, dtype=np.int32)
    is_vertex     = np.zeros(N, dtype=bool)
    fine_face_ids = np.zeros(N, dtype=np.int32)
    fine_bc       = np.zeros((N, 3), dtype=np.float64)

    for i, fr in enumerate(fine_rows):
        sample_ids[i]    = int(fr[0])
        is_vertex[i]     = bool(int(fr[1]))
        fine_face_ids[i] = int(fr[2])
        fine_bc[i]       = [float(fr[3]), float(fr[4]), float(fr[5])]

    ff = fine_face_ids
    fv = b.fineV
    fi = b.fineF
    fine_pts = (fine_bc[:, 0:1] * fv[fi[ff, 0]]
              + fine_bc[:, 1:2] * fv[fi[ff, 1]]
              + fine_bc[:, 2:3] * fv[fi[ff, 2]])

    # Coarse positions via interpolated per-vertex F2C map (same as Mode 0)
    c0 = _f2c_v[fi[ff, 0]]
    c1 = _f2c_v[fi[ff, 1]]
    c2 = _f2c_v[fi[ff, 2]]
    coarse_pts = (fine_bc[:, 0:1] * c0
                + fine_bc[:, 1:2] * c1
                + fine_bc[:, 2:3] * c2)

    t0 = _tracked_mask[fi[ff, 0]]
    t1 = _tracked_mask[fi[ff, 1]]
    t2 = _tracked_mask[fi[ff, 2]]
    tracked = t0 & t1 & t2

    _tracker_sample_ids   = sample_ids
    _tracker_is_vertex    = is_vertex
    _tracker_fine_face_id = fine_face_ids
    _tracker_fine_bc      = fine_bc

    _sample_face_ids  = fine_face_ids
    _sample_bcs       = fine_bc
    _fine_sample_pts  = fine_pts
    _coarse_land_pts  = coarse_pts
    _sample_tracked   = tracked
    _coarse_face_ids  = None  # no per-sample coarse face for Mode 2 (3D blend)

    n_tr = int(tracked.sum())
    print(f"[mode2] {n_tr}/{N} samples on fully-tracked faces  ({100*n_tr/N:.1f}%)")
    return True


# ---------------------------------------------------------------------------
# Mode 3: tracker fine samples → per-sample forward UV-cast F2C query
# ---------------------------------------------------------------------------

_MODE3_CAP = 50_000   # warn + cap if sample count exceeds this

def _compute_tracker_fine_f2c_query() -> bool:
    """
    Read samples_fine_*.txt and run a per-sample forward UV-cast walk for
    each one via _query_f2c_batch() — the most accurate F2C mapping.

    Unlike Mode 2 (which blends 3 per-vertex F2C results), this walks each
    sample's exact BC through the UV_pre→UV_post chain at every collapse,
    staying topologically on a single coarse face throughout.

    Capped at _MODE3_CAP samples to keep response time reasonable.
    """
    global _sample_face_ids, _sample_bcs
    global _fine_sample_pts, _coarse_land_pts, _sample_tracked, _coarse_face_ids
    global _tracker_sample_ids, _tracker_is_vertex, _tracker_fine_face_id, _tracker_fine_bc

    b = _bundle
    if b is None or not b.has_ssp_data:
        print("[mode3] ERROR: bundle or SSP data not ready")
        return False
    if not _tracker_fine_path or not os.path.isfile(_tracker_fine_path):
        print(f"[mode3] ERROR: fine file not found: {_tracker_fine_path}")
        return False

    def _read_txt(path):
        with open(path, "r") as fh:
            lines = fh.readlines()
        data_lines = [l for l in lines if not l.strip().startswith("#")]
        count = int(data_lines[0].strip())
        return [l.split() for l in data_lines[1:count + 1]]

    fine_rows = _read_txt(_tracker_fine_path)
    N_full = len(fine_rows)

    if N_full > _MODE3_CAP:
        print(f"[mode3] {N_full} samples in file — capping at {_MODE3_CAP} for performance")
        fine_rows = fine_rows[:_MODE3_CAP]
    N = len(fine_rows)
    print(f"[mode3] Running per-sample F2C walk for {N} samples …")

    sample_ids    = np.zeros(N, dtype=np.int32)
    is_vertex     = np.zeros(N, dtype=bool)
    fine_face_ids = np.zeros(N, dtype=np.int32)
    fine_bc       = np.zeros((N, 3), dtype=np.float64)

    for i, fr in enumerate(fine_rows):
        sample_ids[i]    = int(fr[0])
        is_vertex[i]     = bool(int(fr[1]))
        fine_face_ids[i] = int(fr[2])
        fine_bc[i]       = [float(fr[3]), float(fr[4]), float(fr[5])]

    fv = b.fineV
    fi = b.fineF
    ff = fine_face_ids
    fine_pts = (fine_bc[:, 0:1] * fv[fi[ff, 0]]
              + fine_bc[:, 1:2] * fv[fi[ff, 1]]
              + fine_bc[:, 2:3] * fv[fi[ff, 2]])

    coarse_pts, tracked, final_ssp_faces = _query_f2c_batch(b, fine_face_ids, fine_bc)

    # Map SSP face indices → compact coarse face indices
    if _inv_faceMap is not None:
        compact_cf = np.array([_inv_faceMap.get(int(f), -1) for f in final_ssp_faces],
                              dtype=np.int32)
    else:
        compact_cf = None

    _tracker_sample_ids   = sample_ids
    _tracker_is_vertex    = is_vertex
    _tracker_fine_face_id = fine_face_ids
    _tracker_fine_bc      = fine_bc

    _sample_face_ids  = fine_face_ids
    _sample_bcs       = fine_bc
    _fine_sample_pts  = fine_pts
    _coarse_land_pts  = coarse_pts
    _sample_tracked   = tracked
    _coarse_face_ids  = compact_cf

    n_tr = int(tracked.sum())
    print(f"[mode3] {n_tr}/{N} samples reached a coarse face  ({100*n_tr/N:.1f}%)")
    return True


# ---------------------------------------------------------------------------
# Load tracker files — Mode 1
# ---------------------------------------------------------------------------

def _load_tracker_samples(fine_path: str, coarse_path: str) -> bool:
    """
    Read samples_fine_*.txt and samples_coarse_*.txt written by
    sample_tracker_save() and populate the shared sample arrays.

    Fine file columns:  sample_id  is_vertex  fine_face_id  b0  b1  b2
    Coarse file columns: sample_id  is_vertex  coarse_face_id  b0  b1  b2  bv0  bv1  bv2

    Coarse landing position = b0*coarseV[bv0] + b1*coarseV[bv1] + b2*coarseV[bv2]
    """
    global _sample_face_ids, _sample_bcs
    global _fine_sample_pts, _coarse_land_pts, _sample_tracked, _coarse_face_ids
    global _tracker_sample_ids, _tracker_is_vertex, _tracker_fine_face_id, _tracker_fine_bc
    global _tracker_coarse_face_id, _tracker_coarse_bc, _tracker_coarse_bv

    b = _bundle
    if b is None:
        print("[tracker] ERROR: no bundle loaded")
        return False
    if not os.path.isfile(fine_path):
        print(f"[tracker] ERROR: fine file not found: {fine_path}")
        return False
    if not os.path.isfile(coarse_path):
        print(f"[tracker] ERROR: coarse file not found: {coarse_path}")
        return False

    def _read_txt(path, skip_header=True):
        with open(path, "r") as fh:
            lines = fh.readlines()
        # strip comment lines starting with '#'
        data_lines = [l for l in lines if not l.strip().startswith("#")]
        # first data line is count
        count = int(data_lines[0].strip())
        rows  = [l.split() for l in data_lines[1:count + 1]]
        return rows

    fine_rows   = _read_txt(fine_path)
    coarse_rows = _read_txt(coarse_path)

    N = len(fine_rows)
    if len(coarse_rows) != N:
        print(f"[tracker] WARNING: fine has {N} samples but coarse has {len(coarse_rows)} — using min")
        N = min(N, len(coarse_rows))

    print(f"[tracker] Loading {N} samples from tracker files")

    sample_ids    = np.zeros(N, dtype=np.int32)
    is_vertex     = np.zeros(N, dtype=bool)
    fine_face_ids = np.zeros(N, dtype=np.int32)
    fine_bc       = np.zeros((N, 3), dtype=np.float64)

    coarse_face_ids = np.zeros(N, dtype=np.int32)
    coarse_bc       = np.zeros((N, 3), dtype=np.float64)
    coarse_bv       = np.zeros((N, 3), dtype=np.int32)

    for i in range(N):
        fr = fine_rows[i]
        cr = coarse_rows[i]
        sample_ids[i]    = int(fr[0])
        is_vertex[i]     = bool(int(fr[1]))
        fine_face_ids[i] = int(fr[2])
        fine_bc[i]       = [float(fr[3]), float(fr[4]), float(fr[5])]

        coarse_face_ids[i] = int(cr[2])
        coarse_bc[i]       = [float(cr[3]), float(cr[4]), float(cr[5])]
        coarse_bv[i]       = [int(cr[6]),   int(cr[7]),   int(cr[8])]

    # Fine positions — BC on original fine mesh faces (gFO in C++)
    ff = fine_face_ids
    fv = b.fineV
    fi = b.fineF
    fine_pts = (fine_bc[:, 0:1] * fv[fi[ff, 0]]
              + fine_bc[:, 1:2] * fv[fi[ff, 1]]
              + fine_bc[:, 2:3] * fv[fi[ff, 2]])

    # Coarse positions — bv0/bv1/bv2 are *global SSP vertex indices* (into gV),
    # NOT compact coarse indices. Convert via vtxMap inverse (global → compact).
    # Fall back to fineV[global] for any global index not in the compact coarse mesh
    # (should not happen for valid coarse landings, but guards edge cases).
    cv = b.coarseV
    fv_fallback = b.fineV
    if b.vtxMap is not None:
        g2c = {int(b.vtxMap[i]): i for i in range(len(b.vtxMap))}
    else:
        g2c = {}

    def _coarse_pos_from_global(global_ids):
        """Resolve N global SSP vertex indices → (N, 3) coarse positions."""
        pts = np.empty((len(global_ids), 3), dtype=np.float64)
        for j, gid in enumerate(global_ids):
            ci = g2c.get(int(gid))
            if ci is not None:
                pts[j] = cv[ci]
            else:
                # vertex was not a coarse survivor — use fine-mesh position
                gi = int(gid)
                pts[j] = fv_fallback[gi] if gi < len(fv_fallback) else cv[0]
        return pts

    bv0_g, bv1_g, bv2_g = coarse_bv[:, 0], coarse_bv[:, 1], coarse_bv[:, 2]
    p0 = _coarse_pos_from_global(bv0_g)
    p1 = _coarse_pos_from_global(bv1_g)
    p2 = _coarse_pos_from_global(bv2_g)
    coarse_pts = (coarse_bc[:, 0:1] * p0
                + coarse_bc[:, 1:2] * p1
                + coarse_bc[:, 2:3] * p2)

    # Report how many bv values had no compact-coarse mapping
    all_globals = np.concatenate([bv0_g, bv1_g, bv2_g])
    n_missing = sum(1 for g in all_globals if int(g) not in g2c)
    if n_missing:
        print(f"[tracker] WARNING: {n_missing}/{len(all_globals)} bv indices not in vtxMap "
              f"(fell back to fineV) — vtxMap present: {b.vtxMap is not None}")

    _tracker_sample_ids    = sample_ids
    _tracker_is_vertex     = is_vertex
    _tracker_fine_face_id  = fine_face_ids
    _tracker_fine_bc       = fine_bc
    _tracker_coarse_face_id = coarse_face_ids
    _tracker_coarse_bc     = coarse_bc
    _tracker_coarse_bv     = coarse_bv

    # Map tracker coarse_face_ids (SSP global face) → compact coarse face indices
    if _inv_faceMap is not None:
        compact_cf = np.array([_inv_faceMap.get(int(f), -1) for f in coarse_face_ids],
                              dtype=np.int32)
    else:
        compact_cf = None

    _sample_face_ids  = fine_face_ids
    _sample_bcs       = fine_bc
    _fine_sample_pts  = fine_pts
    _coarse_land_pts  = coarse_pts
    _sample_tracked   = is_vertex   # vertex samples are "tracked" in tracker terminology
    _coarse_face_ids  = compact_cf

    n_verts = int(is_vertex.sum())
    print(f"[tracker] {N} samples  ({n_verts} vertex, {N - n_verts} interior)")
    return True


# ---------------------------------------------------------------------------
# Auto-detect tracker files in a directory
# ---------------------------------------------------------------------------

def _find_tracker_files(directory: str):
    """Return (fine_path, coarse_path) found in directory, or ('', '')."""
    fine_candidates   = sorted(glob.glob(os.path.join(directory, "samples_fine_*.txt")))
    coarse_candidates = sorted(glob.glob(os.path.join(directory, "samples_coarse_*.txt")))
    if fine_candidates and coarse_candidates:
        return fine_candidates[0], coarse_candidates[0]
    return "", ""


# ---------------------------------------------------------------------------
# Active mask (face filter — Mode 0 only)
# ---------------------------------------------------------------------------

def _active_mask():
    if _sample_face_ids is None:
        return None
    mask = np.ones(len(_sample_face_ids), dtype=bool)
    # fine face filter (Mode 0 only)
    if _viz_mode == 0 and _selected_face >= 0:
        mask &= (_sample_face_ids == _selected_face)
    # coarse face filter (all modes when _coarse_face_ids available)
    if _coarse_selected_face >= 0 and _coarse_face_ids is not None:
        mask &= (_coarse_face_ids == _coarse_selected_face)
    return mask


# ---------------------------------------------------------------------------
# Mode cache helpers
# ---------------------------------------------------------------------------

def _snapshot_mode_state() -> dict:
    """Capture all per-sample globals into a dict for caching."""
    return dict(
        sample_face_ids      = _sample_face_ids,
        sample_bcs           = _sample_bcs,
        fine_sample_pts      = _fine_sample_pts,
        coarse_land_pts      = _coarse_land_pts,
        sample_tracked       = _sample_tracked,
        coarse_face_ids      = _coarse_face_ids,
        tracker_sample_ids   = _tracker_sample_ids,
        tracker_is_vertex    = _tracker_is_vertex,
        tracker_fine_face_id = _tracker_fine_face_id,
        tracker_fine_bc      = _tracker_fine_bc,
        tracker_coarse_face_id = _tracker_coarse_face_id,
        tracker_coarse_bc    = _tracker_coarse_bc,
        tracker_coarse_bv    = _tracker_coarse_bv,
    )


def _apply_mode_cache(mode: int):
    """Restore per-sample globals from cache for the given mode."""
    global _sample_face_ids, _sample_bcs, _fine_sample_pts, _coarse_land_pts
    global _sample_tracked, _coarse_face_ids
    global _tracker_sample_ids, _tracker_is_vertex, _tracker_fine_face_id, _tracker_fine_bc
    global _tracker_coarse_face_id, _tracker_coarse_bc, _tracker_coarse_bv
    global _viz_mode, _selected_face, _selected_sample, _coarse_selected_face

    s = _mode_cache.get(mode)
    if s is None:
        return False
    _sample_face_ids       = s['sample_face_ids']
    _sample_bcs            = s['sample_bcs']
    _fine_sample_pts       = s['fine_sample_pts']
    _coarse_land_pts       = s['coarse_land_pts']
    _sample_tracked        = s['sample_tracked']
    _coarse_face_ids       = s['coarse_face_ids']
    _tracker_sample_ids    = s['tracker_sample_ids']
    _tracker_is_vertex     = s['tracker_is_vertex']
    _tracker_fine_face_id  = s['tracker_fine_face_id']
    _tracker_fine_bc       = s['tracker_fine_bc']
    _tracker_coarse_face_id = s['tracker_coarse_face_id']
    _tracker_coarse_bc     = s['tracker_coarse_bc']
    _tracker_coarse_bv     = s['tracker_coarse_bv']
    _viz_mode              = mode
    _selected_face         = -1
    _selected_sample       = -1
    _coarse_selected_face  = -1
    return True


# ---------------------------------------------------------------------------
# Rebuild helpers
# ---------------------------------------------------------------------------

def _rebuild_meshes():
    fv = _bundle.fineV.copy()
    fv[:, 2] += _z_offset
    fm = ps.register_surface_mesh(FINE_MESH, fv, _bundle.fineF)
    fm.set_color((0.55, 0.55, 0.55))
    fm.set_edge_width(0.3)
    fm.set_smooth_shade(False)
    fm.set_transparency(0.4)

    cm = ps.register_surface_mesh(COARSE_MESH_VIZ, _bundle.coarseV, _bundle.coarseF)
    cm.set_color((0.15, 0.75, 0.35))
    cm.set_edge_width(1.0)
    cm.set_smooth_shade(False)
    cm.set_transparency(0.55)


def _rebuild_fine_sample_pc():
    if ps.has_point_cloud(FINE_SAMPLE_PC):
        ps.remove_point_cloud(FINE_SAMPLE_PC)
    if not _show_fine_dots or _fine_sample_pts is None:
        return
    mask = _active_mask()
    pts = _fine_sample_pts[mask].copy()
    pts[:, 2] += _z_offset
    pc = ps.register_point_cloud(FINE_SAMPLE_PC, pts)
    pc.set_color((0.25, 0.55, 1.0))
    pc.set_radius(0.005, relative=True)


def _rebuild_coarse_landing_pc():
    if ps.has_point_cloud(COARSE_LANDING_PC):
        ps.remove_point_cloud(COARSE_LANDING_PC)
    if not _show_coarse_dots or _coarse_land_pts is None:
        return
    mask = _active_mask()
    pc = ps.register_point_cloud(COARSE_LANDING_PC, _coarse_land_pts[mask])
    pc.set_color((0.95, 0.45, 0.10))
    pc.set_radius(0.005, relative=True)


def _rebuild_arrows():
    if ps.has_point_cloud(ARROWS_PC):
        ps.remove_point_cloud(ARROWS_PC)
    if _fine_sample_pts is None or _coarse_land_pts is None:
        return
    mask = _active_mask()
    src = _fine_sample_pts[mask].copy()
    src[:, 2] += _z_offset
    vecs = _coarse_land_pts[mask] - src
    pc = ps.register_point_cloud(ARROWS_PC, src)
    pc.set_color((1.0, 0.85, 0.0))
    pc.set_radius(0.002, relative=True)
    pc.add_vector_quantity("to_coarse", vecs, vectortype="ambient",
                           enabled=_show_arrows, color=(1.0, 0.55, 0.05))


def _rebuild_selection():
    for name in (SEL_FINE_PC, SEL_COARSE_PC):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
    if _selected_sample < 0 or _fine_sample_pts is None:
        return

    fpos = _fine_sample_pts[_selected_sample].copy()
    fpos[2] += _z_offset
    pc_f = ps.register_point_cloud(SEL_FINE_PC, fpos[np.newaxis])
    pc_f.set_color((1.0, 1.0, 0.0))
    pc_f.set_radius(0.012, relative=True)

    cpos = _coarse_land_pts[_selected_sample].copy()
    pc_c = ps.register_point_cloud(SEL_COARSE_PC, cpos[np.newaxis])
    pc_c.set_color((1.0, 0.0, 0.3))
    pc_c.set_radius(0.012, relative=True)


def _rebuild_all():
    _rebuild_meshes()
    _rebuild_fine_sample_pc()
    _rebuild_coarse_landing_pc()
    _rebuild_arrows()
    _rebuild_selection()


# ---------------------------------------------------------------------------
# UI callback
# ---------------------------------------------------------------------------

def ui_callback():
    global _z_offset, _mesh_span, _n_samples, _seed, _per_face_mode
    global _show_arrows, _show_fine_dots, _show_coarse_dots, _selected_sample
    global _pick_face_mode, _selected_face, _coarse_selected_face
    global _viz_mode
    global _tracker_fine_path, _tracker_coarse_path

    changed = False

    try:
        psim.SetNextWindowSize((400, 520), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Fine -> Coarse (Face Samples)", True)

    # ---- Mode selector ----
    psim.TextUnformatted("Correspondence Mode")
    clicked0 = psim.RadioButton("Mode 0: Calculated F2C (c2f_query)", _viz_mode == 0)
    if clicked0 and _viz_mode != 0:
        if _apply_mode_cache(0):
            _rebuild_all()
        elif _bundle is not None and _f2c_v is not None:
            _viz_mode = 0
            _selected_face = _selected_sample = _coarse_selected_face = -1
            _compute_f2c_samples()
            _rebuild_all()

    clicked1 = psim.RadioButton("Mode 1: SSP Sample Tracker files", _viz_mode == 1)
    if clicked1 and _viz_mode != 1:
        if _apply_mode_cache(1):
            _rebuild_all()
        elif _tracker_fine_path and _tracker_coarse_path:
            _viz_mode = 1
            _selected_face = _selected_sample = _coarse_selected_face = -1
            _load_tracker_samples(_tracker_fine_path, _tracker_coarse_path)
            _rebuild_all()

    clicked2 = psim.RadioButton("Mode 2: Tracker fine samples + F2C query", _viz_mode == 2)
    if clicked2 and _viz_mode != 2:
        if _apply_mode_cache(2):
            _rebuild_all()
        elif _tracker_fine_path and _f2c_v is not None:
            _viz_mode = 2
            _selected_face = _selected_sample = _coarse_selected_face = -1
            _compute_tracker_fine_f2c()
            _rebuild_all()

    clicked3 = psim.RadioButton("Mode 3: Tracker fine samples + per-sample UV-cast", _viz_mode == 3)
    if clicked3 and _viz_mode != 3:
        if _apply_mode_cache(3):
            _rebuild_all()
        elif _tracker_fine_path and _bundle is not None and _bundle.has_ssp_data:
            _viz_mode = 3
            _selected_face = _selected_sample = _coarse_selected_face = -1
            _compute_tracker_fine_f2c_query()
            _rebuild_all()

    psim.Separator()

    # ---- Z offset ----
    _slider_range = max(2.0, _mesh_span * 5)
    c, v = psim.SliderFloat("Z offset", _z_offset, -_slider_range, _slider_range)
    if c:
        _z_offset = v
        changed = True

    psim.Separator()

    # ---- Mode-specific controls ----
    if _viz_mode == 0:
        psim.TextUnformatted("[ Mode 0: Sampling on fine faces via F2C query ]")

        c, v = psim.Checkbox("Per face (N samples per fine face)", _per_face_mode)
        if c:
            _per_face_mode = v

        if _per_face_mode:
            c, v = psim.InputInt("Samples per face", _n_samples)
            if c:
                _n_samples = max(1, min(v, 1000))
            if _bundle is not None:
                FF = _bundle.fineF.shape[0]
                psim.TextDisabled(f"  = {_n_samples} x {FF} = {_n_samples * FF} total")
        else:
            c, v = psim.InputInt("Total samples", _n_samples)
            if c:
                _n_samples = max(1, min(v, 500000))
            psim.TextDisabled("  distributed by face area")

        c, v = psim.InputInt("Seed", _seed)
        if c:
            _seed = max(0, v)

        if psim.Button("Resample"):
            ok = _compute_f2c_samples()
            if ok:
                _selected_sample = -1
                _selected_face   = -1
                _rebuild_all()

        if _fine_sample_pts is not None:
            N    = len(_fine_sample_pts)
            n_tr = int(_sample_tracked.sum()) if _sample_tracked is not None else 0
            psim.TextUnformatted(f"{N} samples  |  {n_tr} on fully-tracked faces")

        psim.Separator()
        psim.TextUnformatted("Face Filter")

        c, v = psim.Checkbox("Pick face mode (click fine mesh)", _pick_face_mode)
        if c:
            _pick_face_mode = v
            if not v:
                _selected_face   = -1
                _selected_sample = -1
                _rebuild_fine_sample_pc()
                _rebuild_coarse_landing_pc()
                _rebuild_arrows()
                _rebuild_selection()

        if _selected_face >= 0:
            n_on = int((_sample_face_ids == _selected_face).sum()) if _sample_face_ids is not None else 0
            psim.TextUnformatted(f"  Face {_selected_face}  ({n_on} samples shown)")
            if psim.Button("Clear face filter"):
                _selected_face   = -1
                _selected_sample = -1
                _rebuild_fine_sample_pc()
                _rebuild_coarse_landing_pc()
                _rebuild_arrows()
                _rebuild_selection()
        elif _pick_face_mode:
            psim.TextDisabled("  Click a fine mesh face to filter")

    elif _viz_mode == 1:
        # Mode 1 — Tracker files (both fine + coarse from disk)
        psim.TextUnformatted("[ Mode 1: SSP Sample Tracker output files ]")

        psim.TextDisabled("Fine file:")
        psim.TextUnformatted(f"  {os.path.basename(_tracker_fine_path) if _tracker_fine_path else '(none)'}")
        psim.TextDisabled("Coarse file:")
        psim.TextUnformatted(f"  {os.path.basename(_tracker_coarse_path) if _tracker_coarse_path else '(none)'}")

        if psim.Button("Reload tracker files"):
            if _tracker_fine_path and _tracker_coarse_path:
                ok = _load_tracker_samples(_tracker_fine_path, _tracker_coarse_path)
                if ok:
                    _selected_sample = -1
                    _rebuild_all()
            else:
                print("[tracker] No tracker file paths set — check _tracker_fine_path")

        if _fine_sample_pts is not None:
            N     = len(_fine_sample_pts)
            n_vtx = int(_tracker_is_vertex.sum()) if _tracker_is_vertex is not None else 0
            psim.TextUnformatted(f"{N} samples  |  {n_vtx} vertex  {N - n_vtx} interior")

    elif _viz_mode == 2:
        # Mode 2 — Tracker fine samples, coarse via F2C query
        psim.TextUnformatted("[ Mode 2: Tracker fine samples -> F2C query ]")
        psim.TextDisabled("Fine file (sample positions):")
        psim.TextUnformatted(f"  {os.path.basename(_tracker_fine_path) if _tracker_fine_path else '(none)'}")
        psim.TextDisabled("Coarse landing: computed via F2C query (not tracker)")

        if psim.Button("Recompute (Mode 2)"):
            if _tracker_fine_path and _f2c_v is not None:
                ok = _compute_tracker_fine_f2c()
                if ok:
                    _selected_sample = -1
                    _rebuild_all()
            else:
                print("[mode2] Need tracker fine file and F2C map")

        if _fine_sample_pts is not None:
            N    = len(_fine_sample_pts)
            n_tr = int(_sample_tracked.sum()) if _sample_tracked is not None else 0
            n_vtx = int(_tracker_is_vertex.sum()) if _tracker_is_vertex is not None else 0
            psim.TextUnformatted(f"{N} samples  |  {n_vtx} vertex  {N - n_vtx} interior")
            psim.TextUnformatted(f"  {n_tr} on fully-tracked faces")

    elif _viz_mode == 3:
        # Mode 3 — per-sample forward UV-cast
        psim.TextUnformatted("[ Mode 3: Tracker fine samples + per-sample UV-cast ]")
        psim.TextDisabled("Fine file (sample positions):")
        psim.TextUnformatted(f"  {os.path.basename(_tracker_fine_path) if _tracker_fine_path else '(none)'}")
        psim.TextDisabled(f"Coarse landing: per-sample forward UV-cast  (cap {_MODE3_CAP:,})")

        if psim.Button("Recompute (Mode 3)"):
            if _tracker_fine_path and _bundle is not None and _bundle.has_ssp_data:
                ok = _compute_tracker_fine_f2c_query()
                if ok:
                    _selected_sample = -1
                    _rebuild_all()
            else:
                print("[mode3] Need tracker fine file and SSP bundle")

        if _fine_sample_pts is not None:
            N    = len(_fine_sample_pts)
            n_tr = int(_sample_tracked.sum()) if _sample_tracked is not None else 0
            n_vtx = int(_tracker_is_vertex.sum()) if _tracker_is_vertex is not None else 0
            psim.TextUnformatted(f"{N} samples  |  {n_vtx} vertex  {N - n_vtx} interior")
            psim.TextUnformatted(f"  {n_tr} reached a coarse face")

    # ---- Coarse face filter (available in Modes 1 and 3) ----
    psim.Separator()
    psim.TextUnformatted("Coarse Face Filter")
    if _coarse_face_ids is not None:
        if _coarse_selected_face >= 0:
            n_on = int((_coarse_face_ids == _coarse_selected_face).sum())
            psim.TextUnformatted(f"  Coarse face {_coarse_selected_face}  ({n_on} samples)")
            if psim.Button("Clear coarse face filter"):
                _coarse_selected_face = -1
                _selected_sample      = -1
                _rebuild_fine_sample_pc()
                _rebuild_coarse_landing_pc()
                _rebuild_arrows()
                _rebuild_selection()
        else:
            psim.TextDisabled("  Click a coarse mesh face to filter")
    else:
        psim.TextDisabled("  (not available for this mode — use Mode 1 or 3)")

    # ---- Display ----
    psim.Separator()
    psim.TextUnformatted("Display")

    c, v = psim.Checkbox("Show fine sample dots", _show_fine_dots)
    if c:
        _show_fine_dots = v
        _rebuild_fine_sample_pc()

    c, v = psim.Checkbox("Show coarse landing dots", _show_coarse_dots)
    if c:
        _show_coarse_dots = v
        _rebuild_coarse_landing_pc()

    c, v = psim.Checkbox("Show arrows", _show_arrows)
    if c:
        _show_arrows = v
        _rebuild_arrows()

    # ---- Sample inspector ----
    psim.Separator()
    if _selected_sample >= 0 and _fine_sample_pts is not None:
        si  = _selected_sample
        fi  = int(_sample_face_ids[si])
        bc  = _sample_bcs[si]
        fp  = _fine_sample_pts[si]
        cp  = _coarse_land_pts[si]
        dist = float(np.linalg.norm(cp - fp))

        if _viz_mode == 0:
            trk = bool(_sample_tracked[si]) if _sample_tracked is not None else False
            psim.TextUnformatted(f"Sample {si}  {'[tracked]' if trk else '[untracked]'}")
            psim.TextUnformatted(f"  Fine face   : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
        elif _viz_mode == 1:
            is_vtx = bool(_tracker_is_vertex[si]) if _tracker_is_vertex is not None else False
            sid    = int(_tracker_sample_ids[si]) if _tracker_sample_ids is not None else si
            cfi    = int(_tracker_coarse_face_id[si]) if _tracker_coarse_face_id is not None else -1
            bv     = _tracker_coarse_bv[si] if _tracker_coarse_bv is not None else [0, 0, 0]
            psim.TextUnformatted(f"Sample {sid}  {'[vertex]' if is_vtx else '[interior]'}")
            psim.TextUnformatted(f"  Fine face   : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
            psim.TextUnformatted(f"  Coarse face : {cfi}  verts=({bv[0]},{bv[1]},{bv[2]})")
        elif _viz_mode == 2:
            is_vtx = bool(_tracker_is_vertex[si]) if _tracker_is_vertex is not None else False
            sid    = int(_tracker_sample_ids[si]) if _tracker_sample_ids is not None else si
            trk    = bool(_sample_tracked[si]) if _sample_tracked is not None else False
            psim.TextUnformatted(f"Sample {sid}  {'[vertex]' if is_vtx else '[interior]'}  "
                                 f"{'[tracked]' if trk else '[untracked]'}")
            psim.TextUnformatted(f"  Fine face   : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
            psim.TextDisabled("  Coarse via per-vertex F2C blend")
        else:  # Mode 3
            is_vtx = bool(_tracker_is_vertex[si]) if _tracker_is_vertex is not None else False
            sid    = int(_tracker_sample_ids[si]) if _tracker_sample_ids is not None else si
            trk    = bool(_sample_tracked[si]) if _sample_tracked is not None else False
            psim.TextUnformatted(f"Sample {sid}  {'[vertex]' if is_vtx else '[interior]'}  "
                                 f"{'[reached coarse]' if trk else '[no collapse hit]'}")
            psim.TextUnformatted(f"  Fine face   : {fi}  BC=({bc[0]:.3f},{bc[1]:.3f},{bc[2]:.3f})")
            psim.TextDisabled("  Coarse via per-sample UV-cast walk")

        psim.TextUnformatted(f"  Fine pos    : ({fp[0]:.5f},{fp[1]:.5f},{fp[2]:.5f})")
        psim.TextUnformatted(f"  Coarse land : ({cp[0]:.5f},{cp[1]:.5f},{cp[2]:.5f})")
        psim.TextUnformatted(f"  Dist F->C   : {dist:.6f}")
        if psim.Button("Clear selection"):
            _selected_sample = -1
            _rebuild_selection()
    else:
        psim.TextDisabled("Click a blue fine sample dot to inspect")

    psim.End()

    # ---- pick handler ----
    io = psim.GetIO()
    if io.MouseClicked[0]:
        try:
            pr = ps.pick(screen_coords=io.MousePos)
        except Exception:
            pr = None
        if pr is not None and getattr(pr, 'is_hit', False):
            sname = getattr(pr, 'structure_name', None)
            sdata = getattr(pr, 'structure_data', None) or {}

            if sname == FINE_MESH and _pick_face_mode and _viz_mode == 0:
                etype = sdata.get('element_type', None)
                fidx  = int(sdata.get('index', -1))
                nF    = _bundle.fineF.shape[0]
                if etype == 'face' and 0 <= fidx < nF:
                    _selected_face   = -1 if fidx == _selected_face else fidx
                    _selected_sample = -1
                    _rebuild_fine_sample_pc()
                    _rebuild_coarse_landing_pc()
                    _rebuild_arrows()
                    _rebuild_selection()
                    if _selected_face >= 0:
                        n_on = int((_sample_face_ids == _selected_face).sum()) \
                               if _sample_face_ids is not None else 0
                        print(f"\n[face filter] fine face {_selected_face}  ({n_on} samples)")

            elif sname == COARSE_MESH_VIZ and _coarse_face_ids is not None:
                etype = sdata.get('element_type', None)
                fidx  = int(sdata.get('index', -1))
                nCF   = _bundle.coarseF.shape[0]
                if etype == 'face' and 0 <= fidx < nCF:
                    _coarse_selected_face = -1 if fidx == _coarse_selected_face else fidx
                    _selected_sample      = -1
                    _rebuild_fine_sample_pc()
                    _rebuild_coarse_landing_pc()
                    _rebuild_arrows()
                    _rebuild_selection()
                    if _coarse_selected_face >= 0:
                        n_on = int((_coarse_face_ids == _coarse_selected_face).sum())
                        print(f"\n[coarse filter] coarse face {_coarse_selected_face}  ({n_on} samples)")

            elif sname == FINE_SAMPLE_PC:
                idx = int(sdata.get('index', -1))
                if _fine_sample_pts is not None and idx >= 0:
                    mask    = _active_mask()
                    indices = np.where(mask)[0]
                    global_idx = int(indices[idx]) if idx < len(indices) else -1
                    if global_idx >= 0:
                        _selected_sample = -1 if global_idx == _selected_sample else global_idx
                        _rebuild_selection()
                        if _selected_sample >= 0:
                            si  = _selected_sample
                            fi  = int(_sample_face_ids[si])
                            bc  = _sample_bcs[si]
                            fp  = _fine_sample_pts[si]
                            cp  = _coarse_land_pts[si]
                            print(f"\n[sample {si}]  fine face={fi}  "
                                  f"BC=({bc[0]:.4f},{bc[1]:.4f},{bc[2]:.4f})")
                            print(f"  fine pos    : ({fp[0]:.6f},{fp[1]:.6f},{fp[2]:.6f})")
                            print(f"  coarse land : ({cp[0]:.6f},{cp[1]:.6f},{cp[2]:.6f})")
                            print(f"  dist F->C   : {np.linalg.norm(cp-fp):.6f}")
                            if _viz_mode == 1 and _tracker_is_vertex is not None:
                                is_vtx = bool(_tracker_is_vertex[si])
                                print(f"  type        : {'vertex' if is_vtx else 'interior'}")

    if changed and _fine_sample_pts is not None:
        _rebuild_all()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _bundle, _bundle_dir, _z_offset, _mesh_span, _f2c_v, _tracked_mask
    global _tracker_fine_path, _tracker_coarse_path, _inv_faceMap

    c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\01_00040057_f8f78dbd17414efda75bc437_trimesh_000\\correspondence_01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial.c2f"
    # c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\0002000_partstudio_14_model_ste_00_1024\\correspondence_0002000_partstudio_14_model_ste_00_1024.c2f"
    # c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\hand\\correspondence_hand.c2f"
    # c2f_path = rf"C:\\Users\\alirz\\Projects\\Graphics\\Neural QMAT\\external\\surf_subgrid_SSP_orig\\10_collapse_viz\\output\\subDiv_cube\\correspondence_subDiv_cube.c2f"

    _bundle_dir = os.path.dirname(os.path.abspath(c2f_path))
    _bundle     = load_bundle(c2f_path)

    fv = _bundle.fineV
    mesh_span = float((fv.max(axis=0) - fv.min(axis=0)).max())
    if mesh_span > 1e-10:
        _z_offset  = mesh_span
        _mesh_span = mesh_span

    print(f"Fine:   {_bundle.fineV.shape[0]} verts  {_bundle.fineF.shape[0]} faces")
    print(f"Coarse: {_bundle.coarseV.shape[0]} verts  {_bundle.coarseF.shape[0]} faces")
    if _bundle.has_ssp_data:
        print(f"SSP:    {len(_bundle.decInfo)} collapses")

    # Auto-detect tracker files in the same output directory as the c2f file
    fine_found, coarse_found = _find_tracker_files(_bundle_dir)
    if fine_found and coarse_found:
        _tracker_fine_path   = fine_found
        _tracker_coarse_path = coarse_found
        print(f"[tracker] Found fine   : {os.path.basename(fine_found)}")
        print(f"[tracker] Found coarse : {os.path.basename(coarse_found)}")
    else:
        print(f"[tracker] No tracker files found in: {_bundle_dir}")

    # Build inverse faceMap: global SSP face index → compact coarse face index
    if _bundle.faceMap is not None:
        _inv_faceMap = {int(_bundle.faceMap[i]): i for i in range(len(_bundle.faceMap))}
        print(f"[main] Built inverse faceMap: {len(_inv_faceMap)} coarse faces")
    else:
        _inv_faceMap = None
        print("[main] WARNING: bundle has no faceMap — coarse face filter unavailable")

    print("\n[main] Computing per-vertex F2C map ...")
    _f2c_v, _tracked_mask = compute_f2c_correspondences(_bundle)
    n_tr = int(_tracked_mask.sum())
    print(f"[main] {n_tr}/{_bundle.fineV.shape[0]} fine vertices tracked by F2C")

    # Pre-compute all modes and cache them for instant switching
    global _viz_mode, _mode_cache
    print("\n[main] Pre-computing Mode 0 (Calculated F2C) ...")
    if _compute_f2c_samples():
        _mode_cache[0] = _snapshot_mode_state()

    if _tracker_fine_path and _tracker_coarse_path:
        print("[main] Pre-computing Mode 1 (SSP Tracker files) ...")
        if _load_tracker_samples(_tracker_fine_path, _tracker_coarse_path):
            _mode_cache[1] = _snapshot_mode_state()

    if _tracker_fine_path and _f2c_v is not None:
        print("[main] Pre-computing Mode 2 (Tracker fine + F2C query) ...")
        if _compute_tracker_fine_f2c():
            _mode_cache[2] = _snapshot_mode_state()

    if _tracker_fine_path and _bundle.has_ssp_data:
        print("[main] Pre-computing Mode 3 (Tracker fine + per-sample UV-cast) ...")
        if _compute_tracker_fine_f2c_query():
            _mode_cache[3] = _snapshot_mode_state()

    # Activate the best available starting mode
    start_mode = 2 if 2 in _mode_cache else (0 if 0 in _mode_cache else None)
    if start_mode is None:
        print("ERROR: could not compute any mode — exiting.")
        sys.exit(1)
    if not _apply_mode_cache(start_mode):
        print("ERROR: cache apply failed — exiting.")
        sys.exit(1)
    print(f"[main] Starting in Mode {start_mode}")

    ps.init()
    ps.set_program_name("Fine -> Coarse (Face Samples)")
    ps.set_up_dir("y_up")

    _rebuild_all()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
