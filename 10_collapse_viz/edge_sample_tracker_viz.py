#!/usr/bin/env python3
"""
edge_sample_tracker_viz.py
  — Visualize edge_sample_tracker's output: samples seeded on the fine
    mesh's boundary/seam edges (from .ma_struct), each carrying its live
    correspondence on the coarse (decimated) mesh.

Reads:
  edge_samples_<stem>.txt   — written by edge_sample_tracker_save() (see
                              edge_sample_tracker.h). One row per sample:
                              id type_id struct_id src_v0 src_v1 t
                              seed_face_id cur_FIdx b0 b1 b2 bv0 bv1 bv2
  correspondence_<stem>.c2f — the .c2f bundle written by
                              coarse_fine_save_bundle() for the same run
                              (gives fineV/fineF and the compact coarseV/
                              coarseF + the compact<->global vertex map
                              needed to resolve bv0/bv1/bv2, which are raw
                              gV indices, not compact coarseV indices).
                              Parsed here with a small self-contained v7
                              reader (see _load_c2f_bundle below) rather
                              than 11_correspond_viz/c2f_query.py, whose
                              magic allowlist predates the current v7
                              bundle format (0xC2F50007, coarse_fine_viz.cpp
                              :627-681) and would reject it.

Layout:
  coarse mesh + its two sample point clouds   at Z = 0
  fine mesh   + its two sample point clouds   at Z = +Z_OFFSET (default 2)
  vector quantity "to_fine" on each coarse-mesh point cloud: arrow from the
    coarse sample position to its live fine-mesh correspondence (toggle via
    the "Coarse -> fine correspondence" checkbox, ON by default).
  vector quantity "to_coarse" on each fine-mesh point cloud: arrow from the
    fine sample position to its live coarse-mesh correspondence (toggle via
    the "Fine -> coarse correspondence" checkbox, OFF by default).

UI (all live, no restart needed):
  - "Z offset (fine mesh)" slider
  - "Show every Nth sample" — subsamples both point clouds by row index
  - "Show seams" / "Show boundaries" checkboxes
  - "Coarse -> fine correspondence (arrows)" checkbox (default ON)
  - "Fine -> coarse correspondence (arrows)" checkbox (default OFF)
  - "Structures (by struct_id)" tree — per-struct_id checkboxes, grouped
    under "Seams" / "Boundaries", each with an All/None shortcut, so any
    individual seam or boundary curve can be toggled on/off independently
    of the others

Colors (match collapse_structure_tracker/simp_viz_tracker.cpp's MS_Seam /
MS_Boundary palette):
  seam      -> orange  (255,165,0)
  boundary  -> crimson (220,20,60)

Usage:
    python edge_sample_tracker_viz.py <edge_samples.txt> [<bundle.c2f>] [--no-gui]

  <bundle.c2f> defaults to correspondence_<stem>.c2f in the same directory
  as <edge_samples.txt> (stem = edge_samples_<stem>.txt -> <stem>).
  --no-gui: load + compute + print stats, then exit without opening the
  polyscope window (used for headless verification).

Dependencies:
    pip install polyscope numpy
    (c2f_query.py must be importable — same directory as this script, or
    on PYTHONPATH; it lives in ../11_correspond_viz)
"""

import sys
import os
import struct
import argparse
from dataclasses import dataclass
import numpy as np

Z_OFFSET_DEFAULT = 2.0
STRIDE_DEFAULT   = 50

SEAM_TYPE_ID     = 1
BOUNDARY_TYPE_ID = 2

COLOR_SEAM     = (1.0, 0.647, 0.0)     # orange   — matches simp_viz_tracker MS_Seam
COLOR_BOUNDARY = (0.863, 0.078, 0.235)  # crimson  — matches simp_viz_tracker MS_Boundary

FINE_MESH   = "fine_mesh"
COARSE_MESH = "coarse_mesh"


# ---------------------------------------------------------------------------
# .c2f bundle (v7) reader — only the header fields this script needs:
# coarseV/coarseF, fineV/fineF, and the compact->global vertex map. See the
# format comment at coarse_fine_viz.cpp:569-582 (base layout) and the v7
# note at coarse_fine_viz.cpp:627-681 (adds NCE + stale-chain-extended
# coarseV/vtxMap, which is what changed vs. the older v1-v6 formats).
# ---------------------------------------------------------------------------

C2F_MAGIC_V7 = 0xC2F50007


@dataclass
class Bundle:
    coarseV: np.ndarray  # (NCE, 3) float64 — includes naked stale-chain verts past NC
    coarseF: np.ndarray  # (FC, 3) int64    — compact indices into coarseV
    fineV: np.ndarray    # (NF, 3) float64
    fineF: np.ndarray    # (FF, 3) int64
    vtxMap: np.ndarray   # (NCE,) int64     — compact coarse index -> global (raw gV) index
    nV_total: int        # global SSP vertex count (gV.rows())


def load_c2f_bundle(path):
    with open(path, "rb") as f:
        data = f.read()

    magic, = struct.unpack_from("<I", data, 0)
    if magic != C2F_MAGIC_V7:
        raise ValueError(
            f"{path}: bundle magic 0x{magic:08X} != expected v7 (0x{C2F_MAGIC_V7:08X}). "
            f"This reader only supports the current v7 .c2f format — regenerate the bundle "
            f"with the current collapse_viz_bin, or extend load_c2f_bundle() for older formats.")

    pos = 4
    def u32():
        nonlocal pos
        v, = struct.unpack_from("<I", data, pos); pos += 4
        return v
    def f64_array(n):
        nonlocal pos
        arr = np.frombuffer(data, dtype="<f8", count=n, offset=pos).copy(); pos += 8 * n
        return arr
    def u32_array(n):
        nonlocal pos
        arr = np.frombuffer(data, dtype="<u4", count=n, offset=pos).astype(np.int64); pos += 4 * n
        return arr
    def i32_array(n):
        nonlocal pos
        arr = np.frombuffer(data, dtype="<i4", count=n, offset=pos).astype(np.int64); pos += 4 * n
        return arr

    NC  = u32()
    FC  = u32()
    NF  = u32()
    FF  = u32()
    NCE = u32()

    coarseV = f64_array(NCE * 3).reshape(NCE, 3)
    coarseF = u32_array(FC * 3).reshape(FC, 3)
    fineV   = f64_array(NF * 3).reshape(NF, 3)
    fineF   = u32_array(FF * 3).reshape(FF, 3)

    # Correspondence table: NC x (3 double bc + 3 uint32 fv) — not needed here, skip.
    pos += NC * (3 * 8 + 3 * 4)

    nV_total = u32()
    _nF_decIM = u32()
    _nFO = u32()

    vtxMap = i32_array(NCE)  # compact -> global, covers stale-chain verts too

    return Bundle(coarseV=coarseV, coarseF=coarseF, fineV=fineV, fineF=fineF,
                 vtxMap=vtxMap, nV_total=nV_total)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_edge_samples(path):
    """
    Returns a dict of numpy arrays, one entry per column, from an
    edge_samples_<stem>.txt file (see edge_sample_tracker.h for the format).
    """
    with open(path, "r") as f:
        header = f.readline()  # "# id type_id struct_id ..."
        n = int(f.readline().strip())
        rows = np.loadtxt(f, max_rows=n)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    cols = ["id", "type_id", "struct_id", "src_v0", "src_v1", "t",
            "seed_face_id", "cur_FIdx", "b0", "b1", "b2", "bv0", "bv1", "bv2"]
    data = {c: rows[:, i] for i, c in enumerate(cols)}
    for c in ("id", "type_id", "struct_id", "src_v0", "src_v1",
              "seed_face_id", "cur_FIdx", "bv0", "bv1", "bv2"):
        data[c] = data[c].astype(np.int64)
    print(f"[edge_sample_viz] loaded {n} edge samples from {path}")
    return data


def default_bundle_path(edge_samples_path):
    d = os.path.dirname(os.path.abspath(edge_samples_path))
    base = os.path.basename(edge_samples_path)
    if not base.startswith("edge_samples_") or not base.endswith(".txt"):
        raise ValueError(f"unexpected edge samples filename: {base}")
    stem = base[len("edge_samples_"):-len(".txt")]
    return os.path.join(d, f"correspondence_{stem}.c2f")


def build_global_to_compact(bundle):
    """
    bundle.vtxMap is (NC,) compact -> global (raw gV/SSP index). Invert it
    so bv0/bv1/bv2 (raw indices from edge_samples_*.txt, same convention as
    face_sample_tracker's cur_BF) can be looked up directly into coarseV.
    """
    n_global = int(bundle.nV_total) if bundle.nV_total > 0 else int(bundle.vtxMap.max()) + 1
    g2c = np.full(n_global, -1, dtype=np.int64)
    g2c[bundle.vtxMap] = np.arange(len(bundle.vtxMap), dtype=np.int64)
    return g2c


def compute_positions(samples, bundle, g2c):
    """
    fine_pos[i]   = lerp(fineV[src_v0], fineV[src_v1], t)   — matches the
                    seeding formula in edge_sample_tracker.cpp exactly
                    (bc(c0)=1-t on src_v0's column, bc(c1)=t on src_v1's).
    coarse_pos[i] = b0*coarseV[c(bv0)] + b1*coarseV[c(bv1)] + b2*coarseV[c(bv2)]
    """
    fineV = bundle.fineV
    fine_pos = ((1.0 - samples["t"])[:, None] * fineV[samples["src_v0"]]
                + samples["t"][:, None] * fineV[samples["src_v1"]])

    c0 = g2c[samples["bv0"]]
    c1 = g2c[samples["bv1"]]
    c2 = g2c[samples["bv2"]]
    missing = (c0 < 0) | (c1 < 0) | (c2 < 0)
    n_missing = int(missing.sum())
    if n_missing:
        print(f"[edge_sample_viz] WARNING: {n_missing}/{len(missing)} samples reference "
              f"a coarse vertex not in this bundle's compact mesh (skipped from coarse viz)")
        c0 = np.where(missing, 0, c0)
        c1 = np.where(missing, 0, c1)
        c2 = np.where(missing, 0, c2)

    coarseV = bundle.coarseV
    b0 = samples["b0"][:, None]
    b1 = samples["b1"][:, None]
    b2 = samples["b2"][:, None]
    coarse_pos = b0 * coarseV[c0] + b1 * coarseV[c1] + b2 * coarseV[c2]

    return fine_pos, coarse_pos, ~missing


def compute_length_weighting(samples, fineV):
    """
    Per-sample data needed to make display density proportional to each
    seam/boundary curve's actual arc length, instead of proportional to its
    raw sample count (which edge_sample_tracker.cpp seeds equally per
    struct_id regardless of curve length — see edge_sample_tracker.h — so a
    naive uniform row-stride under-represents long curves visually).

    Returns:
      local_idx[i]   — 0-based index of sample i within its struct_id's
                        contiguous block (edge_sample_tracker.cpp seeds all
                        samples of one struct_id consecutively, so struct_id
                        forms contiguous runs in file order).
      struct_len[i]  — arc length (sum of unique edge lengths) of sample i's
                        struct_id, computed directly from the fine mesh
                        (no C++/file-format change needed).
      length_ref     — median struct arc length across all structs present;
                        the reference a struct's length is weighted against.
    """
    sid = samples["struct_id"]

    # local_idx: position within this struct_id's contiguous run.
    change = np.empty(sid.shape[0], dtype=bool)
    change[0] = True
    change[1:] = sid[1:] != sid[:-1]
    group_start = np.where(change)[0]
    group_of_row = np.cumsum(change) - 1
    local_idx = np.arange(sid.shape[0]) - group_start[group_of_row]

    # struct_len: dedupe (struct_id, edge) so an edge hit by many samples is
    # only counted once, then sum per struct_id.
    v0 = np.minimum(samples["src_v0"], samples["src_v1"])
    v1 = np.maximum(samples["src_v0"], samples["src_v1"])
    combo = np.stack([sid, v0, v1], axis=1)
    uniq = np.unique(combo, axis=0)
    edge_len = np.linalg.norm(fineV[uniq[:, 1]] - fineV[uniq[:, 2]], axis=1)

    max_sid = int(sid.max()) + 1
    struct_len_by_id = np.zeros(max_sid, dtype=np.float64)
    np.add.at(struct_len_by_id, uniq[:, 0], edge_len)

    present_ids = np.unique(sid)
    lengths_present = struct_len_by_id[present_ids]
    lengths_present = lengths_present[lengths_present > 0]
    length_ref = float(np.median(lengths_present)) if lengths_present.size else 1.0

    struct_len = struct_len_by_id[sid]
    return local_idx, struct_len, length_ref


def build_struct_list(samples, struct_len):
    """
    One entry per distinct struct_id present, for the per-struct checkbox
    tree: (struct_id, type_id, length, count). struct_len is the per-sample
    array from compute_length_weighting() (same value repeated within a
    struct_id's block), so indexing at each struct's first occurrence gives
    that struct's length without recomputing anything.

    Sorted by type_id (seam before boundary) then descending length, so the
    checkbox tree lists longer curves first within each group.
    """
    sid = samples["struct_id"]
    uniq_sid, first_pos, counts = np.unique(sid, return_index=True, return_counts=True)
    types_u   = samples["type_id"][first_pos]
    lengths_u = struct_len[first_pos]
    entries = list(zip(uniq_sid.tolist(), types_u.tolist(), lengths_u.tolist(), counts.tolist()))
    entries.sort(key=lambda e: (e[1], -e[2]))
    return entries  # [(struct_id, type_id, length, count), ...]


# ---------------------------------------------------------------------------
# Visualization (stateful — z-offset is a live slider, meshes/point clouds
# are rebuilt in place whenever it changes)
# ---------------------------------------------------------------------------

_state = {}  # bundle, fine_pos, coarse_pos, is_seam, is_boundary, z_offset, mesh_span,
             # stride, show_seam, show_boundary, show_coarse_to_fine, show_fine_to_coarse,
             # length_weighted, local_idx, struct_len, length_ref


def _visible_mask(type_mask, show):
    """
    type_mask (is_seam or is_boundary) AND-ed with:
      - `show` (the seam/boundary checkbox for that type)
      - `st['struct_vis']` (per-struct_id checkbox tree — see build_struct_list())
      - a stride-based thinning, either:
          uniform:         keep every Nth sample by row index (N = st['stride'])
          length-weighted: keep every Nth sample WITHIN each struct's own
                            contiguous block, where N is scaled per-struct by
                            length_ref / struct_len — so a struct twice as
                            long as the reference gets a stride half as big
                            (roughly 2x the displayed points), making display
                            density approximate arc length rather than raw
                            sample count (see compute_length_weighting()).
    """
    if not show:
        return np.zeros_like(type_mask)
    st = _state
    mask = type_mask & st["struct_vis"][st["struct_id"]]
    stride = max(1, int(st.get("stride", 1)))
    if stride <= 1:
        return mask
    if st.get("length_weighted", False):
        eff_stride = np.maximum(
            1, np.round(stride * st["length_ref"] / np.maximum(st["struct_len"], 1e-9))
        ).astype(np.int64)
        return mask & (st["local_idx"] % eff_stride == 0)
    idx = np.arange(type_mask.shape[0])
    return mask & (idx % stride == 0)


def _rebuild_fine_side(ps):
    st = _state
    fv = st["bundle"].fineV.copy()
    fv[:, 2] += st["z_offset"]
    fm = ps.register_surface_mesh(FINE_MESH, fv, st["bundle"].fineF)
    fm.set_color((0.55, 0.55, 0.55))
    fm.set_edge_width(0.3)
    fm.set_transparency(0.4)

    def add_fine_pc(name, mask, color):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
        if not mask.any():
            return
        pts = st["fine_pos"][mask].copy()
        pts[:, 2] += st["z_offset"]
        pc = ps.register_point_cloud(name, pts)
        pc.set_color(color)
        pc.set_radius(0.0025, relative=True)
        if st["show_fine_to_coarse"]:
            vecs = st["coarse_pos"][mask] - pts  # fine sample (shifted) -> coarse correspondence
            pc.add_vector_quantity("to_coarse", vecs, vectortype="ambient",
                                   enabled=True, color=color)

    add_fine_pc("fine_seam_pts", _visible_mask(st["is_seam"], st["show_seam"]), COLOR_SEAM)
    add_fine_pc("fine_boundary_pts", _visible_mask(st["is_boundary"], st["show_boundary"]), COLOR_BOUNDARY)


def _rebuild_coarse_side(ps):
    st = _state
    cm = ps.register_surface_mesh(COARSE_MESH, st["bundle"].coarseV, st["bundle"].coarseF)
    cm.set_color((0.15, 0.75, 0.35))
    cm.set_edge_width(1.0)
    cm.set_transparency(0.55)

    def add_coarse_pc(name, mask, color):
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
        if not mask.any():
            return
        pts = st["coarse_pos"][mask]
        pc = ps.register_point_cloud(name, pts)
        pc.set_color(color)
        pc.set_radius(0.0025, relative=True)
        if st["show_coarse_to_fine"]:
            fine_pts = st["fine_pos"][mask].copy()
            fine_pts[:, 2] += st["z_offset"]
            vecs = fine_pts - pts  # coarse position -> fine correspondence (shifted)
            pc.add_vector_quantity("to_fine", vecs, vectortype="ambient",
                                   enabled=True, color=color)

    add_coarse_pc("coarse_seam_pts", _visible_mask(st["is_seam"], st["show_seam"]), COLOR_SEAM)
    add_coarse_pc("coarse_boundary_pts", _visible_mask(st["is_boundary"], st["show_boundary"]), COLOR_BOUNDARY)


def _ui_callback():
    import polyscope as ps
    import polyscope.imgui as psim

    st = _state
    try:
        psim.SetNextWindowSize((420, 520), psim.ImGuiCond_FirstUseEver)
    except Exception:
        pass
    psim.Begin("Edge Sample Tracker", True)
    n_seam_shown = int(_visible_mask(st["is_seam"], st["show_seam"]).sum())
    n_bnd_shown  = int(_visible_mask(st["is_boundary"], st["show_boundary"]).sum())
    psim.TextUnformatted(f"{int(st['is_seam'].sum())} seam + "
                          f"{int(st['is_boundary'].sum())} boundary samples total")
    psim.TextUnformatted(f"showing {n_seam_shown} seam + {n_bnd_shown} boundary "
                          f"(every {int(st['stride'])})")
    psim.Separator()

    slider_range = max(2.0, st["mesh_span"] * 5.0)
    changed, v = psim.SliderFloat("Z offset (fine mesh)", st["z_offset"],
                                  -slider_range, slider_range)
    if changed:
        st["z_offset"] = v
        _rebuild_fine_side(ps)
        if st["show_coarse_to_fine"]:
            _rebuild_coarse_side(ps)  # vector endpoints depend on z_offset too

    changed, v = psim.InputInt("Show every Nth sample", int(st["stride"]))
    if changed:
        st["stride"] = max(1, v)
        _rebuild_fine_side(ps)
        _rebuild_coarse_side(ps)

    changed, v = psim.Checkbox("Length-weighted sampling (display density ~ curve length)",
                               st["length_weighted"])
    if changed:
        st["length_weighted"] = v
        _rebuild_fine_side(ps)
        _rebuild_coarse_side(ps)

    psim.Separator()

    changed, v = psim.Checkbox("Show seams", st["show_seam"])
    if changed:
        st["show_seam"] = v
        _rebuild_fine_side(ps)
        _rebuild_coarse_side(ps)

    changed, v = psim.Checkbox("Show boundaries", st["show_boundary"])
    if changed:
        st["show_boundary"] = v
        _rebuild_fine_side(ps)
        _rebuild_coarse_side(ps)

    changed, v = psim.Checkbox("Coarse -> fine correspondence (arrows)", st["show_coarse_to_fine"])
    if changed:
        st["show_coarse_to_fine"] = v
        _rebuild_coarse_side(ps)

    changed, v = psim.Checkbox("Fine -> coarse correspondence (arrows)", st["show_fine_to_coarse"])
    if changed:
        st["show_fine_to_coarse"] = v
        _rebuild_fine_side(ps)

    psim.Separator()
    _struct_tree_ui(ps, psim)

    psim.End()


def _struct_tree_ui(ps, psim):
    """Per-struct_id checkbox tree, grouped by type (Seams / Boundaries)."""
    st = _state
    if psim.CollapsingHeader("Structures (by struct_id)"):
        for type_id, label in ((SEAM_TYPE_ID, "Seams"), (BOUNDARY_TYPE_ID, "Boundaries")):
            group = [e for e in st["struct_list"] if e[1] == type_id]
            if not group:
                continue
            if psim.TreeNode(f"{label} ({len(group)})"):
                if psim.Button(f"All##{label}"):
                    for sid, _, _, _ in group:
                        st["struct_vis"][sid] = True
                    _rebuild_fine_side(ps)
                    _rebuild_coarse_side(ps)
                psim.SameLine()
                if psim.Button(f"None##{label}"):
                    for sid, _, _, _ in group:
                        st["struct_vis"][sid] = False
                    _rebuild_fine_side(ps)
                    _rebuild_coarse_side(ps)

                for sid, _, length, count in group:
                    cur = bool(st["struct_vis"][sid])
                    changed, v = psim.Checkbox(
                        f"struct {sid}  (len={length:.3g}, n={count})##sid{sid}", cur)
                    if changed:
                        st["struct_vis"][sid] = v
                        _rebuild_fine_side(ps)
                        _rebuild_coarse_side(ps)
                psim.TreePop()


def run_viz(edge_samples_path, bundle_path, z_offset, stride=STRIDE_DEFAULT, show_gui=True):
    samples = load_edge_samples(edge_samples_path)
    print(f"[edge_sample_viz] loading bundle: {bundle_path}")
    bundle = load_c2f_bundle(bundle_path)
    print(f"[edge_sample_viz] fine:   {bundle.fineV.shape[0]} verts  {bundle.fineF.shape[0]} faces")
    print(f"[edge_sample_viz] coarse: {bundle.coarseV.shape[0]} verts  {bundle.coarseF.shape[0]} faces")

    g2c = build_global_to_compact(bundle)
    fine_pos, coarse_pos, valid = compute_positions(samples, bundle, g2c)

    is_seam     = (samples["type_id"] == SEAM_TYPE_ID) & valid
    is_boundary = (samples["type_id"] == BOUNDARY_TYPE_ID) & valid
    print(f"[edge_sample_viz] {int(is_seam.sum())} seam samples, "
          f"{int(is_boundary.sum())} boundary samples (valid/plotted)")

    if not show_gui:
        print("[edge_sample_viz] --no-gui: skipping polyscope window")
        return

    import polyscope as ps

    local_idx, struct_len, length_ref = compute_length_weighting(samples, bundle.fineV)
    print(f"[edge_sample_viz] struct arc-length reference (median): {length_ref:.6g}")

    struct_list = build_struct_list(samples, struct_len)
    print(f"[edge_sample_viz] {len(struct_list)} distinct struct_ids "
          f"({sum(1 for e in struct_list if e[1] == SEAM_TYPE_ID)} seam, "
          f"{sum(1 for e in struct_list if e[1] == BOUNDARY_TYPE_ID)} boundary)")
    max_sid = int(samples["struct_id"].max()) + 1
    struct_vis = np.ones(max_sid, dtype=bool)  # all structs visible by default

    mesh_span = float((bundle.fineV.max(axis=0) - bundle.fineV.min(axis=0)).max())
    _state.update(bundle=bundle, fine_pos=fine_pos, coarse_pos=coarse_pos,
                  is_seam=is_seam, is_boundary=is_boundary,
                  z_offset=z_offset, mesh_span=mesh_span if mesh_span > 1e-10 else 1.0,
                  stride=max(1, int(stride)),
                  show_seam=True, show_boundary=True,
                  show_coarse_to_fine=True, show_fine_to_coarse=False,
                  length_weighted=False, local_idx=local_idx,
                  struct_len=struct_len, length_ref=length_ref,
                  struct_id=samples["struct_id"], struct_list=struct_list,
                  struct_vis=struct_vis)

    ps.init()
    ps.set_program_name("Edge Sample Tracker (seam/boundary correspondence)")
    ps.set_up_dir("z_up")

    _rebuild_coarse_side(ps)
    _rebuild_fine_side(ps)

    ps.set_user_callback(_ui_callback)

    print("[edge_sample_viz] window ready — close it to exit.")
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edge_samples", help="path to edge_samples_<stem>.txt")
    ap.add_argument("bundle", nargs="?", default=None,
                    help="path to correspondence_<stem>.c2f "
                         "(default: alongside edge_samples, same stem)")
    ap.add_argument("--z-offset", type=float, default=Z_OFFSET_DEFAULT,
                    help=f"Z separation between fine (top) and coarse (bottom) mesh "
                         f"(default {Z_OFFSET_DEFAULT})")
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT,
                    help=f"show every Nth sample (default {STRIDE_DEFAULT}); "
                         f"also adjustable live via the UI")
    ap.add_argument("--no-gui", action="store_true",
                    help="load + compute + print stats, skip opening the polyscope window")
    args = ap.parse_args()

    bundle_path = args.bundle or default_bundle_path(args.edge_samples)
    if not os.path.isfile(bundle_path):
        print(f"ERROR: bundle not found: {bundle_path}", file=sys.stderr)
        sys.exit(1)

    run_viz(args.edge_samples, bundle_path, args.z_offset, stride=args.stride,
           show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
