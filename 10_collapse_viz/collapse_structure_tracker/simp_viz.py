"""
simp_viz.py  —  Polyscope visualizer for MAT simplification results.

Usage:
    python simp_viz.py <initial_mat_mesh> <simp_info_json>
                       [--simplified <simplified_mat.obj>]
                       [--surface    <surface_mesh.obj>]

Shows:
    • initial_mat      — original MAT mesh (wire + surface)
    • simplified_mat   — simplified MAT mesh (if provided)
    • surface          — original fine triangle mesh at half opacity (if provided)
    • simp_vertices    — simplified MAT vertices as a point cloud colored by topo type
      → click any point to highlight its original ancestors (gold) on the initial MAT

Dependencies:  pip install polyscope numpy
"""

import sys
import argparse
import json
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


# ── mesh loaders ────────────────────────────────────────────────────────────

def load_off(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    idx = 0
    if lines[idx].upper() == 'OFF':
        idx += 1
    nv, nf, _ = map(int, lines[idx].split()); idx += 1
    V = np.array([list(map(float, lines[idx + i].split())) for i in range(nv)], dtype=float)
    idx += nv
    F = []
    for i in range(nf):
        row = list(map(int, lines[idx + i].split()))
        n = row[0]
        F.append(row[1:1 + n])
    F = np.array(F, dtype=int)
    return V[:, :3], F


def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                V.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                # handle "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                vids = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                if len(vids) == 3:
                    F.append(vids)
    return np.array(V, dtype=float), np.array(F, dtype=int)


def load_mesh(path):
    if path.lower().endswith('.off'):
        return load_off(path)
    return load_obj(path)


# ── topo type colors ─────────────────────────────────────────────────────────

TOPO_RGB = {
    0: (100/255, 149/255, 237/255),  # MS_Sheet              cornflower blue
    1: (255/255, 165/255,   0/255),  # MS_Seam               orange
    2: (220/255,  20/255,  60/255),  # MS_Boundary           crimson
    3: (148/255,   0/255, 211/255),  # MS_Junction           violet
    4: (  0/255, 200/255, 100/255),  # MS_Sheet_Boundary     green
    5: (255/255, 215/255,   0/255),  # MS_Seam_Boundary      gold
    6: (255/255,  69/255,   0/255),  # MS_Junction_Boundary  orange-red
    7: (160/255, 160/255, 160/255),  # MS_Unknown            grey
}
TOPO_NAMES = {
    0: 'MS_Sheet', 1: 'MS_Seam', 2: 'MS_Boundary', 3: 'MS_Junction',
    4: 'MS_Sheet_Boundary', 5: 'MS_Seam_Boundary', 6: 'MS_Junction_Boundary',
    7: 'MS_Unknown',
}
GOLD = np.array([[1.0, 0.85, 0.1]])


# ── main ────────────────────────────────────────────────────────────────────

_OUT = (
    r"C:\Users\alirz\Projects\Graphics\Neural QMAT\external\surf_subgrid_SSP_orig"
    r"\10_collapse_viz\output\01_00040057_f8f78dbd17414efda75bc437_trimesh_000"
)
_STEM = "01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial"

DEFAULT_INITIAL_MAT = (
    r"C:\Users\alirz\Projects\Graphics\QMAT_old working version  exe file"
    r"\qmat_x64\qmat\output\01_00040057_f8f78dbd17414efda75bc437_trimesh_000"
    rf"\{_STEM}.off"
)
DEFAULT_JSON       = rf"{_OUT}\{_STEM}_simp_visualize_info.json"
DEFAULT_SIMPLIFIED = rf"{_OUT}\simplified_{_STEM}.obj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('initial_mat',  nargs='?', default=DEFAULT_INITIAL_MAT,
                    help='Initial MAT mesh (.off or .obj)')
    ap.add_argument('json',         nargs='?', default=DEFAULT_JSON,
                    help='*_simp_visualize_info.json')
    ap.add_argument('--simplified', default=DEFAULT_SIMPLIFIED,
                    help='Simplified MAT mesh (.obj)')
    ap.add_argument('--surface',    default=None,
                    help='Original fine surface mesh (.obj/.off) — shown at half opacity')
    args = ap.parse_args()

    if len(sys.argv) == 1:
        print('[simp_viz] No args — using defaults:')
        print(f'  initial_mat: {args.initial_mat}')
        print(f'  json:        {args.json}')
        print(f'  simplified:  {args.simplified}')

    # ── load JSON ──
    with open(args.json) as f:
        data = json.load(f)
    verts = data['vertices']
    N = len(verts)
    print(f'[simp_viz] {N} simplified vertices loaded from JSON')

    simp_pos    = np.array([v['position']           for v in verts], dtype=float)
    topo_types  = np.array([v['topo_type']           for v in verts], dtype=int)
    ancestors   = [v['original_ancestors']            for v in verts]
    struct_ids  = [v['struct_ids']                    for v in verts]

    topo_colors = np.array([TOPO_RGB.get(t, (0.5, 0.5, 0.5)) for t in topo_types])

    # ── load meshes ──
    V_mat, F_mat = load_mesh(args.initial_mat)
    print(f'[simp_viz] initial MAT: {V_mat.shape[0]} verts, {F_mat.shape[0]} faces')

    V_simp, F_simp = (None, None)
    if args.simplified:
        V_simp, F_simp = load_mesh(args.simplified)
        print(f'[simp_viz] simplified MAT: {V_simp.shape[0]} verts, {F_simp.shape[0]} faces')

    V_surf, F_surf = (None, None)
    if args.surface:
        V_surf, F_surf = load_mesh(args.surface)
        print(f'[simp_viz] surface mesh: {V_surf.shape[0]} verts, {F_surf.shape[0]} faces')

    # ── polyscope ──
    ps.init()
    ps.set_ground_plane_mode('none')
    ps.set_up_dir('neg_z_up')

    # original fine surface — half opacity
    if V_surf is not None:
        surf_ps = ps.register_surface_mesh('surface', V_surf, F_surf)
        surf_ps.set_smooth_shade(True)
        surf_ps.set_transparency(0.5)
        surf_ps.set_color((0.8, 0.8, 0.8))

    # initial MAT
    mat_ps = ps.register_surface_mesh('initial_mat', V_mat, F_mat)
    mat_ps.set_smooth_shade(False)
    mat_ps.set_color((0.6, 0.7, 0.9))
    mat_ps.set_edge_width(1.0)

    # simplified MAT surface
    if V_simp is not None:
        simp_mesh_ps = ps.register_surface_mesh('simplified_mat', V_simp, F_simp)
        simp_mesh_ps.set_smooth_shade(False)
        simp_mesh_ps.set_color((0.9, 0.9, 0.5))
        simp_mesh_ps.set_edge_width(1.5)

    # simplified vertices point cloud — colored by topo type
    pc = ps.register_point_cloud('simp_vertices', simp_pos)
    pc.add_color_quantity('topo_type', topo_colors, enabled=True)
    pc.set_radius(0.006, relative=False)

    # ancestor cloud — starts hidden
    anc_pc = ps.register_point_cloud('ancestors', np.zeros((1, 3)))
    anc_pc.set_color(GOLD[0])
    anc_pc.set_radius(0.009, relative=False)
    anc_pc.set_enabled(False)

    # ── state ──
    state = {
        'show_ancestors': True,
        'picked_idx':     -1,
        'last_sel':       None,
        'info_text':      '',
    }

    def callback():
        s = state

        # Show Ancestors toggle
        changed, s['show_ancestors'] = psim.Checkbox('Show Ancestors on Click',
                                                      s['show_ancestors'])
        if changed and not s['show_ancestors']:
            anc_pc.set_enabled(False)

        # Pick detection
        if s['show_ancestors'] and ps.have_selection():
            sel = ps.get_selection()
            sel_key = (sel.structure_name, sel.local_index)
            if sel_key != s['last_sel']:
                s['last_sel'] = sel_key
                name = sel.structure_name
                idx  = sel.local_index
                if name == 'simp_vertices' and 0 <= idx < N:
                    s['picked_idx'] = idx
                    anc = ancestors[idx]
                    pts = V_mat[anc] if len(anc) > 0 else np.zeros((1, 3))
                    # re-register with new points
                    anc_cloud = ps.register_point_cloud('ancestors', pts)
                    anc_cloud.set_color(GOLD[0])
                    anc_cloud.set_radius(0.009, relative=False)
                    anc_cloud.set_enabled(True)

                    tt   = topo_types[idx]
                    sids = struct_ids[idx]
                    s['info_text'] = (
                        f'simp_vertex[{idx}]\n'
                        f'Topo: {TOPO_NAMES.get(tt, str(tt))}\n'
                        f'Struct IDs ({len(sids)}): {sids}\n'
                        f'Ancestors: {len(anc)} original vertices'
                    )

        # Info panel
        if s['picked_idx'] >= 0:
            psim.Separator()
            for line in s['info_text'].split('\n'):
                psim.TextUnformatted(line)
            if psim.Button('Clear'):
                s['picked_idx'] = -1
                s['info_text']  = ''
                s['last_sel']   = None
                anc_pc.set_enabled(False)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == '__main__':
    main()
