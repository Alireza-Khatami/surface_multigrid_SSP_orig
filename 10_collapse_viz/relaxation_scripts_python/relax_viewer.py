"""Step viewer for the explicit relaxation (spec: md_files/relaxation_step_visualizer.md).

Runs the relaxation one iteration at a time (ExplicitRelaxer, the code verified
against the C++) and shows what each iteration computed. Nothing about the
relaxation is computed here: every point, vector, mesh and BVH node shown comes
from the step's checkpoint (StepTrace), filled by the relaxation itself.

  python relax_viewer.py [--bundle ...] [--matstruct_path ...] [run_relax.py flags]

Defaults: ABC 00040057, the bundle of output/relaxation_experiments/src_qslim200_valid
(decimation with validity checks; run_relax.DEFAULT_SOURCE_RUN),
explicit method. The configuration can be changed between any two steps
(panel "Configuration", Apply): the run continues from the current positions.
Panel "Export (PLY)": one button per mesh (relaxation input, committed, step y,
projection Pi(y)) and one for the stages before the relaxation, written to
--export_dir (default output/relaxation_experiments/viewer_exports/<date_time>).
"""
import datetime
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import log_util  # noqa: E402
import run_relax  # noqa: E402
from bundle_io import load_bundle_flat  # noqa: E402
from coarse_subdiv_relax import CoarseRelaxSession  # noqa: E402
from matstruct import load_matstruct  # noqa: E402
from concave_parts import compact, concave_curve_parts  # noqa: E402
from ply_io import write_ply  # noqa: E402
from relax_explicit import StepTrace  # noqa: E402
from relax_exports import export_relax_input, relax_vertex_props  # noqa: E402
from struct_ids import RELAX_CURVE, RELAX_JUNCTION  # noqa: E402

DEFAULT_EXPORT_ROOT = os.path.normpath(os.path.join(HERE, '..', 'output', 'relaxation_experiments',
                                                   'viewer_exports'))
DEFAULT_MS = run_relax.DEFAULT_MS  # bundle / .ma_struct defaults come from run_relax.parse_args
WEIGHTS = ['uniform', 'cotan', 'meanvalue']
METHODS = ['explicit', 'newton (not ported)', 'solve_project (not ported)']
ROLE_NAME = {0: 'sheet', 1: 'curve', 2: 'junction'}


# ------------------------------------------------------------------ point clouds

POINT_MATERIAL = 'flat'  # unlit: a point's colour does not change with the light / view


# Sizes stay the same on screen-independent terms: every radius below is given relative to
# the scene's length scale AT START (LEN0) and set as an absolute size, so the camera speed
# control (which changes the length scale) does not shrink points, curves or vectors.
LEN0 = [None]


def _abs(rel):
    return rel * LEN0[0]


def register_pc(ps, name, points, **kw):
    """ps.register_point_cloud with the unlit material and an absolute radius (every point
    cloud of the viewer)."""
    kw.setdefault('material', POINT_MATERIAL)
    r = kw.pop('radius', 0.005)
    pc = ps.register_point_cloud(name, points, **kw)
    if LEN0[0]:
        pc.set_radius(_abs(r), relative=False)
    return pc


def register_cn(ps, name, nodes, edges, **kw):
    """ps.register_curve_network with an absolute radius."""
    r = kw.pop('radius', 0.005)
    cn = ps.register_curve_network(name, nodes, edges, **kw)
    if LEN0[0]:
        cn.set_radius(_abs(r), relative=False)
    return cn


def add_ambient_vectors(pc, name, vecs, color, radius=0.0012):
    """Ambient vector quantity with an absolute radius (the Python wrapper only sets
    relative radii, so the bound call is used)."""
    import polyscope_bindings as psb
    from polyscope.common import glm3
    q = pc.bound_instance.add_vector_quantity(name, np.asfortranarray(np.asarray(vecs, dtype=np.float32)),
                                              psb.VectorType.ambient)
    q.set_radius(_abs(radius), False)
    q.set_color(glm3(color))
    q.set_enabled(True)
    return q


# ------------------------------------------------------------------ geometry helpers (display only)

_BOX_EDGES = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)


def box_wire(mn, mx):
    """Wire boxes (k boxes) as curve-network nodes / edges."""
    mn = np.asarray(mn).reshape(-1, 3)
    mx = np.asarray(mx).reshape(-1, 3)
    k = len(mn)
    c = np.empty((k, 8, 3))
    for i in range(8):
        c[:, i, 0] = np.where(i & 1, mx[:, 0], mn[:, 0])
        c[:, i, 1] = np.where(i & 2, mx[:, 1], mn[:, 1])
        c[:, i, 2] = np.where(i & 4, mx[:, 2], mn[:, 2])
    E = (_BOX_EDGES[None, :, :] + 8 * np.arange(k)[:, None, None]).reshape(-1, 2)
    return c.reshape(-1, 3), E


def tree_depths(tr):
    """Depth of every node of a projector tree (root 0)."""
    d = np.zeros(len(tr.nodes), dtype=np.int64)
    for i, nd in enumerate(tr.nodes):  # preorder: parents come first
        for ch in (nd[2], nd[3]):
            if ch >= 0:
                d[ch] = d[i] + 1
    return d


def seg_dist(P, A, B):
    AB = B - A
    l2 = np.einsum('ij,ij->i', AB, AB)
    t = np.clip(np.einsum('ij,ij->i', P - A, AB) / np.maximum(l2, 1e-300), 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1)


# ------------------------------------------------------------------ viewer

class RelaxViewer:
    def __init__(self, args, cfg, ps):
        self.ps = ps
        self.args = args
        t0 = time.perf_counter()
        B = self.B = load_bundle_flat(args.bundle)
        ms = self.ms = load_matstruct(args.matstruct_path, B.fineV, B.fineF)
        C = self.C = run_relax.build_c2f(B, args)
        sess = self.sess = CoarseRelaxSession(B, C, ms)
        self.cfg = cfg
        self.Vs = sess.Vs
        self.F = np.ascontiguousarray(sess.M.F)
        self.trace = StepTrace(self.Vs)
        self.hasTrace = False
        self.iters = 0
        self.history = []          # (iteration, configuration text) of the current run
        self.runId = 0             # Reset starts run 1, 2, ...
        self.pastRuns = []         # (run id, history, final iteration) of the earlier runs
        self.rel = None
        self.make_relaxer(state=None)
        self.keep_input()
        self.exportDir = None
        self.exported = []         # paths written this session

        # display state
        rng = np.random.default_rng(12345)
        self.colors = rng.uniform(0.1, 0.95, size=(self.Vs, 3))  # fixed per point for the session
        self.sel = -1
        self.selAt = 0             # 0: x, 1: y, 2: Pi(y)
        self.show = dict(x=True, y=True, p=True, committed=True, vstep=True, vproj=True,
                         bvh_targets=True, bvh_visited=True, bvh_winner=True, region=True)
        self.meshOn = dict(committed=False, step=False, proj=False)
        self.colorMode = 0         # 0 random, 1 concave mask
        self.concAngle = 20.0      # concave corner: sheet interior angle > 180 + this
        self.concCache = None      # ((margin, radius, k), mask)
        self.concRadius = 3.0      # in mean subdivided edge lengths (of the input)
        self.concK = 0
        self.fineSheet = np.array([ids[0] if ids else -1 for ids in ms.faceIds], dtype=np.int64)
        self.concParts = None
        Fm = self.F
        self.meshEdges = np.unique(np.sort(np.concatenate([Fm[:, [0, 1]], Fm[:, [1, 2]], Fm[:, [2, 0]]]), axis=1),
                                   axis=0)
        self.running = False
        self.stepsPerFrame = 1
        self.runTo = 0
        self.record = True
        self.bvhDepth = 64
        self.bvhLeaves = False
        self.bvhOn = {}            # tree -> shown
        self.treeDepth = [tree_depths(tr) for tr in sess.proj.trees]
        self.treeLabel = {}
        for i, t in sorted(sess.proj.sheetTree.items()):
            self.treeLabel[t] = 'sheet %d' % i
        for i, t in sorted(sess.proj.curveTree.items()):
            self.treeLabel[t] = 'curve %d' % i
        if sess.proj.globalTree >= 0:
            self.treeLabel[sess.proj.globalTree] = 'global (all faces)'
        # camera speed control: polyscope moves (scroll zoom, pan, first-person keys) by steps
        # proportional to the length scale; it is set every frame to k * d, d = distance
        # from the camera to the orbit centre (mode 0) or to the point under the cursor (mode 1)
        self.sceneCenter = None    # orbit centre fallback (polyscope < 2.5)
        self.cam = dict(on=True, mode=0, k=0.5, minf=1e-3, maxf=1.0, smooth=0.35, every=2, frame=0,
                        d=float('nan'), scale=None, far=20.0)
        self.ui = {}
        self.status = ''
        log_util.log('[relax_viewer] ready in %.1f s' % (time.perf_counter() - t0))
        self.register()

    # -------------------------------------------------------------- relaxation
    def cfg_text(self, c=None):
        c = c or self.cfg
        return ('lambda %g, tol %g, %s weights, %s graph, no-new-folds %s, %s projection, per-face %s'
                % (c.explicitLambda, c.explicitTol, c.weights, 'directed' if c.explicitDirected else 'symmetric',
                   'on' if c.noNewFolds else 'off', 'global' if c.explicitGlobalProj else 'local',
                   'on' if c.perCoarseFace else 'off'))

    def make_relaxer(self, state):
        """Relaxer for self.cfg, continuing from state (None: the seeds). All derived
        data (graph, weights, hold set, free vertices) is rebuilt by the original code."""
        cfg = self.cfg
        cfg.explicitMaxIter = max(0, self.args.total_max_iter - self.iters)
        self.rel = self.sess.make_relaxer(cfg, '', state=state, it0=self.iters)
        self.history.append((self.iters, self.cfg_text()))
        log_util.log('[relax_viewer] iteration %d: configuration %s' % (self.iters, self.cfg_text()))

    def keep_input(self):
        """The relaxation input of the current run: its relaxer at its initialization (copies)."""
        r0 = self.rel
        self.input0 = SimpleNamespace(X=r0.X.copy(), G=SimpleNamespace(role=r0.G.role.copy()),
                                      setId=r0.setId.copy(), isFree=r0.isFree.copy(), face=r0.face.copy())

    def reset(self):
        """Starts the relaxation again from its input (the seeds, iteration 0) under the
        current configuration. The seeds come from the session (steps 1-3, built once);
        a new relaxer is made for the configuration, as for the first run."""
        self.running = False
        self.runTo = 0
        self.pastRuns.append((self.runId, list(self.history), self.iters))
        self.runId += 1
        self.iters = 0
        self.hasTrace = False
        self.history = []
        log_util.log('[relax_viewer] reset: run %d starts from the relaxation input' % self.runId)
        self.make_relaxer(state=None)
        self.keep_input()
        self.concCache = None
        self.status = 'run %d: started again from the relaxation input' % self.runId
        self.refresh()

    def run_prefix(self):
        return '' if self.runId == 0 else 'run%02d_' % self.runId

    def apply_config(self):
        r = self.rel
        self.make_relaxer(state=(r.X, r.face, r.edge, r.bary))
        self.refresh()

    def step(self, n=1):
        for _ in range(n):
            if self.rel.done:
                self.running = False
                break
            tr = self.trace if self.record else None
            if tr is not None:
                tr.watch = self.sel
            self.rel.step(tr)
            self.iters += 1
            self.hasTrace = self.record
            if self.runTo and self.iters >= self.runTo:
                self.running = False
                self.runTo = 0
                break
        self.refresh()

    # -------------------------------------------------------------- camera speed
    CAM_MODES = ['distance to the orbit centre (C)', 'depth under the cursor (A)']

    def _view(self):
        """(camera position, orbit centre) from the view JSON, which every polyscope 2.x
        version has. Polyscope < 2.5 has no view centre (not in the JSON, no
        get_view_center): its turntable orbits the centre of the scene bounding box."""
        import json
        v = json.loads(self.ps.get_view_as_json())
        M = v.get('viewMat')
        if not M or any(x is None for x in M):
            return None, None
        M = np.asarray(M, dtype=float).reshape(4, 4)  # world -> camera, row-major
        R, t = M[:3, :3], M[:3, 3]
        pos = -R.T @ t
        if 'viewCenter' in v:
            ctr = np.asarray(v['viewCenter'], dtype=float)
        else:
            if self.sceneCenter is None:
                lo, hi = (np.asarray(x, dtype=float) for x in self.ps.get_bounding_box())
                self.sceneCenter = 0.5 * (lo + hi)
            ctr = self.sceneCenter
        return pos, ctr

    def camera_distance(self, psim=None):
        """d for the current mode; None when there is no reading this frame."""
        ps, c = self.ps, self.cam
        pos, ctr = self._view()
        if pos is None:
            return None
        if c['mode'] == 0:
            d = float(np.linalg.norm(ctr - pos))
            return d if np.isfinite(d) else None
        c['frame'] += 1
        if psim is None or c['frame'] % max(1, c['every']):
            return None
        io = psim.GetIO()
        if io.WantCaptureMouse:  # over the GUI
            return None
        mp = io.MousePos
        try:
            if hasattr(ps, 'pick'):
                r = ps.pick(screen_coords=(mp[0], mp[1]))
            else:  # older polyscope
                r = ps.pick_at_screen_coords((mp[0], mp[1]))
        except Exception:  # noqa: BLE001 (outside the window)
            return None
        if not r.is_hit:
            return None
        d = float(np.linalg.norm(r.position - pos))
        return d if np.isfinite(d) else None

    def set_far_clip(self, scale):
        """Keeps the far clip at 20 x the starting length scale (polyscope's far clip is a
        ratio of the current length scale)."""
        want = 20.0 * LEN0[0] / scale
        if abs(want - self.cam['far']) <= 0.01 * self.cam['far']:
            return
        try:
            import json
            v = json.loads(self.ps.get_view_as_json())
            if any(x is None for x in v.get('viewMat', [])):
                return  # no valid view (mock backend)
            v['farClip' if 'farClip' in v else 'farClipRatio'] = want  # name differs across versions
            self.ps.set_view_from_json(json.dumps(v))
            self.cam['far'] = want
        except Exception:  # noqa: BLE001
            pass

    def camera_speed(self, psim=None):
        """Called every frame: length scale = clamp(k d, min, max) x LEN0, smoothed."""
        c = self.cam
        L0 = LEN0[0]
        if not L0:
            return
        if not c['on']:
            target = L0
        else:
            d = self.camera_distance(psim)
            if d is not None:
                c['d'] = d
            if not np.isfinite(c['d']):
                return
            target = min(max(c['k'] * c['d'], c['minf'] * L0), c['maxf'] * L0)
        sc = c['scale'] + c['smooth'] * (target - c['scale'])
        if abs(sc - target) < 1e-4 * L0:
            sc = target
        if sc != c['scale']:
            c['scale'] = sc
            self.ps.set_length_scale(sc)
            self.set_far_clip(sc)

    # -------------------------------------------------------------- PLY export
    EXPORTS = (('input', 'relaxation input (iteration 0)'), ('committed', 'committed (current)'),
               ('step', 'step y (this iteration)'), ('proj', 'projection Pi(y) (this iteration)'))

    def export_root(self):
        """The export folder, with its experiment_config.txt (created on the first export)."""
        if self.exportDir is None:
            d = self.args.export_dir or os.path.join(
                DEFAULT_EXPORT_ROOT, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(d, exist_ok=True)
            a = self.args
            with open(os.path.join(d, 'experiment_config.txt'), 'w', newline='\n') as f:
                f.write('experiment:   %s\n' % os.path.basename(os.path.normpath(d)))
                f.write('description:  PLY exports from relax_viewer.py\n')
                f.write('started:      %s\n' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                f.write('implementation: python (relaxation_scripts_python/relax_viewer.py)\n')
                f.write('bundle:       %s\n' % a.bundle)
                f.write('matstruct:    %s\n' % a.matstruct_path)
                f.write('n_coarse_subdiv_samples: %d\n' % a.n_coarse_subdiv_samples)
                f.write('equal-area refinement: %s\n' % run_relax.equal_area_text(a))
                f.write('exports (iteration | configuration active then | file):\n')
            self.exportDir = d
        return self.exportDir

    def _log_export(self, it, cfgText, name):
        with open(os.path.join(self.export_root(), 'experiment_config.txt'), 'a', newline='\n') as f:
            f.write('  %6d | %s | %s\n' % (it, cfgText, name))

    def export_mesh(self, key):
        """Writes one mesh as PLY; the data is the relaxation's (its state or this
        iteration's checkpoint). Returns the path, or None when there is no data."""
        t, r = self.trace, self.rel
        held = t.held if self.hasTrace else None
        if key == 'input':
            X = self.input0.X
            vp = relax_vertex_props(self.sess, self.input0)
            name = self.run_prefix() + 'relax_input_it000000.ply'
        elif key == 'committed':
            X = r.X
            vp = relax_vertex_props(self.sess, r, held=held)
            name = self.run_prefix() + 'it%06d_committed.ply' % self.iters
        elif key in ('step', 'proj'):
            if not self.hasTrace:
                self.status = 'no checkpoint for this iteration: turn on "record checkpoints" and step'
                return None
            X = t.Y if key == 'step' else t.P
            vp = relax_vertex_props(self.sess, r, face=(None if key == 'step' else t.Pface), held=held)
            if key == 'step':
                del vp['fine_face']  # y is off the surface
            name = self.run_prefix() + 'it%06d_%s.ply' % (self.iters, 'step_y' if key == 'step' else 'projection')
        else:
            raise ValueError(key)
        cc = np.clip(np.round(self.colors * 255.0), 0, 255).astype(np.uint8)
        vp['red'], vp['green'], vp['blue'] = cc[:, 0], cc[:, 1], cc[:, 2]
        fp = dict(coarse_face=self.C.S.faceOrig.astype(np.int32))
        path = os.path.join(self.export_root(), name)
        it = 0 if key == 'input' else self.iters
        cfgText = self.history[0][1] if key == 'input' else self.cfg_text()
        if not write_ply(path, X, self.F, vp, fp, ['relax_viewer %s, iteration %d' % (key, it), cfgText]):
            self.status = 'could not write %s' % path
            return None
        self._log_export(it, cfgText, name)
        self.exported.append(path)
        self.status = 'exported %s' % path
        log_util.log('[relax_viewer] exported %s' % path)
        return path

    def export_stages(self):
        """The stages before the relaxation (coarse, equal-area, subdivided, c2f, input),
        as run_relax.py writes them."""
        d = os.path.join(self.export_root(), 'relax_input')
        paths = export_relax_input(d, '_' + run_relax.bundle_stem(self.args.bundle), self.B, self.C, self.sess, self.input0, 'relax_viewer')
        self._log_export(0, self.history[0][1], 'relax_input/ (%d files)' % len(paths))
        self.exported += paths
        self.status = 'exported %d stage files to %s' % (len(paths), d)
        return paths

    # -------------------------------------------------------------- polyscope structures
    def register(self):
        ps = self.ps
        ps.set_transparency_mode('simple')
        B = self.B
        for g in ('fine MAT', 'points', 'selection', 'step meshes', 'BVH (selected point)', 'BVH inspector'):
            ps.create_group(g)
        m = ps.register_surface_mesh('fine MAT', B.fineV, B.fineF, color=(0.8, 0.8, 0.8), transparency=0.3)
        m.add_to_group('fine MAT')
        LEN0[0] = ps.get_length_scale()  # sizes and the camera speed are relative to this
        ps.set_automatically_compute_scene_extents(False)
        self.cam['scale'] = LEN0[0]
        pc = register_pc(ps, 'points', self.rel.X, radius=0.0015)
        pc.add_color_quantity('random colour', self.colors, enabled=True)
        pc.add_to_group('points')
        self.pc = pc
        for key, name in (('committed', 'mesh: committed'), ('step', 'mesh: step y'), ('proj', 'mesh: projection Pi(y)')):
            sm = ps.register_surface_mesh(name, self.rel.X, self.F, enabled=False, edge_width=0.5)
            sm.add_to_group('step meshes')
        self.refresh()

    def mesh_positions(self, key):
        t, X = self.trace, self.rel.X
        if not self.hasTrace or key == 'committed':
            return X
        return t.Y if key == 'step' else t.P

    def refresh(self):
        ps = self.ps
        X = self.rel.X
        self.pc.update_point_positions(X)
        if self.hasTrace:
            self.pc.add_scalar_quantity('held back (this step)', self.trace.held.astype(float), enabled=False)
        self.apply_colors()
        for key, name in (('committed', 'mesh: committed'), ('step', 'mesh: step y'), ('proj', 'mesh: projection Pi(y)')):
            sm = ps.get_surface_mesh(name)
            sm.set_enabled(self.meshOn[key])
            if self.meshOn[key]:
                sm.update_vertex_positions(self.mesh_positions(key))
                if key == 'committed' and self.hasTrace:
                    sm.add_scalar_quantity('held back', self.trace.held.astype(float), enabled=True,
                                           cmap='reds')
        self.update_selection()

    def apply_colors(self):
        if self.colorMode == 0:
            self.pc.add_color_quantity('random colour', self.colors, enabled=True)
        else:
            mask = self.concave_mask()
            col = np.where(mask[:, None], np.array([0.9, 0.1, 0.1]), np.array([0.75, 0.75, 0.75]))
            self.pc.add_color_quantity('concave mask', col, enabled=True)
        if self.sel >= 0:
            t = np.full(self.Vs, 0.12)
            t[self.sel] = 1.0
            self.pc.add_scalar_quantity('selection transparency', t, enabled=False)
            self.pc.set_transparency_quantity('selection transparency')
        else:
            self.pc.clear_transparency_quantity()

    def concave_mask(self):
        """Subdivided vertices of the relaxation input (iteration 0) at the concave parts of
        the seams / boundaries (concave_parts.py): for a concave corner (v, s), the input
        points whose fine face belongs to sheet s and that lie within concRadius mean
        subdivided edge lengths of v; then k rings over the subdivided mesh. Fixed for the
        session; recomputed only when the margin, the radius or k change."""
        key = (self.concAngle, self.concRadius, self.concK)
        if self.concCache is not None and self.concCache[0] == key:
            return self.concCache[1]
        B, ps = self.B, self.ps
        P = self.concParts = concave_curve_parts(B.fineV, B.fineF, self.ms, self.concAngle)
        for name in ('concave seam / boundary parts', 'concave corners'):
            if name == 'concave seam / boundary parts' and ps.has_curve_network(name):
                ps.remove_curve_network(name)
            if name == 'concave corners' and ps.has_point_cloud(name):
                ps.remove_point_cloud(name)
        if len(P.edges):
            nodes, edges, _ = compact(B.fineV, P.edges)  # only the vertices the edges use
            cn = register_cn(ps, 'concave seam / boundary parts', nodes, edges, color=(0.9, 0.1, 0.1),
                                           radius=0.002, material=POINT_MATERIAL)
            cn.add_scalar_quantity('type (1 seam, 2 boundary)', P.edgeType.astype(float), defined_on='edges')
            cn.add_to_group('fine MAT')
        if len(P.corners):
            cv = np.unique(P.corners[:, 0])
            register_pc(ps, 'concave corners', B.fineV[cv, :3], radius=0.003,
                        color=(0.6, 0.0, 0.0)).add_to_group('fine MAT')
        X0, f0 = self.input0.X, self.input0.face
        ok = f0 >= 0
        pSheet = np.where(ok, self.fineSheet[np.where(ok, f0, 0)], -2)
        E0 = self.meshEdges
        r = self.concRadius * float(np.linalg.norm(X0[E0[:, 0]] - X0[E0[:, 1]], axis=1).mean())
        mask = np.zeros(self.Vs, dtype=bool)
        bySheet = {}
        for v, sh in P.corners.tolist():
            if sh not in bySheet:
                bySheet[sh] = np.nonzero(pSheet == sh)[0]
            idx = bySheet[sh]
            d = np.linalg.norm(X0[idx] - B.fineV[v, :3], axis=1)
            mask[idx[d <= r]] = True
        E = self.meshEdges
        for _ in range(self.concK):
            grow = mask.copy()
            grow[E[mask[E[:, 1]], 0]] = True
            grow[E[mask[E[:, 0]], 1]] = True
            mask = grow
        self.concCache = (key, mask)
        return mask
    def select(self, i):
        self.sel = int(i)
        self.trace.watch = self.sel
        self.apply_colors()
        self.update_selection()

    def clear_selection(self):
        self.sel = -1
        self.trace.watch = -1
        self.remove_group_structures('selection')
        self.remove_group_structures('BVH (selected point)')
        self.apply_colors()

    def remove_group_structures(self, prefix):
        ps = self.ps
        for name in list(self.ui.get(prefix, [])):
            kind, nm = name
            if kind == 'pc' and ps.has_point_cloud(nm):
                ps.remove_point_cloud(nm)
            elif kind == 'cn' and ps.has_curve_network(nm):
                ps.remove_curve_network(nm)
            elif kind == 'sm' and ps.has_surface_mesh(nm):
                ps.remove_surface_mesh(nm)
        self.ui[prefix] = []

    def _add(self, group, kind, name, obj):
        obj.add_to_group(group)
        self.ui.setdefault(group, []).append((kind, name))
        return obj

    def selection_info(self):
        """What the last step computed for the selected point (from the checkpoint)."""
        i, t, r = self.sel, self.trace, self.rel
        if i < 0:
            return None
        k = int(r.setId[i])
        pal = self.sess.pal
        info = dict(i=i, role=ROLE_NAME[int(r.G.role[i])], free=bool(r.isFree[i]), set=k,
                    ids=pal.ids[pal.offsets[k]:pal.offsets[k + 1]], targets=list(self.sess.proj.targets[k]),
                    face=int(r.face[i]), edge=int(r.edge[i]), bary=r.bary[i].copy(), X=r.X[i].copy())
        if self.hasTrace:
            info.update(it=t.it, x0=t.X0[i].copy(), y=t.Y[i].copy(), p=t.P[i].copy(), x1=t.X1[i].copy(),
                        pface=int(t.Pface[i]), pedge=int(t.Pedge[i]), pbary=t.Pbary[i].copy(),
                        held=bool(t.held[i]), regCnt=int(t.regCnt[i]),
                        reg=t.reg[i, :max(0, min(int(t.regCnt[i]), t.reg.shape[1]))].copy(),
                        winTree=int(t.winTree[i]), winLeaf=int(t.winLeaf[i]), local=t.local,
                        watched=(t.watch == i), visited=t.watchVis[:int(t.watchN[0])].copy() if t.watch == i else None)
        return info

    def update_selection(self):
        ps = self.ps
        self.remove_group_structures('selection')
        self.remove_group_structures('BVH (selected point)')
        info = self.selection_info()
        if info is None:
            return
        G1, G2 = 'selection', 'BVH (selected point)'
        sh = self.show
        col = self.colors[info['i']]
        if not self.hasTrace:
            self._add(G1, 'pc', 'sel: point', register_pc(ps, 'sel: point', info['X'][None], radius=0.006,
                                                                    color=col))
            return
        x0, y, p, x1 = info['x0'], info['y'], info['p'], info['x1']
        at = (x0, y, p)[self.selAt]
        self._add(G1, 'pc', 'sel: point', register_pc(ps, 'sel: point', at[None], radius=0.007, color=col))
        if sh['x']:
            c = self._add(G1, 'pc', 'sel: x (start)', register_pc(ps, 'sel: x (start)', x0[None], radius=0.004,
                                                                            color=(0.2, 0.2, 0.2)))
            if sh['vstep']:
                add_ambient_vectors(c, 'step x -> y', (y - x0)[None], (0.1, 0.4, 0.95))
        if sh['y']:
            c = self._add(G1, 'pc', 'sel: y (step)', register_pc(ps, 'sel: y (step)', y[None], radius=0.004,
                                                                           color=(0.1, 0.4, 0.95)))
            if sh['vproj']:
                add_ambient_vectors(c, 'projection y -> Pi(y)', (p - y)[None], (0.95, 0.5, 0.05))
        if sh['p']:
            self._add(G1, 'pc', 'sel: Pi(y) (projection)',
                      register_pc(ps, 'sel: Pi(y) (projection)', p[None], radius=0.004, color=(0.95, 0.5, 0.05)))
        if sh['committed'] and info['held']:
            self._add(G1, 'pc', 'sel: committed (held back)',
                      register_pc(ps, 'sel: committed (held back)', x1[None], radius=0.005, color=(0.9, 0.1, 0.1)))
        # BVH / search region of this point in this step
        proj = self.sess.proj
        B = self.B
        if sh['region'] and info['local'] and info['regCnt'] >= 0 and len(info['reg']):
            if info['role'] == 'curve':
                E = proj.E[info['reg']]
                self._add(G2, 'cn', 'sel: local search region (edges)',
                          register_cn(ps, 'sel: local search region (edges)', B.fineV, E,
                                                    color=(0.2, 0.8, 0.3), radius=0.002))
            else:
                self._add(G2, 'sm', 'sel: local search region (faces)',
                          ps.register_surface_mesh('sel: local search region (faces)', B.fineV, B.fineF[info['reg']],
                                                   color=(0.2, 0.8, 0.3), transparency=0.6))
        if sh['region']:
            if info['role'] == 'curve' and info['pedge'] >= 0:
                self._add(G2, 'cn', 'sel: result edge', register_cn(ps,
                    'sel: result edge', B.fineV, proj.E[[info['pedge']]], color=(0.95, 0.5, 0.05), radius=0.003))
            elif info['pface'] >= 0:
                self._add(G2, 'sm', 'sel: result face', ps.register_surface_mesh(
                    'sel: result face', B.fineV, B.fineF[[info['pface']]], color=(0.95, 0.5, 0.05)))
        if sh['bvh_targets']:
            for t in info['targets']:
                tr = proj.trees[t]
                d = self.treeDepth[t]
                keep = [j for j, nd in enumerate(tr.nodes) if d[j] <= self.bvhDepth and (not self.bvhLeaves or nd[2] < 0)]
                if keep:
                    nodes, edges = box_wire([tr.nodes[j][0] for j in keep], [tr.nodes[j][1] for j in keep])
                    nm = 'sel: target tree %s' % self.treeLabel.get(t, t)
                    self._add(G2, 'cn', nm, register_cn(ps, nm, nodes, edges, color=(0.6, 0.6, 0.6),
                                                                      radius=0.0006, transparency=0.5))
        pj = proj.pj
        if sh['bvh_visited'] and info.get('visited') is not None and len(info['visited']):
            g = info['visited']
            nodes, edges = box_wire(pj[24][g], pj[25][g])
            self._add(G2, 'cn', 'sel: BVH nodes visited', register_cn(ps,
                'sel: BVH nodes visited', nodes, edges, color=(0.1, 0.4, 0.95), radius=0.001))
        if sh['bvh_winner'] and info['winTree'] >= 0:
            gnode = pj[23][info['winTree']] + info['winLeaf']
            nodes, edges = box_wire(pj[24][gnode], pj[25][gnode])
            self._add(G2, 'cn', 'sel: winning BVH leaf', register_cn(ps,
                'sel: winning BVH leaf', nodes, edges, color=(0.95, 0.5, 0.05), radius=0.002))

    def handle_pick(self):
        ps = self.ps
        if not ps.have_selection():
            return
        r = ps.get_selection()
        idx = -1
        if r.structure_name in ('points',):
            idx = r.local_index
        elif r.structure_name in ('mesh: committed', 'mesh: step y', 'mesh: projection Pi(y)'):
            if r.structure_data.get('element_type') == 'vertex':
                idx = r.structure_data.get('index', -1)
        ps.reset_selection()
        if 0 <= idx < self.Vs and idx != self.sel:
            self.select(idx)

    # -------------------------------------------------------------- BVH inspector
    def set_tree(self, t, on):
        ps = self.ps
        nm = 'bvh: %s' % self.treeLabel.get(t, t)
        self.bvhOn[t] = on
        if ps.has_curve_network(nm):
            ps.remove_curve_network(nm)
        if not on:
            return
        tr = self.sess.proj.trees[t]
        d = self.treeDepth[t]
        keep = [j for j, nd in enumerate(tr.nodes) if d[j] <= self.bvhDepth and (not self.bvhLeaves or nd[2] < 0)]
        if not keep:
            return
        nodes, edges = box_wire([tr.nodes[j][0] for j in keep], [tr.nodes[j][1] for j in keep])
        c = register_cn(ps, nm, nodes, edges, radius=0.0006)
        c.add_scalar_quantity('depth', np.repeat(d[keep], 12).astype(float), defined_on='edges', enabled=True,
                              cmap='viridis')
        c.add_to_group('BVH inspector')

    def tree_errors(self, t):
        from test_projector import structure_errors
        return structure_errors(self.sess.proj.trees[t], self.sess.proj.VO)

    # -------------------------------------------------------------- UI
    def gui(self):
        import polyscope.imgui as psim
        self.handle_pick()
        if self.cam['on']:
            try:
                self.camera_speed(psim)
            except Exception as e:  # noqa: BLE001 (polyscope API differences): switch it off, keep the viewer
                self.cam['on'] = False
                self.cam['error'] = '%s: %s' % (type(e).__name__, e)
                log_util.log('[relax_viewer] camera speed control switched off: ' + self.cam['error'])
        if self.running:
            self.step(self.stepsPerFrame)
        r = self.rel

        if psim.CollapsingHeader('Run', psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Text('run %d' % self.runId + ' (Reset: start again from the relaxation input with the configuration below)')
            psim.Text('iteration %d | last max move %.3g x diag | %s'
                      % (self.iters, r.lastMove / r.diag,
                         'converged' if r.R.converged else ('done (max iterations)' if r.done else 'running' if self.running else 'stopped')))
            if psim.Button('Stop' if self.running else 'Run'):
                self.running = not self.running
            psim.SameLine()
            if psim.Button('Step'):
                self.step(1)
            psim.SameLine()
            if psim.Button('Reset'):
                self.reset()
            psim.SameLine()
            _, self.stepsPerFrame = psim.SliderInt('steps / frame', self.stepsPerFrame, 1, 50)
            _, self.runTo = psim.InputInt('run until iteration (0: off)', self.runTo)
            if psim.Button('Run until'):
                self.running = self.runTo > self.iters
            _, self.record = psim.Checkbox('record checkpoints (needed for the step data)', self.record)
            if psim.Button('Metrics'):
                self.status = ('energy %.6g, folded %d (from the relaxation)'
                               % (r.energy(r.X), r.count_folded(r.X)))
            if self.status:
                psim.Text(self.status)

        if psim.CollapsingHeader('Configuration', psim.ImGuiTreeNodeFlags_DefaultOpen):
            c = self.cfg
            psim.Combo('method', 0, METHODS)
            psim.Text('(only the explicit method is ported)')
            ch, w = psim.Combo('weights', WEIGHTS.index(c.weights), WEIGHTS)
            c.weights = WEIGHTS[w]
            _, c.explicitDirected = psim.Checkbox('directed graph', c.explicitDirected)
            _, c.noNewFolds = psim.Checkbox('no new folds', c.noNewFolds)
            _, c.explicitGlobalProj = psim.Checkbox('global projection', c.explicitGlobalProj)
            _, c.perCoarseFace = psim.Checkbox('per coarse face (hold coarse vertices / edges)', c.perCoarseFace)
            _, c.explicitLambda = psim.InputFloat('lambda', c.explicitLambda)
            _, c.explicitTol = psim.InputFloat('tol (x diag)', c.explicitTol, format='%.2e')
            _, self.args.total_max_iter = psim.InputInt('max iterations (total)', self.args.total_max_iter)
            if psim.Button('Reset with this configuration (from iteration 0)'):
                self.reset()
            if psim.Button('Apply (continue from the current positions)'):
                self.running = False
                self.apply_config()
            psim.Text('active since iteration %d: %s' % self.history[-1])
            if len(self.history) > 1 and psim.TreeNode('configuration history'):
                for it, txt in self.history:
                    psim.TextUnformatted('%6d  %s' % (it, txt))
                psim.TreePop()
            if self.pastRuns and psim.TreeNode('earlier runs (before Reset)'):
                for rid, hist, last in self.pastRuns:
                    psim.TextUnformatted('run %d, stopped at iteration %d:' % (rid, last))
                    for it, txt in hist:
                        psim.TextUnformatted('  %6d  %s' % (it, txt))
                psim.TreePop()

        if psim.CollapsingHeader('Display', psim.ImGuiTreeNodeFlags_DefaultOpen):
            ch, self.colorMode = psim.Combo('point colours', self.colorMode, ['random (fixed per point)', 'concave mask'])
            ch1, self.concK = psim.SliderInt('concave mask: rings k', self.concK, 0, 10)
            ch2, self.concAngle = psim.SliderFloat('concave corner: interior angle > 180 + (deg)', self.concAngle,
                                                   0.0, 90.0)
            ch3, self.concRadius = psim.SliderFloat('concave mask: radius (subdiv edges)', self.concRadius, 0.5, 30.0)
            if ch or ch1 or ch2 or ch3:
                self.apply_colors()
            for key, label in (('committed', 'mesh: committed (held back flagged)'), ('step', 'mesh: step y'),
                               ('proj', 'mesh: projection Pi(y)')):
                ch, self.meshOn[key] = psim.Checkbox(label, self.meshOn[key])
                if ch:
                    self.refresh()

        if psim.CollapsingHeader('Camera speed', psim.ImGuiTreeNodeFlags_DefaultOpen):
            c = self.cam
            _, c['on'] = psim.Checkbox('slow down near things (camera speed control)', c['on'])
            _, c['mode'] = psim.Combo('distance measured to', c['mode'], self.CAM_MODES)
            _, c['k'] = psim.SliderFloat('speed per unit distance k', c['k'], 0.01, 2.0, format='%.3f')
            _, c['minf'] = psim.InputFloat('min speed (x start)', c['minf'], format='%.5f')
            c['minf'] = min(max(c['minf'], 1e-5), 1.0)
            _, c['maxf'] = psim.SliderFloat('max speed (x start)', c['maxf'], 0.01, 4.0, format='%.2f')
            _, c['smooth'] = psim.SliderFloat('smoothing', c['smooth'], 0.05, 1.0)
            if c['mode'] == 1:
                _, c['every'] = psim.SliderInt('pick every n frames', c['every'], 1, 10)
            if self.cam.get('error'):
                psim.TextUnformatted('switched off after an error: ' + self.cam['error'])
            psim.Text('distance %.4g | speed %.3g x start' % (c['d'], c['scale'] / LEN0[0] if LEN0[0] else 1.0))
            if psim.Button('Reset speed'):
                c['scale'] = LEN0[0]
                self.ps.set_length_scale(LEN0[0])
                self.set_far_clip(LEN0[0])

        if psim.CollapsingHeader('Export (PLY)', psim.ImGuiTreeNodeFlags_DefaultOpen):
            for key, label in self.EXPORTS:
                if psim.Button('Export ' + label):
                    self.export_mesh(key)
            if psim.Button('Export all four'):
                for key, _ in self.EXPORTS:
                    self.export_mesh(key)
            if psim.Button('Export the stages before the relaxation (coarse ... input)'):
                self.export_stages()
            psim.TextUnformatted('folder: %s' % (self.exportDir or self.args.export_dir
                                                 or 'viewer_exports/<date_time> (made on the first export)'))

        if psim.CollapsingHeader('Selected point', psim.ImGuiTreeNodeFlags_DefaultOpen):
            info = self.selection_info()
            if info is None:
                psim.Text('click a point (or a vertex of a step mesh)')
            else:
                if psim.Button('Clear selection'):
                    self.clear_selection()
                    info = None
            if info is not None:
                ch = False
                for key, label in (('x', 'start x'), ('y', 'step y'), ('p', 'projection Pi(y)'),
                                   ('committed', 'committed (if held back)'), ('vstep', 'vector x -> y'),
                                   ('vproj', 'vector y -> Pi(y)'), ('region', 'local search region / result'),
                                   ('bvh_targets', 'BVH: target trees'), ('bvh_visited', 'BVH: nodes visited'),
                                   ('bvh_winner', 'BVH: winning leaf')):
                    c1, self.show[key] = psim.Checkbox(label, self.show[key])
                    ch |= c1
                psim.Text('selected point drawn at:')
                for j, lab in enumerate(('x', 'y', 'Pi(y)')):
                    psim.SameLine()
                    if psim.RadioButton(lab, self.selAt == j):
                        self.selAt = j
                        ch = True
                if ch:
                    self.update_selection()
                psim.TextUnformatted(self.info_text(info))

        if psim.CollapsingHeader('BVH inspector'):
            if psim.Button('Show all'):
                for t in self.treeLabel:
                    self.set_tree(t, True)
            psim.SameLine()
            if psim.Button('Hide all'):
                for t in self.treeLabel:
                    self.set_tree(t, False)
            ch1, self.bvhDepth = psim.SliderInt('max depth', self.bvhDepth, 0, 64)
            ch2, self.bvhLeaves = psim.Checkbox('leaves only', self.bvhLeaves)
            if ch1 or ch2:
                for t, on in list(self.bvhOn.items()):
                    if on:
                        self.set_tree(t, True)
                self.update_selection()
            for title, mapping in (('sheets', self.sess.proj.sheetTree), ('curves', self.sess.proj.curveTree)):
                if psim.TreeNode('%s (%d trees)' % (title, len(mapping))):
                    for sid, t in sorted(mapping.items()):
                        tr = self.sess.proj.trees[t]
                        lab = '%s: %d %s, %d nodes, depth %d##%d' % (
                            self.treeLabel[t], len(tr.gid), 'edges' if title == 'curves' else 'faces', len(tr.nodes),
                            int(self.treeDepth[t].max()) if len(tr.nodes) else 0, t)
                        ch, on = psim.Checkbox(lab, self.bvhOn.get(t, False))
                        if ch:
                            self.set_tree(t, on)
                    psim.TreePop()
            if psim.Button('Check every tree (boxes nest, leaves contain primitives)'):
                errs = {self.treeLabel[t]: self.tree_errors(t) for t in self.treeLabel}
                bad = {k: v for k, v in errs.items() if v}
                self.status = 'BVH check: %d trees, %s' % (len(errs), 'all correct' if not bad else 'errors in %s' % bad)

    def info_text(self, info):
        f = lambda v: '(%.6g, %.6g, %.6g)' % tuple(v)
        s = ['vertex %d: %s, %s, struct ids %s, target trees %s'
             % (info['i'], info['role'], 'free' if info['free'] else 'fixed', info['ids'],
                [self.treeLabel.get(t, t) for t in info['targets']])]
        if 'it' not in info:
            s.append('no step recorded yet: position %s, fine face %d' % (f(info['X']), info['face']))
            return '\n'.join(s)
        d = self.rel.diag
        s += ['step %d (%s projection):' % (info['it'], 'local' if info['local'] else 'global'),
              '  x      %s' % f(info['x0']),
              '  y      %s   |y - x| = %.3g x diag' % (f(info['y']), np.linalg.norm(info['y'] - info['x0']) / d),
              '  Pi(y)  %s   |Pi(y) - y| = %.3g x diag' % (f(info['p']), np.linalg.norm(info['p'] - info['y']) / d),
              '         fine face %d, edge %d, bary %s' % (info['pface'], info['pedge'], f(info['pbary'])),
              '  committed %s%s' % (f(info['x1']), '  (HELD BACK by no-new-folds)' if info['held'] else '')]
        if info['local']:
            s.append('  local search region: %s' % (('%d %s' % (info['regCnt'], 'edges' if info['role'] == 'curve' else 'faces'))
                                                   if info['regCnt'] >= 0 else 'none (fell back to global)'))
        if info['winTree'] >= 0:
            s.append('  result from BVH %s, leaf node %d' % (self.treeLabel.get(info['winTree']), info['winLeaf']))
        elif not info['local'] or info['regCnt'] < 0:
            s.append('  result: the current location (no BVH primitive was closer)')
        if info['visited'] is None:
            s.append('  BVH nodes visited: recorded from the next step on (point is now watched)')
        else:
            s.append('  BVH nodes visited: %d' % len(info['visited']))
        return '\n'.join(s)


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--total_max_iter', type=int, default=20000)
    pre.add_argument('--mock', action='store_true', help='headless polyscope backend (tests)')
    pre.add_argument('--export_dir', default=None, help='PLY export folder (default: viewer_exports/<date_time>)')
    extra, rest = pre.parse_known_args(argv)
    if '--coarse_subdiv_relax_method' not in rest:
        rest = ['--coarse_subdiv_relax_method', 'explicit'] + rest
    a = run_relax.parse_args(rest)
    a.total_max_iter = extra.total_max_iter
    a.mock = extra.mock
    a.export_dir = extra.export_dir
    cfg = run_relax.config_from_args(a)
    cfg.snapshotIters = ()
    return a, cfg


def main(argv=None):
    import polyscope as ps
    a, cfg = parse(argv)
    ps.init('openGL_mock' if a.mock else 'auto')
    log_util.log('[relax_viewer] python %s, polyscope %s (%s)' % (sys.version.split()[0], getattr(ps, '__version__', '?'),
                                                               os.path.dirname(ps.__file__)))
    ps.set_program_name('relaxation step viewer')
    ps.set_up_dir('z_up')
    ps.set_background_color((0.0, 0.0, 0.0))  # black background
    ps.set_ground_plane_mode('none')  # the tiled ground plane would cover it
    v = RelaxViewer(a, cfg, ps)
    ps.set_user_callback(v.gui)
    if not a.mock:
        ps.show()
    return v


if __name__ == '__main__':
    main()
