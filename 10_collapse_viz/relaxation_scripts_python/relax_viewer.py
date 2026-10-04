"""Step viewer for the explicit relaxation (spec: md_files/relaxation_step_visualizer.md).

Runs the relaxation one iteration at a time (ExplicitRelaxer, the code verified
against the C++) and shows what each iteration computed. Nothing about the
relaxation is computed here: every point, vector, mesh and BVH node shown comes
from the step's checkpoint (StepTrace), filled by the relaxation itself.

  python relax_viewer.py [--bundle ...] [--matstruct_path ...] [run_relax.py flags]

Defaults: ABC 00040057, the bundle of output/relaxation_experiments/clamp_check,
explicit method. The configuration can be changed between any two steps
(panel "Configuration", Apply): the run continues from the current positions.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import log_util  # noqa: E402
import run_relax  # noqa: E402
from bundle_io import load_bundle_flat  # noqa: E402
from coarse_subdiv_relax import CoarseRelaxSession  # noqa: E402
from matstruct import load_matstruct  # noqa: E402
from relax_explicit import StepTrace  # noqa: E402
from struct_ids import RELAX_CURVE, RELAX_JUNCTION  # noqa: E402

DEFAULT_RUN = os.path.normpath(os.path.join(HERE, '..', 'output', 'relaxation_experiments', 'clamp_check'))
DEFAULT_MS = ('D:/datasets/abc_full_10k/out_ABC_v6_knn_poission40_20_15_10/'
              '01_00040057_f8f78dbd17414efda75bc437_trimesh_000/mat/'
              'mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.ma_struct')
WEIGHTS = ['uniform', 'cotan', 'meanvalue']
METHODS = ['explicit', 'newton (not ported)', 'solve_project (not ported)']
ROLE_NAME = {0: 'sheet', 1: 'curve', 2: 'junction'}


# ------------------------------------------------------------------ point clouds

POINT_MATERIAL = 'flat'  # unlit: a point's colour does not change with the light / view


def register_pc(ps, name, points, **kw):
    """ps.register_point_cloud with the unlit material (every point cloud of the viewer)."""
    kw.setdefault('material', POINT_MATERIAL)
    return ps.register_point_cloud(name, points, **kw)


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


def concave_fine_edges(V, F, angle_deg):
    """Fine edges whose two faces meet at a concave dihedral angle larger than
    angle_deg. Normals made consistent across the edge (the second face's normal
    is flipped when both faces run the edge in the same direction); edges with
    other than two faces are skipped. Returns (edges (m,2), mask over origEdges order)."""
    F = np.asarray(F, dtype=np.int64)
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    opp = F[:, [2, 0, 1]].reshape(-1)
    fid = np.repeat(np.arange(F.shape[0]), 3)
    key = (np.minimum(a, b) << 32) | np.maximum(a, b)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[start, len(ks)])
    two = start[cnt == 2]
    h1, h2 = order[two], order[two + 1]
    n = np.cross(V[b] - V[a], V[opp] - V[a])
    nf = n[np.arange(0, len(a), 3)]
    nf = nf / np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-300)
    n1 = nf[fid[h1]]
    n2 = nf[fid[h2]] * np.where(a[h1] == a[h2], -1.0, 1.0)[:, None]  # same direction: inconsistent
    s = np.einsum('ij,ij->i', n1, V[opp[h2]] - V[a[h1]])
    bend = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2), -1.0, 1.0)))
    conc = (s > 0) & (bend > angle_deg)
    E = np.stack([np.minimum(a[h1], b[h1]), np.maximum(a[h1], b[h1])], axis=1)[conc]
    return E


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
        self.history = []          # (iteration, configuration text)
        self.rel = None
        self.make_relaxer(state=None)

        # display state
        rng = np.random.default_rng(12345)
        self.colors = rng.uniform(0.1, 0.95, size=(self.Vs, 3))  # fixed per point for the session
        self.sel = -1
        self.selAt = 0             # 0: x, 1: y, 2: Pi(y)
        self.show = dict(x=True, y=True, p=True, committed=True, vstep=True, vproj=True,
                         bvh_targets=True, bvh_visited=True, bvh_winner=True, region=True)
        self.meshOn = dict(committed=False, step=False, proj=False)
        self.colorMode = 0         # 0 random, 1 concave mask
        self.concK = 2
        self.concAngle = 10.0
        self.concEdges = None
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

    # -------------------------------------------------------------- polyscope structures
    def register(self):
        ps = self.ps
        ps.set_transparency_mode('simple')
        B = self.B
        for g in ('fine MAT', 'points', 'selection', 'step meshes', 'BVH (selected point)', 'BVH inspector'):
            ps.create_group(g)
        m = ps.register_surface_mesh('fine MAT', B.fineV, B.fineF, color=(0.8, 0.8, 0.8), transparency=0.3)
        m.add_to_group('fine MAT')
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
        """Points within k rings of a concave fine edge. Seeds: points whose current
        fine face (from the relaxation state) has a concave edge closer than the mean
        subdivided edge length; then k rings over the subdivided mesh."""
        B = self.B
        if self.concEdges is None or self.concEdges[0] != self.concAngle:
            E = concave_fine_edges(B.fineV, B.fineF, self.concAngle)
            self.concEdges = (self.concAngle, E)
            # per fine face corner c: is edge (F[f,c], F[f,c+1]) concave
            FO = B.fineF
            ck = np.sort((E[:, 0] << 32) | E[:, 1])
            self.concCorner = np.zeros((FO.shape[0], 3), bool)
            for c in range(3):
                a, b = FO[:, c], FO[:, (c + 1) % 3]
                k = (np.minimum(a, b) << 32) | np.maximum(a, b)
                j = np.clip(np.searchsorted(ck, k), 0, max(len(ck) - 1, 0))
                self.concCorner[:, c] = (len(ck) > 0) & (ck[j] == k) if len(ck) else False
            if len(E):
                nodes, edges = B.fineV, E
                self.ps.register_curve_network('concave fine edges', nodes, edges, color=(0.9, 0.1, 0.1),
                                               radius=0.002).add_to_group('fine MAT')
            elif self.ps.has_curve_network('concave fine edges'):
                self.ps.remove_curve_network('concave fine edges')
        X, face = self.rel.X, self.rel.face
        FO = B.fineF
        ok = face >= 0
        mask = np.zeros(self.Vs, bool)
        if not self.concCorner.any():
            return mask
        fc = np.where(ok, face, 0)
        Fe = self.F
        el = np.linalg.norm(X[Fe[:, 0]] - X[Fe[:, 1]], axis=1).mean()
        for c in range(3):
            a, b = FO[fc, c], FO[fc, (c + 1) % 3]
            isc = self.concCorner[fc, c] & ok
            idx = np.nonzero(isc)[0]
            d = seg_dist(X[idx], B.fineV[a[idx]], B.fineV[b[idx]])
            mask[idx[d <= el]] = True
        G = self.rel.G
        rows = np.repeat(np.arange(self.Vs), np.diff(G.rowOffs))
        for _ in range(self.concK):
            grow = np.zeros(self.Vs, bool)
            grow[rows[mask[G.cols]]] = True
            mask |= grow
        return mask

    # -------------------------------------------------------------- selection
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
                c.add_vector_quantity('step x -> y', (y - x0)[None], vectortype='ambient', enabled=True,
                                      color=(0.1, 0.4, 0.95))
        if sh['y']:
            c = self._add(G1, 'pc', 'sel: y (step)', register_pc(ps, 'sel: y (step)', y[None], radius=0.004,
                                                                           color=(0.1, 0.4, 0.95)))
            if sh['vproj']:
                c.add_vector_quantity('projection y -> Pi(y)', (p - y)[None], vectortype='ambient', enabled=True,
                                      color=(0.95, 0.5, 0.05))
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
                          ps.register_curve_network('sel: local search region (edges)', B.fineV, E,
                                                    color=(0.2, 0.8, 0.3), radius=0.002))
            else:
                self._add(G2, 'sm', 'sel: local search region (faces)',
                          ps.register_surface_mesh('sel: local search region (faces)', B.fineV, B.fineF[info['reg']],
                                                   color=(0.2, 0.8, 0.3), transparency=0.6))
        if sh['region']:
            if info['role'] == 'curve' and info['pedge'] >= 0:
                self._add(G2, 'cn', 'sel: result edge', ps.register_curve_network(
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
                    self._add(G2, 'cn', nm, ps.register_curve_network(nm, nodes, edges, color=(0.6, 0.6, 0.6),
                                                                      radius=0.0006, transparency=0.5))
        pj = proj.pj
        if sh['bvh_visited'] and info.get('visited') is not None and len(info['visited']):
            g = info['visited']
            nodes, edges = box_wire(pj[24][g], pj[25][g])
            self._add(G2, 'cn', 'sel: BVH nodes visited', ps.register_curve_network(
                'sel: BVH nodes visited', nodes, edges, color=(0.1, 0.4, 0.95), radius=0.001))
        if sh['bvh_winner'] and info['winTree'] >= 0:
            gnode = pj[23][info['winTree']] + info['winLeaf']
            nodes, edges = box_wire(pj[24][gnode], pj[25][gnode])
            self._add(G2, 'cn', 'sel: winning BVH leaf', ps.register_curve_network(
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
        c = ps.register_curve_network(nm, nodes, edges, radius=0.0006)
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
        if self.running:
            self.step(self.stepsPerFrame)
        r = self.rel

        if psim.CollapsingHeader('Run', psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Text('iteration %d | last max move %.3g x diag | %s'
                      % (self.iters, r.lastMove / r.diag,
                         'converged' if r.R.converged else ('done (max iterations)' if r.done else 'running' if self.running else 'stopped')))
            if psim.Button('Stop' if self.running else 'Run'):
                self.running = not self.running
            psim.SameLine()
            if psim.Button('Step'):
                self.step(1)
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
            if psim.Button('Apply (continue from the current positions)'):
                self.running = False
                self.apply_config()
            psim.Text('active since iteration %d: %s' % self.history[-1])
            if len(self.history) > 1 and psim.TreeNode('configuration history'):
                for it, txt in self.history:
                    psim.TextUnformatted('%6d  %s' % (it, txt))
                psim.TreePop()

        if psim.CollapsingHeader('Display', psim.ImGuiTreeNodeFlags_DefaultOpen):
            ch, self.colorMode = psim.Combo('point colours', self.colorMode, ['random (fixed per point)', 'concave mask'])
            ch1, self.concK = psim.SliderInt('concave mask: rings k', self.concK, 0, 10)
            ch2, self.concAngle = psim.SliderFloat('concave mask: min bend (deg)', self.concAngle, 0.0, 90.0)
            if ch or ch1 or ch2:
                self.apply_colors()
            for key, label in (('committed', 'mesh: committed (held back flagged)'), ('step', 'mesh: step y'),
                               ('proj', 'mesh: projection Pi(y)')):
                ch, self.meshOn[key] = psim.Checkbox(label, self.meshOn[key])
                if ch:
                    self.refresh()

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
    run = DEFAULT_RUN
    b = glob.glob(os.path.join(run, 'correspondence_*.c2f'))
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--total_max_iter', type=int, default=20000)
    pre.add_argument('--mock', action='store_true', help='headless polyscope backend (tests)')
    extra, rest = pre.parse_known_args(argv)
    if '--bundle' not in rest and b:
        rest = ['--bundle', b[0]] + rest
    if '--matstruct_path' not in rest:
        rest = ['--matstruct_path', DEFAULT_MS] + rest
    if '--coarse_subdiv_relax_method' not in rest:
        rest = ['--coarse_subdiv_relax_method', 'explicit'] + rest
    a = run_relax.parse_args(rest)
    a.total_max_iter = extra.total_max_iter
    a.mock = extra.mock
    cfg = run_relax.config_from_args(a)
    cfg.snapshotIters = ()
    return a, cfg


def main(argv=None):
    import polyscope as ps
    a, cfg = parse(argv)
    ps.init('openGL_mock' if a.mock else 'auto')
    ps.set_program_name('relaxation step viewer')
    ps.set_up_dir('z_up')
    v = RelaxViewer(a, cfg, ps)
    ps.set_user_callback(v.gui)
    if not a.mock:
        ps.show()
    return v


if __name__ == '__main__':
    main()
