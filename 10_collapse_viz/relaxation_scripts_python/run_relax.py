"""Run the explicit relaxation of the subdivided coarse mesh from a C++ run's
bundle (.c2f) and the fine .ma_struct, with the same flags as
collapse_viz_bin (main.cpp):

  python run_relax.py --bundle <run>/correspondence_<stem>.c2f \
      --matstruct_path <...>.ma_struct --output_dir <dir> \
      --coarse_subdiv_relax_method explicit [--coarse_subdiv_relax_no_new_folds] ...

Relaxation flags (as in main.cpp):
  --coarse_subdiv_relax_method explicit   (the only ported method; C++ default: newton)
  --coarse_subdiv_relax_per_face          hold vertices on coarse vertices / edges at their seeds
  --coarse_subdiv_relax_no_new_folds      no step may fold a triangle that is not folded
  --coarse_subdiv_relax_local_proj        newton only in C++ (only adds _local to the file name)
  --coarse_subdiv_relax_joint             newton only in C++ (makes the graph symmetric, adds _joint)
  --coarse_subdiv_relax_joint_solve       solve_project only in C++ (warning, adds _jointsolve)
  --explicit_lambda L                     default 0.5
  --explicit_max_iter N                   default 20000
  --explicit_tol T                        default 1e-7 (x diag)
  --explicit_global_proj                  global closest point instead of local
  --explicit_directed_graph               curves pulled only by curve / junction neighbours
  --coarse_subdiv_relax_weights W         uniform | cotan | meanvalue
  --n_coarse_subdiv_samples N             default 200000 (= the runs' --n_subdiv_samples)
  --subdiv_obj_max_verts N                default 2000000
Python-only extras:
  --stem S                  output stem (default: from the bundle name, correspondence_<stem>.c2f)
  --simplified_obj PATH     check the bundle's coarse mesh against this simplified_*.obj
  --log_every N             progress line every N iterations (C++: 100)
  --snapshot_iters a,b,...  OBJ snapshots at these iterations (C++: 1,10,100,1000,10000)
  --no_subdiv_objs          do not write coarse_subdiv_*.obj / coarse_subdiv_at_fine_pos_*.obj
  --description TEXT        written to experiment_config.txt
  --equal_area_levels K     equal-area mode (equal_area_refine.py): split the big coarse faces
                            (longest-edge bisection) into about |F| * 4^K faces of similar area
                            before the uniform subdivision (0 = off, the C++ behaviour)
  --equal_area_target A     same, with the target face area A given directly
"""
import argparse
import dataclasses
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import log_util  # noqa: E402
from bundle_io import load_bundle_flat  # noqa: E402
from c2f_walk import coarse_subdiv_c2f_build  # noqa: E402
from equal_area_refine import refine_from_args  # noqa: E402
from coarse_subdiv_relax import CoarseSubdivRelaxConfig, coarse_subdiv_relax_export, relaxed_obj_prefix  # noqa: E402
from matstruct import load_matstruct  # noqa: E402
from obj_io import read_obj, write_obj  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', required=True)
    p.add_argument('--matstruct_path', required=True)
    p.add_argument('--output_dir', default='.')
    p.add_argument('--coarse_subdiv_relax_method', default='newton')
    p.add_argument('--coarse_subdiv_relax_per_face', action='store_true')
    p.add_argument('--coarse_subdiv_relax_no_new_folds', action='store_true')
    p.add_argument('--coarse_subdiv_relax_local_proj', action='store_true')
    p.add_argument('--coarse_subdiv_relax_joint', action='store_true')
    p.add_argument('--coarse_subdiv_relax_joint_solve', action='store_true')
    p.add_argument('--coarse_subdiv_relax_max_iter', type=int, default=-1)
    p.add_argument('--subdiv_relax_anchor_tol', type=float, default=3e-3)
    p.add_argument('--explicit_lambda', type=float, default=0.5)
    p.add_argument('--explicit_max_iter', type=int, default=20000)
    p.add_argument('--explicit_tol', type=float, default=1e-7)
    p.add_argument('--explicit_global_proj', action='store_true')
    p.add_argument('--explicit_directed_graph', action='store_true')
    p.add_argument('--coarse_subdiv_relax_weights', default='uniform')
    p.add_argument('--n_coarse_subdiv_samples', type=int, default=200000)
    p.add_argument('--subdiv_obj_max_verts', type=int, default=2000000)
    p.add_argument('--stem', default=None)
    p.add_argument('--simplified_obj', default=None)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--snapshot_iters', default='1,10,100,1000,10000')
    p.add_argument('--no_subdiv_objs', action='store_true')
    p.add_argument('--description', default='')
    p.add_argument('--equal_area_levels', type=int, default=0)
    p.add_argument('--equal_area_target', type=float, default=-1.0)
    return p.parse_args(argv)


def build_c2f(B, a):
    """Subdivided coarse mesh + c2f walk; with the equal-area mode, from the refined coarse mesh."""
    R = refine_from_args(B.coarseV, B.coarseF, a.equal_area_levels, a.equal_area_target)
    return coarse_subdiv_c2f_build(B, a.n_coarse_subdiv_samples, refined=R)


def equal_area_text(a):
    if a.equal_area_target > 0:
        return 'on, target face area %g' % a.equal_area_target
    if a.equal_area_levels > 0:
        return 'on, levels %d (about |F| x %d coarse faces of similar area)' % (a.equal_area_levels,
                                                                                 4 ** a.equal_area_levels)
    return 'off'


def config_from_args(a):
    c = CoarseSubdivRelaxConfig()
    c.method = a.coarse_subdiv_relax_method
    c.curveAnchorTol = a.subdiv_relax_anchor_tol
    c.maxIter = a.coarse_subdiv_relax_max_iter
    c.perCoarseFace = a.coarse_subdiv_relax_per_face
    c.noNewFolds = a.coarse_subdiv_relax_no_new_folds
    c.localProjection = a.coarse_subdiv_relax_local_proj
    c.jointPass = a.coarse_subdiv_relax_joint
    c.jointSolve = a.coarse_subdiv_relax_joint_solve
    c.explicitLambda = a.explicit_lambda
    c.explicitMaxIter = a.explicit_max_iter
    c.explicitTol = a.explicit_tol
    c.explicitGlobalProj = a.explicit_global_proj
    c.explicitDirected = a.explicit_directed_graph
    c.weights = a.coarse_subdiv_relax_weights
    c.logEvery = a.log_every
    c.snapshotIters = tuple(int(x) for x in a.snapshot_iters.split(',') if x.strip())
    return c


def write_clamp_csv(C, path):
    with open(path, 'w', newline='\n') as f:
        f.write('vid,mapped,steps,clamped_steps,far_steps,max_neg_bary,sum_neg_bary,max_snap_rel\n')
        for i in range(C.S.V.shape[0]):
            f.write('%d,%d,%d,%d,%d,%.9g,%.9g,%.9g\n' % (i, 1 if C.fineFace[i] >= 0 else 0, C.walkSteps[i],
                                                         C.clampedSteps[i], C.farSteps[i], C.maxNegBary[i],
                                                         C.sumNegBary[i], C.maxSnapRel[i]))


def main(argv=None):
    a = parse_args(argv)
    cfg = config_from_args(a)
    out_dir = a.output_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = a.stem
    if stem is None:
        base = os.path.basename(a.bundle)
        stem = base[len('correspondence_'):] if base.startswith('correspondence_') else base
        stem = stem[:-4] if stem.endswith('.c2f') else stem
    out = lambda prefix, ext: os.path.join(out_dir, prefix + stem + ext)
    objPath = out(relaxed_obj_prefix(cfg), '.obj')
    logPath = os.path.join(out_dir, 'run.log')
    logf = open(logPath, 'w', newline='\n')
    log_util.set_log_file(logf)
    log = log_util.log

    started = datetime.datetime.now()
    cmd = 'python ' + ' '.join([os.path.relpath(__file__)] + (argv if argv is not None else sys.argv[1:]))
    with open(os.path.join(out_dir, 'experiment_config.txt'), 'w', newline='\n') as f:
        f.write('experiment:   %s\n' % os.path.basename(os.path.normpath(out_dir)))
        f.write('description:  %s\n' % a.description)
        f.write('started:      %s\n' % started.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('implementation: python (relaxation_scripts_python)\n')
        f.write('bundle:       %s\n' % a.bundle)
        f.write('matstruct:    %s\n' % a.matstruct_path)
        f.write('n_coarse_subdiv_samples: %d\n' % a.n_coarse_subdiv_samples)
        f.write('relaxation:   method %s, lambda %g, max_iter %d, tol %g x diag, %s projection, %s graph, '
                '%s weights, no-new-folds %s, per coarse face %s\n'
                % (cfg.method, cfg.explicitLambda, cfg.explicitMaxIter, cfg.explicitTol,
                   'global' if cfg.explicitGlobalProj else 'local',
                   'directed' if cfg.explicitDirected and not cfg.jointPass else 'symmetric',
                   cfg.weights, 'on' if cfg.noNewFolds else 'off', 'on' if cfg.perCoarseFace else 'off'))
        f.write('output:       %s , log %s\n' % (out_dir, logPath))
        f.write('command:      %s\n' % cmd)

    t0 = time.perf_counter()
    B = load_bundle_flat(a.bundle)
    if a.simplified_obj:
        V, F = read_obj(a.simplified_obj)
        okF = F.shape == B.coarseF.shape and np.array_equal(F, B.coarseF)
        okV = V.shape == B.coarseV.shape and all(
            '%.15g' % x == '%.15g' % y for x, y in zip(V.ravel().tolist(), B.coarseV.ravel().tolist()))
        log('[run_relax] simplified mesh %s vs bundle coarse mesh: faces %s, vertices %s (15 digits)'
            % (a.simplified_obj, 'identical' if okF else 'DIFFER', 'identical' if okV else 'DIFFER'))
        if not (okF and okV):
            raise SystemExit('simplified mesh does not match the bundle')
    ms = load_matstruct(a.matstruct_path, B.fineV, B.fineF)
    C = build_c2f(B, a)
    if not a.no_subdiv_objs and C.S.V.shape[0] <= a.subdiv_obj_max_verts:
        write_obj(out('coarse_subdiv_', '.obj'), C.S.V, C.S.F)
        write_obj(out('coarse_subdiv_at_fine_pos_', '.obj'), C.P, C.S.F)
        log('[coarse_subdiv] subdivided coarse mesh -> %s' % out('coarse_subdiv_', '.obj'))
        log('[coarse_subdiv] subdivided coarse mesh at fine correspondences -> %s' % out('coarse_subdiv_at_fine_pos_', '.obj'))
    write_clamp_csv(C, out('coarse_subdiv_c2f_clamp_', '.csv'))
    status = 0
    try:
        M, R, q0, q1 = coarse_subdiv_relax_export(B, C, ms, cfg, a.subdiv_obj_max_verts, objPath, log)
        result = [
            '[relax_explicit] %s after %d iterations (last max move %.3g x diag) | move max %.3g mean %.3g (x diag) | '
            'folded %d -> %d, moves held back %d' % ('converged' if R.converged else 'NOT CONVERGED', R.itersSheet,
                                                     R.deltaSheet, R.maxMove, R.meanMove, R.foldedSeed,
                                                     R.foldedResult, R.foldReverts),
            'quality before -> after: edge CV %.4f -> %.4f | min angle p1 %.3f -> %.3f, p5 %.3f -> %.3f, median '
            '%.3f -> %.3f deg | degenerate %d -> %d | flipped vs seed %d'
            % (q0.edgeCV, q1.edgeCV, q0.p1, q1.p1, q0.p5, q1.p5, q0.median, q1.median, q0.degenerate,
               q1.degenerate, q1.flippedVsRef)]
        res = dict(folder=os.path.basename(os.path.normpath(out_dir)), config=dataclasses.asdict(cfg),
                   equal_area=equal_area_text(a), n_subdiv_verts=int(C.S.V.shape[0]),
                   n_subdiv_faces=int(C.S.F.shape[0]),
                   report=dataclasses.asdict(R), quality_before=dataclasses.asdict(q0),
                   quality_after=dataclasses.asdict(q1), seconds=time.perf_counter() - t0, relaxed_obj=objPath)
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(res, f, indent=1, default=float)
    except Exception as e:  # noqa: BLE001
        log('[run_relax] FAILED: %s' % e)
        result = ['FAILED: %s' % e]
        status = 1
    with open(os.path.join(out_dir, 'experiment_config.txt'), 'a', newline='\n') as f:
        f.write('finished:     %s (exit code %d, %.1f s)\n'
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), status, time.perf_counter() - t0))
        f.write('result lines:\n')
        for s in result:
            f.write('  ' + s + '\n')
    log_util.set_log_file(None)
    logf.close()
    return status


if __name__ == '__main__':
    sys.exit(main())
