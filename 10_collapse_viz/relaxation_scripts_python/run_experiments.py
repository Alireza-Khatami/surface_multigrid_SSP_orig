"""Relaxation experiments with the Python port: runs a list of schemes (graph,
weights, folds rule, step, projection, per-face) on one C++ run's bundle and
writes a results table.

  python run_experiments.py [--set default] [--only name1,name2] [--table_only]

Each experiment goes to output/relaxation_experiments/<name>/ (with
experiment_config.txt, run.log, results.json); the table (one row per
experiment, folder column first) to
output/relaxation_experiments/py_experiments_results.{csv,md}.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_relax  # noqa: E402

EXP_ROOT = os.path.normpath(os.path.join(HERE, '..', 'output', 'relaxation_experiments'))
SOURCE_RUN = os.path.join(EXP_ROOT, 'clamp_check')  # C++ run whose bundle all experiments start from
MS = ('D:/datasets/abc_full_10k/out_ABC_v6_knn_poission40_20_15_10/'
      '01_00040057_f8f78dbd17414efda75bc437_trimesh_000/mat/'
      'mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.ma_struct')

E = ['--coarse_subdiv_relax_method', 'explicit']
NF = ['--coarse_subdiv_relax_no_new_folds']
IT = ['--explicit_max_iter', '2000']

# name -> (description, flags)
SETS = {
    'default': [
        ('py_nofold_2k', 'baseline: uniform weights, symmetric graph, no new folds, lambda 0.5, local projection',
         E + NF + IT),
        ('py_nofold_global_2k', 'as baseline, global closest-point projection instead of local',
         E + NF + IT + ['--explicit_global_proj']),
        ('py_nofold_perface_2k', 'as baseline, vertices on coarse vertices / edges held at their seeds',
         E + NF + IT + ['--coarse_subdiv_relax_per_face']),
        ('py_cotan_nofold_dir_2k', 'cotangent weights, directed graph, no new folds',
         E + NF + IT + ['--coarse_subdiv_relax_weights', 'cotan', '--explicit_directed_graph']),
        ('py_mv_nofold_dir_2k', 'mean-value weights, directed graph, no new folds',
         E + NF + IT + ['--coarse_subdiv_relax_weights', 'meanvalue', '--explicit_directed_graph']),
        ('py_mv_nofold_lam1_2k', 'mean-value weights, symmetric graph, no new folds, lambda 1',
         E + NF + IT + ['--coarse_subdiv_relax_weights', 'meanvalue', '--explicit_lambda', '1']),
    ],
}

COLS = [('folder', lambda r: r['folder']),
        ('weights', lambda r: r['config']['weights']),
        ('graph', lambda r: 'directed' if r['config']['explicitDirected'] and not r['config']['jointPass'] else 'symmetric'),
        ('no_new_folds', lambda r: int(r['config']['noNewFolds'])),
        ('per_face', lambda r: int(r['config']['perCoarseFace'])),
        ('projection', lambda r: 'global' if r['config']['explicitGlobalProj'] else 'local'),
        ('lambda', lambda r: '%g' % r['config']['explicitLambda']),
        ('iters', lambda r: r['report']['itersSheet']),
        ('converged', lambda r: int(r['report']['converged'])),
        ('last_move', lambda r: '%.3g' % r['report']['deltaSheet']),
        ('move_max', lambda r: '%.3g' % r['report']['maxMove']),
        ('move_mean', lambda r: '%.3g' % r['report']['meanMove']),
        ('folded_seed', lambda r: r['report']['foldedSeed']),
        ('folded_result', lambda r: r['report']['foldedResult']),
        ('held_back', lambda r: r['report']['foldReverts']),
        ('edge_cv', lambda r: '%.4f -> %.4f' % (r['quality_before']['edgeCV'], r['quality_after']['edgeCV'])),
        ('min_angle_p1', lambda r: '%.3f -> %.3f' % (r['quality_before']['p1'], r['quality_after']['p1'])),
        ('min_angle_p5', lambda r: '%.3f -> %.3f' % (r['quality_before']['p5'], r['quality_after']['p5'])),
        ('min_angle_median', lambda r: '%.3f -> %.3f' % (r['quality_before']['median'], r['quality_after']['median'])),
        ('degenerate', lambda r: r['quality_after']['degenerate']),
        ('flipped_vs_seed', lambda r: r['quality_after']['flippedVsRef']),
        ('seconds', lambda r: '%.0f' % r['seconds'])]


def write_table(folders):
    rows = []
    for d in folders:
        p = os.path.join(EXP_ROOT, d, 'results.json')
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    if not rows:
        return
    head = [c for c, _ in COLS]
    vals = [[str(f(r)) for _, f in COLS] for r in rows]
    with open(os.path.join(EXP_ROOT, 'py_experiments_results.csv'), 'w', newline='\n') as f:
        f.write(','.join(head) + '\n')
        for v in vals:
            f.write(','.join('"%s"' % x if ',' in x else x for x in v) + '\n')
    with open(os.path.join(EXP_ROOT, 'py_experiments_results.md'), 'w', newline='\n') as f:
        f.write('| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head) + '\n')
        for v in vals:
            f.write('| ' + ' | '.join(v) + ' |\n')
    print(open(os.path.join(EXP_ROOT, 'py_experiments_results.md')).read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', default='default')
    p.add_argument('--only', default='')
    p.add_argument('--table_only', action='store_true')
    a = p.parse_args()
    exps = SETS[a.set]
    if a.only:
        keep = set(a.only.split(','))
        exps = [e for e in exps if e[0] in keep]
    bundle = glob.glob(os.path.join(SOURCE_RUN, 'correspondence_*.c2f'))[0]
    if not a.table_only:
        for name, desc, flags in exps:
            out = os.path.join(EXP_ROOT, name)
            print('=== %s: %s' % (name, desc), flush=True)
            run_relax.main(['--bundle', bundle, '--matstruct_path', MS, '--output_dir', out,
                            '--no_subdiv_objs', '--description', desc] + flags)
    write_table([e[0] for e in SETS[a.set]])


if __name__ == '__main__':
    main()
