"""Run the Python relaxation on a C++ experiment folder's bundle and compare
with that run's outputs:

  * coarse_subdiv_*.obj and coarse_subdiv_at_fine_pos_*.obj   byte for byte
  * every relaxed snapshot *_it<N>.obj both runs wrote          byte for byte
  * the final relaxed OBJ (when the same number of iterations)  byte for byte
  * the log lines of the relaxation (graph, seed snap, struct IDs, weights,
    per-iteration progress, final report, quality), with the timings removed

  python verify_relax.py --cpp_dir ../output/relaxation_experiments/nofold \
      --out_dir <dir> -- --coarse_subdiv_relax_method explicit --coarse_subdiv_relax_no_new_folds \
      --explicit_max_iter 100

Everything after "--" goes to run_relax.py (same flags as the C++ run; use a
smaller --explicit_max_iter to compare a prefix of the run).
"""
import argparse
import filecmp
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_relax  # noqa: E402

DEFAULT_MS = ('D:/datasets/abc_full_10k/out_ABC_v6_knn_poission40_20_15_10/'
              '01_00040057_f8f78dbd17414efda75bc437_trimesh_000/mat/'
              'mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.ma_struct')

# log lines compared (prefix match), timings stripped
COMPARED = ('[coarse_subdiv] ', '[coarse_subdiv_relax] coarse struct IDs', '[subdiv_struct_ids]',
            '[coarse_subdiv_relax] seed snap', '[coarse_subdiv_relax] 0 vertices', '[coarse_subdiv_relax] per coarse',
            '[coarse_subdiv_relax] cotangent', '[coarse_subdiv_relax] mean-value', '[subdiv_relax] graph',
            '[subdiv_relax]   kept', '[relax_explicit]', '[coarse_subdiv_relax] relaxation (')
SKIPPED = ('[coarse_subdiv] subdivided coarse mesh', '[coarse_subdiv] c2f clamp statistics per vertex')


def norm_line(s):
    s = re.sub(r'\(\d+\.\d+ s\)', '(T s)', s)          # timings
    # fields that older C++ builds did not log (compared only through the other lines)
    s = re.sub(r' \| projection: [^|]*?\(x diag\)', '', s)          # per-iteration projection stats
    s = re.sub(r' \| (uniform|given) weights( \([^)]*\))?', '', s)  # weights in the start line
    return s.rstrip()


def line_key(s):
    """Which line this is, to pair C++ and python lines whatever the log format."""
    m = re.match(r'\[relax_explicit\] iter (\d+):', s)
    if m:
        return 'iter ' + m.group(1)
    for pat, key in ((r'\[relax_explicit\] \d+ free', 'start'),
                     (r'\[relax_explicit\] (NOT CONVERGED|converged)', 'final'),
                     (r'\[relax_explicit\]   projection over all', 'projection totals'),
                     (r'\[relax_explicit\]   projection:', 'projection'),
                     (r'\[relax_explicit\]   checks', 'checks')):
        if re.match(pat, s):
            return key
    return re.split(r'\d', s, maxsplit=1)[0]


def relevant(lines, maxIter):
    """The coarse-subdivision part of a log: from the first "[coarse_subdiv] "
    line on (the C++ log has the fine-mesh tracker's lines before it)."""
    out = []
    start = next((i for i, s in enumerate(lines) if s.startswith('[coarse_subdiv] ')), len(lines))
    for s in lines[start:]:
        s = s.rstrip('\r\n')
        if not s.startswith(COMPARED) or s.startswith(SKIPPED):
            continue
        m = re.match(r'\[relax_explicit\] iter (\d+):', s)
        if m and int(m.group(1)) > maxIter:
            continue
        out.append(norm_line(s))
    return out


def main():
    argv = sys.argv[1:]
    fwd = []
    if '--' in argv:
        k = argv.index('--')
        argv, fwd = argv[:k], argv[k + 1:]
    p = argparse.ArgumentParser()
    p.add_argument('--cpp_dir', required=True)
    p.add_argument('--cpp_log', default=None, help='default: <cpp_dir>.log')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--matstruct_path', default=DEFAULT_MS)
    p.add_argument('--skip_run', action='store_true', help='compare an existing out_dir')
    a = p.parse_args(argv)
    cpp = os.path.normpath(a.cpp_dir)
    cpp_log = a.cpp_log or cpp + '.log'
    bundle = glob.glob(os.path.join(cpp, 'correspondence_*.c2f'))[0]
    simp = glob.glob(os.path.join(cpp, 'simplified_*.obj'))
    stem = os.path.basename(bundle)[len('correspondence_'):-4]

    if not a.skip_run:
        args = ['--bundle', bundle, '--matstruct_path', a.matstruct_path, '--output_dir', a.out_dir] + fwd
        if simp:
            args += ['--simplified_obj', simp[0]]
        if run_relax.main(args) != 0:
            print('VERIFY: python run failed')
            return 1
    ra = run_relax.parse_args(['--bundle', bundle, '--matstruct_path', a.matstruct_path] + fwd)
    cfg = run_relax.config_from_args(ra)

    ok = True
    report = []
    # OBJs
    pairs = [('coarse_subdiv_' + stem + '.obj',) * 2, ('coarse_subdiv_at_fine_pos_' + stem + '.obj',) * 2]
    relaxed = run_relax.relaxed_obj_prefix(cfg) + stem
    for f in sorted(glob.glob(os.path.join(a.out_dir, relaxed + '_it*.obj'))):
        pairs.append((os.path.basename(f),) * 2)
    cpp_lines = open(cpp_log, encoding='utf-8', errors='replace').read().splitlines()
    total = [int(m.group(1)) for s in cpp_lines for m in [re.search(r'after (\d+) iterations', s)] if m]
    cpp_iters = total[0] if total else -1
    if cpp_iters == cfg.explicitMaxIter:
        pairs.append((relaxed + '.obj',) * 2)
    for cname, pname in pairs:
        cp, pp = os.path.join(cpp, cname), os.path.join(a.out_dir, pname)
        if not os.path.exists(cp):
            report.append('  (no C++ file %s)' % cname)
            continue
        if not os.path.exists(pp):
            report.append('  MISSING python file %s' % pname)
            ok = False
            continue
        same = filecmp.cmp(cp, pp, shallow=False)
        ok &= same
        report.append('  %-9s %s' % ('identical' if same else 'DIFFERENT', cname))

    # logs
    maxIter = cfg.explicitMaxIter
    L1 = relevant(cpp_lines, maxIter)
    L2 = relevant(open(os.path.join(a.out_dir, 'run.log')).read().splitlines(), maxIter)
    if cpp_iters != maxIter:  # prefix run: the final report / quality lines differ by design
        cut = lambda L: [s for s in L if not re.match(r'\[relax_explicit\] (NOT CONVERGED|converged)|'
                                                         r'\[relax_explicit\]   |\[coarse_subdiv_relax\] relaxation \(', s)]
        L1, L2 = cut(L1), cut(L2)
        mx = lambda L: [re.sub(r'max \d+ iterations', 'max N iterations', s) for s in L]
        L1, L2 = mx(L1), mx(L2)
    # c2f clamp line: only in C++ runs built after the clamp statistics were added
    if not any(s.startswith('[coarse_subdiv] c2f clamp') for s in L1):
        L2 = [s for s in L2 if not s.startswith('[coarse_subdiv] c2f clamp')]
    def keyed(L):
        d, cnt = {}, {}
        for s in L:
            k = line_key(s)
            cnt[k] = cnt.get(k, 0) + 1
            d['%s #%d' % (k, cnt[k])] = s
        return d
    K1, K2 = keyed(L1), keyed(L2)
    nl = 0
    for k, x in K1.items():
        y = K2.get(k, '<none>')
        if x != y:
            ok = False
            nl += 1
            if nl <= 10:
                report.append('  LOG DIFF\n    C++: %s\n    py : %s' % (x, y))
    only_py = [k for k in K2 if k not in K1]
    report.insert(0, 'log lines compared: %d (C++) / %d (python), %d differ%s'
                  % (len(K1), len(K2), nl,
                     '; only in the python log (not logged by this C++ build): ' + ', '.join(only_py) if only_py else ''))
    print('VERIFY %s vs %s (C++ run: %d iterations, python: %d)' % (a.out_dir, cpp, cpp_iters, maxIter))
    print('\n'.join(report))
    print('VERIFY RESULT: %s' % ('ALL IDENTICAL' if ok else 'DIFFERENCES FOUND'))
    with open(os.path.join(a.out_dir, 'verify_report.txt'), 'w') as f:
        f.write('C++ run: %s (%d iterations), log %s\n' % (cpp, cpp_iters, cpp_log))
        f.write('\n'.join(report) + '\nRESULT: %s\n' % ('ALL IDENTICAL' if ok else 'DIFFERENCES FOUND'))
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
