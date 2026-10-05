"""One ranked table of every relaxation of the subdivided coarse mesh in
output/relaxation_experiments (Python runs: results.json; C++ runs: the
"[coarse_subdiv_relax] relaxation (...)" line of <name>.log).

Ranking: each run gets a rank per metric (end values; ties share the best rank), and
the runs are ordered by the mean of their ranks:
  edge CV (lower), min angle p1 / p5 / median (higher), folded triangles at the end
  (lower; only explicit runs report it), flipped vs seed (lower), degenerate (lower).
The source decimation (validity checks on / off) is a column: runs from different
sources start from different meshes.

Runs of the subdivided FINE mesh ([subdiv_tracker] relaxation: l3_*, l4_*, relax_*,
m563_*, ...) relax another mesh and are listed apart, unranked.

  python rank_experiments.py   ->  output/relaxation_experiments/all_relaxation_experiments_ranked.{md,html,pdf}

The PDF is the md rendered to HTML and printed by Edge (or Chrome) headless, A3 landscape.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, '..', 'output', 'relaxation_experiments'))
OUT = os.path.join(EXP, 'all_relaxation_experiments_ranked.md')

QLINE = re.compile(r'relaxation \((?P<tag>[^)]*)\), before -> after: edge CV (?P<cv0>[\d.]+) -> (?P<cv1>[\d.]+) \| '
                   r'min angle [\d.]+ -> [\d.]+, p1 (?P<p10>[\d.]+) -> (?P<p11>[\d.]+), p5 (?P<p50>[\d.]+) -> '
                   r'(?P<p51>[\d.]+), median (?P<m0>[\d.]+) -> (?P<m1>[\d.]+) deg \| degenerate (?P<d0>\d+) -> '
                   r'(?P<d1>\d+) \| flipped vs seed (?P<flip>\d+)')
FOLD = re.compile(r'\[relax_explicit\] (?P<conv>converged|NOT CONVERGED) after (?P<it>\d+) iterations.*?'
                  r'folded (?P<f0>\d+) -> (?P<f1>\d+)')
VERTS = re.compile(r'\[coarse_subdiv\] \d+ levels: \|V\| \d+ -> (?P<n>\d+)')


def source_of(bundle):
    b = bundle.replace('\\', '/')
    m = re.search(r'relaxation_experiments/([^/]+)/correspondence_', b)
    return m.group(1) if m else '?'


def python_runs():
    rows = []
    for p in sorted(glob.glob(os.path.join(EXP, '*', 'results.json'))):
        d = os.path.basename(os.path.dirname(p))
        r = json.load(open(p))
        c, R, q0, q1 = r['config'], r['report'], r['quality_before'], r['quality_after']
        cfgtxt = open(os.path.join(EXP, d, 'experiment_config.txt')).read()
        m = re.search(r'^bundle:\s+(.*)$', cfgtxt, re.M)
        src = source_of(m.group(1)) if m else '?'
        n = r.get('n_subdiv_verts')
        if n is None:
            lg = open(os.path.join(EXP, d, 'run.log')).read()
            mv = VERTS.search(lg)
            n = int(mv.group('n')) if mv else None
        eq = r.get('equal_area', 'off')
        eq = 'off' if eq == 'off' else ('min area' if 'smallest' in eq else ('K=' + re.search(r'levels (\d+)', eq).group(1)
                                                                               if 'levels' in eq else eq))
        rows.append(dict(
            folder=d, impl='python', source=src, valid=src == 'src_qslim200_valid', method='explicit',
            weights=c['weights'], graph='directed' if c['explicitDirected'] and not c['jointPass'] else 'symmetric',
            proj='global' if c['explicitGlobalProj'] else 'local', nofold=c['noNewFolds'], perface=c['perCoarseFace'],
            lam=c['explicitLambda'], equal_area=eq, verts=n, iters=R['itersSheet'], converged=R['converged'],
            cv0=q0['edgeCV'], cv1=q1['edgeCV'], p10=q0['p1'], p11=q1['p1'], p50=q0['p5'], p51=q1['p5'],
            m0=q0['median'], m1=q1['median'], fold0=R['foldedSeed'], fold1=R['foldedResult'],
            flip=q1['flippedVsRef'], deg=q1['degenerate']))
    return rows


def cpp_runs(skip):
    rows, fine = [], []
    for lp in sorted(glob.glob(os.path.join(EXP, '*.log'))):
        name = os.path.splitext(os.path.basename(lp))[0]
        if name in skip or not os.path.isdir(os.path.join(EXP, name)):
            continue
        txt = open(lp, errors='replace').read()
        cl = [l for l in txt.splitlines() if l.startswith('[coarse_subdiv_relax] relaxation (')]
        if not cl:
            if any(l.startswith('[subdiv_tracker] relaxation') for l in txt.splitlines()):
                fine.append(name)
            continue
        q = QLINE.search(cl[-1])
        if not q:
            continue
        tag = q.group('tag')
        f = FOLD.search(txt)
        mv = VERTS.search(txt)
        m = re.search(r'explicit_lambda (\S+)', txt[:3000] + open(os.path.join(EXP, name, 'experiment_config.txt')).read()
                      if os.path.exists(os.path.join(EXP, name, 'experiment_config.txt')) else '')
        method = tag.split(',')[0]
        rows.append(dict(
            folder=name, impl='C++', source=name, valid='Validity checks: ENABLED' in txt,
            method=method, weights=('cotan' if 'cotan' in tag else 'meanvalue' if 'meanvalue' in tag else 'uniform'),
            graph='directed' if 'directed' in tag else 'symmetric',
            proj='local',  # C++ explicit default; global runs say so below
            nofold='no new folds' in tag, perface='per coarse face' in tag, lam=float(m.group(1)) if m else 0.5,
            equal_area='off', verts=int(mv.group('n')) if mv else None,
            iters=int(f.group('it')) if f else None, converged=(f.group('conv') == 'converged') if f else None,
            cv0=float(q.group('cv0')), cv1=float(q.group('cv1')), p10=float(q.group('p10')), p11=float(q.group('p11')),
            p50=float(q.group('p50')), p51=float(q.group('p51')), m0=float(q.group('m0')), m1=float(q.group('m1')),
            fold0=int(f.group('f0')) if f else None, fold1=int(f.group('f1')) if f else None,
            flip=int(q.group('flip')), deg=int(q.group('d1'))))
        if 'explicit_global_proj' in txt[:5000] or (os.path.exists(os.path.join(EXP, name, 'experiment_config.txt'))
                                                     and 'explicit_global_proj' in open(os.path.join(EXP, name, 'experiment_config.txt')).read()):
            rows[-1]['proj'] = 'global'
        if method != 'explicit':
            rows[-1]['proj'] = 'local' if 'local projection' in tag else '-'
    return rows, fine


def rank(rows):
    metrics = [('cv1', False), ('p11', True), ('p51', True), ('m1', True), ('fold1', False), ('flip', False),
               ('deg', False)]
    for key, high in metrics:
        vals = [(r[key], i) for i, r in enumerate(rows) if r[key] is not None]
        vals.sort(key=lambda t: -t[0] if high else t[0])
        prev, prank = None, 0
        for pos, (v, i) in enumerate(vals):
            if v != prev:
                prank, prev = pos + 1, v
            rows[i]['r_' + key] = prank
    for r in rows:
        rs = [r['r_' + k] for k, _ in metrics if ('r_' + k) in r]
        r['score'] = sum(rs) / len(rs)
    rows.sort(key=lambda r: (r['score'], r['cv1']))
    return rows


def fmt(v, f='%s'):
    return '-' if v is None else (f % v)


def md_to_html(md, title):
    """The few markdown forms the ranking file uses: #/## headings, paragraphs, '- ' lists,
    pipe tables, `code` and **bold**."""
    import html as H

    def inline(t):
        t = H.escape(t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
        return re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if l.startswith('## '):
            out.append('<h2>%s</h2>' % inline(l[3:]))
        elif l.startswith('# '):
            out.append('<h1>%s</h1>' % inline(l[2:]))
        elif l.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            i -= 1
            out.append('<table><thead><tr>' + ''.join('<th>%s</th>' % inline(c) for c in rows[0]) + '</tr></thead><tbody>')
            for r in rows[2:]:
                out.append('<tr>' + ''.join('<td>%s</td>' % inline(c) for c in r) + '</tr>')
            out.append('</tbody></table>')
        elif l.startswith('- '):
            out.append('<ul>')
            while i < len(lines) and lines[i].startswith('- '):
                out.append('<li>%s</li>' % inline(lines[i][2:]))
                i += 1
            i -= 1
            out.append('</ul>')
        elif l.strip():
            out.append('<p>%s</p>' % inline(l))
        i += 1
    css = ('@page{size:A3 landscape;margin:10mm}body{font-family:Segoe UI,Arial,sans-serif;font-size:9pt;color:#111}'
           'h1{font-size:15pt}h2{font-size:12pt;margin-top:14pt}table{border-collapse:collapse;font-size:7pt;width:100%}'
           'th,td{border:1px solid #bbb;padding:2px 3px;text-align:left;white-space:nowrap}'
           'th{background:#eee}tbody tr:nth-child(even){background:#f7f7f7}code{font-family:Consolas,monospace}'
           'thead{display:table-header-group}tr{page-break-inside:avoid}')
    return ('<!doctype html><html><head><meta charset="utf-8"><title>%s</title><style>%s</style></head><body>%s'
            '</body></html>' % (H.escape(title), css, '\n'.join(out)))


def write_pdf(md_path):
    """md -> html -> pdf (Edge / Chrome headless). Returns the pdf path, or None."""
    import subprocess
    base = os.path.splitext(md_path)[0]
    with open(md_path, encoding='utf-8') as f:
        page = md_to_html(f.read(), os.path.basename(base))
    with open(base + '.html', 'w', encoding='utf-8', newline='\n') as f:
        f.write(page)
    for exe in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
                r'C:\Program Files\Google\Chrome\Application\chrome.exe'):
        if not os.path.exists(exe):
            continue
        r = subprocess.run([exe, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                            '--print-to-pdf=' + base + '.pdf', 'file:///' + (base + '.html').replace('\\', '/')],
                           capture_output=True, timeout=120)
        if os.path.exists(base + '.pdf'):
            return base + '.pdf'
        print('[rank_experiments] %s failed: %s' % (os.path.basename(exe), r.stderr[-300:]))
    return None


def main():
    py = python_runs()
    cpp, fine = cpp_runs(skip=set(r['folder'] for r in py))
    rows = rank(py + cpp)
    valid = {r['source'] for r in rows if 'valid' in r['source']}
    L = ['# All relaxations of the subdivided coarse mesh, ranked', '',
         'ABC 00040057. Generated by `relaxation_scripts_python/rank_experiments.py` from each folder\'s '
         '`results.json` (Python) or `<folder>.log` (C++).', '',
         '**Ranking:** per metric, each run gets a rank (ties share the best rank). Runs are ordered by the mean rank '
         'over: edge CV (lower is better), min angle p1 / p5 / median (higher), folded triangles at the end (lower; '
         'only explicit runs report it), flipped vs seed (lower), degenerate (lower). All values are after the '
         'relaxation; "start" columns show the seed.', '',
         '**Validity checks / source run:** the decimation the relaxation starts from. Python runs read the bundle '
         'of a C++ run (`src_qslim200_valid` or `clamp_check`); each C++ run did its own decimation. Validity checks '
         'ON (`src_qslim200_valid`, 541 coarse faces) and off (all others, 400 coarse faces, the same decimation in '
         'each) start from different meshes (edge CV 0.72 vs 1.08), so compare runs within one group.', '',
         '| rank | folder | impl | validity checks | source run | method | weights | graph | projection | no new folds | per face | '
         'equal area | vertices | iters | converged | edge CV start -> end | p1 start -> end | p5 end | median end | '
         'folded seed -> end | flipped vs seed | degenerate | mean rank |',
         '|' + '---|' * 23]
    for i, r in enumerate(rows, 1):
        L.append('| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.3f -> %.3f | %.2f -> %.2f '
                 '| %.2f | %.1f | %s | %d | %d | %.1f |'
                 % (i, r['folder'], r['impl'], 'ON' if r['valid'] else 'off', r['source'], r['method'], r['weights'], r['graph'], r['proj'],
                    'yes' if r['nofold'] else 'no', 'yes' if r['perface'] else 'no', r['equal_area'],
                    fmt(r['verts']), fmt(r['iters']), fmt(None if r['converged'] is None else
                                                          ('yes' if r['converged'] else 'no')),
                    r['cv0'], r['cv1'], r['p10'], r['p11'], r['p51'], r['m1'],
                    '-' if r['fold1'] is None else '%d -> %d' % (r['fold0'], r['fold1']), r['flip'], r['deg'],
                    r['score']))
    L.append('')
    for v, label in ((True, 'validity checks ON (541 coarse faces)'), (False, 'validity checks off (400 coarse faces)')):
        grp = [r for r in rows if r['valid'] == v]
        if grp:
            b = grp[0]
            L.append('- Best with %s: **`%s`** (edge CV %.3f, p1 / p5 / median %.2f / %.2f / %.1f deg, folded %s, '
                     'flipped vs seed %d, degenerate %d).' % (label, b['folder'], b['cv1'], b['p11'], b['p51'], b['m1'],
                                                              fmt(b['fold1']), b['flip'], b['deg']))
    L += ['', '## Not ranked', '',
          'Relaxations of the subdivided **fine** mesh (`[subdiv_tracker] relaxation`, about 88k samples; another '
          'mesh and problem): ' + ', '.join('`%s`' % n for n in fine) + '.', '',
          'Python runs without results: ' + ', '.join(
              '`%s`' % os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(EXP, 'py_*', 'experiment_config.txt')))
              if not os.path.exists(os.path.join(os.path.dirname(p), 'results.json'))) + ' (stopped before the end).']
    with open(OUT, 'w', newline='\n', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(OUT)
    pdf = write_pdf(OUT)
    print(pdf or '[rank_experiments] no PDF (Edge / Chrome not found or failed); the .html can be printed instead')
    for i, r in enumerate(rows[:10], 1):
        print('%2d %-36s %-22s cv %.3f p1 %.2f p5 %.2f med %.1f fold %s flip %d deg %d  (%.1f)'
              % (i, r['folder'], r['source'], r['cv1'], r['p11'], r['p51'], r['m1'], fmt(r['fold1']), r['flip'],
                 r['deg'], r['score']))


if __name__ == '__main__':
    main()
