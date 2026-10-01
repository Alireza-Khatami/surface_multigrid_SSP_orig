"""Compare explicit runs with different step sizes (lambda) at equal total step
lambda x iterations, from their logs ([relax_explicit] lines, logged every 100
iterations): folded triangles, degenerate triangles and energy (1/2 sum of
squared edge lengths; lower = more even spacing).

usage: python step_size_table.py <run.log> [<run.log> ...]   (e.g. ../output/relaxation_experiments/lam1.log)
"""
import os
import re
import sys

HEAD = re.compile(r'\[relax_explicit\] .*lambda ([0-9.eE+-]+),.*energy ([0-9.eE+-]+), folded (-?\d+)')
ITER = re.compile(r'\[relax_explicit\] iter (\d+): max move ([0-9.eE+-]+) .*energy ([0-9.eE+-]+), folded (-?\d+), '
                  r'degenerate (\d+) \(([0-9.]+) s\)')
BUDGETS = [50, 100, 200, 500, 1000, 2000, 2500]

logs = sys.argv[1:]
if not logs:
    sys.exit(__doc__)
runs = []
for path in logs:
    lam, rows, seed = None, {}, None
    with open(path, errors='replace') as f:
        for line in f:
            m = HEAD.search(line)
            if m:
                lam, seed = float(m.group(1)), (float(m.group(2)), int(m.group(3)))
                continue
            m = ITER.search(line)
            if m:
                rows[int(m.group(1))] = (float(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6)),
                                         float(m.group(2)))
    if lam is None:
        print(f'{path}: no [relax_explicit] header, skipped')
        continue
    runs.append((os.path.basename(path), lam, rows, seed))

if runs:
    e0, f0 = runs[0][3]
    print(f'seed: energy {e0:.4g}, folded {f0}')
for b in BUDGETS:
    line = []
    for name, lam, rows, _ in runs:
        it = round(b / lam)
        if abs(it * lam - b) < 1e-9 and it in rows:
            e, fo, de, t, _ = rows[it]
            line.append(f'{name} (lambda {lam:g}, iter {it}): folded {fo}, degenerate {de}, energy {e:.4g}, {t:.0f} s')
    if line:
        print(f'\nlambda x iterations = {b}')
        for l in line:
            print('  ' + l)
print('\nlast logged iteration:')
for name, lam, rows, _ in runs:
    if rows:
        it = max(rows)
        e, fo, de, t, mv = rows[it]
        print(f'  {name} (lambda {lam:g}): iter {it} (lambda x iter {it * lam:g}), folded {fo}, degenerate {de}, '
              f'energy {e:.4g}, last max move {mv:.2g} x diag, {t:.0f} s')
