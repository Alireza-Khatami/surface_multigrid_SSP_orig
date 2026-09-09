"""
build_and_diagnose.py
Builds the release binary, runs it with the bear params, then inspects all
rejection-log files and prints a structured diagnosis to stdout.

Usage:  python build_and_diagnose.py
        python build_and_diagnose.py --skip-build   (skip cmake, re-use existing binary)
"""

import subprocess, sys, os, re, time
from pathlib import Path

WORKSPACE = Path(__file__).parent.resolve()
CMAKE     = r"C:/Program Files/CMake/bin/cmake.exe"
BINARY    = WORKSPACE / "build/release/Release/collapse_viz_bin.exe"
OUTPUT_DIR = WORKSPACE / "output/bear"

BEAR_ARGS = [
    str(BINARY),
    "--mesh_path",
    r"C:\Users\alirz\Projects\Graphics\Neural QMAT\Experiments\bear\structural_mat\bear\bear_simplified.obj",
    "--matstruct_path",
    r"C:\Users\alirz\Projects\Graphics\Neural QMAT\Experiments\bear\structural_mat\bear\bear.ma_struct",
    "--target_faces", "200",
    "--mode",         "qslim",
    "--n_samples_total", "1000000",
    "--output_dir",   str(OUTPUT_DIR),
    "--validity-checks",
    "--mat_struct_check",
    "--trace_vertices", "trace_vids.txt",
]

# -- helpers ------------------------------------------------------------------

def run(cmd, cwd=None, label=""):
    print(f"\n{'='*60}")
    print(f"  {label or ' '.join(str(c) for c in cmd[:3])}")
    print(f"{'='*60}")
    t0 = time.time()
    # allow_nonzero: collapse_viz_bin returns non-zero on normal queue-exhaustion exit
    allow_nonzero = any("collapse_viz_bin" in str(c) for c in cmd)
    result = subprocess.run(cmd, cwd=str(cwd or WORKSPACE),
                            capture_output=False, text=True)
    elapsed = time.time() - t0
    print(f"  -> exit={result.returncode}  ({elapsed:.1f}s)")
    if result.returncode != 0 and not allow_nonzero:
        print(f"  FAILED", file=sys.stderr)
        sys.exit(result.returncode)
    return result

def count_lines_matching(path, pattern):
    if not path.exists():
        return 0, []
    rx = re.compile(pattern)
    hits = [l.rstrip() for l in path.read_text(errors="replace").splitlines() if rx.search(l)]
    return len(hits), hits

def head(lines, n=10):
    return lines[:n] + (["  ..."] if len(lines) > n else [])

def section(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

# -- build ---------------------------------------------------------------------

skip_build = "--skip-build" in sys.argv

if not skip_build:
    run([CMAKE, "--preset", "windows-release"],
        cwd=WORKSPACE, label="cmake configure (release)")
    run([CMAKE, "--build", "build/release", "--config", "Release", "-j8"],
        cwd=WORKSPACE, label="cmake build (release)")
else:
    print("  [--skip-build] skipping cmake")

if not BINARY.exists():
    print(f"ERROR: binary not found: {BINARY}", file=sys.stderr)
    sys.exit(1)

# -- run -----------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
run(BEAR_ARGS, cwd=WORKSPACE, label="collapse_viz_bin (bear / qslim / release)")

# -- diagnose ------------------------------------------------------------------

LOG_REJECTIONS  = OUTPUT_DIR / "collapse_rejections_bear_simplified.txt"
LOG_EXHAUSTION  = OUTPUT_DIR / "exhausted_queue_rejections.log"
LOG_STRUCT_GATE = OUTPUT_DIR / "struct_gate_log.txt"
LOG_SEAM        = OUTPUT_DIR / "seam_diag_bear_simplified.txt"

section("File existence check")
for f in [LOG_REJECTIONS, LOG_EXHAUSTION, LOG_STRUCT_GATE, LOG_SEAM]:
    exists  = f.exists()
    size    = f.stat().st_size if exists else 0
    status  = f"EXISTS  ({size} bytes)" if exists else "MISSING"
    print(f"  {'OK' if exists and size>0 else ('EMPTY' if exists else 'MISS'):5s}  {f.name}  — {status}")

# -- collapse_rejections -------------------------------------------------------

section("collapse_rejections_bear_simplified.txt")
n_inf,  inf_lines  = count_lines_matching(LOG_REJECTIONS, r"\[QSLIM-INF\]")
n_rej,  rej_lines  = count_lines_matching(LOG_REJECTIONS, r"\[QSLIM-REJECT\]")
n_uv,   uv_lines   = count_lines_matching(LOG_REJECTIONS, r"\[UV-REJECT\]")
n_stat, stat_lines = count_lines_matching(LOG_REJECTIONS, r"\[QSLIM-STATS\]")
print(f"  [QSLIM-INF]     : {n_inf}")
print(f"  [QSLIM-REJECT]  : {n_rej}")
print(f"  [UV-REJECT]     : {n_uv}")
print(f"  [QSLIM-STATS]   : {n_stat}")
if n_inf == 0 and n_rej == 0 and n_uv == 0:
    print("\n  *** collapse_rejections is EMPTY — no rejection paths were reached ***")
elif n_inf > 0:
    print("\n  First [QSLIM-INF] entries:")
    for l in head(inf_lines): print(f"    {l}")

# -- exhausted_queue_rejections ------------------------------------------------

section("exhausted_queue_rejections.log")
if LOG_EXHAUSTION.exists() and LOG_EXHAUSTION.stat().st_size > 0:
    text = LOG_EXHAUSTION.read_text(errors="replace")
    # summary line
    summ = [l for l in text.splitlines() if "EXHAUSTION-SUMMARY" in l]
    pre  = [l for l in text.splitlines() if "EXHAUSTION-PRE-BLOCK" in l]
    inf2 = [l for l in text.splitlines() if "QSLIM-INF" in l]
    rej2 = [l for l in text.splitlines() if "QSLIM-REJECT" in l]
    print(f"  Summary  : {summ[0] if summ else '(not found)'}")
    print(f"  PRE-BLOCK: {len(pre)}")
    print(f"  QSLIM-INF: {len(inf2)}")
    print(f"  QSLIM-REJ: {len(rej2)}")
    if pre:
        print("\n  First [EXHAUSTION-PRE-BLOCK] entries:")
        for l in head(pre): print(f"    {l}")
    if inf2:
        print("\n  First [QSLIM-INF] (at exhaustion):")
        for l in head(inf2): print(f"    {l}")
else:
    print("  *** exhausted_queue_rejections.log is MISSING or EMPTY ***")
    print("  → queue exhaustion branch was never reached, OR binary is stale (needs rebuild)")

# -- struct gate ---------------------------------------------------------------

section("struct_gate_log.txt")
if LOG_STRUCT_GATE.exists() and LOG_STRUCT_GATE.stat().st_size > 0:
    n_pass,  _ = count_lines_matching(LOG_STRUCT_GATE, r"PASS")
    n_block, b = count_lines_matching(LOG_STRUCT_GATE, r"BLOCK")
    print(f"  PASS  : {n_pass}")
    print(f"  BLOCK : {n_block}")
    if n_block > 0:
        print("\n  First BLOCK entries:")
        for l in head(b): print(f"    {l}")
    if n_pass == 0 and n_block > 0:
        print("\n  *** ALL collapses were struct-gate BLOCKED — queue exhausted with 0 collapses ***")
else:
    print("  struct_gate_log.txt MISSING or EMPTY  (--mat_struct_check active?)")

# -- overall verdict -----------------------------------------------------------

section("VERDICT")

rej_empty   = not LOG_REJECTIONS.exists() or LOG_REJECTIONS.stat().st_size == 0
exh_present = LOG_EXHAUSTION.exists() and LOG_EXHAUSTION.stat().st_size > 0

if rej_empty and not exh_present:
    print("  PROBLEM: collapse_rejections is empty AND exhausted_queue_rejections is missing.")
    print("  Likely cause: binary is STALE — rebuild did not pick up source changes.")
elif rej_empty and exh_present:
    summ = [l for l in LOG_EXHAUSTION.read_text(errors="replace").splitlines()
            if "EXHAUSTION-SUMMARY" in l]
    print("  collapse_rejections is empty but exhaustion diagnostic ran.")
    if summ: print(f"  {summ[0]}")
    # parse pre_blocked
    m = re.search(r"pre_blocked=(\d+)", summ[0] if summ else "")
    n_pb = int(m.group(1)) if m else -1
    m2 = re.search(r"inf=(\d+)", summ[0] if summ else "")
    n_i = int(m2.group(1)) if m2 else -1
    if n_pb > 0 and n_i == 0:
        print(f"\n  ROOT CAUSE: All {n_pb} live edges were blocked by gPreFn (struct/stale/seam gate).")
        print("  → collapse_rejections is empty because the cost function was never called.")
        print("  → Fix: check struct_gate_log.txt; consider running without --mat_struct_check")
    elif n_i > 0 and n_pb == 0:
        print(f"\n  ROOT CAUSE: All {n_i} live edges have ∞ cost (degenerate quadrics).")
        print("  → collapse_rejections is empty because cost=∞ bypasses the validity-check branch.")
    elif n_pb > 0 and n_i > 0:
        print(f"\n  ROOT CAUSE: Mixed — {n_pb} pre-blocked + {n_i} degenerate-quadric edges.")
    else:
        print("  → Could not determine root cause from summary line.")
else:
    print("  collapse_rejections is non-empty — rejection logging is working.")
    print(f"  [QSLIM-INF]={n_inf}  [QSLIM-REJECT]={n_rej}  [UV-REJECT]={n_uv}")

print()
