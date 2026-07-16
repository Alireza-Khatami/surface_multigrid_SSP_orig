"""
seam_diag_stats.py  —  parse a seam_diag_<stem>.txt file produced by
SSP_collapse_edge and print a breakdown of why seam collapses are rejected.

Usage:
    python seam_diag_stats.py  seam_diag_<stem>.txt
"""

import sys
import re
from collections import defaultdict, Counter


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "seam_diag.txt"

    tag_counts   = defaultdict(int)
    validity_shared = []   # shared_real_faces values from SEAM-REJECT-VALIDITY
    try_sheets   = []      # active_sheets values from SEAM-TRY
    # per-stage breakdown: tag → list of (e0, e1, sid or -1)
    stage_detail = defaultdict(list)

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r'\[(SEAM-[A-Z_-]+)\]', line)
            if not m:
                continue
            tag = m.group(1)
            tag_counts[tag] += 1

            if tag == "SEAM-REJECT-VALIDITY":
                m2 = re.search(r'shared_real_faces=(\d+)', line)
                if m2:
                    validity_shared.append(int(m2.group(1)))

            elif tag == "SEAM-TRY":
                m2 = re.search(r'active_sheets=(\d+)', line)
                if m2:
                    try_sheets.append(int(m2.group(1)))

            # Capture e=(a,b) and optional sid for all per-sheet tags
            me = re.search(r'e=\((\d+),(\d+)\)', line)
            ms = re.search(r'sid=(-?\d+)', line)
            e_pair = (int(me.group(1)), int(me.group(2))) if me else None
            sid    = int(ms.group(1)) if ms else -1
            stage_detail[tag].append((e_pair, sid))

    total = sum(tag_counts.values())
    n_reject  = tag_counts.get("SEAM-REJECT-VALIDITY", 0)
    n_try     = tag_counts.get("SEAM-TRY", 0)
    n_attempts = n_reject + n_try   # mutually exclusive paths through the code

    print(f"File: {path}")
    print(f"{'=' * 60}")
    print(f"  Total [SEAM-*] entries  : {total}")
    print(f"  Total seam attempts     : {n_attempts}")
    print(f"    rejected by validity  : {n_reject:4d}  "
          f"({100 * n_reject / max(1, n_attempts):.1f}%)")
    print(f"    passed validity       : {n_try:4d}  "
          f"({100 * n_try / max(1, n_attempts):.1f}%)")

    # ---- validity rejection: shared_real_faces distribution ----
    if validity_shared:
        dist = Counter(validity_shared)
        print(f"\n  --- shared_real_faces distribution  [SEAM-REJECT-VALIDITY] ---")
        for k in sorted(dist):
            bar = "#" * min(50, dist[k])
            print(f"  faces={k}  {dist[k]:5d}  {bar}")

    # ---- SEAM-TRY: active_sheets distribution ----
    if try_sheets:
        dist = Counter(try_sheets)
        print(f"\n  --- active_sheets distribution  [SEAM-TRY] ---")
        for k in sorted(dist):
            bar = "#" * min(50, dist[k])
            print(f"  sheets={k}  {dist[k]:5d}  {bar}")

    # ---- per-stage rejection summary (only shown when something passed validity) ----
    stage_tags = [
        "SEAM-FAIL-ONERING",
        "SEAM-SKIP-FEW-FACES",
        "SEAM-SKIP-BSI",
        "SEAM-SKIP-FANWALK",
        "SEAM-SKIP-INV",
        "SEAM-FAIL-LSCM",
        "SEAM-FAIL-ALL-SHEETS",
    ]
    present = [t for t in stage_tags if tag_counts.get(t, 0) > 0]
    if present:
        print(f"\n  --- per-stage rejection counts (after validity passed) ---")
        for tag in stage_tags:
            n = tag_counts.get(tag, 0)
            if n == 0:
                continue
            entries = stage_detail[tag]
            # count unique edges
            unique_edges = len({e for e, _ in entries if e is not None})
            # count per sid
            sid_dist = Counter(s for _, s in entries if s >= 0)
            sid_str = "  sids: " + ", ".join(
                f"{s}×{c}" for s, c in sid_dist.most_common(5)
            ) if sid_dist else ""
            print(f"  [{tag}]  {n:4d} events  {unique_edges} unique edges{sid_str}")

    # ---- unknown / extra tags ----
    known = {
        "SEAM-REJECT-VALIDITY", "SEAM-TRY",
        *stage_tags
    }
    extra = {t: c for t, c in tag_counts.items() if t not in known}
    if extra:
        print(f"\n  --- other tags ---")
        for t, c in sorted(extra.items()):
            print(f"  [{t}]  {c}")

    # ---- root cause verdict ----
    print(f"\n  {'=' * 56}")
    print(f"  ROOT CAUSE VERDICT")
    print(f"  {'=' * 56}")
    if n_reject > 0 and n_try == 0:
        print(f"  ALL {n_reject} seam-edge collapses rejected by igl::edge_collapse_is_valid.")
        print(f"  The manifold link condition requires exactly 2 shared vertices")
        print(f"  between the one-rings of vi and vj.  Seam edges have 3+ shared")
        print(f"  real faces → 3+ shared vertices → check always fails.")
        print(f"")
        print(f"  Proposed fix: skip edge_collapse_is_valid when shared_real_faces > 2")
        print(f"  (i.e., let the per-sheet LSCM loop decide validity for non-manifold edges).")
    elif n_reject > 0 and n_try > 0:
        print(f"  Mixed: {n_reject} rejected by validity, {n_try} passed.")
        if present:
            dominant = max(present, key=lambda t: tag_counts[t])
            print(f"  Dominant post-validity failure: [{dominant}]  ({tag_counts[dominant]}×)")
    elif n_try > 0 and not present:
        print(f"  All {n_try} seam edges that passed validity succeeded (no [SEAM-FAIL-*] logged).")
        print(f"  seam_collapses should be > 0 — check the gSeamAttemptCount counter.")
    elif n_attempts == 0:
        print(f"  No seam-edge collapses attempted.  The seam edges may have lost their")
        print(f"  non-manifold status (topology changed by neighbor collapses) before")
        print(f"  they were popped from the queue — OR the queue was exhausted first.")
    else:
        print(f"  See per-stage breakdown above.")


if __name__ == "__main__":
    main()
