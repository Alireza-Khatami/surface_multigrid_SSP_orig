import sys
import re
import math

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "seam_edge_costs.txt"

    costs = []
    ranks = []
    total_edges = None
    not_found = 0
    inf_count = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("seam_edges="):
                m = re.search(r"total_edges=(\d+)", line)
                if m:
                    total_edges = int(m.group(1))
                continue
            if "NOT_FOUND" in line:
                not_found += 1
                continue
            m_cost = re.search(r"cost=(\S+)", line)
            if not m_cost:
                continue
            val = m_cost.group(1)
            if val == "inf" or val == "-inf":
                inf_count += 1
            else:
                costs.append(float(val))
            m_rank = re.search(r"rank=(\d+)/(\d+)", line)
            if m_rank:
                ranks.append(int(m_rank.group(1)))
                if total_edges is None:
                    total_edges = int(m_rank.group(2))

    finite = costs
    n_finite = len(finite)
    n_total = n_finite + inf_count + not_found

    print(f"File: {path}")
    print(f"  Total seam edges : {n_total}")
    print(f"  Finite cost      : {n_finite}")
    print(f"  Inf cost         : {inf_count}")
    print(f"  Not found in gE  : {not_found}")

    if not finite:
        print("  (no finite costs to summarize)")
        return

    finite_sorted = sorted(finite)
    n = len(finite_sorted)

    def percentile(p):
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return finite_sorted[lo] + (idx - lo) * (finite_sorted[hi] - finite_sorted[lo])

    mean = sum(finite) / n
    variance = sum((x - mean) ** 2 for x in finite) / n
    std = math.sqrt(variance)

    print(f"\n  --- finite cost statistics ---")
    print(f"  min    : {finite_sorted[0]:.6g}")
    print(f"  p1     : {percentile(1):.6g}")
    print(f"  p5     : {percentile(5):.6g}")
    print(f"  p25    : {percentile(25):.6g}")
    print(f"  median : {percentile(50):.6g}")
    print(f"  p75    : {percentile(75):.6g}")
    print(f"  p95    : {percentile(95):.6g}")
    print(f"  p99    : {percentile(99):.6g}")
    print(f"  max    : {finite_sorted[-1]:.6g}")
    print(f"  mean   : {mean:.6g}")
    print(f"  std    : {std:.6g}")

    # Bucket breakdown by order of magnitude
    print(f"\n  --- cost buckets ---")
    buckets = {}
    for c in finite:
        if c == 0.0:
            key = "=0"
        else:
            mag = math.floor(math.log10(c))
            key = f"1e{mag:+d}"
        buckets[key] = buckets.get(key, 0) + 1

    for key in sorted(buckets, key=lambda k: (0, 0) if k == "=0" else (1, int(k[2:]))):
        bar = "#" * min(40, buckets[key])
        print(f"  {key:>6}  {buckets[key]:4d}  {bar}")

    # Rank statistics
    if ranks and total_edges:
        ranks_sorted = sorted(ranks)
        n_r = len(ranks_sorted)

        def pct_rank(p):
            idx = (p / 100) * (n_r - 1)
            lo, hi = int(idx), min(int(idx) + 1, n_r - 1)
            return ranks_sorted[lo] + (idx - lo) * (ranks_sorted[hi] - ranks_sorted[lo])

        print(f"\n  --- queue rank (out of {total_edges} edges, 1=lowest cost=first collapsed) ---")
        print(f"  min rank   : {ranks_sorted[0]}  ({100*ranks_sorted[0]/total_edges:.1f}%)")
        print(f"  p25 rank   : {pct_rank(25):.0f}  ({100*pct_rank(25)/total_edges:.1f}%)")
        print(f"  median rank: {pct_rank(50):.0f}  ({100*pct_rank(50)/total_edges:.1f}%)")
        print(f"  p75 rank   : {pct_rank(75):.0f}  ({100*pct_rank(75)/total_edges:.1f}%)")
        print(f"  max rank   : {ranks_sorted[-1]}  ({100*ranks_sorted[-1]/total_edges:.1f}%)")

        # How many seam edges are in the first N% of the queue
        for threshold_pct in [10, 25, 50]:
            cutoff = int(total_edges * threshold_pct / 100)
            count = sum(1 for r in ranks if r <= cutoff)
            print(f"  seam edges in first {threshold_pct:2d}% of queue : {count}/{n_r} ({100*count/n_r:.1f}%)")

if __name__ == "__main__":
    main()
