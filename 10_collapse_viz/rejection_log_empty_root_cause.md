# Why `collapse_rejections_bear_simplified.txt` Was Empty

## Summary

Three layered bugs — one sequencing bug, one dead-code bug, and one architectural blind spot —
combined to produce an empty rejection log even with `--validity-checks` active.

---

## Bug 1 — `SSP_rej_log_open` called too late (sequencing)

**File:** `10_collapse_viz/main.cpp`  
**Fixed in commit:** `1d6c571`

`SSP_rej_log_open(...)` was called at line 847, well after `init_ssp(...)` at line 696.
During `init_ssp` the qslim cost function runs once over every edge to initialise the
priority queue (the *initial cost pass*). Any `[QSLIM-INF]` or `[QSLIM-REJECT]` entries
generated at that point called `SSP_rej_log_file()`, which returned `nullptr` because the
file had not been opened yet. Those writes were silently dropped.

**Fix:** move `SSP_rej_log_open` to just before `init_ssp`, so the file handle is valid
for the entire lifetime of the program including the initial cost pass.

---

## Bug 2 — `[QSLIM-INF]` block was dead code (commented-out `#ifdef` + runtime gate)

**File:** `src/SSP_qslim_optimal_collapse_edge_callbacks.cpp`  
**Fixed in commit:** `19b0eec`

The block that logs infinite-cost edges was wrapped in two guards that together made it
unreachable:

```cpp
// #ifdef ML_QEM_LOG          ← double-slash comment: ifdef never compiled
if (s_qslim_log_enabled)      ← false during entire initial cost pass
{
    // [QSLIM-INF] logging ...
}
// #endif
```

- The `#ifdef ML_QEM_LOG` was a comment (`//`), so the preprocessor never saw it.
  Even though `ML_QEM_LOG=ON` was set in `CMakePresets.json`, the macro had no effect.
- `SSP_qslim_enable_log(true)` is called *after* `init_ssp` returns (main.cpp line 773),
  so `s_qslim_log_enabled` was `false` during the entire initial cost pass.

**Fix:** make the `[QSLIM-INF]` block unconditional (remove both the dead ifdef and the
runtime gate) with its own static cap (500 entries to file / 20 to stderr).

---

## Bug 3 — Infinite-cost edges bypass the validity-check branch entirely (architectural)

**File:** `src/SSP_qslim_optimal_collapse_edge_callbacks.cpp`

Even after bugs 1 and 2 were fixed, the `[QSLIM-REJECT]` (face-flip / quality) entries
would still not appear for degenerate-quadric edges. The cost function flow is:

```
QEM optimal position
    └─ if cost is NaN/Inf or < threshold → midpoint fallback
           └─ if midpoint also gives Inf → cost = ∞, early return
                                           ↑
              validity checks only run here, AFTER confirming cost is finite
```

An edge whose combined quadric matrix `A` is singular produces `A.inverse()` = NaN → the
midpoint fallback is tried → midpoint quadric evaluation also diverges → `cost = ∞` and
the function returns **without ever reaching the face-flip or quality checks**.
So a mesh dominated by degenerate quadrics (e.g. the bear's sheet geometry) produces zero
`[QSLIM-REJECT]` entries even with `--validity-checks` active, because those edges never
reach the validity-check branch.

**This is expected behaviour**, not a bug. The `[QSLIM-INF]` logging (bug 2 fix) covers
these edges.

---

## Bug 4 — Queue exhaustion before validity-check rejections accumulate (operational)

With `--mat_struct_check` active, `gPreFn` (the struct-ID gate in `pre_collapse`) blocks
edges whose two endpoints belong to different structural components. On the bear mesh, the
exhaustion-time diagnostic revealed:

```
live_edges=9467   pre_blocked=7214 (76%)   inf=238   finite_cost(uv_pending)=2015
```

76 % of all remaining live edges were permanently blocked by the struct gate before
`gCostFn` was even called. The queue exhausted at 6 524 live faces (target: 200) after
only 599 collapses. Because so few finite-cost edges survived the pre-collapse gate and
reached the validity checks, the main rejection log accumulated only ~200 entries before
the queue went dry.

**Implication:** running without `--mat_struct_check` (the Debug launch config) lets many
more collapses attempt the validity-check branch and produces a much richer rejection log.

---

## Fixes applied (chronological)

| Commit    | What changed |
|-----------|-------------|
| `19b0eec` | `[QSLIM-INF]` block made unconditional — fires during initial cost pass |
| `1d6c571` | `SSP_rej_log_open` moved to before `init_ssp` — file handle valid from startup |
| `0342c07` | Exhaustion diagnostic added — re-costs all live edges at queue exhaustion, logs to `exhausted_queue_rejections.log` |
| `051ee14` | Added `gPreFn` call to exhaustion diagnostic — struct/stale/seam-blocked edges logged as `[EXHAUSTION-PRE-BLOCK]` |

---

## What the logs now contain (bear / qslim / release)

| File | Contents |
|------|----------|
| `collapse_rejections_bear_simplified.txt` | `[QSLIM-REJECT]` quality + face-flip entries from actual collapse attempts |
| `exhausted_queue_rejections.log` | Per-edge diagnosis at exhaustion: `[EXHAUSTION-PRE-BLOCK]`, `[QSLIM-INF]`, `[QSLIM-STATS]` + summary line |
| `struct_gate_log.txt` | Per-collapse `PASS`/`BLOCK` decisions from the struct-ID gate |
