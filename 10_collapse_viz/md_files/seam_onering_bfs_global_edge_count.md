# BFS Sheet Partitioning: Local vs Global Edge Count

## The Problem

`update_seam_onering_display()` partitions the one-ring faces of a picked
seam edge into "local sheets" using BFS. The BFS is only allowed to cross
an edge if that edge is **manifold** (exactly 2 incident live faces).

The current implementation counts manifold-ness using only the faces in the
local ring (the one-ring of `vi ∪ vj`). This is wrong.

An edge that has **2 ring-local faces** can still have **3+ faces in the full
mesh** — making it a seam edge globally — but the local count makes it look
safe to cross. The BFS then merges faces from different sheets.

## Two Figures That Show the Distinction

### Figure 1 — tail faces should NOT merge across a globally non-manifold edge

```
        v2
       /  \
  v4--v1---v3--v7
      |         |
      v5        v6
```

Seam edge: `v1–v3` (3+ incident faces globally).

Suppose edge `vA–v1` (connecting a tail face to a seam face) is itself
globally non-manifold (3+ full-mesh faces). In the local ring it appears
to have only 2 faces → local BFS crosses it → wrong merge.

### Figure 2 — tail faces SHOULD merge (same sheet, globally manifold edge)

```
  v6  v5      v4  v3
   \  / \    / \  /
    v1---+--+---v2    (seam edge v1–v2, red)
```

Faces: `v1-v5-v6` (left tail), `v5-v1-v2` (seam face), `v4-v1-v2` (seam face), `v2-v3-v4` (right tail).

Edge `v1–v5` has exactly 2 faces in the **full mesh** (the left tail and its
seam face) → globally manifold → BFS should cross it → correctly merges the
tail into the same local sheet as its seam face.

The local ring count also gives 2 here, so the current code accidentally
works in this case. But the fix must be principled.

## The Fix

Precompute a `globalEdgeCount` map by scanning **all live real faces** in
`gF`/`gV` (not just the ring faces). Use this as the BFS crossing criterion:

```cpp
// Before BFS — scan full mesh once:
std::map<std::pair<int,int>, int> globalEdgeCount;
for (int f = 0; f < gF.rows(); f++) {
    if (is_face_dead(gF, f)) continue;
    if (std::isinf(gV(gF(f,0),0)) || ...) continue;
    for (int c = 0; c < 3; c++) {
        int a = gF(f,c), b = gF(f,(c+1)%3);
        if (a > b) std::swap(a,b);
        globalEdgeCount[{a,b}]++;
    }
}

// Inside BFS — replace local size check:
// OLD (wrong):
//   if ((int)nbrs.size() != 2) continue;
// NEW (correct):
auto git = globalEdgeCount.find({a,b});
if (git == globalEdgeCount.end() || git->second != 2) continue;
```

Apply the same fix to the ImGui inline BFS (sheet count display).

## Files to Change

- `10_collapse_viz/sheet_seam_viz.cpp`
  - `update_seam_onering_display()` — main BFS
  - `sheet_seam_imgui_section()` — inline BFS for sheet count label

## Status

Implemented in commit `267885b`, then **reverted** (`89d0039`) to come back
to later. The revert is intentional — the logic above is correct, just
deferred.
