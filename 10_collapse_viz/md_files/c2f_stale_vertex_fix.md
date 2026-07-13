# Stale Vertex Bug in `query_coarse_to_fine` — Confirmed & Fixed

## What the backward walk needs

To map a coarse vertex `vi` back to the original fine mesh, the walk traces backward
through the collapse history one step at a time:

```
state: BF = (vi, n1, n2),  FIdx = fi   ← which face vi lives in on the coarse mesh

each step:
  1. dIdx  = last collapse that touched fi  (from decIM[fi])
  2. sd    = SheetData snapshot at dIdx  (the one-ring at that moment in history)
  3. v0,v1,v2 = local indices of BF(0..2) inside sd.subsetVIdx
  4. queryUV  = BC(0)*UV_post[v0] + BC(1)*UV_post[v1] + BC(2)*UV_post[v2]
  5. find queryUV in UV_pre triangles → new BC, BF, FIdx
  repeat until decIM[FIdx] is exhausted
```

Step 3 is where the bug lives.

---

## Why gF can't be used

`gF` accumulates every in-place rewrite across ALL collapses:

- **Pass 1**: flap faces are killed in-place (`F(f,c) = IGL_COLLAPSE_EDGE_NULL`)
- **Pass 2**: `d → s` is rewritten in-place in every surviving face of `d` (`F(f,c) = s`)
- **Vertices**: `V.row(s) = p; V.row(d) = p;` in-place

No new rows are ever added to `gF` or `gV`. After 500 collapses, `gF[fi]` is the
accumulated result of every rewrite that ever touched row `fi`.

---

## Why the original code never saw this

In a **manifold** mesh every edge has exactly one sheet. Every face in Pass 2 that gets
the `d→s` rewrite is also in the active one-ring. So every such rewrite gets a
`decIM[f].push_back(dIdx)` entry, and `BF` initialized from `gF` always matches
`subsetVIdx` at the last tracked collapse.

In the **non-manifold** extension a single edge can span multiple sheets. A collapse is
only _active_ on sheets where both `vi` and `vj` have incident faces. But Pass 2 still
rewrites `d→s` in **all** faces of `d` — including faces on non-active sheets — without
adding those faces to `decIM`.

```
manifold:     every Pass 2 rewrite → active sheet → tracked in decIM → BF never stale ✓
non-manifold: some Pass 2 rewrites → non-active sheet → NOT tracked → BF goes stale  ✗
```

---

## SheetData is the time-capsule

At the moment each collapse `dIdx` executes, before any later rewrite can touch those
faces, the code captures a snapshot:

```
sd.subsetVIdx  = global vertex indices AS THEY ARE at dIdx time
sd.FUV_pre     = face connectivity in LOCAL indices (into subsetVIdx), pre-collapse
sd.FIdx_pre    = global gF row for each FUV_pre row
sd.FUV_post    = face connectivity post-collapse (flap faces removed, d→s remapped)
sd.FIdx_post   = global gF row for each FUV_post row
sd.UV_pre      = LSCM UV of every one-ring vertex BEFORE the collapse
sd.UV_post     = LSCM UV of every one-ring vertex AFTER the collapse
```

`sd.subsetVIdx(sd.FUV_pre(r, k))` = global vertex index at column k of face
`FIdx_pre(r)` **as it was at collapse `dIdx`** — not after any later non-active rewrite.

---

## Confirmed diagnostic output

`cerr` logging added before the `break` confirmed the hypothesis on 5 representative
failures:

```
[q2f STALE] qIdx=539 dIdx=25 queryFIdx=859 BF=(100,609,608) lookup_sizes=(0,1,1)
           FIdx_pre_row=0  correct=(610,609,608)  found=(1,1,1)
           col-by-col:  [0] BF=100 correct=610 STALE  [1] BF=609 ok  [2] BF=608 ok

[q2f STALE] qIdx=551 dIdx=274 queryFIdx=2138 BF=(809,645,640) lookup_sizes=(1,1,0)
           FIdx_pre_row=...  correct=(809,647,642)  found=(1,1,1)
           col-by-col:  [0] ok  [1] BF=645 correct=647 ok  [2] BF=640 correct=642 STALE
```

In every failure:
- All **correct** vertices (from `FIdx_pre`) are found in `subsetVIdx` with `size==1` ✓
- The stale `BF` values are either absent or at the wrong position

---

## Two failure modes

### Type A — vertex not found at all (`size == 0`)

`BF[k]` carries a post-rename value `S` that is completely absent from `subsetVIdx`
at `dIdx`. e.g. `BF[0]=100`, but `subsetVIdx` has `610` at that face column.

Caught immediately by the size check. Causes `break` → query is abandoned.

### Type B — vertex found but at the wrong position (`size == 1`, wrong vertex)

`BF[k]` carries `S` which **happens to be a neighbor vertex** in the one-ring at `dIdx`,
so `subsetVIdx` does contain `S` — but at the position of a _different_ face corner.
e.g. `BF[1]=645`, found in `subsetVIdx`, but the face's actual column-1 vertex at this
collapse level is `647`. The size check passes, the query is not abandoned, but
`UV_post[local(645)]` is the wrong UV → silently wrong correspondence.

---

## Fix

### Part 1 — initialization (`coarse_fine_viz.cpp`)

Initialize `BF` from `sd.FIdx_post` (the snapshot's post-collapse state at `dIdx`)
instead of from `gF`. `gF` reflects ALL subsequent non-active rewrites; `FIdx_post`
reflects only the state at `dIdx`. Implemented in `coarse_fine_compute_and_save()`.

### Part 2 — backward walk (`query_coarse_to_fine.cpp`)

Replace the direct `subsetVIdx` search for `BF` values with a `FIdx_pre` column lookup:

```cpp
// Primary: find queryFIdx in FIdx_pre → column-aligned local indices.
// Handles both Type A (vertex absent) and Type B (vertex at wrong position).
int pre_row = -1;
for (int r = 0; r < (int)sd.FIdx_pre.size(); r++)
    if (sd.FIdx_pre(r) == queryFIdx) { pre_row = r; break; }

if (pre_row >= 0) {
    v0 = sd.FUV_pre(pre_row, 0);
    v1 = sd.FUV_pre(pre_row, 1);
    v2 = sd.FUV_pre(pre_row, 2);
} else {
    // Fallback: direct subsetVIdx search (cross-sheet / manifold edge-case).
    VectorXi v0_vec, v1_vec, v2_vec;
    igl::find((sd.subsetVIdx.array() == f(0)).eval(), v0_vec);
    igl::find((sd.subsetVIdx.array() == f(1)).eval(), v1_vec);
    igl::find((sd.subsetVIdx.array() == f(2)).eval(), v2_vec);
    if (v0_vec.size() != 1 || v1_vec.size() != 1 || v2_vec.size() != 1) break;
    v0 = v0_vec(0); v1 = v1_vec(0); v2 = v2_vec(0);
}
```

**Why column order is preserved:** non-active-sheet collapses only change the _value_
stored at a face column (`d→s`), never the column index itself. So `FUV_pre(pre_row, k)`
always aligns with `BF` column `k` regardless of how many non-active rewrites happened
between `dIdx` and now.
