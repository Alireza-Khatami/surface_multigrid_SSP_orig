# `b_si = -1` Bug Analysis

## What `b_si` is

`b_si` is a 2-element vector of **local indices** within a sheet's compact local mesh:
- `b_si(0)` = local index of `vi = s` (surviving vertex, smaller global index)
- `b_si(1)` = local index of `vj = d` (absorbed vertex, larger global index)

`remove_unreferenced_lessF` assigns local indices by **sorted ascending global index**.
Because `s < d` is asserted (line 349), `b_si(0) < b_si(1)` is always true when **both
vertices are found** in the sheet's face ring. The ordering is NOT the bug.

The bug is that **one vertex is not found at all** → `b_si(*) = -1`.

---

## Confirmed Root Cause

**Infinity faces from `connect_boundary_to_infinity` pass through `collect_onering`
into `partition_onering_by_sheet`, making a sheet's face list appear non-empty when
every face in it will be filtered by `get_collapse_onering_faces`.**

### Complete causal chain

```
1. connect_boundary_to_infinity(VO, FO, gV, gF)
      Adds infinity triangles to gF for every boundary edge.
      Each infinity triangle = {boundary_v1, boundary_v2, inf_vertex}.
      NOT null faces — their corners are real vertex indices.

2. partition_into_sheets(gF, gFaceSheetID, numSheets)
      BFS skips edges where faces.size() != 2.
      After step 1, every original boundary edge has exactly 2 faces
      (one real + one infinity triangle) → BFS CROSSES these edges.
      → Infinity triangles get valid sheet IDs, same sheet as their
        adjacent real face.

3. VF built on gF (main.cpp lines 107–110)
      for (int f = 0; f < gF.rows(); f++)
          for (int c = 0; c < 3; c++)
              gVF[gF(f,c)].push_back(f);
      → VF[vi] includes infinity faces for every boundary vertex vi.

4. collect_onering(vi, Ndf, Ndv) — MISSING FILTER
      for (const int f : (*VF)[vi]) {
          if (is_null_face(f)) continue;   // only null check
          faces.push_back(f);              // infinity faces pass through
      }
      → Ndf contains infinity faces.

5. partition_onering_by_sheet(Nsf, Ndf, faceSheetID, sheets_Nsf, sheets_Ndf)
      Uses faceSheetID(f) which is valid for infinity faces (from step 2).
      → sheets_Ndf[si] populated with infinity faces.
      → sheets_Ndf[si].size() > 0   ← non-empty, LOOKS like vi has faces here.

6. Guard in inner overload:
      if (sheets_Nsf[si].empty() || sheets_Ndf[si].empty()) continue;
      → PASSES (sheets_Ndf[si] is non-empty due to infinity faces).

7. get_collapse_onering_faces(V, F, vi, vj, sheets_Nsf[si], sheets_Ndf[si], ...)
      Filters: if (isinf(V(v0,0)) || isinf(V(v1,0)) || isinf(V(v2,0)))
                   is_in_vij_NF = false;
      → ALL faces in sheets_Ndf[si] are filtered out (infinity vertex).
      → vi does NOT appear in F_ring_pre.

8. remove_unreferenced_lessF(V, F_ring_pre, V_pre_si, FUV_pre_si, ..., subsetVIdx_si)
      subsetVIdx_si is built from the vertices of F_ring_pre.
      vi is absent from F_ring_pre → vi is absent from subsetVIdx_si.

9. b_si search loop:
      for (int ii = 0; ii < subsetVIdx_si.size(); ii++) {
          if (subsetVIdx_si(ii) == vi) b_si(0) = ii;   // never found
          else if (subsetVIdx_si(ii) == vj) b_si(1) = ii;
      }
      → b_si(0) = -1

10. assert(b_si(0) < b_si(1))
      Case A: b_si(0)=-1, b_si(1)=k  → assert(-1 < k) PASSES
              but V_post_si.row(-1) = p → OUT-OF-BOUNDS CRASH
      Case B: b_si(0)=k, b_si(1)=-1  → assert(k < -1) FIRES
```

---

## Why this scenario occurs in practice

As decimation progresses, a boundary vertex `vi` can lose all its **real** mesh faces on a
given sheet (they become null — killed as flap faces of previous collapses). After this:

- `VF[vi]` for that sheet retains only the infinity triangle(s) added by
  `connect_boundary_to_infinity`.
- Those infinity triangles are not null → pass the only filter in `collect_onering`.
- The edge `(vi, vj)` may still be in the priority queue with a finite cost from before
  `vi`'s real faces were killed (stale queue entry that was never re-invalidated).
- When that edge is popped and collapsed, the chain above fires.

---

## The Fix

**Root fix — filter infinity faces in `collect_onering`:**

```cpp
auto collect_onering = [&](const int v,
                            std::vector<int> & faces,
                            std::vector<int> & verts)
{
    faces.clear(); verts.clear();
    std::set<int> vset;
    for (const int f : (*VF)[v]) {
        if (F(f,0) == IGL_COLLAPSE_EDGE_NULL &&
            F(f,1) == IGL_COLLAPSE_EDGE_NULL &&
            F(f,2) == IGL_COLLAPSE_EDGE_NULL) continue;
        // ← ADD THIS: skip infinity triangles
        if (isinf(V(F(f,0),0)) || isinf(V(F(f,1),0)) || isinf(V(F(f,2),0))) continue;
        faces.push_back(f);
        for (int c = 0; c < 3; c++)
            if (F(f,c) != v) vset.insert(F(f,c));
    }
    verts.assign(vset.begin(), vset.end());
};
```

After this fix, `sheets_Ndf[si].empty()` correctly returns `true` when `vi` has no real
faces on sheet `si`, and the existing guard correctly skips the sheet.

**Defensive fix on top — replace the assert:**

```cpp
// assert(b_si(0) < b_si(1));  ← remove
if (b_si(0) < 0 || b_si(1) < 0) continue;  // sheet has no valid ring for this edge
```

This catches any remaining edge case (e.g. the stale queue entry above fires before the
root fix could prevent it).

---

## What was ruled out

| Candidate | Verdict | Reason |
|---|---|---|
| `remove_unreferenced_lessF` assigns wrong order | **Not a cause** | Sorts by global index; since s<d always, b_si(0)<b_si(1) is guaranteed when both found |
| Stale (non-null, wrong-vertex) VF entries | **Not a cause** | After every collapse, Pass 2 updates F(f,c)=sv for all remapped faces; non-null VF entries always contain the vertex they're filed under |
| `partition_onering_by_sheet` routing error | **Not a cause** | Uses faceSheetID(f) which is correct for all face indices in gF |
| `get_collapse_onering_faces` bug | **Not a cause** | Correctly filters infinity and null faces; this function is downstream of the missing filter |



