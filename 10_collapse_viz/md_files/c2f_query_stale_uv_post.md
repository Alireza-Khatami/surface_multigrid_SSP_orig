# Coarse -> fine query: stale `UV_post` row of the absorbed vertex

Status: **confirmed and fixed** (2026-09-28). Verification results and fix below.

## Symptom

Mapping points from the coarse mesh to the fine mesh with
`query_coarse_to_fine` (`src/query_coarse_to_fine.cpp`) leaves some fine
triangles without any mapped point ("missing triangles"), while others get too
many.

## Suspected cause

Each backward step of the walk takes a point given as barycentrics `BC` on a
face that exists **after** collapse `dIdx`, and builds its position in that
collapse's post-collapse UV layout:

```cpp
int v0 = sd.FUV_pre(pre_row, 0);   // local corner ids of the face BEFORE the collapse
int v1 = sd.FUV_pre(pre_row, 1);
int v2 = sd.FUV_pre(pre_row, 2);
queryUV = BC0 * sd.UV_post.row(v0) + BC1 * sd.UV_post.row(v1) + BC2 * sd.UV_post.row(v2);
```

- `FUV_pre` is the face before the collapse. If the face had the absorbed
  vertex `d` (local `b(1)`), that corner is still `d`. After the collapse the
  same column holds the survivor `s` (local `b(0)`).
- `UV_post` is built in every `joint_lscm` variant as
  `UV_post = UV_pre; UV_post.row(vi) = <new UV of the survivor>;`, so
  `UV_post.row(b(1))` is **d's pre-collapse UV**, not s's post-collapse UV.
- So for every surviving face that touched `d`, the query triangle has one
  corner at the wrong place. The point is located in the wrong part of the
  pre-collapse ring, and the error compounds over the walk's steps.

## History

- `8490afa` (original SSP): corners looked up by current vertex id
  (`BF` -> `subsetVIdx`). For a surviving face that id is `s` -> correct UV.
- `a8ac091` (2026-07-13, "fix stale-vertex bug"): switched to `FUV_pre`
  columns so that `BF` ids made stale by non-active-sheet remaps could not
  break the lookup. That fixed the lookup but introduced this bug.

## Fix

Keep the column alignment; before reading `UV_post`, replace the absorbed
vertex by the survivor:

```cpp
if (v == sd.b(1)) v = sd.b(0);   // for v0, v1, v2
```

Applied in every copy of the lookup:

- `src/query_coarse_to_fine.cpp` (the query itself)
- `10_collapse_viz/coarse_fine_viz.cpp`, `traced_walk` (viewer's single-vertex trace)
- `11_correspond_viz/c2f_query.py`, `post_corners()` used by `query_coarse_to_fine`
  and `query_point_with_intermediates`

## Related, same path

- `coarse_fine_compute_and_save` (`c2f_*.txt`): when the coarse vertex is not
  in the face's `FUV_pre` row (it is `s`, the row has `d`), it falls back to
  the current `gF` order with all weight on `d`'s column. Through the bug above
  the coarse vertex then lands at `d`'s old position. The same fix covers it.
- Seam steps with no matching sheet are skipped (`b8e1e2e`); the point keeps its
  post-collapse face/bary and continues to older steps untransformed. Not
  observed on the test mesh (the fixed round trip is exact for every sample),
  but still possible in principle.

## Verification plan

Ground truth: the subdivided-sample tracker. Every sample has an exact fine
position and was cast forward through every collapse onto the coarse mesh.
Running the query on each sample's final coarse position must return the
sample's fine position (up to clamping of samples that fell outside the post
ring during the forward cast).

1. Check the data: `UV_post.row(b(1)) != UV_post.row(b(0))` in stored collapses.
2. Round trip, current query vs fixed query: distance to the true fine position,
   and whether the final fine face matches.
3. Coverage: fine faces that receive no mapped sample.

## Results (ABC 00040057, target 200, qslim, `--n_subdiv_samples 200000` -> 350,016 samples, 1245 collapses)

The decimation, the simplified mesh and the `.sdt` are byte-identical with and
without the fix; only the query changes.

**1. Data.** In all 2077 stored collapse sheets, `UV_post.row(b(1))` equals
`UV_pre.row(b(1))` and differs from `UV_post.row(b(0))` (max gap 1.28 in UV).
4353 surviving pre-collapse faces had `d` as a corner.

**2. Round trip (C++).** Each sample's tracked coarse position through
`query_coarse_to_fine`, compared with the sample's true fine position
(error / bbox diagonal). "Clamped" = the 70 samples that fell outside the post
ring by ~1e-12 during the forward cast.

| | before fix | after fix |
|---|---|---|
| median error | 1.8e-2 | 1.7e-16 |
| max error | 0.23 | 9.0e-13 |
| samples with error > 1e-3 | 325,956 of 349,946 | 0 |
| clamped samples, max error | 6.0e-2 | 3.3e-13 |
| samples ending on the wrong fine face | 287,339 | 0 |
| **fine faces with no mapped sample** | **461 of 2715** | **0** |

The forward tracker is an independent implementation (it uses `FUV_post`), so
an exact round trip confirms the fixed query is its inverse.

**3. Python mirror** (`c2f_query.py`, on the same run's bundle):

- Coarse-face corners vs the C++ `c2f_*.txt` correspondences: max difference
  0.19 x diag before the fix, 5.8e-7 after (the 6-digit precision of the text
  file).
- 100,000 random points (250 per coarse face): 629 fine faces without a point
  before the fix, 14 after. The 14 are a sampling artifact: with 300,000
  area-proportional points every fine face is hit, and those 14 get 0.83-1.68x
  the median hits per area.

Note: `c2f_query.py`'s `load_bundle` reads bundles up to v6; the current writer
produces v7 (`md_files/c2f_bundle_v7_stale_chains.md`). For the test the v7
file was converted to v6 (no stale chains on this mesh, so only two empty
fields differ).

## Effect on existing outputs

- `c2f_*.txt` and the `.c2f` bundle's correspondence block change: coarse
  vertices whose face last touched an absorbed vertex no longer land at that
  vertex's old position.
- Everything computed before 2026-09-28 with `query_coarse_to_fine` since
  `a8ac091` (2026-07-13) is affected.
