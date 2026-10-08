# `coarse_subdiv_relaxed_<stem>.sdt`: the relaxed coarse ↔ fine correspondence

`run_relax.py` (in `relaxation_scripts_python/`) writes this file after every relaxation, next
to the relaxed OBJ, in the output folder.

It uses the **same binary format** as the `subdiv_<stem>.sdt` that `collapse_viz_bin` writes
(`subdiv_sample_tracker/subdiv_tracker.cpp`, `subdiv_tracker_save`), in its relaxed variant
(version 2, 15 arrays). The existing reader `subdiv_sample_tracker/tools/read_sdt.py` reads it
unchanged.

## What the samples are

The samples are the vertices of the **subdivided coarse mesh** after the relaxation on the fine
MAT. Every sample has two locations:

- **coarse side:** `coarse_face`, `coarse_bary`. This is where the sample was made on the coarse
  (decimated) mesh. It is not changed by the relaxation.
- **fine side:** `fine_face`, `fine_bary`. This is where the sample is on the fine MAT after the
  relaxation.

Together they are a dense set of point pairs (coarse point ↔ fine point). The sub mesh
(`sub_F`) connects them, so they define a piecewise-linear map between the coarse mesh and the
fine MAT.

The direction is the opposite of the C++ file. The C++ file subdivides the **fine** mesh and
records where its samples land on the coarse mesh. This file subdivides the **coarse** mesh and
records where its samples are on the fine mesh after relaxation.

## Meshes the indices refer to

| name in this doc | what | where to get it |
|---|---|---|
| fine mesh (gVO, gFO) | the fine MAT, in the order of the bundle / the input `.obj` | the bundle `correspondence_<stem>.c2f` (fine V / F), or the MAT `.obj` |
| coarse mesh (Vbase, Fout) | the decimated mesh | `simplified_<stem>.obj` of the C++ run (same face order as the bundle's coarse mesh) |

`<stem>` is the bundle's stem. For the default input it is
`mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00`, from the
C++ run `output/relaxation_experiments/src_qslim200_valid/`.

## Layout

The file is little-endian. A header comes first, then 15 arrays. Each array starts at an
8-byte-aligned offset given in the header, with zero padding between arrays.

### Header (232 bytes)

| offset | type | field | value in this file |
|---|---|---|---|
| 0 | char[8] | magic | `"SUBDIVT\0"` |
| 8 | uint32 | version | 2 |
| 12 | uint32 | header_bytes | 112 + 8 × 15 = 232 |
| 16 | uint32 | n_levels | number of uniform 1→4 subdivision levels (0 when equal-area splitting made all the samples) |
| 20 | uint32 | relaxed | 1 |
| 24 | uint64 | n_samples_requested | the sample count asked for (`--n_coarse_subdiv_samples`, or `--equal_area_samples`) |
| 32 | uint64 | n_sub_verts (Vs) | number of samples |
| 40 | uint64 | n_sub_faces (Fs) | number of sub faces |
| 48 | uint64 | n_fine_verts | \|gVO\| |
| 56 | uint64 | n_fine_faces | \|gFO\| |
| 64 | uint64 | n_orig_edges (nE) | number of fine-mesh edges |
| 72 | uint64 | n_coarse_verts | \|Vbase\| |
| 80 | uint64 | n_coarse_faces | \|Fout\| |
| 88 | uint64 | n_palette (P) | number of struct-id sets |
| 96 | uint64 | n_palette_ids | total ids in the palette |
| 104 | uint64 | n_arrays | 15 |
| 112 | uint64[15] | offsets | byte offset of each array |

### Arrays

All indices are 0-based. Barycentric coordinates are in the corner order of the referenced
face row.

| # | name | type | shape | meaning in this file |
|---|---|---|---|---|
| 0 | `sub_V` | float64 | Vs × 3 | sample positions on the fine MAT **after relaxation** |
| 1 | `sub_F` | int32 | Fs × 3 | the sub mesh's triangles, i.e. the connectivity the relaxation used (its Laplacian graph) |
| 2 | `sub_face_orig` | int32 | Fs | the **coarse** face (Fout row) each sub face belongs to. The C++ file stores the fine face here. |
| 3 | `orig_edges` | int32 | nE × 2 | fine-mesh edges (min, max), indexed by EDGE carriers |
| 4 | `carrier_type` | uint8 | Vs | carrier of the sample's **seed** (its position before relaxation) on the fine mesh: 1 EDGE (samples on a seam / boundary), 2 FACE (all others). 0 VERTEX is not used. |
| 5 | `carrier_index` | int32 | Vs | FACE: the seed's fine face (gFO row). EDGE: the seed's fine edge (`orig_edges` row). |
| 6 | `fine_face` | int32 | Vs | fine face (gFO row) of the **relaxed** position |
| 7 | `fine_bary` | float64 | Vs × 3 | barycentric coordinates of the relaxed position in `gFO[fine_face]`. On a seam / boundary edge, the coordinate of the third corner is 0. |
| 8 | `coarse_face` | int32 | Vs | coarse face (Fout row) the sample was made in |
| 9 | `coarse_bary` | float64 | Vs × 3 | barycentric coordinates in `Fout[coarse_face]`, exact dyadic values (made by edge midpoints) |
| 10 | `struct_set_id` | int32 | Vs | index of the sample's struct-id set in the palette |
| 11 | `palette_offsets` | int32 | P + 1 | set k = `palette_ids[offsets[k] : offsets[k+1]]` |
| 12 | `palette_ids` | int32 | n_palette_ids | `.ma_struct` struct ids |
| 13 | `palette_type_mask` | uint8 | P | bit 0 sheet, bit 1 seam, bit 2 boundary, bit 3 junction (OR over the set's ids) |
| 14 | `sub_V_seed` | float64 | Vs × 3 | sample positions **before relaxation**, i.e. the relaxation input on the fine MAT |

### Role of a sample (from its type mask)

- junction bit set → **junction**: fixed at a fine-mesh junction vertex
- else seam or boundary bit set → **curve**: lies on a fine edge of one of its seams / boundaries
- else → **sheet**: lies on a fine face of one of its sheets

## Reading it

```python
import sys
sys.path.insert(0, "subdiv_sample_tracker/tools")
from read_sdt import read_sdt, barycentric_positions, struct_sets

H, A = read_sdt("coarse_subdiv_relaxed_<stem>.sdt")
# relaxed fine positions, from the correspondence (equal to A["sub_V"])
P_fine = barycentric_positions(fineV, fineF, A["fine_face"], A["fine_bary"])
# coarse positions of the same samples
P_coarse = barycentric_positions(coarseV, coarseF, A["coarse_face"], A["coarse_bary"])
ids, masks = struct_sets(A)   # per-sample struct ids and type masks
```

Here `fineV, fineF` is the fine MAT and `coarseV, coarseF` is `simplified_<stem>.obj`. The
reader's `read_obj` loads both.

`level_faces(A, H, level)` (coarser subdivision levels) only applies when the samples come
from uniform subdivision with no Delaunay flips. In that case `level_faces(A, H, 0)` returns
exactly the coarse faces. With equal-area splitting (`n_levels` = 0) or flips, `sub_F` is not
nested, so the helper does not apply.

## Guarantees (checked by `relaxation_scripts_python/check_relaxed_sdt.py`)

- `(fine_face, fine_bary)` interpolated on the fine mesh equals `sub_V` exactly (difference 0).
- `(coarse_face, coarse_bary)` interpolated on the coarse mesh equals the sample's position on
  the coarse geometry exactly.
- Both sets of barycentric coordinates are ≥ 0 and sum to 1 (up to 2.2e-16).
- `sub_V_seed` equals the relaxation input (`relax_input/04_relax_input_<stem>.ply` of the run).
- `sub_V` and `sub_F` equal the relaxed OBJ written by the same run (15 digits).
- Every EDGE carrier is a fine edge of one of the sample's seams / boundaries.
- The palette type masks equal the `.ma_struct` types of their ids.
- Every sheet sample's relaxed fine face is in one of its sheets. Every curve sample's relaxed
  position is on a fine edge of one of its seams / boundaries.
- Junction samples sit on their junction vertex.
- Every sample is mapped: no `fine_face` or `coarse_face` is -1.

Check a run with:
`python relaxation_scripts_python/check_relaxed_sdt.py --run_dir <run_relax output folder>`.

On ABC 00040057 with the default bundle, 277,700 samples come to about 38 MB.
