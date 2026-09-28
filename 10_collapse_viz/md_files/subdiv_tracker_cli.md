# `collapse_viz_bin`: subdivided-sample tracking and relaxation parameters

This is the reference for other code that calls this program. It covers what
each flag does, which values to use, and what the program writes.

Background docs: `subdiv_tracker_plan.md` (tracker), `subdiv_relax_plan.md` and
`subdiv_relax_results.md` (Newton relaxation),
`subdiv_relax_solve_project_experiments.md` (solve + project, anchors).

## What the program does

1. Reads the MAT mesh (`--mesh_path`) and its structure file
   (`--matstruct_path`).
2. **Subdivides** the MAT uniformly (midpoint 1->4 split per level) until it
   has at least `--n_subdiv_samples` vertices. These vertices are the samples.
   The subdivided mesh is never decimated.
3. **Relaxes** the samples (on by default). Each sample moves along its own
   structure to even out the spacing:
   - sheet samples stay on their sheet;
   - seam/boundary samples stay on their curve;
   - junctions never move.
4. **Decimates** the MAT to `--target_faces` and tracks every sample through
   every collapse, ending with a (coarse face, barycentric) position on the
   simplified mesh.
5. Writes the samples, their connectivity, fine and coarse coordinates and
   structure ids (`.sdt`), plus OBJ/PLY meshes for inspection.

## Recommended command

```
collapse_viz_bin.exe
  --mesh_path      <mat>.obj
  --matstruct_path <mat>.ma_struct
  --target_faces   200
  --mode           qslim
  --n_subdiv_samples 200000
  --mat_struct_check
  --output_dir     <out_dir>
```

The relaxation defaults are the recommended values (`solve_project`, anchor
tolerance 3e-3), so they need not be passed.

Use the **Release** build (`build/release/Release/collapse_viz_bin.exe`). It is
headless and exits when done. The Debug build opens the viewer and is much
slower.

## Parameters

### Input / decimation

| flag | default | recommended | meaning |
|---|---|---|---|
| `--mesh_path PATH` | `bunny.obj` | the MAT `.obj` | Fine MAT mesh. |
| `--matstruct_path PATH` | none | always give it | `.ma_struct` of the same mesh: sheets, seams, boundaries, junctions. Without it, every sample is a sheet sample and relaxation ignores structure. |
| `--target_faces N` | 285 | 200 | Face count to decimate to. |
| `--mode M` | `qslim` | `qslim` | Collapse cost/placement: `qslim`, `midpoint` or `meshlab`. |
| `--mat_struct_check` | off | on | Structure gate: forbids collapses that would break the MAT structure. Needs `--matstruct_path`. |
| `--validity-checks` | off | off | Extra geometric validity checks per collapse. Slower; for debugging. |
| `--output_dir PATH` | `.` | one folder per run | All outputs go here. Keep it short: see "Paths" below. |

### Subdivided samples

| flag | default | recommended | meaning |
|---|---|---|---|
| `--n_subdiv_samples N` | off (no samples) | 50000-200000 | Subdivide until there are at least N samples. Without this flag, no samples are created and every sampling flag below is ignored. |
| `--subdiv_obj_max_verts N` | 2000000 | default | OBJ/PLY exports of the subdivided mesh are skipped above N vertices. The `.sdt` is always written. |

Each level multiplies the sample count by about 4, so N only selects the
level. Levels on ABC 00040057 (1,506 MAT vertices):

| N up to | level | samples | faces |
|---|---|---|---|
| 1,506 | 0 | 1,506 | 2,715 |
| 5,735 | 1 | 5,735 | 10,860 |
| 22,338 | 2 | 22,338 | 43,440 |
| 88,124 | 3 | 88,124 | 173,760 |
| 350,016 | 4 | 350,016 | 695,040 |
| 1,395,080 | 5 | 1,395,080 | 2,780,160 |

### Relaxation

| flag | default | recommended | meaning |
|---|---|---|---|
| `--no_subdiv_relax` | relaxation on | not given | Skip relaxation: samples stay at the exact midpoint positions. The output is then identical to the pre-relaxation code. |
| `--subdiv_relax_method M` | `solve_project` | default | How to relax (see below). |
| `--subdiv_relax_anchor_tol t` | 3e-3 | default | `solve_project` only. Adaptive seam anchors: a seam sample is fixed wherever the seam leaves the straight line between neighbouring anchors by more than `t x` the bbox diagonal. |
| `--subdiv_relax_curve_anchors N` | 0 (off) | not given | `solve_project` only. Overrides the adaptive anchors with a fixed N anchors per seam. For comparison only. |

**The two methods.**

- **`solve_project`** (recommended). One sparse 3D solve of `L x = 0` for the
  seams, projected onto the MAT. Then one solve for the sheets, with the
  projected seams fixed, projected again. Seam anchors stop closed seam loops
  from collapsing.
  - Speed: about 0.1 s at level 3.
  - Quality: the best sample spacing and angles measured.
  - Not an exact resting state: one more relaxation step would still move
    samples slightly (about 4e-4 x diag at tol 3e-3).
- **`newton`**. Repeats until one more relaxation step moves nothing (the
  exact resting state of the rule).
  - Speed: 2 s at level 3, 68 s at level 4, 9.4 min at level 5.
  - Quality: worse worst-case angles, because sheet samples get pulled against
    concave sheet rims.
  - Use it only when the exact resting state is required.

**Anchor tolerance.** Measured break points, where degenerate and flipped
triangles jump:

| level | safe | breaks at |
|---|---|---|
| 3 | <= 1e-2 | about 1.1e-2 |
| 4 | <= 7.5e-3 | about 9e-3 |

- **3e-3 to 5e-3 is safe at both levels**, with about 2x margin. 3e-3 is the
  recommended value.
- **Tighter values change almost nothing.** 1e-3 gives the same numbers as
  3e-3.
- **Keep it in diagonal units.** The break does not scale with sample spacing
  (tested at levels 3 and 4).

Measured quality at level 3 with the recommended settings (seed values are the
unrelaxed subdivision):

| | seed | solve_project, tol 3e-3 | newton |
|---|---|---|---|
| edge-length CV (lower = more even) | 0.368 | **0.354** | 0.373 |
| smallest angle, worst 1% / median | 16.1 / 40.7 deg | **17.9 / 42.4 deg** | 9.6 / 41.7 deg |
| degenerate / flipped triangles | 0 / 0 | 2 / 5 | 1 / 0 |

### Debug

| flag | default | meaning |
|---|---|---|
| `--track_face_flip F` | off | Face-flip debug tracker on fine face F. Needs `--n_subdiv_samples`. |
| `--seam_pin_fixed` | off | Seam collapses pin vj / vi / the merged point at the old fixed UV targets (-0.5, +0.5, +0.5) instead of at their 3D arc-length fractions. The old targets stretch seam samples unevenly (up to about 50x). For comparison only. |

Unknown flags are ignored silently; `--trace_vertices`, still in some launch
entries, is no longer parsed. Check flag spelling in the caller.

## Outputs (in `--output_dir`)

`<stem>` is the mesh file name without its extension. `<method>` is `newton`
or `solve_project`.

| file | when | content |
|---|---|---|
| `subdiv_<stem>.sdt` | always with samples | **Main output.** Raw binary, layout in `subdiv_tracker.cpp`, described below. |
| `subdiv_graph_<stem>.slg` | relaxed | The relaxation graph: which neighbours pull on which sample. Viewer: `subdiv_graph_viewer.exe`. |
| `subdiv_fine_seed_<stem>.obj` | relaxed | Subdivided mesh before relaxation. |
| `subdiv_fine_relaxed_<method>_<stem>.obj` | relaxed | Subdivided mesh after relaxation (the samples the tracker uses). |
| `subdiv_fine_<stem>.obj` | not relaxed | Subdivided mesh (plain midpoints). |
| `subdiv_fine_relaxed_solve_project_with_anchors_<stem>.ply` | anchored runs | Relaxed mesh in grey plus a sphere per seam anchor, one colour per seam/boundary structure. |
| `subdiv_deformed_[relaxed_<method>_]<stem>.obj` | always with samples | Samples at their tracked positions on the simplified mesh (subdivided connectivity). |
| `simplified_<stem>.obj` | always | The simplified coarse mesh. The `.sdt` coarse face indices refer to its face order. |

All the OBJs of one run share the same vertex order and faces, so they can be
compared vertex by vertex.

### `.sdt` contents

Little-endian. Version 1 has 14 arrays (no relaxation); version 2 has 15
(relaxed). The header's `relaxed` field is 0 or 1.

| # | array | type | per |
|---|---|---|---|
| 0 | `sub_V` | f64 x3 | sample: position on the fine MAT (after relaxation) |
| 1 | `sub_F` | i32 x3 | subdivided face: connectivity for Laplacians |
| 2 | `sub_face_orig` | i32 | subdivided face: the fine face it came from |
| 3 | `orig_edges` | i32 x2 | fine edge (min, max) |
| 4 | `carrier_type` | u8 | sample: seed carrier (0 vertex, 1 edge, 2 face) |
| 5 | `carrier_index` | i32 | sample: seed carrier index |
| 6 | `fine_face` | i32 | sample: fine face it lies on |
| 7 | `fine_bary` | f64 x3 | sample: barycentric on `fine_face` |
| 8 | `coarse_face` | i32 | sample: face of `simplified_<stem>.obj` |
| 9 | `coarse_bary` | f64 x3 | sample: barycentric on `coarse_face` |
| 10 | `struct_set_id` | i32 | sample: index into the structure-id palette |
| 11-13 | `palette_offsets`, `palette_ids`, `palette_type_mask` | i32 / i32 / u8 | the distinct structure-id sets. Mask bits: 1 sheet, 2 seam, 4 boundary, 8 junction. |
| 14 | `sub_V_seed` | f64 x3 | sample: position before relaxation (version 2 only) |

## Exit status and checks

- **Exit 0 on success.** Exit 1 on a fatal error, with a `[FATAL]` line on
  stderr. Examples: invalid input, an unwritable output, or failed relaxation
  consistency checks.
- **Every run checks, and aborts on failure:**
  - every sample stays on its own structure;
  - junctions do not move;
  - positions match their barycentric coordinates.

  With `newton`, the run also aborts unless one more relaxation step moves
  nothing.
- **Tracker counters.** The log line
  `[subdiv_tracker] ... collapses ... | order!=FUV_pre 0 | corner mismatch 0 | ...`
  should show all zeros. Callers can grep it as a health check.
  `outside post ring` may show a few samples at float-noise size (max
  about 1e-12); anything larger is a real tracking error, and each affected
  collapse is then logged as `[subdiv_tracker] outside: collapse ...`.

## Paths

Mesh stems in the dataset are long (about 83 characters). The program writes
through Windows' long-path form, so long output paths no longer fail to write.
But many viewers (Explorer, MeshLab) still cannot open paths over 260
characters, so keep `--output_dir` short.
