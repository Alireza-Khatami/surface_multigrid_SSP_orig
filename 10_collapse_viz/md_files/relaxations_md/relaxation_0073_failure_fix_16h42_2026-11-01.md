# ABC 01_00040073: why the relaxation failed, and the fix

Shape `01_00040073_4707da1eed9441d9a8889f6a_trimesh_000`.
- **Bundle:** `Experiments/<mesh>/mats_893_interp_20/subdiv_n1000000/correspondence_<mesh>_mat_initial.c2f`
  (2026-10-30).
- **`.ma_struct`:** `Data/ABC_Dataset/<mesh>/<mesh>_mat_initial.ma_struct`.
- **Relaxation:** the best setting (equal-area split to 277,700 samples, Delaunay flips, momentum
  β 0.9, two-sided graph, no new folds, 300 iterations).

## Symptom

`run_relax.py` stopped with "coarse structure IDs are inconsistent". The log also showed:
- 19,635 samples **unmatched** in the c2f walk and 3,536 that **ended off their fine face**;
- 33 coarse vertices on no coarse face;
- in the `.ma_struct`: 10 fine faces in no sheet, and 3 seam/boundary edges on no face.

**Regenerating the bundle doesn't help.** The old bundle (`c2f_n1000000`, 2026-09-23) and the new
one have exactly the same numbers. There are two separate causes: one in the shape's data, one in
our code.

## Cause 1: the MAT has a 1-D part (the strict check failed)

- **The data.** 33 fine MAT vertices (860–891) lie on edges only, not on any triangle. The MAT
  continues there as a curve.
- **How the curves reach it.** Boundary curves 16 and 26 run onto that part through 3 edges:
  677–889, 721–890 and 724–891. Each joins a vertex that lies on triangles to one that doesn't,
  and none is an edge of a face. These are the "3 seam/boundary edges on no face".
- **What the collapse does.** It keeps these vertices as the 33 coarse vertices on no coarse
  face. The 3 curve edges then join a coarse vertex that lies on faces to one that doesn't, so
  they map to no coarse edge.
- **Result:** the struct-ID check counted them as "3 not a coarse edge" and stopped the run.

**Fix** (`struct_ids.py`, `coarse_matstruct`):
- A fine curve edge whose coarse end lies on no coarse face is skipped and counted, instead of
  failing the run. Log: "3 to a coarse vertex on no coarse face (1-D MAT part, skipped)".
- No sample can lie on such an edge, so it carries no IDs for the subdivided surface.
- A curve edge that really is broken (both ends on faces, but no coarse edge between them) still
  fails the run.

## Cause 2: a bug in the c2f walk (the unmatched and off-face samples)

**How the walk picks a sheet.** To place a sample, the walk goes back through the collapses that
touched its face. For each one it uses the collapse's sheet record whose global sheet ID equals
the face's sheet ID, and finds the face in that sheet's list of pre-collapse faces.

**What this shape has.** One collapse stores two sheet records with the same global ID: the sheets
are (0, 4, 0). That happens when a sheet meets itself along a seam.

**What went wrong.** The walk took the **first** record with a matching ID, which doesn't contain
the face:
- **At the start of the walk:** 30 coarse faces were left **unmatched**. All 30 faces are in the
  second record of sheet 0 of collapse 503. That's the 19,635 samples.
- **During the walk:** the step was silently skipped (`continue`), so the walk carried on with
  older collapses and the sample **ended off its fine face**. That's the 3,536 samples.

**Fix** (`c2f_walk.py`, in the start of the walk and in the walk kernel):
- Among the sheet records with a matching ID, use the one that contains the face.
- When IDs are unique, the result is the same as before. Checked on the 0093 bundle: the c2f
  positions and fine faces are bit-identical to its earlier run (`relax_input/03_subdiv_at_fine`
  PLY).

**Not changed:** our Python walk is a port of the C++ `query_coarse_to_fine`, and copied the same
rule. So the C++ query probably has the same bug, and maybe the pretrainer's
`11_correspond_viz/c2f_query.py` too. Neither has been checked.

## Result after both fixes

Output: `mats_893_interp_20/subdiv_n1000000/relaxation/`.

| | before the fixes | after |
|---|---|---|
| c2f: unmatched / off their fine face | 19,635 / 3,536 | **0 / 0** |
| struct-ID check | failed (3 not a coarse edge) | pass (3 edges of the 1-D part skipped) |
| relaxation | failed | done; all consistency checks 0 |
| `check_relaxed_sdt.py` | — | **pass** |
| edge CV | — | 0.2034 → 0.1998 |
| min angle p1 / p5 / median | — | 21.33 → 20.80 / 32.55 → 40.18 / 44.67 → 53.63 |
| degenerate triangles | — | 0 → 1 |
| bijectivity: flipped triangles | — | seeds 22 → relaxed 31 |
| bijectivity: folded one-rings | — | seeds 92 → relaxed 21 |

The 33 coarse vertices on the 1-D part stay unmapped, which is correct. The `.sdt` marks them with
face −1.

## Still open on this shape

- **The relaxation helps less than on 0057 or 0093.**
  - Edge CV barely moves (0.2034 → 0.1998) and p1 drops slightly (21.33 → 20.80).
  - Flipped triangles go up (22 → 31), while folded one-rings go down (92 → 21).
- **A likely contributor (not tested):** the 10 fine faces in no sheet.
  - Their 1,235 samples have no sheet ID, so they are projected onto the whole MAT (global
    target) and not kept on their own sheet.
  - Faces like 1496 lie entirely inside sheet 44, so the `.ma_struct` sheet labelling looks
    incomplete there.
