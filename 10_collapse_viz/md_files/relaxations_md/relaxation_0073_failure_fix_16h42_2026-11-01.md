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

## Update (2026-11-01, 16:56): the 10 faces in no sheet

Commit `5e6e9c7`. Two changes, both about the 10 fine faces that are in no sheet of the `.ma_struct`
(faces 662, 663, 1244, 1261, 1266, 1484, 1496, 1503, 1562, 1563).

### Change 1: these faces are never collapsed (`main.cpp`)

- **What.** A collapse of edge (u, v) changes every face on u or v. So the corners of the no-sheet
  faces are locked: any collapse using one of them is blocked. It works like the stale-chain lock.
- **When.** Only with `--mat_struct_check` (which `c2f_preprocess.py` passes by default).
- **Log.** `[NO-SHEET] pre-collapse lock: 10 fine faces in no sheet, 18 corner vertices locked`;
  each blocked collapse is `BLOCK-NO-SHEET-FACE` in `struct_gate_log.txt`.
- **Effect on the pipeline.**
  - The coarse mesh keeps these 10 faces exactly as in the fine mesh (checked: 10 / 10).
  - The coarse mesh has a few more faces: 527 instead of 514 (26 collapses blocked).
  - Collapse tracker counters stay 0; c2f walk: 0 unmatched, 0 off their fine face.
  - The struct-ID check passes as before (3 curve edges of the 1-D part skipped).

### Change 2: samples with no sheet ID are not projected (`projector.py`)

- **Before.** A sample whose set has no sheet ID was projected onto the whole MAT (global target),
  so it could slide onto any sheet.
- **Now.** With a `.ma_struct`, such a set gets **no target** (a sheet ID of "−1"). The relaxation
  already keeps a sample with no target fixed, so these samples stay at their c2f seed positions.
  Because of change 1, those seeds lie on the original no-sheet fine faces.
- **Without a `.ma_struct`** nothing changes: every set is empty, so all samples still use the
  global target.
- `check_relaxed_sdt.py`: only the message changed ("no target, kept fixed").
- **Effect on the pipeline.**
  - Every run with a `.ma_struct` is affected, not only 0073: any shape with no-sheet faces now
    has fixed samples there. Shapes without such faces give the same result as before.
  - On 0073: 1,044 samples kept fixed, none moved; all consistency checks 0;
    `check_relaxed_sdt.py` passes.

### Results on 0073

Run with the `c2f_preprocess.py` defaults (40 target faces, 1M subdiv samples, qslim, validity
checks, struct check; equal-area 277,700, momentum β 0.9, 300 iterations, Delaunay flips).

- Output: `mats_893_interp_20/subdiv_n1000000_nosheet_lock/`
  - `relaxation/`: change 1 only (no-sheet samples still use the global target);
  - `relaxation_nosheet_fixed/`: changes 1 and 2.

| | before the update | change 1 | changes 1 + 2 |
|---|---|---|---|
| coarse faces | 514 | 527 | 527 |
| edge CV (seed → relaxed) | 0.2034 → 0.1998 | 0.2024 → 0.2001 | 0.2024 → **0.1989** |
| min angle p1 | 21.33 → 20.80 | 21.75 → 20.80 | 21.75 → **22.81** |
| min angle p5 / median (relaxed) | 40.18 / 53.63 | 40.37 / 53.54 | 40.37 / 53.54 |
| flipped triangles (seeds → relaxed) | 22 → 31 | 29 → **21** | 29 → 26 |
| folded one-rings (seeds → relaxed) | 92 → 21 | 92 → **18** | 92 → 42 |
| `check_relaxed_sdt.py` | pass | pass | pass |

### What this means

- **Change 1** helps the bijectivity: fewer flipped triangles and folded one-rings after
  relaxation.
- **Change 2** helps the mesh quality (edge CV, and p1 now goes up instead of down), but gives more
  folds than the global target. The fixed patches act as walls: their neighbours cannot even out
  across them, so folds collect at the patch borders.
- **Not tried yet:** treat all no-sheet faces as one extra sheet, so their samples can slide inside
  those faces but never leave them.
