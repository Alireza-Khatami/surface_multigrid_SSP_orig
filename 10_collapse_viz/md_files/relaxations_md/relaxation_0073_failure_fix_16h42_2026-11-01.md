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

## Update (2026-11-01, 21:13): no-sheet faces as one extra sheet, with a pseudo boundary

Commits `55031ce` and `77134f4`. This tries the open point above.

### Change 3: `--no_sheet_mode sheet|fixed|global` (`projector.py`, `run_relax.py`)

A new `run_relax.py` flag picks what happens to the samples with no sheet ID:
- **`sheet`** (the new default): the no-sheet faces and the no-sheet sample sets get the pseudo
  sheet ID −1. The samples are projected onto the no-sheet faces only, so they slide inside them
  but never leave them.
- **`fixed`**: change 2 above; no target, the samples stay at their seeds.
- **`global`**: the original rule; projected onto all fine faces. Gives a byte-identical `.sdt` to
  the earlier run.

Without a `.ma_struct` all modes behave as before. The mode is written to `experiment_config.txt`.
The pseudo sheet −1 lives only inside the projector; it is not written to the `.sdt`.

**Effect on the pipeline.** `c2f_preprocess.py` passes no flag, so every new run uses `sheet`.

### Problem with `sheet` alone: 11 degenerate triangles

- 10 of them lie on fine face 1496 (coarse face 485). Their corners are no-sheet samples that
  ended on or next to the edge 724–726: the triangles were flattened into a line.
- **Why.** Edge 724–726 is an open edge of the MAT (only face 1496 has it), but the `.ma_struct`
  gives it no boundary ID. Nothing holds that border, so the smoothing pulls the samples outwards
  and the projection clamps them onto the edge (44 samples on one line).
- The 11th (face 504773, on sheet 52) is degenerate in every mode; it has nothing to do with this.

### Change 4: a pseudo boundary for those edges (`matstruct.py`, `run_relax.py`, `check_relaxed_sdt.py`)

- **What.** In `sheet` mode, every open edge of a no-sheet face that has no seam/boundary ID gets
  one new BOUNDARY struct, like a boundary listed in the file: the edge and its two endpoints get
  the ID. Its samples become curve samples that slide along the edge, and the sheet samples stop
  at them.
- **The ID** is max struct ID + 1 (57 on 0073). It is a real ID in the `.sdt` palette, not −1.
- **Where.** `matstruct.add_no_sheet_borders(ms, FO)`. `run_relax.py` calls it after the c2f
  build, so the seeds do not change. It is deterministic: `check_relaxed_sdt.py` rebuilds the same
  ID when it finds it in the palette. (The c2f positions do not change. A sample that lies on a
  new pseudo-boundary edge becomes a curve sample and is re-snapped onto it, which moves its seed
  by rounding only; see "Regression on other shapes".)
- **On 0073:** 3 such edges: 724–726 (face 1496), 662–727 (face 1244), 727–853 (face 1503).
- **Effect on the pipeline.**
  - `experiment_config.txt` gets the line `no-sheet pseudo boundary: id 57, 3 unlabelled open edges`.
  - The struct-ID check passes with the 3 extra curve edges (216 coarse curve edges instead of 213).
  - **Readers of the `.sdt` that load the original `.ma_struct`** (e.g. the pretrainer's
    `carrier_samples.py`) do not know ID 57: they see curve samples with no known curve. They have
    to call `add_no_sheet_borders` after loading, as the checker does. Not done yet.
  - `relax_viewer.py` does not add the pseudo boundary yet.
  - Vertex 724 now lies on two curves (26 and 57), so its samples can slide along either.

### Results on 0073

Same bundle (`subdiv_n1000000_nosheet_lock/`) and settings as above. Output folders:
`relaxation_nosheet_global/`, `relaxation_nosheet_fixed/`, `relaxation_nosheet_sheet/` (change 3
only) and `relaxation_nosheet_sheet_border/` (changes 3 + 4).

| relaxed result | global | fixed | sheet | **sheet + pseudo boundary** |
|---|---|---|---|---|
| edge CV (seed 0.2024) | 0.2001 | **0.1989** | 0.1999 | 0.1992 |
| min angle p1 (seed 21.75) | 20.80 | 22.81 | 21.22 | **23.08** |
| min angle p5 / median | 40.37 / 53.54 | 40.37 / 53.54 | 40.53 / 53.55 | **40.80** / 53.56 |
| flipped triangles (seeds 29) | **21** | 26 | 22 | 22 |
| folded one-rings (seeds 92) | 18 | 42 | 17 | **12** |
| degenerate triangles | 1 | 1 | 11 | **1** |
| no-sheet samples left on no-sheet faces | 853 / 1044 | 1044 / 1044 | 1044 / 1044 | 975 / 975 |
| `check_relaxed_sdt.py` | pass | pass | pass | pass |

With the pseudo boundary, 69 of the 1,044 no-sheet samples are now curve samples on the new
boundary, so 975 are left with no sheet ID.

### What this means

- **`sheet` + pseudo boundary is the best setting on 0073:** best p1, p5 and folded one-rings, edge
  CV close to the best, and no extra degenerate triangles. Only flipped triangles are one more than
  `global`.
- The degenerate triangles came from a gap in the `.ma_struct` labelling (a border with no
  boundary ID), not from the relaxation.

## Regression on other shapes (2026-11-01, 21:17)

**Question.** Do the code changes (c2f walk fix, 1-D MAT edges skipped, `--no_sheet_mode`, pseudo
boundary) change the result on shapes other than 0073?

**How.**
- **Shapes:** each with its existing bundle in `Experiments/<mesh>/mats_*/…` and the `.ma_struct`
  in `Data/ABC_Dataset/<mesh>/`.
- **Setting:** the best one (equal-area split to 277,700 samples, Delaunay flips, momentum β 0.9,
  two-sided graph, no new folds, 300 iterations).
- **Three runs per shape:**
  - **old:** commit `b76272c` (before all of these changes);
  - **new-global:** HEAD with `--no_sheet_mode global`;
  - **new-default:** HEAD with the new default `sheet`.
- **Comparison:** the relaxed `.sdt` files, byte for byte.
- **Not covered:** the C++ no-sheet collapse lock (change 1). It only matters when a bundle is
  regenerated; these runs reuse the existing bundles.

| shape | bundle | fine faces in no sheet | old c2f: unmatched / off face | old = new-global | old = new-default |
|---|---|---|---|---|---|
| 01_00040057 | `mats_1506_interp_20/c2f_n1000000` | 0 / 2,715 | 0 / 0 | **identical** | **identical** |
| 01_00040093 | `mats_2595_interp_20/subdiv_n1000000` | 1 / 5,078 | 0 / 0 | **identical** | differs (expected) |
| 01_00040563 | `mats_2741_interp_20/subdiv_n1000000` | 0 / 5,260 | 0 / 0 | **identical** | **identical** |
| 01_00040049 | `mats_2721_interp_20/c2f_n1000000` | 0 / 5,329 | 0 / 0 | **identical** | **identical** |

0051, 0052, 0053 and 0092 were in the list too. The run was stopped during 0051 (the system ran
low on memory), so those four are not tested.

**0093, the one difference, is the intended change.**
- **What triggers it.** It has one fine face in no sheet. In `sheet` mode that face's unlabelled
  open edge becomes a pseudo boundary (ID 37), and the palette grows from 56 to 58 sets.
- **Seeds.** 3 samples lie on that edge. They become curve samples (FACE → EDGE carrier) and are
  re-snapped onto the edge, which moves them by at most 8.8e-17 × diag: rounding. 7 carriers
  change type in all.
- **Relaxed positions.** That small change spreads through the relaxation: 167,819 samples end
  in a slightly different place, on average 6.9e-6 × diag apart (at most 0.0042).
- **Quality is unchanged in practice:**

  | 0093 relaxed | old | new-default |
  |---|---|---|
  | edge CV | 0.1859 | 0.1859 |
  | min angle p1 / p5 / median | 37.567 / 44.579 / 54.219 | 37.572 / 44.578 / 54.219 |
  | degenerate | 0 | 0 |
  | bijectivity: flipped triangles / folded one-rings | 11 / 0 | 10 / 0 |

**Conclusion.**
- **Without no-sheet faces** (0057, 0563, 0049), the result is byte-identical to the old code with
  the default settings.
- **With no-sheet faces**, `--no_sheet_mode global` gives exactly the old result (0093). The
  default `sheet` changes it only through the pseudo boundary and the no-sheet samples, by
  rounding-level amounts on 0093.
- **The c2f walk fix and the 1-D-edge skip** change nothing on these shapes: their old walks
  already had 0 unmatched and 0 off-face samples, and none has a 1-D MAT part.
