# Is the coarse → fine map bijective?

## What is checked

The pipeline:
1. splits the coarse mesh to equal area;
2. flips its edges to Delaunay;
3. sends every sample to the fine MAT with the c2f query.

Sample i lands at P[i] on fine face `fineFace[i]`. The subdivided triangles then give a piecewise
linear map from the coarse mesh onto the fine MAT. Relaxation moves the samples on the fine MAT
and keeps the triangles, so it gives a second map. This note checks both maps for bijectivity:
- every point of the fine region is reached once;
- nothing is folded over;
- nothing overlaps.

Script: `relaxation_scripts_python/check_c2f_bijective.py`.

## The four checks

1. **Vertex injectivity.** No two samples land on the same fine point. Counted exactly and within
   1e-9 × the bounding-box diagonal. If two samples coincide, the map sends two different coarse
   points to one fine point, so it isn't one-to-one.

2. **Triangle orientation.** Each image triangle (the triangle between the three samples' fine
   positions) is checked twice:
   - **Not degenerate:** its area must not be zero.
   - **Not flipped:** its normal must agree with the fine surface underneath, which is the
     normal of the fine faces its corners lie on. A flipped triangle means the map turned that
     piece of the surface over, which is a fold.

   Two details:
   - **Coarse faces wound against the fine sheet.** Some coarse faces are wound the opposite way
     to the fine sheet under them. That's a convention, not a fold. So the majority orientation
     at the c2f seeds is taken as "correct", and a triangle counts as flipped only when it
     disagrees with that majority. The relaxed mesh uses the seeds' majority.
   - **Majority per patch.** The majority is taken over a coarse patch, not a single face. A
     patch is the coarse faces joined across edges with exactly two faces and wound consistently;
     seams separate patches. A first version voted per face, and that hid folds; see "Fix: vote
     per patch" below.
   - **Seam samples.** A sample on a seam or crease edge sits on a fine face of any of the sheets
     that meet there, and that face's normal says nothing about this triangle's sheet. So only
     corners strictly inside a fine face give the reference normal; all corners are used only
     when none is.

   The existing fold count, which compares each triangle with its coarse face's normal (the
   relaxation's no-new-folds reference, 595 at the seeds), is reported alongside.

3. **One-ring check at each sheet-interior vertex.** Take a vertex inside a sheet, i.e. every
   edge around it has exactly two triangles. Lay its image triangles flat on the fine tangent
   plane at the vertex and add up their angles at the vertex, signed by orientation.
   - **One full turn (360°):** the triangles around the vertex cover a neighbourhood exactly
     once. The map is one-to-one there.
   - **Less than one turn:** some triangles are turned over (a fold). −1 turn means the whole
     fan is flipped.
   - **Two or more turns (720°):** the fan wraps around the vertex twice, so a neighbourhood
     is covered twice.

4. **c2f walk diagnostics.**
   - **Unmapped samples:** samples the c2f query could not place.
   - **Clamped walks:** walk steps where a point fell outside every triangle and was pushed back
     inside. That pushing can send nearby points to the same place.

Checks 1–3 all passing means the map is **locally bijective**: one-to-one near every sample and
every triangle. **Not tested:** two far-apart parts of one sheet overlapping each other (global
overlap). The log says so on every run.

## Where it runs

- **`run_relax.py`, on every run:**
  - after the c2f query (the "c2f seeds") and after the relaxation ("relaxed");
  - two `[bijectivity]` lines in `run.log`, a `bijectivity` block in `results.json`, and a summary
    line in `experiment_config.txt`;
  - a diagnostic only: it never fails the run.
- **Standalone, the c2f map only (no relaxation):**

  ```
  python check_c2f_bijective.py --output_dir <dir> --equal_area_samples 277700 --delaunay_flips
  ```

  It writes `bijectivity.log` and `bijectivity.json` to `<dir>`, and exits with 1 when the map
  isn't locally bijective.

## Results

**Setup B, the best setting:**
- equal-area split to 277,700 samples, Delaunay flips;
- momentum β 0.9, two-sided graph, no new folds, 300 iterations;
- run folder `py_eqs278k_flip_momentum_b09_300_bij_valid` (the same mesh as
  `py_eqs278k_flip_momentum_b09_300_valid`).

| check | c2f seeds | after relaxation |
|---|---|---|
| unmapped samples | 0 | 0 |
| coincident samples | 0 | 0 |
| flipped triangles (against the fine surface) | **246** (in 20 coarse faces) | **6** (in 6 coarse faces) |
| degenerate triangles | 0 | 7 |
| one-ring, folded (< 1 turn), of 267,190 sheet-interior vertices | **236** (down to −1 turn) | **3** |
| one-ring, wrapping (> 1 turn) | 0 | 0 |
| c2f walk: clamped queries / steps far outside | 0 / 0 | — |
| locally bijective | **no** | **no** (almost) |
| fold count against the coarse normal (for comparison) | 595 | 303 |

**Small test** (20,000 samples, same sampling, 300 iterations):
- **c2f seeds:** 36 flipped triangles (in 13 coarse faces) and 29 folded one-rings.
- **Relaxed:** 3 flipped triangles, 0 folded one-rings, 0 degenerate.

## What it shows

- **The c2f map is not bijective.** In about 20 of the 541 coarse faces, small patches of
  samples land turned over on the fine sheet: flipped triangles, and one-rings that turn backwards.
  - Nothing else is wrong: every sample is placed, no two coincide, and the walk was never
    clamped.
  - So the folds come from the map itself, not from lost or merged samples.
- **Relaxation almost makes it bijective.** 246 flipped triangles drop to 6, 236 folded one-rings
  to 3, and no one-ring wraps. 7 degenerate triangles appear, which matches the 7 in the quality
  numbers.
- **Most of the "folds" counted so far are not folds.**
  - The fold count in the earlier tables (595 at the seeds, 303 after the best relaxation)
    compares each triangle with the flat normal of its coarse face. It flags every triangle
    where the fine sheet bends more than 90° away from that plane.
  - Against the actual fine surface, only 246 triangles are flipped at the seeds and 6 after
    relaxation.
  - **For comparing runs, the flipped count here is the better fold measure.** The old count
    mostly measures how far the sheets bend under their coarse faces.
- **First version, which overcounted.** A first version took the reference normal from all three
  corners and reported 880 flipped triangles on the small test. 849 of those had a corner on a
  seam, where the fine face (and so the normal) may belong to another sheet. After the fix, the
  triangle check and the one-ring check agree.

## Second mesh: ABC 01_00040093 and a fix

Run: `Experiments/01_00040093_6f8f50d16c8a4ef7abda5aea_trimesh_000/mats_2595_interp_20/subdiv_n1000000/relaxation`.
- **Settings:** the best setting (equal-area split to 277,700 samples, Delaunay flips, momentum
  β 0.9, two-sided graph, no new folds, 300 iterations).
- **Inputs:**
  - bundle: `correspondence_*_mat_initial.c2f` in `subdiv_n1000000`;
  - `.ma_struct`: `Data/ABC_Dataset/<mesh>/<mesh>_mat_initial.ma_struct`.

### Fix: vote per patch

The first version decided each coarse face's orientation by a vote among that face's own
triangles. On this mesh that went wrong for 7 coarse faces:
- **What they are:** thin slivers on seams, of area 2–6 against a median of 995 (face 167 has
  edges 1.6 / 21 / 22.6), each with only 2–9 triangles.
- **What went wrong:** most of their triangles are folded at the c2f seeds, so the vote decided
  they were "wound backwards".
- **Why that's wrong:** face 167 is wound like its neighbours 448 and 162 and lies within 1–7° of
  their plane, and those neighbours agree with the fine sheet.
- **Effect at the seeds:** their folds were hidden.
- **Effect after relaxation:** relaxation turned their triangles the right way round, and the
  check then counted those as flips: 38 flipped triangles and 25 folded one-rings, all false.

The vote now covers a whole patch (17 patches on this mesh, 24 on 00040057), so a sliver face
takes its sheet's orientation. The numbers for 00040057 above don't change.

### Results (with the fix)

| check | c2f seeds | after relaxation |
|---|---|---|
| unmapped samples | 2 (coarse vertices 349, 350: on no coarse face) | 2 |
| coincident samples | 0 | 0 |
| flipped triangles (against the fine surface) | **71** (in 28 coarse faces, median area 21) | **11** (in 9 coarse faces) |
| … of them with all corners on seams / boundaries | 9 | 9 |
| one-ring, folded, of 273,613 sheet-interior vertices | 0 | 0 |
| one-ring, wrapping | 0 | 0 |
| degenerate triangles | 0 | 0 |
| c2f walk: clamped queries | 0 | — |
| fold count against the coarse normal | 0 | 0 |

**What it shows:**
- **c2f seeds:** the map folds only in small patches, mostly in thin coarse faces on seams. No
  samples coincide and no sheet interior is turned over (no folded one-ring).
- **After relaxation:** 11 flipped triangles are left. 9 of them have all three corners on seams
  or boundaries, so the triangle crosses between sheets. The relaxation fixes the rest.
- **The fold count against the coarse normal says 0 at both stages.** These thin seam faces share
  the orientation of their fold, so comparing with the coarse face's own normal can't see it.
  This is another reason to use the flipped count from this check.

### Why some log lines appear twice

These are repeated work, not a bug:
- **`coarse struct IDs` and `palette`:** `delaunay_flip.py` computes the struct IDs it needs to
  find the seams. `CoarseRelaxSession` then computes them again for the relaxation, with the same
  result.
- **`graph`:** two different graphs. The directed graph is always built first, because the
  two-sided graph copies its vertex roles. Then the two-sided graph this run uses is built. Its
  line shows `curve 0, junction 0` because a plain mesh graph knows no roles; they are copied
  over just after.
