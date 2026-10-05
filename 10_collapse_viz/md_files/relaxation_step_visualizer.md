# Relaxation step visualizer (polyscope, Python)

A viewer that runs the explicit relaxation of the subdivided coarse mesh one iteration at a
time and shows what each iteration computed. Code: `relaxation_scripts_python/relax_viewer.py`.

## Rule: the data comes from the calculation

The viewer computes nothing about the relaxation. Every quantity it shows is recorded by the
relaxation code (`relax_explicit.py`, verified byte-identical to the C++) while it runs:

- The relaxation becomes a stepper (`ExplicitRelaxer`: set up once, `step()` per iteration).
  `subdiv_relax_explicit` is that stepper run in a loop, so the batch runs and the viewer
  execute the same code.
- Checkpoints in the step record, per vertex, for the current iteration:
  - x, the position at the start of the step
  - y = x + lambda (mean - x), the step target ("solve" result)
  - Pi(y), the projection result, with its fine face, edge and barycentrics
  - the committed position, which differs from Pi(y) when no-new-folds held the vertex back,
    plus a held-back flag
  - what the projector searched:
    - local projection: the triangles / edges of the search region and the number of
      rounds, per vertex
    - global projection: the trees queried and the BVH nodes visited
- Checkpoints are off in batch runs. With them off, the outputs must stay byte-identical
  (`verify_relax.py` is re-run after the change).

## Switching configurations between steps

At any step the run can be paused, the configuration changed, and the run continued from the
current positions. Changeable settings:

- lambda, max iterations, tolerance
- local / global projection
- no-new-folds on / off
- weights: uniform / cotan / meanvalue
- graph: symmetric / directed
- per coarse face (hold vertices on coarse vertices / edges)

On a change, the derived data is rebuilt by the original functions (`build_relax_graph`,
`cotan_weights` / `meanvalue_weights` at the coarse positions, `holdFixed`, the free-vertex
rules) and the state (positions, fine faces, edges, barycentrics) carries over. Each change is
logged with its iteration number.

Only the explicit method is ported. Switching to Newton / solve_project needs those methods
ported first (not part of this step; to be decided).

## Viewer

1. **Points.** All subdivided coarse vertices in one point cloud. Each point has a random
   colour, fixed for the whole session (seeded by vertex index), so a point can be followed
   across steps.
2. **Selection** (click a point). The selected point stands out: all other points become
   more transparent. For the current step it shows:
   - markers at y (step) and Pi(y) (projection), and the committed position if held back;
   - arrows (ambient vectors) x -> y and y -> Pi(y);
   - the BVH used for the point: its structure's tree(s), with the visited nodes for global
     projection or the searched triangles / edges for local projection;
   - checkboxes for each item, and a choice of where the selected point is drawn: x, y or
     Pi(y);
   - a text panel with the numbers (vertex id, role, structure ids, fine face / edge,
     barycentrics, move lengths, held back).

   **Clear selection** removes all of it and restores the normal look.
3. **Meshes per step**, each with a checkbox:
   - committed (current) mesh
   - step mesh (all vertices at y)
   - projection mesh (all vertices at Pi(y))
   - held-back vertices flagged
4. **Groups.** Polyscope groups keep it organized: points, selection, step meshes, BVH,
   fine MAT.
5. **Run control:** Run, Stop (Run resumes), Step (one iteration), and run until a given
   iteration.
   **Reset** starts again from the relaxation input (the seeds, iteration 0) with the
   configuration currently set in the panel.
   - The seeds come from the session, which is built once. A new relaxer is made, as for
     the first run.
   - Runs are numbered. Earlier runs and their configuration histories are listed in the
     Configuration panel.
   - Exports after a reset are named `runNN_...`.
   - Changes that alter the mesh itself (sample count, equal-area level) still need a
     restart.
6. **Colour mode:** random colours or concave mask.
   - **Concave mask (choice a):** a fine edge is concave when its two faces meet at a
     concave dihedral angle, measured with consistently oriented normals (the coarse face
     orientation). The points within k rings (subdivided mesh) of a concave fine edge are
     marked. k and the angle threshold are UI settings.
   - The mask is computed from the points' current fine faces (from the calculation).
7. **BVH inspector.** One collapsible section per structure (each sheet and each curve), with
   a checkbox per tree. Show all / Hide all buttons. Each tree draws its node boxes (wire
   boxes) with a depth slider and a leaves-only option, so you can check that every box
   encloses its children and its triangles / edges.

8. **Export (PLY).** One button per mesh, and one that writes all four:
   - relaxation input (iteration 0)
   - committed (current)
   - step y (this iteration)
   - projection Pi(y) (this iteration)

   A last button writes the stages before the relaxation, as `run_relax.py` does: coarse,
   equal-area refined, subdivided, at c2f positions, input. The data comes from the
   relaxation's state or this iteration's checkpoint, and step y / Pi(y) need a recorded
   checkpoint. Vertex properties: role, set id, free, fine face, held back, the point's
   colour. Face property: coarse face. Files go to `--export_dir`, by default
   `output/relaxation_experiments/viewer_exports/<date_time>/`, whose `experiment_config.txt`
   lists every export with its iteration and configuration.

9. **Camera speed.** Polyscope moves the camera (scroll zoom, pan, first-person keys) in steps
   proportional to the scene length scale. Every frame the viewer sets that scale to
   clamp(k x d, min, max) x the starting scale, smoothed. A drop-down picks what d is:
   - the distance from the camera to the orbit centre (C);
   - the distance to the 3D point under the cursor, from polyscope's pick buffer (A), read
     every n frames; it keeps the last value off the surface or over the GUI.

   The viewer also does two things so this works:
   - It sets all radii (points, curves, vectors) as absolute sizes from the starting length
     scale, so they stay the same size when the scale changes.
   - It raises the far clip ratio as the scale drops, so far geometry is not cut off.

   On by default, in mode C.

## Input

A C++ run's bundle and the fine `.ma_struct`, as `run_relax.py` takes them (default:
ABC 00040057, `output/relaxation_experiments/clamp_check`). The initial configuration is set
with the same flags as `run_relax.py`.

## Status (2026-10-03)

Implemented: `relaxation_scripts_python/relax_viewer.py`. Headless test:
`relaxation_scripts_python/test_viewer.py` (polyscope mock backend).

- The relaxation is a stepper (`ExplicitRelaxer`) with checkpoints (`StepTrace`). With the
  checkpoints off, `verify_relax.py` still gives byte-identical OBJs and matching logs against
  the C++ (clamp_check 1000 iterations, cpp_ref_global_nofold, cpp_ref_perface_dir_cotan).
  With them on, the positions are bit-identical to a run without them, including after a
  configuration switch in the middle of the run (`test_viewer.py`).
- Run: `python relax_viewer.py [run_relax.py flags] [--total_max_iter N]`. Click a point (or a
  vertex of a step mesh) to select it.
- BVH nodes visited are recorded for the selected (watched) point only, from the step after it
  was selected. The search region, winning tree and leaf, and all positions are recorded for
  every point at every step.
- Concave mask (revised 2026-10-05, `concave_parts.py`): the concave parts of the MAT's edge
  structures from the .ma_struct, i.e. its seams (type 1, 3 faces per edge) and boundaries
  (type 2, 1 face per edge).
  - A corner (v, s) at a vertex v on the border of sheet s is concave when the interior
    angle of s at v (the sum of s's face angles at v) is more than 180 degrees + a margin
    (default 20).
  - The marked points are the subdivided vertices of the relaxation input (iteration 0)
    whose fine face is in sheet s and that lie within a radius (default 3 mean subdivided
    edge lengths) of a concave corner, optionally grown by k rings (default 0).
  - The set is fixed for the session; it does not follow the moving points.
  - The curve network "concave seam / boundary parts" holds only the vertices of its edges.
    A point cloud "concave corners" shows the corners.
  - On ABC 00040057 (clamp_check bundle): 224 concave corners out of 2209 sheet border
    corners, 262 seam / boundary edges, and 33,687 of 207,000 points marked.
  - The first version (two-face dihedral edges, seams and boundaries skipped, marked from the
    current positions) is in `relaxation_scripts_python/legacy/relax_viewer_concave_dihedral.py`.
