# Where the relaxed sample positions are written

Short answer: **in the `.sdt` itself**, and every other output of a relaxed run
uses them too. The PLY is only one of several exports.

## How they flow

Relaxation overwrites the subdivided mesh in place, at the end of
`subdiv_relax_solve_project` (`subdiv_relax_solve_project.cpp`, "write back"):

```cpp
M.V        = X;      // relaxed 3D positions
M.fineFace = face;   // fine face each relaxed sample lies on
M.fineBary = bary;   // barycentric on that face
```

The tracker is initialised after this (`subdiv_tracker_init`), so it seeds and
tracks the **relaxed** samples, not the midpoint ones. The positions before
relaxation are kept separately in `gVseed`.

## In `subdiv_<stem>.sdt`

Written by `subdiv_tracker_save` (`subdiv_tracker.cpp`):

| array | contains |
|---|---|
| `sub_V` | relaxed 3D positions (from `gM.V`) |
| `fine_face`, `fine_bary` | relaxed positions on the fine MAT (`oriented_<stem>.obj`) |
| `coarse_face`, `coarse_bary` | where the relaxed samples ended up on the coarse mesh |
| `sub_V_seed` (array 14, relaxed runs only) | positions **before** relaxation |

A relaxed file is version 2 with 15 arrays and `relaxed = 1` in the header.
With `--no_subdiv_relax` it is version 1 with 14 arrays (no `sub_V_seed`), and
`sub_V` holds the plain midpoint positions.

## Other files of a relaxed run

| file | positions |
|---|---|
| `subdiv_fine_relaxed_<method>_<stem>.obj` | relaxed subdivided mesh |
| `subdiv_deformed_relaxed_<method>_<stem>.obj` | relaxed samples at their tracked coarse positions |
| `subdiv_fine_relaxed_solve_project_with_anchors_<stem>.ply` | relaxed mesh plus the seam anchor spheres |
| `subdiv_fine_seed_<stem>.obj` | before relaxation |
| `subdiv_graph_<stem>.slg` | before relaxation (the graph is built on the seed) |

All of these share one vertex order and one face list (`sub_F`), so they can be
compared vertex by vertex.

## Verified

On `output/relaxation_experiments/l3_defaults`:

- positions rebuilt from `(fine_face, fine_bary)` on `oriented_*.obj` equal
  `sub_V` exactly (max difference 0);
- `subdiv_fine_relaxed_*.obj` matches `sub_V` to OBJ text precision (5e-13);
- `subdiv_fine_seed_*.obj` matches `sub_V_seed` (2e-13).

See `subdiv_sdt_reading.md` for the full file layout and a Python reader.
