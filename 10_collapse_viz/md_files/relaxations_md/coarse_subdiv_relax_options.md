# Coarse-subdivision relaxation: options to stop the folding

Context: `md_files/relaxations_md/coarse_subdiv_relax_results.md`. Relaxing the subdivided
coarse mesh on the fine MAT converges exactly and passes every check, but its
resting state folds (Newton: ~10.6k degenerate and ~15.6k flipped of 409,600
triangles on ABC 00040057). Cause: the relaxation weights every neighbour
equally, so it evens out vertex counts, while every coarse face owns the same
1024 triangles and coarse face areas differ by ~2000x. Large coarse faces are
compressed and fold.

## 1. Relax each coarse face on its own (recommended first)

Fix the vertices on coarse vertices and coarse edges, and relax only the
interior of each coarse triangle.

- **Why it helps:** vertex counts can no longer shift between coarse faces, so
  the squeeze of large faces goes away.
- **Cost:** small. Uses the existing code with extra fixed vertices: the
  subdivided vertices whose coarse carrier is a VERTEX or an EDGE
  (`SubdivMesh::carrierType` of the coarse subdivision) are held at their
  seeds; the rest relax as sheet vertices.
- **Downside:** spacing is only evened out inside each coarse face, not across
  faces.

## 2. Weighted relaxation

Weight neighbours by geometry (for example, cotangent weights from the seed)
instead of equally.

- **Why it helps:** it targets the real cause; the resting state no longer
  equalises vertex counts regardless of area.
- **Cost:** changes the shared relaxation code (`build_relax_graph` /
  `subdiv_relax` / `subdiv_relax_solve_project`), which the fine-subdivision
  relaxation also uses.
- **Downside:** the fine-subdivision relaxation changes too unless the weights
  are made optional (uniform stays the default there).

## 3. Flip-preventing line search

Reject any step that flips a triangle.

- **Why it helps:** no folds by construction.
- **Cost:** a flip check per step in the solver's line search.
- **Downside:** where the resting state itself is folded, the relaxation may
  just stop early, far from even spacing.
