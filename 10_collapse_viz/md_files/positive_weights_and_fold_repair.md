# Positive Laplacian weights and fixing fold-overs: theory

Background notes for the coarse-subdivision relaxation
(`coarse_subdiv_relax_experiments.md`). Nothing here is implemented yet.

Context: the best method so far is the explicit relaxation with uniform weights
and the no-new-folds rule (2,383 folded triangles vs 5,168 in the seed). With
cotangent weights from the coarse positions, 24% of the weights were negative
(obtuse coarse triangles; midpoint subdivision keeps the angles) and were
clamped to 0; the result was no better than uniform. That run was global (whole
symmetric mesh graph, no per-coarse-face mode); experiments stay global from
now on.

"Folded" below = a triangle whose orientation disagrees with the majority of its
coarse face (section 1 of the experiments doc).

## 1. Why negative cotangent weights matter

The cotangent weight of edge ij is

    w_ij = 1/2 (cot alpha + cot beta),

where alpha and beta are the two angles opposite the edge. It is negative when
alpha + beta > 180 deg, i.e. the two triangles on the edge are obtuse "against"
it. That breaks two things:

1. **The update stops being an average.** With a negative weight,
   sum_j w_ij x_j / sum_j w_ij is no longer a convex combination of the
   neighbours, so a vertex can be pushed outside the region its neighbours span,
   and that step alone can fold.
2. **The no-fold guarantee is lost.** Tutte's theorem (extended by Floater):
   with **all weights positive** and a convex boundary, the resting state of the
   weighted Laplacian is a valid embedding, with no folds. Negative weights void
   that guarantee.

The energy itself, 1/2 sum w_ij |x_i - x_j|^2, is still well-behaved with some
negative weights (it is positive semi-definite on a valid triangle mesh). Only
the "move to the weighted average" update and the fold guarantee need
positivity.

## 2. Ways to get positive weights

| option | idea | keeps the coarse layout? | cost / catch |
|---|---|---|---|
| **1. Clamp** (implemented: `--coarse_subdiv_relax_weights cotan`) | w <- max(w, 0) or max(w, eps) | No: no longer the true cotangent operator | trivial; on our mesh 24% of the weights were clamped |
| **2. Intrinsic Delaunay** | flip edges *intrinsically* (same surface geometry) until every edge has alpha + beta <= 180 deg, then take cotangents on that triangulation; Delaunay guarantees all w >= 0 (Bobenko & Springborn 2007; Sharp, Soliman & Crane 2019) | Yes: same geometry, exact on flat regions | the Laplacian lives on the *intrinsic* edges, so a vertex's weighted neighbours can differ from its mesh neighbours; needs an intrinsic-triangulation data structure |
| **3. Mean-value weights** (Floater 2003) | w_ij = (tan(gamma/2) + tan(delta/2)) / \|x_i - x_j\|, with gamma, delta the angles at vertex i on either side of edge ij | Yes in the plane: reproduces the layout of flat regions | always positive for valid triangles; **not symmetric** (w_ij != w_ji), so no symmetric energy, but the averaging update works directly |
| **4. Improve the coarse triangles first** | Delaunay edge flips or remeshing of the coarse mesh before subdividing, fewer obtuse angles | Changes the coarse mesh itself | changes connectivity and the coarse -> fine correspondence; probably not allowed here |
| **5. Weights from the current positions** | recompute cotangents every iteration | Changes as the mesh moves | does not fix negativity; unstable once triangles fold |
| **6. Keep negative weights, minimize the energy** | gradient descent or a linear solve on 1/2 sum w_ij \|x_i - x_j\|^2 instead of averaging | Yes | no fold guarantee (section 1, point 2), so it needs the no-new-folds rule or a barrier |

On flat regions, **mean-value weights** (3) and **intrinsic Delaunay** (2) both
keep the property cotangents were chosen for: the resting state reproduces the
coarse layout, so large coarse faces are not squeezed toward equal vertex
counts, while all weights stay positive. Mean-value is a small change (a third
value of `--coarse_subdiv_relax_weights`), so it is the easy first test.
Intrinsic Delaunay is the exact, symmetric version but needs more machinery.

## 3. Fixing fold-overs that already exist

Two different situations:

- **The mesh folds but the surface under it does not.** The vertices are placed
  badly on a valid sheet. Moving vertices can fix this.
- **The coarse -> fine mapping itself folds.** A region of the coarse face lands
  back-to-front on the fine sheet. Moving vertices only along the surface cannot
  fully fix this; the fix belongs in the correspondence.

Methods, from local and cheap to global and guaranteed:

1. **Kernel projection (local).** Each folded triangle has a vertex whose
   1-ring is tangled. Flatten the 1-ring into the local tangent plane (here the
   fine sheet's plane). The *kernel* of the ring polygon is the region from
   which every triangle around the vertex has positive area. Move the vertex into
   it (e.g. its Chebyshev center, via a small linear program), then project back
   onto the structure. Cheap; fixes each fold whose kernel is non-empty. Can fail
   where several neighbouring vertices are tangled together.
2. **Optimization-based untangling (local).** Per vertex, maximize the smallest
   signed area of its triangles (Freitag & Plassmann 2000; a small linear program
   per vertex), and sweep over the vertices until no fold remains. The robust
   version of 1.
3. **Regularized untangling energies (global).** Minimize an energy that stays
   finite for inverted triangles and pushes them back to positive orientation.
   For example, replace the signed area A with
   chi(A) = 1/2 (A + sqrt(A^2 + eps^2)), a smooth positive stand-in, and shrink
   eps step by step (Escobar et al.; Garanzha et al. 2021, "Foldover-free maps in
   50 lines of code"). Related: TLC, "total lifted content" (Du et al. 2020), and
   progressive embedding (Shen et al. 2019), which carries a guarantee. These
   untangle the whole mesh at once, and some also improve triangle quality.
4. **Re-embed folded regions with Tutte.** Cut a disk-shaped patch around each
   folded cluster, map its boundary to a convex polygon in a 2D chart (the coarse
   face's plane, or the fine sheet's local parametrization), recompute the
   interior with **positive** weights (Tutte/Floater then guarantees no folds in
   the chart), and map the result back onto the fine sheet through the chart.
   Each coarse face is a triangle, already convex, so it is a natural patch. Only
   works if the chart -> fine map itself does not fold (the second situation
   above).
5. **Flip-preventing optimization.** The no-new-folds rule stops new folds but
   never actively removes old ones. Line searches with a barrier such as
   -log(area) (Smith & Schaefer 2015) need a fold-free start, so they pair with
   1-4: untangle first, then relax with a barrier so no folds come back.
6. **Edge flips.** Changing the connectivity removes some folds trivially, but
   our connectivity is fixed by the subdivision, so this is probably off the
   table.

## 4. Suggested combination for this pipeline

1. Positive weights (mean-value, or intrinsic Delaunay) + the no-new-folds rule
   during the relaxation.
2. A cleanup pass on the folds that remain, with kernel projection or
   per-vertex untangling (3.1 or 3.2): cheap and local.
3. Before that, measure how many of the remaining ~2,383 folds come from the
   coarse -> fine mapping itself (the second situation). Only a fix to the
   correspondence helps those.

## References

- W. T. Tutte, "How to draw a graph", Proc. London Math. Soc., 1963.
- M. S. Floater, "Parametrization and smooth approximation of surface
  triangulations", CAGD, 1997; "Mean value coordinates", CAGD, 2003.
- A. I. Bobenko, B. A. Springborn, "A discrete Laplace-Beltrami operator for
  simplicial surfaces", DCG, 2007.
- N. Sharp, Y. Soliman, K. Crane, "Navigating intrinsic triangulations", ACM TOG
  (SIGGRAPH), 2019.
- L. A. Freitag, P. Plassmann, "Local optimization-based simplicial mesh
  untangling and improvement", IJNME, 2000.
- J. M. Escobar et al., "Simultaneous untangling and smoothing of tetrahedral
  meshes", CMAME, 2003.
- V. Garanzha et al., "Foldover-free maps in 50 lines of code", ACM TOG
  (SIGGRAPH), 2021.
- X. Du et al., "Lifting simplices to find injectivity", ACM TOG (SIGGRAPH), 2020.
- H. Shen et al., "Progressive embedding", ACM TOG (SIGGRAPH), 2019.
- J. Smith, S. Schaefer, "Bijective parameterization with free boundaries", ACM
  TOG (SIGGRAPH), 2015.
