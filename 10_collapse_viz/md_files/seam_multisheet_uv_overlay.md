# Seam Multi-Sheet UV Overlay + Per-Sheet V_pre/V_post Point Clouds

## Motivation

For a seam collapse, `active_sheets.size() > 1` and each active sheet is handled
by an **independent** call to `joint_lscm` (see `f2c_vprepost_and_seam_lscm.md`
and the seam/sheet walkthrough discussed in this session). Because `-1` is
injected into both `Nsv_local` and `Ndv_local` for every sheet of a seam
collapse (`SSP_collapse_edge.cpp:887-892`), every sheet is routed into
`whichCase == 2` (0-indexed) / Case 3 (1-indexed) — the double-cover solve in
`joint_lscm_double_cover` (`src/joint_lscm.cpp`).

Each sheet's double-cover solve pins its **own** `B_glued[0]` → `(-1,0)` and
`B_glued[1]` → `(+1,0)`. Nothing in the current pipeline forces sheet A's UV
frame to agree with sheet B's UV frame. In particular `vi` and `vj` — the two
collapsing vertices, shared by every active sheet since they sit on the seam —
get their own independently-solved UV coordinates *per sheet*. The working
hypothesis is that these do **not** come out consistent between sheets: sheet A
might place `vi` at UV `(0.3, 0.0)` while sheet B places the same global vertex
`vi` at `(-0.7, 0.0)`.

## Goal

Visually confirm (or refute) this by taking the **exact, unmodified** per-sheet
`joint_lscm` UV solutions for every active sheet of the currently-inspected
seam collapse, and drawing them **together in one shared frame** — the UV
analogue of how the sheets are joined in 3D along the seam edge — without any
rigid re-alignment. If sheets disagree, this will show up directly as their
`vi`/`vj` markers landing at different UV coordinates instead of coinciding.

Additionally, expose the exact 3D `V_pre`/`V_post` ring geometry for **every**
active sheet (not just the currently-selected one) as point clouds in the
canonical view, since `SheetData` already stores this per sheet
(`src/single_collapse_data.h:30-32`, `SSP_collapse_edge.cpp` `store_sheet_data:`
block) but it was previously only ever visualized for whichever sheet was
selected via `apply_sheet_to_snap`.

## Data already available

`gAllSheets` (`10_collapse_viz/visualizer.cpp:130`) holds every active sheet's
`SheetData` for the most recently displayed collapse (`refresh_snap()`,
`visualizer.cpp:212-229`). Each `SheetData` (`src/single_collapse_data.h:22-45`)
already carries, per sheet, in that sheet's own local index space:

- `UV_pre`, `UV_post` — the raw joint_lscm UV solve output
- `FUV_pre`, `FUV_post` — local face connectivity
- `V_pre`, `V_post` — exact 3D ring geometry
- `b` — local indices of `(vi, vj)` within this sheet
- `subsetVIdx` — local → global vertex index map

No new solver-side data was needed; this is purely a visualization feature.

## Implementation

All changes are in `10_collapse_viz/visualizer.cpp`, inside `show_canonical_view()`.

### 1. Seam UV overlay (isolate view)

New toggle `gShowSeamUVOverlay` (only offered in the UI when
`gAllSheets.size() > 1`, i.e. an actual seam collapse). When enabled:

- Every other structure in canonical view (one-ring meshes, regular/DC UV
  panels, DC vertex groups, boundary-vertex highlight, non-active sheet
  meshes) is disabled via a new `isolateSeamOverlay` flag threaded through
  their existing `setEnabled(...)` calls.
- For each sheet in `gAllSheets`, its **raw** `UV_pre` is lifted into 3D as
  `(u, v, 0)` — no rotation, no per-sheet rigid alignment — and registered as
  its own surface mesh (`seam_ov_sheet_<i>_sid<sid>_pre`), colored by an
  evenly-spaced hue across all sheets (shared `hsv_to_rgb` helper, hoisted out
  of the non-active-sheet-coloring lambda so both blocks use one definition).
- A same-colored 2-point point cloud (`seam_ov_sheet_<i>_sid<sid>_vivj`) marks
  that sheet's local `vi`/`vj` UV positions, so the same global vertex pair
  can be visually compared across sheets — misaligned sheets show up as
  same-colored dots that do NOT coincide.

Known limitation: `sample_tracker_show_canonical(...)` (query/coarse-to-fine
sample overlay) is not gated by `isolateSeamOverlay` since it doesn't take a
visibility flag; it may still draw on top of the isolate view. Not addressed
in this pass — flag if it becomes a problem in practice.

### 2. Per-sheet V_pre/V_post point clouds

New toggle `gShowSheetVPts` (default on), independent of the seam-overlay
isolate toggle — available any time in canonical view. For every sheet in
`gAllSheets`:

- `V_pre` is transformed with the same rotation/centering (`rot(...)`) used
  for the one-ring meshes, so it lines up with `one_ring_pre`, and registered
  as `sheet_<i>_sid<sid>_Vpre_pts`.
- `V_post` likewise → `sheet_<i>_sid<sid>_Vpost_pts`.
- Colored the same hue as that sheet's overlay/non-active-sheet color
  (`V_post` uses the color-inverted variant so pre/post are visually
  distinguishable per sheet).

## UI

Both toggles live in the canonical-view "Visibility" panel
(`visualizer.cpp`, near the existing per-sheet selector / seam log section):

- **"Seam UV overlay (isolate)"** checkbox — only shown when there is more
  than one active sheet.
- **"Sheet V_pre/V_post points (all sheets)"** checkbox — always shown in
  canonical view.

Both are wired into the existing `vis` redraw-trigger pattern already used by
every other canonical-view checkbox.

## Revision: standalone UV View + UV_post

The original implementation put the overlay inside `show_canonical_view()`
behind an "isolate" toggle, alongside the `V_pre`/`V_post` point clouds. That
mixed two incompatible coordinate spaces in one view: `V_pre`/`V_post` are in
true mesh/3D-ring scale (rotated + centered to match the one-ring), while the
raw `UV_pre` overlay is unrotated unit-ish UV scale — so the `V_pre`/`V_post`
points appeared "huge" next to the overlay meshes.

Fix: the overlay was pulled out into its own standalone view,
`seam_uv_view.h`/`seam_uv_view.cpp`, with its own `show_seam_uv_view(sheets,
show_pre, show_post)` entry point:

- A **"UV View"** button in the canonical-view panel (shown only when the
  collapse has >1 active sheet) switches to it via
  `polyscope::removeAllStructures()` + `show_seam_uv_view(...)`.
- A **"Back to Canonical View"** button switches back the same way.
- `update_display()` short-circuits to `show_seam_uv_view(...)` whenever
  `gUVView` is set, so step/slider-driven redraws never rebuild canonical/
  normal-view geometry underneath it.
- `V_pre`/`V_post` point clouds (`gShowSheetVPts`) stay exclusively in the 3D
  canonical view, where their true-mesh-scale, rotated coordinates make sense.

`show_seam_uv_view` also now draws **both** `UV_pre`/`FUV_pre` (bright,
saturated) and `UV_post`/`FUV_post` (muted/desaturated) per sheet — both are
the direct, unmodified `joint_lscm` output (traced end-to-end: `joint_lscm`'s
`case 2:` branch → `SheetData::UV_pre`/`UV_post` at
`SSP_collapse_edge.cpp:1347-1348` → `gAllSheets` copy in `refresh_snap()` →
`lift_uv()` in `seam_uv_view.cpp`, which only appends a constant `z=0`
column). Each is independently toggleable ("Pre (bright)" / "Post (muted)"
checkboxes shown while in UV View), sharing the sheet's hue so pre/post pairs
for the same sheet are visually associated.

## Revision: dedicated UV View window

Initially the "Back to Canonical View" button and Pre/Post checkboxes were
appended into the same `if (gSnap.valid) { ... }` block inside the main "SSP
Collapse Visualizer" ImGui window that also drives the canonical/normal view
toggle — so entering UV View still left the whole canonical-view-oriented
panel on screen (step controls, DC stop checkboxes, run-to-# input, sliders,
etc.), none of which apply to a frozen per-sheet UV inspection.

Fix: `ui_callback()` now early-returns into its own dedicated ImGui window,
`"Seam UV View"`, whenever `gUVView` is set — before any of the main panel's
stepping/running logic runs. That window contains only what's relevant to
this view: collapse/sheet count, `vi`/`vj`, the Pre/Post toggles, and "Back to
Canonical View". Stepping/running is implicitly paused while the UV View is
open (the early return skips the continuous-decimation block entirely) since
this view is a frozen snapshot of one collapse's sheets, not something that
should change under you while inspecting it. The old in-main-panel toggle
block was simplified back to a plain two-way Canonical/Main-view switch.

## Revision: Pre/Post groups + orphan-vertex compaction

Each sheet's mesh + vi/vj point cloud is now added to one of two Polyscope
groups, `"UV View: Pre"` / `"UV View: Post"` (`fresh_group()` in
`seam_uv_view.cpp`, rebuilt every redraw via `removeGroup(name, false)` +
`createGroup(name)` to avoid the stale-group-exception issue seen elsewhere
in this file for `"B_arc_chain"`). This keeps the structure list from
interleaving pre/post meshes per sheet.

Also fixed: `UV_post` is stored with the same row count as the one-ring's
local vertex space, but `vi`'s row is not referenced by any face in
`FUV_post` — the collapse re-triangulates vi's faces onto vj, so vi's row
just holds its solved post-collapse UV with no face touching it. Registering
`UV_post`/`FUV_post` as-is left that row as a disconnected, unreferenced mesh
vertex, which visually looked like a stray point unrelated to the post mesh
geometry (and could even look "mixed with pre" since nothing anchored it to
the mesh at all). Fixed by compacting the mesh vertex set with
`igl::remove_unreferenced` before registering the surface mesh (in
`show_one_sheet()`), so the rendered mesh only ever contains vertices that
are actually on a face. The vi/vj marker point cloud is a separate structure
and doesn't have that constraint, so it still plots from the *uncompacted*
UV row — vi's marker still shows its exact solved post-collapse position
even though the mesh itself no longer carries that vertex.

## Revision: post marker is a single point, not two

`get_post_faces` (`src/get_post_faces.cpp:37-41`) always remaps every
occurrence of `vj` to `vi` when building post-collapse connectivity — `vi` is
always the survivor (moved to the collapse placement), `vj` always the
absorbed vertex. `UV_post.row(vi)` therefore holds the correctly-solved
merged post position, but `UV_post.row(vj)` is never touched for the post
case — it's just whatever `UV_pre.row(vj)` was, copied forward unresolved
(`joint_lscm_double_cover`: `UV_post = UV_pre; UV_post.row(vi) = ...;` — `vj`'s
row is never assigned). So the previous post marker, which plotted both
`uv3.row(lvi)` and `uv3.row(lvj)`, was showing one real point (vi, the merged
position) and one bogus point (vj, stale pre-collapse data with no post-side
meaning).

Fixed in `show_one_sheet()` (`seam_uv_view.cpp`): a new `is_post` parameter
makes the post marker plot a single point at `vi` only — the true merged
location — instead of two. The pre marker is unchanged (two real points, vi
and vj, since both are meaningfully distinct pre-collapse).

## Revision: derive everything from face membership, no compaction step

The `igl::remove_unreferenced` mesh compaction and the hardcoded `is_post`
marker special-case were both redundant with a fact already encoded in the
data: `get_post_faces` (`src/get_post_faces.cpp:37-41`) remaps every
occurrence of the absorbed vertex onto the survivor *before* `FUV_post` is
built, so `FUV_post` itself is already the ground truth for "is this vertex
part of this mesh" — there was no need to separately recompute that via
`remove_unreferenced`, or to hardcode "vi survives, vj is absorbed" as an
`is_post` flag.

Simplified `show_one_sheet()` (`seam_uv_view.cpp`) accordingly:
- The mesh is registered directly from `UV`/`FUV` as-is — no compaction. An
  unreferenced row just sits unused in the vertex buffer; it was never
  visually rendered as a stray point in the first place (Polyscope only
  draws vertices that are part of a face for a surface mesh) — the actual
  "stray point" the user saw was the *marker* point cloud, not the mesh.
- The vi/vj marker now uses one generic rule, `referenced_in(FUV, idx)`:
  plot a marker for `vi`/`vj` only if that index is actually referenced
  somewhere in `FUV`. For `FUV_pre` both are referenced (2 points). For
  `FUV_post`, only the survivor is referenced (1 point) — this falls out
  automatically from the face list, with zero collapse-semantics knowledge
  baked into `seam_uv_view.cpp` itself.

No UV value is touched by any of this — `show_one_sheet` still renders
`UV_pre`/`UV_post` and `FUV_pre`/`FUV_post` exactly as `joint_lscm` produced
them; the only change across all these revisions is which precomputed rows
get plotted as a marker, decided purely by face-list membership.

## Status

Implemented and compiles clean (`cmake --build build/debug --target
collapse_viz_bin --config Debug`). Visually verified that the debug build
launches and loads a real seam-heavy mesh (24 sheets, 474 seam edges) without
crashing. The UV View's own window, pre/post toggles, and the
face-membership-derived markers have not yet been interactively exercised
end-to-end in a running session — worth doing next.
