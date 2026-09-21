# Seam UV Pinning Fix — Investigation + Plan

## 1. Investigation: what's currently pinned

Re-confirmed directly from `src/joint_lscm.cpp` (not from memory — grepped
and read the actual lines).

### `joint_lscm_double_cover` (Case 2 — every seam collapse goes through this)

Lines 256–264:
```cpp
int pin_left  = B_glued[0];   // seam endpoint adjacent to vj
int pin_right = B_glued[1];   // seam endpoint adjacent to vi

VectorXi b_UV(4);
VectorXd bc_UV(4);
b_UV  << pin_left,             nVjoint_dc + pin_left,
         pin_right,            nVjoint_dc + pin_right;
bc_UV << 0.0,                 -1.0,    // pin_left  : y=0, x=-1
         0.0,                  1.0;    // pin_right : y=0, x=+1
```
This is the **entire pin set** passed to `flatten()` (the LSCM quadratic
solve). Exactly 2 points are hard-pinned: `B_glued[0]` → `(-1,0)` and
`B_glued[1]` → `(+1,0)` — the two one-ring-arc endpoints adjacent to `vj` and
`vi`. `vi`, `vj`, every `B_reflected` middle-arc vertex, and the post-collapse
slot (`nV`, holding the merged point `vk`) are all **free variables** of the
quadratic minimization.

### `joint_lscm_case1_dc` (Case 1 — true single-sheet mesh boundary, never
fires for a seam collapse: see below)

Lines 520–528, identical pattern: only `B_glued[0]`/`B_glued[1]` pinned.

### Why seam collapses are always Case 2, never Case 1

`SSP_collapse_edge.cpp:887-892` — for `active_sheets.size() > 1` (a seam
collapse), `-1` (the boundary sentinel) is unconditionally injected into
**both** `Nsv_local` and `Ndv_local` for every sheet, forcing
`onBd.sum()==2` for every sheet of every seam collapse. `joint_lscm()`'s
switch (`src/joint_lscm.cpp:869`) routes `onBd.sum()==2` to `case 2:`
(`joint_lscm_double_cover`) unconditionally. So Case 1 genuinely never
triggers for a real seam collapse under the current code — only Case 2 does.

### Why this causes the cross-sheet mismatches found in [[seam_uv_consistency_findings]]

With only 2 points pinned per sheet, and each sheet having a different local
`B_arc`/3D geometry, the conformal energy minimization is free to place
`vi`, `vj`, and `vk` wherever is locally optimal for *that sheet* — which
differs sheet to sheet. Confirmed empirically: 368/545 seam-collapse
attempts in the test run had `vi`/`vj`/`vk` disagree across sheets by more
than `1e-9`, up to `0.28` (see [[seam_uv_consistency_findings]]).

## 2. What the fix does

Add `vi`, `vj`, and `vk` (the merged post-collapse point) as **additional
hard pins**, on top of the existing 2 `B_glued` pins — 5 pinned points total
per sheet instead of 2. Every sheet is then required, by construction, to
place `vi`/`vj`/`vk` at literal fixed UV coordinates that are **the same
across every sheet of a given seam collapse** — eliminating the mismatch
outright rather than just detecting it.

Chosen pin targets (fixed, same for every sheet, sit on the existing y=0
seam line so they're consistent with the DC's existing mirror-symmetry
convention — `check_dc_symmetry` already expects `vi`/`vj` to have `|y|≈0`):

```
vi_target = (-0.5, 0.0)
vj_target = ( 0.5, 0.0)
vk_target = vi_target        // vi is always the survivor (get_post_faces
                              // remaps vj → vi); vk = UV_post at the
                              // survivor's slot, so it's the same identity
                              // as vi, just after the collapse. Pinning it
                              // to vi_target keeps that identity intact in
                              // UV space, not just in 3D.
```

This trades away 3 of the LSCM's previously-free degrees of freedom for
guaranteed cross-sheet consistency — each sheet's map picks up more
distortion elsewhere to satisfy the extra constraints (LSCM's 2-pin gauge is
normally exactly enough to fix translation/rotation/scale and nothing more;
5 pins over-constrains it).

## 3. Implementation plan

### Constraint: do not modify `src/joint_lscm.cpp`; keep new files local to `10_collapse_viz/`

New logic goes in **`10_collapse_viz/joint_lscm_pinned.h`** /
**`10_collapse_viz/joint_lscm_pinned.cpp`** (not `src/`), reusing what
`joint_lscm.h` already exports (`flatten`, `check_valid_UV_lscm`,
`build_double_cover_faces`, `DCVizData`, `check_dc_symmetry`,
`dc_log_sheet_header`) via `#include "joint_lscm.h"`.

Note: `dc_log()` itself is `static` inside `joint_lscm.cpp` (not exported),
so the new file gets its own small log stream (mirroring the existing
`SEAM_LOG`/`dc_log` pattern) rather than sharing that one.

**Build-architecture wrinkle this placement creates, and how it's handled:**
`SSP_collapse_edge.cpp` lives in `src/` and is a *shared* source file — it
gets compiled separately into all three apps (`08_subdiv_remesh`,
`10_collapse_viz`, `11_correspond_viz`), each via its own
`file(GLOB SRCFILES/SSP_SRCS *.cpp ../src/*.cpp)`. If the call-site change
in `SSP_collapse_edge.cpp` unconditionally `#include`s
`joint_lscm_pinned.h`, the other two apps' builds break — that header
doesn't exist on their include path (only `10_collapse_viz`'s
`CMakeLists.txt` would know about it), and it would be wrong to make a
sibling app depend on a directory that lives in `10_collapse_viz/`.

Fix: guard the new `#include` and the call-site branch in
`SSP_collapse_edge.cpp` behind a new preprocessor macro,
`SSP_SEAM_UV_PINNING`, defined only by `10_collapse_viz/CMakeLists.txt`
(`target_compile_definitions`), the same pattern already used for
`C2F_VIZ_DIAGNOSTIC`/`SSP_LSCM_LOG`/`ML_QEM_LOG`. `10_collapse_viz`'s
`CMakeLists.txt` also gets `target_include_directories(... PRIVATE .)` so
the compiler can find `joint_lscm_pinned.h` when compiling
`SSP_collapse_edge.cpp` as part of *this* app's build. `08_subdiv_remesh`
and `11_correspond_viz` never define `SSP_SEAM_UV_PINNING`, so the guarded
block (include + call) doesn't exist in their translation unit at all —
their builds are unaffected, and this fix is opt-in / `10_collapse_viz`-only
for now, matching where it's being developed and tested.

### New functions

1. **`joint_lscm_double_cover_pinned(...)`** — same signature/behavior as
   `joint_lscm_double_cover`, fully self-contained (computes its own
   boundary loop via `igl::boundary_loop`, exactly like the original — it
   does *not* depend on `joint_lscm()`'s internals). Body is a copy of
   `joint_lscm_double_cover`'s B-arc extraction / DC face construction,
   with Step 5 (pinning) extended from 4 `b_UV` entries to 10:
   `B_glued[0]`, `B_glued[1]`, `vi`, `vj`, `nV` (the post-collapse slot),
   each with a U and V entry. This is the function that actually fixes the
   seam-collapse problem — it's fully self-contained, so it can be called
   directly without touching `joint_lscm()`.

2. **`joint_lscm_case1_dc_pinned(...)`** — **DEFERRED, not implemented in
   this pass.** Mirrors `joint_lscm_case1_dc` the same way Case 2 is
   mirrored, but unlike Case 2, the original `joint_lscm_case1_dc` takes
   `bdLoop_in` as an input, computed by `joint_lscm()`'s onBd≤1 branch
   (`joint_lscm.cpp:726-749`, ~25 lines) — not self-contained the way
   `joint_lscm_double_cover` is. Since Case 1 never fires for a real seam
   collapse under current code (§1: seam collapses are always Case 2), and
   the seam UV-consistency problem this fix addresses only exists for
   multi-sheet (seam) collapses, this function has no effect on the problem
   being solved and was deliberately deferred. If Case 1 handling is wanted
   later (API symmetry / future-proofing against a change to the
   onBd-injection logic), it needs the ~25-line `bdLoop` snippet duplicated
   into the new file first — flag this file for that work when it's picked
   back up.

### Call-site change (the only place allowed to "forward")

`SSP_collapse_edge.cpp:938`, where `joint_lscm(...)` is currently called
once per active sheet: branch on `is_seam_collapse` (already computed at
this point). For a seam collapse, call `joint_lscm_double_cover_pinned(...)`
directly instead of `joint_lscm(...)` — bypassing the generic case-dispatch
in `joint_lscm.cpp` entirely for this path, since seam collapses are always
Case 2 anyway. Non-seam collapses (`active_sheets==1`: Case 0 or a genuine
single-sheet mesh-boundary Case 1) keep calling the original `joint_lscm()`
completely unchanged.

### Validity / distortion checking

Re-implement (in the new file) the same post-solve checks the existing
`case 2:` switch block does today (`joint_lscm.cpp:1021-1320`): `dc_ok =
check_valid_UV_lscm(...)`, the orientation-flip auto-correction, QCE
(`quasi_conformal_error`) logging, and the `check_dc_symmetry` call —
these are all reusable free functions declared in `joint_lscm.h`, so this
is call-and-log wiring, not new math.

### Verification

Re-run the existing headless build (`build/headless_verify`) with the seam
UV-consistency logger ([[seam_uv_consistency_findings]]) on the same mesh —
expect **zero** `[SEAM-UV-MISMATCH]` lines afterward, since `vi`/`vj`/`vk`
are now hard-pinned identically across every sheet. Also compare
`[DC-FAIL]`/QCE distortion rates before vs. after to quantify the
distortion-vs-consistency tradeoff from §2, and confirm the final sanity
check (`[SANITY] OK: ...`) still passes.

## Status

Investigation complete, plan written. Not yet implemented — awaiting
confirmation before writing `joint_lscm_pinned.h`/`.cpp` and the
`SSP_collapse_edge.cpp` call-site change.
