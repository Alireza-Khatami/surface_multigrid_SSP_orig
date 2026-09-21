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

## Implementation (as built)

- `10_collapse_viz/joint_lscm_pinned.h` / `.cpp` — `joint_lscm_double_cover_pinned`
  (5-pin variant of `joint_lscm_double_cover`) and `joint_lscm_seam_pinned`
  (top-level entry point, mirrors `joint_lscm()`'s signature). Case 1 deferred
  (not implemented — see note in `joint_lscm_pinned.h`; never fires for a
  real seam collapse under current code).
- `10_collapse_viz/CMakeLists.txt` — new `SSP_SEAM_UV_PINNING` option
  (default ON), gates `target_compile_definitions`, adds
  `joint_lscm_pinned.cpp` to sources, and adds
  `target_include_directories(... PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})` so
  `src/SSP_collapse_edge.cpp` can find the new header when compiled as part
  of this app. `08_subdiv_remesh`/`11_correspond_viz` never define the
  macro, so they're unaffected (confirmed by inspection of their
  `CMakeLists.txt` — both glob `src/*.cpp` independently and don't have
  `10_collapse_viz/` on their include path).
- `src/SSP_collapse_edge.cpp` — `#include "joint_lscm_pinned.h"` and the
  call-site branch (`is_seam_collapse ? joint_lscm_seam_pinned(...) :
  joint_lscm(...)`) both wrapped in `#ifdef SSP_SEAM_UV_PINNING`.
- `10_collapse_viz/main.cpp` — opens/closes
  `seam_uv_pinned_<stem>.txt` (this file's own `[SEAM-PIN-*]` log,
  separate from `dc_log()` which is `static`/private to `joint_lscm.cpp`).

Pin-value convention was re-derived directly from the original code's own
inline comments (`"pin_left : y=0, x=-1"` next to `bc_UV << 0.0, -1.0`), not
assumed: for a pinned row index `idx`, `(b_UV, bc_UV) = {(idx, y_target),
(nVjoint_dc+idx, x_target)}`. Verified `flatten()`/`mqwf_dense_precompute`/
`mqwf_dense_solve` (`src/mqwf_dense.cpp`) are fully generic over the pin
count — no hidden assumption of exactly 2 pins, so extending 4 `b_UV`
entries to 10 was mechanically safe. Verified `vi`/`vj`/the post-collapse
slot (`nV`) are always index-disjoint from `B_glued`/`B_reflected` by
construction (`B_arc` is built from the boundary loop strictly *excluding*
`vi`/`vj`), so none of the 5 pins can ever collide.

## First verification run — negative result, investigated further

Ran the headless build (`build/headless_verify`) on the same test mesh used
for [[seam_uv_consistency_findings]]. Result: **1290/1290 `SEAM-PIN-FAIL`,
0 `SEAM-PIN-PASS`** — every single seam-collapse sheet failed
`check_valid_UV_lscm`. The `seam_uv_consistency` log showed 0 mismatches,
but only because no seam collapse succeeded at all, not because the fix
worked.

Checked `nFpre` (faces per sheet) across all failures: ranged 3–11, i.e.
every seam one-ring patch in this mesh is tiny. Hypothesis: forcing `vi`/`vj`
to fixed absolute UV coordinates (`±0.5`), on top of `B_glued` at `±1`,
leaves a 3–11-face patch essentially no geometric slack to satisfy 5 rigid
constraints without a flipped/folded triangle.

### Diagnostic: isolating which check(s) are actually rejecting

To confirm the hypothesis without guessing, `check_valid_UV_lscm`'s checks
were split into 3 independently reusable functions in `joint_lscm_pinned.cpp`
(mirroring `check_valid_UV_lscm`'s logic exactly; `check_valid_UV_lscm`
itself untouched):

- `check_uv_face_flip(UV, FUV)`
- `check_uv_foldover(UV, FUV, vi, vj)`
- `check_uv_triangle_quality(UV, FUV, threshold=0.01)`

A diagnostic gate, `check_valid_UV_lscm_diag_no_flip_foldover`, was composed
from only the NaN check + `check_uv_triangle_quality` (both pre/post) —
`check_uv_face_flip`/`check_uv_foldover` deliberately excluded — and swapped
in for `joint_lscm_seam_pinned`'s validity call.

**Result: 1647/1713 `SEAM-PIN-PASS`, 66 `SEAM-PIN-FAIL`, still 0
`seam_uv_consistency` mismatches.** Confirms the flip/fold-over checks were
the cause of the earlier 100% failure rate — the pinning mechanism itself
does force cross-sheet agreement when it's allowed to run.

**But this doesn't make the fixed-pin approach viable.** QCE
(`quasi_conformal_error`) on the "passing" cases: max up to `6.3`, means
around `2–4` — well above the existing `[DC-HIGH-DISTORTION]` threshold of
`3.0` used elsewhere in this codebase. The flip/fold-over checks were
correctly rejecting genuinely invalid (inverted/self-overlapping)
parameterizations, not being overly strict; disabling them just stops
catching that, it doesn't fix the geometry.

**`check_valid_UV_lscm_diag_no_flip_foldover` is left in the code as a
diagnostic result, not switched on as the actual validity gate.**
`joint_lscm_seam_pinned` should be reverted to the full 3-check composition
(or `check_valid_UV_lscm` directly) once the pin-target strategy itself is
fixed — see "Next steps" below.

## Fix found: vi/vj pin ordering was backwards

Before jumping to a more complex (reference-sheet-derived) strategy, the
original 2-pin code's own comment (`src/joint_lscm.cpp:245-247`) was
re-checked directly:

```
// B_glued[0] is adjacent to vj and B_glued[1] is adjacent to vi in the one-ring.
// Pinning them on the seam line (y=0) spread in x forces the correct seam ordering:
//   B_glued[0](-1) < vj < vi < B_glued[1](+1)  at y≈0
```

The documented natural cyclic order is `B_glued[0](-1) < vj < vi <
B_glued[1](+1)` — `vj` (adjacent to `B_glued[0]`) belongs near `-1`, `vi`
(adjacent to `B_glued[1]`, closing the loop) belongs near `+1`. The original
implementation of this fix had them **backwards**:
`vi_target=(-0.5,0)`/`vj_target=(0.5,0)` — i.e. `B_glued[0](-1) < vi(-0.5) <
vj(0.5) < B_glued[1](+1)`, contradicting the documented order. Forcing `vi`
and `vj` into the wrong relative position on the seam line, on top of tiny
3-11-face patches with no slack, is exactly the kind of contradiction that
produces flipped/folded triangles.

**Fix**: swapped to `vi_target=(0.5,0)`, `vj_target=(-0.5,0)` (and
`vk_target = vi_target` still, per the earlier reasoning), matching the
documented order. Re-ran headless verification with the validity gate
reverted to the **full** check (`check_valid_UV_lscm_full`, composed from
`check_uv_face_flip` + `check_uv_foldover` + `check_uv_triangle_quality` —
not the relaxed diagnostic gate):

| Metric | Before (backwards pins, full check) | After (correct pins, full check) |
|---|---|---|
| `SEAM-PIN-PASS` | 0 / 1290 | **1211 / 1257 (96.3%)** |
| `seam_uv_consistency` mismatches | 0 (trivial — nothing succeeded) | **0 (real — solves actually agree)** |
| QCE on passing cases | N/A (nothing passed) | max ~1.3–2.2, mean ~1.1–1.8 (well under the 3.0 high-distortion threshold) |

This confirms the fix is real, not just a relaxed check: passing cases now
have good conformal quality, not the ~6.3-max distortion seen when the
flip/fold-over checks were disabled instead.

`check_valid_UV_lscm_diag_no_flip_foldover` and the composed
`check_valid_UV_lscm_full` both remain in `joint_lscm_pinned.cpp` (the
diagnostic as a documented result of the investigation, the full one as the
actual validity gate now in use).

## Remaining: 43/1257 (3.4%) still fail

Not yet investigated. Possible causes to check next: whether these are
correlated with very short `B_arc`s (few `B_reflected` vertices), or
orientation-flip cases that `joint_lscm_case1_dc` has an auto-correction
safety net for (re-solve with swapped pin x-values if majority of top-sheet
faces come out with negative signed area) but `joint_lscm_double_cover`
(and this pinned variant, which mirrors it) does not.
