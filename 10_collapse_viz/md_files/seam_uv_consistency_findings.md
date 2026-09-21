# Seam UV-Consistency Findings — First Headless Run

## What was checked

The logger added in [[seam_multisheet_uv_overlay]] (`SSP_seam_uv_log_open`,
`src/SSP_collapse_edge.cpp`) checks, for every seam collapse (2+ active
sheets), whether the following agree across all active sheets, within
`1e-9`:

- **`vi`** — the surviving vertex's `UV_pre` position
- **`vj`** — the absorbed vertex's `UV_pre` position
- **`vk`** — the merged post-collapse point's `UV_post` position (`UV_post`
  row of the survivor, `vi`)

Each active sheet solves `joint_lscm` independently (Case 2/3, double cover,
each pinning its own `B_glued` arc at `(-1,0)`/`(+1,0)`), so these are *not*
expected to agree by construction — the point of the logger is to quantify
how often and how badly they disagree in practice.

## Run setup

- Mesh: `mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.obj`
  (1506 vertices, 2715 faces, 24 sheets, 474 seam edges pre-decimation)
- Mode: `qslim`, target 200 faces, `--validity-checks` on
- Build: headless verification build (`C2F_VIZ_DIAGNOSTIC=OFF`, no GUI, runs
  to completion automatically) — `10_collapse_viz/build/headless_verify`
- Result: run completed normally, 1159 total collapses, final sanity check
  passed (`[SANITY] OK: obj/json/bundle agree on 347 vertices...`)
- Raw log: `10_collapse_viz/output/01_00040057_f8f78dbd17414efda75bc437_trimesh_000/seam_uv_consistency_mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.txt`

## Results

| Metric | Value |
|---|---|
| Seam collapse attempts (`[SEAM-TRY]` in `seam_diag_*.txt`) | 545 |
| Distinct collapses with ≥1 logged mismatch | **368** |
| Total mismatch log lines | 2208 |
| — of which `point=vi` | 736 |
| — of which `point=vj` | 736 |
| — of which `point=vk` | 736 |
| Smallest logged diff (tol=1e-9) | 3.39e-07 |
| **Largest logged diff** | **2.82e-01** |

368 of 545 seam-collapse attempts (~68%) produced at least one point whose
UV coordinate disagreed across sheets by more than the 1e-9 tolerance — and
the mismatch is split essentially evenly across `vi`, `vj`, and `vk`, so it
is not concentrated in one particular point.

### Worst case

```
collapse=#1148  e=(239,344)  point=vj
  sheet=2(sid=14)  UV=(-0.169064887, -0.000000000)
  ref_sheet=0(sid=0)  UV=(-0.450756991,  0.000000000)
  diff = 0.2816921044

collapse=#1148  e=(239,344)  point=vj
  sheet=1(sid=13)  UV=(-0.186602245,  0.000000000)
  ref_sheet=0(sid=0)  UV=(-0.450756991,  0.000000000)
  diff = 0.2641547458

collapse=#1148  e=(239,344)  point=vk
  sheet=1(sid=13)  UV=( 0.055047196,  0.000000000)
  ref_sheet=0(sid=0)  UV=(-0.182781100,  0.000000000)
  diff = 0.2378282960
```

On a UV domain that spans roughly `[-1, 1]` (the DC pinning range), a
`0.28` disagreement is ~14% of the full span — this is not a rounding
artifact, it's a materially different UV placement for the same global
vertex depending on which sheet solved it.

Second-worst case, `collapse=#3 e=(466,1275)`, shows the same pattern
(`vk` diff `0.255`, `vi` diff `0.237`).

## Interpretation

This confirms the hypothesis behind the UV View feature
([[seam_multisheet_uv_overlay]]): per-sheet independent `joint_lscm` solves
for a seam collapse do not produce a consistent UV frame across sheets for
the majority of seam collapses in this mesh, and the disagreement is
sometimes large relative to the UV domain size, not just numerical noise.

Not yet investigated:
- Whether mismatch magnitude correlates with sheet count, arc length
  (`B_arc`), or local mesh distortion.
- Whether this is specific to this mesh/decimation config, or general across
  meshes.
- Downstream impact — whether/how this affects final coarse-to-fine
  correspondence quality (the run's own sanity checks passed, but those
  don't check cross-sheet UV agreement).

## Reproducing

```
cmake -S 10_collapse_viz -B 10_collapse_viz/build/headless_verify -DC2F_VIZ_DIAGNOSTIC=OFF
cmake --build 10_collapse_viz/build/headless_verify --target collapse_viz_bin --config Debug

10_collapse_viz/build/headless_verify/Debug/collapse_viz_bin.exe \
  --mesh_path <mesh.obj> --matstruct_path <mesh.ma_struct> \
  --target_faces 200 --mode qslim \
  --output_dir <out_dir> --validity-checks --trace_vertices trace_vids.txt
```

Then inspect `<out_dir>/seam_uv_consistency_<stem>.txt`.
