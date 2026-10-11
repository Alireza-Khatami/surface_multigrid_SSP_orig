# Radius validation of a relaxed `.sdt`

Code: `relaxation_scripts_python/validate_radii.py`. It also runs inside `run_relax.py` when
`--surface_mesh` is given.

## What is checked

Every mapped sample of `coarse_subdiv_relaxed_<stem>.sdt` lies on the fine MAT.

- **Its radius** is interpolated from the fine MAT vertex radii (the `v x y z r` lines of the
  `.ma_struct`) with the sample's stored location: r = w0·r_A + w1·r_B + w2·r_C, where
  (A, B, C) = `gFO[fine_face]` and (w0, w1, w2) = `fine_bary`.
- **Its sphere** has centre `sub_V` and radius r. The sphere is checked against the **surface
  mesh** the MAT was computed from, with exact point-to-triangle distances.

| check | condition | why |
|---|---|---|
| touches the surface | \|d − r\| ≤ tol_touch, with d = distance from the centre to the surface | a medial sphere is tangent to the surface: d < r cuts into it, d > r does not reach it |
| two contact points | two surface points within r + tol_touch of the centre are at least sep_abs apart | a medial sphere touches the surface at ≥ 2 points |
| inside the surface bounding box | centre ± r within the box, up to tol_bbox | a sphere inside the shape cannot leave its box |

Tolerances are relative to the surface's bounding-box diagonal:
- tol_touch = 1e-3
- tol_bbox = tol_touch, because a sphere that cuts into the surface by ε sticks out of the box by up to ε
- sep_abs = 1e-3

Samples with `fine_face < 0` (unmapped, see `coarse_subdiv_relaxed_sdt.md`) are skipped and
counted.

## Zero and tiny radii (medial-axis ends, sharp features)

At sharp features of the surface the medial axis reaches the surface. There the radius goes
to 0 (or to the MAT's radius floor, 0.1 on 00040093) and the two contact points merge into
one.

So a sphere with **r ≤ tol_touch** is treated as a medial-axis end:
- It **is exempt** from the two-contact check. It is counted and logged as "medial-axis ends".
- It **must still touch the surface**: |d − r| ≤ tol_touch. For r = 0 that means the centre
  lies on the surface.
- It **must stay inside the bounding box.**

A sample inside a MAT triangle with zero-radius corners gets the interpolated radius, which is
small but positive inside the triangle. If it is ≤ tol_touch it is handled as an end. If it is
larger, it must have two distinct contacts like any other sample.

## Why these tolerances

The MAT's own vertices are checked the same way (`--check_mat_vertices`). The interpolated
spheres cannot be held to a stricter test than the vertices they come from.

On ABC 00040093 (surface 298 vertices / 592 faces, diagonal ≈ 1450):

| spheres | d − r (× diagonal) | leaves the box by | smallest contact separation | ends exempt | result |
|---|---|---|---|---|---|
| MAT vertices (2,595) | −2.9e-4 … +1.3e-5 | ≤ 2.9e-4 | 0.0158 | 380 | pass |
| relaxed samples (277,698) | −2.7e-4 … +2.9e-4 | ≤ 2.7e-4 | 0.00113 | 2,674 | pass |

Without these allowances, two of the checks would fail on the MAT itself:
- **A zero box tolerance:** 53 MAT vertices stick out of the box by up to 0.41 units, which
  equals their touch error.
- **Contacts at least half a radius apart:** 367 MAT vertices fail. These are the medial-axis
  ends, and vertices near them whose contacts are close together, e.g. r = 358 with contacts
  29 apart.

## It catches wrong radii (tested on 00040093)

| wrong input | failed samples |
|---|---|
| radii 2 % too large | 186,855 cut into the surface |
| radii 2 % too small | 186,909 do not reach it |
| radius interpolated in the **old** triangle of a sample that slid (extrapolated weights) | 5 cut into the surface by up to 1.9 units |
| centre at the position before relaxation | ~99k cut in, ~93k do not reach |

## Failure

Each failing check logs its count and its 5 worst samples (radius, distance, contact
separation, box excess), and the script raises `RuntimeError`. Inside `run_relax.py` the run
fails: the error is logged in `run.log` and `experiment_config.txt`, and the exit code is 1.

## Usage

```
python validate_radii.py --sdt <run>/coarse_subdiv_relaxed_<stem>.sdt \
    --matstruct_path <fine MAT .ma_struct> --surface <surface .obj> [--check_mat_vertices]
    [--tol_touch 1e-3] [--tol_bbox <tol_touch>] [--sep_abs 1e-3] [--fine_obj <MAT .obj>]

python run_relax.py ... --surface_mesh <surface .obj> [--radii_tol_touch 1e-3]
```

By default the fine MAT faces come from the bundle `correspondence_*.c2f` one folder above the
`.sdt`. Use `--fine_obj` to give them directly.
