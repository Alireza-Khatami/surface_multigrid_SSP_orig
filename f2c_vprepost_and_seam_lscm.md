# V_pre/V_post per Sheet and Seam LSCM Case

## The problem

`single_collapse_data` has top-level `V_pre`/`V_post` fields (3D ring geometry), but they
are only set from the **first successful sheet** (see `if (!any_sheet_ok)` in
`SSP_collapse_edge.cpp`).  `SheetData` — the per-sheet struct — stores only UV (2D)
geometry.  For seam collapses with 2+ active sheets, the other sheets' 3D ring geometry
is computed into local variables `V_pre_si`/`V_post_si` and then discarded.

The collapse placement `p` is the **same for all sheets** — it is passed once into the
inner overload from `C.row(e)` and substituted identically per sheet:

```cpp
MatrixXd V_post_si = V_pre_si;
V_post_si.row(b_si(0)) = p;   // same p every sheet
```

Then `joint_lscm` is called on each sheet's `V_pre_si`/`V_post_si` independently to
produce that sheet's `UV_pre_si`/`UV_post_si`.

---

## Seam collapses → joint_lscm Case 2 (both endpoints on boundary)

Before calling `joint_lscm`, seam collapses inject `-1` (the infinity-vertex sentinel) at
position 0 of **both** `Nsv_local` and `Ndv_local` (`SSP_collapse_edge.cpp` lines 568-573):

```cpp
if (active_sheets.size() > 1) {
    if (std::find(Nsv_local.begin(), Nsv_local.end(), -1) == Nsv_local.end())
        Nsv_local.insert(Nsv_local.begin(), -1);
    if (std::find(Ndv_local.begin(), Ndv_local.end(), -1) == Ndv_local.end())
        Ndv_local.insert(Ndv_local.begin(), -1);
}
```

Inside `joint_lscm`, the boundary detection (`joint_lscm.cpp` lines 40-55) checks whether
`-1` appears in each walk:

```cpp
if(std::find(Nsv.begin(), Nsv.end(), infVIdx) != Nsv.end())  onBd(0) = 1;
if(std::find(Ndv.begin(), Ndv.end(), infVIdx) != Ndv.end())  onBd(1) = 1;
```

Because `-1` was injected into both walks, `onBd.sum() == 2` for every seam collapse sheet.
This routes to the **"both endpoints on boundary" branch** (`bdLoop` construction at line 144),
which is Case 2 in the code (0-indexed: 0=both interior, 1=one on boundary, 2=both on
boundary).  In the user's 1-based numbering this is Case 3.

Why this is correct: the seam edge borders multiple sheets, so each sheet's one-ring fan for
`v_s` and `v_d` is naturally open at the seam — the two endpoints really do behave as
boundary vertices within that sheet's patch.

---

## Fix: store V_pre/V_post per sheet in SheetData

To enable exact pre/post 3D visualization for all sheets, `V_pre_si`/`V_post_si` must be
stored inside each `SheetData` rather than only at the top level of `single_collapse_data`.

### Changes

**`src/single_collapse_data.h`** — add `V_pre`/`V_post` to `SheetData`:

```cpp
struct SheetData
{
    int global_sheet_id = -1;
    Eigen::VectorXi b;
    Eigen::VectorXi subsetVIdx;
    Eigen::MatrixXd UV_pre, UV_post;
    Eigen::MatrixXi FUV_pre, FUV_post;
    Eigen::VectorXi FIdx_pre, FIdx_post;
    Eigen::MatrixXd V_pre, V_post;   // ← NEW: 3D ring geometry (local indices, see subsetVIdx)
};
```

**`src/SSP_collapse_edge.cpp`** — populate them at the `store_sheet_data:` label:

```cpp
store_sheet_data:
SheetData sd;
sd.global_sheet_id = sid;
sd.b          = b_si;
sd.subsetVIdx = subsetVIdx_si;
sd.UV_pre     = UV_pre_si;
sd.UV_post    = UV_post_si;
sd.FUV_pre    = FUV_pre_si;
sd.FUV_post   = FUV_post_si;
sd.FIdx_pre   = FIdx_pre_si;
sd.FIdx_post  = FIdx_post_si;
sd.V_pre      = V_pre_si;    // ← NEW
sd.V_post     = V_post_si;   // ← NEW
data.sheets.push_back(sd);
```

### How to use during the F2C walk

At step `ci`, after the UV cast gives you `face` and `BC`, the exact 3D position is:

```python
sd = decInfo[ci].sheets[sheet_idx]
# local indices of the post-collapse face row
a, b, c = sd.FUV_post[post_row]
pos3d = BC[0]*sd.V_post[a] + BC[1]*sd.V_post[b] + BC[2]*sd.V_post[c]
```

`sd.V_post` is in local index space; `sd.subsetVIdx` maps local → global if needed.

### Bundle serialization

`V_pre`/`V_post` are currently **not written** to the `.c2f` bundle file.  To make them
available in Python, add a per-sheet serialization block in `coarse_fine_save_bundle`
alongside the existing UV writes.
