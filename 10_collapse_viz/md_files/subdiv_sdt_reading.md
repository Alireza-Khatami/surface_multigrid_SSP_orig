# Reading `subdiv_<stem>.sdt`: fine, coarse and subdivided samples

The `.sdt` is the main output of the subdivided-sample tracker. For every
sample it holds:

- its position on the **fine** MAT: `(fine face, barycentric)` plus `xyz`;
- its tracked position on the **coarse** (simplified) mesh:
  `(coarse face, barycentric)`;
- the **subdivided mesh** connectivity, for Laplacians;
- its structure ids (sheet / seam / boundary / junction).

A ready-to-use Python reader is in `subdiv_sample_tracker/tools/read_sdt.py`
(numpy only). The code below is the same. How to produce the file is in
`subdiv_tracker_cli.md`.

## Files you need from the run folder

| file | used for |
|---|---|
| `subdiv_<stem>.sdt` | samples, connectivity, coordinates, structure ids |
| `oriented_<stem>.obj` | the **fine** mesh that `fine_face` / `fine_bary` refer to |
| `simplified_<stem>.obj` | the **coarse** mesh that `coarse_face` / `coarse_bary` refer to |

`fine_face` and `fine_bary` refer to **`oriented_<stem>.obj`**, not to
`raw_<stem>.obj` or the input mesh. The raw file has the same vertices and
faces, but some faces have a different corner order (re-wound). Barycentric
coordinates are given in corner order, so they only match the oriented file.

## Layout

Little-endian. A fixed header is followed by arrays, each starting at an
8-byte-aligned offset given in the header.

**Header**

| byte | type | field |
|---|---|---|
| 0 | char[8] | magic `SUBDIVT\0` |
| 8 | u32 | version: 1 = not relaxed, 2 = relaxed |
| 12 | u32 | header_bytes = 112 + 8 x n_arrays |
| 16 | u32 | n_levels (subdivision levels) |
| 20 | u32 | relaxed (0 / 1) |
| 24 | u64 x 11 | n_samples_requested, **Vs** (samples), **Fs** (subdivided faces), n_fine_verts, n_fine_faces, **nE** (fine edges), n_coarse_verts, n_coarse_faces, **P** (palette sets), **nPI** (palette ids), **n_arrays** (14, or 15 when relaxed) |
| 112 | u64 x n_arrays | byte offset of each array |

**Arrays** (row-major)

| # | name | type x columns | rows | meaning |
|---|---|---|---|---|
| 0 | `sub_V` | f64 x 3 | Vs | sample position on the fine MAT (after relaxation, if relaxed) |
| 1 | `sub_F` | i32 x 3 | Fs | subdivided faces: the connectivity for Laplacians |
| 2 | `sub_face_orig` | i32 | Fs | fine face each subdivided face came from |
| 3 | `orig_edges` | i32 x 2 | nE | fine-mesh edges (min, max), sorted |
| 4 | `carrier_type` | u8 | Vs | seed carrier: 0 fine vertex, 1 fine edge, 2 fine face interior |
| 5 | `carrier_index` | i32 | Vs | fine vertex / `orig_edges` row / fine face of the carrier |
| 6 | `fine_face` | i32 | Vs | fine face the sample lies on (-1: an original vertex on no face) |
| 7 | `fine_bary` | f64 x 3 | Vs | barycentric in that fine face's corner order |
| 8 | `coarse_face` | i32 | Vs | face of `simplified_<stem>.obj` (-1: untracked) |
| 9 | `coarse_bary` | f64 x 3 | Vs | barycentric in that coarse face's corner order |
| 10 | `struct_set_id` | i32 | Vs | index of the sample's structure-id set in the palette |
| 11 | `palette_offsets` | i32 | P+1 | set k = `palette_ids[offsets[k] : offsets[k+1]]` |
| 12 | `palette_ids` | i32 | nPI | structure ids (from the `.ma_struct`), sorted within each set |
| 13 | `palette_type_mask` | u8 | P | per set: bit 1 sheet, 2 seam, 4 boundary, 8 junction |
| 14 | `sub_V_seed` | f64 x 3 | Vs | version 2 only: position before relaxation |

**Index facts**

- Sample `i` is row `i` of every per-sample array and vertex `i` of `sub_F`
  and of every exported subdivided OBJ.
- Samples `0 .. n_fine_verts-1` are the original MAT vertices, in their
  original order. After relaxation this holds for identity, not position:
  only junctions and fixed samples stay put. The original positions are in
  `sub_V_seed`.
- Samples with `fine_face == -1` are MAT vertices used by no face. They
  include the loose vertices of stale chains (OBJ `l` polylines that are not
  face edges). They are never relaxed, never tracked (`coarse_face == -1`),
  and have no edges in `sub_F`. Their coarse position is their `sub_V`, because
  decimation locks stale-chain vertices.
- The levels are nested. The first samples are exactly the vertices of each
  coarser level (see `level_faces` below).
- `carrier_*` describe the seed position (before relaxation). The structure
  ids were derived from it, and a relaxed sample stays on the same structure.

## Python reader

```python
import struct
import numpy as np

SDT_ARRAYS = [  # (name, dtype, columns or None, row count key)
    ("sub_V",             "<f8", 3, "Vs"),
    ("sub_F",             "<i4", 3, "Fs"),
    ("sub_face_orig",     "<i4", None, "Fs"),
    ("orig_edges",        "<i4", 2, "nE"),
    ("carrier_type",      "u1",  None, "Vs"),
    ("carrier_index",     "<i4", None, "Vs"),
    ("fine_face",         "<i4", None, "Vs"),
    ("fine_bary",         "<f8", 3, "Vs"),
    ("coarse_face",       "<i4", None, "Vs"),
    ("coarse_bary",       "<f8", 3, "Vs"),
    ("struct_set_id",     "<i4", None, "Vs"),
    ("palette_offsets",   "<i4", None, "P+1"),
    ("palette_ids",       "<i4", None, "nPI"),
    ("palette_type_mask", "u1",  None, "P"),
    ("sub_V_seed",        "<f8", 3, "Vs"),   # version 2 (relaxed) only
]


def read_sdt(path):
    """Read a subdiv_<stem>.sdt file. Returns (header dict, dict of numpy arrays)."""
    buf = open(path, "rb").read()
    if buf[:8] != b"SUBDIVT\0":
        raise ValueError("not an .sdt file")
    version, header_bytes, n_levels, relaxed = struct.unpack_from("<4I", buf, 8)
    (n_req, Vs, Fs, nVO, nFO, nE, nCV, nCF, P, nPI, n_arrays) = struct.unpack_from("<11Q", buf, 24)
    offsets = struct.unpack_from("<%dQ" % n_arrays, buf, 112)
    H = dict(version=version, relaxed=relaxed, n_levels=n_levels, n_samples_requested=n_req,
             Vs=Vs, Fs=Fs, n_fine_verts=nVO, n_fine_faces=nFO, nE=nE,
             n_coarse_verts=nCV, n_coarse_faces=nCF, P=P, nPI=nPI, n_arrays=n_arrays)
    rows = {"Vs": Vs, "Fs": Fs, "nE": nE, "P+1": P + 1, "nPI": nPI, "P": P}
    A = {}
    for k in range(n_arrays):
        name, dt, cols, key = SDT_ARRAYS[k]
        n = rows[key] * (cols or 1)
        a = np.frombuffer(buf, dtype=dt, count=n, offset=offsets[k])
        A[name] = a.reshape(rows[key], cols) if cols else a
    return H, A


def read_obj(path):
    """Minimal OBJ reader: vertex positions and triangle faces (0-based)."""
    V, F = [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            V.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            F.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
    return np.array(V), np.array(F, dtype=np.int64)


def barycentric_positions(V, F, face, bary):
    """Positions of points given as (face, barycentric) on mesh (V, F). face < 0 -> NaN."""
    P = np.full((len(face), 3), np.nan)
    ok = face >= 0
    P[ok] = np.einsum("ij,ijk->ik", bary[ok], V[F[face[ok]]])
    return P


def struct_sets(A):
    """Per-sample structure ids (list of int arrays) and type masks."""
    off, ids, mask = A["palette_offsets"], A["palette_ids"], A["palette_type_mask"]
    sets = [ids[off[k]:off[k + 1]] for k in range(len(mask))]
    sid = A["struct_set_id"]
    return [sets[k] for k in sid], mask[sid]


def level_faces(A, H, level):
    """Faces of a coarser subdivision level (nested: uses only the first vertices)."""
    k = H["n_levels"] - level
    block = 4 ** k
    off1 = (block - 1) // 3
    F = A["sub_F"]
    base = np.arange(H["Fs"] // block) * block
    return np.stack([F[base, 0], F[base + off1, 1], F[base + 2 * off1, 2]], axis=1)
```

## Common tasks

```python
import glob
d = "output/relaxation_experiments/l3_defaults/"          # a run folder
H, A = read_sdt(glob.glob(d + "subdiv_*.sdt")[0])

# 1. Subdivided samples and their mesh (for Laplacians)
V_sub, F_sub = A["sub_V"], A["sub_F"]                     # Vs x 3, Fs x 3

# 2. Fine samples: position on the fine MAT, recomputed from (face, bary)
VO, FO = read_obj(glob.glob(d + "oriented_*.obj")[0])     # NOT raw_*.obj
P_fine = barycentric_positions(VO, FO, A["fine_face"], A["fine_bary"])   # == sub_V

# 3. Coarse samples: where each sample ended up on the simplified mesh
CV, CF = read_obj(glob.glob(d + "simplified_*.obj")[0])
P_coarse = barycentric_positions(CV, CF, A["coarse_face"], A["coarse_bary"])

# fine -> coarse correspondence per sample:
#   (A["fine_face"][i],   A["fine_bary"][i])   on the fine MAT
#   (A["coarse_face"][i], A["coarse_bary"][i]) on the coarse mesh

# 4. Structure ids of each sample
ids, mask = struct_sets(A)          # ids[i]: array of struct ids; mask[i]: type bits
is_junction = (mask & 8) > 0
is_curve    = ((mask & (2 | 4)) > 0) & ~is_junction   # seam or boundary
is_sheet    = ~is_junction & ~is_curve

# 5. A coarser, nested level of the same samples
F_lvl1 = level_faces(A, H, 1)       # uses only samples 0 .. max(F_lvl1)

# 6. Positions before relaxation (version 2 only)
if H["relaxed"]:
    V_seed = A["sub_V_seed"]
```

Samples with `fine_face == -1` are original MAT vertices used by no face.
They are untracked, so they also have `coarse_face == -1`. There were none on
the test mesh.

## Verified on a real run

`output/relaxation_experiments/l3_defaults` (level 3, 88,124 samples, default
relaxation):

| check | result |
|---|---|
| fine positions from `(fine_face, fine_bary)` on `oriented_*.obj` vs `sub_V` | max difference 0 |
| coarse positions from `(coarse_face, coarse_bary)` on `simplified_*.obj` vs `subdiv_fine_at_coarse_pos_*.obj` | 7e-12 (OBJ text precision) |
| `subdiv_fine_at_coarse_pos_*.obj` faces vs `sub_F` | identical |
| `subdiv_fine_relaxed_*.obj` vs `sub_V`, `subdiv_fine_seed_*.obj` vs `sub_V_seed` | 5e-13 / 2e-13 (OBJ text precision) |
| `level_faces(A, H, 0)` vs the fine faces | identical |
| `raw_*.obj` vs `oriented_*.obj` | same vertices, same face sets, different corner order |

## C++

The writer is `subdiv_tracker_save` in `subdiv_sample_tracker/subdiv_tracker.cpp`.
Its comment block lists the same layout. To read the file, `memcpy` the header
fields at the byte positions above, then each array from its offset. All
arrays are plain little-endian `double` / `int32` / `uint8`, row-major.
