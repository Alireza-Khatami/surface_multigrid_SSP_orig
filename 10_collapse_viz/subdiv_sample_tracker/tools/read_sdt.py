"""Reader for subdiv_<stem>.sdt (collapse_viz_bin output). See md_files/subdiv_sdt_reading.md."""
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
