"""
check_struct_ids.py
Verifies that every surviving vertex in the simp_visualize_info JSON has
struct_ids that exactly match the ground-truth sets from the .ma_struct file.

Usage:
    python check_struct_ids.py <matstruct_path> <json_path>

Logic:
  - The surviving gV index for each JSON vertex = min(original_ancestors),
    because SSP_collapse_edge always keeps the lower-index vertex (sv = min).
  - struct_ids should equal the ground-truth set from the .ma_struct file,
    since the collapse gate only allows same-struct collapses (union is no-op).
"""

import sys
import json


def parse_matstruct(path):
    """Returns vertex_struct_ids: list of sets (one per vertex)."""
    with open(path) as f:
        tokens = f.read().split()

    pos = 0
    def tok():
        nonlocal pos
        v = tokens[pos]; pos += 1
        return v

    nv, ne, nf = int(tok()), int(tok()), int(tok())

    # Skip vertex lines: "v x y z r"
    for _ in range(nv):
        tok()  # 'v'
        tok(); tok(); tok(); tok()  # x y z r

    # Read edge endpoints
    edge_verts = []
    for _ in range(ne):
        tok()  # 'e'
        u, v = int(tok()), int(tok())
        edge_verts.append((u, v))

    # Read face corners
    face_verts = []
    for _ in range(nf):
        tok()  # 'f'
        a, b, c = int(tok()), int(tok()), int(tok())
        face_verts.append((a, b, c))

    vertex_struct_ids = [set() for _ in range(nv)]

    # Struct section
    if pos >= len(tokens):
        print("No struct section found.")
        return vertex_struct_ids

    num_structs = int(tok())
    for _ in range(num_structs):
        struct_id = int(tok())
        type_id   = int(tok())
        count     = int(tok())
        for _ in range(count):
            elem_id = int(tok())
            if type_id == 3:  # JUNCTION → vertex
                if 0 <= elem_id < nv:
                    vertex_struct_ids[elem_id].add(struct_id)
            elif type_id == 1:  # SEAM → edge endpoints
                if 0 <= elem_id < ne:
                    u, v = edge_verts[elem_id]
                    vertex_struct_ids[u].add(struct_id)
                    vertex_struct_ids[v].add(struct_id)
            elif type_id == 2:  # BOUNDARY → edge endpoints
                if 0 <= elem_id < ne:
                    u, v = edge_verts[elem_id]
                    vertex_struct_ids[u].add(struct_id)
                    vertex_struct_ids[v].add(struct_id)
            elif type_id == 0:  # SHEET → face corners
                if 0 <= elem_id < nf:
                    a, b, c = face_verts[elem_id]
                    vertex_struct_ids[a].add(struct_id)
                    vertex_struct_ids[b].add(struct_id)
                    vertex_struct_ids[c].add(struct_id)

    return vertex_struct_ids


DEFAULT_MATSTRUCT = (
    r"D:\datasets\abc_full_10k\out_ABC_v6_knn_poission40_20_15_10"
    r"\01_00040057_f8f78dbd17414efda75bc437_trimesh_000\mat"
    r"\mat_01_00040057_f8f78dbd17414efda75bc437_trimesh_000.obj__2025-05-06_02_38_00.ma_struct"
)
DEFAULT_JSON = (
    r"C:\Users\alirz\Projects\Graphics\Neural QMAT\external\surf_subgrid_SSP_orig"
    r"\10_collapse_viz\output\01_00040057_f8f78dbd17414efda75bc437_trimesh_000"
    r"\01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial_simp_visualize_info.json"
)


def main():
    matstruct_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATSTRUCT
    json_path      = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_JSON

    if len(sys.argv) == 1:
        print(f"[check_struct_ids] No args — using defaults:")
        print(f"  matstruct: {matstruct_path}")
        print(f"  json:      {json_path}")

    print(f"Parsing {matstruct_path} ...")
    gt = parse_matstruct(matstruct_path)
    print(f"  -> {len(gt)} vertices")

    print(f"Loading {json_path} ...")
    with open(json_path) as f:
        data = json.load(f)
    verts = data["vertices"]
    print(f"  -> {len(verts)} simplified vertices")

    mismatches = []
    no_ancestors = []
    ok = 0

    for i, v in enumerate(verts):
        anc = v["original_ancestors"]
        if not anc:
            no_ancestors.append(i)
            continue

        sv = min(anc)  # surviving gV index (always min, since s < d at each collapse)
        expected = gt[sv]
        actual   = set(v["struct_ids"])

        if expected != actual:
            mismatches.append({
                "json_idx":  i,
                "sv_gv":     sv,
                "expected":  sorted(expected),
                "actual":    sorted(actual),
                "missing":   sorted(expected - actual),
                "extra":     sorted(actual - expected),
            })
        else:
            ok += 1

    print(f"\nResults:")
    print(f"  OK:           {ok} / {len(verts)}")
    print(f"  Mismatches:   {len(mismatches)}")
    print(f"  No ancestors: {len(no_ancestors)}")

    if mismatches:
        print(f"\nFirst 10 mismatches:")
        for m in mismatches[:10]:
            print(f"  json[{m['json_idx']}]  sv_gv={m['sv_gv']}")
            print(f"    expected : {m['expected']}")
            print(f"    actual   : {m['actual']}")
            print(f"    missing  : {m['missing']}")
            print(f"    extra    : {m['extra']}")
    else:
        print("\nAll struct_ids match ground truth.")

    # ── ancestor coverage check ──────────────────────────────────────────────
    # The union of all original_ancestors across every surviving vertex must
    # equal the complete set of initial vertex indices {0 .. nv-1}.
    # Missing = original vertices that were never tracked (lost during collapse).
    # Extra   = ancestor indices that are out of range (> nv-1), which would
    #           indicate a bug in the ancestor tracking code.
    nv = len(gt)
    expected_all = set(range(nv))
    actual_all   = set()
    duplicate_coverage = {}  # ancestor index -> list of json vertex indices that claim it

    for i, v in enumerate(verts):
        for a in v["original_ancestors"]:
            if a in actual_all:
                duplicate_coverage.setdefault(a, []).append(i)
            actual_all.add(a)

    missing_ancestors = sorted(expected_all - actual_all)
    extra_ancestors   = sorted(actual_all   - expected_all)

    print(f"\n── Ancestor coverage (union over all {len(verts)} simplified vertices) ──")
    print(f"  Initial vertices (nv):      {nv}")
    print(f"  Unique ancestors in JSON:   {len(actual_all)}")
    print(f"  Missing from union:         {len(missing_ancestors)}"
          + (" ✓" if not missing_ancestors else ""))
    print(f"  Out-of-range in union:      {len(extra_ancestors)}"
          + (" ✓" if not extra_ancestors else ""))
    print(f"  Ancestors shared by >1 vertex (overlap): {len(duplicate_coverage)}"
          + (" ✓" if not duplicate_coverage else " (unexpected for tree-collapse)"))

    if missing_ancestors:
        show = missing_ancestors[:20]
        print(f"  First missing indices: {show}"
              + (" ..." if len(missing_ancestors) > 20 else ""))
    if extra_ancestors:
        print(f"  Out-of-range indices: {extra_ancestors[:20]}")
    if duplicate_coverage:
        sample = list(duplicate_coverage.items())[:5]
        for anc_idx, claimants in sample:
            print(f"  ancestor {anc_idx} claimed by json verts: {claimants}")

    if not missing_ancestors and not extra_ancestors:
        print("\n  All initial vertices accounted for in ancestor sets. ✓")


if __name__ == "__main__":
    main()
