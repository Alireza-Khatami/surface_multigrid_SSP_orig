#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/c/Users/alirz/Projects/Graphics/Neural QMAT/external/surf_subgrid_SSP_orig"
MESH_FILE="/c/Users/alirz/Projects/Graphics/QMAT_old working version  exe file/qmat_x64/qmat/output/01_00040057_f8f78dbd17414efda75bc437_trimesh_000/01_00040057_f8f78dbd17414efda75bc437_trimesh_000_mat_initial.off"

cd "$PROJECT_DIR"

python check_mesh_orientation.py "$MESH_FILE"