#pragma once
// SSP_rejection_detail.h
// Lightweight structs that carry full diagnostic context for each rejection type.
// Only used for logging — no math here.
#include <Eigen/Core>
#include <vector>
#include <array>
#include <cstdio>

// ── Global collapse counter ───────────────────────────────────────────────────
// main.cpp calls SSP_rej_set_collapse_num(gCollapseCount) after each successful
// collapse so that rejection structs can stamp the collapse number.
void SSP_rej_set_collapse_num(int n);
int  SSP_rej_get_collapse_num();

// ── QSLIM Euclidean rejection (face-flip or skinny-triangle quality) ──────────
struct QslimRejDetail {
    // Collapse context
    int collapse_num = -1;
    int edge = -1, va = -1, vb = -1;
    Eigen::RowVector3d pos_va, pos_vb, pos_opt;   // 3-D positions
    double cost = 0.0;
    bool used_midpoint = false;

    // Rejection trigger
    bool is_flip = false;      // true=face_flip, false=quality
    int  reject_fi = -1;       // face that triggered rejection
    double dot = 0.0;          // normal dot (face_flip)
    double q   = 0.0;          // quality value (quality)
    Eigen::Vector3d n_pre, n_post;  // face normals before/after (face_flip)

    // Rejecting face vertices
    int fv0 = -1, fv1 = -1, fv2 = -1;
    Eigen::RowVector3d fv0_3d, fv1_3d, fv2_3d;   // pre-collapse 3-D

    // One-ring of the collapsed edge (face indices from VF[va] ∪ VF[vb])
    std::vector<int> ring_va;   // VF[va]
    std::vector<int> ring_vb;   // VF[vb]

    void write(FILE* f) const;
};

// ── UV face-flip rejection ────────────────────────────────────────────────────
struct UvFlipRejDetail {
    int collapse_num = -1;
    int sid = -1, eu = -1, ev = -1;   // sheet id, edge endpoints

    // The specific face whose signed area flipped
    int reject_fi = -1;
    int uv_a = -1, uv_b = -1, uv_c = -1;          // UV-face vertex indices
    Eigen::RowVector2d uv_a_pre, uv_b_pre, uv_c_pre;    // pre-collapse UV
    Eigen::RowVector2d uv_a_post, uv_b_post, uv_c_post; // post-collapse UV
    double signed_area_pre  = 0.0;
    double signed_area_post = 0.0;

    // Full post-collapse UV patch (for one-ring context)
    // Stored as face-list: each entry is {uv_idx_0, uv_idx_1, uv_idx_2}
    std::vector<std::array<int,3>>               ring_faces;
    std::vector<std::array<Eigen::RowVector2d,3>> ring_uvs_post;

    void write(FILE* f) const;
};

// ── UV angle-sum rejection ────────────────────────────────────────────────────
struct UvAngleRejDetail {
    int collapse_num = -1;
    int sid = -1, eu = -1, ev = -1;

    int    bad_uv_v = -1;
    double angle_sum = 0.0, diff_from_2pi = 0.0;
    int    n_interior = 0, n_uv = 0, n_fuv = 0, n_faces_for_bad_v = 0;
    bool   degenerate = false;   // true if an edge was near-zero length

    Eigen::RowVector2d bad_v_uv_post;

    // All faces in the post-collapse UV patch that contain bad_uv_v
    std::vector<std::array<int,3>>               bad_v_faces;
    std::vector<std::array<Eigen::RowVector2d,3>> bad_v_face_uvs;

    void write(FILE* f) const;
};
