#include "joint_lscm_pinned.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <unordered_map>

// ---- local diagnostic log ('static', unlike joint_lscm.cpp's dc_log()
// which is private to that translation unit and not reusable here) ----
static FILE * s_seam_pin_log = nullptr;
static FILE * seam_pin_log() { return s_seam_pin_log ? s_seam_pin_log : stderr; }
void seam_uv_pinned_log_open(const char * path) {
    if (s_seam_pin_log) fclose(s_seam_pin_log);
    s_seam_pin_log = path ? fopen(path, "w") : nullptr;
}
void seam_uv_pinned_log_close() {
    if (s_seam_pin_log) { fclose(s_seam_pin_log); s_seam_pin_log = nullptr; }
}

namespace {

// Fixed pin targets, shared by every sheet of every seam collapse — this is
// what actually forces cross-sheet agreement. Chosen on the existing y=0
// seam line (same line B_glued[0]/[1] already pin to at x=-1/+1) so they're
// consistent with the double cover's existing mirror-symmetry convention
// (check_dc_symmetry already expects glued points to have |y|~=0).
// See md_files/seam_uv_pinning_fix.md §2 for the reasoning, including why
// vk (the post-collapse merged point) reuses vi's target: vi is always the
// survivor (get_post_faces remaps vj -> vi), so vk IS vi, just after the
// collapse — pinning it to vi_target keeps that identity in UV space too.
//
// x-ordering matches the ORIGINAL 2-pin code's own documented natural order
// (src/joint_lscm.cpp:242-255): "B_glued[0] is adjacent to vj and
// B_glued[1] is adjacent to vi ... B_glued[0](-1) < vj < vi < B_glued[1](+1)".
// So vj (adjacent to B_glued[0]) goes near -1, and vi (adjacent to
// B_glued[1], closing the loop) goes near +1 — NOT the other way around.
// (An earlier version of this file had these swapped, which contradicted
// that documented ordering and is suspected to be why the fixed-pin solve
// failed check_valid_UV_lscm's flip/fold-over checks on ~100% of sheets —
// see md_files/seam_uv_pinning_fix.md "Next steps".)
constexpr double kSeamPinViX =  0.5, kSeamPinViY = 0.0;
constexpr double kSeamPinVjX = -0.5, kSeamPinVjY = 0.0;
constexpr double kSeamPinVkX = kSeamPinViX, kSeamPinVkY = kSeamPinViY;

} // namespace

// ---------------------------------------------------------------------------
// check_valid_UV_lscm's checks (src/joint_lscm.cpp:1357-1560), split into 3
// separate, independently reusable functions — mirroring that function's
// logic exactly (check_valid_UV_lscm itself is untouched; these are new,
// local functions), so they can be composed differently for diagnostics
// (see check_valid_UV_lscm_diag_no_flip_foldover below, which uses only the
// quality-threshold one).
// ---------------------------------------------------------------------------

// True iff any face has flipped/degenerate signed area (src/joint_lscm.cpp
// pre/post "face normals flip" checks, lines 1357-1427).
static bool check_uv_face_flip(const Eigen::MatrixXd & UV, const Eigen::MatrixXi & FUV)
{
    for (int ii = 0; ii < FUV.rows(); ii++) {
        Eigen::VectorXd v1 = UV.row(FUV(ii,1)) - UV.row(FUV(ii,0));
        Eigen::VectorXd v2 = UV.row(FUV(ii,2)) - UV.row(FUV(ii,0));
        double signedArea = v1(0) * v2(1) - v1(1) * v2(0);
        if (signedArea < 1e-10 || std::isnan(signedArea)) return true;
    }
    return false;
}

// True iff the UV angle sum around vi or vj exceeds 2*pi — a self-overlap /
// fold-over (src/joint_lscm.cpp pre/post "UV face fold over" checks,
// lines 1429-1501).
static bool check_uv_foldover(const Eigen::MatrixXd & UV, const Eigen::MatrixXi & FUV,
                               const int & vi, const int & vj)
{
    Eigen::MatrixXd internalAng;
    igl::internal_angles(UV, FUV, internalAng);
    double angSum_vi = 0, angSum_vj = 0;
    for (int r = 0; r < FUV.rows(); r++)
        for (int c = 0; c < FUV.cols(); c++) {
            if (FUV(r,c) == vi) angSum_vi += internalAng(r,c);
            if (FUV(r,c) == vj) angSum_vj += internalAng(r,c);
        }
    return (angSum_vi - 2*M_PI) > 1e-10 || (angSum_vj - 2*M_PI) > 1e-10;
}

// True iff any triangle's normalized quality falls below threshold, or is
// NaN (src/joint_lscm.cpp UV_pre/UV_post "triangle quality" checks,
// lines 1503-1560).
static bool check_uv_triangle_quality(const Eigen::MatrixXd & UV, const Eigen::MatrixXi & FUV,
                                       double threshold = 0.01)
{
    for (int ii = 0; ii < FUV.rows(); ii++) {
        int v0 = FUV(ii,0), v1 = FUV(ii,1), v2 = FUV(ii,2);
        double l0 = (UV.row(v0) - UV.row(v1)).norm();
        double l1 = (UV.row(v1) - UV.row(v2)).norm();
        double l2 = (UV.row(v2) - UV.row(v0)).norm();
        double x = (l0+l1+l2) / 2;
        double delta = sqrt(x * (x-l0) * (x-l1) * (x-l2));
        double triQ = 4 * sqrt(3) * delta / (l0*l0 + l1*l1 + l2*l2);
        if (triQ < threshold || std::isnan(triQ)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// TEMPORARY DIAGNOSTIC (see md_files/seam_uv_pinning_fix.md "Diagnostic:
// disabling flip/fold-over checks"): the 5-pin solve showed 1290/1290
// SEAM-PIN-FAIL, 0 pass on the test mesh under the full check. Composed from
// only the NaN check + check_uv_triangle_quality (both pre and post) —
// check_uv_face_flip and check_uv_foldover are deliberately NOT called here.
// With this, 1647/1713 passed and the consistency log stayed at 0
// mismatches — but QCE distortion on the "passing" cases (max up to 6.3) is
// well above the existing [DC-HIGH-DISTORTION] threshold (3.0), meaning the
// flip/fold-over checks were correctly rejecting genuinely invalid
// parameterizations, not being overly strict. This function stays as a
// diagnostic result, not the validity gate to actually use going forward —
// switch joint_lscm_seam_pinned back to the full 3-check composition
// (or check_valid_UV_lscm) once the pin-target strategy itself is fixed.
static bool check_valid_UV_lscm_diag_no_flip_foldover(
    const Eigen::MatrixXd & UV_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & UV_post,
    const Eigen::MatrixXi & FUV_post)
{
    if (UV_pre.array().isNaN().sum() > 0 || UV_post.array().isNaN().sum() > 0)
        return false;
    if (check_uv_triangle_quality(UV_pre,  FUV_pre))  return false;
    if (check_uv_triangle_quality(UV_post, FUV_post)) return false;
    return true;
}

// Full validity gate, composed from all 3 modular checks above (equivalent
// to check_valid_UV_lscm's own logic, just built from the split-out
// functions instead of duplicated inline). This is the gate that should
// actually be used — check_valid_UV_lscm_diag_no_flip_foldover above stays
// only as a diagnostic result, not something to switch back on.
static bool check_valid_UV_lscm_full(
    const Eigen::MatrixXd & UV_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & UV_post,
    const Eigen::MatrixXi & FUV_post,
    const int & vi,
    const int & vj)
{
    if (UV_pre.array().isNaN().sum() > 0 || UV_post.array().isNaN().sum() > 0)
        return false;
    if (check_uv_face_flip(UV_pre,  FUV_pre))  return false;
    if (check_uv_face_flip(UV_post, FUV_post)) return false;
    if (check_uv_foldover(UV_pre,  FUV_pre,  vi, vj)) return false;
    if (check_uv_foldover(UV_post, FUV_post, vi, vj)) return false;
    if (check_uv_triangle_quality(UV_pre,  FUV_pre))  return false;
    if (check_uv_triangle_quality(UV_post, FUV_post)) return false;
    return true;
}
// ---------------------------------------------------------------------------

void joint_lscm_double_cover_pinned(
    const Eigen::MatrixXd & V_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & V_post,
    const Eigen::MatrixXi & FUV_post,
    const int & vi,
    const int & vj,
    const bool isDebug,
    Eigen::MatrixXd & UV_pre,
    Eigen::MatrixXd & UV_post,
    Eigen::MatrixXi & FUV_dc_pre,
    Eigen::MatrixXi & FUV_dc_post,
    Eigen::MatrixXd & UV_dc_pre,
    Eigen::MatrixXd & UV_dc_post,
    std::vector<int> & out_B_glued,
    std::vector<int> & out_B_reflected)
{
    using namespace Eigen;
    using namespace std;

    int nV      = V_pre.rows();
    int nVjoint = nV + 1;   // slot nV = vi's post-collapse 3D position ("vk")

    // --- Build Vjoint: rows 0..nV-1 from V_pre, row nV from V_post.row(vi) ---
    MatrixXd Vjoint(nVjoint, 3);
    for (int i = 0; i < nV; i++) Vjoint.row(i) = V_pre.row(i);
    Vjoint.row(nV) = V_post.row(vi);

    // --- Remap vi -> nV in post face connectivity ---
    MatrixXi Fjoint_pre  = FUV_pre;
    MatrixXi Fjoint_post = FUV_post;
    for (int r = 0; r < Fjoint_post.rows(); r++)
        for (int c = 0; c < 3; c++)
            if (Fjoint_post(r, c) == vi) Fjoint_post(r, c) = nV;

    // --- Step 1: Boundary loop and B-arc extraction (identical to the
    //     original joint_lscm_double_cover — see src/joint_lscm.cpp:137-193) ---
    VectorXi bdLoop_eig;
    igl::boundary_loop(Fjoint_pre, bdLoop_eig);
    int n_loop = (int)bdLoop_eig.size();
    vector<int> bdLoop(bdLoop_eig.data(), bdLoop_eig.data() + n_loop);

    auto nan_skip = [&](const char* reason) {
        fprintf(seam_pin_log(), "[SEAM-PIN-SKIP] vi=%d vj=%d nV=%d: %s\n", vi, vj, nV, reason);
        fflush(seam_pin_log());
        UV_pre  = MatrixXd::Constant(nV, 2, numeric_limits<double>::quiet_NaN());
        UV_post = UV_pre;
    };

    int vi_pos = -1, vj_pos = -1;
    for (int i = 0; i < n_loop; i++) {
        if (bdLoop[i] == vi) vi_pos = i;
        if (bdLoop[i] == vj) vj_pos = i;
    }
    if (vi_pos < 0 || vj_pos < 0) { nan_skip("vi or vj not in boundary loop"); return; }

    vector<int> B_arc;
    {
        int start = (vj_pos + 1) % n_loop;
        if (start == vi_pos) {
            int cur = (vi_pos + 1) % n_loop;
            while (cur != vj_pos) { B_arc.push_back(bdLoop[cur]); cur = (cur+1)%n_loop; }
            reverse(B_arc.begin(), B_arc.end());
        } else {
            int cur = start;
            while (cur != vi_pos) { B_arc.push_back(bdLoop[cur]); cur = (cur+1)%n_loop; }
        }
    }
    // B_arc[0] = B vertex adjacent to vj (glued); B_arc.back() = adjacent to
    // vi (glued); B_arc[1..end-1] = middle B vertices (duplicated). By this
    // construction B_arc never contains vi or vj themselves, so the new
    // vi/vj/nV pins below can never collide with a B_glued/B_reflected pin.

    if ((int)B_arc.size() < 3) {
        nan_skip("interior B arc has fewer than 3 vertices (need >=1 middle B to duplicate)");
        return;
    }

    vector<int> B_glued    = { B_arc.front(), B_arc.back() };
    vector<int> B_reflected(B_arc.begin() + 1, B_arc.end() - 1);
    out_B_glued     = B_glued;
    out_B_reflected = B_reflected;

    // --- Step 2: UV variable layout (identical to the original) ---
    int nVjoint_dc = nVjoint + (int)B_reflected.size();

    unordered_map<int,int> bot_remap;
    for (int k = 0; k < (int)B_reflected.size(); k++)
        bot_remap[B_reflected[k]] = nVjoint + k;

    // --- Step 3: Build Vjoint_dc (identical to the original) ---
    MatrixXd Vjoint_dc(nVjoint_dc, 3);
    Vjoint_dc.topRows(nVjoint) = Vjoint;
    for (int k = 0; k < (int)B_reflected.size(); k++)
        Vjoint_dc.row(nVjoint + k) = Vjoint.row(B_reflected[k]);

    // --- Step 4: Build DC face matrices (identical to the original) ---
    auto make_bot_face = [&](int a, int b_v, int c) -> Vector3i {
        auto remap = [&](int v) -> int {
            auto it = bot_remap.find(v);
            return (it != bot_remap.end()) ? it->second : v;
        };
        return Vector3i(remap(a), remap(c), remap(b_v));  // reversed winding
    };

    int nF_pre  = Fjoint_pre.rows();
    int nF_post = Fjoint_post.rows();
    MatrixXi Fdc_pre(2*nF_pre, 3), Fdc_post(2*nF_post, 3);

    Fdc_pre.topRows(nF_pre)   = Fjoint_pre;
    Fdc_post.topRows(nF_post) = Fjoint_post;
    for (int r = 0; r < nF_pre; r++)
        Fdc_pre.row(nF_pre   + r) = make_bot_face(Fjoint_pre(r,0),  Fjoint_pre(r,1),  Fjoint_pre(r,2));
    for (int r = 0; r < nF_post; r++)
        Fdc_post.row(nF_post + r) = make_bot_face(Fjoint_post(r,0), Fjoint_post(r,1), Fjoint_post(r,2));

    // --- Step 5: Pinning — EXTENDED from the original's 2 points to 5 ---
    //
    // Original (src/joint_lscm.cpp:242-264) pins only B_glued[0]/[1]; vi,
    // vj, and the "vk" slot (nV) are free variables, which is why they
    // disagree across sheets (see md_files/seam_uv_consistency_findings.md).
    //
    // Pin-value convention, taken verbatim from the original's own inline
    // comments ("pin_left : y=0, x=-1" for b_UV={pin_left, nVjoint_dc+pin_left},
    // bc_UV={0.0,-1.0}): for a pinned row index `idx`, the pair
    // (b_UV, bc_UV) = {(idx, y), (nVjoint_dc+idx, x)}.
    int pin_left  = B_glued[0];
    int pin_right = B_glued[1];

    VectorXi b_UV(10);
    VectorXd bc_UV(10);
    b_UV  << pin_left,              nVjoint_dc + pin_left,
             pin_right,             nVjoint_dc + pin_right,
             vi,                    nVjoint_dc + vi,
             vj,                    nVjoint_dc + vj,
             nV,                    nVjoint_dc + nV;
    bc_UV << 0.0,                  -1.0,               // pin_left  : y=0, x=-1
             0.0,                   1.0,               // pin_right : y=0, x=+1
             kSeamPinViY,           kSeamPinViX,        // vi        : y=0, x=-0.5
             kSeamPinVjY,           kSeamPinVjX,        // vj        : y=0, x=+0.5
             kSeamPinVkY,           kSeamPinVkX;        // vk (nV)   : same as vi

    if (isDebug) {
        fprintf(seam_pin_log(),
            "[SEAM-PIN] vi=%d vj=%d  B_glued=(%d,%d)->((-1,0),(1,0))  "
            "vi->(%.2f,%.2f)  vj->(%.2f,%.2f)  vk(nV=%d)->(%.2f,%.2f)\n",
            vi, vj, pin_left, pin_right,
            kSeamPinViX, kSeamPinViY, kSeamPinVjX, kSeamPinVjY, nV, kSeamPinVkX, kSeamPinVkY);
        fflush(seam_pin_log());
    }

    // --- Step 6: Solve (identical mechanism to the original; flatten()/
    //     mqwf_dense are generic over the pin count, no change needed there) ---
    VectorXd UVjoint_flat;
    flatten(Vjoint_dc, Fdc_pre, Vjoint_dc, Fdc_post,
            b_UV, bc_UV, nVjoint_dc, isDebug, UVjoint_flat);

    MatrixXd UVjoint(nVjoint_dc, 2);
    for (int col = 0; col < 2; col++)
        UVjoint.col(1 - col) = UVjoint_flat.segment(nVjoint_dc * col, nVjoint_dc);

    // --- Step 7: Extract UV_pre and UV_post (identical to the original) ---
    UV_pre  = UVjoint.topRows(nV);
    UV_post = UV_pre;
    UV_post.row(vi) = UVjoint.row(nV);

    // --- Step 8: DC visualization data (identical to the original) ---
    FUV_dc_pre  = Fdc_pre;
    FUV_dc_post = Fdc_post;
    UV_dc_pre   = UVjoint;
    UV_dc_post  = UVjoint;
}

bool joint_lscm_seam_pinned(
    const Eigen::MatrixXd & V_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & V_post,
    const Eigen::MatrixXi & FUV_post,
    const int & vi,
    const int & vj,
    const std::vector<int> & Nsv,
    const std::vector<int> & Ndv,
    Eigen::MatrixXd & UV_pre,
    Eigen::MatrixXd & UV_post,
    std::optional<int> * out_case,
    DCVizData * dc_viz,
    int collapse_idx)
{
    using namespace Eigen;
    (void)Nsv; (void)Ndv;  // kept for call-site parity with joint_lscm(); unused here

    if (out_case) *out_case = 2;   // seam collapses are always Case 2

    Eigen::MatrixXi FUV_dc_pre, FUV_dc_post;
    Eigen::MatrixXd UV_dc_pre, UV_dc_post;
    std::vector<int> B_glued, B_reflected;

    joint_lscm_double_cover_pinned(
        V_pre, FUV_pre, V_post, FUV_post, vi, vj,
        /*isDebug=*/false,
        UV_pre, UV_post,
        FUV_dc_pre, FUV_dc_post, UV_dc_pre, UV_dc_post,
        B_glued, B_reflected);

    // Full validity check (flip + fold-over + quality) — testing whether
    // correcting the vi/vj pin ordering (see kSeamPinViX/kSeamPinVjX above)
    // to match the original code's documented natural order actually
    // avoids the flips/fold-overs that caused the earlier 100% fail rate,
    // rather than just disabling the checks that were catching them.
    bool ok = check_valid_UV_lscm_full(UV_pre, FUV_pre, UV_post, FUV_post, vi, vj);

    if (!ok) {
        bool has_nan = UV_pre.array().isNaN().any() || UV_post.array().isNaN().any();
        fprintf(seam_pin_log(),
            "[SEAM-PIN-FAIL #%d] vi=%d vj=%d nFpre=%d nFpost=%d nV=%d nan=%d\n",
            collapse_idx, vi, vj, (int)FUV_pre.rows(), (int)FUV_post.rows(),
            (int)V_pre.rows(), (int)has_nan);
    } else {
        Eigen::VectorXd qce_pre, qce_post;
        quasi_conformal_error(V_pre, FUV_pre,  UV_pre,  qce_pre);
        quasi_conformal_error(V_pre, FUV_post, UV_post, qce_post);
        fprintf(seam_pin_log(),
            "[SEAM-PIN-PASS #%d] vi=%d vj=%d  QCE_pre(max=%.3f mean=%.3f) QCE_post(max=%.3f mean=%.3f)\n",
            collapse_idx, vi, vj,
            qce_pre.maxCoeff(), qce_pre.mean(), qce_post.maxCoeff(), qce_post.mean());
    }
    fflush(seam_pin_log());

    if (dc_viz) {
        dc_viz->has_data    = true;
        dc_viz->FUV_dc_pre  = FUV_dc_pre;
        dc_viz->FUV_dc_post = FUV_dc_post;
        dc_viz->UV_dc_pre   = UV_dc_pre;
        dc_viz->UV_dc_post  = UV_dc_post;
        dc_viz->B_glued     = B_glued;
        dc_viz->B_reflected = B_reflected;

        double asym_err = std::numeric_limits<double>::infinity();
        int sym_val;
        int nVjoint = (int)V_pre.rows() + 1;
        if (UV_dc_post.rows() > 0 && !UV_dc_post.array().isNaN().any()) {
            sym_val = check_dc_symmetry(UV_dc_post, B_reflected, nVjoint, vi, vj, 1e-4, asym_err) ? 1 : 0;
        } else {
            sym_val = -1;
        }
        dc_viz->dc_uv_symmetric    = sym_val;
        dc_viz->dc_uv_asym_max_err = asym_err;
    }

    return ok;
}
