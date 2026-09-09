#include "SSP_rejection_detail.h"
#include <cmath>
#include <cstdio>
#include <sstream>
#include <mutex>

// ── Global collapse counter ───────────────────────────────────────────────────
static int s_collapse_num = -1;
void SSP_rej_set_collapse_num(int n) { s_collapse_num = n; }
int  SSP_rej_get_collapse_num()      { return s_collapse_num; }

// ── File-write mutex ──────────────────────────────────────────────────────────
// The initial cost pass runs under igl::parallel_for, so multiple threads can
// call write() concurrently.  We buffer the full detail block into a string and
// flush it in one fputs under the lock so entries are never interleaved.
static std::mutex s_write_mtx;

static void atomic_write(FILE* f, const std::string& s) {
    if (!f || s.empty()) return;
    std::lock_guard<std::mutex> lk(s_write_mtx);
    fputs(s.c_str(), f);
    fflush(f);
}

// ── Formatting helpers (into std::ostringstream) ──────────────────────────────
static void app_vec3(std::ostringstream& o, const char* label, const Eigen::RowVector3d& v) {
    char buf[128];
    snprintf(buf, sizeof(buf), "  %s=(%.5g,%.5g,%.5g)\n", label, v(0), v(1), v(2));
    o << buf;
}
static void app_vec2(std::ostringstream& o, const char* label, const Eigen::RowVector2d& v) {
    char buf[128];
    snprintf(buf, sizeof(buf), "  %s=(%.5g,%.5g)\n", label, v(0), v(1));
    o << buf;
}

// ── QslimRejDetail::write ─────────────────────────────────────────────────────
void QslimRejDetail::write(FILE* f) const {
    if (!f) return;
    std::ostringstream o;
    char buf[512];

    snprintf(buf, sizeof(buf),
        "[QSLIM-REJECT-DETAIL] collapse=#%d  reason=%s"
        "  e=%d  va=%d  vb=%d  fi=%d"
        "  dot=%.5f  q=%.5f  cost=%.6g  midpoint=%s\n",
        collapse_num,
        is_flip ? "face_flip" : "quality",
        edge, va, vb, reject_fi,
        dot, q, cost,
        used_midpoint ? "yes" : "no");
    o << buf;

    app_vec3(o, "va_3d",  pos_va);
    app_vec3(o, "vb_3d",  pos_vb);
    app_vec3(o, "opt_3d", pos_opt);

    snprintf(buf, sizeof(buf), "  reject_face: v=(%d,%d,%d)\n", fv0, fv1, fv2);
    o << buf;
    app_vec3(o, "fv0_pre", fv0_3d);
    app_vec3(o, "fv1_pre", fv1_3d);
    app_vec3(o, "fv2_pre", fv2_3d);

    if (is_flip) {
        snprintf(buf, sizeof(buf),
            "  n_pre=(%.5g,%.5g,%.5g)  n_post=(%.5g,%.5g,%.5g)  dot=%.5f\n",
            n_pre(0), n_pre(1), n_pre(2),
            n_post(0), n_post(1), n_post(2),
            dot);
        o << buf;
    } else {
        snprintf(buf, sizeof(buf), "  quality=%.5f  (threshold=0.2)\n", q);
        o << buf;
    }

    o << "  one_ring_va[" << ring_va.size() << "]:";
    for (int fi : ring_va) o << " " << fi;
    o << "\n";
    o << "  one_ring_vb[" << ring_vb.size() << "]:";
    for (int fi : ring_vb) o << " " << fi;
    o << "\n";

    atomic_write(f, o.str());
}

// ── UvFlipRejDetail::write ────────────────────────────────────────────────────
void UvFlipRejDetail::write(FILE* f) const {
    if (!f) return;
    std::ostringstream o;
    char buf[512];

    snprintf(buf, sizeof(buf),
        "[UV-REJECT-DETAIL] collapse=#%d  reason=uv_face_flip"
        "  sid=%d  e=(%d,%d)  reject_fi=%d\n",
        collapse_num, sid, eu, ev, reject_fi);
    o << buf;

    snprintf(buf, sizeof(buf), "  uv_face_verts=(%d,%d,%d)\n", uv_a, uv_b, uv_c);
    o << buf;
    snprintf(buf, sizeof(buf),
        "  pre:  a=(%.5g,%.5g)  b=(%.5g,%.5g)  c=(%.5g,%.5g)  area=%.5g\n",
        uv_a_pre(0), uv_a_pre(1),
        uv_b_pre(0), uv_b_pre(1),
        uv_c_pre(0), uv_c_pre(1),
        signed_area_pre);
    o << buf;
    snprintf(buf, sizeof(buf),
        "  post: a=(%.5g,%.5g)  b=(%.5g,%.5g)  c=(%.5g,%.5g)  area=%.5g\n",
        uv_a_post(0), uv_a_post(1),
        uv_b_post(0), uv_b_post(1),
        uv_c_post(0), uv_c_post(1),
        signed_area_post);
    o << buf;

    o << "  post_uv_ring[" << ring_faces.size() << "]:\n";
    for (int i = 0; i < (int)ring_faces.size(); ++i) {
        const auto& vv = ring_faces[i];
        const auto& uv = ring_uvs_post[i];
        snprintf(buf, sizeof(buf),
            "    fi=%d  v=(%d,%d,%d)"
            "  uv0=(%.4g,%.4g)  uv1=(%.4g,%.4g)  uv2=(%.4g,%.4g)\n",
            i, vv[0], vv[1], vv[2],
            uv[0](0), uv[0](1),
            uv[1](0), uv[1](1),
            uv[2](0), uv[2](1));
        o << buf;
    }

    atomic_write(f, o.str());
}

// ── UvAngleRejDetail::write ───────────────────────────────────────────────────
void UvAngleRejDetail::write(FILE* f) const {
    if (!f) return;
    std::ostringstream o;
    char buf[512];

    snprintf(buf, sizeof(buf),
        "[UV-REJECT-DETAIL] collapse=#%d  reason=uv_angle_sum"
        "  sid=%d  e=(%d,%d)\n",
        collapse_num, sid, eu, ev);
    o << buf;

    snprintf(buf, sizeof(buf),
        "  bad_uv_v=%d  angle_sum=%.6f  diff_from_2pi=%.6f  degenerate=%s\n",
        bad_uv_v, angle_sum, diff_from_2pi,
        degenerate ? "yes" : "no");
    o << buf;

    snprintf(buf, sizeof(buf),
        "  n_interior=%d  nUV=%d  nFuv=%d  nfaces_for_bad_v=%d\n",
        n_interior, n_uv, n_fuv, n_faces_for_bad_v);
    o << buf;

    app_vec2(o, "bad_v_uv_post", bad_v_uv_post);

    o << "  faces_containing_bad_v[" << bad_v_faces.size() << "]:\n";
    for (int i = 0; i < (int)bad_v_faces.size(); ++i) {
        const auto& vv  = bad_v_faces[i];
        const auto& uvs = bad_v_face_uvs[i];
        snprintf(buf, sizeof(buf),
            "    fi=%d  v=(%d,%d,%d)"
            "  uv0=(%.4g,%.4g)  uv1=(%.4g,%.4g)  uv2=(%.4g,%.4g)\n",
            i, vv[0], vv[1], vv[2],
            uvs[0](0), uvs[0](1),
            uvs[1](0), uvs[1](1),
            uvs[2](0), uvs[2](1));
        o << buf;
    }

    atomic_write(f, o.str());
}
