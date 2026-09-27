#include "debug_trackers.h"
#include "subdiv_tracker.h"
#include "../face_dead.h"

#include <Eigen/Dense>
#include <cstdio>
#include <unordered_set>

using namespace Eigen;

// ---- globals from main.cpp ----
extern MatrixXd gVO;
extern MatrixXi gF;
extern MatrixXi gFO;
extern int gCollapseCount;

// ---- face flip tracker ----

static int      gFFT_faceIdx        = -1;
static int      gFFT_vtx[3]         = {-1, -1, -1};  // subdivided (= fine) vertex ids; -1 untracked
static Vector3d gFFT_origNormal;
static Vector3d gFFT_posBefore[3];
static std::vector<Vector3d> gFFT_traj[3];
static bool     gFFT_flipDetected   = false;
static int      gFFT_flipAtCollapse = -1;

void face_flip_tracker_init(int face_idx)
{
    gFFT_faceIdx        = -1;
    gFFT_flipDetected   = false;
    gFFT_flipAtCollapse = -1;
    for (int i = 0; i < 3; i++) { gFFT_vtx[i] = -1; gFFT_traj[i].clear(); }

    if (!subdiv_tracker_enabled()) {
        fprintf(stderr, "[face_flip_tracker] needs the subdivided tracker (--n_subdiv_samples)\n");
        return;
    }
    if (face_idx < 0 || face_idx >= gFO.rows()) {
        fprintf(stderr, "[face_flip_tracker] face %d out of range\n", face_idx);
        return;
    }
    gFFT_faceIdx = face_idx;

    const int verts[3] = { gFO(face_idx, 0), gFO(face_idx, 1), gFO(face_idx, 2) };
    for (int i = 0; i < 3; i++)
        if (subdiv_tracker_is_tracked(verts[i])) gFFT_vtx[i] = verts[i];

    const Vector3d p0 = gVO.row(verts[0]).transpose();
    const Vector3d p1 = gVO.row(verts[1]).transpose();
    const Vector3d p2 = gVO.row(verts[2]).transpose();
    gFFT_origNormal = (p1 - p0).cross(p2 - p0).normalized();

    for (int i = 0; i < 3; i++)
        gFFT_traj[i].push_back(gVO.row(verts[i]).transpose());

    fprintf(stderr, "[face_flip_tracker] tracking face %d  verts=(%d,%d,%d)  tracked=(%d,%d,%d)\n",
            face_idx, verts[0], verts[1], verts[2],
            gFFT_vtx[0] >= 0, gFFT_vtx[1] >= 0, gFFT_vtx[2] >= 0);
}

void face_flip_tracker_pre_update()
{
    if (gFFT_faceIdx < 0) return;
    for (int i = 0; i < 3; i++)
        if (gFFT_vtx[i] >= 0) gFFT_posBefore[i] = subdiv_tracker_cur_pos(gFFT_vtx[i]);
}

void face_flip_tracker_post_update()
{
    if (gFFT_faceIdx < 0 || gFFT_flipDetected) return;

    Vector3d posAfter[3];
    bool any_moved = false;
    for (int i = 0; i < 3; i++) {
        if (gFFT_vtx[i] < 0) { posAfter[i] = gFFT_posBefore[i]; continue; }
        posAfter[i] = subdiv_tracker_cur_pos(gFFT_vtx[i]);
        if ((posAfter[i] - gFFT_posBefore[i]).norm() > 1e-15) {
            any_moved = true;
            gFFT_traj[i].push_back(posAfter[i]);
        }
    }
    if (!any_moved) return;

    const Vector3d curNormal = (posAfter[1] - posAfter[0]).cross(posAfter[2] - posAfter[0]);
    if (curNormal.dot(gFFT_origNormal) < 0.0) {
        gFFT_flipDetected   = true;
        gFFT_flipAtCollapse = gCollapseCount;
        fprintf(stderr, "[face_flip_tracker] *** FLIP DETECTED at collapse #%d — original face %d ***\n",
                gCollapseCount, gFFT_faceIdx);
    }
}

bool face_flip_tracker_enabled()          { return gFFT_faceIdx >= 0; }
bool face_flip_tracker_flip_detected()    { return gFFT_flipDetected; }
int  face_flip_tracker_flip_at_collapse() { return gFFT_flipAtCollapse; }
int  face_flip_tracker_face_idx()         { return gFFT_faceIdx; }

Vector3d face_flip_tracker_cur_pos(int i)
{
    if (i < 0 || i > 2 || gFFT_vtx[i] < 0) return Vector3d::Zero();
    return subdiv_tracker_cur_pos(gFFT_vtx[i]);
}

const std::vector<Vector3d> & face_flip_tracker_traj(int i)
{
    static const std::vector<Vector3d> empty;
    if (i < 0 || i > 2) return empty;
    return gFFT_traj[i];
}

// ---- vertex watch tracker ----

static int      gVWT_fineVtxId = -1;
static Vector3i gVWT_snapBF(-1, -1, -1);  // corners snapshotted before the collapse attempts
static bool     gVWT_triggered = false;
static int      gVWT_triggerAt = -1;

void vertex_watch_set(int fine_vtx_id)
{
    vertex_watch_clear();
    if (fine_vtx_id < 0 || fine_vtx_id >= (int)gVO.rows()) {
        fprintf(stderr, "[vertex_watch] vertex %d out of range (gVO has %d rows)\n",
                fine_vtx_id, (int)gVO.rows());
        return;
    }
    if (!subdiv_tracker_is_tracked(fine_vtx_id)) {
        fprintf(stderr, "[vertex_watch] fine vertex %d is not tracked (needs --n_subdiv_samples, "
                "and the vertex must be on a face)\n", fine_vtx_id);
        return;
    }
    gVWT_fineVtxId = fine_vtx_id;
    fprintf(stderr, "[vertex_watch] watching fine vertex %d\n", fine_vtx_id);
}

void vertex_watch_clear()
{
    gVWT_fineVtxId = -1;
    gVWT_triggered = false;
    gVWT_triggerAt = -1;
    gVWT_snapBF    = Vector3i(-1, -1, -1);
}

void vertex_watch_pre_step()
{
    if (gVWT_fineVtxId < 0) return;
    gVWT_snapBF = subdiv_tracker_cur_corners(gVWT_fineVtxId);
}

void vertex_watch_check_collapse(int s_vtx, int d_vtx)
{
    if (gVWT_fineVtxId < 0 || gVWT_triggered) return;

    // The watched vertex lives on the coarse face with corners snapBF. Trigger if
    // s or d shares a live coarse face with one of those corners.
    const std::unordered_set<int> watched = { gVWT_snapBF(0), gVWT_snapBF(1), gVWT_snapBF(2) };

    for (int fi = 0; fi < gF.rows(); fi++) {
        if (is_face_dead(gF, fi)) continue;
        bool has_collapse_vtx = false, has_watched_vtx = false;
        int  matching_collapse = -1, matching_watched = -1;
        for (int k = 0; k < 3; k++) {
            const int v = gF(fi, k);
            if (v == s_vtx || v == d_vtx) { has_collapse_vtx = true; matching_collapse = v; }
            if (watched.count(v))          { has_watched_vtx  = true; matching_watched  = v; }
        }
        if (has_collapse_vtx && has_watched_vtx) {
            gVWT_triggered = true;
            gVWT_triggerAt = gCollapseCount;
            fprintf(stderr,
                "[vertex_watch] *** TRIGGERED at collapse #%d ***"
                "  fine_vtx=%d  via coarse face %d"
                "  collapse_vtx=%d(%s)  watched_corner=%d\n",
                gCollapseCount, gVWT_fineVtxId, fi,
                matching_collapse, (matching_collapse == s_vtx) ? "survivor" : "absorbed",
                matching_watched);
            return;
        }
    }
}

bool vertex_watch_active()              { return gVWT_fineVtxId >= 0; }
bool vertex_watch_triggered()           { return gVWT_triggered; }
int  vertex_watch_trigger_at_collapse() { return gVWT_triggerAt; }
int  vertex_watch_fine_vtx()            { return gVWT_fineVtxId; }

Vector3i vertex_watch_cur_BF()
{
    if (gVWT_fineVtxId < 0) return Vector3i(-1, -1, -1);
    return subdiv_tracker_cur_corners(gVWT_fineVtxId);
}
