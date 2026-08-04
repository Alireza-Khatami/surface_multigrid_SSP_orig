// MeshLab-style QEM callbacks for SSP_collapse_edge.
//
// Ports from QEM_meshlab/VcgQuadricSimplifier.{h,cpp}:
//   VcgQuadric::ByPlane / Apply / operator+= / *=
//   InitQuadrics     — area-weighted face quadrics + boundary quadrics + ScaleIndependent
//   ComputePosition  — SVD-truncation solver (QuadricMinimumClosestToPoint) or fullPivLu
//   ComputePriority  — soft quality/normal modifiers + hard veto gates
//   IsFeasible       — simplified link-condition (preserve_topology gate)
//
// No dependency on SlabMesh, Wm4, or QMAT.

#include "meshlab_qslim_callbacks.h"
#include <min_heap.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <set>
#include <cmath>
#include <limits>
#include <iostream>

using namespace Eigen;

// ── VcgQuadric (Eigen port of vcg::math::Quadric<double>) ───────────────────
// f(x) = x^T A x + b·x + c
// A symmetric 3×3 stored as upper triangle: a[0..5] = A00,A01,A02,A11,A12,A22
struct MeshlabQ
{
    double a[6], b[3], c;

    MeshlabQ() : c(-1.0) { a[0]=a[1]=a[2]=a[3]=a[4]=a[5]=0; b[0]=b[1]=b[2]=0; }

    void SetZero()
    {
        a[0]=a[1]=a[2]=a[3]=a[4]=a[5]=0;
        b[0]=b[1]=b[2]=0;
        c = 0.0;
    }

    void ByPlane(const Vector3d & n, double off)
    {
        a[0]=n.x()*n.x(); a[1]=n.y()*n.x(); a[2]=n.z()*n.x();
        a[3]=n.y()*n.y(); a[4]=n.z()*n.y(); a[5]=n.z()*n.z();
        b[0]=-2.0*off*n.x(); b[1]=-2.0*off*n.y(); b[2]=-2.0*off*n.z();
        c = off*off;
    }

    void operator+=(const MeshlabQ & q)
    {
        for (int i=0;i<6;++i) a[i]+=q.a[i];
        b[0]+=q.b[0]; b[1]+=q.b[1]; b[2]+=q.b[2];
        c+=q.c;
    }

    void operator*=(double w)
    {
        for (int i=0;i<6;++i) a[i]*=w;
        b[0]*=w; b[1]*=w; b[2]*=w; c*=w;
    }

    double Apply(const Vector3d & p) const
    {
        return  p.x()*p.x()*a[0]
            +2.0*p.x()*p.y()*a[1]
            +2.0*p.x()*p.z()*a[2]
            +    p.x()*b[0]
            +    p.y()*p.y()*a[3]
            +2.0*p.y()*p.z()*a[4]
            +    p.y()*b[1]
            +    p.z()*p.z()*a[5]
            +    p.z()*b[2]
            +    c;
    }

    MeshlabQ operator+(const MeshlabQ & q) const { MeshlabQ r=*this; r+=q; return r; }

    Matrix3d mat_A() const
    {
        Matrix3d A;
        A << a[0],a[1],a[2], a[1],a[3],a[4], a[2],a[4],a[5];
        return A;
    }

    Vector3d rhs() const   // -b/2  (gradient=0 gives A*x = -b/2)
    { return Vector3d(-b[0]*0.5, -b[1]*0.5, -b[2]*0.5); }
};

// ── Module-level state ────────────────────────────────────────────────────────
static std::vector<MeshlabQ> gMLQ;
static double gMLScaleFactor = 1.0;

#ifdef ML_QEM_LOG
static bool gMLCostLog   = false;
static int gMLRej_nanInf    = 0;
static int gMLRej_hardFlip  = 0;
static int gMLRej_hardQual  = 0;
static int gMLRej_areaRatio = 0;
static int gMLRej_softNanInf = 0;
static int gMLRej_linkCond  = 0;
#endif

void meshlab_enable_cost_logging() {
#ifdef ML_QEM_LOG
    gMLCostLog = true;
#endif
}

// ── Solvers ───────────────────────────────────────────────────────────────────

// SVD: QuadricMinimumClosestToPoint (VcgQuadricSimplifier.cpp:81)
// Truncates singular values where s[i]/s[0] <= 1e-3 and pins those directions
// toward `pt` (midpoint). Always bounded — cannot fly outside the bounding box.
static Vector3d quadric_minimum_svd(const MeshlabQ & q, const Vector3d & pt)
{
    Matrix3d A = q.mat_A();
    Vector3d r = q.rhs();

    JacobiSVD<Matrix3d> svd(A, ComputeFullU | ComputeFullV);
    Vector3d s = svd.singularValues();
    const double qeps = 1e-3;
    const double s0   = s[0];
    s[0] = (s0 > 1e-30) ? 1.0/s0 : 0.0;
    for (int i = 1; i < 3; ++i)
        s[i] = (s0 > 1e-30 && s[i]/s0 > qeps) ? 1.0/s[i] : 0.0;

    // x = pt + V * S^+ * U^T * (rhs - A*pt)
    return pt + svd.matrixV() * s.asDiagonal() * svd.matrixU().transpose() * (r - A*pt);
}

// fullPivLu: faithful vcg QuadricMinimum — can fly off on ill-conditioned A.
static bool quadric_minimum_lu(const MeshlabQ & q, const Vector3d & mid, Vector3d & x)
{
    auto lu = q.mat_A().fullPivLu();
    if (!lu.isInvertible()) { x = mid; return false; }
    x = lu.solve(q.rhs());
    return true;
}

// Dispatch based on cfg.svd_placement.
static Vector3d compute_position(const MeshlabQ & q, const Vector3d & mid,
                                 bool use_svd, bool optimal)
{
    if (!optimal) return mid;
    if (q.Apply(mid) <= 0.0) return mid;
    if (use_svd) return quadric_minimum_svd(q, mid);
    Vector3d x;
    quadric_minimum_lu(q, mid, x);
    return x;
}

// ── Face helpers ──────────────────────────────────────────────────────────────

static bool face_deleted(const MatrixXi & F, int f)
{
    return F(f,0) == 0 && F(f,1) == 0 && F(f,2) == 0;
}

// 2*sqrt(3)*area / (sum of squared edge lengths)  → in [0,1], 1=equilateral.
static double face_quality(const Vector3d & p0, const Vector3d & p1, const Vector3d & p2)
{
    double a2 = (p1-p0).squaredNorm();
    double b2 = (p2-p1).squaredNorm();
    double c2 = (p0-p2).squaredNorm();
    double denom = std::powf(a2, 2.0f) + std::powf(b2, 2.0f) + std::powf(c2, 2.0f);
    if (denom < 1e-30) return 0.0;
    return 4.0*std::sqrt(3.0) * (p1-p0).cross(p2-p0).norm() / denom;
}

static Vector3d face_normal(const Vector3d & p0, const Vector3d & p1, const Vector3d & p2)
{
    Vector3d n = (p1-p0).cross(p2-p0);
    double len = n.norm();
    if (len > 1e-30) return Vector3d(n / len);
    return Vector3d::Zero();
}

static bool is_inf_pos(const Vector3d & p) { return std::isinf(p.x()); }

// ── One-ring walker ───────────────────────────────────────────────────────────
// Calls visitor(p0,p1,p2, orig_p0,orig_p1,orig_p2) for every surviving face
// in the one-ring of va and vb, with optPos substituted for the moved vertex.
//
// Surviving faces:
//   - faces incident to va but NOT vb  (va→optPos)
//   - faces incident to vb but NOT va  (vb→optPos, since they redirect to va)
//   - shared faces (va AND vb) are skipped — they collapse
template<typename Visitor>
static void walk_surviving_faces(
    int va, int vb, const Vector3d & optPos,
    const MatrixXd & V, const MatrixXi & F,
    const std::vector<std::vector<int>> & VF,
    Visitor && vis)
{
    auto get_pos = [&](int v, int moved) -> Vector3d {
        if (v == moved) return optPos;
        return V.row(v).transpose();
    };

    for (int side = 0; side < 2; ++side) {
        int mv  = (side == 0) ? va : vb;   // the vertex being moved/absorbed
        int opp = (side == 0) ? vb : va;   // the other endpoint (shared faces use this)
        const auto & flist = VF[mv];
        for (int f : flist) {
            if (f < 0 || f >= F.rows()) continue;
            if (face_deleted(F, f)) continue;
            int fv0=F(f,0), fv1=F(f,1), fv2=F(f,2);
            // skip shared (collapsed) faces
            if (fv0==opp || fv1==opp || fv2==opp) continue;
            Vector3d q0 = get_pos(fv0, mv);
            Vector3d q1 = get_pos(fv1, mv);
            Vector3d q2 = get_pos(fv2, mv);
            if (is_inf_pos(q0)||is_inf_pos(q1)||is_inf_pos(q2)) continue;
            // original (pre-collapse) positions
            Vector3d o0 = V.row(fv0).transpose();
            Vector3d o1 = V.row(fv1).transpose();
            Vector3d o2 = V.row(fv2).transpose();
            if (is_inf_pos(o0)||is_inf_pos(o1)||is_inf_pos(o2)) continue;
            vis(q0,q1,q2, o0,o1,o2);
        }
    }
}

// ── InitQuadrics ──────────────────────────────────────────────────────────────
static void ml_init_quadrics(
    const MatrixXd & V, const MatrixXi & F,
    const MatrixXi & E, const MatrixXi & EF,
    const MeshlabQEMConfig & cfg)
{
    const int nV = (int)V.rows();
    const int nF = (int)F.rows();
    const int nE = (int)E.rows();

    gMLQ.assign(nV, MeshlabQ());
    for (auto & q : gMLQ) q.SetZero();

    auto is_inf_v     = [&](int v){ return v>=0&&v<nV&&std::isinf(V(v,0)); };
    auto is_real_face = [&](int f)->bool {
        if (f<0||f>=nF) return false;
        for (int c=0;c<3;++c) if (is_inf_v(F(f,c))) return false;
        return true;
    };

    // Step 1: area-weighted face-plane quadrics
    for (int f = 0; f < nF; ++f)
    {
        int v0=F(f,0), v1=F(f,1), v2=F(f,2);
        if (is_inf_v(v0)||is_inf_v(v1)||is_inf_v(v2)) continue;
        Vector3d p0=V.row(v0), p1=V.row(v1), p2=V.row(v2);
        Vector3d cross = (p1-p0).cross(p2-p0);
        double area = cross.norm();
        if (area < 1e-30) continue;
        Vector3d nrm = cross/area;
        double off = nrm.dot(p0);
        MeshlabQ q; q.SetZero();
        q.ByPlane(nrm, off);
        if (cfg.use_area) q *= area;
        gMLQ[v0]+=q; gMLQ[v1]+=q; gMLQ[v2]+=q;
    }

    // Step 2: boundary quadrics
    for (int e = 0; e < nE; ++e)
    {
        int ea=E(e,0), eb=E(e,1);
        if (is_inf_v(ea)||is_inf_v(eb)) continue;
        bool f0r=is_real_face(EF(e,0)), f1r=is_real_face(EF(e,1));
        if (f0r==f1r) continue;
        int realF = f0r ? EF(e,0) : EF(e,1);
        Vector3d fp0=V.row(F(realF,0)), fp1=V.row(F(realF,1)), fp2=V.row(F(realF,2));
        Vector3d fcross=(fp1-fp0).cross(fp2-fp0);
        double farea=fcross.norm();
        if (farea<1e-30) continue;
        Vector3d faceNrm=fcross/farea;
        Vector3d ev=V.row(eb)-V.row(ea);
        double elen=ev.norm();
        if (elen<1e-30) continue;
        Vector3d edgeDir=ev/elen;
        Vector3d borderDir = faceNrm.cross(edgeDir) * cfg.boundary_quadric_weight;
        double boff = borderDir.dot(Vector3d(V.row(ea)));
        MeshlabQ bq; bq.SetZero();
        bq.ByPlane(borderDir, boff);
        gMLQ[ea]+=bq; gMLQ[eb]+=bq;
    }

    // Step 3: ScaleIndependent
    if (cfg.scale_independent) {
        Vector3d mn( 1e300, 1e300, 1e300), mx(-1e300,-1e300,-1e300);
        bool any=false;
        for (int v=0;v<nV;++v) {
            if (is_inf_v(v)) continue;
            for (int d=0;d<3;++d){mn(d)=std::min(mn(d),V(v,d));mx(d)=std::max(mx(d),V(v,d));}
            any=true;
        }
        double diag = any ? (mx-mn).norm() : 0.0;
        gMLScaleFactor = (diag>1e-30) ? 1e8*std::pow(1.0/diag,6.0) : 1.0;
        std::cout << "[meshlab_qslim] diag=" << diag
                  << "  ScaleFactor=" << gMLScaleFactor << "\n";
    } else {
        gMLScaleFactor = 1.0;
    }
}

// ── Link condition (simplified Dey-Edelsbrunner) ──────────────────────────────
// Returns true when collapse (va→vb) satisfies the link condition.
static bool link_condition(
    int va, int vb,
    const MatrixXi & F,
    const std::vector<std::vector<int>> & VF)
{
    // Collect the one-ring neighbours of each endpoint
    auto neighbours = [&](int v) {
        std::set<int> N;
        for (int f : VF[v]) {
            if (face_deleted(F,f)) continue;
            for (int c=0;c<3;++c) { int u=F(f,c); if(u!=v) N.insert(u); }
        }
        return N;
    };
    std::set<int> Na = neighbours(va);
    std::set<int> Nb = neighbours(vb);

    // Shared neighbours (excluding the two endpoints themselves)
    std::set<int> shared;
    for (int v : Na) if (v!=vb && Nb.count(v)) shared.insert(v);

    // Shared faces (faces containing both va and vb)
    int shared_faces = 0;
    for (int f : VF[va]) {
        if (face_deleted(F,f)) continue;
        for (int c=0;c<3;++c) if (F(f,c)==vb) { ++shared_faces; break; }
    }

    // Link condition: #(shared_neighbours) == #(shared_faces)
    return (int)shared.size() == shared_faces;
}

// ── Public entry point ────────────────────────────────────────────────────────
void meshlab_setup_callbacks(
    const MatrixXd                      & V,
    const MatrixXi                      & F,
    const MatrixXi                      & E,
    const MatrixXi                      & EF,
    const std::vector<std::vector<int>> & VF,
    int                                 & v1,
    int                                 & v2,
    const MeshlabQEMConfig              & cfg,
    decimate_cost_and_placement_func    & cost_fn,
    decimate_pre_collapse_func          & pre_fn,
    decimate_post_collapse_func         & post_fn)
{
    ml_init_quadrics(V, F, E, EF, cfg);
    cfg.print();

    const double kInf = std::numeric_limits<double>::infinity();
    // Paper (Neural Subdivision, Hsueh-Ti et al. 2020): dot(n_pre, n_post) > δ=0.2
    // Was: std::cos(150° * π/180) ≈ -0.866 — too loose, allowed almost-complete flips.
    const double cos_hard_flip  = 0.2;
    // Not required by the paper (Neural Subdivision, Hsueh-Ti et al. 2020):
    // const double cos_normal_thr = std::cos(cfg.normal_thr_deg * M_PI / 180.0);

    // ── cost_fn ──────────────────────────────────────────────────────────────
    cost_fn = [&VF, &cfg, cos_hard_flip, kInf](
        const int        e,
        const MatrixXd & V,
        const MatrixXi & F,
        const MatrixXi & E,
        const VectorXi & /*EMAP*/,
        const MatrixXi & /*EF*/,
        const MatrixXi & /*EI*/,
        double         & cost,
        RowVectorXd    & p)
    {
        int va=E(e,0), vb=E(e,1);
        MeshlabQ q = gMLQ[va]; q += gMLQ[vb];

        Vector3d mid = (V.row(va).transpose() + V.row(vb).transpose()) * 0.5;
        Vector3d optPos = compute_position(q, mid, cfg.svd_placement, cfg.optimal_placement);

        // ── base QEM error ───────────────────────────────────────────────────
        double err = gMLScaleFactor * q.Apply(optPos);
        err = std::max(err, cfg.quadric_epsilon);
        if (err <= cfg.quadric_epsilon)
            err *= (V.row(va) - V.row(vb)).norm();

        if (std::isinf(err) || err != err) {
#ifdef ML_QEM_LOG
            if (gMLCostLog && ++gMLRej_nanInf <= 5)
                fprintf(stderr, "[ML-COST-REJECT] nan_inf_error  e=%d  va=%d vb=%d  err=%g\n",
                        e, va, vb, err);
#endif
            cost=kInf; p=RowVectorXd::Zero(3); return;
        }

        // ── Paper §4 one-ring scan (Neural Subdivision, Hsueh-Ti et al. 2020) ─
        // Active checks: hard face flip (δ=0.2) + hard quality (Q>0.2).
        // Not required by the paper — commented out:
        //   quality_check (soft cost modifier), normal_check (soft cost modifier),
        //   area_check (area ratio gate), minPreQual * 0.9 pre-quality comparison.
        bool need_quality = cfg.hard_quality_check;
        bool need_normal  = cfg.hard_normal_check;
        // bool need_area = cfg.area_check;  // not required by the paper

        double minPostQual = 1.0;
        bool   hard_flip   = false;

        // Locals for MSVC nested-lambda capture
        const bool   l_need_quality = need_quality;
        const bool   l_need_normal  = need_normal;
        const bool   l_hard_normal  = cfg.hard_normal_check;
        const double l_cos_flip     = cos_hard_flip;  // = 0.2 (paper's δ)

        if (need_quality || need_normal)
        {
            walk_surviving_faces(va, vb, optPos, V, F, VF,
                [&](const Vector3d & q0, const Vector3d & q1, const Vector3d & q2,
                    const Vector3d & o0, const Vector3d & o1, const Vector3d & o2)
            {
                // Paper: Q > 0.2 for all neighboring triangles (Euclidean domain).
                if (l_need_quality)
                    minPostQual = std::min(minPostQual, face_quality(q0,q1,q2));

                // Paper: dot(n_pre, n_post) > δ=0.2.
                // Removed: || face_quality(q0,q1,q2) < 0.01 — not required by the paper.
                if (l_need_normal) {
                    Vector3d nPre  = face_normal(o0,o1,o2);
                    Vector3d nPost = face_normal(q0,q1,q2);
                    if (l_hard_normal && nPre.dot(nPost) <= l_cos_flip)
                        hard_flip = true;
                }
            });

            if (hard_flip) {
#ifdef ML_QEM_LOG
                if (gMLCostLog && ++gMLRej_hardFlip <= 5)
                    fprintf(stderr, "[ML-COST-REJECT] hard_flip  e=%d  va=%d vb=%d  delta=0.2\n",
                            e, va, vb);
#endif
                cost = kInf; p = RowVectorXd::Zero(3); return;
            }
        }

        // ── hard quality gate (paper: Q > 0.2) ──────────────────────────────
        // Removed: minPostQual < minPreQual * 0.9 condition — not required by the paper.
        if (cfg.hard_quality_check && minPostQual < cfg.hard_quality_thr)
        {
#ifdef ML_QEM_LOG
            if (gMLCostLog && ++gMLRej_hardQual <= 5)
                fprintf(stderr, "[ML-COST-REJECT] hard_quality  e=%d  va=%d vb=%d"
                        "  postQual=%g  thr=%g\n",
                        e, va, vb, minPostQual, cfg.hard_quality_thr);
#endif
            cost = kInf; p = RowVectorXd::Zero(3); return;
        }

        // ── hard area gate ─ not required by the paper ───────────────────────
        // Not required by the paper (Neural Subdivision, Hsueh-Ti et al. 2020).
        // if (cfg.area_check && (preArea + postArea) > 1e-30 &&
        //     std::abs(preArea - postArea) / (preArea + postArea) > 0.01)
        // { cost = kInf; p = RowVectorXd::Zero(3); return; }

        // ── soft quality modifier ─ not required by the paper ────────────────
        // Not required by the paper (Neural Subdivision, Hsueh-Ti et al. 2020).
        // Paper uses a hard veto on Q, not a soft cost penalty.
        // if (cfg.quality_check) { qualFactor = 1.0 / max(minPostQual, cfg.quality_thr); }

        // ── soft normal modifier ─ not required by the paper ─────────────────
        // Not required by the paper (Neural Subdivision, Hsueh-Ti et al. 2020).
        // if (cfg.normal_check) { normFactor = ...; }

        cost = err;  // no soft modifiers; pure QEM error
        p = optPos.transpose().eval();
    };

    // ── pre_fn ───────────────────────────────────────────────────────────────
    pre_fn = [&v1, &v2, &VF, &cfg](
        const MatrixXd &, const MatrixXi & F,
        const MatrixXi & E,
        const VectorXi &, const MatrixXi &, const MatrixXi &,
        const min_heap<std::tuple<double,int,int>> &,
        const VectorXi &, const MatrixXd &,
        const int e) -> bool
    {
        v1 = E(e,0); v2 = E(e,1);
        if (cfg.preserve_topology && !link_condition(v1, v2, F, VF)) {
#ifdef ML_QEM_LOG
            if (++gMLRej_linkCond <= 10)
                fprintf(stderr, "[ML-REJECT] link_condition  e=%d  va=%d vb=%d\n",
                        e, v1, v2);
#endif
            return false;
        }
        return true;
    };

    // ── post_fn ──────────────────────────────────────────────────────────────
    post_fn = [&v1, &v2](
        const MatrixXd &, const MatrixXi &, const MatrixXi &,
        const VectorXi &, const MatrixXi &, const MatrixXi &,
        const min_heap<std::tuple<double,int,int>> &,
        const VectorXi &, const MatrixXd &,
        const int, const int, const int, const int, const int,
        const bool collapsed)
    {
        if (collapsed)
            gMLQ[v1<v2?v1:v2] = gMLQ[v1] + gMLQ[v2];
    };
}
