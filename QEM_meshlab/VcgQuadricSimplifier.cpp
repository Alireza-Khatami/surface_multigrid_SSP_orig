#include "VcgQuadricSimplifier.h"

#if defined(ONLY_USE_QEM_CONDITION_CHECKS)

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <Eigen/Dense>

// ─────────────────────────────────────────────────────────────────────────────
// Local geometry helpers — faithful ports of the vcg free functions used by the
// quadric collapse (vcglib/triangle3.h).
// ─────────────────────────────────────────────────────────────────────────────
namespace {

// vcg Quality(p0,p1,p2) == QualityFace = 2*Area / max(edge^2).  Range [0,0.866].
double QualityFace(const Wm4::Vector3d& p0,
                   const Wm4::Vector3d& p1,
                   const Wm4::Vector3d& p2)
{
	Wm4::Vector3d d10 = p1 - p0;
	Wm4::Vector3d d20 = p2 - p0;
	Wm4::Vector3d d12 = p1 - p2;
	double a = d10.Cross(d20).Length();   // 2*area
	if (a == 0.0) return 0.0;
	double b = d10.SquaredLength();
	if (b == 0.0) return 0.0;
	double t = d20.SquaredLength(); if (b < t) b = t;
	t = d12.SquaredLength();        if (b < t) b = t;
	return a / b;
}

// vcg NormalizedTriangleNormal — (p1-p0) x (p2-p0), normalised (zero on degenerate).
Wm4::Vector3d NormalizedTriangleNormal(const Wm4::Vector3d& p0,
                                       const Wm4::Vector3d& p1,
                                       const Wm4::Vector3d& p2)
{
	Wm4::Vector3d n = (p1 - p0).Cross(p2 - p0);
	double len = n.Length();
	if (len < 1e-30) return Wm4::Vector3d(0.0, 0.0, 0.0);
	return n / len;
}

// vcg math::Quadric<double>::Minimum — solve 2Ax+b=0 (i.e. Ax = -b/2) with
// Eigen fullPivLu and the same relative-error acceptance test (RelativeErrorThr
// = 1e-6).  Returns false (caller falls back to the midpoint) if the solution
// does not fit the system.
bool QuadricMinimum(const VcgQuadric& q, Wm4::Vector3d& x)
{
	Eigen::Matrix3d A;
	Eigen::Vector3d be;
	A << q.a[0], q.a[1], q.a[2],
	     q.a[1], q.a[3], q.a[4],
	     q.a[2], q.a[4], q.a[5];
	be << -q.b[0] / 2.0, -q.b[1] / 2.0, -q.b[2] / 2.0;

	Eigen::Vector3d xe = A.fullPivLu().solve(be);
	// vcg assigns the fullPivLu solution and (in ComputePosition) uses it whether
	// or not the fit check passes — tri_edge_collapse_quadric.h:176 ignores the
	// return value of Minimum(x).  So write x unconditionally; the bool only
	// reports whether the solution fits (be.norm()*RelativeErrorThr, 1e-6).
	x = Wm4::Vector3d(xe[0], xe[1], xe[2]);
	double error = (A * xe - be).norm();
	return error <= be.norm() * 1e-6;
}

// vcg math::Quadric<double>::MinimumClosestToPoint (quadric.h:221) — SVD solve
// that truncates singular values with s[i]/s[0] <= qeps (1e-3) and pins those
// directions toward pt. Always succeeds (bounded). For full-rank quadrics it
// equals Minimum; for ill-conditioned ones it stays near pt (the midpoint).
Wm4::Vector3d QuadricMinimumClosestToPoint(const VcgQuadric& q, const Wm4::Vector3d& pt)
{
	Eigen::Matrix3d A;
	Eigen::Vector3d be;
	A << q.a[0], q.a[1], q.a[2],
	     q.a[1], q.a[3], q.a[4],
	     q.a[2], q.a[4], q.a[5];
	be << -q.b[0] / 2.0, -q.b[1] / 2.0, -q.b[2] / 2.0;

	Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Vector3d s = svd.singularValues();
	const double qeps = 1e-3;
	for (int i = 1; i < 3; ++i) s[i] = (s[i] / s[0] > qeps) ? 1.0 / s[i] : 0.0;
	s[0] = 1.0 / s[0];

	Eigen::Vector3d xp(pt.X(), pt.Y(), pt.Z());
	Eigen::Vector3d xe = xp + (svd.matrixV() * s.asDiagonal() * svd.matrixU().transpose()) * (be - A * xp);
	return Wm4::Vector3d(xe[0], xe[1], xe[2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// FLOAT plane arithmetic, replicating vcg's Point3<float> operations.
//
// MeshLab (MESHLAB_SCALAR=float) builds each face/border plane in float — the
// cross product, normalization, area and offset are all float; only the resulting
// Quadric is double (its a/b/c are double PRODUCTS of the float direction/offset).
// On near-flat edges Apply is ~14 orders of catastrophic cancellation, and the
// float-level residue of the plane is exactly what keeps MeshLab's Apply above the
// QuadricEpsilon clamp; a double-built plane rounds Apply to ~0 and clamps,
// flipping the cheapest-edge order.  Each subexpression is cast to float so no
// double/extended precision leaks in (with /fp:precise the casts are honoured).
// Results are returned in a Wm4::Vector3d (double) holding the exact float values.
// ─────────────────────────────────────────────────────────────────────────────
inline Wm4::Vector3d FDiff(const Wm4::Vector3d& a, const Wm4::Vector3d& b)
{
	return Wm4::Vector3d((float)((float)a.X() - (float)b.X()),
	                     (float)((float)a.Y() - (float)b.Y()),
	                     (float)((float)a.Z() - (float)b.Z()));
}
inline Wm4::Vector3d FCross(const Wm4::Vector3d& a, const Wm4::Vector3d& b)
{
	float ax=(float)a.X(), ay=(float)a.Y(), az=(float)a.Z();
	float bx=(float)b.X(), by=(float)b.Y(), bz=(float)b.Z();
	return Wm4::Vector3d((float)((float)(ay*bz) - (float)(az*by)),
	                     (float)((float)(az*bx) - (float)(ax*bz)),
	                     (float)((float)(ax*by) - (float)(ay*bx)));
}
inline float FNorm(const Wm4::Vector3d& v)
{
	float x=(float)v.X(), y=(float)v.Y(), z=(float)v.Z();
	return (float)std::sqrt((float)((float)((float)(x*x) + (float)(y*y)) + (float)(z*z)));
}
inline Wm4::Vector3d FNormalize(const Wm4::Vector3d& v, float n)
{
	return Wm4::Vector3d((float)((float)v.X() / n),
	                     (float)((float)v.Y() / n),
	                     (float)((float)v.Z() / n));
}
inline float FDot(const Wm4::Vector3d& a, const Wm4::Vector3d& b)
{
	float r = (float)((float)a.X() * (float)b.X());
	r = (float)(r + (float)((float)a.Y() * (float)b.Y()));
	r = (float)(r + (float)((float)a.Z() * (float)b.Z()));
	return r;
}
inline Wm4::Vector3d FScale(const Wm4::Vector3d& v, float w)
{
	return Wm4::Vector3d((float)((float)v.X() * w),
	                     (float)((float)v.Y() * w),
	                     (float)((float)v.Z() * w));
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Small adjacency helpers.
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::EnsureSized()
{
	if (vq.size() < m.vertices.size())    vq.resize(m.vertices.size());
	if (imark.size() < m.vertices.size()) imark.resize(m.vertices.size(), 0);
}

bool VcgQuadricSimplifier::FaceVerts(unsigned fid, unsigned out[3]) const
{
	if (fid >= m.faces.size() || !m.faces[fid].first) return false;
	const auto& vs = m.faces[fid].second->vertices_;
	if (vs.size() != 3) return false;
	int i = 0;
	for (unsigned v : vs) out[i++] = v;   // sorted ascending (std::set)
	return true;
}

double VcgQuadricSimplifier::EdgeLen(unsigned v0, unsigned v1) const
{
	return (Pos(v0) - Pos(v1)).Length();
}

// ─────────────────────────────────────────────────────────────────────────────
// InitQuadric (tri_edge_collapse_quadric.h:556).
//   • Zero every per-vertex quadric.
//   • For each face: build the (area-weighted) plane quadric and add it to the
//     three corner vertices.  For each face edge that is a border (exactly one
//     incident face) — or always, if QualityQuadric — add the perpendicular
//     border quadric scaled by BoundaryQuadricWeight (resp. QualityQuadricWeight).
//   • ScaleIndependent: ScaleFactor = 1e8 * (1/bboxDiag)^6 over all vertices.
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::InitQuadrics()
{
	EnsureSized();

	for (size_t i = 0; i < m.vertices.size(); ++i)
		if (m.vertices[i].first)
			vq[i].SetZero();

	int borderEdgeCount = 0;   // diagnostic: # of detected border (1-face) edges

	for (size_t fid = 0; fid < m.faces.size(); ++fid)
	{
		unsigned fv[3];
		if (!FaceVerts(static_cast<unsigned>(fid), fv)) continue;
		if (!m.vertices[fv[0]].first || !m.vertices[fv[1]].first || !m.vertices[fv[2]].first)
			continue;

		const Wm4::Vector3d& v0 = Pos(fv[0]);
		const Wm4::Vector3d& v1 = Pos(fv[1]);
		const Wm4::Vector3d& v2 = Pos(fv[2]);

		// vcg (FLOAT): dirArea = (v1-v0) x (v2-v0); facePlane direction =
		//   normalize(dirArea); area = |dirArea| (== 2*triangle area); offset = dir·v0.
		// All computed in float to match MeshLab's Point3f plane (see FCross etc.).
		Wm4::Vector3d dirArea = FCross(FDiff(v1, v0), FDiff(v2, v0));
		float area = FNorm(dirArea);
		if (area < 1e-30f) continue;      // degenerate face contributes nothing
		Wm4::Vector3d nrm = FNormalize(dirArea, area);
		double off = (double)FDot(nrm, v0);

		VcgQuadric q;
		q.ByPlane(nrm, off);
		if (params.UseArea) q *= (double)area;

		for (int j = 0; j < 3; ++j)
			vq[fv[j]] += q;

		// Border / quality quadrics, one per face edge.
		for (int j = 0; j < 3; ++j)
		{
			unsigned a = fv[j];
			unsigned b = fv[(j + 1) % 3];

			// Border detection EXACTLY like vcg's UpdateFlags::FaceBorderFromVF:
			// an edge is a border iff incident to an ODD number of faces (its
			// parity toggle marks border for 1, 3, 5, ... faces — NOT just 1).
			// This is what makes vcg treat non-manifold edges (3 faces) as borders
			// and add their border quadrics; counting only ==1 left every 3-face
			// seam/junction edge unweighted, so it collapsed ~3000x too cheaply vs
			// MeshLab (see md_files/qem/test_border_rule.py).
			int incidentFaces = 0;
			if (a < m.vertices.size() && m.vertices[a].first)
				for (unsigned ifid : m.vertices[a].second->faces_)
				{
					if (ifid >= m.faces.size() || !m.faces[ifid].first) continue;
					const auto& ivs = m.faces[ifid].second->vertices_;
					if (ivs.find(b) != ivs.end()) ++incidentFaces;
				}
			bool isB = (incidentFaces % 2 == 1);   // odd => border (FaceBorderFromVF)
			if (isB) ++borderEdgeCount;

			if (!isB && !params.QualityQuadric) continue;

			// vcg (FLOAT): borderDir = faceNormal x normalize(edge); then amplify by
			// the weight (the *direction* is scaled, so the quadric scales by w²).
			Wm4::Vector3d edge = FDiff(Pos(b), Pos(a));
			float elen = FNorm(edge);
			if (elen < 1e-30f) continue;
			Wm4::Vector3d borderDir = FCross(nrm, FNormalize(edge, elen));
			float w = (float)(isB ? params.BoundaryQuadricWeight : params.QualityQuadricWeight);
			borderDir = FScale(borderDir, w);
			double boff = (double)FDot(borderDir, Pos(a));

			VcgQuadric bq;
			bq.ByPlane(borderDir, boff);
			vq[a] += bq;
			vq[b] += bq;
		}
	}

	// ScaleIndependent — bbox over all active vertices (vcg UpdateBounding::Box).
	if (params.ScaleIndependent)
	{
		Wm4::Vector3d mn( std::numeric_limits<double>::infinity(),
		                  std::numeric_limits<double>::infinity(),
		                  std::numeric_limits<double>::infinity());
		Wm4::Vector3d mx(-std::numeric_limits<double>::infinity(),
		                 -std::numeric_limits<double>::infinity(),
		                 -std::numeric_limits<double>::infinity());
		bool any = false;
		for (size_t i = 0; i < m.vertices.size(); ++i)
			if (m.vertices[i].first)
			{
				const Wm4::Vector3d& p = Pos(static_cast<unsigned>(i));
				mn.X() = std::min(mn.X(), p.X()); mn.Y() = std::min(mn.Y(), p.Y()); mn.Z() = std::min(mn.Z(), p.Z());
				mx.X() = std::max(mx.X(), p.X()); mx.Y() = std::max(mx.Y(), p.Y()); mx.Z() = std::max(mx.Z(), p.Z());
				any = true;
			}
		double diag = any ? (mx - mn).Length() : 0.0;
		if (diag > 1e-30) params.ScaleFactor = 1e8 * std::pow(1.0 / diag, 6.0);
		else              params.ScaleFactor = 1.0;
		std::cerr << "[VcgQuadricSimplifier] QEM bbox diag = " << diag
		          << "  (should match MeshLab's _mat_initial.off diagonal)\n";
	}

	std::cerr << "[VcgQuadricSimplifier] InitQuadrics: ScaleFactor=" << params.ScaleFactor
	          << "  BoundaryQuadricWeight=" << params.BoundaryQuadricWeight
	          << "  UseArea=" << params.UseArea
	          << "  borderEdges=" << borderEdgeCount << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputePosition (tri_edge_collapse_quadric.h:158).
//   midpoint, unless OptimalPlacement and the summed quadric is non-flat at the
//   midpoint (Apply(mid) > 2*QuadricEpsilon), in which case the quadric minimum
//   is used.  If the minimum solve does not fit the system, fall back to the
//   midpoint (vcg leaves the position unchanged; the midpoint is the safe value).
// ─────────────────────────────────────────────────────────────────────────────
Wm4::Vector3d VcgQuadricSimplifier::ComputePosition(unsigned v0, unsigned v1)
{
	Wm4::Vector3d mid = ToFloat((Pos(v0) + Pos(v1)) * 0.5);
	if (!params.OptimalPlacement) return Pos(v1);

	if ((vq[v0].Apply(mid) + vq[v1].Apply(mid)) > 2.0 * params.QuadricEpsilon)
	{
		VcgQuadric q = vq[v0];
		q += vq[v1];
		if (params.SVDPlacement) return ToFloat(QuadricMinimumClosestToPoint(q, mid));
		// vcg: q.Minimum(x); newPos = x;  — the return value is IGNORED
		// (tri_edge_collapse_quadric.h:176), so the fullPivLu solution is always
		// used here, exactly as MeshLab does with SVDPlacement=false.
		Wm4::Vector3d x;
		QuadricMinimum(q, x);
		return ToFloat(x);
	}
	return mid;
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputePriority (tri_edge_collapse_quadric.h:300).  Computes the heap error
// and the cached optimal position.  Reproduces all branches; with vcg/MeshLab
// defaults (QualityCheck on, everything else off) it reduces to
//   error = ScaleFactor * (q0+q1).Apply(optPos) / newQual.
// ─────────────────────────────────────────────────────────────────────────────
double VcgQuadricSimplifier::ComputePriority(unsigned v0, unsigned v1, Wm4::Vector3d& outPos,
                                             QemRejectionReason*  outReason,
                                             QemReasonPrimitives* outPrims)
{
	const Wm4::Vector3d oldP0 = Pos(v0);
	const Wm4::Vector3d oldP1 = Pos(v1);

	// ── Collect ORIGINAL normals / area / quality, exactly as vcg does. ──────
	std::vector<Wm4::Vector3d> origNormals;
	auto collectOrigNormals = [&](unsigned va, unsigned skip) {
		for (unsigned fid : m.vertices[va].second->faces_)
		{
			unsigned fv[3];
			if (!FaceVerts(fid, fv)) continue;
			if (fv[0] == skip || fv[1] == skip || fv[2] == skip) continue;  // skip faces with the partner
			origNormals.push_back(NormalizedTriangleNormal(Pos(fv[0]), Pos(fv[1]), Pos(fv[2])));
		}
	};
	if (params.NormalCheck) { collectOrigNormals(v0, v1); collectOrigNormals(v1, v0); }

	double origArea = 0.0;
	auto collectArea = [&](unsigned va, bool skipShared, unsigned partner) {
		for (unsigned fid : m.vertices[va].second->faces_)
		{
			unsigned fv[3];
			if (!FaceVerts(fid, fv)) continue;
			if (skipShared && (fv[0] == partner || fv[1] == partner || fv[2] == partner)) continue;
			origArea += (Pos(fv[1]) - Pos(fv[0])).Cross(Pos(fv[2]) - Pos(fv[0])).Length();  // DoubleArea
		}
	};
	if (params.AreaCheck) { collectArea(v0, false, v1); collectArea(v1, true, v0); }

	double origQual = std::numeric_limits<double>::max();
	auto collectOrigQual = [&](unsigned va, bool skipShared, unsigned partner) {
		for (unsigned fid : m.vertices[va].second->faces_)
		{
			unsigned fv[3];
			if (!FaceVerts(fid, fv)) continue;
			if (skipShared && (fv[0] == partner || fv[1] == partner || fv[2] == partner)) continue;
			origQual = std::min(origQual, QualityFace(Pos(fv[0]), Pos(fv[1]), Pos(fv[2])));
		}
	};
	if (params.HardQualityCheck) { collectOrigQual(v0, false, v1); collectOrigQual(v1, true, v0); }

	// ── Simulate the collapse: both endpoints move to the optimal position. ──
	outPos = ComputePosition(v0, v1);

	// Helper: quality / normal of a surviving face with one endpoint substituted.
	auto faceWithSubst = [&](unsigned fid, unsigned moved, Wm4::Vector3d p[3]) -> bool {
		unsigned fv[3];
		if (!FaceVerts(fid, fv)) return false;
		for (int k = 0; k < 3; ++k)
			p[k] = (fv[k] == moved) ? outPos : Pos(fv[k]);
		return true;
	};

	double MinCos = std::numeric_limits<double>::max();
	if (params.NormalCheck)
	{
		int i = 0;
		auto scanNormals = [&](unsigned va, unsigned skip) {
			for (unsigned fid : m.vertices[va].second->faces_)
			{
				unsigned fv[3];
				if (!FaceVerts(fid, fv)) continue;
				if (fv[0] == skip || fv[1] == skip || fv[2] == skip) continue;
				Wm4::Vector3d p[3];
				faceWithSubst(fid, va, p);
				Wm4::Vector3d nn = NormalizedTriangleNormal(p[0], p[1], p[2]);
				if (i < (int)origNormals.size())
					MinCos = std::min(MinCos, nn.Dot(origNormals[i++]));
			}
		};
		scanNormals(v0, v1);
		scanNormals(v1, v0);
	}

	double newQual = std::numeric_limits<double>::max();
	std::array<std::array<double,3>,3> worstQualTri{};
	bool haveWorstQualTri = false;
	if (params.QualityCheck)
	{
		auto scanQual = [&](unsigned va, unsigned skip) {
			for (unsigned fid : m.vertices[va].second->faces_)
			{
				unsigned fv[3];
				if (!FaceVerts(fid, fv)) continue;
				if (fv[0] == skip || fv[1] == skip || fv[2] == skip) continue;  // destroyed shared face
				Wm4::Vector3d p[3];
				faceWithSubst(fid, va, p);
				double q = QualityFace(p[0], p[1], p[2]);
				if (q < newQual) {
					newQual = q;
					worstQualTri = { { {p[0].X(), p[0].Y(), p[0].Z()},
					                   {p[1].X(), p[1].Y(), p[1].Z()},
					                   {p[2].X(), p[2].Y(), p[2].Z()} } };
					haveWorstQualTri = true;
				}
			}
		};
		scanQual(v0, v1);
		scanQual(v1, v0);
	}

	double newArea = 0.0;
	if (params.AreaCheck)
	{
		auto scanArea = [&](unsigned va, bool skipShared, unsigned partner) {
			for (unsigned fid : m.vertices[va].second->faces_)
			{
				unsigned fv[3];
				if (!FaceVerts(fid, fv)) continue;
				if (skipShared && (fv[0] == partner || fv[1] == partner || fv[2] == partner)) continue;
				Wm4::Vector3d p[3];
				faceWithSubst(fid, va, p);
				newArea += (p[1] - p[0]).Cross(p[2] - p[0]).Length();
			}
		};
		scanArea(v0, false, v1);
		scanArea(v1, true, v0);
	}

	// ── QuadErr = ScaleFactor * (q0+q1).Apply(optPos). ───────────────────────
	VcgQuadric qq = vq[v0];
	qq += vq[v1];
	double QuadErr = params.ScaleFactor * qq.Apply(outPos);

	if (newQual > params.QualityThr) newQual = params.QualityThr;

	if (params.NormalCheck)
	{
		if (MinCos > params.CosineThr) MinCos = params.CosineThr;
		MinCos = std::fabs((MinCos + 1.0) / 2.0);   // → [0,1], 0 = worst
	}

	QuadErr = std::max(QuadErr, params.QuadricEpsilon);
	if (QuadErr <= params.QuadricEpsilon)
		QuadErr *= EdgeLen(v0, v1);

	// UseVertexWeight: W==1 for all vertices in our port → no-op (kept faithful).
	if (params.UseVertexWeight) QuadErr *= 1.0;

	double error;
	if (!params.QualityCheck && !params.NormalCheck) error = QuadErr;
	else if ( params.QualityCheck && !params.NormalCheck) error = QuadErr / newQual;
	else if (!params.QualityCheck &&  params.NormalCheck) error = QuadErr / MinCos;
	else                                                  error = QuadErr / (newQual * MinCos);

	const double MAXVAL = std::numeric_limits<double>::max();
	const bool   capture = (outReason && outPrims);

	if (params.AreaCheck && (std::fabs(origArea - newArea) / (origArea + newArea) > 0.01))
		error = MAXVAL;

	if (params.HardQualityCheck && (newQual < params.HardQualityThr && newQual < origQual * 0.9))
	{
		if (!params.DiagnoseOnly) error = MAXVAL;   // DiagnoseOnly: colour but don't veto
		if (capture) {
			*outReason = QemRejectionReason::HardQualityCheckFailed;
			outPrims->vertices = { v0, v1 };
			outPrims->edges    = { { v0, v1 } };
			outPrims->targ_ver = { outPos.X(), outPos.Y(), outPos.Z() };
			if (haveWorstQualTri) outPrims->tris_after.push_back(worstQualTri);
			outPrims->metrics  = { {"quality (post-collapse)", newQual},
			                       {"hard quality threshold",  params.HardQualityThr},
			                       {"orig quality x0.9",       origQual * 0.9} };
			outPrims->message  =
				"Collapse would create a sliver triangle: the worst surviving face's "
				"quality drops to " + std::to_string(newQual) + ", below the hard threshold "
				+ std::to_string(params.HardQualityThr) + " and below 90% of the original "
				"worst quality (" + std::to_string(origQual * 0.9) + "). Yellow = the sliver.";
		}
	}

	if (params.HardNormalCheck && CheckForFlip(v0, v1, outPos))   // fast path (no capture)
	{
		if (!params.DiagnoseOnly) error = MAXVAL;   // DiagnoseOnly: colour but don't veto
		{
			if (capture) {
				// Second pass only on a real flip: capture the offending face geometry.
				std::array<std::array<std::array<double,3>,3>,2> flipped;
				CheckForFlip(v0, v1, outPos, &flipped);
				*outReason = QemRejectionReason::NormalFlipped;
				outPrims->vertices     = { v0, v1 };
				outPrims->edges        = { { v0, v1 } };
				outPrims->targ_ver     = { outPos.X(), outPos.Y(), outPos.Z() };
				outPrims->flipped_face = flipped;
				outPrims->tris_after.clear();   // flip viz supersedes any sliver viz
				outPrims->metrics.clear();
				outPrims->message =
					"Collapse would flip a surviving face: moving the merged vertex to the "
					"optimal position turns a face inside-out (dihedral > 150 deg) or creates "
					"a near-zero-quality sliver. Red = face before, orange = face after.";
			}
		}
	}

	return error;
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckForFlip (tri_edge_collapse_quadric.h:459).  Returns true if, after the
// collapse, two surviving faces sharing an outgoing edge form a dihedral angle
// over 150° (or any surviving face becomes a near-zero-quality sliver).
// Only invoked when HardNormalCheck is on (off by default).
// ─────────────────────────────────────────────────────────────────────────────
bool VcgQuadricSimplifier::CheckForFlip(unsigned v0, unsigned v1, const Wm4::Vector3d& newPos,
                                        std::array<std::array<std::array<double,3>,3>,2>* outFlipped)
{
	const double angleThrRad = 150.0 * 3.14159265358979323846 / 180.0;

	// For the optional flip visualization: track the surviving face whose normal
	// changes most (smallest dot of before-normal with after-normal) and store
	// its before/after triangle into *outFlipped.
	double worstDot = 2.0;
	auto considerFlipViz = [&](const unsigned fv[3], unsigned va) {
		if (!outFlipped) return;
		Wm4::Vector3d bp[3], ap[3];
		for (int k = 0; k < 3; ++k) {
			bp[k] = Pos(fv[k]);
			ap[k] = (fv[k] == va) ? newPos : Pos(fv[k]);
		}
		Wm4::Vector3d bn = NormalizedTriangleNormal(bp[0], bp[1], bp[2]);
		Wm4::Vector3d an = NormalizedTriangleNormal(ap[0], ap[1], ap[2]);
		double d = bn.Dot(an);
		if (d < worstDot) {
			worstDot = d;
			(*outFlipped)[0] = { { {bp[0].X(), bp[0].Y(), bp[0].Z()},
			                       {bp[1].X(), bp[1].Y(), bp[1].Z()},
			                       {bp[2].X(), bp[2].Y(), bp[2].Z()} } };
			(*outFlipped)[1] = { { {ap[0].X(), ap[0].Y(), ap[0].Z()},
			                       {ap[1].X(), ap[1].Y(), ap[1].Z()},
			                       {ap[2].X(), ap[2].Y(), ap[2].Z()} } };
		}
	};

	// vcg keeps ONE edgeNormMap and ONE maxAngle across BOTH endpoint loops
	// (edge_collapse.h sibling: tri_edge_collapse_quadric.h:461).  Sharing them is
	// required so a dihedral measured across a v0-face / v1-face pair (faces sharing
	// the same "other" vertex) is detected.
	std::vector<std::pair<unsigned, Wm4::Vector3d>> edgeNorm;  // (other-vertex, face-normal)
	double maxAngle = 0.0;

	// Process the faces incident to `va` (skipping faces with the partner `skip`).
	// Returns true the moment a near-zero-quality sliver appears (vcg's early-out).
	auto scan = [&](unsigned va, unsigned skip) -> bool {
		for (unsigned fid : m.vertices[va].second->faces_)
		{
			unsigned fv[3];
			if (!FaceVerts(fid, fv)) continue;
			if (fv[0] == skip || fv[1] == skip || fv[2] == skip) continue;

			considerFlipViz(fv, va);

			Wm4::Vector3d p[3];
			for (int k = 0; k < 3; ++k) p[k] = (fv[k] == va) ? newPos : Pos(fv[k]);
			if (QualityFace(p[0], p[1], p[2]) < 0.01) return true;   // sliver ⇒ flip

			Wm4::Vector3d n = NormalizedTriangleNormal(p[0], p[1], p[2]);
			for (int k = 0; k < 3; ++k)
			{
				if (fv[k] == va) continue;
				unsigned other = fv[k];
				bool found = false;
				for (auto& en : edgeNorm)
					if (en.first == other)
					{
						double d = std::max(-1.0, std::min(1.0, n.Dot(en.second)));
						maxAngle = std::max(maxAngle, std::acos(d));
						found = true;
						break;
					}
				if (!found) edgeNorm.push_back({other, n});
			}
		}
		return false;
	};

	// Fast path (no capture): short-circuit on a sliver exactly like vcg, else
	// the shared maxAngle is compared once after both endpoints.
	if (!outFlipped)
	{
		if (scan(v0, v1)) return true;
		if (scan(v1, v0)) return true;
		return maxAngle > angleThrRad;
	}
	// Capture path: run both scans (no early return) so the flip viz reflects the
	// worst face; a sliver in either scan still counts as a flip.
	bool sliver = false;
	if (scan(v0, v1)) sliver = true;
	if (scan(v1, v0)) sliver = true;
	return sliver || (maxAngle > angleThrRad);
}

// ─────────────────────────────────────────────────────────────────────────────
// IsUpToDate (tri_edge_collapse.h:198) — an element is stale if either endpoint
// has been deleted or has been touched (IMark bumped) since the element was made.
// ─────────────────────────────────────────────────────────────────────────────
bool VcgQuadricSimplifier::IsUpToDate(const HeapElem& e) const
{
	if (e.v0 >= m.vertices.size() || e.v1 >= m.vertices.size()) return false;
	if (!m.vertices[e.v0].first || !m.vertices[e.v1].first)     return false;
	if (e.localMark < imark[e.v0] || e.localMark < imark[e.v1]) return false;
	return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// LinkConditions (edge_collapse.h:133) — faithful port of vcg's
// EdgeCollapser::LinkConditions, the Dey-Edelsbrunner topology-preservation test
// (Lk(v0) ∩ Lk(v1) == Lk(v0-v1)).  It is implemented exactly as vcg does it:
// run over the faces incident to each endpoint (VF adjacency), maintaining
// per-vertex and per-edge virtual counters via std::maps, add the dummy vertex
// and dummy edges for boundary endpoints, and finally compare counters:
//   • any shared edge (EdgeCnt == 2)              ⇒ not feasible
//   • #shared verts (VertCnt == 4) != |Lk(edge)|  ⇒ not feasible
// Returns true when the collapse SATISFIES the link condition (feasible).
//
// vcg uses VertexPointer(0) as the dummy id; here the dummy is UINT_MAX (no real
// SlabMesh vertex ever has that id).  Edges are keyed (min,max) consistently for
// both real and dummy edges, which preserves vcg's dummy-edge collision counting.
// ─────────────────────────────────────────────────────────────────────────────
bool VcgQuadricSimplifier::LinkConditions(unsigned v0, unsigned v1) const
{
	const unsigned DUMMY = std::numeric_limits<unsigned>::max();

	auto edgeKey = [](unsigned a, unsigned b) -> std::pair<unsigned, unsigned> {
		return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
	};

	std::map<unsigned, int>                          VertCnt;
	std::map<std::pair<unsigned, unsigned>, int>     EdgeCnt;
	std::vector<unsigned>                            BoundaryVertexVec[2];

	const unsigned endp[2] = { v0, v1 };

	// Collect vertices and edges of the one-rings of v0 and v1 (vcg's first loop).
	for (int i = 0; i < 2; ++i)
	{
		unsigned vi = endp[i];
		if (vi < m.vertices.size() && m.vertices[vi].first)
		{
			for (unsigned fid : m.vertices[vi].second->faces_)
			{
				unsigned fv[3];
				if (!FaceVerts(fid, fv)) continue;
				// The two vertices of this face other than vi  (== vfi.V1(), vfi.V2()).
				unsigned o[2]; int n = 0;
				for (int k = 0; k < 3; ++k) if (fv[k] != vi) o[n++] = fv[k];
				if (n != 2) continue;   // degenerate (vi appears twice) — skip
				++VertCnt[o[0]];
				++VertCnt[o[1]];
				++EdgeCnt[edgeKey(o[0], o[1])];
			}
		}

		// Add dummy stuff for a boundary endpoint (vcg scans the whole accumulated
		// VertCnt each pass; a vertex counted once is on the fan boundary).
		for (auto& kv : VertCnt)
			if (kv.second == 1) BoundaryVertexVec[i].push_back(kv.first);

		if (BoundaryVertexVec[i].size() == 2)
		{
			VertCnt[DUMMY] += 2;
			++EdgeCnt[edgeKey(DUMMY, BoundaryVertexVec[i][0])];
			++EdgeCnt[edgeKey(DUMMY, BoundaryVertexVec[i][1])];
			++VertCnt[BoundaryVertexVec[i][0]];
			++VertCnt[BoundaryVertexVec[i][1]];
		}
	}

	// Cardinality of Lk(v0-v1): the third vertices of faces containing both v0,v1.
	std::vector<unsigned> LkEdge;
	if (v0 < m.vertices.size() && m.vertices[v0].first)
	{
		for (unsigned fid : m.vertices[v0].second->faces_)
		{
			unsigned fv[3];
			if (!FaceVerts(fid, fv)) continue;
			if (fv[0] != v1 && fv[1] != v1 && fv[2] != v1) continue;  // face must contain v1
			for (int k = 0; k < 3; ++k)
				if (fv[k] != v0 && fv[k] != v1) LkEdge.push_back(fv[k]);
		}
	}
	// Boundary collapsing edge ⇒ add the dummy vertex (so |Lk(edge)| >= 2).
	if (LkEdge.size() == 1) LkEdge.push_back(DUMMY);

	// COUNT.
	size_t SharedEdgeCnt = 0;
	for (auto& kv : EdgeCnt) if (kv.second == 2) ++SharedEdgeCnt;
	if (SharedEdgeCnt > 0) return false;

	size_t SharedVertCnt = 0;
	for (auto& kv : VertCnt) if (kv.second == 4) ++SharedVertCnt;
	if (SharedVertCnt != LkEdge.size()) return false;

	return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// IsFeasible (tri_edge_collapse_quadric.h:149) — link condition, only when
// PreserveTopology is set (off in MeshLab defaults).  Calls the faithful vcg
// LinkConditions port above (replacing the earlier WouldCreateNonManifold path).
// ─────────────────────────────────────────────────────────────────────────────
bool VcgQuadricSimplifier::IsFeasible(unsigned v0, unsigned v1, QemReasonPrimitives* outPrims) const
{
	if (!params.PreserveTopology) return true;
	if (LinkConditions(v0, v1)) return true;   // vcg: EdgeCollapser::LinkConditions

	if (outPrims) {
		outPrims->vertices = { v0, v1 };
		outPrims->edges    = { { v0, v1 } };
		// Offending neighbourhood: vertices shared by the one-rings of v0 and v1
		// (beyond the partner itself) — the link-condition violation.
		std::set<unsigned> r0, r1;
		auto ring = [&](unsigned va, std::set<unsigned>& out) {
			if (va >= m.vertices.size() || !m.vertices[va].first) return;
			for (unsigned fid : m.vertices[va].second->faces_) {
				unsigned fv[3];
				if (!FaceVerts(fid, fv)) continue;
				for (int k = 0; k < 3; ++k) if (fv[k] != va) out.insert(fv[k]);
			}
		};
		ring(v0, r0); ring(v1, r1);
		for (unsigned s : r0)
			if (s != v1 && s != v0 && r1.count(s)) outPrims->vertices.push_back(s);
		outPrims->message =
			"Collapse would break manifoldness (link condition): the one-rings of the two "
			"endpoints share vertices other than the two opposite the collapsed edge, so "
			"merging them would create a non-manifold edge/vertex. Highlighted vertices are "
			"the shared neighbours.";
	}
	return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// AddCollapseToHeap (tri_edge_collapse_quadric.h:577) — compute the priority for
// (v0,v1), and push it unless the error exceeds the admissible max.
//
// vcg: IsSymmetric == OptimalPlacement.  When the collapse is SYMMETRIC
// (OptimalPlacement=true) v0→v1 and v1→v0 are identical, so a single directed
// entry per pair is pushed (the real-target path).  When it is A-SYMMETRIC
// (OptimalPlacement=false) the two directions have different cost and survivor
// (newPos = Pos(v1)), so vcg pushes BOTH directions (tri_edge_collapse_quadric.h:589).
// Routing both seeding and UpdateHeap through here reproduces vcg, where the same
// helper adds the backward edge for the asymmetric case.
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::AddCollapseToHeap(unsigned v0, unsigned v1)
{
	// Push one DIRECTED candidate collapse (v0 deleted into v1 at the computed pos).
	auto push = [&](unsigned a, unsigned b)
	{
		if (a == b) return;
		if (a >= m.vertices.size() || b >= m.vertices.size()) return;
		if (!m.vertices[a].first || !m.vertices[b].first) return;

		HeapElem e;
		e.v0 = a; e.v1 = b;
		e.localMark = globalMark;

		QemRejectionReason  reason = QemRejectionReason::None;
		QemReasonPrimitives prims;
		e.pri = ComputePriority(a, b, e.optPos, &reason, &prims);

		if (e.pri >= std::numeric_limits<double>::max()) {   // maxAdmitErr — hard veto
			if (reason != QemRejectionReason::None)
				RecordRejection(a, b, reason, std::move(prims));
			return;                                          // dropped (not DiagnoseOnly)
		}

		// Finite priority → this candidate WILL go on the heap.  In DiagnoseOnly mode a
		// failing check still set `reason` (but not +inf): colour the edge yet keep it.
		// Otherwise the edge is genuinely fine — clear any stale rejection mark.
		if (reason != QemRejectionReason::None)
			RecordRejection(a, b, reason, std::move(prims));
		else
			ClearRejection(a, b);

		heap.push_back(e);
		std::push_heap(heap.begin(), heap.end(), HeapLess);
	};

	push(v0, v1);
	if (!params.OptimalPlacement)   // vcg !IsSymmetric ⇒ also seed the reverse direction
		push(v1, v0);
}

// ─────────────────────────────────────────────────────────────────────────────
// RecordRejection / ClearRejection — resolve the slab edge id for (v0,v1) and
// store / erase its QEM rejection in SlabMesh's qem_edge_* maps (read by the
// QEM rejection viewer in main_cli).
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::RecordRejection(unsigned v0, unsigned v1,
                                           QemRejectionReason reason, QemReasonPrimitives prims)
{
	unsigned eid;
	if (!m.Edge(v0, v1, eid)) return;   // no slab edge → nothing to colour
	m.qem_edge_last_rejection[eid]    = reason;
	m.qem_edge_reason_primitives[eid] = std::move(prims);
}

void VcgQuadricSimplifier::ClearRejection(unsigned v0, unsigned v1)
{
	unsigned eid;
	if (!m.Edge(v0, v1, eid)) return;
	m.qem_edge_last_rejection.erase(eid);
	m.qem_edge_reason_primitives.erase(eid);
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute — faithful port of vcg's collapse: TriEdgeCollapseQuadric::Execute
// (tri_edge_collapse_quadric.h:182) followed by EdgeCollapser::Do
// (edge_collapse.h:212).  vcg does, in order:
//     QH::Qd(pos.V(1)) += QH::Qd(pos.V(0));            // v1 accumulates v0's quadric
//     EdgeCollapser::Do(m, pos, newPos);               // v0 deleted, v1 survives
// and Do() deletes the faces incident on BOTH endpoints (the AV01 set on the
// collapsing edge), redirects the faces incident only on v0 (AV0) so v0→v1,
// deletes v0, and moves v1 to newPos.
//
// We reproduce this EXACTLY on SlabMesh, keeping v1 as the survivor (it keeps its
// id, like vcg — NOT a fresh id as MergeVertices allocated):
//   • snapshot the AV0 faces as redirected triangles {v1, x, y};
//   • DeleteVertex(v0), which cascades v0's edges/faces away — removing the AV01
//     faces (exactly as vcg) and the AV0 faces (re-created next).  v1's faces that
//     don't touch v0 are left untouched;
//   • re-insert the redirected AV0 faces anchored on v1.
// This uses SlabMesh's own primitives so the edge/face containers stay consistent,
// while producing vcg's precise topology.  MergeVertices is left intact (unused by
// this path); its QMAT cluster/struct bookkeeping diverged from vcg.
// ─────────────────────────────────────────────────────────────────────────────
unsigned VcgQuadricSimplifier::Execute(unsigned v0, unsigned v1, const Wm4::Vector3d& optPos)
{
	if (v0 == v1) return (unsigned)-1;
	if (v0 >= m.vertices.size() || v1 >= m.vertices.size()) return (unsigned)-1;
	if (!m.vertices[v0].first || !m.vertices[v1].first)      return (unsigned)-1;

	// vcg: QH::Qd(v1) += QH::Qd(v0).
	VcgQuadric merged = vq[v0];
	merged += vq[v1];

	const double r0 = m.vertices[v0].second->sphere.radius;
	const double r1 = m.vertices[v1].second->sphere.radius;
	const double rt = 0.5 * (r0 + r1);

#ifdef QMAT_WITH_POLYSCOPE
	if (m.on_collapse_cb)
	{
		// Callback in NORMALIZED coords (matching sphere.center / radius storage);
		// optPos is at the .off scale, so map it back with / qemScale.
		const Wm4::Vector3d c0 = m.vertices[v0].second->sphere.center;
		const Wm4::Vector3d c1 = m.vertices[v1].second->sphere.center;
		Sphere s; s.center = optPos / qemScale; s.radius = rt;
		m.on_collapse_cb(v0, c0, r0, v1, c1, r1, s);
	}
#endif

	// 1. Snapshot the AV0 faces (incident on v0 but NOT v1) as redirected triangles
	//    {v1, x, y}.  Carry struct_id so QMAT face metadata survives the collapse.
	struct RedirectedFace { std::set<unsigned> vset; int struct_id; };
	std::vector<RedirectedFace> redirected;
	for (unsigned fid : m.vertices[v0].second->faces_)
	{
		if (fid >= m.faces.size() || !m.faces[fid].first) continue;
		const std::set<unsigned>& vs = m.faces[fid].second->vertices_;
		if (vs.find(v1) != vs.end()) continue;            // AV01 (on the edge) → drop
		std::set<unsigned> nv;
		for (unsigned vv : vs) nv.insert(vv == v0 ? v1 : vv);
		if (nv.size() != 3) continue;                     // degenerate → drop
		redirected.push_back({ nv, m.faces[fid].second->struct_id });
	}

	// 2. Move the survivor to the optimal position BEFORE rebuilding faces, so the
	//    re-inserted triangles get their slab geometry computed at the final spot.
	//    optPos is at the .off (qemScale) scale; SlabMesh stores normalized centers,
	//    so map it back with / qemScale.
	m.vertices[v1].second->sphere.center = optPos / qemScale;
	m.vertices[v1].second->sphere.radius = rt;

	// 3. Delete v0 — cascades through its edges/faces, deleting BOTH the AV01 faces
	//    (correctly gone, like vcg) and the AV0 faces (re-created in step 4).
	m.DeleteVertex(v0);

	// 4. Re-insert the redirected AV0 faces, now anchored on v1.  InsertFace creates
	//    any missing (v1,*) edges, registers all adjacency, and skips duplicates
	//    (which LinkConditions guarantees cannot arise when PreserveTopology is on).
	for (const RedirectedFace& rf : redirected)
	{
		m.InsertFace(rf.vset);
		if (rf.struct_id >= 0)
		{
			unsigned nfid;
			if (m.Face(rf.vset, nfid) && m.faces[nfid].first)
				m.faces[nfid].second->struct_id = rf.struct_id;
		}
	}

	// 5. Survivor carries the merged quadric.
	EnsureSized();
	vq[v1] = merged;
	return v1;
}

// ─────────────────────────────────────────────────────────────────────────────
// UpdateHeap (tri_edge_collapse_quadric.h:523) — bump GlobalMark and the IMark
// of the surviving vertex and its ring, then re-add every collapse incident to
// the surviving vertex plus the outer-ring edges (pairs of ring vertices that
// share a face with it).  Older entries touching the ring are now stale.
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::UpdateHeap(unsigned vid_tgt)
{
	++globalMark;
	EnsureSized();
	imark[vid_tgt] = globalMark;

	// Ring = vertices sharing a face with vid_tgt (vcg iterates faces via VFIterator).
	std::set<unsigned> ring;
	for (unsigned fid : m.vertices[vid_tgt].second->faces_)
	{
		unsigned fv[3];
		if (!FaceVerts(fid, fv)) continue;
		for (int k = 0; k < 3; ++k)
			if (fv[k] != vid_tgt) ring.insert(fv[k]);
	}
	for (unsigned r : ring) imark[r] = globalMark;

	// Re-add edges from vid_tgt to each ring vertex.
	for (unsigned r : ring)
		AddCollapseToHeap(vid_tgt, r);

	// Re-add the outer-ring edges (the two non-target vertices of each incident
	// face).  Dedup so a shared outer edge is not pushed twice.
	std::set<std::pair<unsigned, unsigned>> seen;
	for (unsigned fid : m.vertices[vid_tgt].second->faces_)
	{
		unsigned fv[3];
		if (!FaceVerts(fid, fv)) continue;
		unsigned a = (unsigned)-1, b = (unsigned)-1;
		for (int k = 0; k < 3; ++k)
			if (fv[k] != vid_tgt) { (a == (unsigned)-1 ? a : b) = fv[k]; }
		if (a == (unsigned)-1 || b == (unsigned)-1) continue;
		auto key = std::minmax(a, b);
		if (seen.insert({key.first, key.second}).second)
			AddCollapseToHeap(a, b);
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// ClearHeap (local_optimization.h:237) — drop stale elements when the heap grows
// past SimplexNumber * HeapSimplexRatio, then rebuild.
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::ClearHeap()
{
	std::vector<HeapElem> kept;
	kept.reserve(heap.size());
	for (const HeapElem& e : heap)
		if (IsUpToDate(e)) kept.push_back(e);
	heap.swap(kept);
	std::make_heap(heap.begin(), heap.end(), HeapLess);
}

// ─────────────────────────────────────────────────────────────────────────────
// Collapse logging — emit one JSONL record per accepted collapse, with vertex
// ids and coordinates in the _mat_initial.off space MeshLab reads (see
// md_files/qem/collapse_records_schema.md).
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::EnableCollapseLog(const std::string& path)
{
	collapseLogPath    = path;
	collapseLogEnabled = true;
}

// slab vid -> OFF index, mirroring ExportMatAsOff's active-vertex compaction.
void VcgQuadricSimplifier::BuildOffIndexMap()
{
	slabToOff.assign(m.vertices.size(), std::numeric_limits<unsigned>::max());
	unsigned next = 0;
	for (unsigned i = 0; i < m.vertices.size(); ++i)
		if (m.vertices[i].first) slabToOff[i] = next++;
}

// Load the exported _mat_initial.off and store its vertex coordinates (parsed as
// FLOAT, exactly as MeshLab's MESHLAB_SCALAR=float reader does) indexed by OFF
// index.  Path is derived from collapseLogPath (<prefix>_collapse_records.jsonl ->
// <prefix>_mat_initial.off).  Returns true on success (sets useOffCoords).
bool VcgQuadricSimplifier::LoadOffCoords()
{
	if (collapseLogPath.empty()) return false;
	std::filesystem::path lp(collapseLogPath);
	std::string stem = lp.filename().string();
	const std::string suf = "_collapse_records.jsonl";
	if (stem.size() > suf.size() && stem.compare(stem.size() - suf.size(), suf.size(), suf) == 0)
		stem.resize(stem.size() - suf.size());
	std::filesystem::path off = lp.parent_path() / (stem + "_mat_initial.off");

	std::ifstream f(off.string());
	if (!f) { std::cerr << "[VcgQuadricSimplifier] LoadOffCoords: cannot open " << off.string() << "\n"; return false; }

	std::string tok;
	f >> tok;                                   // "OFF"
	if (tok != "OFF") { std::cerr << "[VcgQuadricSimplifier] LoadOffCoords: not an OFF file\n"; return false; }
	size_t nv = 0, nf = 0, ne = 0;
	f >> nv >> nf >> ne;
	offCoords.assign(nv, Wm4::Vector3d(0, 0, 0));
	for (size_t i = 0; i < nv; ++i)
	{
		double x, y, z;
		if (!(f >> x >> y >> z)) { std::cerr << "[VcgQuadricSimplifier] LoadOffCoords: truncated at vertex " << i << "\n"; return false; }
		// Cast to float to match MeshLab's float vertex storage (the value MeshLab
		// actually feeds InitQuadric), then keep it as a double in the quadric math.
		offCoords[i] = Wm4::Vector3d((float)x, (float)y, (float)z);
	}
	useOffCoords = true;
	std::cerr << "[VcgQuadricSimplifier] LoadOffCoords: using " << nv
	          << " exact .off coordinates (float) for Pos()\n";
	return true;
}

void VcgQuadricSimplifier::LogCollapse(unsigned v0, unsigned v1, double cost,
                                       const Wm4::Vector3d& newPos)
{
	if (collapseIdx == 0)
	{
		std::filesystem::path p(collapseLogPath);
		if (p.has_parent_path())
		{
			std::error_code ec;
			std::filesystem::create_directories(p.parent_path(), ec);
		}
		collapseLog.open(collapseLogPath, std::ios::out | std::ios::trunc);
		collapseLog << std::setprecision(10);
		std::cerr << "[VcgQuadricSimplifier] collapse log -> " << collapseLogPath << "\n";
	}
	std::ofstream& out = collapseLog;
	if (!out) return;

	auto vid = [&](unsigned v) { return (v < slabToOff.size()) ? slabToOff[v] : v; };
	auto p3  = [&](const Wm4::Vector3d& p) {
		out << "[" << p.X() << "," << p.Y() << "," << p.Z() << "]";
	};

	// Serialize one face: {"id":fid,"vert_ids":[...],"verts":[[...],...]}.
	auto face = [&](unsigned fid) {
		const auto& vs = m.faces[fid].second->vertices_;
		out << "{\"id\":" << fid << ",\"vert_ids\":[";
		bool first = true;
		for (unsigned v : vs) { out << (first ? "" : ",") << vid(v); first = false; }
		out << "],\"verts\":[";
		first = true;
		for (unsigned v : vs) { if (!first) out << ","; p3(Pos(v)); first = false; }
		out << "]}";
	};

	std::vector<unsigned> deleted, modified;
	for (unsigned fid : m.vertices[v0].second->faces_)
	{
		if (fid >= m.faces.size() || !m.faces[fid].first) continue;
		const auto& vs = m.faces[fid].second->vertices_;
		(vs.find(v1) != vs.end() ? deleted : modified).push_back(fid);
	}

	out << "{\"idx\":" << collapseIdx << ",\"cost\":" << cost
	    << ",\"edge\":{\"v0\":{\"id\":" << vid(v0) << ",\"pos\":"; p3(Pos(v0));
	out << "},\"v1\":{\"id\":" << vid(v1) << ",\"pos\":"; p3(Pos(v1));
	out << "}},\"new_pos\":"; p3(newPos);
	out << ",\"deleted_faces\":[";
	for (size_t i = 0; i < deleted.size(); ++i) { if (i) out << ","; face(deleted[i]); }
	out << "],\"modified_faces\":[";
	for (size_t i = 0; i < modified.size(); ++i) { if (i) out << ","; face(modified[i]); }
	out << "]}\n";
	out.flush();
}

// Snapshot of the seeded heap (every candidate collapse + cost), sorted ascending
// by cost — initial_heap_state_schema.md.  Written once before the first collapse.
void VcgQuadricSimplifier::WriteInitialHeapState()
{
	std::filesystem::path lp(collapseLogPath);
	std::string stem = lp.filename().string();
	const std::string suf = "_collapse_records.jsonl";
	if (stem.size() > suf.size() && stem.compare(stem.size() - suf.size(), suf.size(), suf) == 0)
		stem.resize(stem.size() - suf.size());
	std::filesystem::path out = lp.parent_path() / (stem + "_initial_heap_state_.json");
	std::error_code ec;
	if (lp.has_parent_path()) std::filesystem::create_directories(lp.parent_path(), ec);

	std::vector<HeapElem> sorted = heap;
	std::sort(sorted.begin(), sorted.end(),
	          [](const HeapElem& a, const HeapElem& b) { return a.pri < b.pri; });

	std::ofstream f(out.string());
	if (!f) return;
	f << std::setprecision(17);
	auto vid = [&](unsigned v) { return (v < slabToOff.size()) ? slabToOff[v] : v; };

	f << "{\n  \"mesh\": \"" << stem << "\",\n"
	  << "  \"heap_size\": " << sorted.size() << ",\n"
	  << "  \"order\": \"ascending_by_cost\",\n"
	  << "  \"entries\": [\n";
	for (size_t i = 0; i < sorted.size(); ++i)
	{
		const HeapElem& e = sorted[i];
		f << "    {\"rank\": " << i << ", \"v0_id\": " << vid(e.v0)
		  << ", \"v1_id\": " << vid(e.v1) << ", \"cost\": " << e.pri
		  << ", \"opt\": [" << e.optPos.X() << ", " << e.optPos.Y() << ", " << e.optPos.Z() << "]}"
		  << (i + 1 < sorted.size() ? ",\n" : "\n");
	}
	f << "  ]\n}\n";
	std::cerr << "[VcgQuadricSimplifier] initial heap state -> " << out.string() << "\n";
}

// Per-vertex accumulated InitQuadric quadrics (a[6],b[3],c) at the .off scale,
// written to <prefix>_vertex_quadrics_.json so the InitQuadric stage can be
// verified vertex-by-vertex against MeshLab's matching dump.
void VcgQuadricSimplifier::WriteVertexQuadrics()
{
	std::filesystem::path lp(collapseLogPath);
	std::string stem = lp.filename().string();
	const std::string suf = "_collapse_records.jsonl";
	if (stem.size() > suf.size() && stem.compare(stem.size() - suf.size(), suf.size(), suf) == 0)
		stem.resize(stem.size() - suf.size());
	std::filesystem::path out = lp.parent_path() / (stem + "_vertex_quadrics_.json");

	std::ofstream f(out.string());
	if (!f) return;
	f << std::setprecision(17);
	auto vid = [&](unsigned v) { return (v < slabToOff.size()) ? slabToOff[v] : v; };

	// count active vertices first
	size_t n = 0;
	for (size_t i = 0; i < m.vertices.size(); ++i)
		if (m.vertices[i].first && i < vq.size()) ++n;

	f << "{\"mesh\": \"" << stem << "\", \"count\": " << n << ", \"verts\": [\n";
	bool first = true;
	for (size_t i = 0; i < m.vertices.size(); ++i)
	{
		if (!m.vertices[i].first || i >= vq.size()) continue;
		const VcgQuadric& q = vq[i];
		const Wm4::Vector3d p = Pos(static_cast<unsigned>(i));
		if (!first) f << ",\n";
		first = false;
		f << "    {\"id\": " << vid(static_cast<unsigned>(i))
		  << ", \"pos\": [" << p.X() << ", " << p.Y() << ", " << p.Z() << "]"
		  << ", \"a\": [" << q.a[0] << ", " << q.a[1] << ", " << q.a[2] << ", "
		                  << q.a[3] << ", " << q.a[4] << ", " << q.a[5] << "]"
		  << ", \"b\": [" << q.b[0] << ", " << q.b[1] << ", " << q.b[2] << "]"
		  << ", \"c\": " << q.c << "}";
	}
	f << "\n  ]\n}\n";
	std::cerr << "[VcgQuadricSimplifier] vertex quadrics -> " << out.string() << "\n";
}

// The fully-resolved VcgQuadricParameter used for this run, written to
// <prefix>_qem_params.json. Called after InitQuadrics so ScaleFactor (and the
// NormalThrRad/CosineThr adjustments) are final.
void VcgQuadricSimplifier::WriteParams()
{
	std::filesystem::path lp(collapseLogPath);
	std::string stem = lp.filename().string();
	const std::string suf = "_collapse_records.jsonl";
	if (stem.size() > suf.size() && stem.compare(stem.size() - suf.size(), suf.size(), suf) == 0)
		stem.resize(stem.size() - suf.size());
	std::filesystem::path out = lp.parent_path() / (stem + "_qem_params.json");
	std::error_code ec;
	if (lp.has_parent_path()) std::filesystem::create_directories(lp.parent_path(), ec);

	std::ofstream f(out.string());
	if (!f) return;
	f << std::setprecision(17) << std::boolalpha;
	const VcgQuadricParameter& p = params;
	f << "{\n  \"mesh\": \"" << stem << "\",\n"
	  << "  \"qemScale\": " << qemScale << ",\n"
	  << "  \"BoundaryQuadricWeight\": " << p.BoundaryQuadricWeight << ",\n"
	  << "  \"AreaCheck\": " << p.AreaCheck << ",\n"
	  << "  \"HardQualityCheck\": " << p.HardQualityCheck << ",\n"
	  << "  \"HardQualityThr\": " << p.HardQualityThr << ",\n"
	  << "  \"HardNormalCheck\": " << p.HardNormalCheck << ",\n"
	  << "  \"NormalCheck\": " << p.NormalCheck << ",\n"
	  << "  \"NormalThrRad\": " << p.NormalThrRad << ",\n"
	  << "  \"CosineThr\": " << p.CosineThr << ",\n"
	  << "  \"OptimalPlacement\": " << p.OptimalPlacement << ",\n"
	  << "  \"PreserveTopology\": " << p.PreserveTopology << ",\n"
	  << "  \"QuadricEpsilon\": " << p.QuadricEpsilon << ",\n"
	  << "  \"QualityCheck\": " << p.QualityCheck << ",\n"
	  << "  \"QualityThr\": " << p.QualityThr << ",\n"
	  << "  \"QualityQuadric\": " << p.QualityQuadric << ",\n"
	  << "  \"QualityQuadricWeight\": " << p.QualityQuadricWeight << ",\n"
	  << "  \"QualityWeight\": " << p.QualityWeight << ",\n"
	  << "  \"QualityWeightFactor\": " << p.QualityWeightFactor << ",\n"
	  << "  \"ScaleFactor\": " << p.ScaleFactor << ",\n"
	  << "  \"ScaleIndependent\": " << p.ScaleIndependent << ",\n"
	  << "  \"UseArea\": " << p.UseArea << ",\n"
	  << "  \"UseVertexWeight\": " << p.UseVertexWeight << ",\n"
	  << "  \"SVDPlacement\": " << p.SVDPlacement << ",\n"
	  << "  \"DiagnoseOnly\": " << p.DiagnoseOnly << ",\n"
	  << "  \"FaceTarget\": " << p.FaceTarget << "\n}\n";
	std::cerr << "[VcgQuadricSimplifier] qem params -> " << out.string() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// PostSimplificationCleaning (meshfilter.cpp AutoClean) — remove null faces,
// then duplicate vertices, then unreferenced vertices.
// ─────────────────────────────────────────────────────────────────────────────
void VcgQuadricSimplifier::PostSimplificationCleaning()
{
	// 1. Null (zero-area) faces — RemoveFaceOutOfRangeArea(m, 0).
	int nullFaces = 0;
	for (size_t fid = 0; fid < m.faces.size(); ++fid)
	{
		unsigned fv[3];
		if (!FaceVerts(static_cast<unsigned>(fid), fv)) continue;
		double a2 = (Pos(fv[1]) - Pos(fv[0])).Cross(Pos(fv[2]) - Pos(fv[0])).Length();
		if (a2 <= 0.0) { m.DeleteFace(static_cast<unsigned>(fid)); ++nullFaces; }
	}

	// 2. Duplicate vertices — RemoveDuplicateVertex.  Group active verts by exact
	//    position; keep the lowest id, repoint faces of the rest onto it.
	int dupVerts = 0;
	std::map<std::array<double,3>, unsigned> seen;
	for (unsigned i = 0; i < m.vertices.size(); ++i)
	{
		if (!m.vertices[i].first) continue;
		const auto& c = m.vertices[i].second->sphere.center;
		std::array<double,3> key{ c.X(), c.Y(), c.Z() };
		auto it = seen.find(key);
		if (it == seen.end()) { seen[key] = i; continue; }

		unsigned keep = it->second;
		std::vector<std::set<unsigned>> redirect;
		for (unsigned fid : m.vertices[i].second->faces_)
		{
			if (fid >= m.faces.size() || !m.faces[fid].first) continue;
			std::set<unsigned> nv;
			for (unsigned v : m.faces[fid].second->vertices_) nv.insert(v == i ? keep : v);
			if (nv.size() == 3) redirect.push_back(nv);
		}
		m.DeleteVertex(i);
		for (const auto& nv : redirect) m.InsertFace(nv);
		++dupVerts;
	}

	// 3. Unreferenced vertices — RemoveUnreferencedVertex (no incident face).
	int unrefVerts = 0;
	for (unsigned i = 0; i < m.vertices.size(); ++i)
		if (m.vertices[i].first && m.vertices[i].second->faces_.empty())
		{
			m.DeleteVertex(i);
			++unrefVerts;
		}

	std::cerr << "[VcgQuadricSimplifier] PostSimplificationCleaning: "
	          << nullFaces << " null faces, " << dupVerts << " duplicate verts, "
	          << unrefVerts << " unreferenced verts removed; MAT vertices = "
	          << m.numVertices << ", faces = " << m.numFaces << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Run — vcg LocalOptimization::Init + DoOptimization.
//   • seed the heap with one entry per face edge (vcg seeds via faces, so edges
//     with no incident face are never collapsed — matching vcg on a tri mesh);
//   • pop the cheapest, skip if stale, skip if infeasible, else Execute +
//     UpdateHeap.  Stop after maxCollapses collapses (Simplify's threshold) or
//     when only one vertex remains.
// ─────────────────────────────────────────────────────────────────────────────
unsigned VcgQuadricSimplifier::Run(int maxCollapses)
{
	// Decimate at the SAME scale as the _mat_initial.off MeshLab reads: its coords
	// are sphere.center * bb_diagonal_length (ExportMatAsOff).  Using that exact
	// factor makes Pos() return the .off coordinates, so vcg's scale-dependent QEM
	// (area-weighted face quadrics vs unit border quadrics) behaves identically.
	qemScale = (m.pmesh ? m.pmesh->bb_diagonal_length : 1.0);
	std::cerr << "[VcgQuadricSimplifier] qemScale (bb_diagonal_length) = " << qemScale
	          << "  (Pos evaluated at _mat_initial.off scale)\n";

	// MeshLab's QuadricSimplification driver tightens the normal threshold when
	// NormalCheck is on (quadric_simp.cpp:54).  Mirror it before CosineThr.
	if (params.NormalCheck) params.NormalThrRad = 3.14159265358979323846 / 4.0;
	params.CosineThr = std::cos(params.NormalThrRad);

	globalMark = 0;
	imark.assign(m.vertices.size(), 0);
	vq.assign(m.vertices.size(), VcgQuadric());

	// Build the OFF index map and load the exact .off coordinates BEFORE InitQuadrics,
	// so Pos() returns the bit-identical positions MeshLab reads (text → float).
	// (slabToOff reflects the initial active vertices; no collapses have happened yet.)
	BuildOffIndexMap();
	LoadOffCoords();

	InitQuadrics();

	// Seed: every distinct edge that belongs to at least one face.
	heap.clear();
	{
		std::set<std::pair<unsigned, unsigned>> seen;
		for (size_t fid = 0; fid < m.faces.size(); ++fid)
		{
			unsigned fv[3];
			if (!FaceVerts(static_cast<unsigned>(fid), fv)) continue;
			for (int j = 0; j < 3; ++j)
			{
				unsigned a = fv[j], b = fv[(j + 1) % 3];
				auto key = std::minmax(a, b);
				if (seen.insert({key.first, key.second}).second)
					AddCollapseToHeap(key.first, key.second);
			}
		}
	}
	std::make_heap(heap.begin(), heap.end(), HeapLess);

	if (collapseLogEnabled) { BuildOffIndexMap(); WriteInitialHeapState(); WriteVertexQuadrics(); WriteParams(); }

	const float ratio = params.OptimalPlacement ? 4.0f : 8.0f;  // HeapSimplexRatio

	std::cerr << "[VcgQuadricSimplifier] seeded heap with " << heap.size()
	          << " candidate collapses; target collapses = " << maxCollapses
	          << ", start MAT vertices = " << m.numVertices << "\n";

	// When a face target is set (--nf) it governs the run: keep collapsing until
	// the MAT face count reaches it, bypassing the vertex-derived maxCollapses
	// bound.  Otherwise maxCollapses (from --simplify) is the stop.
	const bool useFaceTarget = params.FaceTarget >= 0;
	if (useFaceTarget)
		std::cerr << "[VcgQuadricSimplifier] face target active (--nf): stopping at "
		          << "MAT faces <= " << params.FaceTarget
		          << " (start faces = " << m.numFaces << ")\n";

	unsigned collapses = 0;
	while (m.numVertices > 1 && !heap.empty()
	       && (useFaceTarget ? (int)m.numFaces > params.FaceTarget
	                         : collapses < (unsigned)maxCollapses))
	{
		if (heap.size() > (size_t)(m.numFaces * ratio)) ClearHeap();
		if (heap.empty()) break;

		std::pop_heap(heap.begin(), heap.end(), HeapLess);
		HeapElem e = heap.back();
		heap.pop_back();

		if (!IsUpToDate(e)) continue;
		{
			QemReasonPrimitives prims;
			if (!IsFeasible(e.v0, e.v1, &prims)) {
				RecordRejection(e.v0, e.v1, QemRejectionReason::NonManifoldLinkCondition,
				                std::move(prims));
				// Veto only when not diagnosing; DiagnoseOnly colours it yet collapses.
				if (!params.DiagnoseOnly) continue;
			}
		}

		// Log before Execute — face/vertex data must be pre-collapse.
		if (collapseLogEnabled) { LogCollapse(e.v0, e.v1, e.pri, e.optPos); ++collapseIdx; }

		unsigned vid_tgt = Execute(e.v0, e.v1, e.optPos);
		if (vid_tgt == (unsigned)-1) continue;

		UpdateHeap(vid_tgt);
		++collapses;
	}

	std::cerr << "[VcgQuadricSimplifier] done: " << collapses
	          << " collapses, MAT vertices = " << m.numVertices << "\n";
	return collapses;
}

#endif  // ONLY_USE_QEM_CONDITION_CHECKS
