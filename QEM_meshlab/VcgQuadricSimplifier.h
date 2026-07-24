#ifndef _VCG_QUADRIC_SIMPLIFIER_H
#define _VCG_QUADRIC_SIMPLIFIER_H

// ─────────────────────────────────────────────────────────────────────────────
// VcgQuadricSimplifier
//
// A faithful, self-contained port of vcglib's quadric edge-collapse
// decimation (vcg::tri::TriEdgeCollapseQuadric + the LocalOptimization driver)
// onto QMAT's SlabMesh, treating the medial axis as a triangle mesh whose
// vertices are the sphere centres (radii ignored).  It reproduces, step for
// step, the four pieces that determine the result:
//
//   1. InitQuadric      — area-weighted face plane quadrics + boundary quadrics
//   2. ComputePosition  — optimal placement = quadric minimum (midpoint fallback)
//   3. ComputePriority  — QuadErr·ScaleFactor with the soft/hard quality, normal
//                         and area terms
//   4. the lazy heap    — std::vector + push_heap/pop_heap with the mark-based
//                         IsUpToDate staleness test, UpdateHeap, ClearHeap and
//                         the pop → IsUpToDate → IsFeasible → Execute → UpdateHeap
//                         main loop (vcg's DoOptimization).
//
// Everything new lives in THIS class — no fields are added to SlabMesh,
// SlabVertex or PrimMesh, so a build without ONLY_USE_QEM_CONDITION_CHECKS is
// byte-identical to before.  The whole file compiles to nothing when the flag
// is off.
//
// Reference sources are bundled under qmat/vcglib/.  See
// md_files/vcg_faithful_qem.md for the line-by-line mapping.
// ─────────────────────────────────────────────────────────────────────────────

#if defined(ONLY_USE_QEM_CONDITION_CHECKS)

#include "SlabMesh.h"
#include <vector>
#include <string>
#include <fstream>

// ─────────────────────────────────────────────────────────────────────────────
// Faithful port of vcg::math::Quadric<double> (vcglib/quadric.h).
//
//   f(x) = x^T A x + b·x + c
//
// A is symmetric 3x3 stored as its 6 distinct entries a[] = {a11,a12,a13,a22,
// a23,a33}.  A quadric built ByPlane(n, off) (n = plane normal, off = n·p for a
// point p on the plane) evaluates to the squared distance to that plane:
//   Apply(x) = (n·x − off)^2.
// ─────────────────────────────────────────────────────────────────────────────
struct VcgQuadric
{
	double a[6];   // a11 a12 a13 a22 a23 a33   (upper triangle of symmetric A)
	double b[3];   // r3 vector
	double c;      // scalar (-1 == null/uninitialised, matching vcg)

	VcgQuadric() : c(-1.0) { a[0]=a[1]=a[2]=a[3]=a[4]=a[5]=0.0; b[0]=b[1]=b[2]=0.0; }

	bool IsValid() const { return c >= 0.0; }

	void SetZero()
	{
		a[0]=a[1]=a[2]=a[3]=a[4]=a[5]=0.0;
		b[0]=b[1]=b[2]=0.0;
		c=0.0;
	}

	// vcg Quadric::ByPlane.  n may already be scaled by a weight (vcg amplifies
	// the plane Direction() before building K, so the quadric scales by w²).
	void ByPlane(const Wm4::Vector3d& n, double off)
	{
		a[0] = n.X()*n.X();  a[1] = n.Y()*n.X();  a[2] = n.Z()*n.X();
		a[3] = n.Y()*n.Y();  a[4] = n.Z()*n.Y();
		a[5] = n.Z()*n.Z();
		b[0] = -2.0*off*n.X();
		b[1] = -2.0*off*n.Y();
		b[2] = -2.0*off*n.Z();
		c    = off*off;
	}

	void operator+=(const VcgQuadric& q)
	{
		a[0]+=q.a[0]; a[1]+=q.a[1]; a[2]+=q.a[2];
		a[3]+=q.a[3]; a[4]+=q.a[4]; a[5]+=q.a[5];
		b[0]+=q.b[0]; b[1]+=q.b[1]; b[2]+=q.b[2];
		c+=q.c;
	}

	void operator*=(double w)
	{
		a[0]*=w; a[1]*=w; a[2]*=w; a[3]*=w; a[4]*=w; a[5]*=w;
		b[0]*=w; b[1]*=w; b[2]*=w; c*=w;
	}

	// vcg Quadric::Apply.  Cross terms carry the factor 2 because only the upper
	// triangle of A is stored.
	double Apply(const Wm4::Vector3d& p) const
	{
		return p.X()*p.X()*a[0] + 2.0*p.X()*p.Y()*a[1] + 2.0*p.X()*p.Z()*a[2] + p.X()*b[0]
		     + p.Y()*p.Y()*a[3] + 2.0*p.Y()*p.Z()*a[4] + p.Y()*b[1]
		     + p.Z()*p.Z()*a[5] + p.Z()*b[2] + c;
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// Faithful port of vcg::tri::TriEdgeCollapseQuadricParameter.
// Defaults are the vcglib *source* defaults (note: MeshLab's GUI overrides a
// few — e.g. it ships BoundaryQuadricWeight = 1.0 — see md_files/vcg_faithful_qem.md).
// ─────────────────────────────────────────────────────────────────────────────
struct VcgQuadricParameter
{
	double BoundaryQuadricWeight = 0.5;
	bool   AreaCheck             = false;
	bool   HardQualityCheck      = false;
	double HardQualityThr        = 0.1;
	bool   HardNormalCheck       = false;
	bool   NormalCheck           = false;
	double NormalThrRad          = 3.14159265358979323846 / 2.0;  // M_PI/2
	double CosineThr             = 0.0;   // recomputed in Run() = cos(NormalThrRad)
	bool   OptimalPlacement      = false;
	bool   PreserveTopology      = false;
	double QuadricEpsilon        = 1e-15;
	bool   QualityCheck          = true;
	double QualityThr            = 0.3;
	bool   QualityQuadric        = false;
	double QualityQuadricWeight  = 0.001;
	bool   QualityWeight         = false;
	double QualityWeightFactor   = 100.0;
	double ScaleFactor           = 1.0;   // overwritten by ScaleIndependent
	bool   ScaleIndependent      = true;
	bool   UseArea               = true;
	bool   UseVertexWeight       = false;
	// vcg parameter (struct default false). When true ComputePosition uses
	// MinimumClosestToPoint (SVD, qeps=1e-3 truncation pinned to the midpoint)
	// instead of fullPivLu Minimum. fullPivLu's rank threshold is Eigen-version
	// dependent: on ill-conditioned (sharp-feature) interior quadrics our Eigen
	// keeps them full-rank and the solve flies far (cost ~1e20), so those edges
	// never collapse. SVD truncation bounds them to MeshLab's range.
	bool   SVDPlacement          = false;

	// NOT a vcg parameter.  When true, the hard-reject gates (HardQualityCheck,
	// HardNormalCheck, PreserveTopology) are evaluated for DIAGNOSTICS ONLY: a
	// failing collapse is recorded/coloured as a QEM rejection but is STILL
	// performed, so the decimation reaches its target exactly as if the gates
	// were off.  When false, the gates veto (the faithful behaviour) — which can
	// stop the run early because forbidden collapses can never reach the target.
	bool   DiagnoseOnly          = false;

	// NOT a vcg parameter.  Face-count stop target (from the --nf CLI option).
	// When >= 0 it GOVERNS the run: collapses continue until the MAT face count
	// drops to this value (the vertex-derived maxCollapses bound is bypassed).
	// When < 0 (default) the run is driven by maxCollapses as before.
	int    FaceTarget            = -1;
};

class VcgQuadricSimplifier
{
public:
	explicit VcgQuadricSimplifier(SlabMesh& mesh) : m(mesh) {}

	// Runs the decimation, performing at most `maxCollapses` edge collapses
	// (matches SlabMesh::Simplify's deleteSphereNum < threshold semantics).
	// Returns the number of collapses actually performed.
	unsigned Run(int maxCollapses);

	// Tunable parameters (vcg defaults).  Edit before calling Run().
	VcgQuadricParameter params;

	// Per-collapse JSONL log (collapse_records_schema.md). Call before Run().
	void EnableCollapseLog(const std::string& path);

	// MeshLab AutoClean: drop null faces, duplicate + unreferenced verts.
	void PostSimplificationCleaning();

private:
	SlabMesh& m;

	// Per-vertex quadric and incremental mark, indexed by SlabMesh vertex id.
	// SlabMesh only ever appends vertices (never reuses freed ids), so these
	// stay aligned as long as they are grown to vertices.size().
	std::vector<VcgQuadric> vq;
	std::vector<int>        imark;
	int                     globalMark = 0;

	// One queued candidate collapse.  Mirrors a vcg local-modification object:
	// it carries the cached priority, the optimal target position computed when
	// it was created, and the localMark used by the staleness test.
	struct HeapElem
	{
		unsigned      v0 = 0, v1 = 0;
		double        pri = 0.0;
		int           localMark = 0;
		Wm4::Vector3d optPos;
	};
	std::vector<HeapElem> heap;

	// vcg HeapElem::operator< returns (pri > other.pri); with std::*_heap (a max
	// heap by operator<) the "largest" element is the smallest priority, so
	// pop_heap surfaces the cheapest collapse.  We replicate that ordering.
	static bool HeapLess(const HeapElem& x, const HeapElem& y) { return x.pri > y.pri; }

	// ── ported vcg pieces ──────────────────────────────────────────────────
	void          InitQuadrics();                              // InitQuadric
	Wm4::Vector3d ComputePosition(unsigned v0, unsigned v1);   // ComputePosition
	// ComputePriority.  When a hard-reject gate fires (returns +inf) and the
	// outReason/outPrims pointers are supplied, they are filled with the reason
	// and the offending geometry so the caller can record it for the viewer.
	double        ComputePriority(unsigned v0, unsigned v1, Wm4::Vector3d& outPos,
	                              QemRejectionReason*  outReason = nullptr,
	                              QemReasonPrimitives* outPrims  = nullptr);
	// CheckForFlip.  If outFlipped is supplied and a flip is detected, it is set
	// to the surviving face whose normal changed most: [0]=before, [1]=after.
	bool          CheckForFlip(unsigned v0, unsigned v1, const Wm4::Vector3d& newPos,
	                           std::array<std::array<std::array<double,3>,3>,2>* outFlipped = nullptr);
	bool          IsUpToDate(const HeapElem& e) const;         // IsUpToDate
	// IsFeasible.  If outPrims is supplied and the link condition fails, it is
	// filled with the offending one-ring geometry.
	bool          IsFeasible(unsigned v0, unsigned v1, QemReasonPrimitives* outPrims = nullptr) const;
	// LinkConditions — faithful port of vcg::tri::EdgeCollapser::LinkConditions
	// (vcglib/edge_collapse.h:133).  Returns true when the collapse SATISFIES the
	// link condition (i.e. is topologically feasible), exactly like vcg.  Operates
	// only on VF-style adjacency (the faces incident to each endpoint).
	bool          LinkConditions(unsigned v0, unsigned v1) const;
	// Faithful vcg collapse (EdgeCollapser::Do): v0 is deleted, v1 SURVIVES and
	// moves to optPos.  Returns the surviving vertex id (v1), or (unsigned)-1 on
	// failure.
	unsigned      Execute(unsigned v0, unsigned v1, const Wm4::Vector3d& optPos);
	void          UpdateHeap(unsigned vid_tgt);                // UpdateHeap
	void          ClearHeap();                                 // ClearHeap
	void          AddCollapseToHeap(unsigned v0, unsigned v1); // AddCollapseToHeap

	// ── QEM rejection recording (writes into SlabMesh's qem_edge_* maps) ─────
	// Resolve the slab edge id for (v0,v1) and store / erase its rejection.
	void          RecordRejection(unsigned v0, unsigned v1,
	                              QemRejectionReason reason, QemReasonPrimitives prims);
	void          ClearRejection(unsigned v0, unsigned v1);

	// ── small helpers over SlabMesh adjacency ──────────────────────────────
	void   EnsureSized();                       // grow vq/imark to vertices.size()
	bool   FaceVerts(unsigned fid, unsigned out[3]) const;  // sorted face verts
	double EdgeLen(unsigned v0, unsigned v1) const;
	// Our QEM must decimate the EXACT mesh MeshLab does: the exported
	// _mat_initial.off, whose coords are sphere.center * bb_diagonal_length
	// (ExportMatAsOff in main_cli.cpp).  SlabMesh stores centers at a different
	// normalization (its bbox diagonal = 1), so evaluating QEM at raw sphere.center
	// (diag 1) decimates a √2-smaller mesh than MeshLab sees (the .off, diag ~1.4).
	// vcg's QEM is scale-dependent (face quadrics are area-weighted, border quadrics
	// are not), so we evaluate Pos at the .off scale (× bb_diagonal_length) and map
	// the optimal position back (÷ bb_diagonal_length) when writing it in Execute.
	// qemScale is set in Run() = bb_diagonal_length; using the SAME value the export
	// used guarantees Pos == the .off coordinates, whatever that value is.
	double qemScale = 1.0;
	// MeshLab's mesh CoordType is float (MESHLAB_SCALAR=float); the quadric stays
	// double. Truncate coords to float so plane/mid/opt decisions match MeshLab.
	static Wm4::Vector3d ToFloat(const Wm4::Vector3d& p)
	{
		return Wm4::Vector3d((float)p.X(), (float)p.Y(), (float)p.Z());
	}
	// When the exported _mat_initial.off has been loaded (useOffCoords), Pos returns
	// the EXACT coordinate MeshLab reads (text → float), keyed by the OFF index.
	// Otherwise it falls back to ToFloat(center * qemScale).  MeshLab loads float
	// coords from 10-digit .off text, which does NOT round-trip to (float)(center*
	// qemScale); on flat edges that ~1e-8 position difference is exactly what flips
	// the near-zero Apply across the QuadricEpsilon clamp (see md_files/qem/check_coords.py).
	std::vector<Wm4::Vector3d> offCoords;   // indexed by OFF index (== slabToOff[v])
	bool                       useOffCoords = false;
	Wm4::Vector3d Pos(unsigned v) const
	{
		if (useOffCoords && v < slabToOff.size())
		{
			unsigned oi = slabToOff[v];
			if (oi < offCoords.size()) return offCoords[oi];
		}
		return ToFloat(m.vertices[v].second->sphere.center * qemScale);
	}

	// ── collapse logging ────────────────────────────────────────────────────
	std::string           collapseLogPath;
	bool                  collapseLogEnabled = false;
	unsigned              collapseIdx = 0;
	std::ofstream         collapseLog;
	std::vector<unsigned> slabToOff;       // slab vid -> _mat_initial.off index
	void BuildOffIndexMap();
	bool LoadOffCoords();                  // load _mat_initial.off coords (float) into offCoords
	void LogCollapse(unsigned v0, unsigned v1, double cost, const Wm4::Vector3d& newPos);
	void WriteInitialHeapState();  // initial_heap_state_schema.md
	void WriteVertexQuadrics();    // per-vertex InitQuadric quadrics (a[6],b[3],c) for cross-check
	void WriteParams();            // final VcgQuadricParameter -> <prefix>_qem_params.json
};

#endif  // ONLY_USE_QEM_CONDITION_CHECKS
#endif  // _VCG_QUADRIC_SIMPLIFIER_H
