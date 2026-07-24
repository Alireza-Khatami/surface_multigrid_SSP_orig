// VcgDirectSimplifier — see VcgDirectSimplifier.h.
//
// Recipe taken verbatim from vcglib/apps/tridecimator/tridecimator.cpp:
//   - minimal vcg::TriMesh<MyVertex, MyFace> with VFAdj + per-vertex Quadric
//   - vcg::tri::TriEdgeCollapseQuadric<MyMesh, BasicVertexPair, MyEdgeCollapse,
//                                    QInfoStandard<MyVertex>>
//   - vcg::LocalOptimization<MyMesh>::Init / DoOptimization / Finalize
//
// On top of the canonical recipe we maintain shadow tables (g_vertex_shadow,
// g_edge_shadow, g_face_shadow) tracking QMAT-side per-primitive attributes —
// cluster_type, struct_ids, topo flags / topo_type, face struct_id — across
// collapses.  Merge rules mirror SlabMesh::MergeVertices (vertex struct_ids
// union, topo flags OR/AND per its formula) and SlabEdge::CombineTopoType.
//
// We deliberately keep all vcg includes INSIDE this .cpp so the rest of QMAT
// (which uses Wm4 / CGAL) doesn't see them.

#include "VcgDirectSimplifier.h"

#include "SlabMesh.h"
#include "Mesh.h"      // for pmesh->bb_diagonal_length

#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <unordered_map>
#include <utility>

// ── vcglib ──────────────────────────────────────────────────────────────────
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/edge_collapse.h>        // EdgeCollapser::LinkConditions
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/flag.h>
#include <vcg/complex/algorithms/update/normal.h>

// ─────────────────────────────────────────────────────────────────────────────
// Minimal vcg::TriMesh definition — same as vcglib/apps/tridecimator.
// Vertex stores the Quadric inline; QInfoStandard wraps Qd() so TECQ can find it.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

class VDVertex;
class VDEdge;
class VDFace;

struct VDUsedTypes : public vcg::UsedTypes<
	vcg::Use<VDVertex>::AsVertexType,
	vcg::Use<VDEdge  >::AsEdgeType,
	vcg::Use<VDFace  >::AsFaceType> {};

class VDVertex : public vcg::Vertex<VDUsedTypes,
	vcg::vertex::VFAdj,
	vcg::vertex::Coord3f,
	vcg::vertex::Normal3f,
	vcg::vertex::Mark,
	vcg::vertex::Qualityf,
	vcg::vertex::BitFlags>
{
public:
	vcg::math::Quadric<double>& Qd() { return q; }
private:
	vcg::math::Quadric<double> q;
};

class VDEdge : public vcg::Edge<VDUsedTypes> {};

class VDFace : public vcg::Face<VDUsedTypes,
	vcg::face::VFAdj,
	vcg::face::VertexRef,
	vcg::face::Normal3f,
	vcg::face::BitFlags> {};

class VDMesh : public vcg::tri::TriMesh<std::vector<VDVertex>, std::vector<VDFace>> {};

typedef vcg::tri::BasicVertexPair<VDVertex> VDVertexPair;

// ─────────────────────────────────────────────────────────────────────────────
// Shadow tables — track QMAT slab attributes alongside the vcg mesh.
// Indexed by vcg-vid / canonical (min,max) vcg-vid pair / vcg-fid.  TU-static
// because vcg constructs VDEdgeCollapse instances internally and we have no
// per-instance state to attach.  Set up in RunVcgDirectSimplify, mutated by
// VDEdgeCollapse::Execute, read by ExtractSnapshot.
// ─────────────────────────────────────────────────────────────────────────────

struct VertexShadow {
	uint8_t      cluster_type = 0;          // SlabVertex::ClusterType
	std::set<int> struct_ids;
	bool         is_sheet     = false;
	bool         is_seam      = false;
	bool         is_junction  = false;
	bool         is_boundary  = false;
	// SlabVertex::sphere.radius * bb_diagonal_length — survivor wins on merge.
	double       radius       = 0.0;
	// Initial-MAT vids whose lineage folded into this surviving vertex.
	// Seeded {slab_vid} in BuildFromSlab, unioned in MergeVertexShadow.
	std::set<unsigned> original_ancestors;
};

struct EdgeShadow {
	uint8_t       topo_type = 0;            // SlabEdge::TopoType
	std::set<int> struct_ids;
};

struct FaceShadow {
	int struct_id = -1;
};

using EdgeKey = std::pair<int,int>;
struct EdgeKeyHash {
	size_t operator()(EdgeKey k) const noexcept {
		return std::hash<long long>{}(((long long)k.first << 32) ^ (long long)(uint32_t)k.second);
	}
};

inline EdgeKey canonical(int a, int b) { return (a < b) ? EdgeKey{a,b} : EdgeKey{b,a}; }

std::vector<VertexShadow>                              g_vertex_shadow;
std::unordered_map<EdgeKey, EdgeShadow, EdgeKeyHash>   g_edge_shadow;
std::vector<FaceShadow>                                g_face_shadow;
// Initial-MAT positions (scaled), captured once in BuildFromSlab; copied into
// every snapshot so click-handler can map ancestor ids to coords.
std::vector<std::array<double,3>>                      g_original_positions;

// Last rejection record per canonical edge key.  Written by VDEdgeCollapse's
// IsFeasible override when a candidate fails; read by ExtractSnapshot to fill
// per-edge rejection fields on the snapshot.  Last-rejection-wins.
struct RejectionRecord {
	uint8_t reason       = 255;   // SlabMesh::RejectionReason; 255 = none
	std::array<double,3> target = {0.0, 0.0, 0.0};   // optimal target pos
};
std::unordered_map<EdgeKey, RejectionRecord, EdgeKeyHash> g_edge_last_rejection;

// Live-update plumbing (snapshot+callback live alongside the shadow tables).
LiveUpdateCallback   g_live_cb;
VcgDirectSnapshot    g_live_snap;
VDMesh*              g_mesh = nullptr;

// VDE-specific gate toggles, snapshotted from VcgDirectParams at run start
// so IsFeasible can consult them without touching vcg's BaseParameterClass.
struct ExtraGates {
	bool euler_check      = true;
	bool same_topo_check  = true;
	bool struct_ids_check = true;
};
ExtraGates g_extra_gates;

// ─────────────────────────────────────────────────────────────────────────────
// Forward decls / merge helpers.
// ─────────────────────────────────────────────────────────────────────────────

void ExtractSnapshot(VDMesh& mesh, VcgDirectSnapshot& out);

// SlabEdge::CombineTopoType, duplicated here because the header version is a
// static method on SlabEdge and we only want uint8_t↔uint8_t for the shadow.
uint8_t CombineEdgeTopoType(uint8_t a, uint8_t b)
{
	using TT = SlabEdge::TopoType;
	return static_cast<uint8_t>(SlabEdge::CombineTopoType(
		static_cast<TT>(a), static_cast<TT>(b)));
}

// Union sv1+sv2 into svt following SlabMesh::MergeVertices' rules:
// struct_ids = union (CanMerge would gate equality, but we don't gate yet),
// cluster_type = survivor (sv1) wins,
// topo flags: OR for seam/junction/boundary, sheet = both-sheets && !seam && !boundary.
VertexShadow MergeVertexShadow(const VertexShadow& s, const VertexShadow& d)
{
	VertexShadow out = s;
	out.struct_ids.insert(d.struct_ids.begin(), d.struct_ids.end());
	out.is_seam     = s.is_seam     || d.is_seam;
	out.is_junction = s.is_junction || d.is_junction;
	out.is_boundary = s.is_boundary || d.is_boundary;
	out.is_sheet    = s.is_sheet    && d.is_sheet
	                  && !out.is_seam && !out.is_boundary;
	// Ancestor union — same rule QMAT's MergeVertices uses on SlabVertex.
	out.original_ancestors.insert(d.original_ancestors.begin(),
	                              d.original_ancestors.end());
	return out;
}

// When two parent edges (s,w) and (d,w) fold into one survivor edge (s,w)
// after the d→s collapse — diamond case in SlabMesh::MergeVertices.
EdgeShadow MergeEdgeShadow(const EdgeShadow& a, const EdgeShadow& b)
{
	EdgeShadow out = a;
	out.struct_ids.insert(b.struct_ids.begin(), b.struct_ids.end());
	out.topo_type = CombineEdgeTopoType(a.topo_type, b.topo_type);
	return out;
}

// Collect the 1-ring (vcg-vid set) of a vertex by walking incident faces via
// VFAdj.  Used to snapshot a dying vertex's neighbours pre-collapse so we can
// re-key shadow edges after vcg rewrites face vertex refs.
std::set<int> CollectOneRing(VDMesh& m, int vid)
{
	std::set<int> out;
	if (vid < 0 || vid >= (int)m.vert.size()) return out;
	VDVertex* v = &m.vert[vid];
	if (v->IsD()) return out;
	vcg::face::VFIterator<VDFace> it(v);
	while (!it.End()) {
		VDFace* f = it.F();
		if (f && !f->IsD()) {
			for (int k = 0; k < 3; ++k) {
				VDVertex* nv = f->V(k);
				if (!nv || nv == v) continue;
				int nvi = (int)vcg::tri::Index(m, *nv);
				if (nvi != vid) out.insert(nvi);
			}
		}
		++it;
	}
	return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// VDEdgeCollapse — base TECQ + Execute override that drives the shadow merge
// and fires the live-update callback.
// ─────────────────────────────────────────────────────────────────────────────

class VDEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric<
	VDMesh, VDVertexPair, VDEdgeCollapse, vcg::tri::QInfoStandard<VDVertex>>
{
public:
	typedef vcg::tri::TriEdgeCollapseQuadric<
		VDMesh, VDVertexPair, VDEdgeCollapse, vcg::tri::QInfoStandard<VDVertex>> TECQ;
	inline VDEdgeCollapse(const VDVertexPair& p, int i, vcg::BaseParameterClass* pp) : TECQ(p, i, pp) {}

	// Override IsFeasible to record vcg's hard rejects + layer our own QMAT
	// gates on top.  vcg's base IsFeasible only does LinkConditions (when
	// PreserveTopology=true); QualityCheck / NormalCheck are PENALTIES inside
	// ComputePriority, not rejections.
	// Gates applied in order:
	//   (1) LinkConditions      → NonManifold_LinkCondition
	//   (2) Δχ = S - F_inc      → EulerCharacteristicChange   (catches hole closure)
	//   (3) endpoint topo type  → DifferentClusterType
	//   (4) struct_ids equality → struct_ids_sets_different
	bool IsFeasible(vcg::BaseParameterClass* _pp)
	{
		using QP = vcg::tri::TriEdgeCollapseQuadricParameter;
		QP* pp = static_cast<QP*>(_pp);

		const int ia = (int)vcg::tri::Index(*g_mesh, *this->pos.V(0));
		const int ib = (int)vcg::tri::Index(*g_mesh, *this->pos.V(1));
		const EdgeKey k = canonical(ia, ib);

		auto recordReject = [&](SlabMesh::RejectionReason r) {
			RejectionRecord rec;
			rec.reason = static_cast<uint8_t>(r);
			// Optimal placement (mid-point fallback) for the Rejection Target marker.
			this->ComputePosition(_pp);
			rec.target = { (double)this->optimalPos[0],
			               (double)this->optimalPos[1],
			               (double)this->optimalPos[2] };
			g_edge_last_rejection[k] = rec;
		};

		// (1) vcg's only hard reject: LinkConditions (when PreserveTopology).
		if (pp->PreserveTopology) {
			if (!vcg::tri::EdgeCollapser<VDMesh, VDVertexPair>::LinkConditions(this->pos)) {
				recordReject(SlabMesh::RejectionReason::NonManifold_LinkCondition);
				return false;
			}
		}

		// (2) Direct Euler-characteristic delta: Δχ = S - F_inc, where
		//     S     = |N(v0) ∩ N(v1)|   (common active neighbours)
		//     F_inc = # active faces incident to the collapsing edge.
		// A valid edge collapse on a 2-manifold has Δχ = 0; any non-zero value
		// means the collapse would change topology (close/open a hole, merge
		// boundary loops, etc.).  Layered on top of LinkConditions because the
		// MAT input is non-manifold and vcg's boundary-dummy logic can let
		// hole-closing collapses slip through.  Independent toggle (--qem-euler-check).
		if (g_extra_gates.euler_check) {
			VDVertex* a = this->pos.V(0);
			VDVertex* b = this->pos.V(1);
			int F_inc = 0;
			std::set<VDVertex*> Na, Nb;
			for (vcg::face::VFIterator<VDFace> vfi(a); !vfi.End(); ++vfi) {
				VDFace* f = vfi.F();
				if (!f || f->IsD()) continue;
				bool touches_b = false;
				for (int kk = 0; kk < 3; ++kk) {
					VDVertex* nv = f->V(kk);
					if (nv == b) touches_b = true;
					else if (nv != a) Na.insert(nv);
				}
				if (touches_b) ++F_inc;
			}
			for (vcg::face::VFIterator<VDFace> vfi(b); !vfi.End(); ++vfi) {
				VDFace* f = vfi.F();
				if (!f || f->IsD()) continue;
				for (int kk = 0; kk < 3; ++kk) {
					VDVertex* nv = f->V(kk);
					if (nv != a && nv != b) Nb.insert(nv);
				}
			}
			int S = 0;
			for (VDVertex* v : Na) if (Nb.count(v)) ++S;
			if (S - F_inc != 0) {
				recordReject(SlabMesh::RejectionReason::EulerCharacteristicChange);
				return false;
			}
		}

		// (3) Endpoint topo (cluster_type) equality — QMAT's CanMerge cond 2.
		// Independent toggle (--qem-same-topo-check).
		if (g_extra_gates.same_topo_check &&
		    ia >= 0 && ia < (int)g_vertex_shadow.size() &&
		    ib >= 0 && ib < (int)g_vertex_shadow.size())
		{
			if (g_vertex_shadow[ia].cluster_type != g_vertex_shadow[ib].cluster_type) {
				recordReject(SlabMesh::RejectionReason::DifferentClusterType);
				return false;
			}
		}

		// (4) struct_ids equality: edge.struct_ids must equal both endpoints' sets.
		// Independent toggle (--qem-struct-ids-check).
		if (g_extra_gates.struct_ids_check) {
			auto it = g_edge_shadow.find(k);
			if (it != g_edge_shadow.end() && !it->second.struct_ids.empty()) {
				const auto& eids = it->second.struct_ids;
				bool match =
					ia >= 0 && ia < (int)g_vertex_shadow.size() &&
					ib >= 0 && ib < (int)g_vertex_shadow.size() &&
					g_vertex_shadow[ia].struct_ids == eids &&
					g_vertex_shadow[ib].struct_ids == eids;
				if (!match) {
					recordReject(SlabMesh::RejectionReason::struct_ids_sets_different);
					return false;
				}
			}
		}

		// Feasible: clear any prior rejection so a re-attempt that succeeds
		// doesn't leave a stale reason colour in the viewer.
		g_edge_last_rejection.erase(k);
		return true;
	}

	// LocalModification::Execute is virtual — override to: (1) snapshot 1-rings
	// of both endpoints before the collapse rewrites face vertex refs, (2) call
	// base, (3) detect survivor/dead by IsD(), (4) merge shadow state, (5) fire
	// live hook.
	void Execute(VDMesh& m, vcg::BaseParameterClass* pp)
	{
		const int ia = (int)vcg::tri::Index(m, *this->pos.V(0));
		const int ib = (int)vcg::tri::Index(m, *this->pos.V(1));

		// (1) Pre-collapse neighbour snapshots.
		std::set<int> nbrs_a = CollectOneRing(m, ia);
		std::set<int> nbrs_b = CollectOneRing(m, ib);

		// (2) Base collapse — vcg marks one endpoint IsD and re-points incident
		//     faces' VertexRef to the survivor.
		TECQ::Execute(m, pp);

		// (3) Detect survivor / dead dynamically; don't rely on V(0)/V(1) order.
		const bool a_dead = this->pos.V(0)->IsD();
		const bool b_dead = this->pos.V(1)->IsD();
		if (a_dead == b_dead) {
			// Should never happen — exactly one endpoint must die.
			std::cerr << "[VcgDirectSimplify] WARN: collapse left both endpoints "
			          << (a_dead ? "dead" : "alive") << " for edge (" << ia << "," << ib << ")\n";
		}
		const int s = a_dead ? ib : ia;
		const int d = a_dead ? ia : ib;
		const std::set<int>& nbrs_d = a_dead ? nbrs_a : nbrs_b;

		// (4a) Re-key every (d,w) shadow edge to (s,w); merge into existing
		//      (s,w) on collision (the diamond case).
		for (int w : nbrs_d) {
			if (w == s) continue;                  // the collapsed edge itself
			const EdgeKey old_key = canonical(d, w);
			auto it = g_edge_shadow.find(old_key);
			if (it == g_edge_shadow.end()) continue;
			EdgeShadow data = std::move(it->second);
			g_edge_shadow.erase(it);

			const EdgeKey new_key = canonical(s, w);
			auto [hit, inserted] = g_edge_shadow.try_emplace(new_key, std::move(data));
			if (!inserted) hit->second = MergeEdgeShadow(hit->second, data);
		}
		// Erase the collapsed edge (s,d) entry itself.
		g_edge_shadow.erase(canonical(ia, ib));
		g_edge_last_rejection.erase(canonical(ia, ib));

		// (4b) Merge vertex shadow into survivor.  Leave shadow[d] in place
		//      (harmless leak — vcg never recycles vertex slots, so (d,*) keys
		//      are unreachable from any surviving primitive).
		if (s >= 0 && s < (int)g_vertex_shadow.size() &&
		    d >= 0 && d < (int)g_vertex_shadow.size())
			g_vertex_shadow[s] = MergeVertexShadow(g_vertex_shadow[s], g_vertex_shadow[d]);

		// (5) Live hook.
		if (g_live_cb) {
			ExtractSnapshot(m, g_live_snap);
			g_live_cb(g_live_snap);
		}
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// BuildFromSlab — copy active slab vertices/faces into the vcg mesh AND seed
// the shadow tables from slab-side attributes.
// ─────────────────────────────────────────────────────────────────────────────

void BuildFromSlab(const SlabMesh& sm, VDMesh& mesh, int& nv_active, int& nf_tri)
{
	nv_active = 0;
	nf_tri    = 0;

	g_vertex_shadow.clear();
	g_edge_shadow.clear();
	g_face_shadow.clear();
	g_original_positions.clear();
	g_edge_last_rejection.clear();

	const double scale = sm.pmesh ? sm.pmesh->bb_diagonal_length : 1.0;

	// Capture initial-MAT positions (scaled) once for ancestor lookups.
	// Source is slab's pre-simplification original_positions; fall back to the
	// current sphere centers if the slab didn't populate it (shouldn't happen
	// for vcg-direct, which always runs on the unsimplified slab).
	if (!sm.original_positions.empty()) {
		g_original_positions.reserve(sm.original_positions.size());
		for (const auto& p : sm.original_positions)
			g_original_positions.push_back({ p[0]*scale, p[1]*scale, p[2]*scale });
	} else {
		g_original_positions.reserve(sm.vertices.size());
		for (size_t i = 0; i < sm.vertices.size(); ++i) {
			if (!sm.vertices[i].first || !sm.vertices[i].second) {
				g_original_positions.push_back({0,0,0});
				continue;
			}
			const auto& c = sm.vertices[i].second->sphere.center;
			g_original_positions.push_back({ c[0]*scale, c[1]*scale, c[2]*scale });
		}
	}

	// Slab vertex idx -> compact vcg vertex idx (-1 if vertex is deleted).
	std::vector<int> slabToVcg(sm.vertices.size(), -1);
	for (size_t i = 0; i < sm.vertices.size(); ++i)
		if (sm.vertices[i].first) ++nv_active;

	for (size_t i = 0; i < sm.faces.size(); ++i)
		if (sm.faces[i].first && sm.faces[i].second &&
		    sm.faces[i].second->vertices_.size() == 3)
			++nf_tri;

	if (nv_active == 0 || nf_tri == 0) return;

	g_vertex_shadow.resize(nv_active);
	g_face_shadow.reserve(nf_tri);

	auto vi = vcg::tri::Allocator<VDMesh>::AddVertices(mesh, nv_active);
	int written = 0;
	for (size_t i = 0; i < sm.vertices.size(); ++i)
	{
		if (!sm.vertices[i].first) continue;
		slabToVcg[i] = written;
		const SlabVertex* sv = sm.vertices[i].second;
		const auto& c = sv->sphere.center;
		(*vi).P() = VDMesh::CoordType(
			(float)(c[0] * scale),
			(float)(c[1] * scale),
			(float)(c[2] * scale));
		// Seed vertex shadow.
		VertexShadow& vs = g_vertex_shadow[written];
		vs.cluster_type = static_cast<uint8_t>(sv->nmn_cluster_type);
		vs.struct_ids   = sv->struct_ids;
		vs.is_sheet     = sv->topo_is_sheet;
		vs.is_seam      = sv->topo_is_seam;
		vs.is_junction  = sv->topo_is_junction;
		vs.is_boundary  = sv->topo_is_boundary;
		vs.radius       = sv->sphere.radius * scale;
		// Seed ancestors: copy slab's lineage if present, else {slab_vid}.
		if (!sv->original_ancestors.empty())
			vs.original_ancestors = sv->original_ancestors;
		else
			vs.original_ancestors.insert((unsigned)i);
		++written;
		++vi;
	}

	auto fi = vcg::tri::Allocator<VDMesh>::AddFaces(mesh, nf_tri);
	for (size_t i = 0; i < sm.faces.size(); ++i)
	{
		if (!sm.faces[i].first || !sm.faces[i].second) continue;
		const SlabFace* sf = sm.faces[i].second;
		const auto& vs = sf->vertices_;
		if (vs.size() != 3) continue;
		auto it = vs.begin();
		int a = slabToVcg[*it++];
		int b = slabToVcg[*it++];
		int c = slabToVcg[*it];
		if (a < 0 || b < 0 || c < 0) continue;
		(*fi).V(0) = &mesh.vert[a];
		(*fi).V(1) = &mesh.vert[b];
		(*fi).V(2) = &mesh.vert[c];
		FaceShadow fs;
		fs.struct_id = sf->struct_id;
		g_face_shadow.push_back(fs);
		++fi;
	}

	// Seed edge shadow from active slab edges.  Edges whose endpoints are also
	// active map directly to a vcg edge key; orphan edges (endpoint deleted)
	// are skipped.
	for (size_t i = 0; i < sm.edges.size(); ++i) {
		if (!sm.edges[i].first || !sm.edges[i].second) continue;
		const SlabEdge* se = sm.edges[i].second;
		const unsigned ev0 = se->vertices_.first;
		const unsigned ev1 = se->vertices_.second;
		if (ev0 >= slabToVcg.size() || ev1 >= slabToVcg.size()) continue;
		const int a = slabToVcg[ev0];
		const int b = slabToVcg[ev1];
		if (a < 0 || b < 0) continue;
		EdgeShadow es;
		es.topo_type  = static_cast<uint8_t>(se->topo_type);
		es.struct_ids = se->struct_ids;
		g_edge_shadow.emplace(canonical(a, b), std::move(es));
	}

	// Topology bookkeeping vcg needs before LocalOptimization::Init.
	vcg::tri::UpdateBounding<VDMesh>::Box(mesh);
	vcg::tri::UpdateTopology<VDMesh>::VertexFace(mesh);
	vcg::tri::UpdateFlags<VDMesh>::FaceBorderFromVF(mesh);
	vcg::tri::UpdateNormal<VDMesh>::PerFaceNormalized(mesh);
}

// ─────────────────────────────────────────────────────────────────────────────
// ExtractSnapshot — walk surviving vertices/faces, fill snapshot arrays from
// shadow tables.  Edges are derived from face vertex pairs (deduped via set).
// ─────────────────────────────────────────────────────────────────────────────

inline int FirstStructId(const std::set<int>& ids)
{
	return ids.empty() ? -1 : *ids.begin();
}

inline uint8_t PackVertexTopoFlags(const VertexShadow& vs)
{
	return (uint8_t)((vs.is_sheet    ? 1 : 0)
	             |  (vs.is_seam     ? 2 : 0)
	             |  (vs.is_junction ? 4 : 0)
	             |  (vs.is_boundary ? 8 : 0));
}

void ExtractSnapshot(VDMesh& mesh, VcgDirectSnapshot& out)
{
	out.vertices.clear();
	out.faces.clear();
	out.edges.clear();
	out.vertex_cluster_type.clear();
	out.vertex_first_struct_id.clear();
	out.vertex_struct_ids.clear();
	out.vertex_topo_flags.clear();
	out.vertex_radius.clear();
	out.face_struct_id.clear();
	out.edge_topo_type.clear();
	out.edge_first_struct_id.clear();
	out.edge_struct_ids.clear();
	out.edge_struct_match.clear();
	out.edge_last_rejection.clear();
	out.edge_rejection_target.clear();
	out.vertex_original_ancestors.clear();
	// Initial-MAT positions are run-invariant; copy once per snapshot.
	out.original_positions = g_original_positions;

	std::vector<int> vcgToOut(mesh.vert.size(), -1);
	int next = 0;
	out.vertices.reserve(mesh.VN());
	out.vertex_cluster_type.reserve(mesh.VN());
	out.vertex_first_struct_id.reserve(mesh.VN());
	out.vertex_struct_ids.reserve(mesh.VN());
	out.vertex_topo_flags.reserve(mesh.VN());
	out.vertex_radius.reserve(mesh.VN());
	out.vertex_original_ancestors.reserve(mesh.VN());
	for (size_t i = 0; i < mesh.vert.size(); ++i)
	{
		if (mesh.vert[i].IsD()) continue;
		vcgToOut[i] = next++;
		const auto& p = mesh.vert[i].P();
		out.vertices.push_back({ (double)p[0], (double)p[1], (double)p[2] });
		if (i < g_vertex_shadow.size()) {
			const VertexShadow& vs = g_vertex_shadow[i];
			out.vertex_cluster_type.push_back(vs.cluster_type);
			out.vertex_first_struct_id.push_back(FirstStructId(vs.struct_ids));
			out.vertex_struct_ids.emplace_back(
				vs.struct_ids.begin(), vs.struct_ids.end());
			out.vertex_topo_flags.push_back(PackVertexTopoFlags(vs));
			out.vertex_radius.push_back(vs.radius);
			out.vertex_original_ancestors.emplace_back(
				vs.original_ancestors.begin(), vs.original_ancestors.end());
		} else {
			out.vertex_cluster_type.push_back(0);
			out.vertex_first_struct_id.push_back(-1);
			out.vertex_struct_ids.emplace_back();
			out.vertex_topo_flags.push_back(0);
			out.vertex_radius.push_back(0.0);
			out.vertex_original_ancestors.emplace_back();
		}
	}

	out.faces.reserve(mesh.FN());
	out.face_struct_id.reserve(mesh.FN());
	for (size_t i = 0; i < mesh.face.size(); ++i)
	{
		if (mesh.face[i].IsD()) continue;
		const size_t i0 = vcg::tri::Index(mesh, *mesh.face[i].V(0));
		const size_t i1 = vcg::tri::Index(mesh, *mesh.face[i].V(1));
		const size_t i2 = vcg::tri::Index(mesh, *mesh.face[i].V(2));
		out.faces.push_back({ vcgToOut[i0], vcgToOut[i1], vcgToOut[i2] });
		out.face_struct_id.push_back(i < g_face_shadow.size() ? g_face_shadow[i].struct_id : -1);
	}

	// Derive edges from face vertex pairs (deduped).
	std::set<EdgeKey> seen;
	for (size_t i = 0; i < mesh.face.size(); ++i) {
		if (mesh.face[i].IsD()) continue;
		const int i0 = (int)vcg::tri::Index(mesh, *mesh.face[i].V(0));
		const int i1 = (int)vcg::tri::Index(mesh, *mesh.face[i].V(1));
		const int i2 = (int)vcg::tri::Index(mesh, *mesh.face[i].V(2));
		EdgeKey k0 = canonical(i0, i1);
		EdgeKey k1 = canonical(i1, i2);
		EdgeKey k2 = canonical(i2, i0);
		for (const EdgeKey& k : { k0, k1, k2 }) {
			if (!seen.insert(k).second) continue;
			out.edges.push_back({ vcgToOut[k.first], vcgToOut[k.second] });
			auto it = g_edge_shadow.find(k);
			if (it == g_edge_shadow.end()) {
				out.edge_topo_type.push_back(0);
				out.edge_first_struct_id.push_back(-1);
				out.edge_struct_ids.emplace_back();
				out.edge_struct_match.push_back(0);
			} else {
				out.edge_topo_type.push_back(it->second.topo_type);
				out.edge_first_struct_id.push_back(FirstStructId(it->second.struct_ids));
				out.edge_struct_ids.emplace_back(
					it->second.struct_ids.begin(), it->second.struct_ids.end());
				// Struct-collapsible: edge has non-empty struct_ids AND both
				// endpoints' struct_ids equal the edge's set exactly.  Matches
				// the existing "Structure Collapsible" colouring in main_cli's
				// BuildMatArrays (see line ~603).
				uint8_t match = 0;
				if (!it->second.struct_ids.empty() &&
				    k.first  < (int)g_vertex_shadow.size() &&
				    k.second < (int)g_vertex_shadow.size() &&
				    g_vertex_shadow[k.first ].struct_ids == it->second.struct_ids &&
				    g_vertex_shadow[k.second].struct_ids == it->second.struct_ids)
					match = 1;
				out.edge_struct_match.push_back(match);
			}

			// Per-edge rejection (last-attempt reason + target).  255 = no record.
			auto rit = g_edge_last_rejection.find(k);
			if (rit == g_edge_last_rejection.end()) {
				out.edge_last_rejection.push_back(255);
				out.edge_rejection_target.push_back({0.0, 0.0, 0.0});
			} else {
				out.edge_last_rejection.push_back(rit->second.reason);
				out.edge_rejection_target.push_back(rit->second.target);
			}
		}
	}
}

} // anonymous namespace

bool RunVcgDirectSimplify(const SlabMesh& sm,
                          const VcgDirectParams& params,
                          VcgDirectResult& out,
                          LiveUpdateCallback live_callback)
{
	VDMesh mesh;
	int nv_active = 0, nf_tri = 0;
	BuildFromSlab(sm, mesh, nv_active, nf_tri);

	out.initial_vertex_count = nv_active;
	out.initial_face_count   = nf_tri;

	if (mesh.FN() == 0)
	{
		std::cerr << "[VcgDirectSimplify] no triangular faces; nothing to do\n";
		return false;
	}

	// Map our params onto vcg's TriEdgeCollapseQuadricParameter.  Defaults
	// for unspecified knobs come from the struct itself, so anything we don't
	// touch matches vcg's built-in defaults.
	vcg::tri::TriEdgeCollapseQuadricParameter qparams;
	qparams.QualityThr             = params.QualityThr;
	qparams.BoundaryQuadricWeight  = params.BoundaryQuadricWeight * 0.5; // UI -> internal
	qparams.OptimalPlacement       = params.OptimalPlacement;
	qparams.PreserveTopology       = params.PreserveTopology;
	qparams.NormalCheck            = params.NormalCheck;
	if (qparams.NormalCheck)
		qparams.NormalThrRad = vcg::math::ToRad(45.0);   // MeshLab default (M_PI/4)

	// VDE-specific gate toggles, read by IsFeasible.
	g_extra_gates.euler_check      = params.EulerCheck;
	g_extra_gates.same_topo_check  = params.SameTopoCheck;
	g_extra_gates.struct_ids_check = params.StructIdsCheck;

	// Resolve target face count.  Half the current face count is a reasonable
	// fallback when the caller passes neither a face nor a vertex target.
	int target_faces = params.TargetFaceNum;
	if (target_faces < 0 && params.TargetVertexNum > 0)
		target_faces = std::max(1, params.TargetVertexNum * 2);  // V≈F/2 for closed meshes
	if (target_faces < 0)
		target_faces = std::max(1, mesh.FN() / 2);
	target_faces = std::min(target_faces, mesh.FN());

	std::cerr << "[VcgDirectSimplify] V=" << mesh.VN() << " F=" << mesh.FN()
	          << " -> target faces " << target_faces
	          << " | OptimalPlacement=" << qparams.OptimalPlacement
	          << " PreserveTopology="   << qparams.PreserveTopology
	          << " NormalCheck="        << qparams.NormalCheck
	          << " QualityThr="         << qparams.QualityThr
	          << " BoundaryQuadricWeight=" << qparams.BoundaryQuadricWeight
	          << " EulerCheck="         << g_extra_gates.euler_check
	          << " SameTopoCheck="      << g_extra_gates.same_topo_check
	          << " StructIdsCheck="     << g_extra_gates.struct_ids_check
	          << " | shadow: V=" << g_vertex_shadow.size()
	          << " E=" << g_edge_shadow.size()
	          << " F=" << g_face_shadow.size()
	          << "\n";

	// Install the per-collapse hook for VDEdgeCollapse::Execute to call.
	g_live_cb = std::move(live_callback);
	g_mesh    = &mesh;

	vcg::LocalOptimization<VDMesh> DeciSession(mesh, &qparams);
	DeciSession.Init<VDEdgeCollapse>();
	std::cerr << "[VcgDirectSimplify] initial heap size " << DeciSession.h.size() << "\n";

	DeciSession.SetTargetSimplices(target_faces);
	DeciSession.SetTimeBudget(0.5f);

	while (DeciSession.DoOptimization() && mesh.FN() > target_faces) { /* iterate */ }

	DeciSession.Finalize<VDEdgeCollapse>();
	out.collapses_performed = DeciSession.nPerformedOps;

	// Clear hooks so we don't dangle into the next call.
	g_live_cb = nullptr;
	g_mesh    = nullptr;

	ExtractSnapshot(mesh, out.snapshot);
	out.final_vertex_count = (int)out.snapshot.vertices.size();
	out.final_face_count   = (int)out.snapshot.faces.size();

	std::cerr << "[VcgDirectSimplify] done V=" << out.final_vertex_count
	          << " F=" << out.final_face_count
	          << " collapses=" << out.collapses_performed
	          << " | shadow leftover: V=" << g_vertex_shadow.size()
	          << " E=" << g_edge_shadow.size()
	          << " F=" << g_face_shadow.size()
	          << "\n";
	return true;
}
