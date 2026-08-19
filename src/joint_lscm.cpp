#include "joint_lscm.h"

// All DC-related log messages go to dc_log.txt in the working directory.
static FILE* dc_log() {
    static FILE* f = fopen("dc_log.txt", "w");
    return f ? f : stderr;
}

void build_double_cover_faces(
    const Eigen::MatrixXi & F,
    Eigen::MatrixXi & F_dc)
{
    int nF = F.rows();
    F_dc.resize(2 * nF, 3);
    F_dc.topRows(nF) = F;
    for (int r = 0; r < nF; r++) {
        F_dc(nF + r, 0) = F(r, 0);
        F_dc(nF + r, 1) = F(r, 2);  // swap 1↔2 to reverse winding
        F_dc(nF + r, 2) = F(r, 1);
    }
}

void joint_lscm_double_cover(
    const Eigen::MatrixXd & V_pre,
    const Eigen::MatrixXi & FUV_pre,
    const Eigen::MatrixXd & V_post,
    const Eigen::MatrixXi & FUV_post,
    const int & vi,
    const int & vj,
    const std::set<int> & mesh_bd_verts,  // currently unused; caller passes {vi,vj}
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
    int nVjoint = nV + 1;   // slot nV = vi's post-collapse 3D position

    // --- Build Vjoint: rows 0..nV-1 from V_pre, row nV from V_post.row(vi) ---
    MatrixXd Vjoint(nVjoint, 3);
    for (int i = 0; i < nV; i++) Vjoint.row(i) = V_pre.row(i);
    Vjoint.row(nV) = V_post.row(vi);

    // --- Remap vi → nV in post face connectivity ---
    MatrixXi Fjoint_pre  = FUV_pre;
    MatrixXi Fjoint_post = FUV_post;
    for (int r = 0; r < Fjoint_post.rows(); r++)
        for (int c = 0; c < 3; c++)
            if (Fjoint_post(r, c) == vi) Fjoint_post(r, c) = nV;

    // =========================================================================
    // DC construction: reflected-B double cover
    //
    // The one-ring boundary loop has two arcs:
    //   boundary arc : vi → vj   (the collapsing edge, on the actual mesh boundary)
    //   interior arc : vj → B1 → B2 → … → Bn → vi   (B = mesh-interior vertices)
    //
    // Glued (shared between DC top and bottom sheets): vi, vj, B_arc[0], B_arc.back()
    //   → after gluing, vi and vj become TRUE interior vertices of the combined mesh.
    //
    // Duplicated with separate UV slots:  B_arc[1..end-1]  (the "middle" B vertices)
    //   → B_top stays at original index; B_bot gets a new index nVjoint+k.
    //   → Same 3D position for both — the reversed winding naturally pulls B_bot
    //     to the opposite side of the diamond in UV space.
    //
    // Combined mesh boundary (the UV frame):
    //   B_arc[0] — B1_top — … — B_arc.back() — Bn_bot — … — B1_bot — B_arc[0]
    //
    // TODO (Part 4): The UV frame is now defined by the B vertices, not vi/vj.
    //   check_valid_UV_lscm is frame-invariant so it still applies. Downstream
    //   callers that interpret vi/vj absolute UV positions need updating.
    // =========================================================================

    // --- Step 1: Boundary loop and B-arc extraction ---
    VectorXi bdLoop_eig;
    igl::boundary_loop(Fjoint_pre, bdLoop_eig);
    int n_loop = (int)bdLoop_eig.size();
    vector<int> bdLoop(bdLoop_eig.data(), bdLoop_eig.data() + n_loop);

    auto nan_skip = [&](const char* reason) {
        fprintf(dc_log(), "[DC-SKIP] vi=%d vj=%d nV=%d: %s\n", vi, vj, nV, reason);
        fflush(dc_log());
        UV_pre  = MatrixXd::Constant(nV, 2, numeric_limits<double>::quiet_NaN());
        UV_post = UV_pre;
    };

    int vi_pos = -1, vj_pos = -1;
    for (int i = 0; i < n_loop; i++) {
        if (bdLoop[i] == vi) vi_pos = i;
        if (bdLoop[i] == vj) vj_pos = i;
    }
    if (vi_pos < 0 || vj_pos < 0) { nan_skip("vi or vj not in boundary loop"); return; }

    // Walk the interior arc from vj+1 forward to vi (exclusive).
    // If vi is immediately after vj, the arc goes the other way (vi+1 → vj).
    vector<int> B_arc;
    {
        int start = (vj_pos + 1) % n_loop;
        if (start == vi_pos) {
            // vi immediately follows vj → B arc goes the other direction
            int cur = (vi_pos + 1) % n_loop;
            while (cur != vj_pos) { B_arc.push_back(bdLoop[cur]); cur = (cur+1)%n_loop; }
            reverse(B_arc.begin(), B_arc.end());  // keep B_arc[0] = vj-adjacent
        } else {
            int cur = start;
            while (cur != vi_pos) { B_arc.push_back(bdLoop[cur]); cur = (cur+1)%n_loop; }
        }
    }
    // B_arc[0]    = B vertex adjacent to vj  (glued)
    // B_arc.back()= B vertex adjacent to vi  (glued)
    // B_arc[1..end-1] = middle B vertices    (duplicated)

    if ((int)B_arc.size() < 3) {
        nan_skip("interior B arc has fewer than 3 vertices (need ≥1 middle B to duplicate)");
        return;
    }

    vector<int> B_glued    = { B_arc.front(), B_arc.back() };
    vector<int> B_reflected(B_arc.begin() + 1, B_arc.end() - 1);
    out_B_glued    = B_glued;
    out_B_reflected = B_reflected;

    if (isDebug) {
        fprintf(dc_log(), "[DC] vi=%d vj=%d  B_arc(%d):", vi, vj, (int)B_arc.size());
        for (int b : B_arc) fprintf(dc_log(), " %d", b);
        fprintf(dc_log(), "\n[DC] B_glued: %d %d  |  B_reflected(%d):",
                B_glued[0], B_glued[1], (int)B_reflected.size());
        for (int b : B_reflected) fprintf(dc_log(), " %d", b);
        fprintf(dc_log(), "\n");
    }

    // --- Step 2: UV variable layout ---
    // slots 0..nVjoint-1        : original joint vertices (top sheet, shared glue)
    // slots nVjoint..nVjoint_dc-1 : bottom-sheet copies of B_reflected[k]
    int nVjoint_dc = nVjoint + (int)B_reflected.size();

    // bot_remap: maps B_reflected[k] original index → new bottom-sheet UV slot nVjoint+k
    unordered_map<int,int> bot_remap;
    for (int k = 0; k < (int)B_reflected.size(); k++)
        bot_remap[B_reflected[k]] = nVjoint + k;

    if (isDebug) {
        fprintf(dc_log(), "[DC] bot_remap (top_idx->bot_idx): ");
        for (auto& kv : bot_remap) fprintf(dc_log(), "%d->%d  ", kv.first, kv.second);
        fprintf(dc_log(), "\n[DC] nVjoint=%d  nVjoint_dc=%d\n", nVjoint, nVjoint_dc);
    }

    // --- Step 3: Build Vjoint_dc ---
    // Bottom-sheet B_reflected copies keep the SAME 3D position as the originals.
    // The reversed winding of the bottom faces pulls them to the opposite side of
    // the diamond in UV space — no explicit 3D reflection needed.
    MatrixXd Vjoint_dc(nVjoint_dc, 3);
    Vjoint_dc.topRows(nVjoint) = Vjoint;
    for (int k = 0; k < (int)B_reflected.size(); k++)
        Vjoint_dc.row(nVjoint + k) = Vjoint.row(B_reflected[k]);

    // --- Step 4: Build DC face matrices ---
    // Top faces : original indices, original winding.
    // Bottom faces: swap cols 1↔2 (reverse winding), remap B_reflected → bot indices.
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

    // --- Step 5: Pinning ---
    // Pin the MIDDLE B_reflected vertex (top and bottom copies) as the two LSCM frame points.
    // Middle = true pole of the diamond: equidistant from both arc endpoints, so the two
    // halves of the DC diamond are symmetric. Using B_reflected[0] (arc-adjacent) placed
    // the pin off-center, causing a lopsided UV fold.
    int mid     = (int)B_reflected.size() / 2;
    int pin_top = B_reflected[mid];        // original UV index (top sheet, middle of arc)
    int pin_bot = nVjoint + mid;           // bottom copy of the same middle vertex

    VectorXi b_UV(4);
    VectorXd bc_UV(4);
    b_UV  << pin_top,              nVjoint_dc + pin_top,
             pin_bot,              nVjoint_dc + pin_bot;
    bc_UV << 0.0,                  0.0,
             1.0,                  0.0;
    // → pin_top at UV(col1=0, col0=0) = UV(0,0) ; pin_bot at UV(col1=1, col0=0) = UV(0,1)

    if (isDebug) {
        fprintf(dc_log(), "[DC] pinning: B_ref[mid=%d]_top(idx=%d)->UV(0,0)  "
                        "B_ref[mid=%d]_bot(idx=%d)->UV(0,1)\n", mid, pin_top, mid, pin_bot);
    }

    // --- Step 6: Solve ---
    VectorXd UVjoint_flat;
    flatten(Vjoint_dc, Fdc_pre, Vjoint_dc, Fdc_post,
            b_UV, bc_UV, nVjoint_dc, isDebug, UVjoint_flat);

    // Reshape flat [U-block; V-block] → nVjoint_dc×2 matrix.
    // flatten stores [first-coord; second-coord]; col(1) = first, col(0) = second.
    MatrixXd UVjoint(nVjoint_dc, 2);
    for (int col = 0; col < 2; col++)
        UVjoint.col(1 - col) = UVjoint_flat.segment(nVjoint_dc * col, nVjoint_dc);

    // --- Step 7: Extract UV_pre and UV_post ---
    UV_pre = UVjoint.topRows(nV);           // top-sheet rows 0..nV-1
    UV_post = UV_pre;
    UV_post.row(vi) = UVjoint.row(nV);      // vi's post-collapse UV from slot nV

    // --- Step 8: DC visualization (full double cover, both sheets) ---
    // Use the actual DC face matrices so the visualizer can see the bottom-sheet
    // B_reflected vertices (indices nVjoint..nVjoint_dc-1) in UV space.
    // UVjoint has nVjoint_dc rows — one UV per top-sheet slot AND per bottom-sheet copy.
    FUV_dc_pre  = Fdc_pre;
    FUV_dc_post = Fdc_post;
    UV_dc_pre   = UVjoint;   // full nVjoint_dc×2 (includes bottom-sheet B UV positions)
    UV_dc_post  = UVjoint;

    if (isDebug) {
        auto pUV = [&](const char* lbl, int idx) {
            if (idx < (int)UVjoint.rows())
                fprintf(dc_log(), "  %-22s idx=%-4d UV=(%.4f, %.4f)\n",
                        lbl, idx, UVjoint(idx,0), UVjoint(idx,1));
        };
        fprintf(dc_log(), "[DC] UV results:\n");
        pUV("vi",             vi);
        pUV("vj",             vj);
        pUV("nV(vi_post)",    nV);
        pUV("B_glued[0]",     B_glued[0]);
        pUV("B_glued[1]",     B_glued[1]);
        for (int k = 0; k < (int)B_reflected.size(); k++) {
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "B_ref[%d]_top", k);
            pUV(lbl, B_reflected[k]);
            snprintf(lbl, sizeof(lbl), "B_ref[%d]_bot", k);
            pUV(lbl, nVjoint + k);
        }
        fflush(dc_log());
    }
}

bool joint_lscm(
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
	using namespace std;
	bool verbose = false;
	bool isDebug = false;

  int nV = V_pre.rows();
	int infVIdx = -1;

	// Note:
	// Nsv[end] = E(e,0)
	// Ndv[end] = E(e,1)
	int e0, e1;
	if (Nsv[Nsv.size()-1] == vi)
	{
		e0 = vi; e1 = vj;
	}	
	else if (Ndv[Ndv.size()-1] == vi)
	{
		e1 = vi; e0 = vj;
	}
	
	// count whether vi or vj are on boundary by checking whether Nsv or Ndv contains infVIdx
	VectorXi onBd(2);
  onBd.setZero();
	{
		if (Nsv[Nsv.size()-1] == vj) // if E(e,0) == Nsv[end] == vj
		{
			if(std::find(Nsv.begin(), Nsv.end(), infVIdx) != Nsv.end())
				onBd(0) = 1;
			if(std::find(Ndv.begin(), Ndv.end(), infVIdx) != Ndv.end())
				onBd(1) = 1;
		}
		else if (Ndv[Ndv.size()-1] == vj) // if E(e,1) == Ndv[end] == vj
		{
			if(std::find(Ndv.begin(), Ndv.end(), infVIdx) != Ndv.end())
				onBd(0) = 1;
			if(std::find(Nsv.begin(), Nsv.end(), infVIdx) != Nsv.end())
				onBd(1) = 1;
		}
	}
	if (verbose)
		cout << "finish determining vi or vj on boundary\n";


	// check whether vi vj cut a flap
	bool isFlap = false;
	{
		if (onBd.sum() == 2)
		{
			int Nsv0 = Nsv[0];
			int Ndv0 = Ndv[0];
			int Nsv1 = Nsv[Nsv.size() - 2];
			int Ndv1 = Ndv[Ndv.size() - 2];
			if (Nsv0 == Ndv0 && Nsv0 == infVIdx)
				isFlap = false;
			else if (Nsv1 == Ndv1 && Nsv1 == infVIdx)
				isFlap = false;
			else
				isFlap = true;
		}
	}
	if (isFlap)
		return false;

	// // is vi vj a edge of a flap, if not v_flapCorner = -1
	// int v_flapCorner = -1;
	// {
	// 	if (onBd.sum() == 2)
	// 	{
	// 		if (Nsv.size() == 3)
	// 			v_flapCorner = Ndv[Ndv.size() - 1];
	// 		else if (Ndv.size() == 3)
	// 			v_flapCorner = Nsv[Nsv.size() - 1];
	// 	}
	// }

	// check 3D triangle quality
	if (onBd.sum() > 0)
	{
		double triangleQualityThreshold = 0.3;
		{
			// PROFC_NODE("check triangle quality");
			// check UV_pre triangle quality
			for (int fIdx=0; fIdx<FUV_post.rows(); fIdx++)
			{
				int v0 = FUV_post(fIdx,0);
				int v1 = FUV_post(fIdx,1);
				int v2 = FUV_post(fIdx,2);
				double l0 = (V_post.row(v0) - V_post.row(v1)).norm();
				double l1 = (V_post.row(v1) - V_post.row(v2)).norm();
				double l2 = (V_post.row(v2) - V_post.row(v0)).norm();
				double x = (l0+l1+l2) / 2;
				double delta = sqrt(x * (x-l0) * (x-l1) * (x-l2));
				double triQ = 4 * sqrt(3) * delta / (l0*l0 + l1*l1 + l2*l2); 
				if (triQ < triangleQualityThreshold || isnan(triQ)) 
				{
					if (verbose)
						cout << "bad 3D triangle quality" << endl;
					return false;
				}
			}
		}
	}

	// compute ordered boundary loop
	VectorXi bdLoop;
	{
		// PROFC_NODE("lscm: get boundary");
		if (onBd.sum() == 0 || onBd.sum() == 1)
		{
			// either vi vj are interior or both are interior
			bdLoop.resize(Nsv.size() + Ndv.size() - 4); 
			bdLoop.setZero();
			for (int ii=0; ii<Ndv.size()-2; ii++)
			{
				if (Ndv[Ndv.size() - 3 - ii]  == infVIdx)
					bdLoop(ii) = e0;
				else
					bdLoop(ii) = Ndv[Ndv.size() - 3 - ii];
			}
			for (int ii=0; ii<Nsv.size()-2; ii++)
			{
				if ( Nsv[ii+1] == infVIdx)
					bdLoop(ii+Ndv.size()-2) =  e1;
				else
					bdLoop(ii+Ndv.size()-2) = Nsv[ii+1];
			}
		}
		else if (onBd.sum() == 2)
		{
			// both vi vj are on boundary (general case)
			bdLoop.resize(Nsv.size() + Ndv.size() - 3); 
			bdLoop.setZero();
			for (int ii=0; ii<Ndv.size()-1; ii++)
			{
				if (Ndv[Ndv.size() - 2 - ii]  == infVIdx)
					bdLoop(ii) = e0;
				else
					bdLoop(ii) = Ndv[Ndv.size() - 2 - ii];
			}
			if (Nsv[0] == infVIdx) // do not rewrite the first one
			{
				for (int ii=0; ii<Nsv.size()-2; ii++)
				{
					if (Nsv[ii] == infVIdx)
						bdLoop(ii+Ndv.size()-1) =  e1;
					else
						bdLoop(ii+Ndv.size()-1) = Nsv[ii];
				}
			}
			else
			{
				for (int ii=0; ii<Nsv.size()-1; ii++)
				{
					if (Nsv[ii] == infVIdx)
						bdLoop(ii+Ndv.size()-2) =  e1;
					else
						bdLoop(ii+Ndv.size()-2) = Nsv[ii];
				}
			}
		}
	}
	if (verbose)
	{
		cout << "bfLoop: " << bdLoop.transpose() << endl;
		cout << "finish computing boundary loop\n";
	}

	// double check solution with igl function
	if (verbose)
	{
		VectorXi bdLoop_igl;
		igl::boundary_loop(FUV_pre,bdLoop_igl);
		cout << "boundary loop (igl): " << bdLoop_igl.transpose() << endl;
		VectorXi bdLoop_igl_reordered, secondHalf, firstHalf;
		for (int ii=0; ii<bdLoop_igl.size(); ii++)
		{
			if (bdLoop_igl(ii) == bdLoop(0)) 
			{	
				VectorXi I1 = VectorXi::LinSpaced(bdLoop_igl.size()-ii, ii, bdLoop_igl.size()-1);
				igl::slice(bdLoop_igl,I1,1,firstHalf);

				VectorXi I2 = VectorXi::LinSpaced(ii, 0, ii-1);
				igl::slice(bdLoop_igl,I2,1,secondHalf);
			}
		}
		bdLoop_igl_reordered.resize(bdLoop_igl.size());
				bdLoop_igl_reordered << firstHalf, secondHalf;
		cout << "boundary loop (igl reordered): " << bdLoop_igl_reordered.transpose() << endl;
		assert((bdLoop_igl_reordered - bdLoop).norm() == 0);
	}


	// determine cases:
	// - 0: vi and vj are both intereor vertices
	// - 1: one of the vi or vj is on the boundary
	// - 2: (vi, vj) is an boundary edge and it is not a flap
	// - 3: (vi, vj) is an boundary edge and it is an edge of an ear face (faces with two boundary edges)
	// - 4: (vi, vj) is an interior edge, but both are boundary vertices, and it is an edge of an ear face 
	int whichCase;
	if (onBd.sum() == 0)
		whichCase = 0;
	else if (onBd.sum() == 1)
		whichCase = 1;
	else if (onBd.sum() == 2)
		whichCase = 2;
	else
	{
		cout << "found a new case\n";
		abort();
	}

	if (out_case) *out_case = whichCase;

	switch (whichCase){
		case 0:
			joint_lscm_case0(V_pre,FUV_pre,V_post,FUV_post,vi,vj,bdLoop,verbose,isDebug,onBd,UV_pre,UV_post);
			break;
		case 1:
		case 2:
		{
			// All boundary cases use double cover + 2-point pinning.
			// Falls back to original handler if DC UV fails validation.
			static int dc_log_count = 0;
			bool is_seam_inj = (!Nsv.empty() && Nsv[0] == -1) ||
			                   (!Ndv.empty() && Ndv[0] == -1);

			Eigen::MatrixXi dc_F_pre, dc_F_post;
			Eigen::MatrixXd dc_UV_pre, dc_UV_post;
			Eigen::MatrixXd UV_pre_dc, UV_post_dc;
			std::vector<int> dc_B_glued, dc_B_reflected;
			// TODO: pass full global mesh boundary set so B-arc filtering is accurate.
			std::set<int> mesh_bd_verts_dc = {vi, vj};
			joint_lscm_double_cover(V_pre, FUV_pre, V_post, FUV_post,
			                        vi, vj, mesh_bd_verts_dc, isDebug,
			                        UV_pre_dc, UV_post_dc,
			                        dc_F_pre, dc_F_post, dc_UV_pre, dc_UV_post,
			                        dc_B_glued, dc_B_reflected);

			bool dc_ok = check_valid_UV_lscm(V_pre, UV_pre_dc, FUV_pre,
			                                  V_post, UV_post_dc, FUV_post,
			                                  vi, vj, Nsv, Ndv);

			dc_log_count++;
			if (!dc_ok) {
				// diagnose the exact failure reason
				bool has_nan = UV_pre_dc.array().isNaN().any() ||
				               UV_post_dc.array().isNaN().any();
				auto signed_area_2d = [](const Eigen::MatrixXd & UV,
				                         const Eigen::MatrixXi & F, int fi) {
					double ax = UV(F(fi,0),0), ay = UV(F(fi,0),1);
					double bx = UV(F(fi,1),0), by = UV(F(fi,1),1);
					double cx = UV(F(fi,2),0), cy = UV(F(fi,2),1);
					return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
				};
				bool pre_flip = false, post_flip = false;
				for (int fi = 0; fi < FUV_pre.rows()  && !pre_flip;  fi++) {
					double sa = signed_area_2d(UV_pre_dc,  FUV_pre,  fi);
					if (sa < 1e-10 || std::isnan(sa)) pre_flip = true;
				}
				for (int fi = 0; fi < FUV_post.rows() && !post_flip; fi++) {
					double sa = signed_area_2d(UV_post_dc, FUV_post, fi);
					if (sa < 1e-10 || std::isnan(sa)) post_flip = true;
				}
				fprintf(dc_log(),
				  "[DC-FAIL #%d] case=%d seam=%d vi=%d vj=%d nFpre=%d nFpost=%d nV=%d"
				  "  nan=%d pre_flip=%d post_flip=%d\n",
				  dc_log_count, whichCase, (int)is_seam_inj, vi, vj,
				  (int)FUV_pre.rows(), (int)FUV_post.rows(), (int)V_pre.rows(),
				  (int)has_nan, (int)pre_flip, (int)post_flip);
				fflush(dc_log());
			} else {
				// Passed flip check — measure quasi-conformal distortion on the
				// top-sheet UV to catch collapses that pass but are poorly conditioned.
				// UV_pre_dc/UV_post_dc both have nV rows (vi's post-UV is at row vi,
				// not at a separate nV+1 slot). FUV_pre/FUV_post use indices 0..nV-1.
				Eigen::VectorXd qce_pre, qce_post;
				quasi_conformal_error(V_pre, FUV_pre,  UV_pre_dc,  qce_pre);
				quasi_conformal_error(V_pre, FUV_post, UV_post_dc, qce_post);

				double max_pre  = qce_pre.maxCoeff(),  mean_pre  = qce_pre.mean();
				double max_post = qce_post.maxCoeff(), mean_post = qce_post.mean();
				bool high_dist  = (max_pre > 3.0 || max_post > 3.0);

				fprintf(dc_log(),
				  "[DC-%s #%d] collapse=%d case=%d seam=%d vi=%d vj=%d nFpre=%d nFpost=%d nV=%d"
				  "  B_glued=%d B_ref=%d"
				  "  QCE_pre(max=%.3f mean=%.3f) QCE_post(max=%.3f mean=%.3f)\n",
				  high_dist ? "HIGH-DISTORTION" : "PASS",
				  dc_log_count, collapse_idx, whichCase, (int)is_seam_inj, vi, vj,
				  (int)FUV_pre.rows(), (int)FUV_post.rows(), (int)V_pre.rows(),
				  (int)dc_B_glued.size(), (int)dc_B_reflected.size(),
				  max_pre, mean_pre, max_post, mean_post);

				// --- B_arc geometry + pinning audit ---
				// DC pins ONLY B_reflected[0] top->(0,0) and B_reflected[0] bot->(1,0).
				// No semicircle / equal-spacing of the arc. Log 3D edge lengths and
				// resulting UV positions to diagnose distortion source.
				{
					// Full B_arc order: [B_glued[0], B_ref[0..n-1], B_glued[1]]
					std::vector<int> b_arc_full;
					b_arc_full.push_back(dc_B_glued[0]);
					for (int k : dc_B_reflected) b_arc_full.push_back(k);
					b_arc_full.push_back(dc_B_glued[1]);

					// 3D edge lengths along the interior arc
					double total_len = 0.0;
					fprintf(dc_log(), "[DC-ARC #%d] B_arc(%d) 3D edge lengths: ",
					        dc_log_count, (int)b_arc_full.size());
					for (int k = 0; k + 1 < (int)b_arc_full.size(); k++) {
						double len = (V_pre.row(b_arc_full[k]) - V_pre.row(b_arc_full[k+1])).norm();
						total_len += len;
						fprintf(dc_log(), "%.4f ", len);
					}
					fprintf(dc_log(), " total=%.4f\n", total_len);

					// Boundary arc legs (vi→B_glued[0], vj→B_glued[1])
					double len_vi_bg0 = (V_pre.row(vi) - V_pre.row(dc_B_glued[0])).norm();
					double len_vj_bg1 = (V_pre.row(vj) - V_pre.row(dc_B_glued[1])).norm();
					fprintf(dc_log(), "[DC-ARC #%d] vi(%d)->B_glued[0](%d)=%.4f  vj(%d)->B_glued[1](%d)=%.4f\n",
					        dc_log_count, vi, dc_B_glued[0], len_vi_bg0,
					        vj, dc_B_glued[1], len_vj_bg1);

					// Pin assignment (only 2 vertices pinned, rest are free LSCM)
					if (!dc_B_reflected.empty()) {
						int log_mid = (int)dc_B_reflected.size() / 2;
						fprintf(dc_log(), "[DC-ARC #%d] pinned: B_ref[mid=%d]_top(idx=%d)->(0,0)  B_ref[mid=%d]_bot->(1,0)  [all other arc verts FREE]\n",
						        dc_log_count, log_mid, dc_B_reflected[log_mid], log_mid);
					}

					// UV positions of B_arc vertices after solve
					fprintf(dc_log(), "[DC-ARC #%d] B_arc UV: ", dc_log_count);
					for (int b : b_arc_full) {
						if (b < (int)UV_pre_dc.rows())
							fprintf(dc_log(), "%d=(%.3f,%.3f) ", b, UV_pre_dc(b,0), UV_pre_dc(b,1));
					}
					fprintf(dc_log(), "\n");
					fprintf(dc_log(), "[DC-ARC #%d] vi(%d) UV=(%.3f,%.3f)  vj(%d) UV=(%.3f,%.3f)\n",
					        dc_log_count, vi, UV_pre_dc(vi,0), UV_pre_dc(vi,1),
					        vj, UV_pre_dc(vj,0), UV_pre_dc(vj,1));
				}
				fflush(dc_log());
			}

			if (dc_ok) {
				UV_pre  = UV_pre_dc;
				UV_post = UV_post_dc;
				if (dc_viz) {
					dc_viz->has_data    = true;
					dc_viz->FUV_dc_pre  = dc_F_pre;
					dc_viz->FUV_dc_post = dc_F_post;
					dc_viz->UV_dc_pre   = dc_UV_pre;
					dc_viz->UV_dc_post  = dc_UV_post;
					dc_viz->B_glued     = dc_B_glued;
					dc_viz->B_reflected = dc_B_reflected;
				}
			} else {
				// DC failed — fall back to original handler.
				if (whichCase == 1)
					joint_lscm_case1(V_pre,FUV_pre,V_post,FUV_post,vi,vj,bdLoop,verbose,isDebug,onBd,UV_pre,UV_post);
				else
					joint_lscm_case2(V_pre,FUV_pre,V_post,FUV_post,vi,vj,bdLoop,verbose,isDebug,onBd,UV_pre,UV_post);
			}
			break;
		}
	}
	// return true;
	return check_valid_UV_lscm(V_pre, UV_pre, FUV_pre, V_post, UV_post, FUV_post, vi, vj, Nsv, Ndv);
}

bool check_valid_UV_lscm(
  const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXd & UV_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXd & UV_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
	const std::vector<int> & Nsv,
  const std::vector<int> & Ndv)
{
	using namespace Eigen;
	using namespace std;
	bool isDebug = false;
	bool verbose = false;

	// check whether UV contains NaN
	{
		if (UV_pre.array().isNaN().sum() > 0 || UV_post.array().isNaN().sum() > 0) // contains nan
		{
			if (verbose)
			{
				cout << "UV contains NaN\n";
				cout << "UV_pre: \n" << UV_pre << endl;
				cout << "UV_post: \n" << UV_post << endl;
			}
			return false;
		}
	}

	// check pre face normals flip or not
	{
		// pre faces
		for (int ii=0; ii<FUV_pre.rows(); ii++)
    {
      VectorXd v1 = UV_pre.row(FUV_pre(ii,1)) - UV_pre.row(FUV_pre(ii,0));
      VectorXd v2 = UV_pre.row(FUV_pre(ii,2)) - UV_pre.row(FUV_pre(ii,0));
      double signedArea = v1(0) * v2(1) - v1(1) * v2(0);
			// if (verbose)
			// 	cout << "pre face signed area: " << signedArea << endl;
      if (signedArea < 1e-10 || isnan(signedArea)) 
      {	
				if (verbose)
					cout << "pre flatten face normal flip" << endl;
				
				if (isDebug)
				{
					// writeOBJ_UV("preflipface_UVpre.obj", UV_pre, FUV_pre);
					// writeOBJ_UV("preflipface_UVpost.obj", UV_post, FUV_post);
					igl::writeOBJ("preflipface_Vpre.obj", V_pre, FUV_pre);
					igl::writeOBJ("preflipface_Vpost.obj", V_post, FUV_post);
					// cout << "vi: " << vi << ", vj: " << vj << endl;

					// for(int ii = 0; ii < Nsv.size(); ii++)
					// 	std::cout << Nsv[ii] << ", ";
					// std::cout << "\n";

					// for(int ii = 0; ii < Ndv.size(); ii++)
					// 	std::cout << Ndv[ii] << ", ";
					// std::cout << "\n";
				}
				
        return false;
      }
    }
	}

	// check post face normals flip or not
	{	
		for (int ii=0; ii<FUV_post.rows(); ii++)
    {
      VectorXd v1 = UV_post.row(FUV_post(ii,1)) - UV_post.row(FUV_post(ii,0));
      VectorXd v2 = UV_post.row(FUV_post(ii,2)) - UV_post.row(FUV_post(ii,0));
      double signedArea = v1(0) * v2(1) - v1(1) * v2(0);
			// if (verbose)
			// 	cout << "post face signed area: " << signedArea << endl;
      if (signedArea < 1e-10 || isnan(signedArea)) 
      {
				if (verbose)
					cout << "post flatten face normal flip" << endl;
				
				if (isDebug)
				{
					// writeOBJ_UV("postflipface_UVpre.obj", UV_pre, FUV_pre);
					// writeOBJ_UV("postflipface_UVpost.obj", UV_post, FUV_post);
					igl::writeOBJ("postflipface_Vpre.obj", V_pre, FUV_pre);
					igl::writeOBJ("postflipface_Vpost.obj", V_post, FUV_post);
					// cout << "vi: " << vi << ", vj: " << vj << endl;

					// for(int ii = 0; ii < Nsv.size(); ii++)
					// 	std::cout << Nsv[ii] << ", ";
					// std::cout << "\n";

					// for(int ii = 0; ii < Ndv.size(); ii++)
					// 	std::cout << Ndv[ii] << ", ";
					// std::cout << "\n";
				}
        return false;
      }
    }
	}

	// check pre UV face fold over
	{
		MatrixXd internalAng;
		assert(FUV_pre.maxCoeff() < UV_pre.rows());
		igl::internal_angles(UV_pre, FUV_pre, internalAng);
		double angSum_b0 = 0;
		double angSum_b1 = 0;
		for (int r=0; r<FUV_pre.rows(); r++){
			for (int c=0; c<FUV_pre.cols(); c++){
				if (FUV_pre(r,c) == vi) angSum_b0 += internalAng(r,c);
				if (FUV_pre(r,c) == vj) angSum_b1 += internalAng(r,c);
			}
		}
		if ((angSum_b0 - 2*M_PI) > 1e-10 || (angSum_b1 - 2*M_PI) > 1e-10) 
		{
			if (verbose)
				cout << "pre flatten self-overlap" << endl;
			if (isDebug)
			{
				// writeOBJ_UV("prefoldover_UVpre.obj", UV_pre, FUV_pre);
				// writeOBJ_UV("prefoldover_UVpost.obj", UV_post, FUV_post);
				igl::writeOBJ("prefoldover_Vpre.obj", V_pre, FUV_pre);
				igl::writeOBJ("prefoldover_Vpost.obj", V_post, FUV_post);
				// cout << "vi: " << vi << ", vj: " << vj << endl;

				// for(int ii = 0; ii < Nsv.size(); ii++)
				// 	std::cout << Nsv[ii] << ", ";
				// std::cout << "\n";

				// for(int ii = 0; ii < Ndv.size(); ii++)
				// 	std::cout << Ndv[ii] << ", ";
				// std::cout << "\n";
			}
			return false;
		}
	}

	// check post UV face fold over
	{
		MatrixXd internalAng;
		assert(FUV_post.maxCoeff() < UV_post.rows());
		igl::internal_angles(UV_post, FUV_post, internalAng);
		double angSum_b0 = 0;
		double angSum_b1 = 0;
		for (int r=0; r<FUV_post.rows(); r++){
			for (int c=0; c<FUV_post.cols(); c++){
				if (FUV_post(r,c) == vi) angSum_b0 += internalAng(r,c);
				if (FUV_post(r,c) == vj) angSum_b1 += internalAng(r,c);
			}
		}
		if ((angSum_b0 - 2*M_PI) > 1e-10 || (angSum_b1 - 2*M_PI) > 1e-10) 
		{
			if (verbose)
				cout << "post flatten self-overlap" << endl;
			if (isDebug)
			{
				// writeOBJ_UV("postfoldover_UVpre.obj", UV_pre, FUV_pre);
				// writeOBJ_UV("postfoldover_UVpost.obj", UV_post, FUV_post);
				igl::writeOBJ("postfoldover_Vpre.obj", V_pre, FUV_pre);
				igl::writeOBJ("postfoldover_Vpost.obj", V_post, FUV_post);
				// cout << "vi: " << vi << ", vj: " << vj << endl;

				// for(int ii = 0; ii < Nsv.size(); ii++)
				// 	std::cout << Nsv[ii] << ", ";
				// std::cout << "\n";

				// for(int ii = 0; ii < Ndv.size(); ii++)
				// 	std::cout << Ndv[ii] << ", ";
				// std::cout << "\n";
			}
			return false;
		}
	}

	double triangleQualityThreshold = 0.01;
	// check UV_pre triangle quality
  {
    for (int ii=0; ii<FUV_pre.rows(); ii++)
    {
      int v0 = FUV_pre(ii,0);
      int v1 = FUV_pre(ii,1);
      int v2 = FUV_pre(ii,2);
      double l0 = (UV_pre.row(v0) - UV_pre.row(v1)).norm();
      double l1 = (UV_pre.row(v1) - UV_pre.row(v2)).norm();
      double l2 = (UV_pre.row(v2) - UV_pre.row(v0)).norm();
      double x = (l0+l1+l2) / 2;
      double delta = sqrt(x * (x-l0) * (x-l1) * (x-l2));
      double triQ = 4 * sqrt(3) * delta / (l0*l0 + l1*l1 + l2*l2); 
      if (triQ < triangleQualityThreshold || isnan(triQ)) 
      {
				if (verbose)
					cout << "pre LSCM bad UV triangle quality" << endl;
				if (isDebug)
				{
					// writeOBJ_UV("preUVQuality_UVpre.obj", UV_pre, FUV_pre);
					// writeOBJ_UV("preUVQuality_UVpost.obj", UV_post, FUV_post);
					igl::writeOBJ("preUVQuality_Vpre.obj", V_pre, FUV_pre);
					igl::writeOBJ("preUVQuality_Vpost.obj", V_post, FUV_post);
				}
        return false;
      }
    }
  }

	// check UV_post triangle quality
  {
    for (int ii=0; ii<FUV_post.rows(); ii++)
    {
      int v0 = FUV_post(ii,0);
      int v1 = FUV_post(ii,1);
      int v2 = FUV_post(ii,2);
      double l0 = (UV_post.row(v0) - UV_post.row(v1)).norm();
      double l1 = (UV_post.row(v1) - UV_post.row(v2)).norm();
      double l2 = (UV_post.row(v2) - UV_post.row(v0)).norm();
      double x = (l0+l1+l2) / 2;
      double delta = sqrt(x * (x-l0) * (x-l1) * (x-l2));
      double triQ = 4 * sqrt(3) * delta / (l0*l0 + l1*l1 + l2*l2); 
      if (triQ < triangleQualityThreshold || isnan(triQ)) 
      {
				if (verbose)
        	cout << "post LSCM bad UV triangle quality" << endl;
				if (isDebug)
				{
					// writeOBJ_UV("postUVQuality_UVpre.obj", UV_pre, FUV_pre);
					// writeOBJ_UV("postUVQuality_UVpost.obj", UV_post, FUV_post);
					igl::writeOBJ("postUVQuality_Vpre.obj", V_pre, FUV_pre);
					igl::writeOBJ("postUVQuality_Vpost.obj", V_post, FUV_post);
				}
        return false;
      }
    }
  }

	return true;
	
}

void flatten(
	const Eigen::MatrixXd & Vjoint_pre,
  const Eigen::MatrixXi & Fjoint_pre,
  const Eigen::MatrixXd & Vjoint_post,
  const Eigen::MatrixXi & Fjoint_post,
	const Eigen::VectorXi & b_UV,
	const Eigen::VectorXd & bc_UV,
	const int & nVjoint,
	const bool & isDebug,
	Eigen::VectorXd & UVjoint_flat)
{
	using namespace Eigen;
	using namespace std;

	// get matrices
	// SparseMatrix<double> Q;
	MatrixXd Q;
	{
		// PROFC_NODE("lscm: get matrices");
		MatrixXd A_pre, A_post;

		// Assemble the area matrix (note that A is #Vx2 by #Vx2)
		vector_area_matrix_size(Fjoint_pre,nVjoint,A_pre);
		vector_area_matrix_size(Fjoint_post,nVjoint,A_post);

		// Assemble the cotan laplacian matrix for UV
		MatrixXd L_pre, L_post, LUV_pre, LUV_post;
		cotmatrix_dense(Vjoint_pre,Fjoint_pre,L_pre);
		cotmatrix_dense(Vjoint_post,Fjoint_post,L_post);

		// LUV_pre = [L_pre, 0; 0, L_pre]
		LUV_pre.resize(2*L_pre.rows(), 2*L_pre.cols());
		LUV_pre.setZero();
		LUV_pre.block(0,0,L_pre.rows(), L_pre.cols()) = L_pre;
		LUV_pre.block(L_pre.rows(), L_pre.cols(), L_pre.rows(), L_pre.cols()) = L_pre;

		// LUV_post = [L_post, 0; 0, L_post]
		LUV_post.resize(2*L_post.rows(), 2*L_post.cols());
		LUV_post.setZero();
		LUV_post.block(0,0,L_post.rows(), L_post.cols()) = L_post;
		LUV_post.block(L_post.rows(), L_post.cols(), L_post.rows(), L_post.cols()) = L_post;

		// get matrix for quadratics
		Q = -LUV_pre + 2.*A_pre - LUV_post + 2.*A_post;
	}

	// solve UV jointly
	{
		// PROFC_NODE("lscm: solve");
		const VectorXd RHS = VectorXd::Zero(nVjoint*2);
		mqwf_dense_data data;
		mqwf_dense_precompute(Q,b_UV,data);
		mqwf_dense_solve(data,RHS,bc_UV,UVjoint_flat);

		if (isDebug)
		{
			cout << "UVjoint_flat: " << UVjoint_flat.transpose() << endl;
			for (int ii = 0; ii<UVjoint_flat.size(); ii++)
				assert(!isnan(UVjoint_flat(ii)));
		}
	}

	// // avoid lscm mistakenly turns the mesh 90 degrees
  // double diff = std::abs(bc(0,0) - UV(b(0),0)) + 
  //               std::abs(bc(0,1) - UV(b(0),1));
  // double diff_reverse = std::abs(bc(0,1) - UV(b(0),0)) + 
  //                       std::abs(bc(0,0) - UV(b(0),1));
  // if (diff > diff_reverse){
  //   MatrixXd tmp = UV.col(0);
  //   UV.col(0) = UV.col(1);
  //   UV.col(1)= tmp;
  // }
}

void joint_lscm_case0( // both vi and vj are interior
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
	const bool & verbose,
	const bool & isDebug, 
	const Eigen::VectorXi & onBd,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post)
{
	using namespace Eigen;
	using namespace std;

	int nV = V_pre.rows();
	int infVIdx = -1;

	// create joint mesh
	MatrixXd Vjoint_pre,Vjoint_post;
	MatrixXi Fjoint_pre, Fjoint_post;
	int vi_pre, vj_pre, vi_post;
	VectorXi b_UV;
	VectorXd bc_UV;
	int nVjoint;
	{
		nVjoint = nV + 1;

		// cout << "start create joint mesh\n";
		// compute Vjoint = [V_pre; V_post.row(vi)]
		MatrixXd Vjoint;
		Vjoint.resize(nVjoint,3); 

		VectorXi I = VectorXi::LinSpaced(nV, 0, nV-1);
		igl::slice_into(V_pre,I,1,Vjoint);
		Vjoint.row(nV) = V_post.row(vi);

		Vjoint_post = Vjoint;
		Vjoint_pre = Vjoint;

		// compute Fjoint
		Fjoint_pre = FUV_pre;
		Fjoint_post = FUV_post;
		for (int r=0; r<Fjoint_post.rows(); r++){
			for (int c=0; c<Fjoint_post.cols(); c++){
				if (Fjoint_post(r,c) == vi)
					Fjoint_post(r,c) = nV;
			}
		}
		// cout << "finish create joint mesh\n";

		// assign vi vj
		vi_pre = vi;
		vj_pre = vj;
		vi_post = nV;

		// get boundary conditions for the global solve
		// for case 0: vi_pre = (0,0), vj_pre = (1,0)
		b_UV.resize(2*2,1);
		bc_UV.resize(2*2,1);
		b_UV << vi_pre, vj_pre, vi_pre+nVjoint, vj_pre+nVjoint;
		bc_UV << 0, 1, 0, 0;
		if (verbose)
			cout << "finish reindexing \n";
	}

	// solve UV jointly
	VectorXd UVjoint_flat;
	{
		// PROFC_NODE("lscm: solve");
	  flatten(Vjoint_pre,Fjoint_pre,Vjoint_post,Fjoint_post,b_UV,bc_UV,nVjoint,isDebug,UVjoint_flat);
	}
	// cout << "UVjoint_flat \n" << UVjoint_flat.transpose() << endl;
	if (verbose)
		cout << "finish solving \n";
	
	// reshape UV
	{
		// get UVjoint 
		MatrixXd UVjoint(nVjoint, 2);
		for (unsigned i=0;i<UVjoint.cols();++i)
			UVjoint.col(UVjoint.cols()-i-1) = UVjoint_flat.block(UVjoint.rows()*i,0,UVjoint.rows(),1);

		// assign to UV_pre
		VectorXi I = VectorXi::LinSpaced(nV, 0, nV-1);
		igl::slice(UVjoint,I,1,UV_pre);

		// get UV_post
		UV_post.resize(nV,2);
		UV_post = UV_pre;
		UV_post.row(vi) = UVjoint.row(vi_post);
	}
}

void joint_lscm_case1( // one of vi or vj is on boundary
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
	const bool & verbose,
	const bool & isDebug, 
	const Eigen::VectorXi & onBd,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post)
{
	using namespace Eigen;
	using namespace std;

	int nV = V_pre.rows();
	int infVIdx = -1;

	// create joint mesh
	MatrixXd Vjoint_pre,Vjoint_post;
	MatrixXi Fjoint_pre, Fjoint_post;
	int vi_pre, vj_pre, vi_post;
	VectorXi b_UV;
	VectorXd bc_UV;
	int nVjoint;
	{
		// cout << "start case 1\n";

		nVjoint = nV;

		int v_bd;
		if (onBd(0) == 1)
			v_bd = vi;
		else
			v_bd = vj;

		// cout << "start create joint mesh\n";
		// compute Vjoint = V_pre, but Vjoint.row(v_bd) = V_post.row(vi)
		Vjoint_pre.resize(nVjoint, 3);
		Vjoint_pre = V_pre;
		Vjoint_post.resize(nVjoint, 3);
		Vjoint_post = V_pre;
		Vjoint_post.row(v_bd) = V_post.row(vi);

		Fjoint_pre = FUV_pre;
		Fjoint_post = FUV_post;
		for (int r=0; r<Fjoint_post.rows(); r++){
			for (int c=0; c<Fjoint_post.cols(); c++){
				if (Fjoint_post(r,c) == vi)
					Fjoint_post(r,c) = v_bd;
			}
		}
		// cout << "finish create joint mesh\n";

		// assign vi vj
		vi_pre = vi;
		vj_pre = vj;
		vi_post = v_bd;

		// get boundary conditions for the global solve
		// for case 0: vi_pre = (0,0), vj_pre = (1,0)
		b_UV.resize(2*2,1);
		bc_UV.resize(2*2,1);
		b_UV << vi_pre, vj_pre, vi_pre+nVjoint, vj_pre+nVjoint;
		bc_UV << 0, 1, 0, 0;
	}
	if (verbose)
		cout << "finish reindexing\n";

	VectorXd UVjoint_flat;
	{
		// PROFC_NODE("lscm: solve");
		flatten(Vjoint_pre,Fjoint_pre,Vjoint_post,Fjoint_post,b_UV,bc_UV,nVjoint,isDebug,UVjoint_flat);
	}
	// cout << "UVjoint_flat \n" << UVjoint_flat.transpose() << endl;
	if (verbose)
		cout << "finish solving \n";
	
	// reshape UV
	{
		// get UVjoint 
		MatrixXd UVjoint(nVjoint, 2);
		for (unsigned i=0;i<UVjoint.cols();++i)
			UVjoint.col(UVjoint.cols()-i-1) = UVjoint_flat.block(UVjoint.rows()*i,0,UVjoint.rows(),1);

		// assign to UV_pre
		UV_pre = UVjoint;

		// get UV_post
		UV_post.resize(nV,2);
		UV_post = UV_pre;
		UV_post.row(vi) = UVjoint.row(vi_post);
	}
}

void joint_lscm_case2( // both vi and vj are on boundary, (vi,vj) is a boundary edge of a flap
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
	const bool & verbose,
	const bool & isDebug, 
	const Eigen::VectorXi & onBd,
	// int & snapIdx,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post)
{
	using namespace Eigen;
	using namespace std;

	// snap to vi
	MatrixXd UV_pre_vi, UV_post_vi;
	case2_constraint3_snap1(V_pre,FUV_pre,V_post,FUV_post,vi,vj,bdLoop,verbose,isDebug,onBd,vi,UV_pre_vi,UV_post_vi);

	VectorXd error_pre_vi, error_post_vi;
	quasi_conformal_error(V_pre,FUV_pre,UV_pre_vi,error_pre_vi);
	quasi_conformal_error(V_post,FUV_post,UV_post_vi,error_post_vi);
	double objVal_snap_vi = error_pre_vi.norm() + error_post_vi.norm();
	if (isnan(objVal_snap_vi))
		objVal_snap_vi = std::numeric_limits<int>::max();
	// cout << "objVal_snap_vi: " << objVal_snap_vi << endl; 
	// {
	// 	writeOBJ_UV("case2_vi_pre.obj", UV_pre_vi, FUV_pre);
	// 	writeOBJ_UV("case2_vi__post.obj", UV_post_vi, FUV_post); 
	// }

	// snap vj
	MatrixXd UV_pre_vj, UV_post_vj;
	case2_constraint3_snap1(V_pre,FUV_pre,V_post,FUV_post,vi,vj,bdLoop,verbose,isDebug,onBd,vj,UV_pre_vj,UV_post_vj);

	VectorXd error_pre_vj, error_post_vj;
	quasi_conformal_error(V_pre,FUV_pre,UV_pre_vj,error_pre_vj);
	quasi_conformal_error(V_post,FUV_post,UV_post_vj,error_post_vj);
	double objVal_snap_vj = error_pre_vj.norm() + error_post_vj.norm();
	if (isnan(objVal_snap_vj))
		objVal_snap_vj = std::numeric_limits<int>::max();
	// cout << "objVal_snap_vj: " << objVal_snap_vj << endl; 
	// {
	// 	writeOBJ_UV("case2_vj_pre.obj", UV_pre_vj, FUV_pre);
	// 	writeOBJ_UV("case2_vj_post.obj", UV_post_vj, FUV_post); 
	// }

	MatrixXd UV_pre_nosnap, UV_post_nosnap;
	case2_constraint4(V_pre,FUV_pre,V_post,FUV_post,vi,vj,bdLoop,verbose,isDebug,onBd,UV_pre_nosnap,UV_post_nosnap);

	VectorXd error_pre_n, error_post_n;
	quasi_conformal_error(V_pre,FUV_pre,UV_pre_nosnap,error_pre_n);
	quasi_conformal_error(V_post,FUV_post,UV_post_nosnap,error_post_n);
	double objVal_no_snap = error_pre_n.norm() + error_post_n.norm();
	if (isnan(objVal_no_snap))
		objVal_no_snap = std::numeric_limits<int>::max();
	// cout << "objVal_no_snap: " << objVal_no_snap << endl; 
	// {
	// 	writeOBJ_UV("case2_nosnap_pre.obj", UV_pre_nosnap, FUV_pre);
	// 	writeOBJ_UV("case2_nosnap_post.obj", UV_post_nosnap, FUV_post); 
	// }

	if (objVal_snap_vi <= objVal_snap_vj && objVal_snap_vi <= objVal_no_snap) // objVal_snap_vi is min
	{
		UV_pre.resize(UV_pre_vi.rows(), UV_pre_vi.cols()); 
		UV_pre = UV_pre_vi; 
		UV_post.resize(UV_post_vi.rows(), UV_post_vi.cols()); 
		UV_post = UV_post_vi;
	}	
	else if (objVal_snap_vj <= objVal_snap_vi && objVal_snap_vj <= objVal_no_snap) // objVal_snap_vj is min
	{
		UV_pre.resize(UV_pre_vj.rows(), UV_pre_vj.cols()); 
		UV_pre = UV_pre_vj; 
		UV_post.resize(UV_post_vj.rows(), UV_post_vj.cols()); 
		UV_post = UV_post_vj;
	}	
	else // objVal_no_snap is min
	{
		UV_pre.resize(UV_pre_nosnap.rows(), UV_pre_nosnap.cols()); 
		UV_pre = UV_pre_nosnap; 
		UV_post.resize(UV_post_nosnap.rows(), UV_post_nosnap.cols()); 
		UV_post = UV_post_nosnap;
	}	
}

void case2_constraint3_snap1(
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
	const bool & verbose,
	const bool & isDebug, 
	const Eigen::VectorXi & onBd,
	const int & snapIdx,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post)
{
	using namespace Eigen;
	using namespace std;

	assert((snapIdx == vi) || (snapIdx == vj));
	// // determine whether ui is going to snap to vi or vj
	// if (snapIdx == -1)
	// {
	// 	double dist2vi = (V_post.row(vi) - V_pre.row(vi)).norm();
	// 	double dist2vj = (V_post.row(vi) - V_pre.row(vj)).norm();
	// 	if (dist2vi < dist2vj)
	// 		snapIdx = vi;
	// 	else
	// 		snapIdx = vj;
	// }

	int nV = V_pre.rows();
	int infVIdx = -1;

	// create joint mesh
	MatrixXd Vjoint_pre,Vjoint_post;
	MatrixXi Fjoint_pre, Fjoint_post;
	int vi_pre, vj_pre, vi_post;
	VectorXi b_UV;
	VectorXd bc_UV;
	int nVjoint;
	{
		nVjoint = nV;

		// cout << "start create joint mesh\n";
		// compute Vjoint = [V_pre; V_post.row(vi)]
		Vjoint_pre.resize(nVjoint, 3);
		Vjoint_pre = V_pre;
		Vjoint_post.resize(nVjoint, 3);
		Vjoint_post = V_pre;
		Vjoint_post.row(snapIdx) = V_post.row(vi);

		Fjoint_pre = FUV_pre;
		Fjoint_post = FUV_post;
		for (int r=0; r<Fjoint_post.rows(); r++){
			for (int c=0; c<Fjoint_post.cols(); c++){
				if (Fjoint_post(r,c) == vi)
					Fjoint_post(r,c) = snapIdx;
			}
		}

		// get the vertex that needs to be on the straight line (snapIdx -- (vi or vj) -- vk)
		int vk = -1;
		{
			for (int ii=0; ii<bdLoop.size(); ii++)
			{
				if (bdLoop(ii) == snapIdx)
				{
					// check whether (vi/vj) is on the left
					int preIdx = (ii-1+bdLoop.size()) % bdLoop.size();
					if (bdLoop(preIdx) == vi || bdLoop(preIdx) == vj) // if on the left 
					{
						int prepreIdx = (ii-2+bdLoop.size()) % bdLoop.size();
						vk = bdLoop(prepreIdx);
					}
					
					// check whether (vi/vj) is on the right
					int postIdx = (ii+1) % bdLoop.size();
					if (bdLoop(postIdx) == vi || bdLoop(postIdx) == vj) // if on the right 
					{
						int postpostIdx = (ii+2) % bdLoop.size();
						vk = bdLoop(postpostIdx);
					}
				}
			}
			if(verbose)
				fprintf(stderr, "vk: %d\n", vk);
			assert(vk != -1);
		}

		// assign vi vj
		vi_pre = vi;
		vj_pre = vj;
		vi_post = snapIdx;

		// get boundary conditions for the global solve
		// for case 2: vi_pre = (0,0), vj_pre = (1,0), and all bIdx = (?,0)
		b_UV.resize(5);
		bc_UV.resize(5);
		bc_UV.setZero();

		b_UV << vi_pre, vj_pre, vi_pre+nVjoint, vj_pre+nVjoint, vk+nVjoint;
		bc_UV << 0, 1, 0, 0, 0;
	}
	if (verbose)
		cout << "finish reindexing \n";

	VectorXd UVjoint_flat;
	{
		// PROFC_NODE("lscm: solve");
		flatten(Vjoint_pre,Fjoint_pre,Vjoint_post,Fjoint_post,b_UV,bc_UV,nVjoint,isDebug,UVjoint_flat);
	}
	if (verbose)
		cout << "finish solving \n";
	
	// reshape UV
	{
		// get UVjoint 
		MatrixXd UVjoint(nVjoint, 2);
		for (unsigned i=0;i<UVjoint.cols();++i)
			UVjoint.col(UVjoint.cols()-i-1) = UVjoint_flat.block(UVjoint.rows()*i,0,UVjoint.rows(),1);

		// assign to UV_pre
		VectorXi I = VectorXi::LinSpaced(nV, 0, nV-1);
		igl::slice(UVjoint,I,1,UV_pre);

		// get UV_post
		UV_post.resize(nV,2);
		UV_post = UV_pre;
		UV_post.row(vi) = UVjoint.row(vi_post);
	}
}

void case2_constraint4(
	const Eigen::MatrixXd & V_pre,
  const Eigen::MatrixXi & FUV_pre,
  const Eigen::MatrixXd & V_post,
  const Eigen::MatrixXi & FUV_post,
  const int & vi,
  const int & vj,
  const Eigen::VectorXi & bdLoop,
	const bool & verbose,
	const bool & isDebug, 
	const Eigen::VectorXi & onBd,
  Eigen::MatrixXd & UV_pre,
  Eigen::MatrixXd & UV_post)
{
	using namespace Eigen;
	using namespace std;

	int nV = V_pre.rows();
	int infVIdx = -1;

	// create joint mesh
	MatrixXd Vjoint_pre,Vjoint_post;
	MatrixXi Fjoint_pre, Fjoint_post;
	int vi_pre, vj_pre, vi_post;
	VectorXi b_UV;
	VectorXd bc_UV;
	int nVjoint;
	{
		nVjoint = nV + 1;

		// cout << "start create joint mesh\n";
		// compute Vjoint = [V_pre; V_post.row(vi)]
		MatrixXd Vjoint;
		Vjoint.resize(nVjoint,3); 

		VectorXi I = VectorXi::LinSpaced(nV, 0, nV-1);
		igl::slice_into(V_pre,I,1,Vjoint);
		Vjoint.row(nV) = V_post.row(vi);

		Vjoint_post = Vjoint;
		Vjoint_pre = Vjoint;

		// compute Fjoint
		Fjoint_pre = FUV_pre;
		Fjoint_post = FUV_post;
		for (int r=0; r<Fjoint_post.rows(); r++){
			for (int c=0; c<Fjoint_post.cols(); c++){
				if (Fjoint_post(r,c) == vi)
					Fjoint_post(r,c) = nV;
			}
		}

		// assign vi vj
		vi_pre = vi;
		vj_pre = vj;
		vi_post = nV;

		// get the constrained vertices
		VectorXi bIdx;
		{
			// get boundary loop of the post mesh
			VectorXi bdLoop_post;
			{
				bdLoop_post = bdLoop;
				for (int ii=0; ii<bdLoop_post.size(); ii++)
				{
					if (bdLoop_post(ii) == vj)
						remove_vector_element(ii, bdLoop_post);
				}
			}
			if (verbose)
			{
				cout << "bdLoop_post: " << bdLoop_post.transpose() <<endl;
				cout << "finish getting post boundary loop\n";
			}

			// check boundary loop
			if (verbose)
			{
				VectorXi bdLoop_post_tmp;
				igl::boundary_loop(FUV_post,bdLoop_post_tmp);
				cout << "bdLoop_post_tmp: " << bdLoop_post_tmp.transpose() <<endl;

				// compare
				VectorXi bdLoop_post_reordered, secondHalf, firstHalf;
				for (int ii=0; ii<bdLoop_post.size(); ii++)
				{
					if (bdLoop_post_tmp(ii) == bdLoop_post(0)) 
					{	
						VectorXi I1 = VectorXi::LinSpaced(bdLoop_post_tmp.size()-ii, ii, bdLoop_post_tmp.size()-1);
						igl::slice(bdLoop_post_tmp,I1,1,firstHalf);

						VectorXi I2 = VectorXi::LinSpaced(ii, 0, ii-1);
						igl::slice(bdLoop_post_tmp,I2,1,secondHalf);
					}
				}
				bdLoop_post_reordered.resize(bdLoop_post_tmp.size());
						bdLoop_post_reordered << firstHalf, secondHalf;
				cout << "boundary loop (post reordered): " << bdLoop_post_reordered.transpose() << endl;
				assert((bdLoop_post_reordered - bdLoop_post).norm() == 0);
			}
			
			// get the neighboring vIdx of vi 
			VectorXi neighbor_to_vi(3);
			for (int ii=0; ii<bdLoop_post.size(); ii++)
			{
				if (bdLoop_post(ii) == vi)
				{
					int preIdx = (ii-1+bdLoop_post.size()) % bdLoop_post.size();
					neighbor_to_vi(0) = bdLoop_post(preIdx);

					neighbor_to_vi(1) = vi;

					int postIdx = (ii+1) % bdLoop_post.size();
					neighbor_to_vi(2) = bdLoop_post(postIdx);
				}
			}

			// get other vertices that should not be in the constraint set
			VectorXi freeVIdx, useless;
			igl::setdiff(bdLoop_post, neighbor_to_vi, freeVIdx, useless);
			igl::setdiff(bdLoop, freeVIdx, bIdx, useless);
		}

		// get boundary conditions for the global solve
		// for case 2: vi_pre = (0,0), vj_pre = (1,0), and all bIdx = (?,0)
		b_UV.resize(bIdx.size() + 2 + 1,1);
		bc_UV.resize(bIdx.size() + 2 + 1,1);
		bc_UV.setZero();
		b_UV(0) = vi_pre;
		b_UV(1) = vj_pre;
		b_UV(2) = vi_post+nVjoint;
		bc_UV(1) = 1.0;
		for (int ii=0; ii<bIdx.size(); ii++)
			b_UV(ii+3) = bIdx(ii)+nVjoint;
	}

	VectorXd UVjoint_flat;
	{
		// PROFC_NODE("lscm: solve");
		flatten(Vjoint_pre,Fjoint_pre,Vjoint_post,Fjoint_post,b_UV,bc_UV,nVjoint,isDebug,UVjoint_flat);
	}
	if (verbose)
		cout << "finish solving \n";
	
	// reshape UV
	{
		// get UVjoint 
		MatrixXd UVjoint(nVjoint, 2);
		for (unsigned i=0;i<UVjoint.cols();++i)
			UVjoint.col(UVjoint.cols()-i-1) = UVjoint_flat.block(UVjoint.rows()*i,0,UVjoint.rows(),1);

		// assign to UV_pre
		VectorXi I = VectorXi::LinSpaced(nV, 0, nV-1);
		igl::slice(UVjoint,I,1,UV_pre);

		// get UV_post
		UV_post.resize(nV,2);
		UV_post = UV_pre;
		UV_post.row(vi) = UVjoint.row(vi_post);
	}
}

// void joint_lscm_case2( // both vi and vj are on boundary (no flap)
// 	const Eigen::MatrixXd & V_pre,
//   const Eigen::MatrixXi & FUV_pre,
//   const Eigen::MatrixXd & V_post,
//   const Eigen::MatrixXi & FUV_post,
//   const int & vi,
//   const int & vj,
//   const Eigen::VectorXi & bdLoop,
// 	const bool & verbose,
// 	const bool & isDebug, 
// 	const Eigen::VectorXi & onBd,
//   Eigen::MatrixXd & UV_pre,
//   Eigen::MatrixXd & UV_post)
// {
// 	using namespace Eigen;
// 	using namespace std;

// 	int nV = V_pre.rows();
// 	int infVIdx = -1;

// 	// determine whether ui is going to snap to vi or vj
// 	int snapIdx;
// 	{
// 		double dist2vi = (V_post.row(vi) - V_pre.row(vi)).norm();
// 		double dist2vj = (V_post.row(vi) - V_pre.row(vj)).norm();
// 		if (dist2vi < dist2vj)
// 			snapIdx = vi;
// 		else
// 			snapIdx = vj;
// 	}

// 	// create joint mesh
// 	MatrixXd Vjoint_pre,Vjoint_post;
// 	MatrixXi Fjoint_pre, Fjoint_post;
// 	int vi_pre, vj_pre, vi_post;
// 	VectorXi b_UV;
// 	VectorXd bc_UV;
// 	int nVjoint;
// 	{
// 		nVjoint = nV;

// 		// cout << "start create joint mesh\n";
// 		// compute Vjoint = [V_pre; V_post.row(vi)]
// 		Vjoint_pre.resize(nVjoint, 3);
// 		Vjoint_pre = V_pre;
// 		Vjoint_post.resize(nVjoint, 3);
// 		Vjoint_post = V_pre;
// 		Vjoint_post.row(snapIdx) = V_post.row(vi);

// 		Fjoint_pre = FUV_pre;
// 		Fjoint_post = FUV_post;
// 		for (int r=0; r<Fjoint_post.rows(); r++){
// 			for (int c=0; c<Fjoint_post.cols(); c++){
// 				if (Fjoint_post(r,c) == vi)
// 					Fjoint_post(r,c) = snapIdx;
// 			}
// 		}

// 		// get the vertex that needs to be on the straight line (snapIdx -- (vi or vj) -- vk)
// 		int vk = -1;
// 		{
// 			for (int ii=0; ii<bdLoop.size(); ii++)
// 			{
// 				if (bdLoop(ii) == snapIdx)
// 				{
// 					// check whether (vi/vj) is on the left
// 					int preIdx = (ii-1+bdLoop.size()) % bdLoop.size();
// 					if (bdLoop(preIdx) == vi || bdLoop(preIdx) == vj) // if on the left 
// 					{
// 						int prepreIdx = (ii-2+bdLoop.size()) % bdLoop.size();
// 						vk = bdLoop(prepreIdx);
// 					}
					
// 					// check whether (vi/vj) is on the right
// 					int postIdx = (ii+1) % bdLoop.size();
// 					if (bdLoop(postIdx) == vi || bdLoop(postIdx) == vj) // if on the right 
// 					{
// 						int postpostIdx = (ii+2) % bdLoop.size();
// 						vk = bdLoop(postpostIdx);
// 					}
// 				}
// 			}
// 			cout << "vk: " << vk << endl;
// 			assert(vk != -1);
// 		}

// 		// assign vi vj
// 		vi_pre = vi;
// 		vj_pre = vj;
// 		vi_post = snapIdx;

// 		// get boundary conditions for the global solve
// 		// for case 2: vi_pre = (0,0), vj_pre = (1,0), and all bIdx = (?,0)
// 		b_UV.resize(5);
// 		bc_UV.resize(5);
// 		bc_UV.setZero();

// 		b_UV << vi_pre, vj_pre, vi_pre+nVjoint, vj_pre+nVjoint, vk+nVjoint;
// 		bc_UV << 0, 1, 0, 0, 0;
// 	}
// 	cout << "finish reindexing \n";

// 	VectorXd UVjoint_flat;
// 	{
// 		PROFC_NODE("lscm: solve");
// 		flatten(Vjoint_pre,Fjoint_pre,Vjoint_post,Fjoint_post,b_UV,bc_UV,nVjoint,isDebug,UVjoint_flat);
// 	}
// 	if (verbose)
// 		cout << "finish solving \n";
	
// 	// reshape UV
// 	{
// 		// get UVjoint 
// 		MatrixXd UVjoint(nVjoint, 2);
// 		for (unsigned i=0;i<UVjoint.cols();++i)
// 			UVjoint.col(UVjoint.cols()-i-1) = UVjoint_flat.block(UVjoint.rows()*i,0,UVjoint.rows(),1);

// 		// assign to UV_pre
// 		VectorXi I = VectorXi::LinSpaced(nV, 0, nV-1);
// 		igl::slice(UVjoint,I,1,UV_pre);

// 		// get UV_post
// 		UV_post.resize(nV,2);
// 		UV_post = UV_pre;
// 		UV_post.row(vi) = UVjoint.row(vi_post);
// 	}
// }



	

	// // determine whether ui is going to snap to vi or vj
	// if (snapIdx == -1)
	// {
	// 	double dist2vi = (V_post.row(vi) - V_pre.row(vi)).norm();
	// 	double dist2vj = (V_post.row(vi) - V_pre.row(vj)).norm();
	// 	if (dist2vi < dist2vj)
	// 		snapIdx = vi;
	// 	else
	// 		snapIdx = vj;
	// }


	// int nV = V_pre.rows();
	// int infVIdx = -1;

	// // create joint mesh
	// MatrixXd Vjoint_pre,Vjoint_post;
	// MatrixXi Fjoint_pre, Fjoint_post;
	// int vi_pre, vj_pre, vi_post;
	// VectorXi b_UV;
	// VectorXd bc_UV;
	// int nVjoint;
	// {
	// 	nVjoint = nV;

	// 	// cout << "start create joint mesh\n";
	// 	// compute Vjoint = [V_pre; V_post.row(vi)]
	// 	Vjoint_pre.resize(nVjoint, 3);
	// 	Vjoint_pre = V_pre;
	// 	Vjoint_post.resize(nVjoint, 3);
	// 	Vjoint_post = V_pre;
	// 	Vjoint_post.row(snapIdx) = V_post.row(vi);

	// 	Fjoint_pre = FUV_pre;
	// 	Fjoint_post = FUV_post;
	// 	for (int r=0; r<Fjoint_post.rows(); r++){
	// 		for (int c=0; c<Fjoint_post.cols(); c++){
	// 			if (Fjoint_post(r,c) == vi)
	// 				Fjoint_post(r,c) = snapIdx;
	// 		}
	// 	}

	// 	// get the vertex that needs to be on the straight line (snapIdx -- (vi or vj) -- vk)
	// 	int vk = -1;
	// 	{
	// 		for (int ii=0; ii<bdLoop.size(); ii++)
	// 		{
	// 			if (bdLoop(ii) == snapIdx)
	// 			{
	// 				// check whether (vi/vj) is on the left
	// 				int preIdx = (ii-1+bdLoop.size()) % bdLoop.size();
	// 				if (bdLoop(preIdx) == vi || bdLoop(preIdx) == vj) // if on the left 
	// 				{
	// 					int prepreIdx = (ii-2+bdLoop.size()) % bdLoop.size();
	// 					vk = bdLoop(prepreIdx);
	// 				}
					
	// 				// check whether (vi/vj) is on the right
	// 				int postIdx = (ii+1) % bdLoop.size();
	// 				if (bdLoop(postIdx) == vi || bdLoop(postIdx) == vj) // if on the right 
	// 				{
	// 					int postpostIdx = (ii+2) % bdLoop.size();
	// 					vk = bdLoop(postpostIdx);
	// 				}
	// 			}
	// 		}
	// 		cout << "vk: " << vk << endl;
	// 		assert(vk != -1);
	// 	}

	// 	// assign vi vj
	// 	vi_pre = vi;
	// 	vj_pre = vj;
	// 	vi_post = snapIdx;

	// 	// get boundary conditions for the global solve
	// 	// for case 2: vi_pre = (0,0), vj_pre = (1,0), and all bIdx = (?,0)
	// 	b_UV.resize(5);
	// 	bc_UV.resize(5);
	// 	bc_UV.setZero();

	// 	b_UV << vi_pre, vj_pre, vi_pre+nVjoint, vj_pre+nVjoint, vk+nVjoint;
	// 	bc_UV << 0, 1, 0, 0, 0;
	// }
	// cout << "finish reindexing \n";

	// VectorXd UVjoint_flat;
	// {
	// 	PROFC_NODE("lscm: solve");
	// 	double energyVal = flatten(Vjoint_pre,Fjoint_pre,Vjoint_post,Fjoint_post,b_UV,bc_UV,nVjoint,isDebug,UVjoint_flat);
	// }
	// if (verbose)
	// 	cout << "finish solving \n";
	
	// reshape UV
	// {
	// 	// get UVjoint 
	// 	MatrixXd UVjoint(nVjoint, 2);
	// 	for (unsigned i=0;i<UVjoint.cols();++i)
	// 		UVjoint.col(UVjoint.cols()-i-1) = UVjoint_flat.block(UVjoint.rows()*i,0,UVjoint.rows(),1);

	// 	// assign to UV_pre
	// 	VectorXi I = VectorXi::LinSpaced(nV, 0, nV-1);
	// 	igl::slice(UVjoint,I,1,UV_pre);

	// 	// get UV_post
	// 	UV_post.resize(nV,2);
	// 	UV_post = UV_pre;
	// 	UV_post.row(vi) = UVjoint.row(vi_post);
	// }