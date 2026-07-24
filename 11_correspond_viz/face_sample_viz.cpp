#include "face_sample_viz.h"
#include "bundle.h"

#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/surface_mesh.h>

#include <query_coarse_to_fine.h>

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

using namespace Eigen;

// ---- globals owned here ----
int  gSelectedFace        = -1;
int  gFaceSampleN         = 20;
bool gShowFaceSampleCoarse = true;
bool gShowFaceSampleFine   = true;

static bool gFaceSampleReg      = false;
static bool gHullReg            = false;
static bool gFaceHighlightActive = false;

static polyscope::PointCloud* gCoarsePC = nullptr;
static polyscope::PointCloud* gFinePC   = nullptr;

// ---- globals owned in main.cpp ----
extern Bundle gBundle;
extern float  gZOffset;

// Project pts onto their PCA best-fit plane, compute 2D convex hull via
// Andrew's monotone chain, and return a fan-triangulated face list (indices into pts).
static MatrixXi convex_hull_2d(const MatrixXd & pts)
{
    int n = pts.rows();
    if (n < 3) return MatrixXi(0, 3);

    // Best-fit plane via PCA
    RowVector3d center = pts.colwise().mean();
    MatrixXd C = pts.rowwise() - center;
    Matrix3d cov = (C.transpose() * C) / n;
    SelfAdjointEigenSolver<Matrix3d> eig(cov);
    // eigenvectors in ascending eigenvalue order; last two span the dominant plane
    Vector3d ax0 = eig.eigenvectors().col(2);
    Vector3d ax1 = eig.eigenvectors().col(1);

    // Project to 2D
    MatrixXd p2(n, 2);
    for (int i = 0; i < n; i++) {
        Vector3d d = (pts.row(i) - center).transpose();
        p2(i, 0) = d.dot(ax0);
        p2(i, 1) = d.dot(ax1);
    }

    // Andrew's monotone chain
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return p2(a, 0) < p2(b, 0) || (p2(a, 0) == p2(b, 0) && p2(a, 1) < p2(b, 1));
    });

    auto cross = [&](int O, int A, int B) {
        return (p2(A,0)-p2(O,0))*(p2(B,1)-p2(O,1))
              -(p2(A,1)-p2(O,1))*(p2(B,0)-p2(O,0));
    };

    std::vector<int> hull;
    hull.reserve(2 * n);
    // lower hull
    for (int i = 0; i < n; i++) {
        while (hull.size() >= 2 && cross(hull[hull.size()-2], hull.back(), idx[i]) <= 0)
            hull.pop_back();
        hull.push_back(idx[i]);
    }
    // upper hull
    int lower_sz = (int)hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while ((int)hull.size() > lower_sz && cross(hull[hull.size()-2], hull.back(), idx[i]) <= 0)
            hull.pop_back();
        hull.push_back(idx[i]);
    }
    hull.pop_back(); // last == first

    int h = (int)hull.size();
    if (h < 3) return MatrixXi(0, 3);

    // Fan triangulation from hull[0]
    MatrixXi F(h - 2, 3);
    for (int i = 0; i < h - 2; i++) {
        F(i, 0) = hull[0];
        F(i, 1) = hull[i + 1];
        F(i, 2) = hull[i + 2];
    }
    return F;
}

void rebuild_face_sample_viz()
{
    if (gFaceSampleReg) {
        polyscope::removeStructure("face_sample_coarse");
        polyscope::removeStructure("face_sample_fine");
        gFaceSampleReg = false;
        gCoarsePC = nullptr;
        gFinePC   = nullptr;
    }
    if (gHullReg) {
        polyscope::removeStructure("face_sample_hull");
        gHullReg = false;
    }
    // Clear or update face highlight on the existing coarse mesh
    {
        auto* mesh = polyscope::getSurfaceMesh("coarse_mesh");
        if (mesh) {
            if (gSelectedFace < 0) {
                if (gFaceHighlightActive) {
                    mesh->removeQuantity("face_highlight");
                    gFaceHighlightActive = false;
                }
            } else {
                int FC = gBundle.coarseF.rows();
                MatrixXd faceColors(FC, 3);
                // default coarse mesh color
                for (int f = 0; f < FC; f++)
                    faceColors.row(f) = RowVector3d(0.25, 0.52, 0.95);
                faceColors.row(gSelectedFace) = RowVector3d(1.0, 0.65, 0.0);
                mesh->addFaceColorQuantity("face_highlight", faceColors)->setEnabled(true);
                gFaceHighlightActive = true;
            }
        }
    }
    if (gSelectedFace < 0) return;

    int n = std::max(1, gFaceSampleN);

    // Compact coarse face corners
    int ci0 = gBundle.coarseF(gSelectedFace, 0);
    int ci1 = gBundle.coarseF(gSelectedFace, 1);
    int ci2 = gBundle.coarseF(gSelectedFace, 2);
    RowVector3d cv0 = gBundle.coarseV.row(ci0);
    RowVector3d cv1 = gBundle.coarseV.row(ci1);
    RowVector3d cv2 = gBundle.coarseV.row(ci2);

    MatrixXd coarsePts(n, 3), finePts(n, 3), arrowVecs(n, 3);

    // Fold-sampled barycentric coords + coarse 3D positions
    MatrixXd BC(n, 3);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int k = 0; k < n; k++) {
            double r1 = U(rng), r2 = U(rng);
            if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }
            BC(k, 0) = r1;
            BC(k, 1) = r2;
            BC(k, 2) = 1.0 - r1 - r2;

            RowVector3d cPt = r1 * cv0 + r2 * cv1 + BC(k, 2) * cv2;
            cPt(2) += (double)gZOffset;
            coarsePts.row(k) = cPt;
        }
    }

    if (gBundle.hasSspData) {
        // Map compact→global for corners and the face itself
        int gvi0 = gBundle.vtxMap[ci0];
        int gvi1 = gBundle.vtxMap[ci1];
        int gvi2 = gBundle.vtxMap[ci2];
        int gfi  = gBundle.faceMap[gSelectedFace];

        MatrixXi BF(n, 3);
        VectorXi FIdx(n);
        for (int k = 0; k < n; k++) {
            BF(k, 0) = gvi0;  BF(k, 1) = gvi1;  BF(k, 2) = gvi2;
            FIdx(k)  = gfi;
        }

        // IM and IMF are identity (decimation never reindexes, only nullifies)
        VectorXi IM  = VectorXi::LinSpaced(gBundle.nV_total, 0, gBundle.nV_total - 1);
        VectorXi IMF = VectorXi::LinSpaced(gBundle.nF_total, 0, gBundle.nF_total - 1);

        query_coarse_to_fine(gBundle.decInfo, IM, gBundle.decIM, IMF,
                             gBundle.faceSheetID, BC, BF, FIdx);

        for (int k = 0; k < n; k++) {
            RowVector3d fPt = BC(k,0) * gBundle.fineV.row(BF(k,0))
                            + BC(k,1) * gBundle.fineV.row(BF(k,1))
                            + BC(k,2) * gBundle.fineV.row(BF(k,2));
            finePts.row(k)   = fPt;
            arrowVecs.row(k) = fPt - coarsePts.row(k);
        }

        // Convex hull of the fine correspondence points
        if (n >= 3) {
            MatrixXi hullF = convex_hull_2d(finePts);
            if (hullF.rows() > 0) {
                auto* hull = polyscope::registerSurfaceMesh(
                    "face_sample_hull", finePts, hullF);
                hull->setSurfaceColor({0.10f, 0.90f, 0.75f});
                hull->setTransparency(1.0f);
                hull->setEdgeWidth(0.0f);
                hull->setSmoothShade(false);
                gHullReg = true;
            }
        }
    } else {
        finePts   = coarsePts;
        arrowVecs.setZero();
        std::cerr << "[face_sample] No SSP data — need a v2 bundle for correspondence\n";
    }

    gCoarsePC = polyscope::registerPointCloud("face_sample_coarse", coarsePts);
    gCoarsePC->setPointColor({0.90f, 0.40f, 1.00f});
    gCoarsePC->setPointRadius(0.0048, true);
    gCoarsePC->setEnabled(gShowFaceSampleCoarse);
    auto* vq = gCoarsePC->addVectorQuantity("to_fine", arrowVecs, polyscope::VectorType::AMBIENT);
    vq->setEnabled(true);
    vq->setVectorColor({0.70f, 0.00f, 1.00f});
    vq->setVectorLengthScale(1.0, false);

    gFinePC = polyscope::registerPointCloud("face_sample_fine", finePts);
    gFinePC->setPointColor({0.20f, 1.00f, 0.85f});
    gFinePC->setPointRadius(0.0048, true);
    gFinePC->setEnabled(gShowFaceSampleFine);

    gFaceSampleReg = true;
}

void update_face_sample_visibility()
{
    if (gCoarsePC) gCoarsePC->setEnabled(gShowFaceSampleCoarse);
    if (gFinePC)   gFinePC->setEnabled(gShowFaceSampleFine);
}
