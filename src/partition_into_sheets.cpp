#include "partition_into_sheets.h"

#include <map>
#include <queue>
#include <utility>
#include <vector>
#include <cstdio>

void partition_into_sheets(
    const Eigen::MatrixXi & F,
    Eigen::VectorXi & faceSheetID,
    int & numSheets)
{
    using namespace std;
    const int nF = F.rows();
    faceSheetID.setConstant(nF, -1);
    numSheets = 0;

    // Build undirected edge → list of incident face indices
    map<pair<int,int>, vector<int>> edgeFaces;
    for (int f = 0; f < nF; f++) {
        for (int c = 0; c < 3; c++) {
            int a = F(f,c), b = F(f,(c+1)%3);
            if (a > b) swap(a,b);
            edgeFaces[{a,b}].push_back(f);
        }
    }

    // BFS: cross only manifold interior edges (exactly 2 incident faces).
    // Boundary edges (1 face) and non-manifold edges (3+ faces) act as sheet boundaries.
    for (int seed = 0; seed < nF; seed++) {
        if (faceSheetID(seed) != -1) continue;

        queue<int> q;
        q.push(seed);
        faceSheetID(seed) = numSheets;

        while (!q.empty()) {
            int f = q.front(); q.pop();
            for (int c = 0; c < 3; c++) {
                int a = F(f,c), b = F(f,(c+1)%3);
                if (a > b) swap(a,b);
                const auto & nbrs = edgeFaces[{a,b}];
                if ((int)nbrs.size() != 2) continue;  // boundary or non-manifold
                for (int nf : nbrs) {
                    if (faceSheetID(nf) == -1) {
                        faceSheetID(nf) = numSheets;
                        q.push(nf);
                    }
                }
            }
        }
        numSheets++;
    }

    fprintf(stderr, "[partition_into_sheets] %d faces → %d sheets\n", nF, numSheets);
}
