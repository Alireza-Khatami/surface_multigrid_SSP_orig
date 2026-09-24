#include "coarse_mesh_export.h"

#include <igl/writeOBJ.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

static std::ofstream open_or_throw(const std::string & path, std::ios::openmode mode = std::ios::out)
{
    std::ofstream f(path, mode);
    if (!f) throw std::runtime_error("[coarse-export] cannot open " + path);
    return f;
}

static void write_l_lines(std::ofstream & f, const CoarseMeshCompaction & cmc)
{
    for (const auto & chain : cmc.staleChains) {
        f << "l";
        for (int idx : chain) f << " " << (idx + 1); // OBJ is 1-based
        f << "\n";
    }
}

void write_coarse_obj(const std::string & path, const CoarseMeshCompaction & cmc)
{
    if (!igl::writeOBJ(path, cmc.Vbase, cmc.Fout))
        throw std::runtime_error("[coarse-export] writeOBJ failed: " + path);
    if (!cmc.staleChains.empty()) {
        std::ofstream f = open_or_throw(path, std::ios::app);
        write_l_lines(f, cmc);
    }
    fprintf(stderr, "[coarse-export] OBJ %d verts  %d faces  %d chains -> %s\n",
            (int)cmc.Vbase.rows(), (int)cmc.Fout.rows(), (int)cmc.staleChains.size(), path.c_str());
}

void write_coarse_ply(const std::string & path, const CoarseMeshCompaction & cmc)
{
    std::vector<std::pair<int,int>> edges;
    for (const auto & chain : cmc.staleChains)
        for (size_t k = 0; k + 1 < chain.size(); k++)
            edges.push_back({chain[k], chain[k+1]});

    std::ofstream f = open_or_throw(path);
    const int nV = (int)cmc.Vbase.rows(), nF = (int)cmc.Fout.rows(), nE = (int)edges.size();
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << nV << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "element face " << nF << "\nproperty list uchar int vertex_indices\n";
    if (nE > 0) f << "element edge " << nE << "\nproperty int vertex1\nproperty int vertex2\n";
    f << "end_header\n";
    for (int i = 0; i < nV; i++)
        f << cmc.Vbase(i,0) << " " << cmc.Vbase(i,1) << " " << cmc.Vbase(i,2) << "\n";
    for (int i = 0; i < nF; i++)
        f << "3 " << cmc.Fout(i,0) << " " << cmc.Fout(i,1) << " " << cmc.Fout(i,2) << "\n";
    for (const auto & [a, b] : edges)
        f << a << " " << b << "\n";
    fprintf(stderr, "[coarse-export] PLY %d verts  %d faces  %d chain edges -> %s\n",
            nV, nF, nE, path.c_str());
}

void write_stale_chains_obj(const std::string & path, const CoarseMeshCompaction & cmc)
{
    std::ofstream f = open_or_throw(path);
    for (int i = 0; i < cmc.Vbase.rows(); i++)
        f << "v " << cmc.Vbase(i,0) << " " << cmc.Vbase(i,1) << " " << cmc.Vbase(i,2) << "\n";
    write_l_lines(f, cmc);
    fprintf(stderr, "[coarse-export] stale chains OBJ %d chains  %d verts -> %s\n",
            (int)cmc.staleChains.size(), (int)cmc.Vbase.rows(), path.c_str());
}
