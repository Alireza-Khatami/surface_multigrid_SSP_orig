#pragma once
#include <Eigen/Dense>
#include <array>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// Stale chains: naked l-element edges in the fine mesh that SSP must not touch.
extern std::vector<std::vector<int>> gStaleChains;    // ordered vertex ID sequences
extern std::unordered_set<int>       gStaleVertexSet; // fast lookup for the pre-collapse lock

// OBJ loader that also captures 'l' line-elements (avoiding libigl's parser warning).
bool load_obj_vfl(const std::string & path,
                  Eigen::MatrixXd & V,
                  Eigen::MatrixXi & F,
                  std::vector<std::pair<int,int>> & l_edges);

// Filters l-edges to naked-only, builds maximal chain decomposition.
// Populates gStaleChains and gStaleVertexSet.
void detect_stale_chains(const std::vector<std::pair<int,int>> & l_edges_raw,
                         const Eigen::MatrixXi & FO);

#ifdef C2F_VIZ_DIAGNOSTIC
extern std::vector<uint8_t> gStaleChainVisible;
extern bool                 gStaleChainShowAll;

std::array<float,3> stale_hsv_rgb(float h, float s, float v);
void update_stale_chains_display();
#endif
