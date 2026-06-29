This is the core CMake pattern — add_subdirectory(). Here's how it works:

  Root CMakeLists.txt pulls in sub-projects:
  cmake_minimum_required(VERSION 3.20)
  project(my_root)

  add_subdirectory(01_single_SSP)
  add_subdirectory(08_subdiv_remesh)

  Each subdirectory's CMakeLists.txt runs as part of the same build, and all binaries land in one build tree.

  ---
  But there's a catch with this repo specifically. Each subfolder calls project(main) and produces an executable named main_bin. If you add_subdirectory() all of them, you'd get 9 targets all named main_bin —
  a conflict.

  The clean fix is to rename each target using the folder name:
  # inside 08_subdiv_remesh/CMakeLists.txt
  add_executable(subdiv_remesh_bin ${SRCFILES})  # instead of ${PROJECT_NAME}_bin

  ---
  When sub-projects produce libraries instead of executables (the more common real-world case), the root project links against them:
  # sub-project CMakeLists.txt
  add_library(ssp_core STATIC src/SSP_collapse_edge.cpp src/get_prolong.cpp ...)
  target_include_directories(ssp_core PUBLIC include/)

  # root CMakeLists.txt
  add_subdirectory(ssp_core)
  add_subdirectory(08_subdiv_remesh)

  # 08_subdiv_remesh/CMakeLists.txt
  target_link_libraries(subdiv_remesh_bin PRIVATE ssp_core igl::core ...)

  The PUBLIC/PRIVATE/INTERFACE keywords on target_include_directories and target_link_libraries control what propagates to dependents automatically — that's the key mechanism that makes multi-project CMake
  work cleanly.

  ---
  Summary of the three patterns:

  ┌────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │      Pattern       │                              Use when                               │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ add_subdirectory() │ Sub-projects are part of your own source tree                       │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ find_package()     │ Sub-project is already installed on the system                      │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ FetchContent       │ Sub-project is an external dependency to download at configure time │
  └────────────────────┴─────────────────────────────────────────────────────────────────────┘

  For your case — wrapping these 9 examples under one root — add_subdirectory() with renamed targets is the right move. Want me to set that up?