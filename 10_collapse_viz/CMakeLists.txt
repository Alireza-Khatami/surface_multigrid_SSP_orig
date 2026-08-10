cmake_minimum_required(VERSION 3.14)
project(collapse_viz)

# ---- polyscope via FetchContent ----
# Force real OpenGL/GLFW backend; disable headless mock before polyscope's CMakeLists runs.
set(POLYSCOPE_BACKEND_OPENGL3_GLFW "ON"  CACHE BOOL "Enable real OpenGL3+GLFW backend" FORCE)
set(POLYSCOPE_BACKEND_OPENGL_MOCK  "OFF" CACHE BOOL "Disable headless mock backend"    FORCE)

include(FetchContent)
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG        v2.3.0
)
FetchContent_MakeAvailable(polyscope)

# ---- libigl (core only – polyscope owns OpenGL/GLFW) ----
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../08_subdiv_remesh/cmake)
option(LIBIGL_WITH_OPENGL      "Use OpenGL" OFF)
option(LIBIGL_WITH_OPENGL_GLFW "Use GLFW"   OFF)
find_package(LIBIGL REQUIRED)

include_directories("../src/")

# Core sources — always compiled.
# visualizer.cpp and sheet_seam_viz.cpp are added below only when C2F_VIZ_DIAGNOSTIC is ON.
set(APP_SRCS
    main.cpp
    load_matstruct.cpp
    coarse_fine_viz.cpp
    face_sample_tracker.cpp
    orient_faces_consistently.cpp
    meshlab_qslim_callbacks.cpp
    meshlab_qslim_config.cpp
)

# SSP library sources — glob is fine here; all files belong to the shared library
file(GLOB SSP_SRCS ../src/*.cpp)

add_executable(${PROJECT_NAME}_bin ${APP_SRCS} ${SSP_SRCS})

target_compile_features(${PROJECT_NAME}_bin PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME}_bin igl::core polyscope)

if(MSVC)
  target_compile_definitions(${PROJECT_NAME}_bin PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

option(C2F_VIZ_DIAGNOSTIC "Enable per-collapse ring_post c2f diagnostic polyscope visualization (traced walk, point clouds, arrows)" ON)
if(C2F_VIZ_DIAGNOSTIC)
    target_compile_definitions(${PROJECT_NAME}_bin PRIVATE C2F_VIZ_DIAGNOSTIC)
    target_sources(${PROJECT_NAME}_bin PRIVATE visualizer.cpp sheet_seam_viz.cpp)
endif()

option(SSP_LSCM_LOG "Enable SSP LSCM mapping diagnostic logs (fan_walk, pre-jlscm, LSCM-FAIL, NON-ACTIVE-SHEET, etc.)" ON)
if(SSP_LSCM_LOG)
    target_compile_definitions(${PROJECT_NAME}_bin PRIVATE SSP_LSCM_LOG)
endif()

option(ML_QEM_LOG "Enable MeshLab QEM rejection logs (ML-COST-REJECT, ML-REJECT)" ON)
if(ML_QEM_LOG)
    target_compile_definitions(${PROJECT_NAME}_bin PRIVATE ML_QEM_LOG)
endif()
