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

# App sources — list explicitly so new files are visible at a glance
set(APP_SRCS
    main.cpp
    visualizer.cpp
    orient_faces_consistently.cpp
)

# SSP library sources — glob is fine here; all files belong to the shared library
file(GLOB SSP_SRCS ../src/*.cpp)

add_executable(${PROJECT_NAME}_bin ${APP_SRCS} ${SSP_SRCS})

target_link_libraries(${PROJECT_NAME}_bin igl::core polyscope)

if(MSVC)
  target_compile_definitions(${PROJECT_NAME}_bin PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()
