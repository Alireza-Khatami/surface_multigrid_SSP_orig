cmake_minimum_required(VERSION 3.14)
project(correspond_viz)

set(POLYSCOPE_BACKEND_OPENGL3_GLFW "ON"  CACHE BOOL "Enable real OpenGL3+GLFW backend" FORCE)
set(POLYSCOPE_BACKEND_OPENGL_MOCK  "OFF" CACHE BOOL "Disable headless mock backend"    FORCE)

include(FetchContent)
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG        v2.3.0
)
FetchContent_MakeAvailable(polyscope)

add_executable(${PROJECT_NAME}_bin main.cpp)
target_link_libraries(${PROJECT_NAME}_bin polyscope)

if(MSVC)
  target_compile_definitions(${PROJECT_NAME}_bin PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()
