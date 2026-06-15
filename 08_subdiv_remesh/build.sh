#!/usr/bin/env bash
# Build script for 08_subdiv_remesh on Windows with MSVC + vcpkg
#
# CMakeLists.txt change required for MSVC:
#   Added the following block after target_link_libraries(...):
#
#     if(MSVC)
#       target_compile_definitions(${PROJECT_NAME}_bin PRIVATE _USE_MATH_DEFINES)
#     endif()
#
#   Reason: MSVC does not define M_PI by default (it is a POSIX extension).
#   Without _USE_MATH_DEFINES, the build fails with C2065: 'M_PI' undeclared identifier.
#
# Uses the Windows cmake explicitly so the Visual Studio generator is available,
# bypassing Strawberry Perl's Unix-only cmake.
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 is required because this CMakeLists.txt
# declares VERSION 3.1 which newer cmake no longer accepts without an override.

CMAKE="/c/Program Files/CMake/bin/cmake.exe"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
VCPKG="C:/Users/alirz/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Propagate policy override to all child cmake invocations (e.g. libigl's Eigen download)
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Remove stale build dir to avoid generator conflicts
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

"$CMAKE" -G "Visual Studio 17 2022" -A x64 \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG" \
    -S "$SCRIPT_DIR" \
    -B "$BUILD_DIR"

"$CMAKE" --build "$BUILD_DIR" --config Release -j8
