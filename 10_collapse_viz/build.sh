#!/usr/bin/env bash
# Build 10_collapse_viz with MSVC + vcpkg.
# Polyscope is fetched automatically via CMake FetchContent (needs internet on first run).

CMAKE="/c/Program Files/CMake/bin/cmake.exe"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
VCPKG="C:/Users/alirz/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake"

export CMAKE_POLICY_VERSION_MINIMUM=3.5

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

"$CMAKE" -G "Visual Studio 17 2022" -A x64 \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG" \
    -S "$SCRIPT_DIR" \
    -B "$BUILD_DIR"

"$CMAKE" --build "$BUILD_DIR" --config Release -j8
