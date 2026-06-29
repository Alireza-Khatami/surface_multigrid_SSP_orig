  CMakeLists.txt — two fixes:
  - POLYSCOPE_BACKEND_OPENGL_MOCK OFF forced before FetchContent_MakeAvailable so polyscope never builds the mock
  backend
  - LIBIGL_WITH_OPENGL / LIBIGL_WITH_OPENGL_GLFW OFF kept — libigl stays headless, polyscope owns the window

  CMakePresets.json — two presets (windows-debug, windows-release), each pointing to build/debug or build/release, with
  vcpkg toolchain and the CMAKE_POLICY_VERSION_MINIMUM env var baked in.

  .vscode/settings.json — tells CMake Tools to use presets and the right cmake binary.

  .vscode/tasks.json — four tasks: configure + build for both configs. Build tasks depend on their configure task so a
  single Ctrl+Shift+B does everything.

  .vscode/launch.json — Debug and Release launch configs. On F5 you'll be prompted for mesh path and target face count.
  Uses cppvsdbg (MSVC debugger) so breakpoints work in the SSP source.

  To use: open 10_collapse_viz/ as a folder in VSCode, select the windows-debug or windows-release preset in the CMake
  Tools status bar, then press F5.