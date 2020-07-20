cmake_minimum_required(VERSION 3.14)
project(MSIS LANGUAGES Fortran)
enable_testing()

if(NOT msis_src)
  if(DEFINED ENV{GEMINI_ROOT})
    set(msis_src $ENV{GEMINI_ROOT}/src/vendor/msis00)
  else()
    message(FATAL_ERROR "Please define msis_src or GEMINI_ROOT with top-level gemiin3d path")
  endif()
endif()

add_subdirectory(${msis_src} ${PROJECT_BINARY_DIR}/msis)
# this source_dir location is for Python, so the robust importlib.resources can get it.
set_target_properties(msis_setup PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
