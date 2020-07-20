cmake_minimum_required(VERSION 3.14)
project(MSIS LANGUAGES Fortran)
enable_testing()

set(gemini_url "https://github.com/gemini3d/gemini")

if(NOT gemini_root)
  if(DEFINED ENV{GEMINI_ROOT})
    set(gemini_root $ENV{GEMINI_ROOT})
  else()
    set(gemini_root ${CMAKE_CURRENT_SOURCE_DIR}/../../../gemini/)
  endif()
endif()

set(msis_path ${gemini_root}/src/vendor/msis00)

include(${CMAKE_CURRENT_SOURCE_DIR}/git.cmake)
clone_if_missing(${gemini_root} ${gemini_url})

add_subdirectory(${msis_path} ${PROJECT_BINARY_DIR}/msis)
# this source_dir location is for Python, so the robust importlib.resources can get it.
set_target_properties(msis_setup PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
