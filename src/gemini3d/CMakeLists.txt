cmake_minimum_required(VERSION 3.15)

include(cmake/compiler_find.cmake)

project(MSIS LANGUAGES Fortran)

include(CTest)
if(NOT DEFINED ${PROJECT_NAME}_BUILD_TESTING)
  set(${PROJECT_NAME}_BUILD_TESTING ${BUILD_TESTING})
endif()

set(gemini_url "https://github.com/gemini3d/gemini3d.git")
set(gemini_tag "master")

if(NOT gemini_root)
  if(DEFINED ENV{GEMINI_ROOT})
    set(gemini_root $ENV{GEMINI_ROOT})
  else()
    set(gemini_root ${CMAKE_CURRENT_SOURCE_DIR}/gemini-fortran/)
  endif()
endif()

set(msis_path ${gemini_root}/src/vendor/msis00)

include(cmake/git.cmake)

git_download(${gemini_root} ${gemini_url} ${gemini_tag})

add_subdirectory(${msis_path} ${PROJECT_BINARY_DIR}/msis)
# this source_dir location is for Python
# importlib.resources finds it.
set_target_properties(msis_setup PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
