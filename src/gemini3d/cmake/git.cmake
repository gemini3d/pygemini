function(git_download root_dir url)
# tag argument is optional

find_package(Git REQUIRED)

if(NOT IS_DIRECTORY ${root_dir})
  execute_process(COMMAND ${GIT_EXECUTABLE} clone ${url} ${root_dir}
    RESULT_VARIABLE _gitstat
    TIMEOUT 120)
  if(NOT _gitstat STREQUAL 0)
    message(FATAL_ERROR "could not Git clone ${url}, return code ${_gitstat}")
  endif()
endif()

if(ARGC EQUAL 3)
  set(tag ${ARGV2})
  # use WORKING_DIRECTORY for legacy HPC Git
  execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${tag}
    RESULT_VARIABLE _gitstat
    TIMEOUT 30
    WORKING_DIRECTORY ${root_dir})
  if(NOT _gitstat STREQUAL 0)
    message(FATAL_ERROR "could not Git checkout ${tag}, return code ${_gitstat}")
  endif()
endif()

endfunction(git_download)
