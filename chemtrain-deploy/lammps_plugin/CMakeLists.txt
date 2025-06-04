# -*- CMake -*- file for plugin examples.
# The is meant to be used as a template for plugins that are
# distributed independent from the LAMMPS package.
##########################################

cmake_minimum_required(VERSION 3.16)

# enforce out-of-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed. You must create and use a build directory. "
    "Please remove CMakeCache.txt and CMakeFiles first.")
endif()

message("Source dir: ${CMAKE_SOURCE_DIR}\nBinary dir: ${CMAKE_BINARY_DIR}")

project(plugins VERSION 1.0 LANGUAGES CXX)

# when this file is included as subdirectory in the LAMMPS build, many settings are directly imported
if(LAMMPS_DIR)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})
else()
  # NOTE: the next line should be commented out when used outside of the LAMMPS package
  # get_filename_component(LAMMPS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../src ABSOLUTE)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR} CACHE PATH "Location of LAMMPS headers")
  if(NOT LAMMPS_HEADER_DIR)
    message(FATAL_ERROR "Must set LAMMPS_HEADER_DIR")
  endif()
  # by default, install into $HOME/.local (not /usr/local),
  # so that no root access (and sudo) is needed
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Default install path" FORCE)
  endif()
  # ugly hacks for MSVC which by default always reports an old C++ standard in the __cplusplus macro
  # and prints lots of pointless warnings about "unsafe" functions
  if(MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      add_compile_options(/Zc:__cplusplus)
      add_compile_options(/wd4244)
      add_compile_options(/wd4267)
      add_compile_options(/wd4250)
      add_compile_options(/EHsc)
    endif()
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  endif()
endif()

# C++11 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_C_COMPILER "/usr/lib/llvm-18/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/lib/llvm-18/bin/clang++")


set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script")

add_compile_options(-std=c++17)

# Need -restrict with Intel compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -restrict")
endif()

set(SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../connector)
set(LIB_DIR ${CMAKE_SOURCE_DIR}/../lib)

include(CheckIncludeFileCXX)
if(NOT LAMMPS_DIR)
  include(${SOURCE_DIR}/LAMMPSInterfaceCXX)
endif()

##########################
# building the plugins

include_directories(${INCLUDE_DIR})
add_library(chemtrain_deploy MODULE
        ${SOURCE_DIR}/chemtrain_deploy.cpp
        ${SOURCE_DIR}/pair_chemtrain_deploy.cpp
        ${INCLUDE_DIR}/libconnector.h)
target_link_libraries(
        chemtrain_deploy PRIVATE lammps ${LIB_DIR}/libconnector.so
)
set_target_properties(chemtrain_deploy PROPERTIES PREFIX "" SUFFIX "plugin.so" LINKER_LANGUAGE CXX CXX_STANDARD 17)


# MacOS seems to need this
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set_target_properties(chemtrain_deploy
    PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
# tell CMake to export all symbols to a .dll on Windows with special case for MinGW cross-compilers
  set_target_properties(chemtrain_deploy
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(chemtrain_deploy
      PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
  endif()
else()
  set_target_properties(chemtrain_deploy PROPERTIES
    LINK_FLAGS "-rdynamic")
endif()

add_custom_target(plugins ALL ${CMAKE_COMMAND} -E echo "Building Plugins"
  DEPENDS chemtrain_deploy)
