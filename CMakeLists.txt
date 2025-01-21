# ##################################################################################################
# Defines the sys-vm library and associated tooling. See docs/cmake.md for instructions on how to
# build sys-vm or integrate with another system with CMake.
# ##################################################################################################
cmake_minimum_required(VERSION 3.8)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)
set(VERSION_SUFFIX rc1)
project(sys-vm VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

if (VERSION_SUFFIX)
   set(VERSION_FULL "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}-${VERSION_SUFFIX}")
else()
   set(VERSION_FULL "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
endif()

message(STATUS "Building sys-vm v${VERSION_FULL}...")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads)
link_libraries(Threads::Threads)

# ##################################################################################################
# Gate against Windows builds.
# ##################################################################################################
if (WIN32)
   message(FATAL_ERROR "SYS-VM currently only builds on Unix like OSes (Linux, Mac OSX)")
endif()

# ##################################################################################################
# Defined options for building or integrating sys-vm.
# ##################################################################################################
include(CMakeDependentOption)
option(ENABLE_SOFTFLOAT "enable the backend to use deterministic software floating point operations"
	ON)
option(FULL_DEBUG_BUILD "enables stack dumping and instruction tracing" OFF)
option(ENABLE_INSTALL "enable this library to be installed" ON)
option(ENABLE_MEMORY_OPS_ALIGNMENT "enable the backend to obey alignment hints" OFF)
option(ENABLE_TOOLS "enable building of tools" ON)
option(ENABLE_TESTS "enable building of unit tests, spec tests" OFF)
cmake_dependent_option(ENABLE_SPEC_TESTS "enable wasm spec tests" ON "ENABLE_TESTS" ON)
cmake_dependent_option(ENABLE_FUZZ_TESTS "enable fuzz testing" OFF "ENABLE_TESTS" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules")
include(SysVMBuildUtils)

# ##################################################################################################
# Create the sys-vm library.
# ##################################################################################################
add_library(sys-vm INTERFACE)
target_include_directories(sys-vm
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                     ${CMAKE_CURRENT_SOURCE_DIR}/external/softfloat/source/include)
# ignore the C++17 register warning until clean up
target_compile_options( sys-vm INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-register> )

# ##################################################################################################
# Enable debugging stats for sys-vm.
# ##################################################################################################
if(FULL_DEBUG_BUILD)
   target_compile_definitions(sys-vm INTERFACE -DSYS_VM_FULL_DEBUG)
endif()

# ##################################################################################################
# Option specific build configurations.
# ##################################################################################################
if(ENABLE_SOFTFLOAT)
   if(NOT TARGET softfloat)
      # Suppress warnings on 3rdParty Library
      add_definitions( -w )
      add_subdirectory(external)
      remove_definitions( -w )
   endif()
   target_compile_definitions(sys-vm INTERFACE -DSYS_VM_SOFTFLOAT)
   target_link_libraries(sys-vm INTERFACE softfloat)
endif()

if(ENABLE_MEMORY_OPS_ALIGNMENT)
   target_compile_definitions(sys-vm INTERFACE -DSYS_VM_ALIGN_MEMORY_OPS)
endif()

# ##################################################################################################
# Build sys-vm tools.
# ##################################################################################################
if(ENABLE_TOOLS)
add_subdirectory(tools)
endif()

# ##################################################################################################
# Build sys-vm tests.
# ##################################################################################################
if(ENABLE_TESTS)
   include(CTest)
   set(CATCH_BUILD_TESTING OFF CACHE BOOL "disable building tests")
   set(CATCH_INSTALL_DOCS OFF CACHE BOOL "disable installing docs")
   set(CATCH_INSTALL_HELPERS OFF CACHE BOOL "disable installing helpers")
   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2)
   include(external/Catch2/contrib/Catch.cmake)
   add_subdirectory(tests)
   if (NOT EXISTS ${CMAKE_BINARY_DIR}/wasms AND ENABLE_SPEC_TESTS)
      include(ExternalProject)
      ExternalProject_Add(
         sys_vm_test_wasms
         GIT_REPOSITORY "https://github.com/SYSIO/sys-vm-test-wasms.git"
         SOURCE_DIR ${CMAKE_BINARY_DIR}/wasms
         BINARY_DIR ${CMAKE_BINARY_DIR}/wasms
         BUILD_ALWAYS 0
         BUILD_COMMAND ""
         TEST_COMMAND ""
         INSTALL_COMMAND ""
         UPDATE_COMMAND ""
         PATCH_COMMAND ""
         CONFIGURE_COMMAND ""
      )
   endif()
endif()

# ##################################################################################################
# Installation.
# ##################################################################################################
if(ENABLE_INSTALL)
   if(SYSVM_INSTALL_COMPONENT)
      set(INSTALL_LIBRARY_TARGETS_COMPONENT_ARGS COMPONENT ${SYSVM_INSTALL_COMPONENT} EXCLUDE_FROM_ALL)
      set(INSTALL_PUBLIC_HEADER_TARGETS_COMPONENT_ARGS COMPONENT ${SYSVM_INSTALL_COMPONENT} EXCLUDE_FROM_ALL)
      set(INSTALL_DIRECTORY_COMPONENT_ARGS COMPONENT ${SYSVM_INSTALL_COMPONENT} EXCLUDE_FROM_ALL)
   else()
      set(INSTALL_LIBRARY_TARGETS_COMPONENT_ARGS COMPONENT Libraries)
      set(INSTALL_PUBLIC_HEADER_TARGETS_COMPONENT_ARGS COMPONENT Headers)
   endif()

   include(GNUInstallDirs)
   message(STATUS "Installing...")
   install(TARGETS sys-vm
            LIBRARY
               DESTINATION ${CMAKE_INSTALL_LIBDIR}
               ${INSTALL_LIBRARY_TARGETS_COMPONENT_ARGS}
            PUBLIC_HEADER
               DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
               ${INSTALL_PUBLIC_HEADER_TARGETS_COMPONENT_ARGS})

   install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/sysio
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      ${INSTALL_DIRECTORY_COMPONENT_ARGS})
endif()

