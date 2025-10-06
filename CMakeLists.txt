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

if(VERSION_SUFFIX)
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

# ##################################################################################################
# Gate against Windows builds.
# ##################################################################################################
if(WIN32)
   message(FATAL_ERROR "SYS-VM currently only builds on Unix like OSes (Linux, Mac OSX)")
endif()

# ##################################################################################################
# Defined options for building or integrating sys-vm.
# ##################################################################################################
include(CMakeDependentOption)
option(
   ENABLE_SOFTFLOAT
   "enable the backend to use deterministic software floating point operations"
   ON
)
option(ENABLE_SOFTFLOAT_INSTALL "enable softfloat install" ON)
option(
   FULL_DEBUG_BUILD
   "enables stack dumping and instruction tracing"
   $<IF:$<CONFIG:Debug>,ON,OFF>
)
option(ENABLE_INSTALL "enable this library to be installed" ON)
option(ENABLE_MEMORY_OPS_ALIGNMENT "enable the backend to obey alignment hints" OFF)
option(ENABLE_TOOLS "enable building of tools" ON)
option(ENABLE_TESTS "enable building of unit tests, spec tests" OFF)

cmake_dependent_option(ENABLE_SPEC_TESTS "enable wasm spec tests" ON "ENABLE_TESTS" ON)
cmake_dependent_option(ENABLE_FUZZ_TESTS "enable fuzz testing" OFF "ENABLE_TESTS" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(dependencies NO_POLICY_SCOPE)
include(target_helpers NO_POLICY_SCOPE)
include(SysVMBuildUtils NO_POLICY_SCOPE)

# ##################################################################################################
# SYS_VM: Create the sys-vm library.
# ##################################################################################################
add_library(sys-vm INTERFACE)

if(ENABLE_SOFTFLOAT)
   target_compile_definitions(sys-vm INTERFACE -DSYS_VM_SOFTFLOAT)
   target_link_libraries(sys-vm INTERFACE softfloat::softfloat)
endif()

# SYS_VM: ignore the C++17 register warning until clean up
target_compile_options(
   sys-vm
   INTERFACE
   $<$<COMPILE_LANGUAGE:CXX>:-Wno-register>
)

# ##################################################################################################
# DEBUG: Enable debugging stats for sys-vm.
# ##################################################################################################
if(FULL_DEBUG_BUILD)
   target_compile_definitions(sys-vm INTERFACE -DSYS_VM_FULL_DEBUG)
endif()

# ##################################################################################################
# MEMORY ALIGNMENT
# ##################################################################################################
if(ENABLE_MEMORY_OPS_ALIGNMENT)
   target_compile_definitions(sys-vm INTERFACE -DSYS_VM_ALIGN_MEMORY_OPS)
endif()

# ##################################################################################################
# TOOLS: Build sys-vm tools.
# ##################################################################################################
if(ENABLE_TOOLS)
   add_subdirectory(tools)
endif()

# ##################################################################################################
# Build sys-vm tests.
# ##################################################################################################
if(ENABLE_TESTS)
   include(CTest)

   add_subdirectory(tests)
endif()

# ##################################################################################################
# Installation.
# ##################################################################################################
if(ENABLE_INSTALL)
   include(GNUInstallDirs)
   message(STATUS "Installing...")
   install(
      TARGETS sys-vm
      EXPORT sys-vm-config
      RUNTIME DESTINATION bin
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      PUBLIC_HEADER DESTINATION include
   )

   install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

   install(
      EXPORT sys-vm-config
      FILE sys-vm-config.cmake
      NAMESPACE sys-vm::
      DESTINATION share/sys-vm
   )

   # IF SOFTFLOAT IS ENABLED, PASS ALONG THE INSTALL ARTIFACTS
   if(ENABLE_SOFTFLOAT AND ENABLE_SOFTFLOAT_INSTALL)
      install(DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/softfloat/" DESTINATION share/softfloat)
      install(DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/softfloat/" DESTINATION include/softfloat)
      install(FILES "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libsoftfloat.a" DESTINATION lib)

      # INCLUDE DEBUG ARTIFACTS IF THEY EXIST
      if(IS_DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/include/softfloat")
         install(DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/include/softfloat/" DESTINATION debug/include/softfloat)
      endif()
      if(EXISTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib/libsoftfloat.a")
         install(FILES "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib/libsoftfloat.a" DESTINATION debug/lib)
      endif()
   endif()
endif()

