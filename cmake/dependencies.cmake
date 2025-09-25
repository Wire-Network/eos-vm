# LOAD CMAKE TOOLS
# find_package(PkgConfig REQUIRED)

include(GNUInstallDirs)

# Catch2
find_package(Catch2 CONFIG REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/catch2")
file(MAKE_DIRECTORY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/catch2")
file(COPY "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/catch.hpp" DESTINATION "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/catch2")

# Softfloat
find_package(softfloat CONFIG REQUIRED) 