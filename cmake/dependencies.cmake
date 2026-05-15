# LOAD CMAKE TOOLS
# find_package(PkgConfig REQUIRED)

include(GNUInstallDirs)

# Softfloat
find_package(Threads REQUIRED)
find_package(softfloat CONFIG REQUIRED)
