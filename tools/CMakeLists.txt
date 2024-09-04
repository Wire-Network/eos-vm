# ##################################################################################################
# Create the sys-vm interpreter driver.
# ##################################################################################################
add_executable(sys-vm-interp ${CMAKE_CURRENT_SOURCE_DIR}/interp.cpp)
target_link_libraries(sys-vm-interp sys-vm)

add_executable(bench-interp ${CMAKE_CURRENT_SOURCE_DIR}/bench_interp.cpp)
target_link_libraries(bench-interp sys-vm)

add_executable(hello-driver ${CMAKE_CURRENT_SOURCE_DIR}/hello_driver.cpp)
target_link_libraries(hello-driver sys-vm)

add_executable(sys-vm-addr2line addr2line.cpp)
target_link_libraries(sys-vm-addr2line sys-vm)
