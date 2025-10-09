# ##################################################################################################
# Create the sys-vm interpreter driver.
# ##################################################################################################
add_executable(sys-vm-interp ${CMAKE_CURRENT_SOURCE_DIR}/interp.cpp)
setup_vm_target(sys-vm-interp)
target_link_libraries(sys-vm-interp sys-vm)

add_executable(bench-interp ${CMAKE_CURRENT_SOURCE_DIR}/bench_interp.cpp)
setup_vm_target(bench-interp)
target_link_libraries(bench-interp sys-vm)

add_executable(hello-driver ${CMAKE_CURRENT_SOURCE_DIR}/hello_driver.cpp)
setup_vm_target(hello-driver)
target_link_libraries(hello-driver sys-vm)

add_executable(sys-vm-addr2line addr2line.cpp)
setup_vm_target(sys-vm-addr2line)
target_link_libraries(sys-vm-addr2line sys-vm)
