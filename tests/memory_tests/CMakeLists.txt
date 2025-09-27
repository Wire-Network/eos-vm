file (GLOB memory_tests_files *.cpp)

add_executable( memory_tests ${memory_tests_files} )
setup_vm_target(memory_tests)
target_link_libraries( memory_tests sys-vm Catch2::Catch2 )

catch_discover_tests( memory_tests )
