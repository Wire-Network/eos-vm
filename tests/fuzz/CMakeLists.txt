include(CheckCXXSourceCompiles)

set(SYS_VM_FUZZ_SANITIZER_FLAGS "-fsanitize=fuzzer,address")
set(SYS_VM_FUZZ_DEBUG_FLAGS "-g" "-O1" "-gline-tables-only")

set(CMAKE_REQUIRED_FLAGS_SAVE "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${SYS_VM_FUZZ_SANITIZER_FLAGS}")
check_cxx_source_compiles([=[
   #include <cstddef>
   #include <cstdint>

   extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t*, std::size_t) {
      return 0;
   }
]=] SYS_VM_HAS_FUZZER_ADDRESS_SANITIZER)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_SAVE}")

if(NOT SYS_VM_HAS_FUZZER_ADDRESS_SANITIZER)
    message(FATAL_ERROR "ENABLE_FUZZ_TESTS requires compiler/linker support for ${SYS_VM_FUZZ_SANITIZER_FLAGS}")
endif()

function(add_libfuzzer_test name)
    set(multi_arg_options "SOURCES")
    cmake_parse_arguments(
        "add_libfuzzer_test" "" "" "${multi_arg_options}" ${ARGN})
    if("${add_libfuzzer_test_SOURCES}" STREQUAL "")
        message(FATAL_ERROR "Source files must be specified")
    endif()
    add_executable(LLVMFuzzer-${name}
        ${add_libfuzzer_test_SOURCES}
    )
    setup_vm_target(LLVMFuzzer-${name})
    target_compile_options(LLVMFuzzer-${name} PUBLIC ${SYS_VM_FUZZ_DEBUG_FLAGS} ${SYS_VM_FUZZ_SANITIZER_FLAGS})
    target_link_libraries(LLVMFuzzer-${name} sys-vm ${SYS_VM_FUZZ_SANITIZER_FLAGS})
endfunction()

add_libfuzzer_test(wasm_fuzz_test SOURCES fuzz_driver.cpp)

add_test(NAME wasm_libfuzzer COMMAND ${CMAKE_CURRENT_BINARY_DIR}/LLVMFuzzer-wasm_fuzz_test -runs=1 -jobs=1 -rss_limit_mb=6144 -timeout=30 ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
