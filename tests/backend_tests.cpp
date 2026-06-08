#include "utils.hpp"
#include <catch2/catch.hpp>
#include <sysio/vm/backend.hpp>

using namespace sysio::vm;

extern wasm_allocator wa;

TEST_CASE("JIT capability flags match configured compile definitions", "[config]") {
#if SYS_VM_HAS_JIT_BACKEND
   CHECK(sys_vm_has_jit_backend);
#else
   CHECK_FALSE(sys_vm_has_jit_backend);
#endif

#if SYS_VM_HAS_JIT_PROFILE
   CHECK(sys_vm_has_jit_profile);
#else
   CHECK_FALSE(sys_vm_has_jit_profile);
#endif

#if SYS_VM_TARGET_X86_64
   CHECK(sys_vm_target_x86_64);
   CHECK(sys_vm_amd64);
#endif

#if SYS_VM_TARGET_ARM64
   CHECK(sys_vm_target_aarch64);
#endif

#if SYS_VM_HAS_AARCH64_JIT_BACKEND
   CHECK(sys_vm_has_aarch64_jit_backend);
#else
   CHECK_FALSE(sys_vm_has_aarch64_jit_backend);
#endif
}

BACKEND_TEST_CASE("Tests that the arguments of top level calls are validated", "[call_typecheck]") {
   /*
    * (module
    *  (func (export "f0"))
    *  (func (export "f1") (param i32))
    * )
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x60,
                                 0x00, 0x00, 0x60, 0x01, 0x7f, 0x00, 0x03, 0x03, 0x02, 0x00, 0x01, 0x07,
                                 0x0b, 0x02, 0x02, 0x66, 0x30, 0x00, 0x00, 0x02, 0x66, 0x31, 0x00, 0x01,
                                 0x0a, 0x07, 0x02, 0x02, 0x00, 0x0b, 0x02, 0x00, 0x0b };

   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(code, &wa);

   CHECK_THROWS_AS(bkend.call("env", "f0", 0), std::exception);                        // too many arguments
   CHECK_THROWS_AS(bkend.call("env", "f1"), std::exception);                           // too few arguments
   CHECK_THROWS_AS(bkend.call("env", "f1", UINT64_C(0)), std::exception);              // wrong type of argument
   CHECK_THROWS_AS(bkend.call("env", "f1", UINT32_C(0), UINT32_C(0)), std::exception); // too many arguments
}
