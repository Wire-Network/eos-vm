#include <sysio/vm/backend.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

using namespace sysio::vm;

namespace {
constexpr uint32_t         benchmark_input      = 1000;
constexpr uint32_t         benchmark_iterations = 20000;
constexpr std::string_view module_name          = "env";
constexpr std::string_view function_name        = "sum_to";

/**
 * Returns a loop-heavy WASM module supported by both the interpreter and the AArch64 JIT.
 *
 * The exported function computes the sum of integers in [0, n).
 */
std::vector<uint8_t> make_sum_to_module() {
   return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
            0x03, 0x02, 0x01, 0x00, 0x07, 0x0a, 0x01, 0x06, 0x73, 0x75, 0x6d, 0x5f, 0x74, 0x6f, 0x00, 0x00,
            0x0a, 0x2d, 0x01, 0x2b, 0x01, 0x02, 0x7f, 0x41, 0x00, 0x21, 0x01, 0x41, 0x00, 0x21, 0x02, 0x02,
            0x40, 0x03, 0x40, 0x20, 0x01, 0x20, 0x00, 0x4f, 0x0d, 0x01, 0x20, 0x02, 0x20, 0x01, 0x6a, 0x21,
            0x02, 0x20, 0x01, 0x41, 0x01, 0x6a, 0x21, 0x01, 0x0c, 0x00, 0x0b, 0x0b, 0x20, 0x02, 0x0b };
}

/**
 * Measures repeated calls to the benchmark function and returns elapsed milliseconds.
 */
template <typename Backend>
double benchmark_backend(Backend& backend, uint64_t& checksum) {
   for (uint32_t i = 0; i < 100; ++i) {
      checksum += backend.call_with_return(module_name, function_name, uint32_t{ benchmark_input })->to_ui32();
   }

   const auto started = std::chrono::steady_clock::now();
   for (uint32_t i = 0; i < benchmark_iterations; ++i) {
      checksum += backend.call_with_return(module_name, function_name, uint32_t{ benchmark_input })->to_ui32();
   }
   const auto elapsed = std::chrono::steady_clock::now() - started;
   return std::chrono::duration<double, std::milli>(elapsed).count();
}
} // namespace

/**
 * Runs an interpreter vs AArch64 JIT microbenchmark.
 */
int main() {
#if !SYS_VM_HAS_JIT_BACKEND || !SYS_VM_TARGET_ARM64
   std::cerr << "AArch64 JIT is not enabled in this build.\n";
   return 1;
#else
   wasm_allocator interpreter_allocator;
   wasm_allocator jit_allocator;
   auto           interpreter_code = make_sum_to_module();
   auto           jit_code         = make_sum_to_module();

   backend<std::nullptr_t, interpreter>           interpreter_backend(interpreter_code, &interpreter_allocator);
   backend<std::nullptr_t, jit> jit_backend(jit_code, &jit_allocator);

   uint64_t     interpreter_checksum = 0;
   uint64_t     jit_checksum         = 0;
   const double interpreter_ms       = benchmark_backend(interpreter_backend, interpreter_checksum);
   const double jit_ms               = benchmark_backend(jit_backend, jit_checksum);

   if (interpreter_checksum != jit_checksum) {
      std::cerr << "checksum mismatch: interpreter=" << interpreter_checksum << " jit=" << jit_checksum << "\n";
      return 2;
   }

   std::cout << "kernel=sum_to"
             << "\ninput=" << benchmark_input << "\niterations=" << benchmark_iterations
             << "\ninterpreter_ms=" << interpreter_ms << "\naarch64_jit_ms=" << jit_ms
             << "\nspeedup=" << (interpreter_ms / jit_ms) << "\nchecksum=" << jit_checksum << "\n";
   return 0;
#endif
}
