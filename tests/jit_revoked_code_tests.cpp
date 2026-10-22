// Recovery from a fault on revoked JIT code pages, through the real machinery. An expiring
// timed_run revokes the shared module's code pages (disable_code, PROT_NONE), and a peer's
// exit guard clears timed_run_has_timed_out before re-enabling them, so a run entering the
// code in that window faults with the flag already false. The handler must still unwind the
// run into an exception; gating on the flag ended the process (wire-sysio CI run 39807473127).
// The state is built directly rather than raced: compile, revoke, clear the flag, execute.
#include <sysio/vm/backend.hpp>
#include <sysio/vm/config.hpp>
#include <sysio/vm/exceptions.hpp>
#include <sysio/vm/signals.hpp>
#include <sysio/vm/utils.hpp>

#include <atomic>
#include <cstdint>
#include <exception>
#include <thread>

#include <catch2/catch.hpp>

#if SYS_VM_HAS_JIT_BACKEND && SYS_VM_TARGET_X86_64

extern sysio::vm::wasm_allocator wa;

namespace {
   using backend_t = sysio::vm::backend<std::nullptr_t, sysio::vm::jit>;

   // (module (func (export "f")))
   sysio::vm::wasm_code make_empty_export_module() {
      return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic, version
               0x01, 0x04, 0x01, 0x60, 0x00, 0x00,             // type: () -> ()
               0x03, 0x02, 0x01, 0x00,                         // one function of type 0
               0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x00,       // export "f" = func 0
               0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b };           // body: no locals, end
   }

   struct worker_result {
      bool               returned_normally = false;
      std::exception_ptr error;
   };

   // Executes the export on its own thread, as a read-only thread would, and reports what
   // happened without asserting off the main thread.
   worker_result call_f_on_worker(backend_t& bkend) {
      worker_result result;
      std::thread   worker{ [&] {
         try {
            bkend.call("env", "f");
            result.returned_normally = true;
         } catch (...) {
            result.error = std::current_exception();
         }
      } };
      worker.join();
      return result;
   }

   // Leaves the shared timeout state the way the other tests expect it, even on a failed REQUIRE.
   struct timeout_flag_guard {
      ~timeout_flag_guard() { sysio::vm::timed_run_has_timed_out.store(false, std::memory_order_release); }
   };
}

TEST_CASE("Executing revoked JIT code with the timeout flag clear unwinds into an exception",
          "[jit][revoked_code_recovery]") {
   timeout_flag_guard flag_guard;
   auto               code = make_empty_export_module();
   backend_t          bkend(code, &wa);
   auto&              allocator = bkend.get_module().allocator;
   REQUIRE(!allocator.get_code_span().empty());

   // Control: the module runs when its code is executable.
   {
      auto result = call_f_on_worker(bkend);
      REQUIRE(!result.error);
      CHECK(result.returned_normally);
   }

   // Revoke the code pages as an expiring watchdog does, then clear the flag as a peer thread's
   // exit guard does before it re-enables the pages.
   REQUIRE(allocator.disable_code());
   REQUIRE(allocator.code_disabled());
   auto reenable = sysio::vm::scope_guard{ [&] {
      if (allocator.code_disabled())
         allocator.enable_code(true);
   } };
   sysio::vm::timed_run_has_timed_out.store(false, std::memory_order_release);

   // The run must come back as the ordinary out-of-bounds unwind, not reach the previous handler.
   auto result = call_f_on_worker(bkend);
   CHECK(!result.returned_normally);
   REQUIRE(result.error);
   CHECK_THROWS_AS(std::rethrow_exception(result.error), sysio::vm::wasm_memory_exception);

   // Once the pages are executable again the same module runs normally: the recovery left the
   // backend, the allocator and the handler state intact.
   REQUIRE(allocator.enable_code(true));
   REQUIRE(!allocator.code_disabled());
   {
      auto after = call_f_on_worker(bkend);
      REQUIRE(!after.error);
      CHECK(after.returned_normally);
   }
}

#endif // SYS_VM_HAS_JIT_BACKEND && SYS_VM_TARGET_X86_64
