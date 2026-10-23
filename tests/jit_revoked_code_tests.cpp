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
#include <sysio/vm/watchdog.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
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

// A host function may run another backend. The nested run installs its own spans and must put
// the outer run's back on return; otherwise a later fault of the outer run no longer matches
// them and is declined to the previous handler.
namespace {
   // (module (import "env" "host.call" (func)) (func (export "f") (call 0)))
   sysio::vm::wasm_code make_host_call_module() {
      return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,                         // magic, version
               0x01, 0x04, 0x01, 0x60, 0x00, 0x00,                                     // type: () -> ()
               0x02, 0x11, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x09, 0x68, 0x6f, 0x73, 0x74, // import env.host.call
               0x2e, 0x63, 0x61, 0x6c, 0x6c, 0x00, 0x00,
               0x03, 0x02, 0x01, 0x00,                                                 // one function of type 0
               0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x01,                               // export "f" = func 1
               0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x00, 0x0b };                       // body: call 0, end
   }

   struct nested_host {
      backend_t*                     inner               = nullptr;
      sysio::vm::growable_allocator* outer_allocator     = nullptr;
      bool                           revoke_outer        = false;
      bool                           outer_span_restored = false;

      void host_call() {
         inner->call("env", "f"); // a nested run with its own spans
         const auto outer_span = outer_allocator->get_code_span();
         outer_span_restored   = sysio::vm::code_memory_range.data() == outer_span.data() &&
                               sysio::vm::code_memory_range.size() == outer_span.size();
         if (revoke_outer) {
            // The peer-thread race state, applied to the OUTER module while its run is still active:
            // returning into its revoked code must unwind, which needs the outer span back in place.
            outer_allocator->disable_code();
            sysio::vm::timed_run_has_timed_out.store(false, std::memory_order_release);
         }
      }
   };
   using nested_rhf_t     = sysio::vm::registered_host_functions<nested_host>;
   using nested_backend_t = sysio::vm::backend<nested_rhf_t, sysio::vm::jit>;
}

TEST_CASE("A nested run restores the outer run's spans, so a later outer fault still unwinds",
          "[jit][revoked_code_recovery][nested]") {
   timeout_flag_guard flag_guard;
   nested_rhf_t::add<&nested_host::host_call>("env", "host.call");

   sysio::vm::wasm_allocator inner_wa;
   auto                      inner_code = make_empty_export_module();
   backend_t                 inner(inner_code, &inner_wa);

   nested_host      host;
   nested_backend_t outer(make_host_call_module(), host, &wa);
   auto&            outer_allocator = outer.get_module().allocator;
   host.inner                       = &inner;
   host.outer_allocator             = &outer_allocator;

   // Control: the nested call returns and the outer spans are back in place afterwards.
   CHECK_NOTHROW(outer.call(host, "env", "f"));
   CHECK(host.outer_span_restored);

   host.revoke_outer = true;
   auto reenable     = sysio::vm::scope_guard{ [&] {
      if (outer_allocator.code_disabled())
         outer_allocator.enable_code(true);
   } };
   worker_result result;
   std::thread   worker{ [&] {
      try {
         outer.call(host, "env", "f");
         result.returned_normally = true;
      } catch (...) {
         result.error = std::current_exception();
      }
   } };
   worker.join();
   CHECK(!result.returned_normally);
   REQUIRE(result.error);
   CHECK_THROWS_AS(std::rethrow_exception(result.error), sysio::vm::wasm_memory_exception);
}

// timed_run is the production entry point. Its registration and exit transitions are serialized, so a timeout
// stays in force, with the pages revoked, until the last registered run exits, and that exit restores both.
// The watchdog here fires when the test says so, which pins each interleaving instead of racing for it.
namespace {
   struct manual_watchdog {
      std::function<void()> timeout;   // the callback timed_run armed; null outside a run
      std::function<void()> on_disarm; // runs as the run's guard is destroyed, before its exit transition

      struct guard {
         manual_watchdog* owner;
         explicit guard(manual_watchdog* watchdog) : owner(watchdog) {}
         guard(const guard&)            = delete;
         guard& operator=(const guard&) = delete;
         ~guard() {
            owner->timeout = nullptr;
            if (owner->on_disarm)
               owner->on_disarm();
         }
      };

      template <typename F>
      [[nodiscard]] guard scoped_run(F&& callback) {
         timeout = std::forward<F>(callback);
         return guard(this);
      }
      void fire() { timeout(); }
   };

   // The import is the witness: reaching it means the module body executed.
   struct witness_host {
      bool body_executed = false;
      void host_call() { body_executed = true; }
   };
   using witness_rhf_t     = sysio::vm::registered_host_functions<witness_host>;
   using witness_backend_t = sysio::vm::backend<witness_rhf_t, sysio::vm::jit>;

   void register_witness_host() {
      static const bool registered = (witness_rhf_t::add<&witness_host::host_call>("env", "host.call"), true);
      (void)registered;
   }

   // Stands in for another run of the module that is registered but not executing.
   struct registered_peer {
      sysio::vm::growable_allocator& allocator;
      bool                           held = true;
      explicit registered_peer(sysio::vm::growable_allocator& module_allocator) : allocator(module_allocator) {
         ++sysio::vm::total_timed_run_in_progress;
         ++allocator.timed_run_in_progress;
      }
      void release() {
         if (held) {
            --sysio::vm::total_timed_run_in_progress;
            --allocator.timed_run_in_progress;
            held = false;
         }
      }
      ~registered_peer() { release(); }
   };

   bool timed_out() { return sysio::vm::timed_run_has_timed_out.load(std::memory_order_acquire); }

   // The production shape: one backend compiles the module (single parsing, no execution context of its own), and
   // every thread that runs it holds a backend sharing that module with its own context and memory.
   using witness_context_t = sysio::vm::jit_execution_context<witness_rhf_t>;

   struct shared_runner {
      witness_host              host;
      sysio::vm::wasm_allocator memory;
      witness_context_t         context;
      witness_backend_t         bkend;

      explicit shared_runner(witness_backend_t& compiled) {
         bkend.share(compiled);
         context.set_module(&compiled.get_module());
         bkend.set_context(&context);
         bkend.reset_max_call_depth();
         bkend.reset_max_pages();
         bkend.set_wasm_allocator(&memory);
         bkend.initialize(&host);
      }
      void run_f() { bkend.call(host, "env", "f"); }
   };
}

TEST_CASE("A timeout fired after the entry check stops the run before its first instruction, and stays in force "
          "until the last registered run exits",
          "[jit][revoked_code_recovery][timed_run]") {
   timeout_flag_guard flag_guard;
   register_witness_host();
   witness_host      host;
   witness_backend_t bkend(make_host_call_module(), host, &wa);
   auto&             allocator = bkend.get_module().allocator;
   auto              reenable  = sysio::vm::scope_guard{ [&] {
      if (allocator.code_disabled())
         allocator.enable_code(true);
   } };
   manual_watchdog wd;

   // Another run of the module is registered throughout the first run, so that run's exit is not the last.
   registered_peer peer(allocator);
   bool            revoked_before_first_instruction   = false;
   bool            timed_out_before_first_instruction = false;
   CHECK_THROWS_AS(bkend.timed_run(wd,
                                   [&] {
                                      // Past timed_run's entry check, before the module's first instruction.
                                      wd.fire();
                                      revoked_before_first_instruction   = allocator.code_disabled();
                                      timed_out_before_first_instruction = timed_out();
                                      bkend.call(host, "env", "f");
                                   }),
                   sysio::vm::timeout_exception);
   CHECK(revoked_before_first_instruction);
   CHECK(timed_out_before_first_instruction);
   CHECK(!host.body_executed);
   // The peer is still registered: the exit restored nothing.
   CHECK(allocator.code_disabled());
   CHECK(timed_out());

   // The next run finds the timeout in force and does not execute; being the last registered run, its exit
   // clears the flag and re-enables the pages.
   peer.release();
   CHECK_THROWS_AS(bkend.timed_run(wd, [&] { bkend.call(host, "env", "f"); }), sysio::vm::timeout_exception);
   CHECK(!host.body_executed);
   CHECK(!allocator.code_disabled());
   CHECK(!timed_out());

   // Restored: the module runs again.
   bkend.timed_run(wd, [&] { bkend.call(host, "env", "f"); });
   CHECK(host.body_executed);
}

TEST_CASE("A run that outlives a newer run's timeout restores the flag and the pages only as the last registered "
          "run",
          "[jit][revoked_code_recovery][timed_run]") {
   timeout_flag_guard flag_guard;
   register_witness_host();
   auto                     code = make_host_call_module();
   sysio::vm::wasm_code_ptr code_ptr(code.data(), code.size());
   witness_backend_t        compiled(code_ptr, code.size(), nullptr, sysio::vm::default_options{}, true, false);
   auto&                    allocator = compiled.get_module().allocator;
   auto                     reenable  = sysio::vm::scope_guard{ [&] {
      if (allocator.code_disabled())
         allocator.enable_code(true);
   } };
   shared_runner older(compiled);
   shared_runner newer(compiled);

   // While the older run is still registered (its guard is being disarmed, ahead of its exit transition), the
   // newer run of the same module times out on its own thread and exits. That exit must leave the timeout in
   // force; the older run's exit, the last one, restores both.
   manual_watchdog older_wd;
   manual_watchdog newer_wd;
   bool            newer_timed_out                 = false;
   bool            restored_while_older_registered = false;
   older_wd.on_disarm = [&] {
      std::thread worker{ [&] {
         try {
            newer.bkend.timed_run(newer_wd, [&] {
               newer_wd.fire();
               newer.run_f();
            });
         } catch (const sysio::vm::timeout_exception&) {
            newer_timed_out = true;
         } catch (...) {
         }
      } };
      worker.join();
      restored_while_older_registered = !allocator.code_disabled() || !timed_out();
   };
   older.bkend.timed_run(older_wd, [&] { older.run_f(); }); // its body ran before the newer timeout
   CHECK(older.host.body_executed);
   CHECK(newer_timed_out);
   CHECK(!newer.host.body_executed);
   CHECK(!restored_while_older_registered);
   CHECK(!allocator.code_disabled());
   CHECK(!timed_out());

   // Restored: both threads' backends run the module again.
   older.bkend.timed_run(older_wd, [&] { older.run_f(); });
   newer.bkend.timed_run(newer_wd, [&] { newer.run_f(); });
   CHECK(newer.host.body_executed);
}

#endif // SYS_VM_HAS_JIT_BACKEND && SYS_VM_TARGET_X86_64
