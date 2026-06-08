#include <atomic>
#include <chrono>
#include <sysio/vm/backend.hpp>
#include <sysio/vm/exceptions.hpp>
#include <sysio/vm/watchdog.hpp>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

using sysio::vm::watchdog;

extern sysio::vm::wasm_allocator wa;

namespace {
struct execution_thread_interrupt_watchdog {
   /// Joins the watchdog thread when timed_run leaves its protected execution scope.
   struct guard {
      std::thread callback_thread;

      ~guard() {
         if (callback_thread.joinable())
            callback_thread.join();
      }
   };

   /// Fires after JIT execution starts so the test proves generated code can be interrupted.
   template <typename F>
   guard scoped_run(F&& callback) {
      return guard{ std::thread{ [callback = std::forward<F>(callback)]() mutable {
         std::this_thread::sleep_for(std::chrono::milliseconds(10));
         callback();
      } } };
   }
};

struct cooperative_timeout_watchdog {
   /// Joins the callback thread after the protected scope observes the timeout.
   struct guard {
      std::thread callback_thread;

      ~guard() {
         if (callback_thread.joinable())
            callback_thread.join();
      }
   };

   /// Fires from another thread but asks timed_run not to send a signal into the execution thread.
   template <typename F>
   guard scoped_run(F&& callback) {
      return guard{ std::thread{ [callback = std::forward<F>(callback)]() mutable {
         std::this_thread::sleep_for(std::chrono::milliseconds(10));
         callback();
      } } };
   }

   /// Identifies watchdog expiry paths that should be observed without asynchronously interrupting execution.
   bool should_interrupt_execution_thread() const {
      return false;
   }
};

struct late_execution_thread_interrupt_watchdog {
   template <typename F>
   struct guard {
      F callback;

      ~guard() {
         std::thread callback_thread{[this]() { callback(); }};
         callback_thread.join();
      }
   };

   /// Fires when timed_run leaves the user callback, after the JIT signal frame has been restored.
   template <typename F>
   guard<std::decay_t<F>> scoped_run(F&& callback) {
      return guard<std::decay_t<F>>{std::forward<F>(callback)};
   }
};

/// Builds a module with an exported function that returns before the watchdog guard is destroyed.
std::vector<uint8_t> make_return_42_wasm_module() {
   /*
    * (module
    *   (func (export "run") (result i32)
    *     i32.const 42))
    */
   return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00,
            0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e,
            0x00, 0x00, 0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b };
}

/// Builds a module with an exported function whose body never reaches a timed_run boundary.
std::vector<uint8_t> make_infinite_loop_wasm_module() {
   /*
    * (module
    *   (func (export "loop")
    *     (loop br 0)))
    */
   return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
            0x03, 0x02, 0x01, 0x00, 0x07, 0x08, 0x01, 0x04, 0x6c, 0x6f, 0x6f, 0x70, 0x00, 0x00,
            0x0a, 0x09, 0x01, 0x07, 0x00, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b };
}
} // namespace

TEST_CASE("watchdog interrupt", "[watchdog_interrupt]") {
   std::atomic<bool> okay = false;
   watchdog          w{ std::chrono::milliseconds(50) };
   {
      auto g = w.scoped_run([&]() { okay = true; });
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
   }
   CHECK(okay);
}

TEST_CASE("watchdog no interrupt", "[watchdog_no_interrupt]") {
   std::atomic<bool> okay = true;
   watchdog          w{ std::chrono::milliseconds(50) };
   {
      auto g = w.scoped_run([&]() { okay = false; });
   } // the guard goes out of scope here, cancelling the timer
   std::this_thread::sleep_for(std::chrono::milliseconds(100));
   CHECK(okay);
}

#if SYS_VM_HAS_AARCH64_JIT_BACKEND
TEST_CASE("AArch64 JIT watchdog interrupt signals the execution thread", "[watchdog_interrupt][jit][aarch64]") {
   using backend_t = sysio::vm::backend<std::nullptr_t, sysio::vm::jit>;
   auto      code  = make_infinite_loop_wasm_module();
   backend_t bkend(code, &wa);

   CHECK_THROWS_AS(bkend.timed_run(execution_thread_interrupt_watchdog{}, [&]() {
                      bkend.call("env", "loop");
                   }),
                   sysio::vm::timeout_exception);
}

TEST_CASE("AArch64 JIT cooperative watchdog does not signal the execution thread", "[watchdog_interrupt][jit][aarch64]") {
   using backend_t = sysio::vm::backend<std::nullptr_t, sysio::vm::jit>;
   auto      code  = make_infinite_loop_wasm_module();
   backend_t bkend(code, &wa);

   CHECK_THROWS_AS(bkend.timed_run(cooperative_timeout_watchdog{}, []() {
                      std::this_thread::sleep_for(std::chrono::milliseconds(50));
                   }),
                   sysio::vm::timeout_exception);
}

TEST_CASE("AArch64 JIT watchdog late signal is reported as timeout", "[watchdog_interrupt][jit][aarch64]") {
   using backend_t = sysio::vm::backend<std::nullptr_t, sysio::vm::jit>;
   auto      code  = make_return_42_wasm_module();
   backend_t bkend(code, &wa);

   CHECK_THROWS_AS(bkend.timed_run(late_execution_thread_interrupt_watchdog{}, [&]() {
                      bkend.call("env", "run");
                   }),
                   sysio::vm::timeout_exception);
}
#endif
