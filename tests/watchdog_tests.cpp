#include <atomic>
#include <chrono>
#include <sysio/vm/backend.hpp>
#include <sysio/vm/exceptions.hpp>
#include <sysio/vm/watchdog.hpp>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

using sysio::vm::watchdog;

extern sysio::vm::wasm_allocator wa;

namespace {
struct cooperative_interrupt_watchdog {
   /// Keeps the watchdog scope API compatible with sysio::vm::watchdog.
   struct guard {};

   /// Fires the callback from a watchdog thread, matching chain interrupt timers.
   template <typename F>
   guard scoped_run(F&& callback) {
      std::thread callback_thread{ std::forward<F>(callback) };
      callback_thread.join();
      return {};
   }

   /// Cooperative interrupts must be observed at safe timed_run boundaries.
   bool should_interrupt_execution_thread() const { return false; }
};

/// Builds a minimal valid module so backend::timed_run has initialized allocator state.
std::vector<uint8_t> make_empty_wasm_module() { return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 }; }
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

#if SYS_VM_TARGET_ARM64
TEST_CASE("cooperative watchdog interrupt does not signal execution thread", "[watchdog_interrupt][aarch64]") {
   using backend_t = sysio::vm::backend<std::nullptr_t, sysio::vm::interpreter>;
   auto      code  = make_empty_wasm_module();
   backend_t bkend(code, &wa);

   bool executed = false;
   CHECK_THROWS_AS(bkend.timed_run(cooperative_interrupt_watchdog{}, [&]() { executed = true; }),
                   sysio::vm::timeout_exception);
   CHECK_FALSE(executed);
}
#endif
