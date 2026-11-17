#include "utils.hpp"

#include <sysio/vm/backend.hpp>
#include <sysio/vm/exceptions.hpp>

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

using namespace sysio::vm;

extern wasm_allocator wa;

namespace {

/// Wall-clock time each host call spends, so host calls dominate the run as an expensive intrinsic does.
constexpr std::chrono::microseconds host_call_duration{ 200 };
/// Time from the start of the run until the watchdog expires.
constexpr std::chrono::milliseconds timeout_after{ 20 };
/// Host calls the export makes when nothing stops it. Far past the timeout, and at two branch polls per iteration
/// well under timeout_poll_interval, so only a host-call timeout check can stop the AArch64 JIT run early.
constexpr uint32_t host_call_limit = 5000;
static_assert(2 * host_call_limit < timeout_poll_interval);
/// Host calls a run makes when it has no timeout.
constexpr uint32_t short_count = 3;

/// Host whose one import spends host_call_duration, and counts the calls that see the watchdog expired.
struct timeout_host {
   const std::atomic<bool>* expired            = nullptr;
   uint32_t                 calls              = 0;
   uint32_t                 calls_after_expiry = 0;

   void work() {
      ++calls;
      if (expired->load(std::memory_order_acquire))
         ++calls_after_expiry;
      const std::chrono::steady_clock::time_point until = std::chrono::steady_clock::now() + host_call_duration;
      while (std::chrono::steady_clock::now() < until) {}
   }
};

using timeout_rhf_t = registered_host_functions<timeout_host>;

void register_timeout_host() {
   static const bool registered = (timeout_rhf_t::add<&timeout_host::work>("env", "work"), true);
   (void)registered;
}

/// Joins the watchdog thread when timed_run leaves its protected scope.
struct watchdog_thread_guard {
   std::thread thread;

   ~watchdog_thread_guard() {
      if (thread.joinable())
         thread.join();
   }
};

/// Expires timeout_after into the run: fires the VM's timeout first, then marks the expiry for the host, so a host
/// call that sees the expiry is one the VM stops when it returns.
struct marking_watchdog {
   std::atomic<bool>& expired;

   template <typename F>
   watchdog_thread_guard scoped_run(F&& callback) {
      return watchdog_thread_guard{ std::thread{ [&expired = expired, callback = std::forward<F>(callback)]() mutable {
         std::this_thread::sleep_for(timeout_after);
         callback();
         expired.store(true, std::memory_order_release);
      } } };
   }
};

/// Builds a module whose export calls the host import directly once per iteration, up to its argument's count.
std::vector<uint8_t> make_direct_host_call_loop_wasm_module() {
   /*
    * (module
    *   (import "env" "work" (func $work))
    *   (func (export "run") (param $n i32) (result i32) (local $i i32)
    *     block
    *       loop
    *         local.get $i
    *         local.get $n
    *         i32.ge_u
    *         br_if 1
    *         call $work
    *         local.get $i
    *         i32.const 1
    *         i32.add
    *         local.set $i
    *         br 0
    *       end
    *     end
    *     local.get $i))
    */
   return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x09, 0x02, 0x60, 0x00, 0x00, 0x60, 0x01,
            0x7f, 0x01, 0x7f, 0x02, 0x0c, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x04, 0x77, 0x6f, 0x72, 0x6b, 0x00,
            0x00, 0x03, 0x02, 0x01, 0x01, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x01, 0x0a, 0x20,
            0x01, 0x1e, 0x01, 0x01, 0x7f, 0x02, 0x40, 0x03, 0x40, 0x20, 0x01, 0x20, 0x00, 0x4f, 0x0d, 0x01,
            0x10, 0x00, 0x20, 0x01, 0x41, 0x01, 0x6a, 0x21, 0x01, 0x0c, 0x00, 0x0b, 0x0b, 0x20, 0x01, 0x0b };
}

/// Builds a module whose export calls the host import through a table once per iteration, up to its argument's count.
std::vector<uint8_t> make_indirect_host_call_loop_wasm_module() {
   /*
    * (module
    *   (type $void (func))
    *   (import "env" "work" (func $work (type $void)))
    *   (table 1 funcref)
    *   (elem (i32.const 0) $work)
    *   (func (export "run") (param $n i32) (result i32) (local $i i32)
    *     block
    *       loop
    *         local.get $i
    *         local.get $n
    *         i32.ge_u
    *         br_if 1
    *         i32.const 0
    *         call_indirect (type $void)
    *         local.get $i
    *         i32.const 1
    *         i32.add
    *         local.set $i
    *         br 0
    *       end
    *     end
    *     local.get $i))
    */
   return { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x09, 0x02, 0x60, 0x00, 0x00, 0x60, 0x01, 0x7f,
            0x01, 0x7f, 0x02, 0x0c, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x04, 0x77, 0x6f, 0x72, 0x6b, 0x00, 0x00, 0x03,
            0x02, 0x01, 0x01, 0x04, 0x04, 0x01, 0x70, 0x00, 0x01, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00,
            0x01, 0x09, 0x07, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00, 0x0a, 0x23, 0x01, 0x21, 0x01, 0x01, 0x7f,
            0x02, 0x40, 0x03, 0x40, 0x20, 0x01, 0x20, 0x00, 0x4f, 0x0d, 0x01, 0x41, 0x00, 0x11, 0x00, 0x00, 0x20,
            0x01, 0x41, 0x01, 0x6a, 0x21, 0x01, 0x0c, 0x00, 0x0b, 0x0b, 0x20, 0x01, 0x0b };
}

/// Without a timeout, the export runs its count of host calls and returns it.
template <typename Impl>
void check_runs_to_count(const std::vector<uint8_t>& code) {
   register_timeout_host();
   std::atomic<bool>            expired{ false };
   timeout_host                 host{ .expired = &expired };
   std::vector<uint8_t>         module_code = code;
   backend<timeout_rhf_t, Impl> bkend(module_code, &wa);

   auto result = bkend.call_with_return(host, "env", "run", short_count);
   REQUIRE(result);
   CHECK(result->to_ui32() == short_count);
   CHECK(host.calls == short_count);
}

/// A timeout that fires during the host calls stops the run when the call in progress returns.
template <typename Impl>
void check_timeout_stops_host_calls(const std::vector<uint8_t>& code) {
   register_timeout_host();
   std::atomic<bool>            expired{ false };
   timeout_host                 host{ .expired = &expired };
   std::vector<uint8_t>         module_code = code;
   backend<timeout_rhf_t, Impl> bkend(module_code, &wa);

   CHECK_THROWS_AS(
         bkend.timed_run(marking_watchdog{ expired }, [&]() { bkend.call(host, "env", "run", host_call_limit); }),
         timeout_exception);
   REQUIRE(expired.load(std::memory_order_acquire));
   // Only the call in progress when the expiry was marked can see it.
   CHECK(host.calls_after_expiry <= 1);
   CHECK(host.calls < host_call_limit);
}

} // namespace

BACKEND_TEST_CASE("Direct host-call loop runs to its count without a timeout", "[watchdog_interrupt][host_functions]") {
   check_runs_to_count<TestType>(make_direct_host_call_loop_wasm_module());
}

BACKEND_TEST_CASE("A timeout during direct host calls stops the run before another host call",
                  "[watchdog_interrupt][host_functions]") {
   check_timeout_stops_host_calls<TestType>(make_direct_host_call_loop_wasm_module());
}

BACKEND_TEST_CASE("Indirect host-call loop runs to its count without a timeout",
                  "[watchdog_interrupt][host_functions]") {
   check_runs_to_count<TestType>(make_indirect_host_call_loop_wasm_module());
}

BACKEND_TEST_CASE("A timeout during indirect host calls stops the run before another host call",
                  "[watchdog_interrupt][host_functions]") {
   check_timeout_stops_host_calls<TestType>(make_indirect_host_call_loop_wasm_module());
}
