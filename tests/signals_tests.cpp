#include <sysio/vm/signals.hpp>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <span>
#include <thread>
#include <iostream>

#include <catch2/catch.hpp>

struct test_exception {};

TEST_CASE("Testing signals", "[invoke_with_signal_handler]") {
   bool okay = false;
   // These tests call the signal handler directly, outside a real VM execution.
   // A null memory allocator intentionally leaves memory_range empty so the
   // handler exercises its legacy catch-all path.
   sysio::vm::growable_allocator code_allocator;
   try {
      sysio::vm::invoke_with_signal_handler([]() {
         std::raise(SIGSEGV);
      }, [](int sig) {
         throw test_exception{};
      }, code_allocator, nullptr);
   } catch(test_exception&) {
      okay = true;
   }
   CHECK(okay);
}

TEST_CASE("Testing throw", "[signal_handler_throw]") {
   // See the signal test above: no wasm memory is involved in this direct path.
   sysio::vm::growable_allocator code_allocator;
   CHECK_THROWS_AS(sysio::vm::invoke_with_signal_handler([](){
      sysio::vm::throw_<sysio::vm::wasm_exit_exception>( "Exiting" );
   }, [](int){}, code_allocator, nullptr), sysio::vm::wasm_exit_exception);
}

static volatile sig_atomic_t sig_handled;
static volatile sig_atomic_t sig_handled_count;

static void handle_signal(int sig) {
   sig_handled_count = sig_handled_count + 1;
   sig_handled = 42 + sig;
}

static void handle_signal_sigaction(int sig, siginfo_t* info, void* uap) {
   sig_handled_count = sig_handled_count + 1;
   sig_handled = 142 + sig;
}

TEST_CASE("Test signal handler forwarding", "[signal_handler_forward]") {
   // reset backup signal handlers
   auto guard = sysio::vm::scope_guard{[]{
      std::signal(SIGSEGV, SIG_DFL);
      std::signal(SIGBUS, SIG_DFL);
      std::signal(SIGFPE, SIG_DFL);
      sysio::vm::setup_signal_handler_impl(); // This is normally only called once
   }};
   {
      std::signal(SIGSEGV, &handle_signal);
      std::signal(SIGBUS, &handle_signal);
      std::signal(SIGFPE, &handle_signal);
      sysio::vm::setup_signal_handler_impl();
      sig_handled = 0;
      sig_handled_count = 0;
      std::raise(SIGSEGV);
      CHECK(sig_handled == 42 + SIGSEGV);
      CHECK(sig_handled_count == 1);
#ifndef __linux__
      sig_handled = 0;
      sig_handled_count = 0;
      std::raise(SIGBUS);
      CHECK(sig_handled == 42 + SIGBUS);
      CHECK(sig_handled_count == 1);
#endif
      sig_handled = 0;
      sig_handled_count = 0;
      std::raise(SIGFPE);
      CHECK(sig_handled == 42 + SIGFPE);
      CHECK(sig_handled_count == 1);
   }
   {
      struct sigaction sa;
      sa.sa_sigaction = &handle_signal_sigaction;
      sigemptyset(&sa.sa_mask);
      sa.sa_flags = SA_NODEFER | SA_SIGINFO;
      sigaction(SIGSEGV, &sa, nullptr);
      sigaction(SIGBUS, &sa, nullptr);
      sigaction(SIGFPE, &sa, nullptr);
      sysio::vm::setup_signal_handler_impl();
      sig_handled = 0;
      sig_handled_count = 0;
      std::raise(SIGSEGV);
      CHECK(sig_handled == 142 + SIGSEGV);
      CHECK(sig_handled_count == 1);
#ifndef __linux__
      sig_handled = 0;
      sig_handled_count = 0;
      std::raise(SIGBUS);
      CHECK(sig_handled == 142 + SIGBUS);
      CHECK(sig_handled_count == 1);
#endif
      sig_handled = 0;
      sig_handled_count = 0;
      std::raise(SIGFPE);
      CHECK(sig_handled == 142 + SIGFPE);
      CHECK(sig_handled_count == 1);
   }
}

// Regression: read-only-trx shared-code SIGSEGV (wire-sysio CI run 39807473127).
//
// Read-only threads share one compiled module: one code arena, one process-wide
// `timed_run_has_timed_out`. The x86_64 timeout mechanism revokes the shared code pages
// (mprotect PROT_NONE); a thread executing them then faults on an instruction fetch into
// its own `code_memory_range`. A PEER thread's timed_run exit guard can already have
// cleared `timed_run_has_timed_out` to false before it re-enables those pages, so a
// handler that classified the fault by that flag read "not timed out", returned false,
// and chained to the previous (sys-vm-oc / default) handler — aborting the process.
//
// `handle_active_vm_signal` must instead classify by the faulting LOCATION: a fault whose
// address or interrupted PC lies in this thread's code arena is always a VM fault and must
// be jumped out of, regardless of the flag. These cases call the handler directly with a
// synthetic siginfo/ucontext (no real VM execution), setting the thread-local ranges that
// `invoke_with_signal_handler` would otherwise populate.
namespace {
   // A real, mapped buffer so the code-arena span points at valid memory. Page-aligned and
   // oversized to resemble a JIT code segment; its bytes are never executed by the test.
   alignas(4096) std::byte regression_code_arena[8192];

   struct vm_signal_state_guard {
      vm_signal_state_guard() = default;
      ~vm_signal_state_guard() {
         // Restore the empty-range / cleared state the other cases in this file rely on
         // (the thread-local ranges persist across Catch cases on the same thread).
         sysio::vm::code_memory_range = std::span<std::byte>{};
         sysio::vm::memory_range      = std::span<std::byte>{};
         sysio::vm::timed_run_has_timed_out.store(false, std::memory_order_release);
         sysio::vm::active_timed_run_has_timed_out = nullptr;
      }
   };

   // Build a SIGSEGV siginfo/ucontext whose fault address (and, on x86_64 Linux, the
   // interrupted instruction pointer) is `addr` — the instruction-fetch signature of a
   // revoked-code-page fault, where the kernel reports si_addr == RIP.
   void make_instruction_fetch_fault(siginfo_t& info, ucontext_t& uctx, void* addr) {
      info = siginfo_t{};
      info.si_signo = SIGSEGV;
      info.si_code  = SEGV_ACCERR;
      info.si_addr  = addr;
      uctx = ucontext_t{};
#if defined(__linux__) && defined(__x86_64__)
      uctx.uc_mcontext.gregs[REG_RIP] = reinterpret_cast<greg_t>(addr);
#endif
   }
}

TEST_CASE("A fault in the shared code arena jumps out even when timed_run_has_timed_out is false",
          "[handle_active_vm_signal_code_range]") {
   vm_signal_state_guard guard;

   sysio::vm::code_memory_range =
      std::span<std::byte>{ regression_code_arena, sizeof(regression_code_arena) };
   sysio::vm::memory_range = std::span<std::byte>{};            // no linear memory in this unit path
   // THE regression condition: a peer thread already cleared the flag.
   sysio::vm::timed_run_has_timed_out.store(false, std::memory_order_release);
   sysio::vm::active_timed_run_has_timed_out = nullptr;

   siginfo_t info;
   ucontext_t uctx;
   make_instruction_fetch_fault(info, uctx, regression_code_arena + 0x40); // inside the arena

   bool jumped_out = false;
   sigjmp_buf dest;
   if (sigsetjmp(dest, 1) == 0) {
      sysio::vm::handle_active_vm_signal(SIGSEGV, &info, &uctx, &dest);
      FAIL("handle_active_vm_signal returned instead of jumping out of the revoked code arena");
   } else {
      jumped_out = true;
   }
   CHECK(jumped_out);
}

TEST_CASE("A fault outside every VM range is still declined (chained) when not timed out",
          "[handle_active_vm_signal_code_range]") {
   vm_signal_state_guard guard;

   sysio::vm::code_memory_range =
      std::span<std::byte>{ regression_code_arena, sizeof(regression_code_arena) };
   sysio::vm::memory_range = std::span<std::byte>{};
   sysio::vm::timed_run_has_timed_out.store(false, std::memory_order_release);
   sysio::vm::active_timed_run_has_timed_out = nullptr;

   // A fault well away from both the code arena and (empty) memory range.
   int unrelated_stack_object = 0;
   siginfo_t info;
   ucontext_t uctx;
   make_instruction_fetch_fault(info, uctx, &unrelated_stack_object);

   sigjmp_buf dest;
   // Not timed out and not in any VM range: the handler must return false so the signal
   // chains to the previous handler, rather than being swallowed.
   CHECK(sysio::vm::handle_active_vm_signal(SIGSEGV, &info, &uctx, &dest) == false);
}
