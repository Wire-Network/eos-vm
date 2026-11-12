#include <sysio/vm/signals.hpp>

#include <catch2/catch.hpp>

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <limits>
#include <type_traits>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int chained_signal = SIGILL;
constexpr int child_setup_failure = 81;
constexpr int child_unexpected_vm_catch = 82;
constexpr int child_unexpected_siginfo = 83;
constexpr unsigned child_timeout_seconds = 5;

/// Capture the original ABI's arguments without modifying process-wide signal actions.
struct handler_observation {
   int sig;
   siginfo_t* info;
   void* context;
};
handler_observation observed{};

/// Record a legacy predecessor invocation.
void record_legacy_handler(int sig) {
   observed = {sig, nullptr, nullptr};
}

/// Record every argument forwarded to an SA_SIGINFO predecessor.
void record_siginfo_handler(int sig, siginfo_t* info, void* context) {
   observed = {sig, info, context};
}

/// Fail a child when a deferred timeout incorrectly reaches its predecessor.
void reject_deferred_predecessor(int) {
   _exit(child_unexpected_vm_catch);
}

#if defined(__x86_64__)
/// Verify real IDIV/#GP metadata and the original signal context in a forwarded SA_SIGINFO predecessor.
void verify_host_fault_info(int sig, siginfo_t* info, void* context) {
   bool expected = info && sig == SIGFPE && info->si_code == FPE_INTDIV;
#ifdef __linux__
   expected = expected || (info && sig == SIGSEGV && info->si_code == SI_KERNEL && info->si_addr == nullptr);
#endif
   _exit(expected && context ? EXIT_SUCCESS : child_unexpected_siginfo);
}
#endif

/// Isolate fatal signal tests, bound a broken handler's spin, and leave the parent signal state intact.
template <typename F>
int run_in_subprocess(F&& test) {
   const auto child = fork();
   REQUIRE(child >= 0);
   if (child == 0) {
      const struct rlimit no_core{0, 0};
      if (setrlimit(RLIMIT_CORE, &no_core) != 0)
         _exit(child_setup_failure);
#ifdef __linux__
      // Linux ignores RLIMIT_CORE for piped crash reporters; keep fatal test children non-dumpable as well.
      if (prctl(PR_SET_DUMPABLE, 0L) != 0)
         _exit(child_setup_failure);
#endif
      alarm(child_timeout_seconds);
      sigset_t unblocked;
      sigemptyset(&unblocked);
      for (int sig : std::array{SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGALRM}) {
         std::signal(sig, SIG_DFL);
         sigaddset(&unblocked, sig);
      }
      if (sigprocmask(SIG_UNBLOCK, &unblocked, nullptr) != 0)
         _exit(child_setup_failure);
      test();
      _exit(EXIT_SUCCESS);
   }
   int status = 0;
   pid_t waited;
   do {
      waited = waitpid(child, &status, 0);
   } while (waited == -1 && errno == EINTR);
   REQUIRE(waited == child);
   return status;
}

/// Round-trip a disposition through the kernel before exercising the direct chaining entry point.
int chain_in_subprocess(struct sigaction action) {
   return run_in_subprocess([action]() mutable {
      if (sigemptyset(&action.sa_mask) != 0 || sigaction(chained_signal, &action, nullptr) != 0 ||
          sigaction(chained_signal, nullptr, &sysio::vm::prev_signal_handler<chained_signal>) != 0)
         _exit(child_setup_failure);
      std::signal(chained_signal, SIG_DFL);
      sysio::vm::chain_previous_signal_handler(chained_signal, nullptr, nullptr);
   });
}

/// Select a kernel-backed fault without invoking undefined C++ arithmetic or pointer operations.
enum class host_signal_kind {
   protected_page,
   raised_signal,
#if defined(__linux__) && defined(__x86_64__)
   noncanonical_store,
   misaligned_simd_load,
#endif
#if defined(__x86_64__)
   divide_by_zero,
   divide_overflow,
#endif
};

/// Model the legacy process-wide timeout or the per-run state used by the AArch64 JIT.
enum class timeout_state { none, process_wide, per_run };

/// Deliver the selected fault after installing the VM over the chosen predecessor.
int deliver_host_signal(struct sigaction action, host_signal_kind kind, bool active_vm_frame, timeout_state timeout) {
   return run_in_subprocess([=]() mutable {
      if (sigemptyset(&action.sa_mask) != 0 || sigaction(SIGSEGV, &action, nullptr) != 0 ||
          sigaction(SIGBUS, &action, nullptr) != 0 || sigaction(SIGFPE, &action, nullptr) != 0)
         _exit(child_setup_failure);
      sysio::vm::setup_signal_handler_impl();
      sysio::vm::signal_dest.store(nullptr);
      sysio::vm::code_memory_range = {};
      sysio::vm::memory_range = {};
      // Set both indicators for per-run timeouts so this branch is exercised on every host architecture.
      sysio::vm::timed_run_has_timed_out.store(timeout != timeout_state::none);
      std::atomic<bool> thread_timed_out{timeout == timeout_state::per_run};
      sysio::vm::active_timed_run_has_timed_out = timeout == timeout_state::per_run ? &thread_timed_out : nullptr;

      // A nonempty, unrelated code range makes an active VM frame decline this host fault.
      std::byte unrelated_code{};
      sigjmp_buf destination;
      if (sigsetjmp(destination, 1) != 0)
         _exit(child_unexpected_vm_catch);
      if (active_vm_frame) {
         sysio::vm::code_memory_range = {&unrelated_code, 1};
         sysio::vm::signal_dest.store(&destination);
      }
      switch (kind) {
         case host_signal_kind::protected_page: {
            const auto page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0)
               _exit(child_setup_failure);
            auto* page = mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (page == MAP_FAILED)
               _exit(child_setup_failure);
            *static_cast<volatile unsigned char*>(page) = 1;
            break;
         }
         case host_signal_kind::raised_signal:
            if (raise(SIGSEGV) != 0)
               _exit(child_setup_failure);
            break;
#if defined(__linux__) && defined(__x86_64__)
         case host_signal_kind::noncanonical_store: {
            constexpr std::uintptr_t noncanonical_address = 0x8000000000000000ULL;
            asm volatile("movb $0, (%0)" : : "r"(noncanonical_address) : "memory");
            break;
         }
         case host_signal_kind::misaligned_simd_load: {
            alignas(16) std::array<std::byte, 32> storage{};
            asm volatile("movaps (%0), %%xmm0" : : "r"(storage.data() + 1) : "xmm0", "memory");
            break;
         }
#endif
#if defined(__x86_64__)
         case host_signal_kind::divide_by_zero:
         case host_signal_kind::divide_overflow: {
            int dividend = kind == host_signal_kind::divide_overflow ? std::numeric_limits<int>::min() : 1;
            const int divisor = kind == host_signal_kind::divide_overflow ? -1 : 0;
            asm volatile("cltd\n\tidivl %1" : "+&a"(dividend) : "r"(divisor) : "edx", "cc");
            break;
         }
#endif
      }
   });
}

} // namespace

/// SIG_IGN remains non-callable even when the saved action carries SA_SIGINFO.
TEST_CASE("Ignored predecessors never become callback addresses", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = SIG_IGN;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const auto status = chain_in_subprocess(action);
   REQUIRE(WIFEXITED(status));
   CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
}

/// Default actions re-raise the original signal instead of jumping through the null sentinel.
TEST_CASE("Default predecessors retain the original terminating signal", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = SIG_DFL;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const auto status = chain_in_subprocess(action);
   REQUIRE(WIFSIGNALED(status));
   CHECK(WTERMSIG(status) == chained_signal);
}

/// Real handlers retain their original calling convention and argument addresses.
TEST_CASE("Callable predecessors retain their original arguments", "[signal_handler_chain]") {
   const bool with_siginfo = GENERATE(false, true);
   const auto previous = sysio::vm::prev_signal_handler<chained_signal>;
   auto restore = sysio::vm::scope_guard{[previous] {
      sysio::vm::prev_signal_handler<chained_signal> = previous;
   }};
   struct sigaction action{};
   if (with_siginfo) {
      action.sa_flags = SA_SIGINFO;
      action.sa_sigaction = record_siginfo_handler;
   } else {
      action.sa_handler = record_legacy_handler;
   }
   sysio::vm::prev_signal_handler<chained_signal> = action;
   siginfo_t info{};
   int context = 0;
   observed = {};
   sysio::vm::chain_previous_signal_handler(chained_signal, &info, &context);
   CHECK(observed.sig == chained_signal);
   CHECK(observed.info == (with_siginfo ? &info : nullptr));
   CHECK(observed.context == (with_siginfo ? &context : nullptr));
}

/// A saved VM handler still falls back to default termination rather than recursing.
TEST_CASE("A predecessor that is the VM handler does not recurse", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_flags = SA_SIGINFO;
   action.sa_sigaction = sysio::vm::signal_handler;
   const auto status = chain_in_subprocess(action);
   REQUIRE(WIFSIGNALED(status));
   CHECK(WTERMSIG(status) == chained_signal);
}

/// Ignoring a predecessor cannot suppress a kernel fault outside VM code, with or without an active frame.
TEST_CASE("Host memory faults terminate instead of repeating an ignored instruction", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = GENERATE(false, true) ? SIG_IGN : SIG_DFL;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const bool active_vm_frame = GENERATE(false, true);
   const auto status =
      deliver_host_signal(action, host_signal_kind::protected_page, active_vm_frame, timeout_state::none);
   CAPTURE(action.sa_handler == SIG_IGN, action.sa_flags, active_vm_frame, status);
   REQUIRE(WIFSIGNALED(status));
   // A protected-page access is SIGSEGV on Linux; Unix platforms may report it as SIGBUS.
   CHECK((WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS));
}

/// raise() is not a synchronous fault and must still honor an ignored predecessor, including SA_SIGINFO.
TEST_CASE("Asynchronously raised host signals retain ignored dispositions", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = SIG_IGN;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const bool active_vm_frame = GENERATE(false, true);
   const auto status =
      deliver_host_signal(action, host_signal_kind::raised_signal, active_vm_frame, timeout_state::none);
   CAPTURE(action.sa_handler == SIG_IGN, action.sa_flags, active_vm_frame, status);
   REQUIRE(WIFEXITED(status));
   CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
}

/// Only signal-specific kernel fault codes may override an ignored disposition.
TEST_CASE("Synchronous fault classification includes arithmetic faults and excludes sent signals",
          "[signal_handler_chain]") {
   using signal_fault = std::pair<int, int>;
   const auto fault = GENERATE(signal_fault{SIGSEGV, SEGV_MAPERR}, signal_fault{SIGSEGV, SEGV_ACCERR},
                              signal_fault{SIGBUS, BUS_ADRALN}, signal_fault{SIGBUS, BUS_ADRERR},
                              signal_fault{SIGBUS, BUS_OBJERR}, signal_fault{SIGILL, ILL_ILLOPC},
                              signal_fault{SIGFPE, FPE_INTDIV}, signal_fault{SIGFPE, FPE_INTOVF},
                              signal_fault{SIGFPE, FPE_FLTDIV}, signal_fault{SIGFPE, FPE_FLTOVF},
                              signal_fault{SIGFPE, FPE_FLTUND}, signal_fault{SIGFPE, FPE_FLTRES},
                              signal_fault{SIGFPE, FPE_FLTINV}, signal_fault{SIGFPE, FPE_FLTSUB});
   siginfo_t info{};
   info.si_code = fault.second;
   CHECK(sysio::vm::fault_is_synchronous(fault.first, &info, nullptr));
   for (int code : {SI_USER, SI_QUEUE}) {
      info.si_code = code;
      CHECK_FALSE(sysio::vm::fault_is_synchronous(fault.first, &info, nullptr));
   }
}

#if defined(__APPLE__) && defined(__aarch64__)
/// Darwin's fault-like siginfo cannot turn a saved SVC64 context into a hardware memory fault.
TEST_CASE("Darwin syscall contexts exclude synchronous memory faults", "[signal_handler_chain]") {
   siginfo_t info{};
   info.si_code = GENERATE(SEGV_MAPERR, SEGV_ACCERR);
   std::remove_pointer_t<mcontext_t> machine_context{};
   ucontext_t context{};
   context.uc_mcontext = &machine_context;
   constexpr unsigned exception_class_shift = 26;
   constexpr unsigned supervisor_call_64 = 0x15;
   constexpr unsigned data_abort_from_user = 0x24;
   constexpr unsigned instruction_abort_from_user = 0x20;

   machine_context.__es.__esr = supervisor_call_64 << exception_class_shift;
   CHECK_FALSE(sysio::vm::fault_is_synchronous(SIGSEGV, &info, &context));
   machine_context.__es.__esr = data_abort_from_user << exception_class_shift;
   CHECK(sysio::vm::fault_is_synchronous(SIGSEGV, &info, &context));
   machine_context.__es.__esr = instruction_abort_from_user << exception_class_shift;
   CHECK(sysio::vm::fault_is_synchronous(SIGSEGV, &info, &context));
}
#endif

#if defined(__linux__) && defined(__x86_64__)
/// Linux reports both non-canonical stores and misaligned MOVAPS as SI_KERNEL SIGSEGV, with no fault address.
TEST_CASE("General-protection host faults terminate with ignored predecessors", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = GENERATE(false, true) ? SIG_IGN : SIG_DFL;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const bool active_vm_frame = GENERATE(false, true);
   const auto kind = GENERATE(host_signal_kind::noncanonical_store, host_signal_kind::misaligned_simd_load);
   const auto timeout = GENERATE(timeout_state::none, timeout_state::per_run);
   const auto status = deliver_host_signal(action, kind, active_vm_frame, timeout);
   CAPTURE(action.sa_handler == SIG_IGN, action.sa_flags, active_vm_frame, kind, timeout, status);
   REQUIRE(WIFSIGNALED(status));
   CHECK(WTERMSIG(status) == SIGSEGV);
}

/// Widening fatal chaining must not make a previously unrecognized #GP eligible for a VM unwind.
TEST_CASE("General-protection codes do not widen VM ownership", "[signal_handler_chain]") {
   const auto status = run_in_subprocess([] {
      std::byte code{};
      sysio::vm::code_memory_range = {&code, 1};
      sysio::vm::memory_range = {};
      sysio::vm::timed_run_has_timed_out.store(false);
      sysio::vm::active_timed_run_has_timed_out = nullptr;
      siginfo_t info{};
      info.si_code = SI_KERNEL;
      ucontext_t context{};
      context.uc_mcontext.gregs[REG_RIP] = reinterpret_cast<greg_t>(&code);
      sigjmp_buf destination;
      if (sigsetjmp(destination, 1) != 0)
         _exit(child_unexpected_vm_catch);
      if (sysio::vm::handle_active_vm_signal(SIGSEGV, &info, &context, &destination))
         _exit(child_unexpected_vm_catch);
   });
   REQUIRE(WIFEXITED(status));
   CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
}
#endif

#if defined(__x86_64__)
/// IDIV generates a real arithmetic fault for both zero divisors and signed quotient overflow.
TEST_CASE("Host integer division faults retain default termination", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = GENERATE(false, true) ? SIG_IGN : SIG_DFL;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const bool active_vm_frame = GENERATE(false, true);
   const auto kind = GENERATE(host_signal_kind::divide_by_zero, host_signal_kind::divide_overflow);
   const auto status = deliver_host_signal(action, kind, active_vm_frame, timeout_state::none);
   CAPTURE(action.sa_handler == SIG_IGN, action.sa_flags, active_vm_frame, kind, status);
   REQUIRE(WIFSIGNALED(status));
   CHECK(WTERMSIG(status) == SIGFPE);
}
#endif

/// An expired run cannot swallow or longjmp across a genuine host memory fault.
TEST_CASE("Timed-out host memory faults must reach the predecessor", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = GENERATE(false, true) ? SIG_IGN : SIG_DFL;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const auto timeout = GENERATE(timeout_state::process_wide, timeout_state::per_run);
   const auto status = deliver_host_signal(action, host_signal_kind::protected_page, true, timeout);
   CAPTURE(action.sa_handler == SIG_IGN, action.sa_flags, timeout, status);
   REQUIRE(WIFSIGNALED(status));
   CHECK((WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS));
}

/// A directed timeout interrupt in a host frame must still wait for a safe VM polling point.
TEST_CASE("Timed-out host interruptions remain deferred", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_handler = reject_deferred_predecessor;
   action.sa_flags = GENERATE(0, SA_SIGINFO);
   const auto status = deliver_host_signal(action, host_signal_kind::raised_signal, true, timeout_state::per_run);
   CAPTURE(action.sa_flags, status);
   REQUIRE(WIFEXITED(status));
   CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
}

#if defined(__x86_64__)
/// Hardware IDIV and Linux #GP must retain their kernel-supplied code and context when forwarded.
TEST_CASE("Host hardware faults forward the original siginfo", "[signal_handler_chain]") {
   struct sigaction action{};
   action.sa_sigaction = verify_host_fault_info;
   action.sa_flags = SA_SIGINFO;
   const bool active_vm_frame = GENERATE(false, true);
   const auto timeout = GENERATE(timeout_state::none, timeout_state::per_run);
#ifdef __linux__
   const auto kind = GENERATE(host_signal_kind::noncanonical_store, host_signal_kind::misaligned_simd_load,
                              host_signal_kind::divide_by_zero, host_signal_kind::divide_overflow);
#else
   const auto kind = GENERATE(host_signal_kind::divide_by_zero, host_signal_kind::divide_overflow);
#endif
   const auto status = deliver_host_signal(action, kind, active_vm_frame, timeout);
   CAPTURE(active_vm_frame, kind, timeout, status);
   REQUIRE(WIFEXITED(status));
   CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
}
#endif
