#include <sysio/vm/signals.hpp>

#include <catch2/catch.hpp>

#include <array>
#include <cerrno>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int chained_signal = SIGILL;
constexpr int child_setup_failure = 81;
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

/// Round-trip a disposition through the kernel and isolate default termination or a broken sentinel call.
int chain_in_subprocess(struct sigaction action) {
   const auto child = fork();
   REQUIRE(child >= 0);
   if (child == 0) {
      const struct rlimit no_core{0, 0};
      if (setrlimit(RLIMIT_CORE, &no_core) != 0)
         _exit(child_setup_failure);
      alarm(child_timeout_seconds);
      sigset_t unblocked;
      sigemptyset(&unblocked);
      for (int sig : std::array{SIGSEGV, SIGBUS, SIGFPE, SIGILL}) {
         std::signal(sig, SIG_DFL);
         sigaddset(&unblocked, sig);
      }
      if (sigprocmask(SIG_UNBLOCK, &unblocked, nullptr) != 0 || sigemptyset(&action.sa_mask) != 0 ||
          sigaction(chained_signal, &action, nullptr) != 0 ||
          sigaction(chained_signal, nullptr, &sysio::vm::prev_signal_handler<chained_signal>) != 0)
         _exit(child_setup_failure);
      std::signal(chained_signal, SIG_DFL);
      sysio::vm::chain_previous_signal_handler(chained_signal, nullptr, nullptr);
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
