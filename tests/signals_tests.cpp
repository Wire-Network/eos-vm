#include <sysio/vm/signals.hpp>
#include <chrono>
#include <csignal>
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

static void handle_signal(int sig) {
   sig_handled = 42 + sig;
}

static void handle_signal_sigaction(int sig, siginfo_t* info, void* uap) {
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
      std::raise(SIGSEGV);
      CHECK(sig_handled == 42 + SIGSEGV);
#ifndef __linux__
      sig_handled = 0;
      std::raise(SIGBUS);
      CHECK(sig_handled == 42 + SIGBUS);
#endif
      sig_handled = 0;
      std::raise(SIGFPE);
      CHECK(sig_handled == 42 + SIGFPE);
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
      std::raise(SIGSEGV);
      CHECK(sig_handled == 142 + SIGSEGV);
#ifndef __linux__
      sig_handled = 0;
      std::raise(SIGBUS);
      CHECK(sig_handled == 142 + SIGBUS);
#endif
      sig_handled = 0;
      std::raise(SIGFPE);
      CHECK(sig_handled == 142 + SIGFPE);
   }
}
