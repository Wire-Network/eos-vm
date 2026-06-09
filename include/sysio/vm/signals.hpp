#pragma once

#include <sysio/vm/allocator.hpp>
#include <sysio/vm/config.hpp>
#include <sysio/vm/exceptions.hpp>
#include <sysio/vm/span.hpp>
#include <sysio/vm/utils.hpp>

#include <atomic>
#include <cstdlib>
#include <exception>
#include <utility>
#include <signal.h>
#include <setjmp.h>

namespace sysio { namespace vm {

   // Fixes a duplicate symbol build issue when building with `-fvisibility=hidden`
   __attribute__((visibility("default")))
   inline thread_local std::atomic<sigjmp_buf*> signal_dest{nullptr};

   __attribute__((visibility("default")))
   inline thread_local std::span<std::byte> code_memory_range;

   __attribute__((visibility("default")))
   inline thread_local std::span<std::byte> memory_range;

   __attribute__((visibility("default")))
   inline std::atomic<bool> timed_run_has_timed_out{false};

   __attribute__((visibility("default")))
   /// Points to the timeout state for the timed_run currently executing on this thread.
   inline thread_local std::atomic<bool>* active_timed_run_has_timed_out{nullptr};

   /// Describes where the AArch64 timeout path is relative to VM execution.
   enum class aarch64_timeout_phase {
      none,
      executing_jit,
      returned_from_jit,
   };

   __attribute__((visibility("default")))
   /// Tracks whether a directed AArch64 timeout signal must escape VM execution or may be ignored as late.
   inline thread_local aarch64_timeout_phase active_aarch64_timeout_phase = aarch64_timeout_phase::none;

   // Fixes a duplicate symbol build issue when building with `-fvisibility=hidden`
   __attribute__((visibility("default")))
   inline thread_local std::exception_ptr saved_exception{nullptr};

   template<int Sig>
   inline struct sigaction prev_signal_handler;

   inline void signal_handler(int sig, siginfo_t* info, void* uap) {
      ignore_unused_variable_warning(uap);
      sigjmp_buf* dest = std::atomic_load(&signal_dest);

      if (dest) {
         const void* addr = info->si_addr;

         //neither range set means legacy catch-all behavior; useful for some of the old tests
         if (code_memory_range.empty() && memory_range.empty())
            siglongjmp(*dest, sig);

         //a failure in the memory range is always jumped out of
         if (addr >= memory_range.data() && addr < memory_range.data() + memory_range.size())
            siglongjmp(*dest, sig);

         // x86_64 uses shared code-page revocation for timeout interruption. The timeout state must be global
         // because any executing thread can fault after another watchdog disables executable pages.
         //
         // The AArch64 JIT uses a directed signal instead. Its timeout state stays thread-local so one interrupted
         // execution does not cause unrelated faults in another execution thread to be classified as timeouts.
         const bool execution_timed_out = [] {
            if constexpr (sys_vm_has_aarch64_jit_backend) {
               return active_timed_run_has_timed_out &&
                      active_timed_run_has_timed_out->load(std::memory_order_acquire);
            } else {
               return timed_run_has_timed_out.load(std::memory_order_acquire);
            }
         }();
         // SEGV/BUS/ILL can come from a revoked code page or from the directed AArch64 interruption signal.
         // On linux no SIGBUS handler is registered (see setup_signal_handler_impl()) so it will never occur here
         if ((sig == SIGSEGV || sig == SIGBUS || sig == SIGILL) && !execution_timed_out) {
            goto chain_previous_handler;
         }
         if constexpr (sys_vm_has_aarch64_jit_backend) {
            if (sig == SIGSEGV && execution_timed_out) {
               active_aarch64_timeout_phase = aarch64_timeout_phase::returned_from_jit;
            }
         }
         //otherwise, jump out
         siglongjmp(*dest, sig);

         //if dest not set, fall through and let chained handler an opportunity to handle
      }

      if constexpr (sys_vm_has_aarch64_jit_backend) {
         // The AArch64 timeout path interrupts the execution thread with a directed SIGSEGV instead of revoking
         // code pages. Only suppress it after VM execution has returned; while VM execution is still active, the
         // signal must remain an interrupt so tight loops can escape through siglongjmp.
         const bool directed_signal = !info || info->si_code <= 0;
         if (sig == SIGSEGV && directed_signal && active_timed_run_has_timed_out &&
             active_timed_run_has_timed_out->load(std::memory_order_acquire) &&
             active_aarch64_timeout_phase == aarch64_timeout_phase::returned_from_jit) {
            return;
         }
      }

chain_previous_handler:
      struct sigaction* prev_action;
      switch(sig) {
         case SIGSEGV: prev_action = &prev_signal_handler<SIGSEGV>; break;
         case SIGBUS: prev_action = &prev_signal_handler<SIGBUS>; break;
         case SIGFPE: prev_action = &prev_signal_handler<SIGFPE>; break;
         case SIGILL: prev_action = &prev_signal_handler<SIGILL>; break;
         default: std::abort();
      }
      if (!prev_action) std::abort();
      if ((prev_action->sa_flags & SA_SIGINFO) && prev_action->sa_sigaction == &signal_handler) {
         signal(sig, SIG_DFL);
         raise(sig);
         std::abort();
      }
      if (prev_action->sa_flags & SA_SIGINFO) {
         // FIXME: We need to be at least as strict as the original
         // flags and relax the mask as needed.
         prev_action->sa_sigaction(sig, info, uap);
      } else {
         if(prev_action->sa_handler == SIG_DFL) {
            // The default for all three signals is to terminate the process.
            sigaction(sig, prev_action, nullptr);
            raise(sig);
         } else if(prev_action->sa_handler == SIG_IGN) {
            // Do nothing
         } else {
            prev_action->sa_handler(sig);
         }
      }
   }

   template<int Sig>
   inline void install_signal_handler_once(const struct sigaction& sa) {
      struct sigaction current;
      sigaction(Sig, nullptr, &current);
      if ((current.sa_flags & SA_SIGINFO) && current.sa_sigaction == &signal_handler)
         return;
      sigaction(Sig, &sa, &prev_signal_handler<Sig>);
   }

   // only valid inside invoke_with_signal_handler.
   // This is a workaround for the fact that it
   // is currently unsafe to throw an exception through
   // a jit frame.
   template<typename F>
   inline void longjmp_on_exception(F&& f) {
      static_assert(std::is_trivially_destructible_v<std::decay_t<F>>, "longjmp has undefined behavior when it bypasses destructors.");
      bool caught_exception = false;
      try {
         f();
      } catch(...) {
         saved_exception = std::current_exception();
         // Cannot safely longjmp from inside the catch,
         // as that will leak the exception.
         caught_exception = true;
      }
      if (caught_exception) {
         sigset_t block_mask;
         sigemptyset(&block_mask);
         sigaddset(&block_mask, SIGPROF);
         pthread_sigmask(SIG_BLOCK, &block_mask, nullptr);
         sigjmp_buf* dest = std::atomic_load(&signal_dest);
         siglongjmp(*dest, -1);
      }
   }

   template<typename E>
   [[noreturn]] inline void throw_(const char* msg) {
      saved_exception = std::make_exception_ptr(E{msg});
      sigset_t block_mask;
      sigemptyset(&block_mask);
      sigaddset(&block_mask, SIGPROF);
      pthread_sigmask(SIG_BLOCK, &block_mask, nullptr);
      sigjmp_buf* dest = std::atomic_load(&signal_dest);
      siglongjmp(*dest, -1);
   }

   inline void setup_signal_handler_impl() {
      struct sigaction sa;
      sa.sa_sigaction = &signal_handler;
      sigemptyset(&sa.sa_mask);
      sigaddset(&sa.sa_mask, SIGPROF);
      sa.sa_flags = SA_NODEFER | SA_SIGINFO;
      install_signal_handler_once<SIGSEGV>(sa);
#ifndef __linux__
      install_signal_handler_once<SIGBUS>(sa);
#endif
      install_signal_handler_once<SIGFPE>(sa);
      // Apple Silicon may deliver disabled-JIT-page instruction fetches as SIGILL.
      if constexpr (sys_vm_has_aarch64_jit_backend) {
         install_signal_handler_once<SIGILL>(sa);
      }
   }

   inline void setup_signal_handler() {
      static int init_helper = (setup_signal_handler_impl(), 0);
      ignore_unused_variable_warning(init_helper);
      static_assert(std::atomic<sigjmp_buf*>::is_always_lock_free, "Atomic pointers must be lock-free to be async signal safe.");
   }

   /// Call a function with a signal handler installed.  If this thread is
   /// signalled during the execution of f, the function e will be called with
   /// the signal number as an argument.  If f creates any automatic variables
   /// with non-trivial destructors, then it must mask the relevant signals
   /// during the lifetime of these objects or the behavior is undefined.
   ///
   /// signals handled: SIGSEGV, SIGBUS (except on Linux), SIGFPE, and SIGILL on AArch64
   ///
   // Make this noinline to prevent possible corruption of the caller's local variables.
   // It's unlikely, but I'm not sure that it can definitely be ruled out if both
   // this and f are inlined and f modifies locals from the caller.
   template<typename F, typename E>
   [[gnu::noinline]] auto invoke_with_signal_handler(F&& f, E&& e, growable_allocator& code_allocator, wasm_allocator* mem_allocator) {
      setup_signal_handler();
      sigjmp_buf dest;
      sigjmp_buf* volatile old_signal_handler = nullptr;
      code_memory_range = code_allocator.get_code_span();
      memory_range = mem_allocator ? mem_allocator->get_span() : std::span<std::byte>{};
      int sig;
      if((sig = sigsetjmp(dest, 1)) == 0) {
         // Note: Cannot use RAII, as non-trivial destructors w/ longjmp
         // have undefined behavior. [csetjmp.syn]
         //
         // Warning: The order of operations is critical here.
         // We also have to register signal_dest before unblocking
         // signals to make sure that only our signal handler is executed
         // if the caller has previously blocked signals.
         old_signal_handler = std::atomic_exchange(&signal_dest, &dest);
         sigset_t unblock_mask, old_sigmask; // Might not be preserved across longjmp
         sigemptyset(&unblock_mask);
         sigaddset(&unblock_mask, SIGSEGV);
         sigaddset(&unblock_mask, SIGBUS);
         sigaddset(&unblock_mask, SIGFPE);
         if constexpr (sys_vm_has_aarch64_jit_backend) {
            sigaddset(&unblock_mask, SIGILL);
         }
         sigaddset(&unblock_mask, SIGPROF);
         pthread_sigmask(SIG_UNBLOCK, &unblock_mask, &old_sigmask);
         try {
            f();
            pthread_sigmask(SIG_SETMASK, &old_sigmask, nullptr);
            std::atomic_store(&signal_dest, old_signal_handler);
         } catch(...) {
            pthread_sigmask(SIG_SETMASK, &old_sigmask, nullptr);
            std::atomic_store(&signal_dest, old_signal_handler);
            throw;
         }
      } else {
         std::atomic_store(&signal_dest, old_signal_handler);
         if (sig == -1) {
            std::exception_ptr exception = std::move(saved_exception);
            saved_exception = nullptr;
            std::rethrow_exception(exception);
         } else {
            e(sig);
         }
      }
   }

}} // namespace sysio::vm
