#pragma once

#include <sysio/vm/allocator.hpp>
#include <sysio/vm/bitcode_writer.hpp>
#include <sysio/vm/config.hpp>
#include <sysio/vm/debug_visitor.hpp>
#include <sysio/vm/execution_context.hpp>
#include <sysio/vm/interpret_visitor.hpp>
#include <sysio/vm/null_writer.hpp>
#include <sysio/vm/parser.hpp>
#include <sysio/vm/types.hpp>

#if SYS_VM_HAS_JIT_BACKEND && SYS_VM_TARGET_X86_64
#   include <sysio/vm/x86_64.hpp>
#endif
#if SYS_VM_HAS_AARCH64_JIT_BACKEND
#   include <sysio/vm/aarch64.hpp>
#endif

#include <atomic>
#include <exception>
#include <iostream>
#include <optional>
#include <pthread.h>
#include <signal.h>
#include <string_view>
#include <system_error>
#include <vector>

namespace sysio { namespace vm {

   __attribute__((visibility("default"))) inline std::atomic<uint32_t> total_timed_run_in_progress{ 0 };

#if SYS_VM_HAS_JIT_BACKEND
   /// Normal JIT backend capability advertised by the selected architecture.
   struct jit {
      template <typename Host>
      using context = jit_execution_context<Host>;
      template <typename Host, typename Options, typename DebugInfo>
      using parser = binary_parser<machine_code_writer<jit_execution_context<Host>>, Options, DebugInfo>;
      static constexpr bool is_jit = true;
   };
#endif

#if SYS_VM_HAS_JIT_PROFILE
   /// JIT backend variant with async profiling/backtrace support.
   struct jit_profile {
      template <typename Host>
      using context = jit_execution_context<Host, true>;
      template <typename Host, typename Options, typename DebugInfo>
      using parser                 = binary_parser<machine_code_writer<context<Host>>, Options, DebugInfo>;
      static constexpr bool is_jit = true;
   };
#endif

   struct interpreter {
      template <typename Host>
      using context = execution_context<Host>;
      template <typename Host, typename Options, typename DebugInfo>
      using parser                 = binary_parser<bitcode_writer, Options, DebugInfo>;
      static constexpr bool is_jit = false;
   };

   struct null_backend {
      template <typename Host>
      using context = null_execution_context<Host>;
      template <typename Host, typename Options, typename DebugInfo>
      using parser                 = binary_parser<null_writer, Options, DebugInfo>;
      static constexpr bool is_jit = false;
   };

   template <typename T>
   struct maybe_unique_ptr {
      maybe_unique_ptr(T* ptr = nullptr, bool owns = true) : ptr(ptr), owns(owns) {}
      maybe_unique_ptr(const maybe_unique_ptr&)            = delete;
      maybe_unique_ptr& operator=(const maybe_unique_ptr&) = delete;
      ~maybe_unique_ptr() {
         if (ptr && owns)
            delete ptr;
      }
      T&   operator*() const { return *ptr; }
      T*   operator->() const { return ptr; }
      T*   get() const { return ptr; }
      void reset(T* new_ptr, bool new_owns = true) {
         if (ptr && owns)
            delete ptr;
         this->ptr  = new_ptr;
         this->owns = new_owns;
      }

    private:
      T* ptr;

    public:
      bool owns;
   };

   template <typename HostFunctions = std::nullptr_t, typename Impl = interpreter, typename Options = default_options,
             typename DebugInfo = null_debug_info>
   class backend {
      using host_t    = detail::host_type_t<HostFunctions>;
      using context_t = typename Impl::template context<HostFunctions>;
      using parser_t  = typename Impl::template parser<HostFunctions, Options, DebugInfo>;

      void construct(host_t* host = nullptr) {
         mod->finalize();
         if (ctx.owns) {
            ctx->set_wasm_allocator(memory_alloc);
         }
         // Now data required by JIT is finalized; create JIT module
         // such that memory used in parsing can be released.
         if constexpr (Impl::is_jit) {
            mod->make_jit_module();

            // Important. Release the memory used by parsing.
            mod->allocator.release_base_memory();
         }
         if (ctx.owns) {
            ctx->initialize_globals();
         }
         if constexpr (!std::is_same_v<HostFunctions, std::nullptr_t>)
            HostFunctions::resolve(*mod);
         // FIXME: should not hard code knowledge of null_backend here
         if (ctx.owns) {
            if constexpr (!std::is_same_v<Impl, null_backend>)
               initialize(host);
         }
      }

    public:
      backend() {}
      backend(wasm_code&& code, host_t& host, wasm_allocator* alloc, const Options& options = Options{})
          : memory_alloc(alloc), mod(std::make_shared<module>()),
            ctx(new context_t{ parse_module(code, options), detail::get_max_call_depth(options) }),
            mod_sharable{ true } {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct(&host);
      }
      backend(wasm_code&& code, wasm_allocator* alloc, const Options& options = Options{})
          : memory_alloc(alloc), mod(std::make_shared<module>()),
            ctx(new context_t{ parse_module(code, options), detail::get_max_call_depth(options) }),
            mod_sharable{ true } {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct();
      }
      backend(wasm_code& code, host_t& host, wasm_allocator* alloc, const Options& options = Options{})
          : memory_alloc(alloc), mod(std::make_shared<module>()),
            ctx(new context_t{ parse_module(code, options), detail::get_max_call_depth(options) }),
            mod_sharable{ true } {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct(&host);
      }
      backend(wasm_code& code, wasm_allocator* alloc, const Options& options = Options{})
          : memory_alloc(alloc), mod(std::make_shared<module>()),
            ctx(new context_t{ (parse_module(code, options)), detail::get_max_call_depth(options) }),
            mod_sharable{ true } {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct();
      }
      backend(wasm_code_ptr& ptr, size_t sz, host_t& host, wasm_allocator* alloc, const Options& options = Options{})
          : memory_alloc(alloc), mod(std::make_shared<module>()),
            ctx(new context_t{ parse_module2(ptr, sz, options, true), detail::get_max_call_depth(options) }),
            mod_sharable{ true } { // single parsing. original behavior {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct(&host);
      }
      // Leap:
      //  * Contract validation only needs single parsing as the instantiated module is not cached.
      //  * JIT execution needs single parsing only.
      //  * Interpreter execution requires two-passes parsing to prevent memory mappings exhaustion
      //  * Leap reuses execution context per thread; ctx.owns is set
      //  to false when a backend is constructued
      backend(wasm_code_ptr& ptr, size_t sz, wasm_allocator* alloc, const Options& options = Options{},
              bool single_parsing = true, bool exec_ctx_by_backend = true)
          : memory_alloc(alloc), mod(std::make_shared<module>()), ctx(nullptr, exec_ctx_by_backend),
            mod_sharable{ true }, initial_max_call_depth(detail::get_max_call_depth(options)),
            initial_max_pages(detail::get_max_pages(options)) {
         if (ctx.owns) {
            ctx.reset(new context_t{ parse_module2(ptr, sz, options, single_parsing), initial_max_call_depth });
            ctx->set_max_pages(initial_max_pages);
         } else {
            parse_module2(ptr, sz, options, single_parsing);
         }
         construct();
      }

      module& parse_module(wasm_code& code, const Options& options) {
         mod->allocator.use_default_memory();
         return parser_t{ mod->allocator, options }.parse_module(code, *mod, debug);
      }

      module& parse_module2(wasm_code_ptr& ptr, size_t sz, const Options& options, bool single_parsing) {
         if (single_parsing) {
            mod->allocator.use_default_memory();
            return parser_t{ mod->allocator, options }.parse_module2(ptr, sz, *mod, debug);
         } else {
            // To prevent large number of memory mappings used, two-passes of
            // parsing are performed.
            wasm_code_ptr orig_ptr     = ptr;
            size_t        largest_size = 0;

            // First pass: finds max size of memory required by parsing.
            {
               // Memory used by this pass is freed when going out of the scope
               vm::module first_pass_module;
               first_pass_module.allocator.use_default_memory();
               parser_t{ first_pass_module.allocator, options }.parse_module2(ptr, sz, first_pass_module, debug);
               first_pass_module.finalize();
               largest_size = first_pass_module.allocator.largest_used_size();
            }

            // Second pass: uses actual required memory for final parsing
            mod->allocator.use_fixed_memory(largest_size);
            return parser_t{ mod->allocator, options }.parse_module2(orig_ptr, sz, *mod, debug);
         }
      }

      // Shares compiled module with another backend which never compiles
      // module itself.
      void share(const backend& from) {
         assert(from.mod_sharable); // `from` backend's mod is sharable
         assert(!mod_sharable);     // `to` backend's mod must not be sharable
         mod                    = from.mod;
         ctx.owns               = from.ctx.owns;
         initial_max_call_depth = from.initial_max_call_depth;
         initial_max_pages      = from.initial_max_pages;
      }

      void set_context(context_t* ctx_ptr) {
         // ctx cannot be set if it is created by the backend
         assert(!ctx.owns);
         ctx.reset(ctx_ptr, false);
      }

      inline void reset_max_call_depth() {
         // max_call_depth cannot be reset if ctx is created by the backend
         assert(!ctx.owns);
         ctx->set_max_call_depth(initial_max_call_depth);
      }

      inline void reset_max_pages() {
         // max_pages cannot be reset if ctx is created by the backend
         assert(!ctx.owns);
         ctx->set_max_pages(initial_max_pages);
      }

      template <typename... Args>
      inline auto operator()(host_t& host, const std::string_view& mod, const std::string_view& func, Args&&... args) {
         return call(host, mod, func, std::forward<Args>(args)...);
      }

      template <typename... Args>
      inline bool operator()(const std::string_view& mod, const std::string_view& func, Args&&... args) {
         return call(mod, func, std::forward<Args>(args)...);
      }

      // Only dynamic options matter.  Parser options will be ignored.
      inline backend& initialize(host_t* host, const Options& new_options) {
         ctx->set_max_call_depth(detail::get_max_call_depth(new_options));
         ctx->set_max_pages(detail::get_max_pages(new_options));
         initialize(host);
         return *this;
      }

      inline backend& initialize(host_t* host = nullptr) {
         if (memory_alloc) {
            ctx->reset();
            ctx->execute_start(host, interpret_visitor(*ctx));
         }
         return *this;
      }

      inline backend& initialize(host_t& host) { return initialize(&host); }

      template <typename... Args>
      inline bool call_indirect(host_t* host, uint32_t func_index, Args&&... args) {
         if constexpr (sys_vm_debug) {
            ctx->execute_func_table(host, debug_visitor(*ctx), func_index, std::forward<Args>(args)...);
         } else {
            ctx->execute_func_table(host, interpret_visitor(*ctx), func_index, std::forward<Args>(args)...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(host_t* host, uint32_t func_index, Args&&... args) {
         if constexpr (sys_vm_debug) {
            ctx->execute(host, debug_visitor(*ctx), func_index, std::forward<Args>(args)...);
         } else {
            ctx->execute(host, interpret_visitor(*ctx), func_index, std::forward<Args>(args)...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(host_t& host, const std::string_view& mod, const std::string_view& func, Args&&... args) {
         if constexpr (sys_vm_debug) {
            ctx->execute(&host, debug_visitor(*ctx), func, std::forward<Args>(args)...);
         } else {
            ctx->execute(&host, interpret_visitor(*ctx), func, std::forward<Args>(args)...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(const std::string_view& mod, const std::string_view& func, Args&&... args) {
         if constexpr (sys_vm_debug) {
            ctx->execute(nullptr, debug_visitor(*ctx), func, std::forward<Args>(args)...);
         } else {
            ctx->execute(nullptr, interpret_visitor(*ctx), func, std::forward<Args>(args)...);
         }
         return true;
      }

      template <typename... Args>
      inline auto call_with_return(host_t& host, const std::string_view& mod, const std::string_view& func,
                                   Args&&... args) {
         if constexpr (sys_vm_debug) {
            return ctx->execute(&host, debug_visitor(*ctx), func, std::forward<Args>(args)...);
         } else {
            return ctx->execute(&host, interpret_visitor(*ctx), func, std::forward<Args>(args)...);
         }
      }

      template <typename... Args>
      inline auto call_with_return(const std::string_view& mod, const std::string_view& func, Args&&... args) {
         if constexpr (sys_vm_debug) {
            return ctx->execute(nullptr, debug_visitor(*ctx), func, std::forward<Args>(args)...);
         } else {
            return ctx->execute(nullptr, interpret_visitor(*ctx), func, std::forward<Args>(args)...);
         }
      }

      template <typename Watchdog, typename F>
      inline void timed_run(Watchdog&& wd, F&& f) {
         if constexpr (sys_vm_has_aarch64_jit_backend && Impl::is_jit) {
            // The AArch64 JIT timeout path uses generated sampled branch polls instead of directed SIGSEGV. The review
            // suggestion to interrupt the execution thread with repeated signals is unsafe in practice: delivery can
            // land inside host/libc/helper frames where siglongjmp would corrupt process state, while consuming and
            // re-sending the signal flakes under shared-code concurrency. Generated control-flow polls observe this
            // thread-local flag; completed executions observe it at the normal timed_run boundaries below.
            std::atomic<bool>        local_timed_run_has_timed_out{ false };
            std::atomic<bool>* const previous_timed_run_has_timed_out = active_timed_run_has_timed_out;
            active_timed_run_has_timed_out                            = &local_timed_run_has_timed_out;
            auto restore_timeout_state = scope_guard{ [previous_timed_run_has_timed_out]() {
               active_timed_run_has_timed_out = previous_timed_run_has_timed_out;
            } };
            try {
               {
                  sigset_t timeout_signal_mask;
                  sigemptyset(&timeout_signal_mask);
                  sigaddset(&timeout_signal_mask, SIGSEGV);
                  sigset_t previous_signal_mask;
                  bool     timeout_signal_blocked      = false;
                  auto     restore_timeout_signal_mask = scope_guard{ [&]() {
                     if (timeout_signal_blocked) {
                        if (local_timed_run_has_timed_out.load(std::memory_order_acquire)) {
                           sigset_t pending_signals;
                           sigpending(&pending_signals);
                           if (sigismember(&pending_signals, SIGSEGV)) {
                              int drained_signal = 0;
                              sigwait(&timeout_signal_mask, &drained_signal);
                           }
                        }
                        pthread_sigmask(SIG_SETMASK, &previous_signal_mask, nullptr);
                     }
                  } };
                  pthread_sigmask(SIG_BLOCK, &timeout_signal_mask, &previous_signal_mask);
                  timeout_signal_blocked = true;
                  auto wd_guard          = std::forward<Watchdog>(wd).scoped_run(
                        [&local_timed_run_has_timed_out]() {
                           local_timed_run_has_timed_out.store(true, std::memory_order_release);
                        });
                  if (local_timed_run_has_timed_out.load(std::memory_order_acquire))
                     throw timeout_exception{ "execution timed out" };
                  std::forward<F>(f)();
                  if (local_timed_run_has_timed_out.load(std::memory_order_acquire))
                     throw timeout_exception{ "execution timed out" };
               }
               if (local_timed_run_has_timed_out.load(std::memory_order_acquire)) {
                  throw timeout_exception{ "execution timed out" };
               }
            } catch (wasm_memory_exception&) {
               if (local_timed_run_has_timed_out.load(std::memory_order_acquire)) {
                  throw timeout_exception{ "execution timed out" };
               } else {
                  throw;
               }
            }
         } else {
            // timed_run_has_timed_out lives in signal handling code because the signal handler must inspect it when
            // revoked executable pages produce a SEGV/BUS/ILL. For x86_64 this state is process-wide, not thread-local:
            // disabling shared JIT code pages can interrupt any thread currently executing that code. In production
            // this is either a normal transaction on the main thread or read-only transactions sharing the same timeout
            // window, so timing out one run intentionally stops all active executions. The older thread-local timeout
            // marker was unsafe because the watchdog callback can run on a different thread and update the wrong
            // thread's flag, leaving the actual execution thread running.
            auto reenable_code = scope_guard{ [this]() {
               --total_timed_run_in_progress;
               --mod->allocator.timed_run_in_progress;
               if (total_timed_run_in_progress == 0) {
                  timed_run_has_timed_out.store(false, std::memory_order_release);
               }
               if (mod->allocator.timed_run_in_progress == 0 && mod->allocator.code_disabled()) {
                  mod->allocator.enable_code(Impl::is_jit);
               }
            } };
            try {
               ++total_timed_run_in_progress;
               ++mod->allocator.timed_run_in_progress;
               auto wd_guard = std::forward<Watchdog>(wd).scoped_run([this]() {
                  timed_run_has_timed_out.store(true, std::memory_order_release);
                  mod->allocator.disable_code();
               });
               if (timed_run_has_timed_out.load(std::memory_order_acquire))
                  throw timeout_exception{ "execution timed out" };
               std::forward<F>(f)();
               if (timed_run_has_timed_out.load(std::memory_order_acquire))
                  throw timeout_exception{ "execution timed out" };
            } catch (wasm_memory_exception&) {
               if (timed_run_has_timed_out.load(std::memory_order_acquire)) {
                  throw timeout_exception{ "execution timed out" };
               } else {
                  throw;
               }
            }
         }
      }

      template <typename Watchdog>
      inline void execute_all(Watchdog&& wd, host_t& host) {
         timed_run(std::forward<Watchdog>(wd), [&]() {
            for (int i = 0; i < mod->exports.size(); i++) {
               if (mod->exports[i].kind == external_kind::Function) {
                  std::string s{ (const char*)mod->exports[i].field_str.raw(), mod->exports[i].field_str.size() };
                  ctx->execute(host, interpret_visitor(*ctx), s);
               }
            }
         });
      }

      template <typename Watchdog>
      inline void execute_all(Watchdog&& wd) {
         timed_run(std::forward<Watchdog>(wd), [&]() {
            for (int i = 0; i < mod->exports.size(); i++) {
               if (mod->exports[i].kind == external_kind::Function) {
                  std::string s{ (const char*)mod->exports[i].field_str.raw(), mod->exports[i].field_str.size() };
                  ctx->execute(nullptr, interpret_visitor(*ctx), s);
               }
            }
         });
      }

      inline void set_wasm_allocator(wasm_allocator* alloc) {
         memory_alloc = alloc;
         ctx->set_wasm_allocator(memory_alloc);
      }

      inline module& get_module() { return *mod; }
      inline void    exit(const std::error_code& ec) { ctx->exit(ec); }
      inline auto&   get_context() { return *ctx; }

      const DebugInfo& get_debug() const { return debug; }

    private:
      wasm_allocator*             memory_alloc = nullptr; // non owning pointer
      std::shared_ptr<module>     mod          = nullptr;
      DebugInfo                   debug;
      maybe_unique_ptr<context_t> ctx                    = nullptr;
      bool                        mod_sharable           = false; // true if mod is sharable (compiled by the backend)
      uint32_t                    initial_max_call_depth = 0;
      uint32_t                    initial_max_pages      = 0;
   };
}} // namespace sysio::vm
