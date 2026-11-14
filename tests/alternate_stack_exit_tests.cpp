#include <sysio/vm/backend.hpp>

#include <catch2/catch.hpp>

#include <pthread.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if SYS_VM_HAS_JIT_BACKEND

using namespace sysio::vm;

namespace {

/*
(module
  (type (;0;) (func))
  (import "env" "fail" (func (;0;) (type 0)))
  (import "env" "reenter" (func (;1;) (type 0)))
  (func (;2;) (type 0) call 0)
  (func (;3;) (type 0) unreachable)
  (func (;4;) (type 0) i32.const -1 i32.load drop)
  (func (;5;) (type 0) call 1)
  ;; Never called. Its 4000 locals make the worst-case stack too large for the caller's stack, so every call
  ;; into this module runs on the alternate stack.
  (func (;6;) (type 0) (local i64 ... 4000 times))
  (memory (;0;) 1)
  (export "host_throw" (func 2))
  (export "trap" (func 3))
  (export "fault" (func 4))
  (export "reenter" (func 5)))
*/
wasm_code alternate_stack_wasm = {
   0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
   // type section
   0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
   // import section
   0x02, 0x1a, 0x02,
   0x03, 0x65, 0x6e, 0x76, 0x04, 0x66, 0x61, 0x69, 0x6c, 0x00, 0x00,
   0x03, 0x65, 0x6e, 0x76, 0x07, 0x72, 0x65, 0x65, 0x6e, 0x74, 0x65, 0x72, 0x00, 0x00,
   // function section
   0x03, 0x06, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
   // memory section
   0x05, 0x03, 0x01, 0x00, 0x01,
   // export section
   0x07, 0x27, 0x04,
   0x0a, 0x68, 0x6f, 0x73, 0x74, 0x5f, 0x74, 0x68, 0x72, 0x6f, 0x77, 0x00, 0x02,
   0x04, 0x74, 0x72, 0x61, 0x70, 0x00, 0x03,
   0x05, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x00, 0x04,
   0x07, 0x72, 0x65, 0x65, 0x6e, 0x74, 0x65, 0x72, 0x00, 0x05,
   // code section
   0x0a, 0x1e, 0x05,
   0x04, 0x00, 0x10, 0x00, 0x0b,
   0x03, 0x00, 0x00, 0x0b,
   0x08, 0x00, 0x41, 0x7f, 0x28, 0x02, 0x00, 0x1a, 0x0b,
   0x04, 0x00, 0x10, 0x01, 0x0b,
   0x05, 0x01, 0xa0, 0x1f, 0x7e, 0x0b
};

struct host_failure {};

/// How a call into the module ended.
enum class call_exit { returned, host_exception, trap, memory_fault, unexpected };

template <typename F>
call_exit exit_of(F&& f) {
   try {
      f();
   } catch (const host_failure&) {
      return call_exit::host_exception;
   } catch (const wasm_interpreter_exception&) {
      return call_exit::trap;
   } catch (const wasm_memory_exception&) {
      return call_exit::memory_fault;
   } catch (...) { return call_exit::unexpected; }
   return call_exit::returned;
}

/// A thread whose stack has a large unmapped hole directly above it.
///
/// mmap allocates top-down on Linux, so a mapping made while the thread runs lands in that hole or higher: above
/// the thread's stack. That is the layout in which glibc's fortified longjmp rejects a jump from the mapping back
/// to the thread's stack.
class low_stack_thread {
 public:
   low_stack_thread() {
      void* reservation = mmap(nullptr, stack_size + hole_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      REQUIRE(reservation != MAP_FAILED);
      _stack = static_cast<char*>(reservation);
      REQUIRE(mprotect(_stack, stack_size, PROT_READ | PROT_WRITE) == 0);
      REQUIRE(munmap(_stack + stack_size, hole_size) == 0);
   }
   low_stack_thread(const low_stack_thread&)            = delete;
   low_stack_thread& operator=(const low_stack_thread&) = delete;
   ~low_stack_thread() { munmap(_stack, stack_size); }

   /// One past the highest address of the thread's stack.
   const char* stack_end() const { return _stack + stack_size; }

   /// Runs `f` to completion on the thread. `f` must not throw or use Catch2 assertions.
   template <typename F>
   void run(F&& f) {
      pthread_attr_t attr;
      REQUIRE(pthread_attr_init(&attr) == 0);
      REQUIRE(pthread_attr_setstack(&attr, _stack, stack_size) == 0);
      pthread_t thread;
      const int created = pthread_create(
            &thread, &attr,
            [](void* data) -> void* {
               (*static_cast<std::remove_reference_t<F>*>(data))();
               return nullptr;
            },
            &f);
      pthread_attr_destroy(&attr);
      REQUIRE(created == 0);
      REQUIRE(pthread_join(thread, nullptr) == 0);
   }

 private:
   static constexpr std::size_t stack_size = std::size_t{ 8 } * 1024 * 1024;
   static constexpr std::size_t hole_size  = std::size_t{ 4 } * 1024 * 1024 * 1024;

   char* _stack = nullptr;
};

} // namespace

// Each way generated code can stop early is a siglongjmp: a host function that throws, a trap raised by the
// generated code, a memory fault taken in the signal handler, and any of those ending a nested call.
TEST_CASE("Early exits from the alternate stack reach a caller whose stack lies below it", "[alternate_stack_exit]") {
   struct exit_host;
   using rhf_t     = registered_host_functions<exit_host>;
   using backend_t = backend<rhf_t, jit>;
   struct exit_host {
      backend_t*  bkend = nullptr;
      /// Frame of the last `fail` call, which runs on the stack the generated code uses.
      const void* host_call_frame = nullptr;

      void fail() {
         host_call_frame = __builtin_frame_address(0);
         throw host_failure{};
      }
      void reenter() { bkend->call(*this, "env", "trap"); }
   };
   rhf_t::add<&exit_host::fail>("env", "fail");
   rhf_t::add<&exit_host::reenter>("env", "reenter");

   // Everything the module maps exists before the thread's stack is placed, so the hole above that stack is
   // still free for the alternate stack of each call.
   wasm_allocator wa;
   exit_host      host;
   backend_t      bkend(alternate_stack_wasm, host, &wa);
   host.bkend = &bkend;

   low_stack_thread thread;
   call_exit        host_throw = call_exit::unexpected;
   call_exit        trap       = call_exit::unexpected;
   call_exit        fault      = call_exit::unexpected;
   call_exit        reenter    = call_exit::unexpected;
   thread.run([&] {
      host_throw = exit_of([&] { bkend.call(host, "env", "host_throw"); });
      trap       = exit_of([&] { bkend.call(host, "env", "trap"); });
      fault      = exit_of([&] { bkend.call(host, "env", "fault"); });
      reenter    = exit_of([&] { bkend.call(host, "env", "reenter"); });
   });

#if defined(__linux__)
   // The generated code ran on a mapping above the thread's stack, not on that stack.
   REQUIRE(static_cast<const char*>(host.host_call_frame) >= thread.stack_end());
#endif
   CHECK(host_throw == call_exit::host_exception);
   CHECK(trap == call_exit::trap);
   CHECK(fault == call_exit::memory_fault);
   CHECK(reenter == call_exit::trap);
}

#endif
