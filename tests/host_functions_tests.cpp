#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <string>

#include <catch2/catch.hpp>

#include <sysio/vm/backend.hpp>
#include "wasm_config.hpp"
#include "utils.hpp"

using namespace sysio;
using namespace sysio::vm;

// host functions that are C-style functions
// wasm hex
/* Code used to generate test, compile with sysio-cpp v1.6.2 with minor manual edits to remove unneeded imports
 * extern "C" {
      struct state_t { float f; int i; };
      [[sysio::wasm_import]]
      void c_style_host_function_0();
      [[sysio::wasm_import]]
      void c_style_host_function_1(int);
      [[sysio::wasm_import]]
      void c_style_host_function_2(int, int);
      [[sysio::wasm_import]]
      void c_style_host_function_3(int, float);
      [[sysio::wasm_import]]
      void c_style_host_function_4(const state_t&);

      [[sysio::wasm_entry]]
      void apply(unsigned long long a, unsigned long long b, unsigned long long c) {
         if (a == 0)
            c_style_host_function_0();
         else if (a == 1)
            c_style_host_function_1((int)b);
         else if (a == 2)
            c_style_host_function_2((int)b, (int)c);
         else if (a == 3)
            c_style_host_function_3((int)b, *((int*)&c));
         else if (a == 4) {
            state_t s = {*((float*)&c), (int)b};
            c_style_host_function_4(s);
         }
      }
   } */

#include "host_functions_tests_0.wasm.hpp"
// no return value and no input parameters
int c_style_host_function_state = 0;
struct state_t {
   float f = 0;
   int   i = 0;
};
void c_style_host_function_0() {
   c_style_host_function_state = 1;
}
void c_style_host_function_1(int s) {
   c_style_host_function_state = s;
}
void c_style_host_function_2(int a, int b) {
   c_style_host_function_state = a+b;
}
void c_style_host_function_3(int a, float b) {
   c_style_host_function_state = a+b;
}
void c_style_host_function_4(const state_t& ss) {
   c_style_host_function_state = ss.i;
}

// Combinations:
// Member/Free
// host object/no host object - done
// parameters: none/bool/int32_t/uint32_t/int64_t/uint64_t/float/double/cv-pointer/cv-reference/two
// result: void/bool/int32_t/uint32_t/int64_t/uint64_t/float/double/cv-pointer/cv_reference - done
// call/call_indirect/direct execution - done
//
// Things that a host function might do:
// - call back into wasm
// - exit
// - wasm -> native -> wasm -> native -> exit
// - throw
// - calling into a different execution context
// - wasm1/wasm2 mixed -> native -> exit wasm2


namespace {

template<typename Host>
struct cnv : type_converter<Host> {
   using type_converter<Host>::type_converter;
   using type_converter<Host>::from_wasm;
   using type_converter<Host>::to_wasm;
   template<typename T>
   auto from_wasm(void* ptr) const -> std::enable_if_t<std::is_pointer_v<T>, T> {
      return static_cast<T>(ptr);
   }
   template<typename T>
   auto from_wasm(void* ptr) const -> std::enable_if_t<std::is_lvalue_reference_v<T>, T> {
      return *static_cast<std::remove_reference_t<T>*>(ptr);
   }
   template<typename T>
   auto to_wasm(T*&& ptr) -> const volatile void* {
      return ptr;
   }
   template<typename T>
   auto to_wasm(T& ref) -> const volatile void* {
      return &ref;
   }
};

template<typename T>
struct ref {
   ref() = default;
   ref(T& arg) : val(&arg) {}
   operator T&() { return *val; }
   T* val;
};

template<typename T>
using maybe_ref = std::conditional_t<std::is_reference_v<T>, ref<std::remove_reference_t<T>>, T>;

template<typename T>
maybe_ref<T> global_test_value;

struct static_host_function {
   template<typename T>
   static void put(T t) { global_test_value<T> = t; }
   template<typename T>
   static T get() { return global_test_value<T>; }
};

struct member_host_function {
   template<typename T>
   void put(T t) { global_test_value<T> = t; }
   template<typename T>
   T get() const { return global_test_value<T>; }
};

}

#include "host_functions_tests_1.wasm.hpp"

wasm_code host_functions_tests_1_code{
   host_functions_tests_1_wasm + 0,
   host_functions_tests_1_wasm + sizeof(host_functions_tests_1_wasm)};

template<class Functions, class Host, class Impl>
struct init_backend {
   init_backend(Host* host) : bkend{host_functions_tests_1_code, (init_host_functions(), &wa)}, _host(host) {
      rhf_t::resolve(bkend.get_module());
   }

   static void init_host_functions() {
      add<bool>("b");
      add<int32_t>("i32");
      add<uint32_t>("ui32");
      add<int64_t>("i64");
      add<uint64_t>("ui64");
      add<float>("f32");
      add<double>("f64");
      add<char*>("ptr");
      add<const char*>("cptr");
      add<volatile char*>("vptr");
      add<const volatile char*>("cvptr");
      add<char&>("ref");
      add<const char&>("cref");
      add<volatile char&>("vref");
      add<const volatile char&>("cvref");
   }

   template<typename T>
   static void add(const std::string& name) {
      rhf_t::template add<&Functions::template put<T>>("env", "put_" + name);
      rhf_t::template add<&Functions::template get<T>>("env", "get_" + name);
   }
   // forwarding functions
   template<typename... A>
   auto call_with_return(A&&... a) {
      if constexpr (std::is_same_v<Host, std::nullptr_t>) {
         return bkend.call_with_return(static_cast<A&&>(a)...);
      } else {
         return bkend.call_with_return(*_host, static_cast<A&&>(a)...);
      }
   }
   template<typename... A>
   auto call(A&&... a) {
      if constexpr (std::is_same_v<Host, std::nullptr_t>) {
         return bkend.call(static_cast<A&&>(a)...);
      } else {
         return bkend.call(*_host, static_cast<A&&>(a)...);
      }
   }
   decltype(auto) get_context() { return bkend.get_context(); }

   using rhf_t     = sysio::vm::registered_host_functions<Host, execution_interface, cnv<Host>>;
   using backend_t = sysio::vm::backend<rhf_t, Impl>;
   wasm_allocator wa;
   backend_t bkend{host_functions_tests_1_code, &wa};
   Host * _host;
};

const std::vector<std::string> fun_prefixes = { "", "call.", "call_indirect." };

template<typename T>
std::vector<T> test_values = { 0, 1, std::numeric_limits<T>::min(), std::numeric_limits<T>::max() };

template<>
std::vector<bool> test_values<bool> = { true, false };

int next_ptr_offset() {
   static int counter = 4;
   return counter++;
}

template<typename T, typename B>
void check_put(B& bkend, const std::string& name) {
   for(auto fun : fun_prefixes) {
      fun += "put_" + name;
      for(const T value : test_values<T>) {
         bkend.call("env", fun, value);
         CHECK(global_test_value<T> == value);
      }
   }
}

template<typename T, typename B>
void check_put_ptr(B& bkend, const std::string& name) {
   for(auto fun : fun_prefixes) {
      fun += "put_" + name;
      int offset = next_ptr_offset();
      const T value = const_cast<T>(bkend.get_context().linear_memory() + offset);
      bkend.call("env", fun, value);
      CHECK(global_test_value<T> == value);
   }
}

template<typename T, typename B>
void check_put_ref(B& bkend, const std::string& name) {
   for(auto fun : fun_prefixes) {
      fun += "put_" + name;
      int offset = next_ptr_offset();
      const auto value = &const_cast<T>(*(bkend.get_context().linear_memory() + offset));
      bkend.call("env", fun, value);
      CHECK(global_test_value<T>.val == value);
   }
}

template<typename T, typename B, typename F>
void check_get(B& bkend, const std::string& name, F getter) {
   for(auto fun : fun_prefixes) {
      fun += "get_" + name;
      for(const T value : test_values<T>) {
         global_test_value<T> = value;
         CHECK(getter(bkend.call_with_return("env", fun)) == value);
      }
   }
}

template<typename T, typename B>
void check_get_ptr(B& bkend, const std::string& name) {
   for(auto fun : fun_prefixes) {
      fun += "get_" + name;
      int offset = next_ptr_offset();
      global_test_value<T> = const_cast<T>(bkend.get_context().linear_memory() + offset);
      CHECK(bkend.call_with_return("env", fun)->to_ui32() == offset);
   }
}

template<typename T, typename B>
void check_get_ref(B& bkend, const std::string& name) {
   for(auto fun : fun_prefixes) {
      fun += "get_" + name;
      int offset = next_ptr_offset();
      global_test_value<T> = const_cast<T>(*(bkend.get_context().linear_memory() + offset));
      CHECK(bkend.call_with_return("env", fun)->to_ui32() == offset);
   }
}

template<class Backend>
void test_parameters(Backend&& bkend) {
   for(auto fun : fun_prefixes) {
      fun += "put_b";
      bkend.call("env", fun, true);
      CHECK(global_test_value<bool> == true);
      bkend.call("env", fun, false);
      CHECK(global_test_value<bool> == false);
      // Extra tests for bool:
      bkend.call("env", fun, 42);
      CHECK(global_test_value<bool> == true);
      bkend.call("env", fun, 0x10000);
      CHECK(global_test_value<bool> == true);
   }

   check_put<int32_t>(bkend, "i32");
   check_put<uint32_t>(bkend, "ui32");
   check_put<int64_t>(bkend, "i64");
   check_put<uint64_t>(bkend, "ui64");
   check_put<float>(bkend, "f32");
   check_put<double>(bkend, "f64");
   check_put_ptr<char*>(bkend, "ptr");
   check_put_ptr<const char*>(bkend, "cptr");
   check_put_ptr<volatile char*>(bkend, "vptr");
   check_put_ptr<const volatile char*>(bkend, "cvptr");
   check_put_ref<char&>(bkend, "ref");
   check_put_ref<const char&>(bkend, "cref");
   check_put_ref<volatile char&>(bkend, "vref");
   check_put_ref<const volatile char&>(bkend, "cvref");
}

BACKEND_TEST_CASE( "Test host function parameters", "[host_functions_parameters]" ) {
   test_parameters(init_backend<static_host_function, nullptr_t, TestType>{nullptr});
   member_host_function mhf;
   test_parameters(init_backend<member_host_function, member_host_function, TestType>{&mhf});
}

template<class Backend>
void test_results(Backend&& bkend) {
   for(auto fun : {/*"get_b",*/ "call.get_b", "call_indirect.get_b"}) {
      global_test_value<bool> = false;
      CHECK(bkend.call_with_return("env", fun)->to_ui32() == 0u);
      global_test_value<bool> = true;
      CHECK(bkend.call_with_return("env", fun)->to_ui32() == 1u);
   }

   check_get<int32_t>(bkend, "i32", [](auto x){ return x->to_i32(); });
   check_get<uint32_t>(bkend, "ui32", [](auto x){ return x->to_ui32(); });
   check_get<int64_t>(bkend, "i64", [](auto x){ return x->to_i64(); });
   check_get<uint64_t>(bkend, "ui64", [](auto x){ return x->to_ui64(); });
   check_get<float>(bkend, "f32", [](auto x){ return x->to_f32(); });
   check_get<double>(bkend, "f64", [](auto x){ return x->to_f64(); });
   check_get_ptr<char*>(bkend, "ptr");
   check_get_ptr<const char*>(bkend, "cptr");
   check_get_ptr<volatile char*>(bkend, "vptr");
   check_get_ptr<const volatile char*>(bkend, "cvptr");
   check_get_ref<char&>(bkend, "ref");
   check_get_ref<const char&>(bkend, "cref");
   check_get_ref<volatile char&>(bkend, "vref");
   check_get_ref<const volatile char&>(bkend, "cvref");
}

BACKEND_TEST_CASE( "Test host function results", "[host_functions_results]" ) {
   test_results(init_backend<static_host_function, nullptr_t, TestType>{nullptr});
   member_host_function mhf;
   test_results(init_backend<member_host_function, member_host_function, TestType>{&mhf});
}

BACKEND_TEST_CASE( "Test C-style host function system", "[C-style_host_functions_tests]") {
   wasm_allocator wa;
   using rhf_t     = sysio::vm::registered_host_functions<standalone_function_t, execution_interface, cnv<standalone_function_t>>;
   using backend_t = sysio::vm::backend<rhf_t, TestType>;
   rhf_t::add<&c_style_host_function_0>("env", "c_style_host_function_0");
   rhf_t::add<&c_style_host_function_1>("env", "c_style_host_function_1");
   rhf_t::add<&c_style_host_function_2>("env", "c_style_host_function_2");
   rhf_t::add<&c_style_host_function_3>("env", "c_style_host_function_3");
   rhf_t::add<&c_style_host_function_4>("env", "c_style_host_function_4");

   backend_t bkend(host_functions_test_0_wasm, &wa);

   bkend.call("env", "apply", (uint64_t)0, (uint64_t)0, (uint64_t)0);
   CHECK(c_style_host_function_state == 1);

   bkend.call("env", "apply", (uint64_t)1, (uint64_t)2, (uint64_t)0);
   CHECK(c_style_host_function_state == 2);

   bkend.call("env", "apply", (uint64_t)2, (uint64_t)1, (uint64_t)2);
   CHECK(c_style_host_function_state == 3);

   float f = 2.4f;
   bkend.call("env", "apply", (uint64_t)3, (uint64_t)2, (uint64_t)bit_cast<uint32_t>(f));
   CHECK(c_style_host_function_state == 0x40199980);

   bkend.call("env", "apply", (uint64_t)4, (uint64_t)5, (uint64_t)bit_cast<uint32_t>(f));
   CHECK(c_style_host_function_state == 5);
}

struct my_host_functions {
   static int test(int value) { return value + 42; }
   static int test2(int value) { return value * 42; }
};

extern wasm_allocator wa;

BACKEND_TEST_CASE( "Testing host functions", "[host_functions_test]" ) {
   my_host_functions host;
   using rhf_t = registered_host_functions<standalone_function_t>;
   rhf_t::add<&my_host_functions::test>("host", "test");
   rhf_t::add<&my_host_functions::test2>("host", "test2");

   using backend_t = backend<rhf_t, TestType>;

   auto code = read_wasm( host_wasm );
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "test", UINT32_C(5))->to_i32() == 49);
   CHECK(bkend.call_with_return("env", "test.indirect", UINT32_C(5), UINT32_C(0))->to_i32() == 47);
   CHECK(bkend.call_with_return("env", "test.indirect", UINT32_C(5), UINT32_C(1))->to_i32() == 210);
   CHECK(bkend.call_with_return("env", "test.indirect", UINT32_C(5), UINT32_C(2))->to_i32() == 49);
   CHECK_THROWS_AS(bkend.call("env", "test.indirect", UINT32_C(5), UINT32_C(3)), std::exception);
   CHECK(bkend.call_with_return("env", "test.local-call", UINT32_C(5))->to_i32() == 147);
}

struct has_stateful_conversion {
   uint32_t value;
};

struct stateful_conversion {
   operator has_stateful_conversion() const { return { value }; }
   uint32_t value;
};

struct host_functions_stateful_converter {
   static int test(has_stateful_conversion x) { return x.value + 42; }
   static int test2(has_stateful_conversion x) { return x.value * 42; }
};

struct stateful_cnv : type_converter<standalone_function_t> {
   using type_converter::type_converter;
   using type_converter::from_wasm;
   template<typename T>
   auto from_wasm(uint32_t val) const
      -> std::enable_if_t<std::is_same_v<T, has_stateful_conversion>,
                          stateful_conversion> {
      return { val };
   }
};

BACKEND_TEST_CASE( "Testing stateful ", "[host_functions_stateful_converter]") {
   host_functions_stateful_converter host;
   using rhf_t = registered_host_functions<standalone_function_t, execution_interface, stateful_cnv>;
   rhf_t::add<&host_functions_stateful_converter::test>("host", "test");
   rhf_t::add<&host_functions_stateful_converter::test2>("host", "test2");

   using backend_t = backend<rhf_t, TestType>;

   auto code = read_wasm( host_wasm );
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "test", UINT32_C(5))->to_i32() == 49);
   CHECK(bkend.call_with_return("env", "test.indirect", UINT32_C(5), UINT32_C(0))->to_i32() == 47);
   CHECK(bkend.call_with_return("env", "test.indirect", UINT32_C(5), UINT32_C(1))->to_i32() == 210);
   CHECK(bkend.call_with_return("env", "test.indirect", UINT32_C(5), UINT32_C(2))->to_i32() == 49);
   CHECK_THROWS_AS(bkend.call("env", "test.indirect", UINT32_C(5), UINT32_C(3)), std::exception);
   CHECK(bkend.call_with_return("env", "test.local-call", UINT32_C(5))->to_i32() == 147);
}

// Test overloaded frow_wasm and order of destruction of converters


#warning TODO figure out a way to make this work with the new host function system

#if 0

struct has_multi_converter {
   uint32_t v1;
   uint32_t v2;
};
static std::vector<int> multi_converter_destructor_order;
struct multi_converter {
   uint32_t v1;
   uint32_t v2;
   operator has_multi_converter() const { return { v1, v2 }; }
   ~multi_converter() { multi_converter_destructor_order.push_back(v1); }
   multi_converter(multi_converter&&) = delete;
};

struct host_functions_multi_converter {
   static unsigned test(has_multi_converter x, has_multi_converter y) {
      return 1*x.v1 + 10*x.v2 + 100*y.v1 + 1000*y.v2;
   }
};

struct multi_cnv : type_converter<standalone_function_t> {
   using type_converter::type_converter;
   using type_converter::from_wasm;
   template<typename T>
   auto from_wasm(uint32_t v0, uint32_t v1) const
      -> std::enable_if_t<std::is_same_v<T, has_multi_converter>,
                          multi_converter> {
      return { v0, v1 };
   }
};

BACKEND_TEST_CASE( "Testing multi ", "[host_functions_multi_converter]") {
   host_functions_multi_converter host;
   using rhf_t = registered_host_functions<standalone_function_t, execution_interface, multi_cnv>;
   rhf_t::add<&host_functions_multi_converter::test>("host", "test");

   using backend_t = backend<rhf_t, TestType>;

   /*
     (module
       (func (export "test") (import "host" "test") (param i32 i32 i32 i32) (result i32))
     )
   */

   wasm_code code = {
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x09, 0x01, 0x60,
      0x04, 0x7f, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, 0x02, 0x0d, 0x01, 0x04, 0x68,
      0x6f, 0x73, 0x74, 0x04, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00, 0x07, 0x08,
      0x01, 0x04, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00
   };
   backend_t bkend( code, &wa );

   multi_converter_destructor_order.clear();
   CHECK(bkend.call_with_return("env", "test", UINT32_C(1), UINT32_C(2), UINT32_C(3), UINT32_C(4))->to_i32() == 4321);
   CHECK(multi_converter_destructor_order == std::vector{1, 3});
}

#endif

struct test_exception {};

struct host_functions_throw {
   static int test(int) { throw test_exception{}; }
};

BACKEND_TEST_CASE( "Testing throwing host functions", "[host_functions_throw_test]" ) {
   host_functions_throw host;
   using rhf_t = registered_host_functions<standalone_function_t>;
   rhf_t::add<&host_functions_throw::test>("host", "test");
   rhf_t::add<&host_functions_throw::test>("host", "test2");

   using backend_t = backend<rhf_t, TestType>;

   auto code = read_wasm( host_wasm );
   backend_t bkend( code, &wa );;

   CHECK_THROWS_AS(bkend.call("env", "test", UINT32_C(2)), test_exception);
}

template<typename Impl>
struct host_functions_exit {
   typename Impl::template context<registered_host_functions<host_functions_exit>> * context;
   int test(int) { context->exit(); return 0; }
};

BACKEND_TEST_CASE( "Testing exiting host functions", "[host_functions_exit_test]" ) {
   using rhf_t = registered_host_functions<host_functions_exit<TestType>>;
   rhf_t::template add<&host_functions_exit<TestType>::test>("host", "test");
   rhf_t::template add<&host_functions_exit<TestType>::test>("host", "test2");

   using backend_t = backend<rhf_t, TestType>;

   auto code = read_wasm( host_wasm );
   backend_t bkend( code, &wa );
   host_functions_exit<TestType> host{&bkend.get_context()};

   CHECK(!bkend.call_with_return(host, "env", "test", UINT32_C(2)));
}
