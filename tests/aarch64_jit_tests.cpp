#include "utils.hpp"
#include <catch2/catch.hpp>
#include <sysio/vm/backend.hpp>

using namespace sysio::vm;

extern wasm_allocator wa;

#if SYS_VM_HAS_AARCH64_JIT_BACKEND
namespace {
struct aarch64_jit_imports {
   /// Returns a small deterministic i32 value so the AArch64 host-call bridge can be tested in isolation.
   static uint32_t add_one(uint32_t lhs, uint32_t rhs) { return lhs + rhs + 1; }

   /// Applies positional weights so the AArch64 imported-call bridge verifies both stack staging and argument order.
   static uint32_t sum_nine(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6,
                            uint32_t a7, uint32_t a8) {
      return a0 + (a1 * 2u) + (a2 * 3u) + (a3 * 4u) + (a4 * 5u) + (a5 * 6u) + (a6 * 7u) + (a7 * 8u) + (a8 * 9u);
   }
};

/// Appends an unsigned LEB128 value to a generated WASM module section.
void append_u32_leb(std::vector<uint8_t>& bytes, uint32_t value) {
   do {
      uint8_t byte = value & UINT8_C(0x7f);
      value >>= 7;
      if (value != 0) {
         byte |= UINT8_C(0x80);
      }
      bytes.push_back(byte);
   } while (value != 0);
}

/// Appends a signed i32 LEB128 value for generated i32.const instruction immediates.
void append_i32_leb(std::vector<uint8_t>& bytes, int32_t value) {
   bool more = true;
   while (more) {
      uint8_t byte = static_cast<uint8_t>(value) & UINT8_C(0x7f);
      value >>= 7;
      const bool sign_bit = (byte & UINT8_C(0x40)) != 0;
      more                = !((value == 0 && !sign_bit) || (value == -1 && sign_bit));
      if (more) {
         byte |= UINT8_C(0x80);
      }
      bytes.push_back(byte);
   }
}

/// Appends a complete WASM section with a generated LEB128 payload length.
void append_wasm_section(std::vector<uint8_t>& module, uint8_t section_id, const std::vector<uint8_t>& payload) {
   module.push_back(section_id);
   append_u32_leb(module, static_cast<uint32_t>(payload.size()));
   module.insert(module.end(), payload.begin(), payload.end());
}

/// Builds a minimal module whose local frame requires large AArch64 SP and local-slot offsets.
std::vector<uint8_t> make_aarch64_large_local_frame_wasm() {
   constexpr uint32_t local_count      = 512;
   constexpr uint32_t high_local_index = local_count - 1;
   constexpr uint8_t  section_type     = 1;
   constexpr uint8_t  section_function = 3;
   constexpr uint8_t  section_export   = 7;
   constexpr uint8_t  section_code     = 10;
   constexpr uint8_t  wasm_i64         = 0x7e;

   std::vector<uint8_t> module = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };

   std::vector<uint8_t> type_section;
   append_u32_leb(type_section, 1);
   type_section.push_back(0x60);
   append_u32_leb(type_section, 0);
   append_u32_leb(type_section, 1);
   type_section.push_back(wasm_i64);
   append_wasm_section(module, section_type, type_section);

   std::vector<uint8_t> function_section;
   append_u32_leb(function_section, 1);
   append_u32_leb(function_section, 0);
   append_wasm_section(module, section_function, function_section);

   std::vector<uint8_t>       export_section;
   const std::vector<uint8_t> export_name = { 'r', 'e', 'a', 'd' };
   append_u32_leb(export_section, 1);
   append_u32_leb(export_section, static_cast<uint32_t>(export_name.size()));
   export_section.insert(export_section.end(), export_name.begin(), export_name.end());
   export_section.push_back(0);
   append_u32_leb(export_section, 0);
   append_wasm_section(module, section_export, export_section);

   std::vector<uint8_t> body;
   append_u32_leb(body, 1);
   append_u32_leb(body, local_count);
   body.push_back(wasm_i64);
   body.push_back(0x42);
   append_u32_leb(body, 42);
   body.push_back(0x21);
   append_u32_leb(body, high_local_index);
   body.push_back(0x20);
   append_u32_leb(body, high_local_index);
   body.push_back(0x0b);

   std::vector<uint8_t> code_section;
   append_u32_leb(code_section, 1);
   append_u32_leb(code_section, static_cast<uint32_t>(body.size()));
   code_section.insert(code_section.end(), body.begin(), body.end());
   append_wasm_section(module, section_code, code_section);

   return module;
}

/// Builds a module whose internal direct call has more arguments than available staging registers.
std::vector<uint8_t> make_aarch64_wide_internal_call_wasm() {
   constexpr uint32_t argument_count   = 9;
   constexpr uint8_t  section_type     = 1;
   constexpr uint8_t  section_function = 3;
   constexpr uint8_t  section_export   = 7;
   constexpr uint8_t  section_code     = 10;
   constexpr uint8_t  wasm_i64         = 0x7e;

   std::vector<uint8_t> module = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };

   std::vector<uint8_t> type_section;
   append_u32_leb(type_section, 2);
   type_section.push_back(0x60);
   append_u32_leb(type_section, argument_count);
   for (uint32_t i = 0; i < argument_count; ++i) { type_section.push_back(wasm_i64); }
   append_u32_leb(type_section, 1);
   type_section.push_back(wasm_i64);
   type_section.push_back(0x60);
   append_u32_leb(type_section, 0);
   append_u32_leb(type_section, 1);
   type_section.push_back(wasm_i64);
   append_wasm_section(module, section_type, type_section);

   std::vector<uint8_t> function_section;
   append_u32_leb(function_section, 2);
   append_u32_leb(function_section, 0);
   append_u32_leb(function_section, 1);
   append_wasm_section(module, section_function, function_section);

   std::vector<uint8_t>       export_section;
   const std::vector<uint8_t> export_name = { 'r', 'u', 'n' };
   append_u32_leb(export_section, 1);
   append_u32_leb(export_section, static_cast<uint32_t>(export_name.size()));
   export_section.insert(export_section.end(), export_name.begin(), export_name.end());
   export_section.push_back(0);
   append_u32_leb(export_section, 1);
   append_wasm_section(module, section_export, export_section);

   std::vector<uint8_t> sum_body;
   append_u32_leb(sum_body, 0);
   sum_body.push_back(0x20);
   append_u32_leb(sum_body, 0);
   sum_body.push_back(0x42);
   append_u32_leb(sum_body, 1);
   sum_body.push_back(0x7e);
   for (uint32_t i = 1; i < argument_count; ++i) {
      sum_body.push_back(0x20);
      append_u32_leb(sum_body, i);
      sum_body.push_back(0x42);
      append_u32_leb(sum_body, i + 1u);
      sum_body.push_back(0x7e);
      sum_body.push_back(0x7c);
   }
   sum_body.push_back(0x0b);

   std::vector<uint8_t> run_body;
   append_u32_leb(run_body, 0);
   for (uint32_t i = 1; i <= argument_count; ++i) {
      run_body.push_back(0x42);
      append_u32_leb(run_body, i);
   }
   run_body.push_back(0x10);
   append_u32_leb(run_body, 0);
   run_body.push_back(0x0b);

   std::vector<uint8_t> code_section;
   append_u32_leb(code_section, 2);
   append_u32_leb(code_section, static_cast<uint32_t>(sum_body.size()));
   code_section.insert(code_section.end(), sum_body.begin(), sum_body.end());
   append_u32_leb(code_section, static_cast<uint32_t>(run_body.size()));
   code_section.insert(code_section.end(), run_body.begin(), run_body.end());
   append_wasm_section(module, section_code, code_section);

   return module;
}

/// Builds a module whose imported direct call has more arguments than available staging registers.
std::vector<uint8_t> make_aarch64_wide_imported_call_wasm() {
   constexpr uint32_t argument_count   = 9;
   constexpr uint8_t  section_type     = 1;
   constexpr uint8_t  section_import   = 2;
   constexpr uint8_t  section_function = 3;
   constexpr uint8_t  section_export   = 7;
   constexpr uint8_t  section_code     = 10;
   constexpr uint8_t  wasm_i32         = 0x7f;

   std::vector<uint8_t> module = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };

   std::vector<uint8_t> type_section;
   append_u32_leb(type_section, 2);
   type_section.push_back(0x60);
   append_u32_leb(type_section, argument_count);
   for (uint32_t i = 0; i < argument_count; ++i) { type_section.push_back(wasm_i32); }
   append_u32_leb(type_section, 1);
   type_section.push_back(wasm_i32);
   type_section.push_back(0x60);
   append_u32_leb(type_section, 0);
   append_u32_leb(type_section, 1);
   type_section.push_back(wasm_i32);
   append_wasm_section(module, section_type, type_section);

   std::vector<uint8_t>       import_section;
   const std::vector<uint8_t> module_name = { 'e', 'n', 'v' };
   const std::vector<uint8_t> import_name = { 's', 'u', 'm', '9' };
   append_u32_leb(import_section, 1);
   append_u32_leb(import_section, static_cast<uint32_t>(module_name.size()));
   import_section.insert(import_section.end(), module_name.begin(), module_name.end());
   append_u32_leb(import_section, static_cast<uint32_t>(import_name.size()));
   import_section.insert(import_section.end(), import_name.begin(), import_name.end());
   import_section.push_back(0);
   append_u32_leb(import_section, 0);
   append_wasm_section(module, section_import, import_section);

   std::vector<uint8_t> function_section;
   append_u32_leb(function_section, 1);
   append_u32_leb(function_section, 1);
   append_wasm_section(module, section_function, function_section);

   std::vector<uint8_t>       export_section;
   const std::vector<uint8_t> export_name = { 'r', 'u', 'n' };
   append_u32_leb(export_section, 1);
   append_u32_leb(export_section, static_cast<uint32_t>(export_name.size()));
   export_section.insert(export_section.end(), export_name.begin(), export_name.end());
   export_section.push_back(0);
   append_u32_leb(export_section, 1);
   append_wasm_section(module, section_export, export_section);

   std::vector<uint8_t> run_body;
   append_u32_leb(run_body, 0);
   for (uint32_t i = 1; i <= argument_count; ++i) {
      run_body.push_back(0x41);
      append_u32_leb(run_body, i);
   }
   run_body.push_back(0x10);
   append_u32_leb(run_body, 0);
   run_body.push_back(0x0b);

   std::vector<uint8_t> code_section;
   append_u32_leb(code_section, 1);
   append_u32_leb(code_section, static_cast<uint32_t>(run_body.size()));
   code_section.insert(code_section.end(), run_body.begin(), run_body.end());
   append_wasm_section(module, section_code, code_section);

   return module;
}

/// Builds a module whose br_table targets require different operand-stack drop counts.
std::vector<uint8_t> make_aarch64_mixed_br_table_drop_wasm() {
   constexpr uint8_t section_type     = 1;
   constexpr uint8_t section_function = 3;
   constexpr uint8_t section_export   = 7;
   constexpr uint8_t section_code     = 10;
   constexpr uint8_t wasm_i32         = 0x7f;

   std::vector<uint8_t> module = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };

   std::vector<uint8_t> type_section;
   append_u32_leb(type_section, 1);
   type_section.push_back(0x60);
   append_u32_leb(type_section, 1);
   type_section.push_back(wasm_i32);
   append_u32_leb(type_section, 1);
   type_section.push_back(wasm_i32);
   append_wasm_section(module, section_type, type_section);

   std::vector<uint8_t> function_section;
   append_u32_leb(function_section, 1);
   append_u32_leb(function_section, 0);
   append_wasm_section(module, section_function, function_section);

   std::vector<uint8_t>       export_section;
   const std::vector<uint8_t> export_name = { 'p', 'i', 'c', 'k' };
   append_u32_leb(export_section, 1);
   append_u32_leb(export_section, static_cast<uint32_t>(export_name.size()));
   export_section.insert(export_section.end(), export_name.begin(), export_name.end());
   export_section.push_back(0);
   append_u32_leb(export_section, 0);
   append_wasm_section(module, section_export, export_section);

   std::vector<uint8_t> body;
   append_u32_leb(body, 0);
   body.push_back(0x02);
   body.push_back(wasm_i32);
   body.push_back(0x41);
   append_i32_leb(body, 100);
   body.push_back(0x02);
   body.push_back(wasm_i32);
   body.push_back(0x41);
   append_i32_leb(body, 10);
   body.push_back(0x02);
   body.push_back(wasm_i32);
   body.push_back(0x41);
   append_i32_leb(body, 1);
   body.push_back(0x20);
   append_u32_leb(body, 0);
   body.push_back(0x0e);
   append_u32_leb(body, 2);
   append_u32_leb(body, 0);
   append_u32_leb(body, 1);
   append_u32_leb(body, 2);
   body.push_back(0x0b);
   body.push_back(0x6a);
   body.push_back(0x0c);
   append_u32_leb(body, 1);
   body.push_back(0x0b);
   body.push_back(0x6a);
   body.push_back(0x0b);
   body.push_back(0x0b);

   std::vector<uint8_t> code_section;
   append_u32_leb(code_section, 1);
   append_u32_leb(code_section, static_cast<uint32_t>(body.size()));
   code_section.insert(code_section.end(), body.begin(), body.end());
   append_wasm_section(module, section_code, code_section);

   return module;
}
} // namespace

TEST_CASE("AArch64 JIT executes a zero-argument i32 function", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "run") (result i32)
    *     i32.const 42
    *     i32.const 58
    *     i32.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00,
                                 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e,
                                 0x00, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0x2a, 0x41, 0x3a, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "run");
   REQUIRE(result);
   CHECK(result->to_ui32() == 100u);
}

TEST_CASE("AArch64 JIT executes a zero-argument i64 function", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "run") (result i64)
    *     i64.const 40
    *     i64.const 2
    *     i64.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00,
                                 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e,
                                 0x00, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x42, 0x28, 0x42, 0x02, 0x7c, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "run");
   REQUIRE(result);
   CHECK(result->to_ui64() == 42u);
}

TEST_CASE("AArch64 JIT materializes dense all-ones i64 constants", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "run") (result i64)
    *     i64.const -1))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60,
                                 0x00, 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x72,
                                 0x75, 0x6e, 0x00, 0x00, 0x0a, 0x06, 0x01, 0x04, 0x00, 0x42, 0x7f, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "run");
   REQUIRE(result);
   CHECK(result->to_ui64() == UINT64_MAX);
}

TEST_CASE("AArch64 JIT executes an i32 function with parameters", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "sum") (param i32 i32) (result i32)
    *     local.get 0
    *     local.get 1
    *     i32.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f,
                                 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x73, 0x75, 0x6d,
                                 0x00, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "sum", UINT32_C(41), UINT32_C(1));
   REQUIRE(result);
   CHECK(result->to_ui32() == 42u);
}

TEST_CASE("AArch64 JIT executes an i64 function with locals", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "bump") (param i64) (result i64)
    *     (local i64)
    *     local.get 0
    *     i64.const 2
    *     i64.mul
    *     local.set 1
    *     local.get 1
    *     i64.const 5
    *     i64.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01,
                                 0x7e, 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x07, 0x08, 0x01, 0x04, 0x62, 0x75,
                                 0x6d, 0x70, 0x00, 0x00, 0x0a, 0x12, 0x01, 0x10, 0x01, 0x01, 0x7e, 0x20, 0x00,
                                 0x42, 0x02, 0x7e, 0x21, 0x01, 0x20, 0x01, 0x42, 0x05, 0x7c, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "bump", UINT64_C(10));
   REQUIRE(result);
   CHECK(result->to_ui64() == 25u);
}

TEST_CASE("AArch64 JIT executes high local slots in large frames", "[jit][aarch64]") {
   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(make_aarch64_large_local_frame_wasm(), &wa);

   auto result = bkend.call_with_return("env", "read");
   REQUIRE(result);
   CHECK(result->to_ui64() == 42u);
}

TEST_CASE("AArch64 JIT executes i32 memory load and store", "[jit][aarch64]") {
   /*
    * (module
    *   (memory 1)
    *   (func (export "roundtrip") (param i32 i32) (result i32)
    *     local.get 0
    *     local.get 1
    *     i32.store offset=4
    *     local.get 0
    *     i32.load offset=4))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
                                 0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x05, 0x03, 0x01,
                                 0x00, 0x01, 0x07, 0x0d, 0x01, 0x09, 0x72, 0x6f, 0x75, 0x6e, 0x64, 0x74,
                                 0x72, 0x69, 0x70, 0x00, 0x00, 0x0a, 0x10, 0x01, 0x0e, 0x00, 0x20, 0x00,
                                 0x20, 0x01, 0x36, 0x02, 0x04, 0x20, 0x00, 0x28, 0x02, 0x04, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "roundtrip", UINT32_C(16), UINT32_C(0x12345678));
   REQUIRE(result);
   CHECK(result->to_ui32() == UINT32_C(0x12345678));
}

TEST_CASE("AArch64 JIT preserves store values with large static memory offsets", "[jit][aarch64]") {
   /*
    * (module
    *   (memory 1)
    *   (func (export "word_roundtrip") (param i32) (result i32)
    *     i32.const 0
    *     local.get 0
    *     i32.store offset=8192
    *     i32.const 0
    *     i32.load offset=8192)
    *   (func (export "byte_roundtrip") (param i32) (result i32)
    *     i32.const 0
    *     local.get 0
    *     i32.store8 offset=8192
    *     i32.const 0
    *     i32.load8_u offset=8192))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01,
                                 0x7f, 0x01, 0x7f, 0x03, 0x03, 0x02, 0x00, 0x00, 0x05, 0x03, 0x01, 0x00, 0x01,
                                 0x07, 0x23, 0x02, 0x0e, 0x77, 0x6f, 0x72, 0x64, 0x5f, 0x72, 0x6f, 0x75, 0x6e,
                                 0x64, 0x74, 0x72, 0x69, 0x70, 0x00, 0x00, 0x0e, 0x62, 0x79, 0x74, 0x65, 0x5f,
                                 0x72, 0x6f, 0x75, 0x6e, 0x64, 0x74, 0x72, 0x69, 0x70, 0x00, 0x01, 0x0a, 0x23,
                                 0x02, 0x10, 0x00, 0x41, 0x00, 0x20, 0x00, 0x36, 0x02, 0x80, 0x40, 0x41, 0x00,
                                 0x28, 0x02, 0x80, 0x40, 0x0b, 0x10, 0x00, 0x41, 0x00, 0x20, 0x00, 0x3a, 0x00,
                                 0x80, 0x40, 0x41, 0x00, 0x2d, 0x00, 0x80, 0x40, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto word_result = bkend.call_with_return("env", "word_roundtrip", UINT32_C(0x12345678));
   REQUIRE(word_result);
   CHECK(word_result->to_ui32() == UINT32_C(0x12345678));

   auto byte_result = bkend.call_with_return("env", "byte_roundtrip", UINT32_C(0x000000a5));
   REQUIRE(byte_result);
   CHECK(byte_result->to_ui32() == UINT32_C(0xa5));
}

TEST_CASE("AArch64 JIT executes i64 memory load and store", "[jit][aarch64]") {
   /*
    * (module
    *   (memory 1)
    *   (func (export "roundtrip") (param i32 i64) (result i64)
    *     local.get 0
    *     local.get 1
    *     i64.store offset=8
    *     local.get 0
    *     i64.load offset=8))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
                                 0x02, 0x7f, 0x7e, 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x05, 0x03, 0x01,
                                 0x00, 0x01, 0x07, 0x0d, 0x01, 0x09, 0x72, 0x6f, 0x75, 0x6e, 0x64, 0x74,
                                 0x72, 0x69, 0x70, 0x00, 0x00, 0x0a, 0x10, 0x01, 0x0e, 0x00, 0x20, 0x00,
                                 0x20, 0x01, 0x37, 0x03, 0x08, 0x20, 0x00, 0x29, 0x03, 0x08, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "roundtrip", UINT32_C(24), UINT64_C(0x123456789abcdef0));
   REQUIRE(result);
   CHECK(result->to_ui64() == UINT64_C(0x123456789abcdef0));
}

TEST_CASE("AArch64 JIT executes a forward direct call", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "run") (param i32) (result i32)
    *     local.get 0
    *     call 1
    *     i32.const 2
    *     i32.mul)
    *   (func (param i32) (result i32)
    *     local.get 0
    *     i32.const 1
    *     i32.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01,
                                 0x7f, 0x01, 0x7f, 0x03, 0x03, 0x02, 0x00, 0x00, 0x07, 0x07, 0x01, 0x03, 0x72,
                                 0x75, 0x6e, 0x00, 0x00, 0x0a, 0x13, 0x02, 0x09, 0x00, 0x20, 0x00, 0x10, 0x01,
                                 0x41, 0x02, 0x6c, 0x0b, 0x07, 0x00, 0x20, 0x00, 0x41, 0x01, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "run", UINT32_C(20));
   REQUIRE(result);
   CHECK(result->to_ui32() == 42u);
}

TEST_CASE("AArch64 JIT executes wide internal direct call arguments", "[jit][aarch64]") {
   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(make_aarch64_wide_internal_call_wasm(), &wa);

   auto result = bkend.call_with_return("env", "run");
   REQUIRE(result);
   CHECK(result->to_ui64() == 285u);
}

TEST_CASE("AArch64 JIT executes if else with i32 comparison", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "max") (param i32 i32) (result i32)
    *     local.get 0
    *     local.get 1
    *     i32.gt_s
    *     if (result i32)
    *       local.get 0
    *     else
    *       local.get 1
    *     end))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02,
                                 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x6d,
                                 0x61, 0x78, 0x00, 0x00, 0x0a, 0x11, 0x01, 0x0f, 0x00, 0x20, 0x00, 0x20, 0x01,
                                 0x4a, 0x04, 0x7f, 0x20, 0x00, 0x05, 0x20, 0x01, 0x0b, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto left = bkend.call_with_return("env", "max", INT32_C(99), INT32_C(-5));
   REQUIRE(left);
   CHECK(left->to_i32() == 99);

   auto right = bkend.call_with_return("env", "max", INT32_C(-5), INT32_C(99));
   REQUIRE(right);
   CHECK(right->to_i32() == 99);
}

TEST_CASE("AArch64 JIT executes a counted loop", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "sum_to") (param i32) (result i32)
    *     (local i32 i32)
    *     i32.const 0
    *     local.set 1
    *     i32.const 0
    *     local.set 2
    *     block
    *       loop
    *         local.get 1
    *         local.get 0
    *         i32.ge_u
    *         br_if 1
    *         local.get 2
    *         local.get 1
    *         i32.add
    *         local.set 2
    *         local.get 1
    *         i32.const 1
    *         i32.add
    *         local.set 1
    *         br 0
    *       end
    *     end
    *     local.get 2))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7f,
                                 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0a, 0x01, 0x06, 0x73, 0x75, 0x6d, 0x5f,
                                 0x74, 0x6f, 0x00, 0x00, 0x0a, 0x2d, 0x01, 0x2b, 0x01, 0x02, 0x7f, 0x41, 0x00, 0x21,
                                 0x01, 0x41, 0x00, 0x21, 0x02, 0x02, 0x40, 0x03, 0x40, 0x20, 0x01, 0x20, 0x00, 0x4f,
                                 0x0d, 0x01, 0x20, 0x02, 0x20, 0x01, 0x6a, 0x21, 0x02, 0x20, 0x01, 0x41, 0x01, 0x6a,
                                 0x21, 0x01, 0x0c, 0x00, 0x0b, 0x0b, 0x20, 0x02, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "sum_to", UINT32_C(10));
   REQUIRE(result);
   CHECK(result->to_ui32() == 45u);
}

TEST_CASE("AArch64 JIT executes drop and select", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "choose") (param i32) (result i32)
    *     i32.const 7
    *     drop
    *     i32.const 11
    *     i32.const 22
    *     local.get 0
    *     select))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60,
                                 0x01, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0a, 0x01, 0x06,
                                 0x63, 0x68, 0x6f, 0x6f, 0x73, 0x65, 0x00, 0x00, 0x0a, 0x0e, 0x01, 0x0c,
                                 0x00, 0x41, 0x07, 0x1a, 0x41, 0x0b, 0x41, 0x16, 0x20, 0x00, 0x1b, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto true_result = bkend.call_with_return("env", "choose", UINT32_C(1));
   REQUIRE(true_result);
   CHECK(true_result->to_ui32() == 11u);

   auto false_result = bkend.call_with_return("env", "choose", UINT32_C(0));
   REQUIRE(false_result);
   CHECK(false_result->to_ui32() == 22u);
}

TEST_CASE("AArch64 JIT executes i32 shifts, rotates, and bit counts", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "i32_ops") (param i32) (result i32)
    *     local.get 0
    *     i32.clz
    *     local.get 0
    *     i32.ctz
    *     i32.add
    *     local.get 0
    *     i32.popcnt
    *     i32.add
    *     local.get 0
    *     i32.const 5
    *     i32.shl
    *     i32.const 2
    *     i32.shr_u
    *     i32.add
    *     local.get 0
    *     i32.const 4
    *     i32.rotl
    *     i32.add
    *     local.get 0
    *     i32.const 4
    *     i32.rotr
    *     i32.add
    *     i32.const -16
    *     i32.const 2
    *     i32.shr_s
    *     i32.const 28
    *     i32.shr_u
    *     i32.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7f,
                                 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0b, 0x01, 0x07, 0x69, 0x33, 0x32, 0x5f,
                                 0x6f, 0x70, 0x73, 0x00, 0x00, 0x0a, 0x2d, 0x01, 0x2b, 0x00, 0x20, 0x00, 0x67, 0x20,
                                 0x00, 0x68, 0x6a, 0x20, 0x00, 0x69, 0x6a, 0x20, 0x00, 0x41, 0x05, 0x74, 0x41, 0x02,
                                 0x76, 0x6a, 0x20, 0x00, 0x41, 0x04, 0x77, 0x6a, 0x20, 0x00, 0x41, 0x04, 0x78, 0x6a,
                                 0x41, 0x70, 0x41, 0x02, 0x75, 0x41, 0x1c, 0x76, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "i32_ops", UINT32_C(0x10));
   REQUIRE(result);
   CHECK(result->to_ui32() == 432u);
}

TEST_CASE("AArch64 JIT executes i64 shifts, rotates, and bit counts", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "i64_ops") (param i64) (result i64)
    *     local.get 0
    *     i64.clz
    *     local.get 0
    *     i64.ctz
    *     i64.add
    *     local.get 0
    *     i64.popcnt
    *     i64.add
    *     local.get 0
    *     i64.const 8
    *     i64.shl
    *     i64.const 4
    *     i64.shr_u
    *     i64.add
    *     local.get 0
    *     i64.const 4
    *     i64.rotl
    *     i64.add
    *     local.get 0
    *     i64.const 4
    *     i64.rotr
    *     i64.add
    *     i64.const -16
    *     i64.const 2
    *     i64.shr_s
    *     i64.const 60
    *     i64.shr_u
    *     i64.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7e,
                                 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0b, 0x01, 0x07, 0x69, 0x36, 0x34, 0x5f,
                                 0x6f, 0x70, 0x73, 0x00, 0x00, 0x0a, 0x2d, 0x01, 0x2b, 0x00, 0x20, 0x00, 0x79, 0x20,
                                 0x00, 0x7a, 0x7c, 0x20, 0x00, 0x7b, 0x7c, 0x20, 0x00, 0x42, 0x08, 0x86, 0x42, 0x04,
                                 0x88, 0x7c, 0x20, 0x00, 0x42, 0x04, 0x89, 0x7c, 0x20, 0x00, 0x42, 0x04, 0x8a, 0x7c,
                                 0x42, 0x70, 0x42, 0x02, 0x87, 0x42, 0x3c, 0x88, 0x7c, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "i64_ops", UINT64_C(0x10));
   REQUIRE(result);
   CHECK(result->to_ui64() == 592u);
}

TEST_CASE("AArch64 JIT executes i32 division and remainder", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "i32_divrem") (param i32 i32) (result i32)
    *     local.get 0
    *     local.get 1
    *     i32.div_s
    *     local.get 0
    *     local.get 1
    *     i32.rem_s
    *     i32.add
    *     local.get 0
    *     local.get 1
    *     i32.div_u
    *     i32.add
    *     local.get 0
    *     local.get 1
    *     i32.rem_u
    *     i32.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f,
                                 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0e, 0x01, 0x0a, 0x69, 0x33, 0x32,
                                 0x5f, 0x64, 0x69, 0x76, 0x72, 0x65, 0x6d, 0x00, 0x00, 0x0a, 0x1b, 0x01, 0x19, 0x00,
                                 0x20, 0x00, 0x20, 0x01, 0x6d, 0x20, 0x00, 0x20, 0x01, 0x6f, 0x6a, 0x20, 0x00, 0x20,
                                 0x01, 0x6e, 0x6a, 0x20, 0x00, 0x20, 0x01, 0x70, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "i32_divrem", UINT32_C(13), UINT32_C(5));
   REQUIRE(result);
   CHECK(result->to_ui32() == 10u);
}

TEST_CASE("AArch64 JIT executes i64 division and remainder", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "i64_divrem") (param i64 i64) (result i64)
    *     local.get 0
    *     local.get 1
    *     i64.div_s
    *     local.get 0
    *     local.get 1
    *     i64.rem_s
    *     i64.add
    *     local.get 0
    *     local.get 1
    *     i64.div_u
    *     i64.add
    *     local.get 0
    *     local.get 1
    *     i64.rem_u
    *     i64.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7e,
                                 0x7e, 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0e, 0x01, 0x0a, 0x69, 0x36, 0x34,
                                 0x5f, 0x64, 0x69, 0x76, 0x72, 0x65, 0x6d, 0x00, 0x00, 0x0a, 0x1b, 0x01, 0x19, 0x00,
                                 0x20, 0x00, 0x20, 0x01, 0x7f, 0x20, 0x00, 0x20, 0x01, 0x81, 0x7c, 0x20, 0x00, 0x20,
                                 0x01, 0x80, 0x7c, 0x20, 0x00, 0x20, 0x01, 0x82, 0x7c, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "i64_divrem", UINT64_C(13), UINT64_C(5));
   REQUIRE(result);
   CHECK(result->to_ui64() == 10u);
}

TEST_CASE("AArch64 JIT traps integer division by zero", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "trap") (param i32) (result i32)
    *     local.get 0
    *     i32.const 0
    *     i32.div_u))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7f,
                                 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x08, 0x01, 0x04, 0x74, 0x72, 0x61, 0x70,
                                 0x00, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x41, 0x00, 0x6e, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   CHECK_THROWS_AS(bkend.call_with_return("env", "trap", UINT32_C(1)), wasm_interpreter_exception);
}

TEST_CASE("AArch64 JIT traps unreachable at runtime", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "trap") (result i32)
    *     unreachable))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60,
                                 0x00, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x08, 0x01, 0x04, 0x74,
                                 0x72, 0x61, 0x70, 0x00, 0x00, 0x0a, 0x05, 0x01, 0x03, 0x00, 0x00, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   CHECK_THROWS_AS(bkend.call_with_return("env", "trap"), wasm_interpreter_exception);
}

TEST_CASE("AArch64 JIT executes integer wrap and extension operators", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "convert") (param i64) (result i64)
    *     local.get 0
    *     i32.wrap_i64
    *     i64.extend_u_i32
    *     i32.const -1
    *     i64.extend_s_i32
    *     i64.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60,
                                 0x01, 0x7e, 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0b, 0x01, 0x07,
                                 0x63, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x00, 0x00, 0x0a, 0x0c, 0x01,
                                 0x0a, 0x00, 0x20, 0x00, 0xa7, 0xad, 0x41, 0x7f, 0xac, 0x7c, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "convert", UINT64_C(0x10000002a));
   REQUIRE(result);
   CHECK(result->to_ui64() == 41u);
}

TEST_CASE("AArch64 JIT executes an imported integer host call", "[jit][aarch64]") {
   /*
    * (module
    *   (import "env" "add_one" (func $add_one (param i32 i32) (result i32)))
    *   (func (export "run") (param i32) (result i32)
    *     local.get 0
    *     i32.const 41
    *     call $add_one
    *     i32.const 2
    *     i32.mul))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x02, 0x60, 0x02, 0x7f,
                                 0x7f, 0x01, 0x7f, 0x60, 0x01, 0x7f, 0x01, 0x7f, 0x02, 0x0f, 0x01, 0x03, 0x65, 0x6e,
                                 0x76, 0x07, 0x61, 0x64, 0x64, 0x5f, 0x6f, 0x6e, 0x65, 0x00, 0x00, 0x03, 0x02, 0x01,
                                 0x01, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x01, 0x0a, 0x0d, 0x01, 0x0b,
                                 0x00, 0x20, 0x00, 0x41, 0x29, 0x10, 0x00, 0x41, 0x02, 0x6c, 0x0b };

   using rhf_t = registered_host_functions<std::nullptr_t, execution_interface,
                                           type_converter<std::nullptr_t, execution_interface>>;
   rhf_t::add<&aarch64_jit_imports::add_one>("env", "add_one");
   using backend_t = backend<rhf_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "run", UINT32_C(0));
   REQUIRE(result);
   CHECK(result->to_ui32() == 84u);
}

TEST_CASE("AArch64 JIT executes wide imported direct call arguments", "[jit][aarch64]") {
   using rhf_t = registered_host_functions<std::nullptr_t, execution_interface,
                                           type_converter<std::nullptr_t, execution_interface>>;
   rhf_t::add<&aarch64_jit_imports::sum_nine>("env", "sum9");
   using backend_t = backend<rhf_t, jit>;
   backend_t bkend(make_aarch64_wide_imported_call_wasm(), &wa);

   auto result = bkend.call_with_return("env", "run");
   REQUIRE(result);
   CHECK(result->to_ui32() == 285u);
}

TEST_CASE("AArch64 JIT executes an indirect integer call", "[jit][aarch64]") {
   /*
    * (module
    *   (type $inc_t (func (param i32) (result i32)))
    *   (type $run_t (func (param i32 i32) (result i32)))
    *   (func $run (type $run_t)
    *     local.get 0
    *     local.get 1
    *     call_indirect (type $inc_t))
    *   (func $inc (type $inc_t)
    *     local.get 0
    *     i32.const 1
    *     i32.add)
    *   (table 1 funcref)
    *   (elem (i32.const 0) $inc)
    *   (export "run" (func $run)))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x02, 0x60,
                                 0x01, 0x7f, 0x01, 0x7f, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x03,
                                 0x02, 0x01, 0x00, 0x04, 0x04, 0x01, 0x70, 0x00, 0x01, 0x07, 0x07, 0x01,
                                 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00, 0x09, 0x07, 0x01, 0x00, 0x41, 0x00,
                                 0x0b, 0x01, 0x01, 0x0a, 0x13, 0x02, 0x09, 0x00, 0x20, 0x00, 0x20, 0x01,
                                 0x11, 0x00, 0x00, 0x0b, 0x07, 0x00, 0x20, 0x00, 0x41, 0x01, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "run", UINT32_C(41), UINT32_C(0));
   REQUIRE(result);
   CHECK(result->to_ui32() == 42u);
}

TEST_CASE("AArch64 JIT executes br_table control flow", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "pick") (param i32) (result i32)
    *     (local i32)
    *     block
    *       block
    *         block
    *           local.get 0
    *           br_table 0 1 2
    *         end
    *         i32.const 10
    *         local.set 1
    *         br 1
    *       end
    *       i32.const 20
    *       local.set 1
    *     end
    *     local.get 1))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7f,
                                 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x08, 0x01, 0x04, 0x70, 0x69, 0x63, 0x6b,
                                 0x00, 0x00, 0x0a, 0x22, 0x01, 0x20, 0x01, 0x01, 0x7f, 0x02, 0x40, 0x02, 0x40, 0x02,
                                 0x40, 0x20, 0x00, 0x0e, 0x02, 0x00, 0x01, 0x02, 0x0b, 0x41, 0x0a, 0x21, 0x01, 0x0c,
                                 0x01, 0x0b, 0x41, 0x14, 0x21, 0x01, 0x0b, 0x20, 0x01, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto first = bkend.call_with_return("env", "pick", UINT32_C(0));
   REQUIRE(first);
   CHECK(first->to_ui32() == 10u);

   auto second = bkend.call_with_return("env", "pick", UINT32_C(1));
   REQUIRE(second);
   CHECK(second->to_ui32() == 20u);

   auto fallback = bkend.call_with_return("env", "pick", UINT32_C(9));
   REQUIRE(fallback);
   CHECK(fallback->to_ui32() == 0u);
}

TEST_CASE("AArch64 JIT executes br_table cases with mixed stack drops", "[jit][aarch64]") {
   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(make_aarch64_mixed_br_table_drop_wasm(), &wa);

   auto first = bkend.call_with_return("env", "pick", UINT32_C(0));
   REQUIRE(first);
   CHECK(first->to_ui32() == 11u);

   auto second = bkend.call_with_return("env", "pick", UINT32_C(1));
   REQUIRE(second);
   CHECK(second->to_ui32() == 101u);

   auto fallback = bkend.call_with_return("env", "pick", UINT32_C(9));
   REQUIRE(fallback);
   CHECK(fallback->to_ui32() == 1u);
}

TEST_CASE("AArch64 JIT executes mutable integer globals", "[jit][aarch64]") {
   /*
    * (module
    *   (global (mut i32) (i32.const 5))
    *   (func (export "bump") (param i32) (result i32)
    *     global.get 0
    *     local.get 0
    *     i32.add
    *     global.set 0
    *     global.get 0))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7f,
                                 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x06, 0x06, 0x01, 0x7f, 0x01, 0x41, 0x05, 0x0b,
                                 0x07, 0x08, 0x01, 0x04, 0x62, 0x75, 0x6d, 0x70, 0x00, 0x00, 0x0a, 0x0d, 0x01, 0x0b,
                                 0x00, 0x23, 0x00, 0x20, 0x00, 0x6a, 0x24, 0x00, 0x23, 0x00, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto first = bkend.call_with_return("env", "bump", UINT32_C(37));
   REQUIRE(first);
   CHECK(first->to_ui32() == 42u);

   auto second = bkend.call_with_return("env", "bump", UINT32_C(1));
   REQUIRE(second);
   CHECK(second->to_ui32() == 43u);
}

TEST_CASE("AArch64 JIT executes memory size and grow", "[jit][aarch64]") {
   /*
    * (module
    *   (memory 1 2)
    *   (func (export "grow") (param i32) (result i32)
    *     memory.size
    *     local.get 0
    *     memory.grow
    *     i32.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01,
                                 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x05, 0x04, 0x01, 0x01, 0x01, 0x02,
                                 0x07, 0x08, 0x01, 0x04, 0x67, 0x72, 0x6f, 0x77, 0x00, 0x00, 0x0a, 0x0b, 0x01,
                                 0x09, 0x00, 0x3f, 0x00, 0x20, 0x00, 0x40, 0x00, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "grow", UINT32_C(1));
   REQUIRE(result);
   CHECK(result->to_ui32() == 2u);
}

TEST_CASE("AArch64 JIT executes narrow i32 memory load and store operators", "[jit][aarch64]") {
   /*
    * (module
    *   (memory 1)
    *   (func (export "narrow32") (param i32 i32) (result i32)
    *     local.get 0
    *     local.get 1
    *     i32.store8 offset=3
    *     local.get 0
    *     i32.load8_s offset=3
    *     local.get 0
    *     local.get 1
    *     i32.store16 offset=8
    *     local.get 0
    *     i32.load16_u offset=8
    *     i32.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
                                 0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x05, 0x03, 0x01,
                                 0x00, 0x01, 0x07, 0x0c, 0x01, 0x08, 0x6e, 0x61, 0x72, 0x72, 0x6f, 0x77,
                                 0x33, 0x32, 0x00, 0x00, 0x0a, 0x1d, 0x01, 0x1b, 0x00, 0x20, 0x00, 0x20,
                                 0x01, 0x3a, 0x00, 0x03, 0x20, 0x00, 0x2c, 0x00, 0x03, 0x20, 0x00, 0x20,
                                 0x01, 0x3b, 0x01, 0x08, 0x20, 0x00, 0x2f, 0x01, 0x08, 0x6a, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "narrow32", UINT32_C(16), UINT32_C(0xffff));
   REQUIRE(result);
   CHECK(result->to_ui32() == 65534u);
}

TEST_CASE("AArch64 JIT executes narrow i64 memory load and store operators", "[jit][aarch64]") {
   /*
    * (module
    *   (memory 1)
    *   (func (export "narrow64") (param i32 i64) (result i64)
    *     local.get 0
    *     local.get 1
    *     i64.store32 offset=4
    *     local.get 0
    *     i64.load32_s offset=4
    *     local.get 0
    *     local.get 1
    *     i64.store16 offset=16
    *     local.get 0
    *     i64.load16_u offset=16
    *     i64.add
    *     local.get 0
    *     local.get 1
    *     i64.store8 offset=24
    *     local.get 0
    *     i64.load8_u offset=24
    *     i64.add))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f,
                                 0x7e, 0x01, 0x7e, 0x03, 0x02, 0x01, 0x00, 0x05, 0x03, 0x01, 0x00, 0x01, 0x07, 0x0c,
                                 0x01, 0x08, 0x6e, 0x61, 0x72, 0x72, 0x6f, 0x77, 0x36, 0x34, 0x00, 0x00, 0x0a, 0x2a,
                                 0x01, 0x28, 0x00, 0x20, 0x00, 0x20, 0x01, 0x3e, 0x02, 0x04, 0x20, 0x00, 0x34, 0x02,
                                 0x04, 0x20, 0x00, 0x20, 0x01, 0x3d, 0x01, 0x10, 0x20, 0x00, 0x33, 0x01, 0x10, 0x7c,
                                 0x20, 0x00, 0x20, 0x01, 0x3c, 0x00, 0x18, 0x20, 0x00, 0x31, 0x00, 0x18, 0x7c, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   auto result = bkend.call_with_return("env", "narrow64", UINT32_C(16), UINT64_C(0xffffffffffffffff));
   REQUIRE(result);
   CHECK(result->to_ui64() == 65789u);
}

TEST_CASE("AArch64 JIT executes softfloat arithmetic and comparisons", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "a") (param f32) (result f32)
    *     local.get 0
    *     f32.const 1.5
    *     f32.add)
    *   (func (export "b") (param f64) (result f64)
    *     local.get 0
    *     f64.const 2.0
    *     f64.mul
    *     f64.const 1.0
    *     f64.sub)
    *   (func (export "c") (param f32 f32) (result i32)
    *     local.get 0
    *     local.get 1
    *     f32.gt))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x11, 0x03, 0x60, 0x01, 0x7d,
                                 0x01, 0x7d, 0x60, 0x01, 0x7c, 0x01, 0x7c, 0x60, 0x02, 0x7d, 0x7d, 0x01, 0x7f, 0x03,
                                 0x04, 0x03, 0x00, 0x01, 0x02, 0x07, 0x0d, 0x03, 0x01, 0x61, 0x00, 0x00, 0x01, 0x62,
                                 0x00, 0x01, 0x01, 0x63, 0x00, 0x02, 0x0a, 0x2d, 0x03, 0x0a, 0x00, 0x20, 0x00, 0x43,
                                 0x00, 0x00, 0xc0, 0x3f, 0x92, 0x0b, 0x18, 0x00, 0x20, 0x00, 0x44, 0x00, 0x00, 0x00,
                                 0x00, 0x00, 0x00, 0x00, 0x40, 0xa2, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0,
                                 0x3f, 0xa1, 0x0b, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x5e, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   wasm_allocator                       interpreter_allocator;
   auto                                 interpreter_code = code;
   backend<std::nullptr_t, interpreter> interpreter_backend(interpreter_code, &interpreter_allocator);

   auto f32_result = bkend.call_with_return("env", "a", bit_cast<float>(UINT32_C(0x3f000000)));
   REQUIRE(f32_result);
   CHECK(bit_cast<uint32_t>(f32_result->to_f32()) == UINT32_C(0x40000000));

   auto interpreter_f32_result =
         interpreter_backend.call_with_return("env", "a", bit_cast<float>(UINT32_C(0x3f000000)));
   REQUIRE(interpreter_f32_result);
   CHECK(bit_cast<uint32_t>(f32_result->to_f32()) == bit_cast<uint32_t>(interpreter_f32_result->to_f32()));

   auto f64_result = bkend.call_with_return("env", "b", bit_cast<double>(UINT64_C(0x4008000000000000)));
   REQUIRE(f64_result);
   CHECK(bit_cast<uint64_t>(f64_result->to_f64()) == UINT64_C(0x4014000000000000));

   auto interpreter_f64_result =
         interpreter_backend.call_with_return("env", "b", bit_cast<double>(UINT64_C(0x4008000000000000)));
   REQUIRE(interpreter_f64_result);
   CHECK(bit_cast<uint64_t>(f64_result->to_f64()) == bit_cast<uint64_t>(interpreter_f64_result->to_f64()));

   auto cmp_result = bkend.call_with_return("env", "c", bit_cast<float>(UINT32_C(0x40800000)),
                                            bit_cast<float>(UINT32_C(0x40000000)));
   REQUIRE(cmp_result);
   CHECK(cmp_result->to_ui32() == 1u);

   auto interpreter_cmp_result = interpreter_backend.call_with_return("env", "c", bit_cast<float>(UINT32_C(0x40800000)),
                                                                      bit_cast<float>(UINT32_C(0x40000000)));
   REQUIRE(interpreter_cmp_result);
   CHECK(cmp_result->to_ui32() == interpreter_cmp_result->to_ui32());
}

TEST_CASE("AArch64 JIT executes float memory and global raw-bit round trips", "[jit][aarch64]") {
   /*
    * (module
    *   (memory 1)
    *   (func (export "f") (param i32 f32) (result f32)
    *     local.get 0
    *     local.get 1
    *     f32.store
    *     local.get 0
    *     f32.load)
    *   (func (export "d") (param i32 f64) (result f64)
    *     local.get 0
    *     local.get 1
    *     f64.store
    *     local.get 0
    *     f64.load))
    */
   std::vector<uint8_t> memory_code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0d, 0x02, 0x60, 0x02,
                                        0x7f, 0x7d, 0x01, 0x7d, 0x60, 0x02, 0x7f, 0x7c, 0x01, 0x7c, 0x03, 0x03, 0x02,
                                        0x00, 0x01, 0x05, 0x03, 0x01, 0x00, 0x01, 0x07, 0x09, 0x02, 0x01, 0x66, 0x00,
                                        0x00, 0x01, 0x64, 0x00, 0x01, 0x0a, 0x1f, 0x02, 0x0e, 0x00, 0x20, 0x00, 0x20,
                                        0x01, 0x38, 0x02, 0x00, 0x20, 0x00, 0x2a, 0x02, 0x00, 0x0b, 0x0e, 0x00, 0x20,
                                        0x00, 0x20, 0x01, 0x39, 0x03, 0x00, 0x20, 0x00, 0x2b, 0x03, 0x00, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t memory_backend(memory_code, &wa);

   auto f32_result = memory_backend.call_with_return("env", "f", UINT32_C(32), bit_cast<float>(UINT32_C(0xc0200000)));
   REQUIRE(f32_result);
   CHECK(bit_cast<uint32_t>(f32_result->to_f32()) == UINT32_C(0xc0200000));

   auto f64_result =
         memory_backend.call_with_return("env", "d", UINT32_C(64), bit_cast<double>(UINT64_C(0x400921fb54442d18)));
   REQUIRE(f64_result);
   CHECK(bit_cast<uint64_t>(f64_result->to_f64()) == UINT64_C(0x400921fb54442d18));

   /*
    * (module
    *   (global (mut f32) (f32.const 1.0))
    *   (global (mut f64) (f64.const 2.0))
    *   (func (export "f") (param f32) (result f32)
    *     local.get 0
    *     global.set 0
    *     global.get 0)
    *   (func (export "d") (param f64) (result f64)
    *     local.get 0
    *     global.set 1
    *     global.get 1))
    */
   std::vector<uint8_t> global_code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0b, 0x02, 0x60,
                                        0x01, 0x7d, 0x01, 0x7d, 0x60, 0x01, 0x7c, 0x01, 0x7c, 0x03, 0x03, 0x02,
                                        0x00, 0x01, 0x06, 0x15, 0x02, 0x7d, 0x01, 0x43, 0x00, 0x00, 0x80, 0x3f,
                                        0x0b, 0x7c, 0x01, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
                                        0x0b, 0x07, 0x09, 0x02, 0x01, 0x66, 0x00, 0x00, 0x01, 0x64, 0x00, 0x01,
                                        0x0a, 0x13, 0x02, 0x08, 0x00, 0x20, 0x00, 0x24, 0x00, 0x23, 0x00, 0x0b,
                                        0x08, 0x00, 0x20, 0x00, 0x24, 0x01, 0x23, 0x01, 0x0b };
   backend_t            global_backend(global_code, &wa);

   auto global_f32 = global_backend.call_with_return("env", "f", bit_cast<float>(UINT32_C(0x7fc00001)));
   REQUIRE(global_f32);
   CHECK(bit_cast<uint32_t>(global_f32->to_f32()) == UINT32_C(0x7fc00001));

   auto global_f64 = global_backend.call_with_return("env", "d", bit_cast<double>(UINT64_C(0x7ff8000000000001)));
   REQUIRE(global_f64);
   CHECK(bit_cast<uint64_t>(global_f64->to_f64()) == UINT64_C(0x7ff8000000000001));
}

TEST_CASE("AArch64 JIT executes softfloat conversions and traps NaN truncation", "[jit][aarch64]") {
   /*
    * (module
    *   (func (export "c") (param i32 i64 f32 f64) (result i64)
    *     local.get 0
    *     f64.convert_s_i32
    *     local.get 1
    *     f64.convert_u_i64
    *     f64.add
    *     local.get 2
    *     f64.promote_f32
    *     f64.add
    *     local.get 3
    *     f64.add
    *     i64.trunc_s_f64)
    *   (func (export "t") (result i32)
    *     f32.const nan
    *     i32.trunc_s_f32))
    */
   std::vector<uint8_t> code = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0d, 0x02, 0x60, 0x04, 0x7f,
                                 0x7e, 0x7d, 0x7c, 0x01, 0x7e, 0x60, 0x00, 0x01, 0x7f, 0x03, 0x03, 0x02, 0x00, 0x01,
                                 0x07, 0x09, 0x02, 0x01, 0x63, 0x00, 0x00, 0x01, 0x74, 0x00, 0x01, 0x0a, 0x1c, 0x02,
                                 0x11, 0x00, 0x20, 0x00, 0xb7, 0x20, 0x01, 0xba, 0xa0, 0x20, 0x02, 0xbb, 0xa0, 0x20,
                                 0x03, 0xa0, 0xb0, 0x0b, 0x08, 0x00, 0x43, 0x00, 0x00, 0xc0, 0x7f, 0xa8, 0x0b };

   using backend_t = backend<std::nullptr_t, jit>;
   backend_t bkend(code, &wa);

   wasm_allocator                       interpreter_allocator;
   auto                                 interpreter_code = code;
   backend<std::nullptr_t, interpreter> interpreter_backend(interpreter_code, &interpreter_allocator);

   auto result = bkend.call_with_return("env", "c", UINT32_C(2), UINT64_C(3), bit_cast<float>(UINT32_C(0x3fc00000)),
                                        bit_cast<double>(UINT64_C(0x4012000000000000)));
   REQUIRE(result);
   CHECK(result->to_ui64() == 11u);

   auto interpreter_result = interpreter_backend.call_with_return("env", "c", UINT32_C(2), UINT64_C(3),
                                                                  bit_cast<float>(UINT32_C(0x3fc00000)),
                                                                  bit_cast<double>(UINT64_C(0x4012000000000000)));
   REQUIRE(interpreter_result);
   CHECK(result->to_ui64() == interpreter_result->to_ui64());

   CHECK_THROWS_AS(bkend.call_with_return("env", "t"), wasm_interpreter_exception);
   CHECK_THROWS_AS(interpreter_backend.call_with_return("env", "t"), wasm_interpreter_exception);
}
#endif
