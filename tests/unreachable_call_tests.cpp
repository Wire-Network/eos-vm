#include <sysio/vm/backend.hpp>

#include "utils.hpp"
#include <catch2/catch.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

using namespace sysio::vm;

namespace {

/// Leading bytes of every module: the magic number and version 1.
const std::vector<uint8_t> module_header = { 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };

/// Section identifiers used by the test modules.
constexpr uint8_t type_section_id     = 1;
constexpr uint8_t function_section_id = 3;
constexpr uint8_t table_section_id    = 4;
constexpr uint8_t code_section_id     = 10;

/// Leading byte of a function type.
constexpr uint8_t func_type_form = 0x60;
/// Limits flag of a table without a maximum size.
constexpr uint8_t limits_without_maximum = 0x00;
/// Initial size of the table declared for call_indirect.
constexpr uint8_t table_initial_size = 1;
/// Reserved table index byte that ends call_indirect.
constexpr uint8_t call_indirect_reserved = 0x00;

/// A function type: parameter types and an optional result type.
struct signature {
   std::vector<uint8_t>   params;
   std::optional<uint8_t> result;
};

/// Appends `value` to `out` as unsigned LEB128.
void append_varuint32(std::vector<uint8_t>& out, uint32_t value) {
   constexpr uint32_t payload_bits = 7;
   constexpr uint8_t  payload_mask = 0x7f;
   constexpr uint8_t  continuation = 0x80;
   do {
      uint8_t byte = value & payload_mask;
      value >>= payload_bits;
      if (value != 0)
         byte |= continuation;
      out.push_back(byte);
   } while (value != 0);
}

/// Appends a section with identifier `id` holding `contents` to `out`.
void append_section(std::vector<uint8_t>& out, uint8_t id, const std::vector<uint8_t>& contents) {
   out.push_back(id);
   append_varuint32(out, contents.size());
   out.insert(out.end(), contents.begin(), contents.end());
}

/// Builds a module whose function `i` has type `signatures[function_types[i]]` and instructions `bodies[i]`, with no
/// locals. With `with_table`, the module also declares a funcref table for call_indirect.
std::vector<uint8_t> make_module(const std::vector<signature>& signatures, const std::vector<uint32_t>& function_types,
                                 const std::vector<std::vector<uint8_t>>& bodies, bool with_table) {
   std::vector<uint8_t> wasm = module_header;

   std::vector<uint8_t> type_section;
   append_varuint32(type_section, signatures.size());
   for (const signature& sig : signatures) {
      type_section.push_back(func_type_form);
      append_varuint32(type_section, sig.params.size());
      type_section.insert(type_section.end(), sig.params.begin(), sig.params.end());
      append_varuint32(type_section, sig.result ? 1 : 0);
      if (sig.result)
         type_section.push_back(*sig.result);
   }
   append_section(wasm, type_section_id, type_section);

   std::vector<uint8_t> function_section;
   append_varuint32(function_section, function_types.size());
   for (uint32_t type_index : function_types)
      append_varuint32(function_section, type_index);
   append_section(wasm, function_section_id, function_section);

   if (with_table) {
      std::vector<uint8_t> table_section;
      append_varuint32(table_section, 1);
      table_section.push_back(types::anyfunc);
      table_section.push_back(limits_without_maximum);
      append_varuint32(table_section, table_initial_size);
      append_section(wasm, table_section_id, table_section);
   }

   std::vector<uint8_t> code_section;
   append_varuint32(code_section, bodies.size());
   for (const std::vector<uint8_t>& instructions : bodies) {
      std::vector<uint8_t> body;
      append_varuint32(body, 0); // no local entries
      body.insert(body.end(), instructions.begin(), instructions.end());
      append_varuint32(code_section, body.size());
      code_section.insert(code_section.end(), body.begin(), body.end());
   }
   append_section(wasm, code_section_id, code_section);

   return wasm;
}

/// Parses `wasm` with backend `Impl`; true when the module is accepted, false when parsing rejects it.
template <typename Impl>
bool validates(std::vector<uint8_t> wasm) {
   try {
      backend<std::nullptr_t, Impl> bkend(wasm, get_wasm_allocator());
      return true;
   } catch (const sysio::vm::exception&) {
      return false;
   }
}

/// Instructions of every callee: valid for any signature.
const std::vector<uint8_t> callee_body = { opcodes::unreachable, opcodes::end };

/// Type indices of the parity module.
constexpr uint32_t void_type       = 0; ///< () -> ()
constexpr uint32_t i32_result_type = 1; ///< () -> i32
constexpr uint32_t pair_type       = 2; ///< (i32, i64) -> i32
constexpr uint32_t wide_type       = 3; ///< (i32 x wide_param_count) -> ()
constexpr uint32_t pair_i64_type   = 4; ///< (i32, i64) -> i64

/// Function indices of the parity module; the function under test follows the callees.
constexpr uint8_t pair_callee     = 0;
constexpr uint8_t wide_callee     = 1;
constexpr uint8_t pair_i64_callee = 2;
/// A function index past the end of the parity module.
constexpr uint8_t missing_function = 9;
/// A type index past the end of the parity module.
constexpr uint8_t missing_type = 9;
/// A call_indirect table byte other than the reserved zero.
constexpr uint8_t nonzero_reserved = 1;

/// Number of parameters of `wide_type`.
constexpr uint32_t wide_param_count = 64;

/// Builds the parity module: the callees, then a function of type `tested_type` with instructions `body`.
std::vector<uint8_t> make_parity_module(uint32_t tested_type, const std::vector<uint8_t>& body, bool with_table) {
   const std::vector<signature> signatures = {
      { {}, std::nullopt },
      { {}, types::i32 },
      { { types::i32, types::i64 }, types::i32 },
      { std::vector<uint8_t>(wide_param_count, types::i32), std::nullopt },
      { { types::i32, types::i64 }, types::i64 },
   };
   return make_module(signatures, { pair_type, wide_type, pair_i64_type, tested_type },
                      { callee_body, callee_body, callee_body, body }, with_table);
}

/// A function body and whether validation accepts it.
struct parity_case {
   const char*          description;
   uint32_t             tested_type;
   std::vector<uint8_t> body;
   bool                 valid;
   bool                 with_table = true;
};

/// Concatenates instruction sequences.
std::vector<uint8_t> concat(std::initializer_list<std::vector<uint8_t>> parts) {
   std::vector<uint8_t> result;
   for (const std::vector<uint8_t>& part : parts)
      result.insert(result.end(), part.begin(), part.end());
   return result;
}

/// `count` copies of `instruction`.
std::vector<uint8_t> repeat(const std::vector<uint8_t>& instruction, uint32_t count) {
   std::vector<uint8_t> result;
   result.reserve(instruction.size() * count);
   for (uint32_t i = 0; i < count; ++i)
      result.insert(result.end(), instruction.begin(), instruction.end());
   return result;
}

const std::vector<uint8_t> i32_zero = { opcodes::i32_const, 0x00 };
const std::vector<uint8_t> i64_zero = { opcodes::i64_const, 0x00 };
const std::vector<uint8_t> f32_zero = { opcodes::f32_const, 0x00, 0x00, 0x00, 0x00 };
const std::vector<uint8_t> f64_zero = { opcodes::f64_const, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

/// Function bodies around calls in and out of unreachable code, with the result the parser has always given.
std::vector<parity_case> parity_cases() {
   using O = opcodes;
   const uint8_t empty = types::pseudo;
   return {
      { "call after unreachable takes its arguments from the polymorphic stack", void_type,
        { O::unreachable, O::call, pair_callee, O::drop, O::end }, true },
      { "call result after unreachable stays on the stack", void_type,
        { O::unreachable, O::call, pair_callee, O::end }, false },
      { "call after unreachable checks the operand above the polymorphic base", void_type,
        concat({ { O::unreachable }, i64_zero, { O::call, pair_callee, O::drop, O::end } }), true },
      { "call after unreachable rejects an operand of the wrong type", void_type,
        concat({ { O::unreachable }, i32_zero, { O::call, pair_callee, O::drop, O::end } }), false },
      { "call after unreachable with every argument supplied", void_type,
        concat({ { O::unreachable }, i32_zero, i64_zero, { O::call, pair_callee, O::drop, O::end } }), true },
      { "call after unreachable with arguments in the wrong order", void_type,
        concat({ { O::unreachable }, i64_zero, i32_zero, { O::call, pair_callee, O::drop, O::end } }), false },
      { "call after unreachable leaves the operand below its arguments", void_type,
        concat({ { O::unreachable }, f32_zero, i32_zero, i64_zero, { O::call, pair_callee, O::drop, O::end } }),
        false },
      { "call after unreachable with the operand below its arguments dropped", void_type,
        concat({ { O::unreachable }, f32_zero, i32_zero, i64_zero,
                 { O::call, pair_callee, O::drop, O::drop, O::end } }),
        true },
      { "call after a second unreachable", void_type,
        { O::unreachable, O::unreachable, O::call, pair_callee, O::drop, O::end }, true },
      { "call result of the right type is an argument of the next call", void_type,
        { O::unreachable, O::call, pair_i64_callee, O::call, pair_callee, O::drop, O::end }, true },
      { "call result of the wrong type is an argument of the next call", void_type,
        { O::unreachable, O::call, pair_callee, O::call, pair_callee, O::drop, O::end }, false },
      { "block inside unreachable code starts a stack that is not polymorphic", void_type,
        { O::unreachable, O::block, empty, O::call, pair_callee, O::drop, O::end, O::end }, false },
      { "call after unreachable inside a block", void_type,
        { O::block, empty, O::unreachable, O::call, pair_callee, O::drop, O::end, O::end }, true },
      { "call after unreachable inside a loop", void_type,
        { O::loop, empty, O::unreachable, O::call, pair_callee, O::drop, O::end, O::end }, true },
      { "call after unreachable inside an if", void_type,
        concat({ i32_zero, { O::if_, empty, O::unreachable, O::call, pair_callee, O::drop, O::end, O::end } }),
        true },
      { "call after unreachable inside an else", void_type,
        concat({ i32_zero,
                 { O::if_, empty, O::else_, O::unreachable, O::call, pair_callee, O::drop, O::end, O::end } }),
        true },
      { "polymorphic stack ends with its block", void_type,
        { O::block, empty, O::unreachable, O::end, O::call, pair_callee, O::drop, O::end }, false },
      { "call after br", void_type,
        { O::block, empty, O::br, 0, O::call, pair_callee, O::drop, O::end, O::end }, true },
      { "call after br_table", void_type,
        concat({ { O::block, empty }, i32_zero, { O::br_table, 0, 0, O::call, pair_callee, O::drop, O::end, O::end } }),
        true },
      { "call after return", void_type, { O::return_, O::call, pair_callee, O::drop, O::end }, true },
      { "block result from a call after unreachable", void_type,
        { O::block, types::i32, O::unreachable, O::call, pair_callee, O::end, O::drop, O::end }, true },
      { "block result of the wrong type from a call after unreachable", void_type,
        { O::block, types::i64, O::unreachable, O::call, pair_callee, O::end, O::drop, O::end }, false },
      { "function result from a call after unreachable", i32_result_type,
        { O::unreachable, O::call, pair_callee, O::end }, true },
      { "function result of the wrong type from a call after unreachable", i32_result_type,
        { O::unreachable, O::call, pair_i64_callee, O::end }, false },
      { "many parameters after unreachable without operands", void_type,
        { O::unreachable, O::call, wide_callee, O::end }, true },
      { "many parameters after unreachable with some operands", void_type,
        concat({ { O::unreachable }, repeat(i32_zero, wide_param_count / 2), { O::call, wide_callee, O::end } }),
        true },
      { "many parameters after unreachable with every operand", void_type,
        concat({ { O::unreachable }, repeat(i32_zero, wide_param_count), { O::call, wide_callee, O::end } }), true },
      { "many parameters after unreachable with one operand too many", void_type,
        concat({ { O::unreachable }, repeat(i32_zero, wide_param_count + 1), { O::call, wide_callee, O::end } }),
        false },
      { "many parameters after unreachable with a wrong operand under the others", void_type,
        concat({ { O::unreachable }, f64_zero, repeat(i32_zero, wide_param_count - 1),
                 { O::call, wide_callee, O::end } }),
        false },
      { "call in reachable code without operands", void_type, { O::call, pair_callee, O::drop, O::end }, false },
      { "call in reachable code with only the last operand", void_type,
        concat({ i64_zero, { O::call, pair_callee, O::drop, O::end } }), false },
      { "call in reachable code with every operand", void_type,
        concat({ i32_zero, i64_zero, { O::call, pair_callee, O::drop, O::end } }), true },
      { "call to a missing function after unreachable", void_type,
        { O::unreachable, O::call, missing_function, O::drop, O::end }, false },
      { "call_indirect after unreachable takes its operands from the polymorphic stack", void_type,
        { O::unreachable, O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end }, true },
      { "call_indirect after unreachable rejects an index of the wrong type", void_type,
        concat({ { O::unreachable }, i64_zero,
                 { O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end } }),
        false },
      { "call_indirect after unreachable with only the index", void_type,
        concat({ { O::unreachable }, i32_zero,
                 { O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end } }),
        true },
      { "call_indirect after unreachable with every operand", void_type,
        concat({ { O::unreachable }, i32_zero, i64_zero, i32_zero,
                 { O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end } }),
        true },
      { "call_indirect after unreachable rejects an argument of the wrong type", void_type,
        concat({ { O::unreachable }, i32_zero, i32_zero,
                 { O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end } }),
        false },
      { "call_indirect after unreachable with many parameters", void_type,
        { O::unreachable, O::call_indirect, wide_type, call_indirect_reserved, O::end }, true },
      { "call_indirect inside a block inside unreachable code", void_type,
        { O::unreachable, O::block, empty, O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end,
          O::end },
        false },
      { "call_indirect inside a loop after unreachable", void_type,
        { O::loop, empty, O::unreachable, O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end,
          O::end },
        true },
      { "call_indirect in reachable code without operands", void_type,
        { O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end }, false },
      { "call_indirect to a missing type after unreachable", void_type,
        { O::unreachable, O::call_indirect, missing_type, call_indirect_reserved, O::drop, O::end }, false },
      { "call_indirect with a nonzero reserved byte after unreachable", void_type,
        { O::unreachable, O::call_indirect, pair_type, nonzero_reserved, O::drop, O::end }, false },
      { "call_indirect without a table after unreachable", void_type,
        { O::unreachable, O::call_indirect, pair_type, call_indirect_reserved, O::drop, O::end }, false, false },
   };
}

/// Parameter count and call count of the module that calls a wide function in unreachable code.
constexpr uint32_t large_param_count = 1u << 18;
constexpr uint32_t large_call_count  = 1u << 18;
/// Upper bound on validating that module when each call costs constant time.
constexpr std::chrono::seconds large_module_time_limit{ 5 };

/// Type indices of the large module.
constexpr uint32_t large_void_type = 0; ///< () -> ()
constexpr uint32_t large_wide_type = 1; ///< (i32 x large_param_count) -> ()
/// Function index of the callee of the large module.
constexpr uint8_t large_wide_callee = 0;

/// Builds a module whose second function makes `large_call_count` copies of `call_instruction` in unreachable code,
/// where function `large_wide_callee` has type `large_wide_type`.
std::vector<uint8_t> make_large_module(const std::vector<uint8_t>& call_instruction) {
   const std::vector<signature> signatures = {
      { {}, std::nullopt },
      { std::vector<uint8_t>(large_param_count, types::i32), std::nullopt },
   };
   const std::vector<uint8_t> body =
      concat({ { opcodes::unreachable }, repeat(call_instruction, large_call_count), { opcodes::end } });
   return make_module(signatures, { large_wide_type, large_void_type }, { callee_body, body }, true);
}

/// Validates `wasm` with backend `Impl` and checks that it is accepted within `large_module_time_limit`.
template <typename Impl>
void check_validates_quickly(const std::vector<uint8_t>& wasm) {
   const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
   CHECK(validates<Impl>(wasm));
   const std::chrono::steady_clock::duration elapsed = std::chrono::steady_clock::now() - start;
   CHECK(elapsed < large_module_time_limit);
}

} // namespace

/// Runs a validation test for the null backend and every execution backend compiled in.
#if SYS_VM_HAS_JIT_BACKEND
#define VALIDATION_TEST_CASE(name, tags) TEMPLATE_TEST_CASE(name, tags, null_backend, interpreter, jit)
#else
#define VALIDATION_TEST_CASE(name, tags) TEMPLATE_TEST_CASE(name, tags, null_backend, interpreter)
#endif

VALIDATION_TEST_CASE("Calls in unreachable code are accepted or rejected as before", "[unreachable_call]") {
   for (const parity_case& c : parity_cases()) {
      INFO(c.description);
      CHECK(validates<TestType>(make_parity_module(c.tested_type, c.body, c.with_table)) == c.valid);
   }
}

VALIDATION_TEST_CASE("Many calls with many parameters in unreachable code validate quickly", "[unreachable_call]") {
   check_validates_quickly<TestType>(make_large_module({ opcodes::call, large_wide_callee }));
}

VALIDATION_TEST_CASE("Many call_indirects with many parameters in unreachable code validate quickly",
                     "[unreachable_call]") {
   check_validates_quickly<TestType>(
         make_large_module({ opcodes::call_indirect, large_wide_type, call_indirect_reserved }));
}
