#pragma once

#include <sysio/vm/allocator.hpp>
#include <sysio/vm/exceptions.hpp>
#include <sysio/vm/signals.hpp>
#include <sysio/vm/softfloat.hpp>
#include <sysio/vm/types.hpp>
#include <sysio/vm/utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace sysio { namespace vm {

   /**
    * Experimental AArch64 machine-code writer used to prove Apple Silicon JIT feasibility.
    *
    * This writer implements the MVP scalar surface with integer-register codegen. Floating-point
    * operations are routed through softfloat helpers that receive and return raw bit patterns.
    * Every unsupported opcode fails during parsing instead of silently falling back or producing
    * incomplete native code.
    *
    * This backend is enabled only for Apple AArch64 builds, which are developer-only. The
    * relocation fixups below intentionally fail closed when generated code outgrows AArch64's
    * immediate branch ranges: B/BL reach +/-128 MiB, while B.cond and CBZ/CBNZ reach +/-1 MiB.
    * x86_64's rel32 branches do not have the same practical module-size limit, so any future
    * consensus use of this backend must either preserve this documented acceptance difference or
    * add long-branch lowering.
    */
   template <typename Context>
   class aarch64_machine_code_writer {
    public:
      enum class branch_kind { b, b_cond, cbz, cbnz };
      enum class call_arg_layout { parameter_order, host_stack_order };

      struct branch_t {
         void*       address = nullptr;
         branch_kind kind    = branch_kind::b;
         // For CBZ/CBNZ this is the tested register; for B.cond this is the condition code.
         uint32_t reg = 0;
      };
      using label_t = void*;

      /// Creates a new Apple Silicon AArch64 writer for one WASM code section.
      explicit aarch64_machine_code_writer(growable_allocator& alloc, std::size_t source_bytes, module& mod)
          : _allocator(alloc), _module(mod), _source_bytes(source_bytes), _code_segment_base(_allocator.start_code()) {}

      /// Finalizes the JIT code segment when parsing leaves the writer scope.
      ~aarch64_machine_code_writer() {
         if (_code_segment_base) {
            _allocator.end_code<true>(_code_segment_base);
         }
      }

      aarch64_machine_code_writer(const aarch64_machine_code_writer&)            = delete;
      aarch64_machine_code_writer& operator=(const aarch64_machine_code_writer&) = delete;

      /// Emits a runtime trap for WASM unreachable.
      void emit_unreachable() { emit_call_helper(&on_unreachable); }

      /// Emits no code for WASM nop.
      void emit_nop() {}

      /// Returns the current native-code address for WASM end relocation handling.
      label_t emit_end() { return _code; }

      /// Emits a function return branch.
      branch_t emit_return(uint32_t depth_change) { return emit_br(depth_change); }

      /// Emits no code for a block label.
      void emit_block() {}

      /// Returns the native-code address used as a loop branch target.
      label_t emit_loop() { return _code; }

      /// Emits a conditional branch to the else/end path when the i32 condition is false.
      branch_t emit_if() {
         pop_x(scratch0);
         return emit_cbz_placeholder(scratch0);
      }

      /// Emits an else delimiter and returns a branch that skips the else body.
      branch_t emit_else(branch_t if_loc) {
         auto result = emit_b_placeholder();
         fix_branch(if_loc, _code);
         return result;
      }

      /// Emits an unconditional structured branch.
      branch_t emit_br(uint32_t depth_change) {
         drop_branch_values(depth_change);
         return emit_b_placeholder();
      }

      /// Emits a conditional structured branch.
      branch_t emit_br_if(uint32_t depth_change) {
         pop_x(scratch0);
         if (depth_change == 0u || depth_change == 0x80000001u) {
            return emit_cbnz_placeholder(scratch0);
         }

         auto skip = emit_cbz_placeholder(scratch0);
         drop_branch_values(depth_change);
         auto result = emit_b_placeholder();
         fix_branch(skip, _code);
         return result;
      }

      /// Emits a linear br_table dispatcher for the AArch64 backend.
      struct br_table_parser {
         /// Emits one br_table case branch.
         branch_t emit_case(uint32_t depth_change) {
            _this->emit_mov_w_imm(_case_index, scratch1);
            _this->emit_cmp_w_reg(scratch0, scratch1);
            auto skip = _this->emit_b_cond_placeholder(condition_ne);
            _this->drop_branch_values(depth_change);
            auto result = _this->emit_b_placeholder();
            _this->fix_branch(skip, _this->_code);
            ++_case_index;
            return result;
         }

         /// Emits the br_table default branch.
         branch_t emit_default(uint32_t depth_change) {
            _this->drop_branch_values(depth_change);
            return _this->emit_b_placeholder();
         }

         aarch64_machine_code_writer* _this       = nullptr;
         uint32_t                     _case_index = 0;
      };

      /// Emits a br_table selector pop and returns the case parser.
      br_table_parser emit_br_table(uint32_t /*table_size*/) {
         // TODO: Replace linear br_table dispatch with a jump-table lowering when code-size and range handling allow.
         pop_x(scratch0);
         return { this, 0 };
      }

      /// Emits a direct internal function call.
      void emit_call(const func_type& ft, uint32_t funcnum) {
         for (uint32_t i = 0; i < ft.param_types.size(); ++i) {
            SYS_VM_ASSERT(is_supported_value_type(ft.param_types[i]), wasm_parse_exception,
                          "AArch64 JIT supports only scalar call parameters");
         }
         SYS_VM_ASSERT(ft.return_count == 0 || is_supported_value_type(ft.return_type), wasm_parse_exception,
                       "AArch64 JIT supports only scalar call returns");
         emit_enter_call();
         if (funcnum < _module.get_imported_functions_size()) {
            emit_imported_call(ft, funcnum);
            return;
         }

         const uint32_t cleanup_bytes = stage_call_args(ft.param_types.size());
         emit_mov_sp_to_reg(args_reg);
         emit_mov_x_reg(incoming_linear_memory_reg, linear_memory_reg);
         register_call(emit_bl_placeholder(), funcnum);
         if (cleanup_bytes) {
            emit_add_sp_bytes(cleanup_bytes);
         }
         if (ft.return_count) {
            push_x(return_reg);
         }
         emit_exit_call();
      }

      /// Emits a checked indirect call for integer-only function signatures.
      void emit_call_indirect(const func_type& ft, uint32_t functypeidx) {
         for (uint32_t i = 0; i < ft.param_types.size(); ++i) {
            SYS_VM_ASSERT(is_supported_value_type(ft.param_types[i]), wasm_parse_exception,
                          "AArch64 JIT supports only scalar call_indirect parameters");
         }
         SYS_VM_ASSERT(ft.return_count == 0 || is_supported_value_type(ft.return_type), wasm_parse_exception,
                       "AArch64 JIT supports only scalar call_indirect returns");
         emit_enter_call();
         pop_x(scratch4);
         const uint32_t cleanup_bytes = stage_call_args(ft.param_types.size());

         load_context();
         load_linear_memory();
         emit_mov_x_reg(incoming_linear_memory_reg, linear_memory_reg);
         emit_mov_sp_to_reg(args_reg);
         emit_mov_x_reg(3, scratch4);
         emit_mov_w_imm(functypeidx, 4);
         emit_call_helper(&execute_call_indirect);

         if (cleanup_bytes) {
            emit_add_sp_bytes(cleanup_bytes);
         }
         if (ft.return_count) {
            push_x(return_reg);
         }
         emit_exit_call();
      }

      /// Drops the top value from the native operand stack.
      void emit_drop() { pop_x(scratch0); }

      /// Emits a WASM select for scalar slots.
      void emit_select() {
         pop_x(scratch0);
         pop_x(scratch1);
         pop_x(scratch2);
         emit_cmp_w_imm_zero(scratch0);
         emit_csel_x(scratch0, scratch2, scratch1, condition_ne);
         push_x(scratch0);
      }

      /// Loads a local or parameter slot onto the native operand stack.
      void emit_get_local(uint32_t local_idx) {
         SYS_VM_ASSERT(local_idx < _local_slot_count, wasm_parse_exception, "AArch64 JIT local index out of range");
         load_local(local_idx, scratch0);
         push_x(scratch0);
      }

      /// Stores the top native operand stack value into a local or parameter slot.
      void emit_set_local(uint32_t local_idx) {
         SYS_VM_ASSERT(local_idx < _local_slot_count, wasm_parse_exception, "AArch64 JIT local index out of range");
         pop_x(scratch0);
         store_local(local_idx, scratch0);
      }

      /// Stores the top native operand stack value into a local slot without consuming it.
      void emit_tee_local(uint32_t local_idx) {
         SYS_VM_ASSERT(local_idx < _local_slot_count, wasm_parse_exception, "AArch64 JIT local index out of range");
         pop_x(scratch0);
         push_x(scratch0);
         store_local(local_idx, scratch0);
      }

      /// Emits a scalar global load as a raw native_value slot.
      void emit_get_global(uint32_t globalidx) {
         const auto& gl = _module.globals[globalidx];
         load_context();
         emit_mov_w_imm(globalidx, 1);
         switch (gl.type.content_type) {
            case types::i32:
               emit_call_helper(&get_global_i32);
               emit_zero_extend_w(return_reg);
               break;
            case types::i64: emit_call_helper(&get_global_i64); break;
            case types::f32:
               emit_call_helper(&get_global_f32);
               emit_zero_extend_w(return_reg);
               break;
            case types::f64: emit_call_helper(&get_global_f64); break;
            default: unsupported("get_global non-scalar");
         }
         push_x(return_reg);
         load_context();
         load_linear_memory();
      }

      /// Emits a scalar global store from a raw native_value slot.
      void emit_set_global(uint32_t globalidx) {
         const auto& gl = _module.globals[globalidx];
         pop_x(scratch0);
         load_context();
         emit_mov_w_imm(globalidx, 1);
         emit_mov_x_reg(2, scratch0);
         switch (gl.type.content_type) {
            case types::i32: emit_call_helper(&set_global_i32); break;
            case types::i64: emit_call_helper(&set_global_i64); break;
            case types::f32: emit_call_helper(&set_global_f32); break;
            case types::f64: emit_call_helper(&set_global_f64); break;
            default: unsupported("set_global non-scalar");
         }
         load_context();
         load_linear_memory();
      }

      /// Emits an i32 memory load.
      void emit_i32_load(uint32_t, uint32_t offset) {
         load_memory_address(offset);
         emit_ldr_w_unsigned(scratch0, scratch0, 0);
         push_x(scratch0);
      }

      /// Emits an f32 memory load as raw bits in a native_value slot.
      void emit_f32_load(uint32_t, uint32_t offset) {
         require_softfloat("f32.load");
         emit_i32_load(0, offset);
      }

      /// Emits an i64 memory load.
      void emit_i64_load(uint32_t, uint32_t offset) {
         load_memory_address(offset);
         emit_ldr_x_unsigned(scratch0, scratch0, 0);
         push_x(scratch0);
      }

      /// Emits an f64 memory load as raw bits in a native_value slot.
      void emit_f64_load(uint32_t, uint32_t offset) {
         require_softfloat("f64.load");
         emit_i64_load(0, offset);
      }

      /// Emits a sign-extending i8-to-i32 memory load.
      void emit_i32_load8_s(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x39c00000u); }

      /// Emits a sign-extending i16-to-i32 memory load.
      void emit_i32_load16_s(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x79c00000u); }

      /// Emits a zero-extending i8-to-i32 memory load.
      void emit_i32_load8_u(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x39400000u); }

      /// Emits a zero-extending i16-to-i32 memory load.
      void emit_i32_load16_u(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x79400000u); }

      /// Emits a sign-extending i8-to-i64 memory load.
      void emit_i64_load8_s(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x39800000u); }

      /// Emits a sign-extending i16-to-i64 memory load.
      void emit_i64_load16_s(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x79800000u); }

      /// Emits a sign-extending i32-to-i64 memory load.
      void emit_i64_load32_s(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0xb9800000u); }

      /// Emits a zero-extending i8-to-i64 memory load.
      void emit_i64_load8_u(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x39400000u); }

      /// Emits a zero-extending i16-to-i64 memory load.
      void emit_i64_load16_u(uint32_t, uint32_t offset) { emit_narrow_integer_load(offset, 0x79400000u); }

      /// Emits a zero-extending i32-to-i64 memory load.
      void emit_i64_load32_u(uint32_t, uint32_t offset) {
         load_memory_address(offset);
         emit_ldr_w_unsigned(scratch0, scratch0, 0);
         push_x(scratch0);
      }

      /// Emits an i32 memory store.
      void emit_i32_store(uint32_t, uint32_t offset) {
         pop_x(scratch1);
         load_memory_address(offset, scratch2);
         emit_str_w_unsigned(scratch1, scratch0, 0);
      }

      /// Emits an f32 memory store from raw bits in a native_value slot.
      void emit_f32_store(uint32_t, uint32_t offset) {
         require_softfloat("f32.store");
         emit_i32_store(0, offset);
      }

      /// Emits an i64 memory store.
      void emit_i64_store(uint32_t, uint32_t offset) {
         pop_x(scratch1);
         load_memory_address(offset, scratch2);
         emit_str_x_unsigned(scratch1, scratch0, 0);
      }

      /// Emits an f64 memory store from raw bits in a native_value slot.
      void emit_f64_store(uint32_t, uint32_t offset) {
         require_softfloat("f64.store");
         emit_i64_store(0, offset);
      }

      /// Emits an i32 low-byte memory store.
      void emit_i32_store8(uint32_t, uint32_t offset) { emit_narrow_integer_store(offset, 0x39000000u); }

      /// Emits an i32 low-halfword memory store.
      void emit_i32_store16(uint32_t, uint32_t offset) { emit_narrow_integer_store(offset, 0x79000000u); }

      /// Emits an i64 low-byte memory store.
      void emit_i64_store8(uint32_t, uint32_t offset) { emit_narrow_integer_store(offset, 0x39000000u); }

      /// Emits an i64 low-halfword memory store.
      void emit_i64_store16(uint32_t, uint32_t offset) { emit_narrow_integer_store(offset, 0x79000000u); }

      /// Emits an i64 low-word memory store.
      void emit_i64_store32(uint32_t, uint32_t offset) { emit_narrow_integer_store(offset, 0xb9000000u); }

      /// Emits the WASM memory.size instruction.
      void emit_current_memory() {
         load_context();
         emit_call_helper(&current_memory);
         emit_zero_extend_w(return_reg);
         push_x(return_reg);
         load_context();
         load_linear_memory();
      }

      /// Emits the WASM memory.grow instruction.
      void emit_grow_memory() {
         pop_x(scratch0);
         load_context();
         emit_mov_x_reg(1, scratch0);
         emit_call_helper(&grow_memory);
         emit_zero_extend_w(return_reg);
         push_x(return_reg);
         load_context();
         load_linear_memory();
      }

      /// Emits an i32 constant onto the native operand stack.
      void emit_i32_const(uint32_t value) {
         emit_mov_w_imm(value, scratch0);
         push_x(scratch0);
      }

      /// Emits an i64 constant onto the native operand stack.
      void emit_i64_const(uint64_t value) {
         emit_mov_x_imm(value, scratch0);
         push_x(scratch0);
      }

      /// Emits an f32 constant as a raw bit-pattern native_value slot.
      void emit_f32_const(float value) {
         require_softfloat("f32.const");
         uint32_t bits = 0;
         std::memcpy(&bits, &value, sizeof(bits));
         emit_i32_const(bits);
      }

      /// Emits an f64 constant as a raw bit-pattern native_value slot.
      void emit_f64_const(double value) {
         require_softfloat("f64.const");
         uint64_t bits = 0;
         std::memcpy(&bits, &value, sizeof(bits));
         emit_i64_const(bits);
      }

      /// Emits f32 equality through a raw-bit softfloat helper.
      void emit_f32_eq() { emit_float_compare(false, &soft_f32_eq); }

      /// Emits f32 inequality through a raw-bit softfloat helper.
      void emit_f32_ne() { emit_float_compare(false, &soft_f32_ne); }

      /// Emits f32 less-than through a raw-bit softfloat helper.
      void emit_f32_lt() { emit_float_compare(false, &soft_f32_lt); }

      /// Emits f32 greater-than through a raw-bit softfloat helper.
      void emit_f32_gt() { emit_float_compare(false, &soft_f32_gt); }

      /// Emits f32 less-than-or-equal through a raw-bit softfloat helper.
      void emit_f32_le() { emit_float_compare(false, &soft_f32_le); }

      /// Emits f32 greater-than-or-equal through a raw-bit softfloat helper.
      void emit_f32_ge() { emit_float_compare(false, &soft_f32_ge); }

      /// Emits f64 equality through a raw-bit softfloat helper.
      void emit_f64_eq() { emit_float_compare(true, &soft_f64_eq); }

      /// Emits f64 inequality through a raw-bit softfloat helper.
      void emit_f64_ne() { emit_float_compare(true, &soft_f64_ne); }

      /// Emits f64 less-than through a raw-bit softfloat helper.
      void emit_f64_lt() { emit_float_compare(true, &soft_f64_lt); }

      /// Emits f64 greater-than through a raw-bit softfloat helper.
      void emit_f64_gt() { emit_float_compare(true, &soft_f64_gt); }

      /// Emits f64 less-than-or-equal through a raw-bit softfloat helper.
      void emit_f64_le() { emit_float_compare(true, &soft_f64_le); }

      /// Emits f64 greater-than-or-equal through a raw-bit softfloat helper.
      void emit_f64_ge() { emit_float_compare(true, &soft_f64_ge); }

      /// Emits i32 count-leading-zeroes.
      void emit_i32_clz() { emit_integer_clz(false); }

      /// Emits i32 count-trailing-zeroes.
      void emit_i32_ctz() { emit_integer_ctz(false); }

      /// Emits i32 population count through a scalar helper.
      void emit_i32_popcnt() { emit_integer_unop_helper(false, &i32_popcnt); }

      /// Emits i32 equality with zero.
      void emit_i32_eqz() {
         pop_x(scratch0);
         emit_cmp_w_imm_zero(scratch0);
         emit_cset_w(scratch0, condition_eq);
         push_x(scratch0);
      }

      /// Emits i32 equality comparison.
      void emit_i32_eq() { emit_integer_compare(false, condition_eq); }

      /// Emits i32 inequality comparison.
      void emit_i32_ne() { emit_integer_compare(false, condition_ne); }

      /// Emits signed i32 less-than comparison.
      void emit_i32_lt_s() { emit_integer_compare(false, condition_lt); }

      /// Emits unsigned i32 less-than comparison.
      void emit_i32_lt_u() { emit_integer_compare(false, condition_lo); }

      /// Emits signed i32 greater-than comparison.
      void emit_i32_gt_s() { emit_integer_compare(false, condition_gt); }

      /// Emits unsigned i32 greater-than comparison.
      void emit_i32_gt_u() { emit_integer_compare(false, condition_hi); }

      /// Emits signed i32 less-than-or-equal comparison.
      void emit_i32_le_s() { emit_integer_compare(false, condition_le); }

      /// Emits unsigned i32 less-than-or-equal comparison.
      void emit_i32_le_u() { emit_integer_compare(false, condition_ls); }

      /// Emits signed i32 greater-than-or-equal comparison.
      void emit_i32_ge_s() { emit_integer_compare(false, condition_ge); }

      /// Emits unsigned i32 greater-than-or-equal comparison.
      void emit_i32_ge_u() { emit_integer_compare(false, condition_hs); }

      /// Emits i64 equality with zero.
      void emit_i64_eqz() {
         pop_x(scratch0);
         emit_cmp_x_imm_zero(scratch0);
         emit_cset_w(scratch0, condition_eq);
         push_x(scratch0);
      }

      /// Emits i64 equality comparison.
      void emit_i64_eq() { emit_integer_compare(true, condition_eq); }

      /// Emits i64 inequality comparison.
      void emit_i64_ne() { emit_integer_compare(true, condition_ne); }

      /// Emits signed i64 less-than comparison.
      void emit_i64_lt_s() { emit_integer_compare(true, condition_lt); }

      /// Emits unsigned i64 less-than comparison.
      void emit_i64_lt_u() { emit_integer_compare(true, condition_lo); }

      /// Emits signed i64 greater-than comparison.
      void emit_i64_gt_s() { emit_integer_compare(true, condition_gt); }

      /// Emits unsigned i64 greater-than comparison.
      void emit_i64_gt_u() { emit_integer_compare(true, condition_hi); }

      /// Emits signed i64 less-than-or-equal comparison.
      void emit_i64_le_s() { emit_integer_compare(true, condition_le); }

      /// Emits unsigned i64 less-than-or-equal comparison.
      void emit_i64_le_u() { emit_integer_compare(true, condition_ls); }

      /// Emits signed i64 greater-than-or-equal comparison.
      void emit_i64_ge_s() { emit_integer_compare(true, condition_ge); }

      /// Emits unsigned i64 greater-than-or-equal comparison.
      void emit_i64_ge_u() { emit_integer_compare(true, condition_hs); }

      /// Emits i32 addition.
      void emit_i32_add() { emit_integer_binop(0x0b000000u); }

      /// Emits i32 subtraction.
      void emit_i32_sub() { emit_integer_binop(0x4b000000u); }

      /// Emits i32 multiplication.
      void emit_i32_mul() { emit_integer_mul(0x1b000000u); }

      /// Emits signed i32 division through a trap-aware helper.
      void emit_i32_div_s() { emit_integer_divrem(false, &i32_div_s); }

      /// Emits unsigned i32 division through a trap-aware helper.
      void emit_i32_div_u() { emit_integer_divrem(false, &i32_div_u); }

      /// Emits signed i32 remainder through a trap-aware helper.
      void emit_i32_rem_s() { emit_integer_divrem(false, &i32_rem_s); }

      /// Emits unsigned i32 remainder through a trap-aware helper.
      void emit_i32_rem_u() { emit_integer_divrem(false, &i32_rem_u); }

      /// Emits i32 bitwise and.
      void emit_i32_and() { emit_integer_binop(0x0a000000u); }

      /// Emits i32 bitwise or.
      void emit_i32_or() { emit_integer_binop(0x2a000000u); }

      /// Emits i32 bitwise xor.
      void emit_i32_xor() { emit_integer_binop(0x4a000000u); }

      /// Emits i32 left shift.
      void emit_i32_shl() { emit_integer_shift(false, 0x1ac02000u); }

      /// Emits signed i32 right shift.
      void emit_i32_shr_s() { emit_integer_shift(false, 0x1ac02800u); }

      /// Emits unsigned i32 right shift.
      void emit_i32_shr_u() { emit_integer_shift(false, 0x1ac02400u); }

      /// Emits i32 rotate-left.
      void emit_i32_rotl() { emit_integer_rotate_left(false); }

      /// Emits i32 rotate-right.
      void emit_i32_rotr() { emit_integer_shift(false, 0x1ac02c00u); }

      /// Emits i64 count-leading-zeroes.
      void emit_i64_clz() { emit_integer_clz(true); }

      /// Emits i64 count-trailing-zeroes.
      void emit_i64_ctz() { emit_integer_ctz(true); }

      /// Emits i64 population count through a scalar helper.
      void emit_i64_popcnt() { emit_integer_unop_helper(true, &i64_popcnt); }

      /// Emits i64 addition.
      void emit_i64_add() { emit_integer_binop(0x8b000000u); }

      /// Emits i64 subtraction.
      void emit_i64_sub() { emit_integer_binop(0xcb000000u); }

      /// Emits i64 multiplication.
      void emit_i64_mul() { emit_integer_mul(0x9b000000u); }

      /// Emits signed i64 division through a trap-aware helper.
      void emit_i64_div_s() { emit_integer_divrem(true, &i64_div_s); }

      /// Emits unsigned i64 division through a trap-aware helper.
      void emit_i64_div_u() { emit_integer_divrem(true, &i64_div_u); }

      /// Emits signed i64 remainder through a trap-aware helper.
      void emit_i64_rem_s() { emit_integer_divrem(true, &i64_rem_s); }

      /// Emits unsigned i64 remainder through a trap-aware helper.
      void emit_i64_rem_u() { emit_integer_divrem(true, &i64_rem_u); }

      /// Emits i64 bitwise and.
      void emit_i64_and() { emit_integer_binop(0x8a000000u); }

      /// Emits i64 bitwise or.
      void emit_i64_or() { emit_integer_binop(0xaa000000u); }

      /// Emits i64 bitwise xor.
      void emit_i64_xor() { emit_integer_binop(0xca000000u); }

      /// Emits f32 absolute value through integer-register raw-bit code.
      void emit_f32_abs() { emit_value_unop_helper(false, &soft_f32_abs); }

      /// Emits f32 negation through integer-register raw-bit code.
      void emit_f32_neg() { emit_value_unop_helper(false, &soft_f32_neg); }

      /// Emits f32 ceil through a raw-bit softfloat helper.
      void emit_f32_ceil() { emit_value_unop_helper(false, &soft_f32_ceil); }

      /// Emits f32 floor through a raw-bit softfloat helper.
      void emit_f32_floor() { emit_value_unop_helper(false, &soft_f32_floor); }

      /// Emits f32 trunc through a raw-bit softfloat helper.
      void emit_f32_trunc() { emit_value_unop_helper(false, &soft_f32_trunc); }

      /// Emits f32 nearest through a raw-bit softfloat helper.
      void emit_f32_nearest() { emit_value_unop_helper(false, &soft_f32_nearest); }

      /// Emits f32 sqrt through a raw-bit softfloat helper.
      void emit_f32_sqrt() { emit_value_unop_helper(false, &soft_f32_sqrt); }

      /// Emits f32 addition through a raw-bit softfloat helper.
      void emit_f32_add() { emit_float_binop(false, &soft_f32_add); }

      /// Emits f32 subtraction through a raw-bit softfloat helper.
      void emit_f32_sub() { emit_float_binop(false, &soft_f32_sub); }

      /// Emits f32 multiplication through a raw-bit softfloat helper.
      void emit_f32_mul() { emit_float_binop(false, &soft_f32_mul); }

      /// Emits f32 division through a raw-bit softfloat helper.
      void emit_f32_div() { emit_float_binop(false, &soft_f32_div); }

      /// Emits f32 min through a raw-bit softfloat helper.
      void emit_f32_min() { emit_float_binop(false, &soft_f32_min); }

      /// Emits f32 max through a raw-bit softfloat helper.
      void emit_f32_max() { emit_float_binop(false, &soft_f32_max); }

      /// Emits f32 copysign through integer-register raw-bit code.
      void emit_f32_copysign() { emit_float_binop(false, &soft_f32_copysign); }

      /// Emits f64 absolute value through integer-register raw-bit code.
      void emit_f64_abs() { emit_value_unop_helper(true, &soft_f64_abs); }

      /// Emits f64 negation through integer-register raw-bit code.
      void emit_f64_neg() { emit_value_unop_helper(true, &soft_f64_neg); }

      /// Emits f64 ceil through a raw-bit softfloat helper.
      void emit_f64_ceil() { emit_value_unop_helper(true, &soft_f64_ceil); }

      /// Emits f64 floor through a raw-bit softfloat helper.
      void emit_f64_floor() { emit_value_unop_helper(true, &soft_f64_floor); }

      /// Emits f64 trunc through a raw-bit softfloat helper.
      void emit_f64_trunc() { emit_value_unop_helper(true, &soft_f64_trunc); }

      /// Emits f64 nearest through a raw-bit softfloat helper.
      void emit_f64_nearest() { emit_value_unop_helper(true, &soft_f64_nearest); }

      /// Emits f64 sqrt through a raw-bit softfloat helper.
      void emit_f64_sqrt() { emit_value_unop_helper(true, &soft_f64_sqrt); }

      /// Emits f64 addition through a raw-bit softfloat helper.
      void emit_f64_add() { emit_float_binop(true, &soft_f64_add); }

      /// Emits f64 subtraction through a raw-bit softfloat helper.
      void emit_f64_sub() { emit_float_binop(true, &soft_f64_sub); }

      /// Emits f64 multiplication through a raw-bit softfloat helper.
      void emit_f64_mul() { emit_float_binop(true, &soft_f64_mul); }

      /// Emits f64 division through a raw-bit softfloat helper.
      void emit_f64_div() { emit_float_binop(true, &soft_f64_div); }

      /// Emits f64 min through a raw-bit softfloat helper.
      void emit_f64_min() { emit_float_binop(true, &soft_f64_min); }

      /// Emits f64 max through a raw-bit softfloat helper.
      void emit_f64_max() { emit_float_binop(true, &soft_f64_max); }

      /// Emits f64 copysign through integer-register raw-bit code.
      void emit_f64_copysign() { emit_float_binop(true, &soft_f64_copysign); }

      /// Emits i32.trunc_s_f32 through a raw-bit trap-aware softfloat helper.
      void emit_i32_trunc_s_f32() { emit_value_unop_helper(false, &soft_f32_trunc_i32s); }

      /// Emits i32.trunc_u_f32 through a raw-bit trap-aware softfloat helper.
      void emit_i32_trunc_u_f32() { emit_value_unop_helper(false, &soft_f32_trunc_i32u); }

      /// Emits i32.trunc_s_f64 through a raw-bit trap-aware softfloat helper.
      void emit_i32_trunc_s_f64() { emit_value_unop_helper(false, &soft_f64_trunc_i32s); }

      /// Emits i32.trunc_u_f64 through a raw-bit trap-aware softfloat helper.
      void emit_i32_trunc_u_f64() { emit_value_unop_helper(false, &soft_f64_trunc_i32u); }

      /// Emits i64.trunc_s_f32 through a raw-bit trap-aware softfloat helper.
      void emit_i64_trunc_s_f32() { emit_value_unop_helper(true, &soft_f32_trunc_i64s); }

      /// Emits i64.trunc_u_f32 through a raw-bit trap-aware softfloat helper.
      void emit_i64_trunc_u_f32() { emit_value_unop_helper(true, &soft_f32_trunc_i64u); }

      /// Emits i64.trunc_s_f64 through a raw-bit trap-aware softfloat helper.
      void emit_i64_trunc_s_f64() { emit_value_unop_helper(true, &soft_f64_trunc_i64s); }

      /// Emits i64.trunc_u_f64 through a raw-bit trap-aware softfloat helper.
      void emit_i64_trunc_u_f64() { emit_value_unop_helper(true, &soft_f64_trunc_i64u); }

      /// Emits f32.convert_s_i32 through a raw-bit softfloat helper.
      void emit_f32_convert_s_i32() { emit_value_unop_helper(false, &soft_i32_to_f32); }

      /// Emits f32.convert_u_i32 through a raw-bit softfloat helper.
      void emit_f32_convert_u_i32() { emit_value_unop_helper(false, &soft_ui32_to_f32); }

      /// Emits f32.convert_s_i64 through a raw-bit softfloat helper.
      void emit_f32_convert_s_i64() { emit_value_unop_helper(false, &soft_i64_to_f32); }

      /// Emits f32.convert_u_i64 through a raw-bit softfloat helper.
      void emit_f32_convert_u_i64() { emit_value_unop_helper(false, &soft_ui64_to_f32); }

      /// Emits f32.demote_f64 through a raw-bit softfloat helper.
      void emit_f32_demote_f64() { emit_value_unop_helper(false, &soft_f64_demote_f32); }

      /// Emits f64.convert_s_i32 through a raw-bit softfloat helper.
      void emit_f64_convert_s_i32() { emit_value_unop_helper(true, &soft_i32_to_f64); }

      /// Emits f64.convert_u_i32 through a raw-bit softfloat helper.
      void emit_f64_convert_u_i32() { emit_value_unop_helper(true, &soft_ui32_to_f64); }

      /// Emits f64.convert_s_i64 through a raw-bit softfloat helper.
      void emit_f64_convert_s_i64() { emit_value_unop_helper(true, &soft_i64_to_f64); }

      /// Emits f64.convert_u_i64 through a raw-bit softfloat helper.
      void emit_f64_convert_u_i64() { emit_value_unop_helper(true, &soft_ui64_to_f64); }

      /// Emits f64.promote_f32 through a raw-bit softfloat helper.
      void emit_f64_promote_f32() { emit_value_unop_helper(true, &soft_f32_promote_f64); }

      /// Emits i32.reinterpret_f32 without changing the raw native_value slot.
      void emit_i32_reinterpret_f32() { require_softfloat("i32.reinterpret_f32"); }

      /// Emits i64.reinterpret_f64 without changing the raw native_value slot.
      void emit_i64_reinterpret_f64() { require_softfloat("i64.reinterpret_f64"); }

      /// Emits f32.reinterpret_i32 without changing the raw native_value slot.
      void emit_f32_reinterpret_i32() { require_softfloat("f32.reinterpret_i32"); }

      /// Emits f64.reinterpret_i64 without changing the raw native_value slot.
      void emit_f64_reinterpret_i64() { require_softfloat("f64.reinterpret_i64"); }

      /// Emits i64 left shift.
      void emit_i64_shl() { emit_integer_shift(true, 0x9ac02000u); }

      /// Emits signed i64 right shift.
      void emit_i64_shr_s() { emit_integer_shift(true, 0x9ac02800u); }

      /// Emits unsigned i64 right shift.
      void emit_i64_shr_u() { emit_integer_shift(true, 0x9ac02400u); }

      /// Emits i64 rotate-left.
      void emit_i64_rotl() { emit_integer_rotate_left(true); }

      /// Emits i64 rotate-right.
      void emit_i64_rotr() { emit_integer_shift(true, 0x9ac02c00u); }

      /// Emits i32.wrap_i64 by retaining the low word.
      void emit_i32_wrap_i64() {
         pop_x(scratch0);
         emit_zero_extend_w(scratch0);
         push_x(scratch0);
      }

      /// Emits signed i32-to-i64 extension.
      void emit_i64_extend_s_i32() {
         pop_x(scratch0);
         emit_sxtw(scratch0, scratch0);
         push_x(scratch0);
      }

      /// Emits unsigned i32-to-i64 extension.
      void emit_i64_extend_u_i32() {
         pop_x(scratch0);
         emit_zero_extend_w(scratch0);
         push_x(scratch0);
      }

      /// Resolves a structured branch relocation.
      void fix_branch(branch_t branch, label_t target) { fix_structured_branch(branch, target); }

      /// Emits an AArch64 function prologue and copies integer parameters into frame slots.
      void emit_prologue(const func_type& ft, const guarded_vector<local_entry>& locals, uint32_t funcnum) {
         for (uint32_t i = 0; i < ft.param_types.size(); ++i) {
            SYS_VM_ASSERT(is_supported_value_type(ft.param_types[i]), wasm_parse_exception,
                          "AArch64 JIT supports only scalar parameters");
         }

         _param_count      = ft.param_types.size();
         _local_slot_count = _param_count;
         for (uint32_t i = 0; i < locals.size(); ++i) {
            SYS_VM_ASSERT(is_supported_value_type(locals[i].type), wasm_parse_exception,
                          "AArch64 JIT supports only scalar locals");
            SYS_VM_ASSERT(_local_slot_count <= 0xffffffffu - locals[i].count, wasm_parse_exception,
                          "AArch64 JIT local count overflow");
            _local_slot_count += locals[i].count;
         }

         const auto function_source_bytes = funcnum < _module.code.size() ? _module.code[funcnum].size : _source_bytes;
         const auto local_zeroing_bytes   = static_cast<std::size_t>(_local_slot_count) * local_zeroing_code_bytes;
         const auto function_bytes =
               std::max(initial_function_code_bytes, (function_source_bytes * function_code_size_ratio) +
                                                           local_zeroing_bytes + function_code_margin_bytes);
         _code_start = _allocator.alloc<unsigned char>(function_bytes);
         _code       = _code_start;
         _code_end   = _code_start + function_bytes;

         emit_u32(0xa9bf7bfdu); // stp x29, x30, [sp, #-16]!
         emit_u32(0x910003fdu); // mov x29, sp

         _context_slot              = _local_slot_count;
         _linear_memory_slot        = _context_slot + 1;
         _preserved_x19_slot        = _linear_memory_slot + 1;
         const uint32_t frame_bytes = align_to_16((_local_slot_count + hidden_frame_slot_count) * local_slot_bytes);
         if (frame_bytes) {
            emit_sub_sp_bytes(frame_bytes);
         }

         start_function(_code_start, funcnum + _module.get_imported_functions_size());
         store_local(_preserved_x19_slot, linear_memory_reg);
         emit_mov_x_reg(linear_memory_reg, incoming_linear_memory_reg);
         store_local(_context_slot, return_reg);
         store_local(_linear_memory_slot, linear_memory_reg);
         for (uint32_t i = 0; i < _local_slot_count; ++i) { store_local(i, zero_reg); }
         for (uint32_t i = 0; i < _param_count; ++i) {
            load_arg(i, scratch0);
            store_local(i, scratch0);
         }
      }

      /// Emits an AArch64 function epilogue and return sequence.
      void emit_epilogue(const func_type& ft, const guarded_vector<local_entry>&, uint32_t) {
         if (ft.return_count) {
            pop_x(return_reg);
         }
         load_local(_preserved_x19_slot, linear_memory_reg);
         emit_u32(0x910003bfu); // mov sp, x29
         emit_u32(0xa8c17bfdu); // ldp x29, x30, [sp], #16
         emit_u32(0xd65f03c0u); // ret
      }

      /// Records the emitted native offset for the current WASM function body.
      void finalize(function_body& body) {
         body.jit_code_offset =
               static_cast<unsigned char*>(_code_start) - static_cast<unsigned char*>(_code_segment_base);
      }

      /// Returns the current native-code write address.
      const void* get_addr() const { return _code; }

      /// Returns the native-code segment base.
      const void* get_base_addr() const { return _code_segment_base; }

    private:
      static constexpr uint32_t    scratch0                           = 9;
      static constexpr uint32_t    scratch1                           = 10;
      static constexpr uint32_t    scratch2                           = 11;
      static constexpr uint32_t    scratch3                           = 12;
      static constexpr uint32_t    scratch4                           = 13;
      static constexpr uint32_t    incoming_linear_memory_reg         = 1;
      static constexpr uint32_t    linear_memory_reg                  = 19;
      static constexpr uint32_t    args_reg                           = 2;
      static constexpr uint32_t    frame_reg                          = 29;
      static constexpr uint32_t    zero_reg                           = 31;
      static constexpr uint32_t    stack_reg                          = 31;
      static constexpr uint32_t    return_reg                         = 0;
      static constexpr uint32_t    local_slot_bytes                   = sizeof(native_value);
      static constexpr uint32_t    operand_stack_slot_bytes           = 16;
      static constexpr uint32_t    hidden_frame_slot_count            = 3;
      static constexpr uint32_t    max_aligned_sp_adjustment_bytes    = 4080;
      static constexpr uint32_t    max_scaled_unsigned_x_offset_bytes = 4095 * local_slot_bytes;
      static constexpr uint32_t    condition_eq                       = 0;
      static constexpr uint32_t    condition_ne                       = 1;
      static constexpr uint32_t    condition_hs                       = 2;
      static constexpr uint32_t    condition_lo                       = 3;
      static constexpr uint32_t    condition_hi                       = 8;
      static constexpr uint32_t    condition_ls                       = 9;
      static constexpr uint32_t    condition_ge                       = 10;
      static constexpr uint32_t    condition_lt                       = 11;
      static constexpr uint32_t    condition_gt                       = 12;
      static constexpr uint32_t    condition_le                       = 13;
      static constexpr std::size_t initial_function_code_bytes        = 4096;
      static constexpr std::size_t function_code_growth_bytes         = 4096;
      static constexpr std::size_t function_code_margin_bytes         = 1024;
      static constexpr std::size_t function_code_size_ratio           = 128;
      static constexpr std::size_t local_zeroing_code_bytes           = 32;
      static_assert(local_slot_bytes == 8, "AArch64 JIT expects 8-byte native value slots");
      static_assert(max_aligned_sp_adjustment_bytes <= 4095, "AArch64 SP adjustment chunk must fit imm12");
      static_assert((max_aligned_sp_adjustment_bytes % 16) == 0, "AArch64 SP adjustment chunk must keep SP aligned");

      /// Returns true when a WASM value type is currently supported by the AArch64 backend.
      static bool is_supported_value_type(value_type type) {
         return type == types::i32 || type == types::i64 || type == types::f32 || type == types::f64;
      }

      /// Rounds a byte count up to the next 16-byte AArch64 stack-aligned size.
      static constexpr uint32_t align_to_16(uint32_t size) { return (size + 15u) & ~15u; }

      /// Throws a parse exception for an unsupported AArch64 operation.
      [[noreturn]] static void unsupported(const char* opcode) { throw wasm_parse_exception{ opcode }; }

      /// Fails while emitting float opcodes when deterministic softfloat helpers are not compiled in.
      static void require_softfloat(const char* opcode) {
#ifndef SYS_VM_SOFTFLOAT
         unsupported(opcode);
#else
         (void)opcode;
#endif
      }

      /// Emits a raw little-endian AArch64 instruction.
      void emit_u32(uint32_t instruction) {
         ensure_code_capacity(sizeof(instruction));
         std::memcpy(_code, &instruction, sizeof(instruction));
         _code += sizeof(instruction);
      }

      /// Extends the current function code range while preserving already-recorded native labels.
      void ensure_code_capacity(std::size_t byte_count) {
         SYS_VM_ASSERT(_code, wasm_bad_alloc, "AArch64 JIT emitted code before function prologue");
         if (_code + byte_count <= _code_end) {
            return;
         }

         const auto used_bytes     = static_cast<std::size_t>(_code - _code_start);
         const auto reserved_bytes = static_cast<std::size_t>(_code_end - _code_start);
         const auto needed_bytes   = used_bytes + byte_count;
         const auto growth_bytes   = std::max(function_code_growth_bytes, needed_bytes - reserved_bytes);
         auto*      extension      = _allocator.alloc<unsigned char>(growth_bytes);
         SYS_VM_ASSERT(extension == _code_end, wasm_bad_alloc,
                       "AArch64 JIT function code buffer extension was not contiguous");
         _code_end += growth_bytes;
      }

      /// Emits a SUB immediate instruction whose immediate must fit in one unshifted AArch64 imm12.
      void emit_sub_imm(uint32_t dst, uint32_t src, uint32_t byte_count) {
         SYS_VM_ASSERT(byte_count <= 4095, wasm_parse_exception, "AArch64 JIT immediate offset is too large");
         emit_u32(0xd1000000u | (byte_count << 10) | (src << 5) | dst);
      }

      /// Emits an ADD immediate instruction whose immediate must fit in one unshifted AArch64 imm12.
      void emit_add_imm(uint32_t dst, uint32_t src, uint32_t byte_count) {
         SYS_VM_ASSERT(byte_count <= 4095, wasm_parse_exception, "AArch64 JIT immediate offset is too large");
         emit_u32(0x91000000u | (byte_count << 10) | (src << 5) | dst);
      }

      /// Emits an add-to-stack-pointer operation, splitting large adjustments into valid aligned chunks.
      void emit_add_sp_bytes(uint32_t byte_count) {
         while (byte_count > 0) {
            const uint32_t chunk = std::min<uint32_t>(byte_count, max_aligned_sp_adjustment_bytes);
            emit_add_sp_imm(chunk);
            byte_count -= chunk;
         }
      }

      /// Emits a stack-pointer addition for fixed frame deallocation.
      void emit_add_sp_imm(uint32_t byte_count) { emit_add_imm(stack_reg, stack_reg, byte_count); }

      /// Emits a 64-bit ADD register instruction.
      void emit_add_reg(uint32_t dst, uint32_t lhs, uint32_t rhs) {
         emit_u32(0x8b000000u | (rhs << 16) | (lhs << 5) | dst);
      }

      /// Emits a 64-bit SUB register instruction.
      void emit_sub_reg(uint32_t dst, uint32_t lhs, uint32_t rhs) {
         emit_u32(0xcb000000u | (rhs << 16) | (lhs << 5) | dst);
      }

      /// Emits a 32-bit register compare.
      void emit_cmp_w_reg(uint32_t lhs, uint32_t rhs) { emit_u32(0x6b00001fu | (rhs << 16) | (lhs << 5)); }

      /// Emits a 64-bit register compare.
      void emit_cmp_x_reg(uint32_t lhs, uint32_t rhs) { emit_u32(0xeb00001fu | (rhs << 16) | (lhs << 5)); }

      /// Emits a 32-bit compare with immediate zero.
      void emit_cmp_w_imm_zero(uint32_t reg) { emit_u32(0x7100001fu | (reg << 5)); }

      /// Emits a 64-bit compare with immediate zero.
      void emit_cmp_x_imm_zero(uint32_t reg) { emit_u32(0xf100001fu | (reg << 5)); }

      /// Emits a CSET into a W register.
      void emit_cset_w(uint32_t dst, uint32_t condition) { emit_u32(0x1a9f07e0u | ((condition ^ 1u) << 12) | dst); }

      /// Emits a 64-bit conditional select instruction.
      void emit_csel_x(uint32_t dst, uint32_t lhs, uint32_t rhs, uint32_t condition) {
         emit_u32(0x9a800000u | (rhs << 16) | (condition << 12) | (lhs << 5) | dst);
      }

      /// Emits a move from SP into a general-purpose register.
      void emit_mov_sp_to_reg(uint32_t dst) { emit_add_imm(dst, stack_reg, 0); }

      /// Emits a 64-bit register move.
      void emit_mov_x_reg(uint32_t dst, uint32_t src) { emit_u32(0xaa0003e0u | (src << 16) | dst); }

      /// Emits a stack-pointer subtraction for fixed frame allocation.
      void emit_sub_sp_imm(uint32_t byte_count) { emit_sub_imm(stack_reg, stack_reg, byte_count); }

      /// Emits a subtract-from-stack-pointer operation, splitting large adjustments into valid aligned chunks.
      void emit_sub_sp_bytes(uint32_t byte_count) {
         while (byte_count > 0) {
            const uint32_t chunk = std::min<uint32_t>(byte_count, max_aligned_sp_adjustment_bytes);
            emit_sub_sp_imm(chunk);
            byte_count -= chunk;
         }
      }

      /// Emits an unsigned-offset X-register load.
      void emit_ldr_x_unsigned(uint32_t dst, uint32_t base, uint32_t byte_offset) {
         SYS_VM_ASSERT((byte_offset % local_slot_bytes) == 0, wasm_parse_exception,
                       "AArch64 JIT load offset must be slot-aligned");
         const uint32_t scaled_offset = byte_offset / local_slot_bytes;
         SYS_VM_ASSERT(scaled_offset <= 4095, wasm_parse_exception, "AArch64 JIT load offset is too large");
         emit_u32(0xf9400000u | (scaled_offset << 10) | (base << 5) | dst);
      }

      /// Emits an unsigned-offset W-register load.
      void emit_ldr_w_unsigned(uint32_t dst, uint32_t base, uint32_t byte_offset) {
         SYS_VM_ASSERT((byte_offset % 4) == 0, wasm_parse_exception,
                       "AArch64 JIT i32 load offset must be 4-byte aligned");
         const uint32_t scaled_offset = byte_offset / 4;
         SYS_VM_ASSERT(scaled_offset <= 4095, wasm_parse_exception, "AArch64 JIT i32 load offset is too large");
         emit_u32(0xb9400000u | (scaled_offset << 10) | (base << 5) | dst);
      }

      /// Emits an unsigned-offset X-register store.
      void emit_str_x_unsigned(uint32_t src, uint32_t base, uint32_t byte_offset) {
         SYS_VM_ASSERT((byte_offset % local_slot_bytes) == 0, wasm_parse_exception,
                       "AArch64 JIT store offset must be slot-aligned");
         const uint32_t scaled_offset = byte_offset / local_slot_bytes;
         SYS_VM_ASSERT(scaled_offset <= 4095, wasm_parse_exception, "AArch64 JIT store offset is too large");
         emit_u32(0xf9000000u | (scaled_offset << 10) | (base << 5) | src);
      }

      /// Computes an address at a byte offset from the native stack pointer.
      void emit_stack_offset_address(uint32_t dst, uint32_t byte_offset) {
         emit_mov_sp_to_reg(dst);
         if (byte_offset <= 4095) {
            emit_add_imm(dst, dst, byte_offset);
         } else {
            emit_mov_x_imm(byte_offset, scratch1);
            emit_add_reg(dst, dst, scratch1);
         }
      }

      /// Loads an X register from an arbitrary aligned byte offset above the native stack pointer.
      void emit_ldr_x_stack_offset(uint32_t dst, uint32_t byte_offset) {
         if (byte_offset <= max_scaled_unsigned_x_offset_bytes) {
            emit_ldr_x_unsigned(dst, stack_reg, byte_offset);
            return;
         }

         emit_stack_offset_address(scratch2, byte_offset);
         emit_ldr_x_unsigned(dst, scratch2, 0);
      }

      /// Stores an X register to an arbitrary aligned byte offset above the native stack pointer.
      void emit_str_x_stack_offset(uint32_t src, uint32_t byte_offset) {
         if (byte_offset <= max_scaled_unsigned_x_offset_bytes) {
            emit_str_x_unsigned(src, stack_reg, byte_offset);
            return;
         }

         emit_stack_offset_address(scratch2, byte_offset);
         emit_str_x_unsigned(src, scratch2, 0);
      }

      /// Materializes a contiguous native_value argument array from 16-byte WASM operand-stack slots.
      uint32_t stage_call_args(uint32_t param_count, call_arg_layout layout = call_arg_layout::parameter_order) {
         SYS_VM_ASSERT(param_count <= std::numeric_limits<uint32_t>::max() / operand_stack_slot_bytes,
                       wasm_parse_exception, "AArch64 JIT call argument stack is too large");
         SYS_VM_ASSERT(param_count <= std::numeric_limits<uint32_t>::max() / local_slot_bytes, wasm_parse_exception,
                       "AArch64 JIT call argument array is too large");

         const uint32_t args_bytes    = align_to_16(param_count * local_slot_bytes);
         const uint32_t operand_bytes = param_count * operand_stack_slot_bytes;
         if (args_bytes) {
            emit_sub_sp_bytes(args_bytes);
         }

         for (uint32_t i = 0; i < param_count; ++i) {
            // Output slot i maps to source argument i for direct calls and to reversed host-stack order for imports.
            const uint32_t source_arg    = layout == call_arg_layout::parameter_order ? i : (param_count - 1u - i);
            const uint32_t source_offset = args_bytes + ((param_count - 1u - source_arg) * operand_stack_slot_bytes);
            emit_ldr_x_stack_offset(scratch0, source_offset);
            emit_str_x_stack_offset(scratch0, i * local_slot_bytes);
         }

         return args_bytes + operand_bytes;
      }

      /// Emits an unsigned-offset W-register store.
      void emit_str_w_unsigned(uint32_t src, uint32_t base, uint32_t byte_offset) {
         SYS_VM_ASSERT((byte_offset % 4) == 0, wasm_parse_exception,
                       "AArch64 JIT i32 store offset must be 4-byte aligned");
         const uint32_t scaled_offset = byte_offset / 4;
         SYS_VM_ASSERT(scaled_offset <= 4095, wasm_parse_exception, "AArch64 JIT i32 store offset is too large");
         emit_u32(0xb9000000u | (scaled_offset << 10) | (base << 5) | src);
      }

      /// Emits a zero-offset AArch64 memory access after the effective address has already been materialized.
      void emit_memory_op_zero_offset(uint32_t opcode_base, uint32_t reg, uint32_t base) {
         emit_u32(opcode_base | (base << 5) | reg);
      }

      /// Emits a narrow integer load after applying the WASM address and static offset.
      void emit_narrow_integer_load(uint32_t offset, uint32_t opcode_base) {
         load_memory_address(offset);
         emit_memory_op_zero_offset(opcode_base, scratch0, scratch0);
         push_x(scratch0);
      }

      /// Emits a narrow integer store after applying the WASM address and static offset.
      void emit_narrow_integer_store(uint32_t offset, uint32_t opcode_base) {
         pop_x(scratch1);
         load_memory_address(offset, scratch2);
         emit_memory_op_zero_offset(opcode_base, scratch1, scratch0);
      }

      /// Pops a WASM i32 address and resolves it to a native linear-memory address.
      void load_memory_address(uint32_t offset, uint32_t offset_reg = scratch1) {
         pop_x(scratch0);
         emit_zero_extend_w(scratch0);
         if (offset <= 4095) {
            emit_add_imm(scratch0, scratch0, offset);
         } else {
            emit_mov_x_imm(offset, offset_reg);
            emit_add_reg(scratch0, scratch0, offset_reg);
         }
         emit_add_reg(scratch0, linear_memory_reg, scratch0);
      }

      /// Computes the address of a frame local slot into scratch2.
      void emit_local_address(uint32_t local_idx) {
         const uint32_t byte_offset = (local_idx + 1) * local_slot_bytes;
         if (byte_offset <= 4095) {
            emit_sub_imm(scratch2, frame_reg, byte_offset);
         } else {
            emit_mov_x_imm(byte_offset, scratch2);
            emit_sub_reg(scratch2, frame_reg, scratch2);
         }
      }

      /// Loads a native_value argument from the incoming argument array.
      void load_arg(uint32_t arg_idx, uint32_t dst) { emit_ldr_x_unsigned(dst, args_reg, arg_idx * local_slot_bytes); }

      /// Loads a frame local slot into a register.
      void load_local(uint32_t local_idx, uint32_t dst) {
         emit_local_address(local_idx);
         emit_ldr_x_unsigned(dst, scratch2, 0);
      }

      /// Stores a register into a frame local slot.
      void store_local(uint32_t local_idx, uint32_t src) {
         emit_local_address(local_idx);
         emit_str_x_unsigned(src, scratch2, 0);
      }

      /// Emits a comparison that leaves a 0/1 i32 on the WASM operand stack.
      void emit_integer_compare(bool is_64_bit, uint32_t condition) {
         pop_x(scratch0);
         pop_x(scratch1);
         if (is_64_bit) {
            emit_cmp_x_reg(scratch1, scratch0);
         } else {
            emit_cmp_w_reg(scratch1, scratch0);
         }
         emit_cset_w(scratch0, condition);
         push_x(scratch0);
      }

      /// Restores the JIT context pointer in x0 after an internal call uses x0 for its return value.
      void load_context() { load_local(_context_slot, return_reg); }

      /// Restores the linear-memory base pointer in x1 after a C ABI helper call.
      void load_linear_memory() { load_local(_linear_memory_slot, linear_memory_reg); }

      /// Emits an imported direct function call through jit_execution_context::call_host_function.
      void emit_imported_call(const func_type& ft, uint32_t funcnum) {
         const uint32_t cleanup_bytes = stage_call_args(ft.param_types.size(), call_arg_layout::host_stack_order);

         load_context();
         emit_mov_sp_to_reg(1);
         emit_mov_w_imm(funcnum, 2);
         emit_mov_x_imm(reinterpret_cast<uint64_t>(&call_host_function), scratch3);
         emit_blr(scratch3);

         if (cleanup_bytes) {
            emit_add_sp_bytes(cleanup_bytes);
         }
         if (ft.return_count) {
            push_x(return_reg);
         }
         emit_exit_call();
      }

      /// Calls the existing JIT host-function adapter while preserving x86's longjmp exception boundary.
      static native_value call_host_function(Context* context, native_value* stack, uint32_t idx) {
         native_value result{ uint64_t{ 0 } };
         vm::longjmp_on_exception([&]() { result = context->call_host_function(stack, idx); });
         return result;
      }

      /// Reserves one execution-context call-depth slot for generated calls.
      static void enter_call(Context* context) {
         vm::longjmp_on_exception([&]() { context->enter_jit_call(); });
      }

      /// Releases one execution-context call-depth slot without unwinding through generated code.
      static void exit_call(Context* context) {
         vm::longjmp_on_exception([&]() { context->exit_jit_call(); });
      }

      /// Returns the current linear-memory page count through the execution context.
      static int32_t current_memory(Context* context) { return context->current_linear_memory(); }

      /// Grows linear memory through the execution context.
      static int32_t grow_memory(Context* context, int32_t pages) { return context->grow_linear_memory(pages); }

      /// Returns an i32 global through the execution context.
      static int32_t get_global_i32(Context* context, uint32_t index) { return context->get_global_i32(index); }

      /// Returns an i64 global through the execution context.
      static int64_t get_global_i64(Context* context, uint32_t index) { return context->get_global_i64(index); }

      /// Returns raw f32 global bits through the execution context.
      static uint32_t get_global_f32(Context* context, uint32_t index) { return context->get_global_f32(index); }

      /// Returns raw f64 global bits through the execution context.
      static uint64_t get_global_f64(Context* context, uint32_t index) { return context->get_global_f64(index); }

      /// Stores an i32 global through the execution context.
      static void set_global_i32(Context* context, uint32_t index, int32_t value) {
         context->set_global_i32(index, value);
      }

      /// Stores an i64 global through the execution context.
      static void set_global_i64(Context* context, uint32_t index, int64_t value) {
         context->set_global_i64(index, value);
      }

      /// Stores raw f32 global bits through the execution context.
      static void set_global_f32(Context* context, uint32_t index, uint32_t value) {
         context->set_global_f32(index, value);
      }

      /// Stores raw f64 global bits through the execution context.
      static void set_global_f64(Context* context, uint32_t index, uint64_t value) {
         context->set_global_f64(index, value);
      }

      /// Raises the WASM unreachable trap without unwinding through the generated frame.
      static void on_unreachable() { vm::throw_<wasm_interpreter_exception>("unreachable"); }

      /// Produces a readable failure when a float opcode is reached without softfloat support.
      [[noreturn]] static void softfloat_disabled(const char* opcode) { throw wasm_parse_exception{ opcode }; }

      /// Runs a trap-capable softfloat conversion behind the JIT longjmp boundary.
      template <typename Result, typename Function>
      static Result trap_softfloat(Function function) {
         Result result{};
         vm::longjmp_on_exception([&]() { result = function(); });
         return result;
      }

#ifdef SYS_VM_SOFTFLOAT
      static float    bits_to_f32(uint32_t bits) { return ::from_softfloat32(softfloat32_t{ bits }); }
      static double   bits_to_f64(uint64_t bits) { return ::from_softfloat64(softfloat64_t{ bits }); }
      static uint32_t f32_to_bits(float value) { return ::to_softfloat32(value).v; }
      static uint64_t f64_to_bits(double value) { return ::to_softfloat64(value).v; }

      /// Raw-bit host adapters for the existing interpreter softfloat implementation.
      static uint32_t soft_f32_eq(uint32_t lhs, uint32_t rhs) {
         return _sysio_f32_eq(bits_to_f32(lhs), bits_to_f32(rhs));
      }
      static uint32_t soft_f32_ne(uint32_t lhs, uint32_t rhs) {
         return _sysio_f32_ne(bits_to_f32(lhs), bits_to_f32(rhs));
      }
      static uint32_t soft_f32_lt(uint32_t lhs, uint32_t rhs) {
         return _sysio_f32_lt(bits_to_f32(lhs), bits_to_f32(rhs));
      }
      static uint32_t soft_f32_gt(uint32_t lhs, uint32_t rhs) {
         return _sysio_f32_gt(bits_to_f32(lhs), bits_to_f32(rhs));
      }
      static uint32_t soft_f32_le(uint32_t lhs, uint32_t rhs) {
         return _sysio_f32_le(bits_to_f32(lhs), bits_to_f32(rhs));
      }
      static uint32_t soft_f32_ge(uint32_t lhs, uint32_t rhs) {
         return _sysio_f32_ge(bits_to_f32(lhs), bits_to_f32(rhs));
      }
      static uint32_t soft_f64_eq(uint64_t lhs, uint64_t rhs) {
         return _sysio_f64_eq(bits_to_f64(lhs), bits_to_f64(rhs));
      }
      static uint32_t soft_f64_ne(uint64_t lhs, uint64_t rhs) {
         return _sysio_f64_ne(bits_to_f64(lhs), bits_to_f64(rhs));
      }
      static uint32_t soft_f64_lt(uint64_t lhs, uint64_t rhs) {
         return _sysio_f64_lt(bits_to_f64(lhs), bits_to_f64(rhs));
      }
      static uint32_t soft_f64_gt(uint64_t lhs, uint64_t rhs) {
         return _sysio_f64_gt(bits_to_f64(lhs), bits_to_f64(rhs));
      }
      static uint32_t soft_f64_le(uint64_t lhs, uint64_t rhs) {
         return _sysio_f64_le(bits_to_f64(lhs), bits_to_f64(rhs));
      }
      static uint32_t soft_f64_ge(uint64_t lhs, uint64_t rhs) {
         return _sysio_f64_ge(bits_to_f64(lhs), bits_to_f64(rhs));
      }

      static uint32_t soft_f32_abs(uint32_t value) { return f32_to_bits(_sysio_f32_abs(bits_to_f32(value))); }
      static uint32_t soft_f32_neg(uint32_t value) { return f32_to_bits(_sysio_f32_neg(bits_to_f32(value))); }
      static uint32_t soft_f32_add(uint32_t lhs, uint32_t rhs) {
         return f32_to_bits(_sysio_f32_add(bits_to_f32(lhs), bits_to_f32(rhs)));
      }
      static uint32_t soft_f32_sub(uint32_t lhs, uint32_t rhs) {
         return f32_to_bits(_sysio_f32_sub(bits_to_f32(lhs), bits_to_f32(rhs)));
      }
      static uint32_t soft_f32_mul(uint32_t lhs, uint32_t rhs) {
         return f32_to_bits(_sysio_f32_mul(bits_to_f32(lhs), bits_to_f32(rhs)));
      }
      static uint32_t soft_f32_div(uint32_t lhs, uint32_t rhs) {
         return f32_to_bits(_sysio_f32_div(bits_to_f32(lhs), bits_to_f32(rhs)));
      }
      static uint32_t soft_f32_min(uint32_t lhs, uint32_t rhs) {
         return f32_to_bits(_sysio_f32_min(bits_to_f32(lhs), bits_to_f32(rhs)));
      }
      static uint32_t soft_f32_max(uint32_t lhs, uint32_t rhs) {
         return f32_to_bits(_sysio_f32_max(bits_to_f32(lhs), bits_to_f32(rhs)));
      }
      static uint32_t soft_f32_copysign(uint32_t lhs, uint32_t rhs) {
         return f32_to_bits(_sysio_f32_copysign(bits_to_f32(lhs), bits_to_f32(rhs)));
      }
      static uint32_t soft_f32_sqrt(uint32_t value) { return f32_to_bits(_sysio_f32_sqrt(bits_to_f32(value))); }
      static uint32_t soft_f32_ceil(uint32_t bits) { return f32_to_bits(_sysio_f32_ceil(bits_to_f32(bits))); }
      static uint32_t soft_f32_floor(uint32_t bits) { return f32_to_bits(_sysio_f32_floor(bits_to_f32(bits))); }
      static uint32_t soft_f32_trunc(uint32_t bits) { return f32_to_bits(_sysio_f32_trunc(bits_to_f32(bits))); }
      static uint32_t soft_f32_nearest(uint32_t bits) { return f32_to_bits(_sysio_f32_nearest(bits_to_f32(bits))); }

      static uint64_t soft_f64_abs(uint64_t value) { return f64_to_bits(_sysio_f64_abs(bits_to_f64(value))); }
      static uint64_t soft_f64_neg(uint64_t value) { return f64_to_bits(_sysio_f64_neg(bits_to_f64(value))); }
      static uint64_t soft_f64_add(uint64_t lhs, uint64_t rhs) {
         return f64_to_bits(_sysio_f64_add(bits_to_f64(lhs), bits_to_f64(rhs)));
      }
      static uint64_t soft_f64_sub(uint64_t lhs, uint64_t rhs) {
         return f64_to_bits(_sysio_f64_sub(bits_to_f64(lhs), bits_to_f64(rhs)));
      }
      static uint64_t soft_f64_mul(uint64_t lhs, uint64_t rhs) {
         return f64_to_bits(_sysio_f64_mul(bits_to_f64(lhs), bits_to_f64(rhs)));
      }
      static uint64_t soft_f64_div(uint64_t lhs, uint64_t rhs) {
         return f64_to_bits(_sysio_f64_div(bits_to_f64(lhs), bits_to_f64(rhs)));
      }
      static uint64_t soft_f64_min(uint64_t lhs, uint64_t rhs) {
         return f64_to_bits(_sysio_f64_min(bits_to_f64(lhs), bits_to_f64(rhs)));
      }
      static uint64_t soft_f64_max(uint64_t lhs, uint64_t rhs) {
         return f64_to_bits(_sysio_f64_max(bits_to_f64(lhs), bits_to_f64(rhs)));
      }
      static uint64_t soft_f64_copysign(uint64_t lhs, uint64_t rhs) {
         return f64_to_bits(_sysio_f64_copysign(bits_to_f64(lhs), bits_to_f64(rhs)));
      }
      static uint64_t soft_f64_sqrt(uint64_t value) { return f64_to_bits(_sysio_f64_sqrt(bits_to_f64(value))); }
      static uint64_t soft_f64_ceil(uint64_t bits) { return f64_to_bits(_sysio_f64_ceil(bits_to_f64(bits))); }
      static uint64_t soft_f64_floor(uint64_t bits) { return f64_to_bits(_sysio_f64_floor(bits_to_f64(bits))); }
      static uint64_t soft_f64_trunc(uint64_t bits) { return f64_to_bits(_sysio_f64_trunc(bits_to_f64(bits))); }
      static uint64_t soft_f64_nearest(uint64_t bits) { return f64_to_bits(_sysio_f64_nearest(bits_to_f64(bits))); }

      static uint32_t soft_f32_trunc_i32s(uint32_t bits) {
         return trap_softfloat<uint32_t>(
               [&]() { return static_cast<uint32_t>(_sysio_f32_trunc_i32s(bits_to_f32(bits))); });
      }
      static uint32_t soft_f32_trunc_i32u(uint32_t bits) {
         return trap_softfloat<uint32_t>([&]() { return _sysio_f32_trunc_i32u(bits_to_f32(bits)); });
      }
      static uint32_t soft_f64_trunc_i32s(uint64_t bits) {
         return trap_softfloat<uint32_t>(
               [&]() { return static_cast<uint32_t>(_sysio_f64_trunc_i32s(bits_to_f64(bits))); });
      }
      static uint32_t soft_f64_trunc_i32u(uint64_t bits) {
         return trap_softfloat<uint32_t>([&]() { return _sysio_f64_trunc_i32u(bits_to_f64(bits)); });
      }
      static uint64_t soft_f32_trunc_i64s(uint32_t bits) {
         return trap_softfloat<uint64_t>(
               [&]() { return static_cast<uint64_t>(_sysio_f32_trunc_i64s(bits_to_f32(bits))); });
      }
      static uint64_t soft_f32_trunc_i64u(uint32_t bits) {
         return trap_softfloat<uint64_t>([&]() { return _sysio_f32_trunc_i64u(bits_to_f32(bits)); });
      }
      static uint64_t soft_f64_trunc_i64s(uint64_t bits) {
         return trap_softfloat<uint64_t>(
               [&]() { return static_cast<uint64_t>(_sysio_f64_trunc_i64s(bits_to_f64(bits))); });
      }
      static uint64_t soft_f64_trunc_i64u(uint64_t bits) {
         return trap_softfloat<uint64_t>([&]() { return _sysio_f64_trunc_i64u(bits_to_f64(bits)); });
      }

      static uint32_t soft_i32_to_f32(uint32_t value) {
         return f32_to_bits(_sysio_i32_to_f32(static_cast<int32_t>(value)));
      }
      static uint32_t soft_ui32_to_f32(uint32_t value) { return f32_to_bits(_sysio_ui32_to_f32(value)); }
      static uint32_t soft_i64_to_f32(uint64_t value) {
         return f32_to_bits(_sysio_i64_to_f32(static_cast<int64_t>(value)));
      }
      static uint32_t soft_ui64_to_f32(uint64_t value) { return f32_to_bits(_sysio_ui64_to_f32(value)); }
      static uint32_t soft_f64_demote_f32(uint64_t value) { return f32_to_bits(_sysio_f64_demote(bits_to_f64(value))); }
      static uint64_t soft_i32_to_f64(uint32_t value) {
         return f64_to_bits(_sysio_i32_to_f64(static_cast<int32_t>(value)));
      }
      static uint64_t soft_ui32_to_f64(uint32_t value) { return f64_to_bits(_sysio_ui32_to_f64(value)); }
      static uint64_t soft_i64_to_f64(uint64_t value) {
         return f64_to_bits(_sysio_i64_to_f64(static_cast<int64_t>(value)));
      }
      static uint64_t soft_ui64_to_f64(uint64_t value) { return f64_to_bits(_sysio_ui64_to_f64(value)); }
      static uint64_t soft_f32_promote_f64(uint32_t value) {
         return f64_to_bits(_sysio_f32_promote(bits_to_f32(value)));
      }
#else
      static uint32_t soft_f32_eq(uint32_t, uint32_t) { softfloat_disabled("f32.eq"); }
      static uint32_t soft_f32_ne(uint32_t, uint32_t) { softfloat_disabled("f32.ne"); }
      static uint32_t soft_f32_lt(uint32_t, uint32_t) { softfloat_disabled("f32.lt"); }
      static uint32_t soft_f32_gt(uint32_t, uint32_t) { softfloat_disabled("f32.gt"); }
      static uint32_t soft_f32_le(uint32_t, uint32_t) { softfloat_disabled("f32.le"); }
      static uint32_t soft_f32_ge(uint32_t, uint32_t) { softfloat_disabled("f32.ge"); }
      static uint32_t soft_f64_eq(uint64_t, uint64_t) { softfloat_disabled("f64.eq"); }
      static uint32_t soft_f64_ne(uint64_t, uint64_t) { softfloat_disabled("f64.ne"); }
      static uint32_t soft_f64_lt(uint64_t, uint64_t) { softfloat_disabled("f64.lt"); }
      static uint32_t soft_f64_gt(uint64_t, uint64_t) { softfloat_disabled("f64.gt"); }
      static uint32_t soft_f64_le(uint64_t, uint64_t) { softfloat_disabled("f64.le"); }
      static uint32_t soft_f64_ge(uint64_t, uint64_t) { softfloat_disabled("f64.ge"); }

      static uint32_t soft_f32_abs(uint32_t) { softfloat_disabled("f32.abs"); }
      static uint32_t soft_f32_neg(uint32_t) { softfloat_disabled("f32.neg"); }
      static uint32_t soft_f32_add(uint32_t, uint32_t) { softfloat_disabled("f32.add"); }
      static uint32_t soft_f32_sub(uint32_t, uint32_t) { softfloat_disabled("f32.sub"); }
      static uint32_t soft_f32_mul(uint32_t, uint32_t) { softfloat_disabled("f32.mul"); }
      static uint32_t soft_f32_div(uint32_t, uint32_t) { softfloat_disabled("f32.div"); }
      static uint32_t soft_f32_min(uint32_t, uint32_t) { softfloat_disabled("f32.min"); }
      static uint32_t soft_f32_max(uint32_t, uint32_t) { softfloat_disabled("f32.max"); }
      static uint32_t soft_f32_copysign(uint32_t, uint32_t) { softfloat_disabled("f32.copysign"); }
      static uint32_t soft_f32_sqrt(uint32_t) { softfloat_disabled("f32.sqrt"); }
      static uint32_t soft_f32_ceil(uint32_t) { softfloat_disabled("f32.ceil"); }
      static uint32_t soft_f32_floor(uint32_t) { softfloat_disabled("f32.floor"); }
      static uint32_t soft_f32_trunc(uint32_t) { softfloat_disabled("f32.trunc"); }
      static uint32_t soft_f32_nearest(uint32_t) { softfloat_disabled("f32.nearest"); }

      static uint64_t soft_f64_abs(uint64_t) { softfloat_disabled("f64.abs"); }
      static uint64_t soft_f64_neg(uint64_t) { softfloat_disabled("f64.neg"); }
      static uint64_t soft_f64_add(uint64_t, uint64_t) { softfloat_disabled("f64.add"); }
      static uint64_t soft_f64_sub(uint64_t, uint64_t) { softfloat_disabled("f64.sub"); }
      static uint64_t soft_f64_mul(uint64_t, uint64_t) { softfloat_disabled("f64.mul"); }
      static uint64_t soft_f64_div(uint64_t, uint64_t) { softfloat_disabled("f64.div"); }
      static uint64_t soft_f64_min(uint64_t, uint64_t) { softfloat_disabled("f64.min"); }
      static uint64_t soft_f64_max(uint64_t, uint64_t) { softfloat_disabled("f64.max"); }
      static uint64_t soft_f64_copysign(uint64_t, uint64_t) { softfloat_disabled("f64.copysign"); }
      static uint64_t soft_f64_sqrt(uint64_t) { softfloat_disabled("f64.sqrt"); }
      static uint64_t soft_f64_ceil(uint64_t) { softfloat_disabled("f64.ceil"); }
      static uint64_t soft_f64_floor(uint64_t) { softfloat_disabled("f64.floor"); }
      static uint64_t soft_f64_trunc(uint64_t) { softfloat_disabled("f64.trunc"); }
      static uint64_t soft_f64_nearest(uint64_t) { softfloat_disabled("f64.nearest"); }

      static uint32_t soft_f32_trunc_i32s(uint32_t) {
         return trap_softfloat<uint32_t>([&]() -> uint32_t { softfloat_disabled("i32.trunc_s_f32"); });
      }
      static uint32_t soft_f32_trunc_i32u(uint32_t) {
         return trap_softfloat<uint32_t>([&]() -> uint32_t { softfloat_disabled("i32.trunc_u_f32"); });
      }
      static uint32_t soft_f64_trunc_i32s(uint64_t) {
         return trap_softfloat<uint32_t>([&]() -> uint32_t { softfloat_disabled("i32.trunc_s_f64"); });
      }
      static uint32_t soft_f64_trunc_i32u(uint64_t) {
         return trap_softfloat<uint32_t>([&]() -> uint32_t { softfloat_disabled("i32.trunc_u_f64"); });
      }
      static uint64_t soft_f32_trunc_i64s(uint32_t) {
         return trap_softfloat<uint64_t>([&]() -> uint64_t { softfloat_disabled("i64.trunc_s_f32"); });
      }
      static uint64_t soft_f32_trunc_i64u(uint32_t) {
         return trap_softfloat<uint64_t>([&]() -> uint64_t { softfloat_disabled("i64.trunc_u_f32"); });
      }
      static uint64_t soft_f64_trunc_i64s(uint64_t) {
         return trap_softfloat<uint64_t>([&]() -> uint64_t { softfloat_disabled("i64.trunc_s_f64"); });
      }
      static uint64_t soft_f64_trunc_i64u(uint64_t) {
         return trap_softfloat<uint64_t>([&]() -> uint64_t { softfloat_disabled("i64.trunc_u_f64"); });
      }

      static uint32_t soft_i32_to_f32(uint32_t) { softfloat_disabled("f32.convert_s_i32"); }
      static uint32_t soft_ui32_to_f32(uint32_t) { softfloat_disabled("f32.convert_u_i32"); }
      static uint32_t soft_i64_to_f32(uint64_t) { softfloat_disabled("f32.convert_s_i64"); }
      static uint32_t soft_ui64_to_f32(uint64_t) { softfloat_disabled("f32.convert_u_i64"); }
      static uint32_t soft_f64_demote_f32(uint64_t) { softfloat_disabled("f32.demote_f64"); }
      static uint64_t soft_i32_to_f64(uint32_t) { softfloat_disabled("f64.convert_s_i32"); }
      static uint64_t soft_ui32_to_f64(uint32_t) { softfloat_disabled("f64.convert_u_i32"); }
      static uint64_t soft_i64_to_f64(uint64_t) { softfloat_disabled("f64.convert_s_i64"); }
      static uint64_t soft_ui64_to_f64(uint64_t) { softfloat_disabled("f64.convert_u_i64"); }
      static uint64_t soft_f32_promote_f64(uint32_t) { softfloat_disabled("f64.promote_f32"); }
#endif

      /// Resolves and executes a checked call_indirect table entry.
      static native_value execute_call_indirect(Context* context, void* linear_memory, native_value* stack,
                                                uint32_t table_index, uint32_t functypeidx) {
         native_value result{ uint64_t{ 0 } };
         vm::longjmp_on_exception([&]() {
            auto& full_module = context->get_module();
            auto& jit_module  = *full_module.jit_mod;
            SYS_VM_ASSERT(!jit_module.tables.empty(), wasm_interpreter_exception, "call_indirect out of range");
            const auto& table = jit_module.tables[0].table;
            SYS_VM_ASSERT(table_index < table.size(), wasm_interpreter_exception, "call_indirect out of range");
            SYS_VM_ASSERT(functypeidx < jit_module.type_aliases.size(), wasm_interpreter_exception,
                          "call_indirect incorrect function type");
            const uint32_t expected_type = jit_module.type_aliases[functypeidx];
            const uint32_t funcnum       = table[table_index];
            SYS_VM_ASSERT(funcnum < jit_module.fast_functions.size(), wasm_interpreter_exception,
                          "call_indirect out of range");
            SYS_VM_ASSERT(jit_module.fast_functions[funcnum] == expected_type, wasm_interpreter_exception,
                          "call_indirect incorrect function type");
            const uint32_t imported_count = jit_module.get_imported_functions_size();
            if (funcnum < imported_count) {
               const auto&  ft          = jit_module.get_function_type(funcnum);
               const size_t param_count = ft.param_types.size();
               for (size_t i = 0; i < param_count / 2; ++i) { std::swap(stack[i], stack[param_count - i - 1]); }
               result = context->call_host_function(stack, funcnum);
            } else {
               using jit_fn_type = native_value (*)(void*, void*, native_value*);
               auto* code        = reinterpret_cast<unsigned char*>(full_module.allocator._code_base) +
                                   jit_module.jit_code_offset[funcnum - imported_count];
               auto  fn          = reinterpret_cast<jit_fn_type>(code);
               result            = fn(context, linear_memory, stack);
            }
         });
         return result;
      }

      /// Counts set bits in an i32 value without using floating-point or SIMD instructions.
      static uint32_t i32_popcnt(uint32_t value) { return static_cast<uint32_t>(__builtin_popcount(value)); }

      /// Counts set bits in an i64 value without using floating-point or SIMD instructions.
      static uint64_t i64_popcnt(uint64_t value) { return static_cast<uint64_t>(__builtin_popcountll(value)); }

      /// Performs WASM signed i32 division with exact trap semantics.
      static uint32_t i32_div_s(uint32_t lhs_bits, uint32_t rhs_bits) {
         uint32_t result = 0;
         vm::longjmp_on_exception([&]() {
            const auto lhs = static_cast<int32_t>(lhs_bits);
            const auto rhs = static_cast<int32_t>(rhs_bits);
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i32.div_s divide by zero");
            SYS_VM_ASSERT(!(lhs == std::numeric_limits<int32_t>::min() && rhs == -1), wasm_interpreter_exception,
                          "i32.div_s traps when I32_MAX/-1");
            result = static_cast<uint32_t>(lhs / rhs);
         });
         return result;
      }

      /// Performs WASM unsigned i32 division with exact trap semantics.
      static uint32_t i32_div_u(uint32_t lhs, uint32_t rhs) {
         uint32_t result = 0;
         vm::longjmp_on_exception([&]() {
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i32.div_u divide by zero");
            result = lhs / rhs;
         });
         return result;
      }

      /// Performs WASM signed i32 remainder with exact trap semantics.
      static uint32_t i32_rem_s(uint32_t lhs_bits, uint32_t rhs_bits) {
         uint32_t result = 0;
         vm::longjmp_on_exception([&]() {
            const auto lhs = static_cast<int32_t>(lhs_bits);
            const auto rhs = static_cast<int32_t>(rhs_bits);
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i32.rem_s divide by zero");
            if (lhs == std::numeric_limits<int32_t>::min() && rhs == -1) {
               result = 0;
            } else {
               result = static_cast<uint32_t>(lhs % rhs);
            }
         });
         return result;
      }

      /// Performs WASM unsigned i32 remainder with exact trap semantics.
      static uint32_t i32_rem_u(uint32_t lhs, uint32_t rhs) {
         uint32_t result = 0;
         vm::longjmp_on_exception([&]() {
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i32.rem_u divide by zero");
            result = lhs % rhs;
         });
         return result;
      }

      /// Performs WASM signed i64 division with exact trap semantics.
      static uint64_t i64_div_s(uint64_t lhs_bits, uint64_t rhs_bits) {
         uint64_t result = 0;
         vm::longjmp_on_exception([&]() {
            const auto lhs = static_cast<int64_t>(lhs_bits);
            const auto rhs = static_cast<int64_t>(rhs_bits);
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i64.div_s divide by zero");
            SYS_VM_ASSERT(!(lhs == std::numeric_limits<int64_t>::min() && rhs == -1), wasm_interpreter_exception,
                          "i64.div_s traps when I64_MAX/-1");
            result = static_cast<uint64_t>(lhs / rhs);
         });
         return result;
      }

      /// Performs WASM unsigned i64 division with exact trap semantics.
      static uint64_t i64_div_u(uint64_t lhs, uint64_t rhs) {
         uint64_t result = 0;
         vm::longjmp_on_exception([&]() {
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i64.div_u divide by zero");
            result = lhs / rhs;
         });
         return result;
      }

      /// Performs WASM signed i64 remainder with exact trap semantics.
      static uint64_t i64_rem_s(uint64_t lhs_bits, uint64_t rhs_bits) {
         uint64_t result = 0;
         vm::longjmp_on_exception([&]() {
            const auto lhs = static_cast<int64_t>(lhs_bits);
            const auto rhs = static_cast<int64_t>(rhs_bits);
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i64.rem_s divide by zero");
            if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
               result = 0;
            } else {
               result = static_cast<uint64_t>(lhs % rhs);
            }
         });
         return result;
      }

      /// Performs WASM unsigned i64 remainder with exact trap semantics.
      static uint64_t i64_rem_u(uint64_t lhs, uint64_t rhs) {
         uint64_t result = 0;
         vm::longjmp_on_exception([&]() {
            SYS_VM_ASSERT(rhs != 0, wasm_interpreter_exception, "i64.rem_u divide by zero");
            result = lhs % rhs;
         });
         return result;
      }

      /// Emits a placeholder BL instruction and returns its address for later relocation.
      void* emit_bl_placeholder() {
         void* branch = _code;
         emit_u32(0x94000000u);
         return branch;
      }

      /// Emits an indirect branch-and-link through an X register.
      void emit_blr(uint32_t reg) { emit_u32(0xd63f0000u | (reg << 5)); }

      /// Emits an indirect C ABI helper call through a materialized absolute function address.
      template <typename Function>
      void emit_call_helper(Function helper) {
         emit_mov_x_imm(reinterpret_cast<uint64_t>(helper), scratch3);
         emit_blr(scratch3);
      }

      /// Emits a placeholder unconditional B instruction.
      branch_t emit_b_placeholder() {
         void* branch = _code;
         emit_u32(0x14000000u);
         return { branch, branch_kind::b, 0 };
      }

      /// Emits a placeholder conditional B.cond instruction.
      branch_t emit_b_cond_placeholder(uint32_t condition) {
         void* branch = _code;
         emit_u32(0x54000000u | condition);
         return { branch, branch_kind::b_cond, condition };
      }

      /// Emits a placeholder CBZ instruction.
      branch_t emit_cbz_placeholder(uint32_t reg) {
         void* branch = _code;
         emit_u32(0x34000000u | reg);
         return { branch, branch_kind::cbz, reg };
      }

      /// Emits a placeholder CBNZ instruction.
      branch_t emit_cbnz_placeholder(uint32_t reg) {
         void* branch = _code;
         emit_u32(0x35000000u | reg);
         return { branch, branch_kind::cbnz, reg };
      }

      /// Resolves an AArch64 unconditional B relocation.
      static void fix_b(void* branch, void* target) {
         const auto* branch_addr = static_cast<const unsigned char*>(branch);
         const auto* target_addr = static_cast<const unsigned char*>(target);
         const auto  relative    = target_addr - branch_addr;
         SYS_VM_ASSERT((relative % 4) == 0, wasm_parse_exception, "AArch64 JIT branch target is unaligned");
         const auto immediate = relative / 4;
         SYS_VM_ASSERT(immediate >= -(1 << 25) && immediate < (1 << 25), wasm_parse_exception,
                       "AArch64 JIT branch target is out of range");
         const uint32_t instruction = 0x14000000u | (static_cast<uint32_t>(immediate) & 0x03ffffffu);
         std::memcpy(branch, &instruction, sizeof(instruction));
      }

      /// Resolves an AArch64 CBZ/CBNZ relocation.
      ///
      /// CBZ/CBNZ use a 19-bit immediate scaled by 4, so large structured bodies may exceed the
      /// +/-1 MiB reach that x86_64 conditional branches do not hit in practice.
      static void fix_cb(branch_t branch, void* target) {
         const auto* branch_addr = static_cast<const unsigned char*>(branch.address);
         const auto* target_addr = static_cast<const unsigned char*>(target);
         const auto  relative    = target_addr - branch_addr;
         SYS_VM_ASSERT((relative % 4) == 0, wasm_parse_exception, "AArch64 JIT branch target is unaligned");
         const auto immediate = relative / 4;
         SYS_VM_ASSERT(immediate >= -(1 << 18) && immediate < (1 << 18), wasm_parse_exception,
                       "AArch64 JIT conditional branch target is out of range");
         const uint32_t base        = branch.kind == branch_kind::cbz ? 0x34000000u : 0x35000000u;
         const uint32_t instruction = base | ((static_cast<uint32_t>(immediate) & 0x7ffffu) << 5) | branch.reg;
         std::memcpy(branch.address, &instruction, sizeof(instruction));
      }

      /// Resolves an AArch64 B.cond relocation.
      ///
      /// B.cond has the same +/-1 MiB reach as CBZ/CBNZ. The current writer fails closed instead
      /// of synthesizing an inverted short branch plus long unconditional branch.
      static void fix_b_cond(branch_t branch, void* target) {
         const auto* branch_addr = static_cast<const unsigned char*>(branch.address);
         const auto* target_addr = static_cast<const unsigned char*>(target);
         const auto  relative    = target_addr - branch_addr;
         SYS_VM_ASSERT((relative % 4) == 0, wasm_parse_exception, "AArch64 JIT branch target is unaligned");
         const auto immediate = relative / 4;
         SYS_VM_ASSERT(immediate >= -(1 << 18) && immediate < (1 << 18), wasm_parse_exception,
                       "AArch64 JIT conditional branch target is out of range");
         const uint32_t instruction =
               0x54000000u | ((static_cast<uint32_t>(immediate) & 0x7ffffu) << 5) | (branch.reg & 0xfu);
         std::memcpy(branch.address, &instruction, sizeof(instruction));
      }

      /// Drops values from the WASM operand stack for structured branch transfer.
      void drop_branch_values(uint32_t depth_change) {
         const bool     preserve_result = (depth_change & 0x80000000u) != 0;
         const uint32_t count           = depth_change & 0x7fffffffu;
         if (preserve_result) {
            if (count <= 1) {
               return;
            }
            pop_x(scratch0);
            emit_add_sp_bytes((count - 1) * 16u);
            push_x(scratch0);
         } else if (count) {
            emit_add_sp_bytes(count * 16u);
         }
      }

      /// Resolves an AArch64 BL relocation.
      static void fix_bl(void* branch, void* target) {
         const auto* branch_addr = static_cast<const unsigned char*>(branch);
         const auto* target_addr = static_cast<const unsigned char*>(target);
         const auto  relative    = target_addr - branch_addr;
         SYS_VM_ASSERT((relative % 4) == 0, wasm_parse_exception, "AArch64 JIT branch target is unaligned");
         const auto immediate = relative / 4;
         SYS_VM_ASSERT(immediate >= -(1 << 25) && immediate < (1 << 25), wasm_parse_exception,
                       "AArch64 JIT branch target is out of range");
         const uint32_t instruction = 0x94000000u | (static_cast<uint32_t>(immediate) & 0x03ffffffu);
         std::memcpy(branch, &instruction, sizeof(instruction));
      }

      /// Resolves a structured branch relocation.
      static void fix_structured_branch(branch_t branch, void* target) {
         switch (branch.kind) {
            case branch_kind::b: fix_b(branch.address, target); break;
            case branch_kind::b_cond: fix_b_cond(branch, target); break;
            case branch_kind::cbz:
            case branch_kind::cbnz: fix_cb(branch, target); break;
         }
      }

      /// Registers or resolves a direct internal function call relocation.
      void register_call(void* branch, uint32_t funcnum) {
         ensure_function_slots(funcnum);
         if (_function_starts[funcnum]) {
            fix_bl(branch, _function_starts[funcnum]);
         } else {
            _pending_calls[funcnum].push_back(branch);
         }
      }

      /// Records a function start and resolves forward calls waiting on it.
      void start_function(void* func_start, uint32_t funcnum) {
         ensure_function_slots(funcnum);
         _function_starts[funcnum] = func_start;
         for (void* branch : _pending_calls[funcnum]) { fix_bl(branch, func_start); }
         _pending_calls[funcnum].clear();
      }

      /// Ensures relocation vectors have storage for an absolute function index.
      void ensure_function_slots(uint32_t funcnum) {
         if (funcnum >= _function_starts.size()) {
            _function_starts.resize(funcnum + 1);
            _pending_calls.resize(funcnum + 1);
         }
      }

      /// Encodes a 32-bit immediate into a W register using movz/movk.
      void emit_mov_w_imm(uint32_t value, uint32_t reg) {
         emit_u32(0x52800000u | ((value & 0xffffu) << 5) | reg);
         const uint32_t high = (value >> 16) & 0xffffu;
         if (high) {
            emit_u32(0x72800000u | (1u << 21) | (high << 5) | reg);
         }
      }

      /// Encodes a 64-bit immediate into an X register using movz/movk.
      /// TODO: Consider MOVN for dense 0xffff... constants to reduce code size.
      void emit_mov_x_imm(uint64_t value, uint32_t reg) {
         emit_u32(0xd2800000u | (static_cast<uint32_t>(value & 0xffffu) << 5) | reg);
         for (uint32_t chunk = 1; chunk < 4; ++chunk) {
            const auto part = static_cast<uint32_t>((value >> (chunk * 16)) & 0xffffu);
            if (part) {
               emit_u32(0xf2800000u | (chunk << 21) | (part << 5) | reg);
            }
         }
      }

      /// Pushes an X register into a 16-byte stack slot to preserve ABI alignment.
      void push_x(uint32_t reg) {
         constexpr uint32_t pre_index_stp       = 0xa9800000u;
         constexpr uint32_t imm7_minus_16_bytes = 126u;
         emit_u32(pre_index_stp | (imm7_minus_16_bytes << 15) | (zero_reg << 10) | (stack_reg << 5) | reg);
      }

      /// Pops an X register from a 16-byte stack slot.
      void pop_x(uint32_t reg) {
         constexpr uint32_t post_index_ldp     = 0xa8c00000u;
         constexpr uint32_t imm7_plus_16_bytes = 2u;
         emit_u32(post_index_ldp | (imm7_plus_16_bytes << 15) | (zero_reg << 10) | (stack_reg << 5) | reg);
      }

      /// Emits a binary integer operation with the rhs at the top of the WASM operand stack.
      void emit_integer_binop(uint32_t opcode_base) {
         pop_x(scratch0);
         pop_x(scratch1);
         emit_u32(opcode_base | (scratch0 << 16) | (scratch1 << 5) | scratch0);
         push_x(scratch0);
      }

      /// Emits integer multiplication with the rhs at the top of the WASM operand stack.
      void emit_integer_mul(uint32_t opcode_base) {
         pop_x(scratch0);
         pop_x(scratch1);
         emit_u32(opcode_base | (scratch0 << 16) | (zero_reg << 10) | (scratch1 << 5) | scratch0);
         push_x(scratch0);
      }

      /// Emits a helper-backed integer div/rem operation whose helper enforces WASM traps.
      template <typename Function>
      void emit_integer_divrem(bool is_64_bit, Function helper) {
         pop_x(scratch0);
         pop_x(scratch1);
         emit_mov_x_reg(0, scratch1);
         emit_mov_x_reg(1, scratch0);
         emit_call_helper(helper);
         if (!is_64_bit) {
            emit_zero_extend_w(return_reg);
         }
         push_x(return_reg);
         // TODO: Audit helper-backed ops and remove dead context/linear-memory reloads where consumers reload them.
         load_context();
         load_linear_memory();
      }

      /// Alias for helper-backed integer unary operations at integer opcode call sites.
      template <typename Function>
      void emit_integer_unop_helper(bool is_64_bit, Function helper) {
         pop_x(scratch0);
         emit_mov_x_reg(0, scratch0);
         emit_call_helper(helper);
         if (!is_64_bit) {
            emit_zero_extend_w(return_reg);
         }
         push_x(return_reg);
         load_context();
         load_linear_memory();
      }

      /// Emits a helper-backed unary operation with raw native_value slots.
      template <typename Function>
      void emit_value_unop_helper(bool result_is_64_bit, Function helper) {
         require_softfloat("float unary helper");
         pop_x(scratch0);
         emit_mov_x_reg(0, scratch0);
         emit_call_helper(helper);
         if (!result_is_64_bit) {
            emit_zero_extend_w(return_reg);
         }
         push_x(return_reg);
         load_context();
         load_linear_memory();
      }

      /// Emits a helper-backed floating-point comparison whose helper returns i32 0/1.
      template <typename Function>
      void emit_float_compare(bool /*is_f64*/, Function helper) {
         require_softfloat("float compare helper");
         pop_x(scratch0);
         pop_x(scratch1);
         emit_mov_x_reg(0, scratch1);
         emit_mov_x_reg(1, scratch0);
         emit_call_helper(helper);
         emit_zero_extend_w(return_reg);
         push_x(return_reg);
         load_context();
         load_linear_memory();
      }

      /// Emits a helper-backed floating-point binary operation using raw bit-pattern operands.
      template <typename Function>
      void emit_float_binop(bool result_is_64_bit, Function helper) {
         require_softfloat("float binary helper");
         pop_x(scratch0);
         pop_x(scratch1);
         emit_mov_x_reg(0, scratch1);
         emit_mov_x_reg(1, scratch0);
         emit_call_helper(helper);
         if (!result_is_64_bit) {
            emit_zero_extend_w(return_reg);
         }
         push_x(return_reg);
         load_context();
         load_linear_memory();
      }

      /// Emits a register-variable integer shift whose rhs is masked by AArch64 to the WASM width.
      void emit_integer_shift(bool is_64_bit, uint32_t opcode_base) {
         pop_x(scratch0);
         pop_x(scratch1);
         emit_u32(opcode_base | (scratch0 << 16) | (scratch1 << 5) | scratch0);
         (void)is_64_bit;
         push_x(scratch0);
      }

      /// Emits a call-depth reservation before a generated direct or indirect call.
      void emit_enter_call() {
         // TODO: Consider inlining AArch64 call-depth tracking to avoid C-call overhead on every generated call.
         load_context();
         emit_call_helper(&enter_call);
         load_context();
         load_linear_memory();
      }

      /// Emits a call-depth release after a generated direct or indirect call.
      void emit_exit_call() {
         load_context();
         emit_call_helper(&exit_call);
         load_context();
         load_linear_memory();
      }

      /// Emits rotate-left through AArch64's rotate-right instruction and a negated shift count.
      void emit_integer_rotate_left(bool is_64_bit) {
         pop_x(scratch0);
         pop_x(scratch1);
         emit_neg_reg(is_64_bit, scratch0);
         emit_u32((is_64_bit ? 0x9ac02c00u : 0x1ac02c00u) | (scratch0 << 16) | (scratch1 << 5) | scratch0);
         push_x(scratch0);
      }

      /// Emits integer count-leading-zeroes.
      void emit_integer_clz(bool is_64_bit) {
         pop_x(scratch0);
         emit_u32((is_64_bit ? 0xdac01000u : 0x5ac01000u) | (scratch0 << 5) | scratch0);
         push_x(scratch0);
      }

      /// Emits integer count-trailing-zeroes through rbit+clz.
      void emit_integer_ctz(bool is_64_bit) {
         pop_x(scratch0);
         emit_u32((is_64_bit ? 0xdac00000u : 0x5ac00000u) | (scratch0 << 5) | scratch0);
         emit_u32((is_64_bit ? 0xdac01000u : 0x5ac01000u) | (scratch0 << 5) | scratch0);
         push_x(scratch0);
      }

      /// Emits a two's-complement negation into the same integer register.
      void emit_neg_reg(bool is_64_bit, uint32_t reg) {
         emit_u32((is_64_bit ? 0xcb000000u : 0x4b000000u) | (reg << 16) | (zero_reg << 5) | reg);
      }

      /// Emits an i32-to-i64 sign extension.
      void emit_sxtw(uint32_t dst, uint32_t src) { emit_u32(0x93407c00u | (src << 5) | dst); }

      /// Rewrites a W register into itself so AArch64 clears the high X-register half.
      void emit_zero_extend_w(uint32_t reg) { emit_u32(0x2a0003e0u | (reg << 16) | reg); }

      growable_allocator&             _allocator;
      module&                         _module;
      std::size_t                     _source_bytes       = 0;
      void*                           _code_segment_base  = nullptr;
      unsigned char*                  _code_start         = nullptr;
      unsigned char*                  _code               = nullptr;
      unsigned char*                  _code_end           = nullptr;
      uint32_t                        _param_count        = 0;
      uint32_t                        _local_slot_count   = 0;
      uint32_t                        _context_slot       = 0;
      uint32_t                        _linear_memory_slot = 0;
      uint32_t                        _preserved_x19_slot = 0;
      std::vector<void*>              _function_starts;
      std::vector<std::vector<void*>> _pending_calls;
   };

   /// Maps the normal JIT backend name to the AArch64 writer on arm64 builds.
   template <typename Context>
   using machine_code_writer = aarch64_machine_code_writer<Context>;

}} // namespace sysio::vm
