#pragma once

/*
 * definitions from https://github.com/WebAssembly/design/blob/master/BinaryEncoding.md
 */

#include <sysio/vm/allocator.hpp>
#include <sysio/vm/constants.hpp>
#include <sysio/vm/exceptions.hpp>
#include <sysio/vm/stack_elem.hpp>
#include <sysio/vm/types.hpp>
#include <sysio/vm/vector.hpp>

namespace sysio { namespace vm {
   using std::nullptr_t;

   template <typename ElemT, size_t ElemSz, typename Allocator = nullptr_t >
   class stack {
    public:
      template <typename Alloc=Allocator, typename = std::enable_if_t<std::is_same_v<Alloc, std::nullptr_t>, int>>
      stack()
         : _store(ElemSz) {}

      template <typename Alloc=Allocator, typename = std::enable_if_t<!std::is_same_v<Alloc, nullptr_t>, int>>
      stack(Alloc&& alloc)
         : _store(alloc, ElemSz) {}

      template <typename Alloc=Allocator, typename = std::enable_if_t<!std::is_same_v<Alloc, nullptr_t>, int>>
      stack(uint32_t n, Alloc&& alloc)
         : _store(alloc, n) {}

      void push(ElemT&& e) { 
         if constexpr (std::is_same_v<Allocator, nullptr_t>) {
            if (_index >= _store.size())
               _store.resize(_store.size()*2);
         }
         _store[_index++] = std::forward<ElemT>(e);
      }

      ElemT pop() { return _store[--_index]; }

      ElemT& get(uint32_t index) const {
         SYS_VM_ASSERT(index <= _index, wasm_interpreter_exception, "invalid stack index");
         return (ElemT&)_store[index];
      }
      void set(uint32_t index, const ElemT& el) {
         SYS_VM_ASSERT(index <= _index, wasm_interpreter_exception, "invalid stack index");
         _store[index] = el;
      }
      void  eat(uint32_t index) { _index = index; }
      // compact the last element to the element pointed to by index
      void compact(uint32_t index) {
         _store[index] = _store[_index-1];
         _index = index+1;
      }
      size_t       current_index() const { return _index; }
      ElemT&       peek() { return _store[_index - 1]; }
      const ElemT& peek() const { return _store[_index - 1]; }
      ElemT&       peek(size_t i) { return _store[_index - 1 - i]; }
      ElemT&       get_back(size_t i) { return _store[_index - 1 - i]; }
      const ElemT& get_back(size_t i)const { return _store[_index - 1 - i]; }
      void         trim(size_t amt) { _index -= amt; }
      size_t       size() const { return _index; }
      size_t       capacity() const { return _store.size(); }

    private:
      using base_data_store_t = std::conditional_t<std::is_same_v<Allocator, std::nullptr_t>, unmanaged_vector<ElemT>, managed_vector<ElemT, Allocator>>;

      base_data_store_t _store;
      size_t            _index = 0;
   };

   using operand_stack = stack<operand_stack_elem, constants::initial_stack_size>;
   using call_stack    = stack<activation_frame,   constants::max_call_depth + 1, bounded_allocator>;

}} // namespace sysio::vm
