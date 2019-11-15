#pragma once

#include <cstdint>

namespace eosio { namespace vm {

struct options {
   std::uint32_t max_mutable_global_bytes;
   std::uint32_t max_table_elements;
   std::uint32_t max_section_elements;
};

struct default_options {
   static constexpr std::uint32_t max_mutable_global_bytes = 1024;
   static constexpr std::uint32_t max_table_elements = 1024;
};

}}
