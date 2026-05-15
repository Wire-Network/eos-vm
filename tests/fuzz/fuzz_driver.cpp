#include <sysio/vm/backend.hpp>
#include <sysio/vm/watchdog.hpp>

#include <chrono>
#include <cstring>
#include <exception>

using namespace sysio;
using namespace sysio::vm;

extern "C" int LLVMFuzzerTestOneInput( const uint8_t* data, size_t size ) {
   wasm_allocator wa;
   wasm_code wc; 
   wc.resize(size);
   memcpy((uint8_t*)wc.data(), data, size);
   try {
      backend<std::nullptr_t> bkend( wc, &wa );
      bkend.execute_all(watchdog{std::chrono::milliseconds(100)});
   } catch(const std::exception&) {
      // Invalid or long-running fuzz inputs are expected to throw VM exceptions.
      // Returning 0 tells libFuzzer the input was rejected cleanly.
   }
   return 0;
}
