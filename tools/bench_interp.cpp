#include <sysio/vm/backend.hpp>
#include <sysio/vm/error_codes.hpp>
#include <sysio/vm/watchdog.hpp>

#include <chrono>
#include <iostream>

using namespace sysio;
using namespace sysio::vm;

int main(int argc, char** argv) {
   wasm_allocator wa;

   if (argc < 2) {
      std::cerr << "Error, no wasm file provided\n";
      return -1;
   }
   auto t3 = std::chrono::high_resolution_clock::now();
   try {

      auto code = read_wasm( argv[1] );

      auto t1 = std::chrono::high_resolution_clock::now();
      backend bkend( code, &wa );
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << "Startup " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() << "\n";

      auto t3 = std::chrono::high_resolution_clock::now();
      bkend.execute_all(null_watchdog());
      auto t4 = std::chrono::high_resolution_clock::now();
      std::cout << "Execution " << std::chrono::duration_cast<std::chrono::nanoseconds>(t4-t3).count() << "\n";

   } catch ( const sysio::vm::exception& ex ) {
      auto t4 = std::chrono::high_resolution_clock::now();
      std::cout << "Execution " << std::chrono::duration_cast<std::chrono::nanoseconds>(t4-t3).count() << "\n";
      std::cerr << "sys-vm interpreter error\n";
      std::cerr << ex.what() << " : " << ex.detail() <<  "\n";
   }
   return 0;
}
