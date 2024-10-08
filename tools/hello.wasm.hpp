// Code used to generate this wasm
// compile with sysio-cpp -o hello.wasm hello.cpp
//#include <sysio/sysio.hpp>
//#include <sysio/print.hpp>
//
// extern "C" {
//[[sysio::wasm_import]] void print_name(const char*);
//[[sysio::wasm_import]] void print_num(uint64_t);
//[[sysio::wasm_import]] void print_span(const char*, std::size_t);
//
// void apply(uint64_t a, uint64_t b, uint64_t c)
//{
//    const char* test_str = "hellohellohello";
//    print_num(a);
//    print_num(b);
//    print_num(c);
//    sysio::check(b == c, "Failure B != C");
//    for (uint64_t i = 0; i < a; i++)
//        print_name("sys-vm");
//    print_span(test_str, 5);
//    print_span(test_str, 10);
//}
//}

std::vector<uint8_t> hello_wasm = {
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x1e, 0x06, 0x60,
  0x01, 0x7e, 0x00, 0x60, 0x02, 0x7f, 0x7f, 0x00, 0x60, 0x01, 0x7f, 0x00,
  0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, 0x60, 0x00, 0x00, 0x60, 0x03,
  0x7e, 0x7e, 0x7e, 0x00, 0x02, 0x53, 0x05, 0x03, 0x65, 0x6e, 0x76, 0x09,
  0x70, 0x72, 0x69, 0x6e, 0x74, 0x5f, 0x6e, 0x75, 0x6d, 0x00, 0x00, 0x03,
  0x65, 0x6e, 0x76, 0x0c, 0x73, 0x79, 0x73, 0x69, 0x6f, 0x5f, 0x61, 0x73,
  0x73, 0x65, 0x72, 0x74, 0x00, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x0a, 0x70,
  0x72, 0x69, 0x6e, 0x74, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x00, 0x02, 0x03,
  0x65, 0x6e, 0x76, 0x0a, 0x70, 0x72, 0x69, 0x6e, 0x74, 0x5f, 0x73, 0x70,
  0x61, 0x6e, 0x00, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x06, 0x6d, 0x65, 0x6d,
  0x73, 0x65, 0x74, 0x00, 0x03, 0x03, 0x05, 0x04, 0x04, 0x04, 0x02, 0x05,
  0x04, 0x05, 0x01, 0x70, 0x01, 0x01, 0x01, 0x05, 0x03, 0x01, 0x00, 0x01,
  0x06, 0x16, 0x03, 0x7f, 0x01, 0x41, 0x80, 0xc0, 0x00, 0x0b, 0x7f, 0x00,
  0x41, 0xb6, 0xc0, 0x00, 0x0b, 0x7f, 0x00, 0x41, 0xb6, 0xc0, 0x00, 0x0b,
  0x07, 0x09, 0x01, 0x05, 0x61, 0x70, 0x70, 0x6c, 0x79, 0x00, 0x08, 0x0a,
  0x93, 0x01, 0x04, 0x04, 0x00, 0x10, 0x06, 0x0b, 0x36, 0x01, 0x01, 0x7f,
  0x23, 0x00, 0x41, 0x10, 0x6b, 0x22, 0x00, 0x41, 0x00, 0x36, 0x02, 0x0c,
  0x41, 0x00, 0x20, 0x00, 0x28, 0x02, 0x0c, 0x28, 0x02, 0x00, 0x41, 0x07,
  0x6a, 0x41, 0x78, 0x71, 0x22, 0x00, 0x36, 0x02, 0x84, 0x40, 0x41, 0x00,
  0x20, 0x00, 0x36, 0x02, 0x80, 0x40, 0x41, 0x00, 0x3f, 0x00, 0x36, 0x02,
  0x8c, 0x40, 0x0b, 0x02, 0x00, 0x0b, 0x52, 0x00, 0x10, 0x05, 0x20, 0x00,
  0x10, 0x00, 0x20, 0x01, 0x10, 0x00, 0x20, 0x02, 0x10, 0x00, 0x02, 0x40,
  0x20, 0x01, 0x20, 0x02, 0x51, 0x0d, 0x00, 0x41, 0x00, 0x41, 0xa0, 0xc0,
  0x00, 0x10, 0x01, 0x0b, 0x02, 0x40, 0x20, 0x00, 0x50, 0x0d, 0x00, 0x03,
  0x40, 0x41, 0xaf, 0xc0, 0x00, 0x10, 0x02, 0x20, 0x00, 0x42, 0x7f, 0x7c,
  0x22, 0x00, 0x50, 0x45, 0x0d, 0x00, 0x0b, 0x0b, 0x41, 0x90, 0xc0, 0x00,
  0x41, 0x05, 0x10, 0x03, 0x41, 0x90, 0xc0, 0x00, 0x41, 0x0a, 0x10, 0x03,
  0x41, 0x00, 0x10, 0x07, 0x0b, 0x0b, 0x45, 0x04, 0x00, 0x41, 0x90, 0xc0,
  0x00, 0x0b, 0x10, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x68, 0x65, 0x6c, 0x6c,
  0x6f, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x00, 0x00, 0x41, 0xa0, 0xc0, 0x00,
  0x0b, 0x0f, 0x46, 0x61, 0x69, 0x6c, 0x75, 0x72, 0x65, 0x20, 0x42, 0x20,
  0x21, 0x3d, 0x20, 0x43, 0x00, 0x00, 0x41, 0xaf, 0xc0, 0x00, 0x0b, 0x07,
  0x65, 0x6f, 0x73, 0x2d, 0x76, 0x6d, 0x00, 0x00, 0x41, 0x00, 0x0b, 0x04,
  0x38, 0x20, 0x00, 0x00
};
