# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

> Important: eos-vm is synonymous with sys-vm in this repository.

Wire SYS VM (formerly EOS VM) is a high-performance WebAssembly engine designed for blockchain applications. It prioritizes deterministic execution, time-bounded execution, and security. The library is **header-only** (except for the softfloat dependency) and targets **Unix-like OSes only** (Linux, macOS). Current version: 1.1.0.

## Build Commands

### Initial Setup
```bash
git submodule update --init --recursive
./vcpkg/bootstrap-vcpkg.sh
```

### Configure and Build
```bash
# Debug build, use build/claude
# Release build, use build/claude-release
cmake -S . -B build/claude \
  -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DENABLE_TESTS=ON \
  -DENABLE_SPEC_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/claude -- -j$(nproc)
```

### Run All Tests
```bash
cd build/claude && ctest -j$(nproc) --output-on-failure
```

### Run a Single Test
Tests use Catch2 and are discovered via CTest. To run a specific test by name:
various ways:
```bash
cd build/claude && ctest -R <test_name_regex> --output-on-failure
```
Or run a test executable directly with Catch2 filtering:
```bash
./build/claude/tests/unit_tests "<test case name>"
./build/claude/tests/sys_vm_spec_tests "<test case name>"
```

### Key CMake Options
| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_SOFTFLOAT` | ON | Deterministic software floating-point |
| `ENABLE_TESTS` | OFF | Build unit + spec tests |
| `ENABLE_SPEC_TESTS` | ON (if tests) | WebAssembly spec compliance tests |
| `ENABLE_FUZZ_TESTS` | OFF | Fuzzing harness |
| `ENABLE_TOOLS` | ON | Build example tools (interp, bench, hello_driver) |
| `FULL_DEBUG_BUILD` | ON in Debug | Stack dumps and instruction tracing |
| `ENABLE_MEMORY_OPS_ALIGNMENT` | OFF | Obey alignment hints |

## Architecture

### Core Library (`include/sysio/vm/`)

The engine is template-heavy C++20. All core types live in namespace `sysio::vm`.

**Backend** (`backend.hpp`): The main API entry point. Template-parameterized by:
- `HostFunctions` — registered native functions callable from WASM
- `Impl` — execution backend: `interpreter` (default), `jit` (x86-64 only), `jit_profile`, `null_backend`
- `Options` — compile-time or runtime configurable limits
- `DebugInfo` — optional debug information

**Execution flow**: `wasm_allocator` → `watchdog` (timeout) → `backend` (parse + instantiate) → execute exports via `backend(host, "module", "func", args...)`.

**Parser** (`parser.hpp`, ~1800 lines): Single-pass O(n) binary WASM parser with bounds checking. Handles LEB128 decoding (`leb128.hpp`), section parsing (`sections.hpp`), and validation (`validation.hpp`).

**Interpreter** (`execution_context.hpp` + `interpret_visitor.hpp`): Stack-machine bytecode interpreter using a visitor pattern with static dispatch (not virtual). Opcodes defined in `opcodes_def.hpp` and `opcodes.hpp`.

**JIT** (`x86_64.hpp`, ~2600 lines): x86-64 native code generation backend. Only available on x86_64.

**Memory** (`allocator.hpp`): Multiple allocator strategies — `bounded_allocator`, `stack_allocator` (guard-paged stacks), `contiguous_allocator` (mmap-based), `growable_allocator`. Guard paging provides memory sandboxing.

**Host Functions** (`host_function.hpp`): Type-safe registration via `registered_host_functions<HostClass>::add<&method>("module", "name")`. Supports C functions, static methods, and instance methods with automatic WASM↔native type conversion.

**Watchdog** (`watchdog.hpp`): Time-bounded execution using background thread timers. `null_watchdog` for unbounded execution. Multi-threaded timeout uses atomic tracking of in-progress threads.

**Custom Data Types**: `variant.hpp` (fast discriminated union for opcodes/stack), `vector.hpp` (non-owning fast vector backed by allocators), `wasm_stack.hpp`.

**Signals** (`signals.hpp`): Signal handling for watchdog timeout interrupts and guard page violations.

### Tests (`tests/`)

- **Unit tests** (`tests/*.cpp`): ~30 test files covering allocators, host functions, watchdog, signals, configuration limits, backend behavior. Built as `unit_tests` executable.
- **Spec tests** (`tests/spec/`): ~79 WebAssembly spec compliance test suites. Built as `sys_vm_spec_tests` executable. Test WASM binaries fetched from `wire-network/wire-sys-vm-test-wasms`.
- **Fuzz tests** (`tests/fuzz/`): Optional fuzzing harness.

Test framework: **Catch2** configured with `CATCH_CONFIG_NO_POSIX_SIGNALS` (the VM handles signals itself).

### Tools (`tools/`)

- `sys-vm-interp` — standalone WASM interpreter, runs all exports
- `bench-interp` — benchmarking (parse/instantiate + execution time)
- `hello-driver` — host function integration example

### Dependencies

Managed via **vcpkg** (git submodule):
- **softfloat** — deterministic IEEE-754 floating-point
- **Catch2** — testing (with no-posix-signals feature)

## Key Design Constraints

- No unbounded recursion or loops in parsing/execution
- All collections are bounded or validated
- Guard paging makes pointer validation in host functions unnecessary (but host functions must be able to fail hard without calling destructors)
- Non-owning data structures — objects are constructed in-place in allocator memory and never copied
- Atomic synchronization for multi-threaded `timed_run` coordination
- Softfloat ensures cross-platform determinism for consensus
