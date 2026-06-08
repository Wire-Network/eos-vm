# AArch64 JIT Benchmark Report

## Scope

This report captures the local Apple Silicon feasibility benchmark for the experimental AArch64
`sysio::vm::jit` backend. The benchmark compares the existing `sysio::vm::interpreter` runtime
against the AArch64 JIT on the same WASM kernel.

The benchmark is a microbenchmark, not a production block-processing profile. It is intended to
answer whether the JIT path is materially faster than `sys-vm` for a loop-heavy scalar workload.

## Build

Build directory:

```text
build/macos-arm64-port
```

Benchmark target:

```text
build/macos-arm64-port/tests/aarch64_prototype_benchmark
```

## Validation

The full local CTest suite passed with both interpreter and promoted AArch64 JIT tests enabled:

```text
100% tests passed, 0 tests failed out of 283
```

Command:

```sh
ctest --test-dir build/macos-arm64-port --output-on-failure
```

## Benchmark

Command:

```sh
build/macos-arm64-port/tests/aarch64_prototype_benchmark
```

Result:

```text
kernel=sum_to
input=1000
iterations=20000
interpreter_ms=5393.78
aarch64_prototype_jit_ms=97.81
speedup=55.1455
checksum=10039950000
```

## Interpretation

For this scalar loop benchmark, the AArch64 JIT is about 55.1x faster than the interpreter. This is
a strong enough feasibility signal to keep the macOS arm64 JIT path in scope for developer builds.

The benchmark does not prove block-production suitability. macOS arm64 builds remain developer-only,
and the JIT should continue to use softfloat for floating-point behavior.
