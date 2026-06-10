#pragma once

#ifndef SYS_VM_HAS_JIT_BACKEND
#   if defined(__x86_64__) || defined(_M_X64) ||                                                                       \
         ((defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)) && defined(__APPLE__))
#      define SYS_VM_HAS_JIT_BACKEND 1
#   else
#      define SYS_VM_HAS_JIT_BACKEND 0
#   endif
#   define SYS_VM_HAS_JIT_BACKEND_DEFAULTED 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
#   define SYS_VM_TARGET_X86_64 1
#else
#   define SYS_VM_TARGET_X86_64 0
#endif

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#   define SYS_VM_TARGET_ARM64 1
#else
#   define SYS_VM_TARGET_ARM64 0
#endif

#if defined(__APPLE__)
#   define SYS_VM_TARGET_APPLE 1
#else
#   define SYS_VM_TARGET_APPLE 0
#endif

#ifndef SYS_VM_ENABLE_AARCH64_JIT
#   define SYS_VM_ENABLE_AARCH64_JIT (SYS_VM_TARGET_ARM64 && SYS_VM_TARGET_APPLE)
#endif

#ifndef SYS_VM_HAS_AARCH64_JIT_BACKEND
#   define SYS_VM_HAS_AARCH64_JIT_BACKEND                                                                              \
      (SYS_VM_HAS_JIT_BACKEND && SYS_VM_TARGET_ARM64 && SYS_VM_TARGET_APPLE && SYS_VM_ENABLE_AARCH64_JIT)
#endif

#if SYS_VM_HAS_JIT_BACKEND && !SYS_VM_TARGET_X86_64 && !SYS_VM_HAS_AARCH64_JIT_BACKEND &&                              \
      !defined(SYS_VM_HAS_JIT_BACKEND_DEFAULTED)
#   error "SYS_VM_HAS_JIT_BACKEND requires x86_64 or Apple AArch64"
#endif

#ifndef SYS_VM_HAS_JIT_PROFILE
#   define SYS_VM_HAS_JIT_PROFILE (SYS_VM_HAS_JIT_BACKEND && SYS_VM_TARGET_X86_64)
#endif

namespace sysio { namespace vm {

// create constexpr flags for whether the backend should obey alignment hints
#ifdef SYS_VM_ALIGN_MEMORY_OPS
   inline constexpr bool should_align_memory_ops = true;
#else
   inline constexpr bool should_align_memory_ops = false;
#endif

#ifdef SYS_VM_SOFTFLOAT
   inline constexpr bool use_softfloat = true;
#else
   inline constexpr bool use_softfloat = false;
#endif

#ifdef SYS_VM_FULL_DEBUG
   inline constexpr bool sys_vm_debug = true;
#else
   inline constexpr bool sys_vm_debug = false;
#endif

   /// True when this build is targeting an x86_64 host.
   inline constexpr bool sys_vm_target_x86_64 = SYS_VM_TARGET_X86_64 != 0;

   /// True when this build is targeting an AArch64 host.
   inline constexpr bool sys_vm_target_aarch64 = SYS_VM_TARGET_ARM64 != 0;

   /// Backwards-compatible alias for older integrations that check the x86_64 target flag.
   inline constexpr bool sys_vm_amd64 = sys_vm_target_x86_64;

   /// True when the selected sys-vm package contains a normal JIT backend.
   inline constexpr bool sys_vm_has_jit_backend = SYS_VM_HAS_JIT_BACKEND != 0;

   /// True when this build supports async JIT profiling/backtrace helpers.
   inline constexpr bool sys_vm_has_jit_profile = SYS_VM_HAS_JIT_PROFILE != 0;

   /// True when this build contains the Apple AArch64 JIT backend.
   inline constexpr bool sys_vm_has_aarch64_jit_backend = SYS_VM_HAS_AARCH64_JIT_BACKEND != 0;

}} // namespace sysio::vm
