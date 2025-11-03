function(setup_vm_target TARGET)
   target_include_directories(
      ${TARGET}
      PRIVATE
      "${CMAKE_SOURCE_DIR}/include"
   )

   # SYS_VM: Add softfloat include path, defs & links directly for legacy support
   if(ENABLE_SOFTFLOAT)
      target_include_directories(
         ${TARGET}
         PRIVATE
         "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include"
         "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/softfloat"
      )

      target_link_libraries(${TARGET} softfloat::softfloat)
   endif()
endfunction()