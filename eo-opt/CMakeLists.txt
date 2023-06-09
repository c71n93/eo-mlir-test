get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIREo
        pugixml
        )
add_llvm_executable(eo-opt eo-opt.cpp)

llvm_update_compile_flags(eo-opt)
target_link_libraries(eo-opt PRIVATE ${LIBS})
