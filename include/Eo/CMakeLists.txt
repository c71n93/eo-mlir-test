## Calls mlir_tablegen(...), add_public_tablegen_target(...), set(LLVM_TARGET_DEFINITIONS ...)
## It don't works for some reason.
#add_mlir_dialect(EoOps eo)
#add_mlir_doc(EoDialect EoDialect Eo/ -gen-dialect-doc)
#add_mlir_doc(EoOps EoOps Eo/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS EoOps.td)
mlir_tablegen(EoOps.h.inc -gen-op-decls)
mlir_tablegen(EoOps.cpp.inc -gen-op-defs)
mlir_tablegen(EoDialect.h.inc -gen-dialect-decls)
mlir_tablegen(EoDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIREoOpsIncGen)

add_mlir_doc(EoDialect EoDialect Eo/ -gen-dialect-doc)
add_mlir_doc(EoOps EoOps Eo/ -gen-op-doc)
