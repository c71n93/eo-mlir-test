add_mlir_dialect_library(MLIRHello
        EoDialect.cpp
        EoOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Eo

        DEPENDS
        MLIREoOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )