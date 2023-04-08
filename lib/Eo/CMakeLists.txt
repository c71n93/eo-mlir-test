add_mlir_dialect_library(MLIREo
        EoDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Eo

        DEPENDS
        MLIREoOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )