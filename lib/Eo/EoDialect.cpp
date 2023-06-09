#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Eo/EoDialect.h"
#include "Eo/EoOps.h"

using namespace mlir;
using namespace eo;

#include "Eo/EoDialect.cpp.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void EoDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Eo/EoOps.cpp.inc"
    >();
}


