#include "mlir/IR/OpImplementation.h"

#include "Eo/EoDialect.h"
#include "Eo/EoOps.h"

using namespace mlir;
using namespace eo;

#define GET_OP_CLASSES
#include "Eo/EoOps.cpp.inc"
