#ifndef EO_MLIR_TEST_EOOPS_H
#define EO_MLIR_TEST_EOOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "Eo/EoOps.h.inc"

#endif //EO_MLIR_TEST_EOOPS_H
